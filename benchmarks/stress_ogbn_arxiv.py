"""Stress test on ogbn-arxiv: every SSL method through the mini-batch path.

The citation benchmark (run_benchmark.py) is full-batch on graphs of a few thousand nodes,
so it never touches what this script exercises:

- NeighborLoader training (loss on the seed nodes only) on 169k nodes / 2.3M edges;
- the trainer's callback hooks, with an evaluation (kNN, effective rank) run *during* training;
- full-graph embedding extraction and linear-probe / kNN evaluation at that scale;
- mini-batch embedding extraction, checked against the full-graph pass.

It does not exercise ``protected_nodes``: the shared augmentations (``edge_drop``,
``feat_mask``) never remove a node, so there is nothing to protect.

It is a robustness test, not a benchmark: one seed, no tuning. A model that fails (exception,
out of memory, non-finite loss) does not stop the run; its error is saved and the next model
starts. Per model it records the status, per-epoch loss / wall time / peak GPU memory / mean
sampled-subgraph size, the kNN accuracy and effective rank during training, the linear probe
before and after it (one fit of the probe takes minutes on 91k training nodes, so it is not
repeated at every evaluation), and the same encoder left untrained as the reference.

Usage::

    pip install "graphssl[benchmark]"
    # smoke test, a few minutes: 2 short epochs per model
    python benchmarks/stress_ogbn_arxiv.py --epochs 2 --max-steps 5 --eval-every 1 \\
        --skip-linear-probe --out-dir benchmarks/stress/smoke
    # the real run, inside tmux
    python -u benchmarks/stress_ogbn_arxiv.py --epochs 50 2>&1 | tee -a stress_arxiv.log

One JSON per model goes to ``--out-dir`` (default ``benchmarks/stress/ogbn_arxiv``), next to
a ``summary.md`` table and to the trained model (``<model>.pt``, git-ignored;
``graphssl.config.load_model`` reads it back). The exit code is 1 if any model failed.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch
import torch_geometric
from torch.optim import AdamW
from torch_geometric.transforms import ToUndirected

from graphssl.config.load import build_model, save_model
from graphssl.core.callback import Callback
from graphssl.data import DataModule
from graphssl.evaluation import KNNEvaluator, LogRegEvaluator, effective_rank, extract_embeddings
from graphssl.training import DINOTrainer
from graphssl.utils.positive_miner import HAS_FAISS

REPO_ROOT = Path(__file__).resolve().parent.parent
SSL_MODELS = ["dgi", "graphcl", "vicreg", "barlow_twins", "bgrl", "afgrl", "graphdino"]
# Mini-batch and full-graph embeddings differ only by float summation order.
EXTRACTION_TOLERANCE = 1e-3


def fmt_eval(ev: dict) -> str:
    line = f"knn {ev['test_acc_knn']:.4f} | rank {ev['eff_rank']:.1f}"
    if "test_acc_linear" in ev:
        note = "" if ev["probe_converged"] else ", NOT converged"
        line = (
            f"linear {ev['test_acc_linear']:.4f} (L2 {ev['probe_weight_decay']:g}{note}) | {line}"
        )
    return line


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", nargs="+", default=["all"], choices=[*SSL_MODELS, "all"])
    p.add_argument("--encoder", default="gin", choices=["gin", "gcn", "transformer"])
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=1024, help="seed nodes per mini-batch")
    p.add_argument("--fanout", type=int, default=10, help="neighbors sampled per hop")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="kNN accuracy and effective rank every N epochs during training (0: never)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop each epoch after N mini-batches (smoke tests)",
    )
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument(
        "--skip-linear-probe",
        action="store_true",
        help="no linear probe before/after training (minutes per fit on ogbn-arxiv): smoke tests",
    )
    p.add_argument(
        "--skip-extraction-check",
        action="store_true",
        help="don't compare mini-batch embedding extraction with the full-graph pass",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-dir", default="data/ogbn-arxiv")
    p.add_argument("--out-dir", default="benchmarks/stress/ogbn_arxiv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if "all" in args.model:
        args.model = SSL_MODELS
    return args


def load_arxiv(data_dir: str):
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as e:
        raise ImportError('Install ogb: pip install "graphssl[benchmark]"') from e

    # ogb (<= 1.3.6) loads its processed file with a bare torch.load(); since PyTorch 2.6
    # that defaults to weights_only=True and rejects PyG's Data classes unless allow-listed.
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage

        add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=data_dir)
    data = ToUndirected()(dataset[0])  # directed citations -> undirected
    split = dataset.get_idx_split()
    return data, split["train"], split["valid"], split["test"], dataset.num_classes


def make_config(model_name: str, args: argparse.Namespace, total_steps: int) -> dict:
    """The citation benchmark's protocol, with the EMA schedules spanning the whole run.

    GraphDINO keeps the library defaults (EMA 0.9 -> 0.996, teacher temperature 0.07): this
    tests the library as shipped, unlike run_benchmark.py, which pins reference DINO values.
    """
    augment = [{"name": "edge_drop", "p": 0.5}, {"name": "feat_mask", "p": 0.2}]
    cfg: dict = {
        "name": model_name,
        "encoder": {
            "name": args.encoder,
            "hidden_dim": args.hidden,
            "num_layers": args.layers,
            "norm_type": "batch",
            "pool": False,  # node-level task: no graph pooling
            "drop": 0.0,
        },
    }
    if model_name in ("graphcl", "vicreg", "barlow_twins", "bgrl"):
        cfg["augment"] = augment
    if model_name in ("bgrl", "afgrl"):
        cfg.update(pred_hidden=args.hidden, ema_tau=0.99, ema_tau_end=1.0, total_steps=total_steps)
    if model_name == "graphdino":
        cfg.update(
            augment_teacher=augment,
            augment_student=augment,
            total_steps=total_steps,
            head={
                "name": "dino",
                "proj_hidden": args.hidden,
                "bottleneck_dim": max(args.hidden // 4, 8),
                "n_prototypes": 128,
            },
        )
    return cfg


class LimitedLoader:
    """Yields at most ``max_steps`` batches per epoch from ``loader`` (all if None)."""

    def __init__(self, loader, max_steps: Optional[int]):
        self.loader = loader
        self.max_steps = max_steps

    def __len__(self) -> int:
        n = len(self.loader)
        return n if self.max_steps is None else min(n, self.max_steps)

    def __iter__(self) -> Iterator:
        for step, batch in enumerate(self.loader):
            if self.max_steps is not None and step >= self.max_steps:
                break
            yield batch


class StressMonitor(Callback):
    """Per-epoch loss, wall time, peak GPU memory and subgraph size; periodic evaluation.

    Raises on the first non-finite loss, so a diverged model is reported as failed instead
    of training on NaNs for hours.
    """

    def __init__(self, eval_fn: Callable[[], dict], eval_every: int, num_epochs: int):
        self.eval_fn = eval_fn
        self.eval_every = eval_every
        self.num_epochs = num_epochs
        self.epochs: List[dict] = []
        self.curve: List[dict] = []
        self._t0 = 0.0
        self._nodes = 0
        self._edges = 0
        self._steps = 0

    def on_epoch_start(self, trainer: Any, model: torch.nn.Module, epoch: int) -> None:
        self._t0 = time.time()
        self._nodes = self._edges = self._steps = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_batch_end(self, trainer: Any, model: torch.nn.Module, loss: torch.Tensor, batch) -> None:
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss ({loss.item()}) at step {self._steps}")
        self._nodes += batch.num_nodes
        self._edges += batch.edge_index.size(1)
        self._steps += 1

    def on_epoch_end(
        self, trainer: Any, model: torch.nn.Module, epoch: int, metrics: Dict[str, Any]
    ) -> None:
        steps = max(self._steps, 1)
        row = {
            "epoch": epoch + 1,
            "loss": metrics["loss"],
            "wall_time_s": time.time() - self._t0,
            "steps": self._steps,
            "mean_batch_nodes": self._nodes / steps,
            "mean_batch_edges": self._edges / steps,
            "peak_gpu_mem_gb": (
                torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
            ),
        }
        self.epochs.append(row)
        line = (
            f"  epoch {row['epoch']:>3}/{self.num_epochs}: loss {row['loss']:.4f} | "
            f"{row['wall_time_s']:.1f}s | {row['mean_batch_nodes']:,.0f} nodes/batch"
        )
        if row["peak_gpu_mem_gb"] is not None:
            line += f" | peak {row['peak_gpu_mem_gb']:.2f} GiB"
        is_last = epoch + 1 == self.num_epochs
        if self.eval_every > 0 and (epoch + 1) % self.eval_every == 0 and not is_last:
            ev = {"epoch": epoch + 1, **self.eval_fn()}
            self.curve.append(ev)
            line += f" | {fmt_eval(ev)}"
            model.train()  # the evaluation put the model in eval mode
        print(line, flush=True)


def evaluate(
    model: torch.nn.Module,
    dm: DataModule,
    num_classes: int,
    device: torch.device,
    knn_k: int,
    linear_probe: bool = True,
) -> dict:
    """kNN accuracy and effective rank of the full-graph embeddings, plus the linear probe."""
    z, y = extract_embeddings(model, dm, device=str(device))
    if not torch.isfinite(z).all():
        raise FloatingPointError("non-finite values in the extracted embeddings")
    assert y is not None and dm.train_idx is not None
    assert dm.val_idx is not None and dm.test_idx is not None
    knn = KNNEvaluator(k=knn_k).evaluate(z, y, dm.train_idx, dm.val_idx, dm.test_idx)
    result = {
        "val_acc_knn": knn["val_acc"],
        "test_acc_knn": knn["test_acc"],
        "eff_rank": effective_rank(z),
    }
    if linear_probe:
        # One L-BFGS fit per L2 strength, the weakest taking thousands of iterations on 91k
        # training nodes: minutes, on the training device.
        lin = LogRegEvaluator().evaluate(
            z.to(device), y, dm.train_idx, dm.val_idx, dm.test_idx, num_classes=num_classes
        )
        result.update(
            val_acc_linear=lin["val_acc"],
            test_acc_linear=lin["test_acc"],
            probe_weight_decay=lin["weight_decay"],
            probe_converged=lin["converged"],
        )
    return result


def extraction_rel_diff(
    model: torch.nn.Module, dm: DataModule, layers: int, device: torch.device
) -> float:
    """Largest difference between mini-batch and full-graph embeddings, relative to scale.

    With every neighbor sampled over as many hops as the encoder has layers, a seed node's
    embedding must equal the full-graph one up to float error: this checks the reassembly of
    NeighborLoader batches into node order.
    """
    z_full, _ = extract_embeddings(model, dm, device=str(device))
    z_mini, _ = extract_embeddings(
        model, dm, device=str(device), mini_batch=True, num_neighbors=[-1] * layers
    )
    return ((z_mini - z_full).abs().max() / z_full.abs().max().clamp_min(1e-12)).item()


def run_model(
    model_name: str,
    args: argparse.Namespace,
    dm: DataModule,
    num_classes: int,
    device: torch.device,
) -> dict:
    assert dm.data is not None
    loader = LimitedLoader(
        dm.neighbor_loader(
            num_neighbors=[args.fanout] * args.layers,
            input_nodes=dm.train_idx,
            shuffle=True,
        ),
        args.max_steps,
    )
    config = make_config(model_name, args, total_steps=args.epochs * len(loader))
    report: dict = {"model": model_name, "model_config": config, "status": "failed"}
    monitor: Optional[StressMonitor] = None
    t0 = time.time()
    try:
        if model_name == "afgrl" and not HAS_FAISS:
            raise ImportError("faiss is not installed (pip install faiss-cpu)")
        torch.manual_seed(args.seed)
        model = build_model(config, in_channels=dm.data.num_features)
        probe = not args.skip_linear_probe
        report["untrained"] = evaluate(model, dm, num_classes, device, args.knn_k, probe)
        print(f"  untrained: {fmt_eval(report['untrained'])}", flush=True)

        monitor = StressMonitor(
            eval_fn=lambda: evaluate(
                model, dm, num_classes, device, args.knn_k, linear_probe=False
            ),
            eval_every=args.eval_every,
            num_epochs=args.epochs,
        )
        optimizer = AdamW(model.student_parameters(), lr=args.lr, weight_decay=args.weight_decay)
        trainer = DINOTrainer(grad_clip_norm=None, device=device, callbacks=[monitor])
        trainer.train(model, loader, optimizer, num_epochs=args.epochs)

        report["final"] = evaluate(model, dm, num_classes, device, args.knn_k, probe)
        print(f"  final: {fmt_eval(report['final'])}", flush=True)
        # Kept so that another evaluation does not need another run.
        checkpoint = Path(args.out_dir) / f"{model_name}.pt"
        save_model(model, checkpoint, config, in_channels=dm.data.num_features)
        report["checkpoint"] = checkpoint.name
        if not args.skip_extraction_check:
            rel = extraction_rel_diff(model, dm, args.layers, device)
            report["minibatch_extraction_rel_diff"] = rel
            print(f"  mini-batch vs full-graph extraction: rel. diff {rel:.1e}", flush=True)
            if rel > EXTRACTION_TOLERANCE:
                raise AssertionError(
                    f"mini-batch extraction differs from the full-graph pass (rel. diff {rel:.1e})"
                )
        report["status"] = "ok"
    except Exception as e:  # a stress test reports failures instead of stopping at the first
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
        print(f"  FAILED: {report['error']}", flush=True)
    finally:
        report["wall_time_s"] = time.time() - t0
        if monitor is not None:
            report["epochs"] = monitor.epochs
            report["curve"] = monitor.curve
        model = None  # type: ignore[assignment]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


def git_info() -> dict:
    def _run(cmd: List[str]) -> Optional[str]:
        try:
            return subprocess.run(
                cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return None

    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(dirty) if dirty is not None else None,
    }


def summary_row(r: dict) -> str:
    def pct(section: str, key: str) -> str:
        v = r.get(section, {}).get(key)
        return "—" if v is None else f"{v * 100:.2f}"

    epochs = r.get("epochs") or []
    s_per_epoch = sum(e["wall_time_s"] for e in epochs) / len(epochs) if epochs else math.nan
    mems = [e["peak_gpu_mem_gb"] for e in epochs if e["peak_gpu_mem_gb"] is not None]
    rank = r.get("final", {}).get("eff_rank")
    status = "ok" if r["status"] == "ok" else f"FAILED: {r.get('error', '')[:60]}"
    return (
        f"| {r['model']} | {status} | {len(epochs)} | {s_per_epoch:.1f} "
        f"| {f'{max(mems):.2f}' if mems else '—'} | {pct('untrained', 'test_acc_linear')} "
        f"| {pct('final', 'test_acc_linear')} | {pct('final', 'test_acc_knn')} "
        f"| {'—' if rank is None else f'{rank:.1f}'} |"
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, train_idx, val_idx, test_idx, num_classes = load_arxiv(args.data_dir)
    print(
        f"\nogbn-arxiv | nodes={data.num_nodes:,} edges={data.num_edges:,} "
        f"features={data.num_features} classes={num_classes} | {device}\n",
        flush=True,
    )
    dm = DataModule(
        data=data,
        is_graph_level=False,
        batch_size=args.batch,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        num_workers=args.workers,
    )
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "git": git_info(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }

    reports = []
    for model_name in args.model:
        print(f"--- {model_name} ---", flush=True)
        report = run_model(model_name, args, dm, num_classes, device)
        probe = LogRegEvaluator()
        hyperparameters = {
            **vars(args),
            "linear_probe": {
                "standardize": probe.standardize,
                "weight_decays": list(probe.weight_decays),
                "max_iter": probe.max_iter,
            },
        }
        report.update(dataset="ogbn-arxiv", hyperparameters=hyperparameters, provenance=provenance)
        path = out_dir / f"{model_name}__{provenance['timestamp_utc']}.json"
        path.write_text(json.dumps(report, indent=2))
        print(
            f"  -> {report['status']} in {report['wall_time_s']:.0f}s | saved to {path}\n",
            flush=True,
        )
        reports.append(report)

    header = (
        "| Model | Status | Epochs | s/epoch | Peak GPU (GiB) | Untrained linear "
        "| Linear | kNN | Rank |\n|---|---|---|---|---|---|---|---|---|"
    )
    table = "\n".join([header, *(summary_row(r) for r in reports)])
    (out_dir / "summary.md").write_text(table + "\n")
    print(table)
    if any(r["status"] != "ok" for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
