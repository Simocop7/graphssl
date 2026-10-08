"""Every SSL method on ZINC-12k: molecules, graph-level, regression.

Each method pre-trains the encoder on the training molecules without labels; the encoder is
then frozen and a ridge regression on its graph embeddings predicts the target (constrained
solubility). Lower test MAE is better.

Shared, untuned protocol — it compares the objectives at equal budget, like run_benchmark.py:

- ZINC subset (10,000 / 1,000 / 1,000 molecules), atom type (28 classes) and bond type
  (4 classes) as categorical inputs;
- encoder: GIN with edge features (GINE), 4 layers, 128 units, batch norm, mean readout;
- pre-training: 100 epochs, batch 256, AdamW (lr 1e-3, weight decay 1e-5);
- augmentations (GraphCL, VICReg, Barlow Twins, BGRL, GraphDINO): 20% of the atoms replaced
  by a mask token + 20% of the bonds dropped; DGI shuffles the atoms, AFGRL uses none;
- evaluation: ``RidgeEvaluator`` fitted on the training molecules, L2 strength selected on
  validation, MAE on test; effective rank of the embeddings.

Two references, both with the same encoder:

- the encoder left untrained, evaluated the same way (saved with every run);
- ``--model supervised``: trained end to end on the labels with an L1 loss; its test MAE is
  the head's, at the epoch with the best validation MAE. It is not an SSL method.

``--finetune-epochs N`` adds the usual protocol for molecules after pre-training: the
pre-trained encoder and a new linear head are trained on the labels for N epochs (same
optimizer, L1 loss, best-validation epoch). Compare it with the supervised reference, which
is the same training from a random initialisation.

Usage::

    pip install "graphssl[benchmark]"
    # smoke test
    python benchmarks/run_zinc.py --model bgrl --epochs 2 --seeds 1 --out-dir benchmarks/zinc/smoke
    # the real run, inside tmux
    python -u benchmarks/run_zinc.py --model all supervised --seeds 3 --finetune-epochs 100 \\
        2>&1 | tee -a zinc.log

One JSON per model goes to ``--out-dir`` (default ``benchmarks/zinc``), next to a
``summary.md`` table of every result found there.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch_geometric
from stress_ogbn_arxiv import git_info
from torch.optim import AdamW
from torch_geometric.datasets import ZINC

from graphssl.config.load import build_model
from graphssl.core import pretrained_encoder
from graphssl.core.callback import Callback
from graphssl.data import DataModule
from graphssl.evaluation import RidgeEvaluator, effective_rank, extract_embeddings
from graphssl.training import DINOTrainer
from graphssl.utils.positive_miner import HAS_FAISS

SSL_MODELS = ["dgi", "graphcl", "vicreg", "barlow_twins", "bgrl", "afgrl", "graphdino"]
N_ATOM_TYPES, N_BOND_TYPES = 28, 4
MASK_TOKEN = N_ATOM_TYPES  # one more class than the data has, used only by attr_mask
DISPLAY = {
    "dgi": "DGI",
    "graphcl": "GraphCL",
    "vicreg": "VICReg",
    "barlow_twins": "Barlow Twins",
    "bgrl": "BGRL",
    "afgrl": "AFGRL",
    "graphdino": "GraphDINO",
    "supervised": "*Supervised* †",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model",
        nargs="+",
        default=["all"],
        choices=[*SSL_MODELS, "supervised", "all"],
        help="'all' expands to every SSL model (not the supervised reference)",
    )
    p.add_argument("--encoder", default="gin", choices=["gin", "transformer"])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument(
        "--readout",
        default="mean",
        choices=["mean", "sum", "max"],
        help="how node embeddings are pooled into a graph embedding",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--finetune-epochs",
        type=int,
        default=0,
        help="after pre-training, train encoder + head on the labels for N epochs (0: skip)",
    )
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-dir", default="data/ZINC")
    p.add_argument("--out-dir", default="benchmarks/zinc")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    models: List[str] = []
    for name in args.model:
        models += SSL_MODELS if name == "all" else [name]
    args.model = list(dict.fromkeys(models))
    return args


def make_config(model_name: str, args: argparse.Namespace, total_steps: int) -> dict:
    augment = [
        {"name": "attr_mask", "p": 0.2, "mask_value": MASK_TOKEN},
        {"name": "edge_drop", "p": 0.2},
    ]
    cfg: dict = {
        "name": model_name,
        "encoder": {
            "name": args.encoder,
            "hidden_dim": args.hidden,
            "num_layers": args.layers,
            "norm_type": "batch",
            "pool": True,  # graph-level task
            "readout": args.readout,
            "drop": 0.0,
            "node_emb_num_classes": N_ATOM_TYPES + 1,
            "edge_dim": args.hidden,
            "edge_emb_num_classes": N_BOND_TYPES,
        },
    }
    if model_name == "supervised":
        cfg["task"] = "regression"
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


def evaluate(model: torch.nn.Module, splits: Dict[str, DataModule], device: str) -> dict:
    """Ridge-probe MAE and effective rank of the frozen graph embeddings."""
    z_parts, y_parts, idx, start = [], [], {}, 0
    for name in ("train", "val", "test"):
        z, y = extract_embeddings(model, splits[name], device=device)
        assert y is not None
        z_parts.append(z)
        y_parts.append(y.float().view(z.size(0), -1))
        idx[name] = torch.arange(start, start + z.size(0))
        start += z.size(0)
    z, y = torch.cat(z_parts), torch.cat(y_parts)
    if not torch.isfinite(z).all():
        raise FloatingPointError("non-finite values in the extracted embeddings")
    probe = RidgeEvaluator().evaluate(z, y, idx["train"], idx["val"], idx["test"])
    return {
        "val_mae": probe["val_mae"],
        "test_mae": probe["test_mae"],
        "probe_weight_decay": probe["weight_decay"],
        "eff_rank": effective_rank(z),
    }


@torch.no_grad()
def head_mae(model: torch.nn.Module, dm: DataModule, device: torch.device) -> float:
    """MAE of the supervised model's own head on one split."""
    model.eval()
    error, count = 0.0, 0
    for batch in dm.eval_dataloader():
        batch = batch.to(device)
        pred = model.head(model(batch)).squeeze(-1)
        error += (pred - batch.y.float()).abs().sum().item()
        count += batch.y.numel()
    return error / count


class BestValidation(Callback):
    """Supervised reference: test MAE of the head at the epoch with the best validation MAE."""

    def __init__(self, splits: Dict[str, DataModule], device: torch.device):
        self.splits = splits
        self.device = device
        self.best: Dict[str, Any] = {}

    def on_epoch_end(
        self, trainer: Any, model: torch.nn.Module, epoch: int, metrics: Dict[str, Any]
    ) -> None:
        val = head_mae(model, self.splits["val"], self.device)
        if not self.best or val < self.best["val_mae_head"]:
            self.best = {
                "val_mae_head": val,
                "test_mae_head": head_mae(model, self.splits["test"], self.device),
                "best_epoch": epoch + 1,
            }
        model.train()


def finetune(
    pretrained: torch.nn.Module, args: argparse.Namespace, splits: Dict[str, DataModule], seed: int
) -> dict:
    """Train the pre-trained encoder and a new head on the labels; best-validation epoch."""
    device = torch.device(args.device)
    torch.manual_seed(seed)
    model = build_model(make_config("supervised", args, 1), in_channels=1, num_classes=1)
    model.encoder.load_state_dict(pretrained_encoder(pretrained).state_dict())
    best = BestValidation(splits, device)
    optimizer = AdamW(model.student_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = DINOTrainer(grad_clip_norm=None, device=device, callbacks=[best])
    trainer.train(
        model, splits["train"].train_dataloader(), optimizer, num_epochs=args.finetune_epochs
    )
    return {
        "val_mae_finetune": best.best["val_mae_head"],
        "test_mae_finetune": best.best["test_mae_head"],
        "finetune_best_epoch": best.best["best_epoch"],
    }


def run_one(
    model_name: str, args: argparse.Namespace, splits: Dict[str, DataModule], seed: int
) -> dict:
    device = torch.device(args.device)
    loader = splits["train"].train_dataloader()
    config = make_config(model_name, args, total_steps=max(args.epochs * len(loader), 1))
    torch.manual_seed(seed)
    # in_channels is ignored: the atom type goes through an embedding. num_classes = 1 target.
    model = build_model(config, in_channels=1, num_classes=1)
    result: Dict[str, Any] = {"seed": seed, "untrained": evaluate(model, splits, args.device)}

    best: Optional[BestValidation] = None
    callbacks: List[Callback] = []
    if model_name == "supervised":
        best = BestValidation(splits, device)
        callbacks.append(best)
    optimizer = AdamW(model.student_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = DINOTrainer(grad_clip_norm=None, device=device, callbacks=callbacks)
    t0 = time.time()
    losses = trainer.train(model, loader, optimizer, num_epochs=args.epochs)
    result["train_time_s"] = time.time() - t0
    if losses:
        result["first_loss"], result["last_loss"] = losses[0], losses[-1]
        if not all(torch.isfinite(torch.tensor(losses))):
            raise FloatingPointError("non-finite training loss")
    result.update(evaluate(model, splits, args.device))
    if best is not None:
        result.update(best.best)
    elif args.finetune_epochs > 0:
        result.update(finetune(model, args, splits, seed))
    return result


def mean_std(values: List[float]) -> Dict[str, float]:
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": statistics.mean(values), "std": std}


def fmt(agg: Optional[Dict[str, float]], digits: int = 3) -> str:
    return "—" if agg is None else f"{agg['mean']:.{digits}f} ± {agg['std']:.{digits}f}"


def write_summary(out_dir: Path) -> str:
    """Table of the latest result per model found in ``out_dir``."""
    latest: Dict[str, dict] = {}
    for path in sorted(out_dir.glob("*.json")):
        d = json.loads(path.read_text())
        prev = latest.get(d["model"])
        if prev is None or d["provenance"]["timestamp_utc"] > prev["provenance"]["timestamp_utc"]:
            latest[d["model"]] = d
    lines = [
        "| Method | Frozen encoder | Fine-tuned | Untrained encoder | Rank | Epochs | Seeds |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in [*SSL_MODELS, "supervised"]:
        if name not in latest:
            continue
        d = latest[name]
        agg = d["aggregate"]
        # The supervised reference is the same training from a random initialisation.
        tuned = agg.get("test_mae_head") if name == "supervised" else agg.get("test_mae_finetune")
        lines.append(
            f"| {DISPLAY[name]} | {fmt(agg['test_mae'])} | {fmt(tuned)} "
            f"| {fmt(agg['untrained_test_mae'])} | {fmt(agg['eff_rank'], 1)} "
            f"| {d['hyperparameters']['epochs']} | {len(d['seeds'])} |"
        )
    lines += [
        "",
        "Test MAE, lower is better; mean ± sample std over the seeds. *Frozen encoder* and "
        "*Untrained encoder*: ridge regression on the graph embeddings. *Fine-tuned*: encoder "
        "and a linear head trained on the labels after pre-training, best-validation epoch.",
    ]
    if "supervised" in latest:
        lines += [
            "",
            "† Not an SSL method: trained end to end on the labels from a random "
            "initialisation. Its *Fine-tuned* column is that training, the reference for "
            "the others.",
        ]
    table = "\n".join(lines)
    (out_dir / "summary.md").write_text(table + "\n")
    return table


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        name: DataModule(
            dataset_obj=ZINC(root=args.data_dir, subset=True, split=name),
            is_graph_level=True,
            batch_size=args.batch,
            num_workers=args.workers,
        )
        for name in ("train", "val", "test")
    }
    sizes = " / ".join(str(len(splits[name].dataset_obj)) for name in ("train", "val", "test"))
    print(f"\nZINC-12k | {sizes} molecules | {args.device}\n", flush=True)

    probe = RidgeEvaluator()
    for model_name in args.model:
        if model_name == "afgrl" and not HAS_FAISS:
            print("[afgrl] skipped — faiss-cpu not installed", flush=True)
            continue
        print(f"--- {model_name} ({args.seeds} seeds) ---", flush=True)
        per_seed = []
        for seed in range(args.seeds):
            r = run_one(model_name, args, splits, seed)
            per_seed.append(r)
            head = f" | head {r['test_mae_head']:.3f}" if "test_mae_head" in r else ""
            if "test_mae_finetune" in r:
                head = f" | fine-tuned {r['test_mae_finetune']:.3f}"
            print(
                f"  seed {seed}: test MAE {r['test_mae']:.3f} (untrained "
                f"{r['untrained']['test_mae']:.3f}){head} | rank {r['eff_rank']:.1f} "
                f"| {r['train_time_s']:.0f}s",
                flush=True,
            )
        aggregate = {
            "test_mae": mean_std([r["test_mae"] for r in per_seed]),
            "untrained_test_mae": mean_std([r["untrained"]["test_mae"] for r in per_seed]),
            "eff_rank": mean_std([r["eff_rank"] for r in per_seed]),
        }
        for key in ("test_mae_head", "test_mae_finetune"):
            if key in per_seed[0]:
                aggregate[key] = mean_std([r[key] for r in per_seed])
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report = {
            "dataset": "ZINC-12k",
            "model": model_name,
            "model_config": make_config(model_name, args, total_steps=1),
            "hyperparameters": {
                **vars(args),
                "ridge_probe": {
                    "standardize": probe.standardize,
                    "weight_decays": list(probe.weight_decays),
                },
            },
            "seeds": [r["seed"] for r in per_seed],
            "per_seed": per_seed,
            "aggregate": aggregate,
            "provenance": {
                "timestamp_utc": timestamp,
                "git": git_info(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_geometric": torch_geometric.__version__,
                "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            },
        }
        path = out_dir / f"{model_name}__{timestamp}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"  -> test MAE {fmt(aggregate['test_mae'])} | saved to {path}\n", flush=True)

    print(write_summary(out_dir))


if __name__ == "__main__":
    main()
