"""Unified multi-seed, multi-model benchmark runner for node-classification datasets.

Produces one reproducible JSON result file per (dataset, model) combination
under ``benchmarks/results/``, with per-seed metrics, aggregate mean/std, the
exact hyperparameters used, and enough provenance (git commit, package
versions, timestamp) to reproduce the run later.

This replaces copy-pasting a new per-dataset/per-model script every time —
compare it to ``examples/benchmark_planetoid.py``, which does the same thing
but only for BGRL. ``examples/*.py`` remain as minimal single-file "learn the
API" demos for onboarding and are not superseded by this script; this one is
for producing citable, comparable numbers across the whole model zoo.

Usage::

    # One model, one dataset, quick sweep
    python benchmarks/run_benchmark.py --dataset Cora --model bgrl --seeds 5

    # Cross-method comparison: every SSL model, same encoder/budget, same dataset
    python benchmarks/run_benchmark.py --dataset Cora --model all --seeds 10

    # Multiple datasets in one invocation
    python benchmarks/run_benchmark.py --dataset Cora CiteSeer PubMed --model all

Note: ``--model supervised`` is accepted but currently skipped at run time —
see the comment in ``main()`` for why (a real masking gap surfaced while
building this, not a placeholder).

Then render Markdown tables from the saved results::

    python benchmarks/render_tables.py
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_geometric
from torch.optim import AdamW
from torch_geometric.datasets import Planetoid

from graphssl.config.load import build_model
from graphssl.data import DataModule
from graphssl.evaluation import KNNEvaluator, LogRegEvaluator, extract_embeddings
from graphssl.training import DINOTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent

# All 7 SSL models ("all" expands to this). "supervised" is deliberately not
# included in "all" — it's a baseline with a different training contract
# (needs labels during "pretraining"), not an SSL method to compare against
# the others by default. Pass it explicitly via --model to include it.
SSL_MODELS = ["dgi", "graphcl", "vicreg", "barlow_twins", "bgrl", "afgrl", "graphdino"]
ALL_MODEL_CHOICES = [*SSL_MODELS, "supervised"]

# Only Planetoid citation networks are wired up today (CPU-friendly, already
# validated — see CLAUDE.md). Add an entry here to extend to a new dataset;
# everything else in this script (model construction, training, evaluation,
# result storage) is already dataset-agnostic.
PLANETOID_DATASETS = ["Cora", "CiteSeer", "PubMed"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", nargs="+", default=["Cora"], choices=PLANETOID_DATASETS)
    p.add_argument(
        "--model",
        nargs="+",
        default=["all"],
        choices=[*ALL_MODEL_CHOICES, "all"],
        help="'all' expands to every SSL model (excludes 'supervised', pass it explicitly)",
    )
    p.add_argument("--encoder", default="gin", choices=["gin", "gcn", "transformer"])
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="benchmarks/results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if "all" in args.model:
        args.model = SSL_MODELS
    return args


# ---------------------------------------------------------------------------
# Per-model config construction — reuses each model's own dataclass defaults
# (schema.py) wherever possible instead of re-declaring every hyperparameter
# here; only fields that must vary with --hidden/--epochs are set explicitly.
# ---------------------------------------------------------------------------


def build_model_config(model_name: str, encoder_cfg: dict, args: argparse.Namespace) -> dict:
    augment = [{"name": "edge_drop", "p": 0.5}, {"name": "feat_mask", "p": 0.2}]
    cfg: dict = {"name": model_name, "encoder": encoder_cfg}

    if model_name in ("bgrl", "afgrl", "graphdino"):
        cfg["ema_tau"] = 0.99
        cfg["ema_tau_end"] = 1.0
        cfg["total_steps"] = args.epochs

    if model_name == "graphdino":
        cfg["ema_tau"] = 0.996
        cfg["ema_tau_base"] = 0.996
        cfg["freeze_last_layer_epochs"] = min(1, args.epochs)
        cfg["augment_teacher"] = augment
        cfg["augment_student"] = augment
        cfg["head"] = {
            "name": "dino",
            "proj_hidden": args.hidden,
            "bottleneck_dim": max(args.hidden // 4, 8),
            "n_prototypes": 128,
            "warmup_teacher_temp_epochs": min(30, args.epochs // 4),
        }
        return cfg

    if model_name in ("bgrl", "graphcl", "vicreg", "barlow_twins"):
        cfg["augment"] = augment
    if model_name in ("bgrl", "afgrl"):
        cfg["pred_hidden"] = args.hidden
    # dgi and supervised need nothing beyond `encoder` — their dataclasses
    # already default corruption/shuffle_ratio and nothing extra, respectively.
    return cfg


# ---------------------------------------------------------------------------
# One (dataset, model, seed) run
# ---------------------------------------------------------------------------


def run_one(
    model_name: str,
    args: argparse.Namespace,
    data,
    num_classes: int,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)

    encoder_cfg = {
        "name": args.encoder,
        "hidden_dim": args.hidden,
        "num_layers": args.layers,
        "norm_type": "batch",
        "pool": False,  # node-level task: no graph pooling
        "drop": 0.0,
    }
    config = build_model_config(model_name, encoder_cfg, args)

    dm = DataModule(
        data=data,
        is_graph_level=False,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    t0 = time.time()
    model = build_model(config, in_channels=data.num_features)
    loader = dm.train_dataloader()

    optimizer = AdamW(model.student_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = DINOTrainer(grad_clip_norm=None, device=device)
    trainer.train(model, loader, optimizer, num_epochs=args.epochs)
    wall_time_s = time.time() - t0

    z, y = extract_embeddings(model, dm, device=device)
    lin = LogRegEvaluator().evaluate(z, y, train_idx, val_idx, test_idx, num_classes=num_classes)
    knn = KNNEvaluator(k=args.knn_k).evaluate(z, y, train_idx, val_idx, test_idx)

    return {
        "seed": seed,
        "val_acc_linear": lin["val_acc"],
        "test_acc_linear": lin["test_acc"],
        "val_acc_knn": knn["val_acc"],
        "test_acc_knn": knn["test_acc"],
        "wall_time_s": wall_time_s,
    }


# ---------------------------------------------------------------------------
# Provenance + result storage
# ---------------------------------------------------------------------------


def _git_info() -> dict:
    def _run(cmd: list[str]) -> str | None:
        try:
            return subprocess.run(
                cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def _mean_std(values: list[float]) -> dict:
    if len(values) > 1:
        return {"mean": statistics.mean(values), "std": statistics.stdev(values)}
    return {"mean": values[0], "std": 0.0}


def save_result(
    out_dir: Path, dataset: str, model_name: str, args: argparse.Namespace, per_seed: list[dict]
) -> Path:
    out_dir = out_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{model_name}__{timestamp}.json"

    result = {
        "dataset": dataset,
        "model": model_name,
        "hyperparameters": {
            "encoder": args.encoder,
            "hidden_dim": args.hidden,
            "num_layers": args.layers,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "knn_k": args.knn_k,
        },
        "seeds": [r["seed"] for r in per_seed],
        "per_seed": per_seed,
        "aggregate": {
            "test_acc_linear": _mean_std([r["test_acc_linear"] for r in per_seed]),
            "test_acc_knn": _mean_std([r["test_acc_knn"] for r in per_seed]),
        },
        "provenance": {
            "timestamp_utc": timestamp,
            "git": _git_info(),
            "versions": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_geometric": torch_geometric.__version__,
            },
        },
    }
    path.write_text(json.dumps(result, indent=2))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_planetoid(name: str, data_dir: str):
    dataset = Planetoid(root=f"{data_dir}/{name}", name=name)
    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    val_idx = data.val_mask.nonzero(as_tuple=True)[0]
    test_idx = data.test_mask.nonzero(as_tuple=True)[0]
    return data, dataset.num_classes, train_idx, val_idx, test_idx


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = REPO_ROOT / args.out_dir

    for dataset_name in args.dataset:
        data, num_classes, train_idx, val_idx, test_idx = load_planetoid(
            dataset_name, args.data_dir
        )
        print(
            f"\n=== {dataset_name} | nodes={data.num_nodes:,} edges={data.num_edges:,} "
            f"features={data.num_features} classes={num_classes} ==="
        )

        for model_name in args.model:
            if model_name == "afgrl":
                try:
                    import faiss  # noqa: F401
                except ImportError:
                    print(f"[{dataset_name}/afgrl] skipped — faiss-cpu not installed")
                    continue

            if model_name == "supervised":
                # Not yet supported here: Supervised.compute_loss() has no
                # masking of its own in full-batch mode (it only crops to
                # `batch.batch_size` when that attribute is present — see
                # src/graphssl/models/supervised.py) — a plain
                # train_dataloader() full-graph Data object would leak
                # val/test labels into the loss. The correct fix is
                # NeighborLoader(input_nodes=train_idx), which in turn needs
                # pyg-lib or torch-sparse installed (neither is a core
                # dependency) — verified by hitting exactly this ImportError
                # in a real run. Tracked as a follow-up rather than adding
                # that dependency or a fragile workaround here.
                print(
                    f"[{dataset_name}/supervised] skipped — see comment above run_benchmark.main()"
                )
                continue

            print(f"\n--- {dataset_name} / {model_name} ({args.seeds} seeds) ---")
            per_seed = []
            for seed in range(args.seeds):
                t0 = time.time()
                r = run_one(
                    model_name, args, data, num_classes, train_idx, val_idx, test_idx, seed, device
                )
                per_seed.append(r)
                print(
                    f"  seed {seed}: linear test={r['test_acc_linear']:.4f} | "
                    f"knn test={r['test_acc_knn']:.4f} | {time.time() - t0:.1f}s"
                )

            path = save_result(out_dir, dataset_name, model_name, args, per_seed)
            agg = _mean_std([r["test_acc_linear"] for r in per_seed])
            agg_knn = _mean_std([r["test_acc_knn"] for r in per_seed])
            print(
                f"  -> linear {agg['mean'] * 100:.2f} ± {agg['std'] * 100:.2f} | "
                f"knn {agg_knn['mean'] * 100:.2f} ± {agg_knn['std'] * 100:.2f} | saved to {path.relative_to(REPO_ROOT)}"
            )


if __name__ == "__main__":
    main()
