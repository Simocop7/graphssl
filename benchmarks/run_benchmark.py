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

    # Supervised references (not part of "all")
    python benchmarks/run_benchmark.py --dataset Cora --model supervised supervised_reg

Both supervised references train full-batch on the public split's ``train_mask``
only. ``supervised`` keeps the shared SSL protocol unchanged; ``supervised_reg``
uses the standard Kipf & Welling recipe (dropout 0.5, Adam lr 0.01 + L2 5e-4,
best-validation checkpoint), recorded in the JSON's hyperparameters. Their encoder
embeddings get the same linear probe + kNN evaluation; the accuracy of the trained
head is saved alongside as ``test_acc_head``.

Then render Markdown tables from the saved results::

    python benchmarks/render_tables.py

Every result also records the effective rank of the evaluated embeddings (``eff_rank``,
to spot dimensional collapse) and, for teacher-student models, the same metrics for the
other encoder (``*_alt``: BGRL/AFGRL target, GraphDINO student). Ablations go outside
``benchmarks/results/`` — e.g. ``--out-dir benchmarks/ablations/<name>`` together with
``--ema-tau``/``--epochs``/``--encoder`` — and are rendered with
``benchmarks/render_ablation.py``, which keeps every configuration.
"""

from __future__ import annotations

import argparse
import copy
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
from torch.optim import Adam, AdamW
from torch_geometric.datasets import Planetoid

from graphssl.config.load import build_model
from graphssl.data import DataModule
from graphssl.evaluation import (
    KNNEvaluator,
    LogRegEvaluator,
    effective_rank,
    extract_embeddings,
)
from graphssl.models import Supervised
from graphssl.training import DINOTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent

# All 7 SSL models ("all" expands to this).
SSL_MODELS = ["dgi", "graphcl", "vicreg", "barlow_twins", "bgrl", "afgrl", "graphdino"]

# Supervised references, deliberately not part of "all": they train on labels, so they're
# reference points rather than methods to rank. Pass them explicitly via --model.
#   supervised      the shared SSL protocol unchanged (no dropout, AdamW, last checkpoint).
#                   With 120-140 labels and no regularization it overfits, so it understates
#                   what the labels are worth.
#   supervised_reg  the standard semi-supervised recipe of Kipf & Welling (2017) below.
SUPERVISED_MODELS = ["supervised", "supervised_reg"]
SUPERVISED_REG_RECIPE = {
    "optimizer": "adam",  # L2 through Adam's coupled weight_decay, as in the GCN paper
    "lr": 0.01,
    "weight_decay": 5e-4,
    "dropout": 0.5,
    "model_selection": "best_val_acc",  # checkpoint with the best validation accuracy
}
ALL_MODEL_CHOICES = [*SSL_MODELS, *SUPERVISED_MODELS]

# Teacher-student models: model.forward() evaluates one encoder (online for BGRL/AFGRL,
# teacher for GraphDINO); the runner also evaluates the other one, saved with an "_alt"
# suffix. It costs one extra probe, no extra training.
ALT_ENCODER = {"bgrl": "target", "afgrl": "target", "graphdino": "student"}

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
        help="'all' expands to every SSL model (excludes the supervised references)",
    )
    p.add_argument("--encoder", default="gin", choices=["gin", "gcn", "transformer"])
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument(
        "--ema-tau",
        type=float,
        default=None,
        help="starting EMA momentum of the teacher (bgrl/afgrl: annealed to 1.0; graphdino: "
        "annealed to 0.996, fixed if higher). Default: the benchmark's per-model value",
    )
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--data-dir", default="data")
    p.add_argument(
        "--out-dir",
        default="benchmarks/results",
        help="where JSONs go; keep ablations out of benchmarks/results, whose latest file "
        "per (dataset, model) is what render_tables.py reports",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if "all" in args.model:
        args.model = SSL_MODELS
    if args.ema_tau is not None and not 0.0 < args.ema_tau < 1.0:
        p.error(f"--ema-tau must be in (0, 1), got {args.ema_tau}")
    return args


# ---------------------------------------------------------------------------
# Per-model config construction — reuses each model's own dataclass defaults
# (schema.py) wherever possible instead of re-declaring every hyperparameter
# here; only fields that must vary with --hidden/--epochs are set explicitly.
# ---------------------------------------------------------------------------


def make_config(model_name: str, args: argparse.Namespace) -> dict:
    """Full model config for one (model, args) combination — identical for every seed."""
    encoder_cfg = {
        "name": args.encoder,
        "hidden_dim": args.hidden,
        "num_layers": args.layers,
        "norm_type": "batch",
        "pool": False,  # node-level task: no graph pooling
        "drop": SUPERVISED_REG_RECIPE["dropout"] if model_name == "supervised_reg" else 0.0,
    }
    return build_model_config(model_name, encoder_cfg, args)


def build_model_config(model_name: str, encoder_cfg: dict, args: argparse.Namespace) -> dict:
    augment = [{"name": "edge_drop", "p": 0.5}, {"name": "feat_mask", "p": 0.2}]
    name = "supervised" if model_name in SUPERVISED_MODELS else model_name
    cfg: dict = {"name": name, "encoder": encoder_cfg}
    ema_tau = getattr(args, "ema_tau", None)

    if model_name in ("bgrl", "afgrl", "graphdino"):
        cfg["ema_tau"] = 0.99 if ema_tau is None else ema_tau
        cfg["ema_tau_end"] = 1.0
        cfg["total_steps"] = args.epochs

    if model_name == "graphdino":
        # GraphDINO names it the other way round: ema_tau_base -> ema_tau.
        cfg["ema_tau_base"] = 0.996 if ema_tau is None else ema_tau
        cfg["ema_tau"] = max(cfg["ema_tau_base"], 0.996)
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
    # already default corruption/shuffle_ratio and nothing extra, respectively
    # (Supervised's num_classes is passed to build_model() separately).
    return cfg


# ---------------------------------------------------------------------------
# One (dataset, model, seed) run
# ---------------------------------------------------------------------------


def evaluate_embeddings(
    z: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    knn_k: int,
    suffix: str = "",
) -> dict:
    """Linear probe + kNN accuracy and effective rank of frozen embeddings."""
    lin = LogRegEvaluator().evaluate(z, y, train_idx, val_idx, test_idx, num_classes=num_classes)
    knn = KNNEvaluator(k=knn_k).evaluate(z, y, train_idx, val_idx, test_idx)
    return {
        f"val_acc_linear{suffix}": lin["val_acc"],
        f"test_acc_linear{suffix}": lin["test_acc"],
        f"val_acc_knn{suffix}": knn["val_acc"],
        f"test_acc_knn{suffix}": knn["test_acc"],
        f"eff_rank{suffix}": effective_rank(z),
    }


def run_one(
    model_name: str,
    config: dict,
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

    dm = DataModule(
        data=data,
        is_graph_level=False,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    t0 = time.time()
    # num_classes is only used by Supervised; build_model ignores it for SSL models.
    # Supervised's full-batch loss is restricted to data.train_mask (the public split).
    model = build_model(config, in_channels=data.num_features, num_classes=num_classes)
    loader = dm.train_dataloader()

    trainer = DINOTrainer(grad_clip_norm=None, device=device)
    best_epoch = None
    if model_name == "supervised_reg":
        assert isinstance(model, Supervised)
        optimizer: torch.optim.Optimizer = Adam(
            model.student_parameters(),
            lr=SUPERVISED_REG_RECIPE["lr"],
            weight_decay=SUPERVISED_REG_RECIPE["weight_decay"],
        )
        best_epoch = train_best_val(
            model, loader, optimizer, trainer, data, val_idx, args.epochs, device
        )
    else:
        optimizer = AdamW(model.student_parameters(), lr=args.lr, weight_decay=args.weight_decay)
        trainer.train(model, loader, optimizer, num_epochs=args.epochs)
    wall_time_s = time.time() - t0

    splits = (num_classes, train_idx, val_idx, test_idx, args.knn_k)
    z, y = extract_embeddings(model, dm, device=device)
    result = {"seed": seed, **evaluate_embeddings(z, y, *splits), "wall_time_s": wall_time_s}
    alt = ALT_ENCODER.get(model_name)
    if alt is not None:
        z_alt, _ = extract_embeddings(model, dm, encoder_source=alt, device=device)
        result.update(evaluate_embeddings(z_alt, y, *splits, suffix="_alt"))
    if isinstance(model, Supervised):
        result.update(head_accuracy(model, data, val_idx, test_idx, device))
    if best_epoch is not None:
        result["best_epoch"] = best_epoch
    return result


@torch.no_grad()
def head_predictions(model: Supervised, data, device: torch.device) -> torch.Tensor:
    """Class predictions of Supervised's own trained head for every node (on CPU)."""
    model.eval()
    return model.head(model(data.to(device))).argmax(dim=-1).cpu()


def head_accuracy(
    model: Supervised,
    data,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    device: torch.device,
) -> dict:
    """Accuracy of Supervised's own trained head (end-to-end, no probe retraining)."""
    pred, y = head_predictions(model, data, device), data.y.cpu()
    return {
        "val_acc_head": (pred[val_idx] == y[val_idx]).float().mean().item(),
        "test_acc_head": (pred[test_idx] == y[test_idx]).float().mean().item(),
    }


def train_best_val(
    model: Supervised,
    loader,
    optimizer: torch.optim.Optimizer,
    trainer: DINOTrainer,
    data,
    val_idx: torch.Tensor,
    epochs: int,
    device: torch.device,
) -> int:
    """Train for ``epochs``, then restore the checkpoint with the best validation accuracy.

    Selection looks at validation labels only, never at the test split. Returns the
    selected epoch (the earliest one on ties).
    """
    model.to(device)
    y_val = data.y.cpu()[val_idx]
    best_acc, best_epoch, best_state = -1.0, -1, None
    for epoch in range(epochs):
        trainer.train_epoch(model, loader, optimizer, device=device, epoch=epoch)
        acc = (head_predictions(model, data, device)[val_idx] == y_val).float().mean().item()
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = copy.deepcopy(model.state_dict())
    assert best_state is not None
    model.load_state_dict(best_state)
    return best_epoch


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
    # Tracked files only: untracked outputs (results/*.json, *.log) don't change the
    # code that ran, and counting them would flag every run after the first as dirty.
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def _mean_std(values: list[float]) -> dict:
    if len(values) > 1:
        return {"mean": statistics.mean(values), "std": statistics.stdev(values)}
    return {"mean": values[0], "std": 0.0}


AGGREGATED_METRICS = [
    "test_acc_linear",
    "test_acc_knn",
    "eff_rank",
    "test_acc_head",
    "test_acc_linear_alt",
    "test_acc_knn_alt",
    "eff_rank_alt",
]


def save_result(
    out_dir: Path,
    dataset: str,
    model_name: str,
    config: dict,
    args: argparse.Namespace,
    per_seed: list[dict],
) -> Path:
    out_dir = out_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{model_name}__{timestamp}.json"

    aggregate = {
        metric: _mean_std([r[metric] for r in per_seed])
        for metric in AGGREGATED_METRICS
        if all(metric in r for r in per_seed)
    }

    hyperparameters = {
        "encoder": args.encoder,
        "hidden_dim": args.hidden,
        "num_layers": args.layers,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "knn_k": args.knn_k,
    }
    if model_name == "supervised_reg":
        # Its recipe replaces the shared optimizer settings: record what actually ran.
        hyperparameters.update(SUPERVISED_REG_RECIPE)
    if args.ema_tau is not None and model_name in ALT_ENCODER:
        hyperparameters["ema_tau"] = args.ema_tau

    result = {
        "dataset": dataset,
        "model": model_name,
        "hyperparameters": hyperparameters,
        # The exact dict passed to build_model(): everything needed to rebuild the model.
        "model_config": config,
        **({"alt_encoder": ALT_ENCODER[model_name]} if model_name in ALT_ENCODER else {}),
        "seeds": [r["seed"] for r in per_seed],
        "per_seed": per_seed,
        "aggregate": aggregate,
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
    path.write_text(json.dumps(result, indent=2) + "\n")
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

            if args.ema_tau is not None and model_name not in ALT_ENCODER:
                print(f"[{dataset_name}/{model_name}] note: --ema-tau has no effect here")
            config = make_config(model_name, args)
            print(f"\n--- {dataset_name} / {model_name} ({args.seeds} seeds) ---")
            per_seed = []
            for seed in range(args.seeds):
                t0 = time.time()
                r = run_one(
                    model_name,
                    config,
                    args,
                    data,
                    num_classes,
                    train_idx,
                    val_idx,
                    test_idx,
                    seed,
                    device,
                )
                per_seed.append(r)
                alt = ""
                if "test_acc_linear_alt" in r:
                    alt = (
                        f"{ALT_ENCODER[model_name]}: linear={r['test_acc_linear_alt']:.4f} "
                        f"rank={r['eff_rank_alt']:.1f} | "
                    )
                print(
                    f"  seed {seed}: linear test={r['test_acc_linear']:.4f} | "
                    f"knn test={r['test_acc_knn']:.4f} | rank {r['eff_rank']:.1f} | "
                    f"{alt}{time.time() - t0:.1f}s"
                )

            path = save_result(out_dir, dataset_name, model_name, config, args, per_seed)
            agg = _mean_std([r["test_acc_linear"] for r in per_seed])
            agg_knn = _mean_std([r["test_acc_knn"] for r in per_seed])
            head = ""
            if "test_acc_head" in per_seed[0]:
                agg_head = _mean_std([r["test_acc_head"] for r in per_seed])
                head = f"head {agg_head['mean'] * 100:.2f} ± {agg_head['std'] * 100:.2f} | "
            # --out-dir may point outside the repo (absolute path): print it as-is then.
            shown = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
            print(
                f"  -> linear {agg['mean'] * 100:.2f} ± {agg['std'] * 100:.2f} | "
                f"knn {agg_knn['mean'] * 100:.2f} ± {agg_knn['std'] * 100:.2f} | "
                f"{head}saved to {shown}"
            )


if __name__ == "__main__":
    main()
