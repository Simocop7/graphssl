"""BGRL benchmark on ZINC-12k (graph-level SSL, molecular regression).

ZINC atom features are integer atom types (28 classes) and edge features are
integer bond types (4 classes). A ridge regression on the frozen graph embeddings
gives the test MAE; ``benchmarks/run_zinc.py`` runs every method this way.

Usage::

    pip install "graphssl[benchmark]"
    python examples/zinc_bgrl.py
    python examples/zinc_bgrl.py --encoder transformer --hidden 128 --layers 4
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.optim import AdamW
from torch_geometric.datasets import ZINC

from graphssl.config.load import build_model
from graphssl.data import DataModule
from graphssl.evaluation import RidgeEvaluator, extract_embeddings
from graphssl.training import DINOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BGRL on ZINC-12k")
    p.add_argument("--encoder", default="gin", choices=["gin", "transformer"])
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--data-dir", default="data/ZINC")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_config(args: argparse.Namespace) -> dict:
    return {
        "name": "bgrl",
        "encoder": {
            "name": args.encoder,
            "hidden_dim": args.hidden,
            "num_layers": args.layers,
            "norm_type": "layer",
            "pool": True,
            "drop": 0.0,
            # ZINC: 28 atom types (+ 1 mask token used by attr_mask) and 4 bond types
            "node_emb_num_classes": 29,
            "edge_dim": args.hidden,
            "edge_emb_num_classes": 4,
        },
        "augment": [
            {"name": "edge_drop", "p": 0.2},
            # Hide 20% of the atoms behind the mask token. feat_mask would hide the single
            # atom-type column for every atom at once.
            {"name": "attr_mask", "p": 0.2, "mask_value": 28},
        ],
        "pred_hidden": args.hidden * 2,
        "ema_tau": 0.99,
        "ema_tau_end": 1.0,
        # The EMA schedule spans the run: 10,000 training molecules per epoch.
        "total_steps": args.epochs * -(-10_000 // args.batch),
    }


def ridge_probe_mae(model: torch.nn.Module, splits: dict, device: torch.device) -> dict:
    """Ridge regression on frozen embeddings: L2 strength chosen on validation, MAE on test."""
    z_parts, y_parts, idx, start = [], [], {}, 0
    for name in ("train", "val", "test"):
        z, y = extract_embeddings(model, splits[name], device=device)
        z_parts.append(z)
        y_parts.append(y.float())
        idx[name] = torch.arange(start, start + z.size(0))
        start += z.size(0)
    return RidgeEvaluator().evaluate(
        torch.cat(z_parts), torch.cat(y_parts), idx["train"], idx["val"], idx["test"]
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"\nZINC-12k / BGRL | encoder={args.encoder} | {device}\n")

    splits = {
        name: DataModule(
            dataset_obj=ZINC(root=args.data_dir, subset=True, split=name),
            is_graph_level=True,
            batch_size=args.batch,
            num_workers=4,
        )
        for name in ("train", "val", "test")
    }
    train_loader = splits["train"].train_dataloader()

    # in_channels is ignored when node_emb_num_classes is set
    model = build_model(make_config(args), in_channels=1)
    optimizer = AdamW(model.student_parameters(), lr=args.lr, weight_decay=1e-5)
    trainer = DINOTrainer(grad_clip_norm=None, device=device)

    t0 = time.time()
    losses = trainer.train(model, train_loader, optimizer, num_epochs=args.epochs)
    print(f"Training done in {time.time() - t0:.1f}s | final SSL loss: {losses[-1]:.4f}\n")

    probe = ridge_probe_mae(model, splits, device)
    print(f"Val  MAE (ridge probe): {probe['val_mae']:.4f}")
    print(f"Test MAE (ridge probe): {probe['test_mae']:.4f}\n")


if __name__ == "__main__":
    main()
