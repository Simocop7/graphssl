"""BGRL benchmark on Cora (node-level SSL, 7-class classification).

Cora has 2,708 nodes and 1,433-dim bag-of-words features; the whole graph
fits in memory, so training runs full-batch (no NeighborLoader needed).
This is the fastest end-to-end example in the repository — useful as a
smoke test or a quick demo of the library on CPU.

Usage::

    python examples/cora_bgrl.py
    python examples/cora_bgrl.py --hidden 128 --epochs 300
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.optim import AdamW
from torch_geometric.datasets import Planetoid

from graphssl.config.load import build_model
from graphssl.data import DataModule
from graphssl.evaluation import LogRegEvaluator, extract_embeddings
from graphssl.training import DINOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BGRL on Cora")
    p.add_argument("--encoder", default="gin", choices=["gin", "gcn", "transformer"])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--data-dir", default="data/Cora")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_config(args: argparse.Namespace) -> dict:
    return {
        "name": "bgrl",
        "encoder": {
            "name": args.encoder,
            "hidden_dim": args.hidden,
            "num_layers": args.layers,
            "norm_type": "batch",
            "pool": False,  # node-level task: no graph pooling
            "drop": 0.0,
        },
        "augment": [
            {"name": "edge_drop", "p": 0.5},
            {"name": "feat_mask", "p": 0.2},
        ],
        "pred_hidden": args.hidden,
        "ema_tau": 0.99,
        "ema_tau_end": 1.0,
        "total_steps": args.epochs,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"\nCora / BGRL | encoder={args.encoder} | {device}\n")

    dataset = Planetoid(root=args.data_dir, name="Cora")
    data = dataset[0]
    num_classes = dataset.num_classes

    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    val_idx = data.val_mask.nonzero(as_tuple=True)[0]
    test_idx = data.test_mask.nonzero(as_tuple=True)[0]

    print(f"Nodes: {data.num_nodes:,} | Edges: {data.num_edges:,} | Features: {data.num_features}")
    print(f"Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}\n")

    dm = DataModule(
        data=data,
        is_graph_level=False,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    loader = dm.train_dataloader()  # full-batch: single Data object per "batch"

    model = build_model(make_config(args), in_channels=data.num_features)
    optimizer = AdamW(model.student_parameters(), lr=args.lr, weight_decay=1e-5)
    trainer = DINOTrainer(grad_clip_norm=None, device=device)

    t0 = time.time()
    losses = trainer.train(model, loader, optimizer, num_epochs=args.epochs)
    print(f"Training done in {time.time() - t0:.1f}s | final SSL loss: {losses[-1]:.4f}\n")

    z, y = extract_embeddings(model, dm, device=device)

    evaluator = LogRegEvaluator()
    results = evaluator.evaluate(z, y, train_idx, val_idx, test_idx, num_classes=num_classes)

    print(f"Val  Acc (linear probe): {results['val_acc']:.4f}")
    print(f"Test Acc (linear probe): {results['test_acc']:.4f}\n")


if __name__ == "__main__":
    main()
