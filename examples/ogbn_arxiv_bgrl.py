"""BGRL benchmark on ogbn-arxiv (node-level SSL, 40-class classification).

ogbn-arxiv has 169,343 nodes with 128-dim continuous features; its scale
requires NeighborLoader for mini-batch training. Linear evaluation reports
accuracy on the official train/val/test splits.

Usage::

    pip install "graphssl[benchmark]"
    python examples/ogbn_arxiv_bgrl.py
    python examples/ogbn_arxiv_bgrl.py --hidden 256 --layers 3 --epochs 500
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.optim import AdamW

from graphssl.config.load import build_model
from graphssl.data import DataModule
from graphssl.evaluation import LogRegEvaluator, extract_embeddings
from graphssl.training import DINOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BGRL on ogbn-arxiv")
    p.add_argument("--encoder", default="gin", choices=["gin", "gcn", "transformer"])
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", default="data/ogbn-arxiv")
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
            {"name": "feat_mask", "p": 0.1},
        ],
        "pred_hidden": args.hidden,
        "ema_tau": 0.99,
        "ema_tau_end": 1.0,
        "total_steps": args.epochs * 100,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"\nogbn-arxiv / BGRL | encoder={args.encoder} | {device}\n")

    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as e:
        raise ImportError("Install ogb: pip install ogb") from e

    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=args.data_dir)
    data = dataset[0]
    from torch_geometric.transforms import ToUndirected

    data = ToUndirected()(data)  # convert directed citation edges to undirected

    split_idx = dataset.get_idx_split()
    num_classes = dataset.num_classes

    print(f"Nodes: {data.num_nodes:,} | Edges: {data.num_edges:,} | Features: {data.num_features}")
    print(
        f"Train: {len(split_idx['train']):,} | Val: {len(split_idx['valid']):,} | Test: {len(split_idx['test']):,}\n"
    )

    dm = DataModule(
        data=data,
        is_graph_level=False,
        batch_size=args.batch,
        train_idx=split_idx["train"],
        val_idx=split_idx["valid"],
        test_idx=split_idx["test"],
        num_workers=4,
    )
    loader = dm.neighbor_loader(
        num_neighbors=[10] * args.layers,
        input_nodes=split_idx["train"],
    )

    model = build_model(make_config(args), in_channels=data.num_features)
    optimizer = AdamW(model.student_parameters(), lr=args.lr, weight_decay=1e-5)
    trainer = DINOTrainer(grad_clip_norm=None, device=device)

    t0 = time.time()
    losses = trainer.train(model, loader, optimizer, num_epochs=args.epochs)
    print(f"Training done in {time.time() - t0:.1f}s | final SSL loss: {losses[-1]:.4f}\n")

    model.eval()
    full_loader = dm.train_dataloader()
    z_full, y_full = extract_embeddings(model, full_loader, device=device)

    evaluator = LogRegEvaluator()
    results = evaluator.evaluate(
        z_full,
        y_full,
        train_idx=split_idx["train"],
        val_idx=split_idx["valid"],
        test_idx=split_idx["test"],
        num_classes=num_classes,
    )

    print(f"Val  Acc (linear probe): {results['val_acc']:.4f}")
    print(f"Test Acc (linear probe): {results['test_acc']:.4f}\n")


if __name__ == "__main__":
    main()
