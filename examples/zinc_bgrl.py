"""BGRL benchmark on ZINC-12k (graph-level SSL, molecular regression).

ZINC atom features are integer atom types (28 classes) and edge features are
integer bond types (4 classes). The linear probe evaluates test MAE.

Usage::

    pip install "graphssl[benchmark]"
    python examples/zinc_bgrl.py
    python examples/zinc_bgrl.py --encoder transformer --hidden 128 --layers 4
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.datasets import ZINC

from graphssl.config.load import build_model
from graphssl.data import DataModule
from graphssl.training import DINOTrainer
from graphssl.evaluation import extract_embeddings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BGRL on ZINC-12k")
    p.add_argument("--encoder",  default="gin", choices=["gin", "transformer"])
    p.add_argument("--hidden",   type=int, default=64)
    p.add_argument("--layers",   type=int, default=4)
    p.add_argument("--epochs",   type=int, default=100)
    p.add_argument("--batch",    type=int, default=128)
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--data-dir", default="data/ZINC")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
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
            # ZINC: 28 atom types (nodes) + 4 bond types (edges)
            "node_emb_num_classes": 28,
            "edge_dim": args.hidden,
            "edge_emb_num_classes": 4,
        },
        "augment": [
            {"name": "edge_drop", "p": 0.3},
            {"name": "feat_mask", "p": 0.1},
        ],
        "pred_hidden": args.hidden * 2,
        "ema_tau": 0.99,
        "ema_tau_end": 1.0,
        "total_steps": args.epochs * (12000 // args.batch + 1),
    }


def linear_probe_mae(
    model: nn.Module,
    train_dm: DataModule,
    test_dm: DataModule,
    hidden: int,
    device: torch.device,
    epochs: int = 100,
) -> float:
    """Fit a linear regression head on frozen embeddings and return test MAE."""
    model.eval()
    z_train, y_train = extract_embeddings(model, train_dm, device=device)
    z_test,  y_test  = extract_embeddings(model, test_dm,  device=device)
    y_train = y_train.float().view(-1, 1)
    y_test  = y_test.float().view(-1, 1)

    head = nn.Linear(hidden, 1).to(device)
    opt  = AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        head.train()
        loss = nn.functional.l1_loss(head(z_train), y_train)
        opt.zero_grad(); loss.backward(); opt.step()

    head.eval()
    with torch.no_grad():
        return nn.functional.l1_loss(head(z_test), y_test).item()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"\nZINC-12k / BGRL | encoder={args.encoder} | {device}\n")

    train_ds = ZINC(root=args.data_dir, subset=True, split="train")
    test_ds  = ZINC(root=args.data_dir, subset=True, split="test")

    train_dm = DataModule(dataset_obj=train_ds, is_graph_level=True, batch_size=args.batch, num_workers=4)
    test_dm  = DataModule(dataset_obj=test_ds,  is_graph_level=True, batch_size=args.batch, num_workers=4)
    train_loader = train_dm.train_dataloader()

    # in_channels is ignored when node_emb_num_classes is set
    model = build_model(make_config(args), in_channels=1)
    optimizer = AdamW(model.student_parameters(), lr=args.lr, weight_decay=1e-5)
    trainer = DINOTrainer(grad_clip_norm=None, device=device)

    t0 = time.time()
    losses = trainer.train(model, train_loader, optimizer, num_epochs=args.epochs)
    print(f"Training done in {time.time() - t0:.1f}s | final SSL loss: {losses[-1]:.4f}\n")

    mae = linear_probe_mae(model, train_dm, test_dm, args.hidden, device)
    print(f"Test MAE (linear probe): {mae:.4f}\n")


if __name__ == "__main__":
    main()
