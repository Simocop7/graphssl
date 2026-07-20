"""Multi-seed BGRL benchmark on Planetoid datasets (Cora, CiteSeer, PubMed).

Reproduces the evaluation protocol described in the paper (GIN-2L encoder,
hidden_dim=256, linear probing + KNN(k=5)), averaged over multiple seeds with
the standard public Planetoid train/val/test split.

Usage::

    python examples/benchmark_planetoid.py --dataset Cora
    python examples/benchmark_planetoid.py --dataset CiteSeer --seeds 10
    python examples/benchmark_planetoid.py --dataset PubMed --seeds 10 --epochs 300
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch.optim import AdamW
from torch_geometric.datasets import Planetoid

from graphssl.config.load import build_model
from graphssl.data import DataModule
from graphssl.evaluation import LogRegEvaluator, KNNEvaluator, extract_embeddings
from graphssl.training import DINOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed BGRL benchmark on Planetoid datasets")
    p.add_argument("--dataset",  default="Cora", choices=["Cora", "CiteSeer", "PubMed"])
    p.add_argument("--encoder",  default="gin", choices=["gin", "gcn", "transformer"])
    p.add_argument("--hidden",   type=int, default=256)
    p.add_argument("--layers",   type=int, default=2)
    p.add_argument("--epochs",   type=int, default=300)
    p.add_argument("--lr",       type=float, default=5e-4)
    p.add_argument("--seeds",    type=int, default=10)
    p.add_argument("--knn-k",    type=int, default=5)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_config(args: argparse.Namespace) -> dict:
    return {
        "name": "bgrl",
        "encoder": {
            "name": args.encoder,
            "hidden_dim": args.hidden,
            "num_layers": args.layers,
            "norm_type": "batch",
            "pool": False,   # node-level task: no graph pooling
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


def run_seed(args, data, num_classes, train_idx, val_idx, test_idx, seed, device):
    torch.manual_seed(seed)

    dm = DataModule(
        data=data,
        is_graph_level=False,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    loader = dm.train_dataloader()

    model = build_model(make_config(args), in_channels=data.num_features)
    optimizer = AdamW(model.student_parameters(), lr=args.lr, weight_decay=1e-5)
    trainer = DINOTrainer(grad_clip_norm=None, device=device)
    trainer.train(model, loader, optimizer, num_epochs=args.epochs)

    z, y = extract_embeddings(model, dm, device=device)

    lin = LogRegEvaluator().evaluate(z, y, train_idx, val_idx, test_idx, num_classes=num_classes)
    knn = KNNEvaluator(k=args.knn_k).evaluate(z, y, train_idx, val_idx, test_idx)
    return lin["val_acc"], lin["test_acc"], knn["val_acc"], knn["test_acc"]


def fmt(xs: list[float]) -> str:
    if len(xs) > 1:
        return f"{statistics.mean(xs) * 100:.2f} ± {statistics.stdev(xs) * 100:.2f}"
    return f"{xs[0] * 100:.2f}"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset = Planetoid(root=f"{args.data_dir}/{args.dataset}", name=args.dataset)
    data = dataset[0]
    num_classes = dataset.num_classes

    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    val_idx   = data.val_mask.nonzero(as_tuple=True)[0]
    test_idx  = data.test_mask.nonzero(as_tuple=True)[0]

    print(f"\n{args.dataset} / BGRL | encoder={args.encoder} d={args.hidden} | "
          f"seeds={args.seeds} | epochs={args.epochs} | {device}")
    print(f"Nodes: {data.num_nodes:,} | Edges: {data.num_edges:,} | "
          f"Features: {data.num_features} | Classes: {num_classes}")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}\n")

    lin_val, lin_test, knn_val, knn_test = [], [], [], []
    t0 = time.time()
    for seed in range(args.seeds):
        ts = time.time()
        lv, lt, kv, kt = run_seed(args, data, num_classes, train_idx, val_idx, test_idx, seed, device)
        lin_val.append(lv); lin_test.append(lt); knn_val.append(kv); knn_test.append(kt)
        print(f"  seed {seed}: linear test={lt:.4f} | knn test={kt:.4f} | {time.time() - ts:.1f}s")

    print(f"\n{args.dataset} results over {args.seeds} seeds ({time.time() - t0:.1f}s total):")
    print(f"  Linear probe (Adam) — val: {fmt(lin_val)} | test: {fmt(lin_test)}")
    print(f"  KNN (k={args.knn_k})            — val: {fmt(knn_val)} | test: {fmt(knn_test)}")


if __name__ == "__main__":
    main()
