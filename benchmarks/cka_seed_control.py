"""Control for the CKA grouping of analyze_representations.py: is the similarity of two models
with the same backbone shared architecture or shared initialization?

Models of one seed start from the same initial encoder weights whatever the objective, and the
per-seed CKA matrices of representation_metrics.py compare models of the same seed. This script
compares pairs of models across seeds as well, so that same-backbone pairs with different
initializations can be told apart from those with the same one. Linear CKA (Kornblith et al.,
2019) on centered embeddings, and on standardized ones (as the regularized probe sees them).

Writes one row per (dataset, embedding scaling, pair group) with the mean CKA, its standard
deviation over pairs and the number of pairs.

Usage::

    python benchmarks/cka_seed_control.py --dataset Cora --seeds 5 \\
        --out benchmarks/paper/analysis/cka_seed_control
"""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from node_datasets import load_node_dataset
from paper_figures import METHODS_V2

from graphssl.config.load import load_model
from graphssl.data.datamodule import DataModule
from graphssl.evaluation import extract_embeddings

BACKBONES = ["gcn", "gin", "transformer"]


def prepared(z: torch.Tensor, standardize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Centered (optionally standardized) float64 embeddings and the norm of their Gram matrix."""
    z = z.double()
    z = z - z.mean(0)
    if standardize:
        z = z / z.std(0).clamp_min(1e-12)
    return z, torch.linalg.matrix_norm(z.T @ z)


def cka(a: tuple, b: tuple) -> float:
    return float(torch.linalg.matrix_norm(b[0].T @ a[0]) ** 2 / (a[1] * b[1]))


def group(x: tuple, y: tuple) -> str | None:
    (b1, m1, s1), (b2, m2, s2) = x, y
    if "untrained" in (m1, m2):
        if b1 != b2:
            return None
        if m1 == m2:
            return "untrained, same backbone, other seed"
        # how far training moved a model from where it started (E5: 300 vs 3,000 steps)
        return f"trained vs its untrained encoder ({b1})" if s1 == s2 else None
    seed = "same seed" if s1 == s2 else "other seed"
    if b1 == b2 and m1 != m2:
        return f"same backbone, other objective, {seed}"
    if b1 == b2 and m1 == m2:
        return "same backbone, same objective, other seed"
    if m1 == m2:
        return f"other backbone, same objective, {seed}"
    return "neither"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--root", type=Path, default=Path("benchmarks/paper"))
    p.add_argument(
        "--models", default="models", help="trained models under --root (E5: models_e5/steps3000)"
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", type=Path, default=Path("benchmarks/paper/analysis/cka_seed_control"))
    args = p.parse_args()
    data, *_ = load_node_dataset(args.dataset, args.data_dir)
    dm = DataModule(data=data, is_graph_level=False)
    models = args.root / args.models / args.dataset
    # Every objective with a checkpoint for every backbone and seed; the others are named.
    methods = [
        m
        for m in METHODS_V2
        if all(
            (models / b / f"{m}_seed{s}.pt").exists() for b in BACKBONES for s in range(args.seeds)
        )
    ]
    left_out = [m for m in METHODS_V2 if m not in methods]
    print(f"objectives: {', '.join(methods)}" + (f"; left out: {left_out}" if left_out else ""))
    raw = {}
    for b in BACKBONES:
        for s in range(args.seeds):
            for m in methods:
                path = models / b / f"{m}_seed{s}.pt"
                raw[(b, m, s)] = extract_embeddings(load_model(path), dm, device="cpu")[0]
            path = args.root / "models_untrained" / args.dataset / b / f"bgrl_seed{s}.pt"
            raw[(b, "untrained", s)] = extract_embeddings(load_model(path), dm, device="cpu")[0]
    rows = []
    for standardize in (False, True):
        emb = {k: prepared(z, standardize) for k, z in raw.items()}
        groups = defaultdict(list)
        for x, y in itertools.combinations(emb, 2):
            g = group(x, y)
            if g is not None:
                groups[g].append(cka(emb[x], emb[y]))
        for g, values in sorted(groups.items()):
            rows.append(
                {
                    "dataset": args.dataset,
                    "embeddings": "standardized" if standardize else "centered",
                    "group": g,
                    "cka": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "pairs": len(values),
                }
            )
            print(f"{rows[-1]['embeddings']:12s} {g:46s} {rows[-1]['cka']:.3f} (n={len(values)})")
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / f"{args.dataset}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
