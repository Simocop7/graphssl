"""What training did to the representation, beyond accuracy: label-free metrics of saved models.

For every checkpoint written by ``run_benchmark.py`` / ``run_ogb.py --save-models DIR``
(``DIR/<dataset>/<encoder>/<model>_seed<N>.pt``) this extracts the full-graph embeddings once
and records:

``cka_to_untrained``  linear CKA (Kornblith et al., 2019) between the embeddings and those of
                      the untrained encoder of the same backbone and seed (``--untrained-root``,
                      ``bgrl_seed<N>.pt``): 1 = training left the representation where the
                      backbone put it, up to rotation and isotropic scaling.
``eff_rank``          effective rank of the centered embeddings (dimensional collapse).
``feature_std_mean``  mean per-dimension std: the scale a probe without standardization sees.
``pair_cos``          mean cosine similarity of random node pairs (1 = every embedding parallel,
                      which the centered effective rank does not show).
``edge_cos``          mean cosine similarity across the graph's edges; ``edge_cos - pair_cos``
                      is how much more alike neighbours are than random nodes (smoothing).
``uniformity``        log E exp(-2 ||u_i - u_j||^2) over random pairs of L2-normalised
                      embeddings (Wang & Isola, 2020): lower = spread more uniformly.

It also writes, per seed, the linear-CKA matrix between every pair of checkpoints of that seed
(methods and backbones, untrained encoders included): do embeddings group by backbone or by
method? No metric reads a label.

Usage::

    python benchmarks/representation_metrics.py --dataset Cora \\
        --checkpoints benchmarks/paper/models/Cora/*/*.pt benchmarks/paper/models_untrained/Cora/*/*.pt \\
        --untrained-root benchmarks/paper/models_untrained --out benchmarks/paper/metrics/Cora
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import torch_geometric
from analyze_factors import CHECKPOINT
from node_datasets import ALL_DATASETS, load_node_dataset
from run_benchmark import _git_info
from torch import Tensor

from graphssl.config.load import load_model
from graphssl.data import DataModule
from graphssl.evaluation import effective_rank, extract_embeddings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--untrained-root", required=True)
    p.add_argument("--pairs", type=int, default=100_000, help="random node pairs per metric")
    p.add_argument("--no-cka-matrix", action="store_true", help="skip the per-seed CKA matrices")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def linear_cka(x: Tensor, y: Tensor) -> float:
    """Linear CKA of two [N, D] matrices of the same nodes, in float64."""
    x = x.double() - x.double().mean(0)
    y = y.double() - y.double().mean(0)
    cross = (x.T @ y).pow(2).sum()
    norm = (x.T @ x).pow(2).sum().sqrt() * (y.T @ y).pow(2).sum().sqrt()
    return (cross / norm).item() if norm > 0 else float("nan")


def random_pairs(num_nodes: int, n: int) -> tuple:
    """``n`` pairs of distinct nodes, the same for every checkpoint of a dataset."""
    g = torch.Generator().manual_seed(0)
    i = torch.randint(num_nodes, (n,), generator=g)
    j = (i + torch.randint(1, num_nodes, (n,), generator=g)) % num_nodes
    return i, j


def geometry(z: Tensor, edge_index: Tensor, pairs: tuple, n_edges: int) -> Dict[str, float]:
    u = F.normalize(z.double(), dim=1)
    i, j = pairs
    g = torch.Generator().manual_seed(0)
    e = torch.randint(edge_index.size(1), (min(n_edges, edge_index.size(1)),), generator=g)
    src, dst = edge_index[0, e], edge_index[1, e]
    sq = (u[i] - u[j]).pow(2).sum(1)
    return {
        "pair_cos": (u[i] * u[j]).sum(1).mean().item(),
        "edge_cos": (u[src] * u[dst]).sum(1).mean().item(),
        "uniformity": torch.log(torch.exp(-2 * sq).mean()).item(),
        "feature_std_mean": z.std(0).mean().item(),
        "eff_rank": effective_rank(z),
    }


def main() -> None:
    args = parse_args()
    data, *_ = load_node_dataset(args.dataset, args.data_dir)
    dm = DataModule(data=data, is_graph_level=False)
    pairs = random_pairs(data.num_nodes, args.pairs)
    untrained_root = Path(args.untrained_root).resolve()

    def embed(path: Path) -> Tensor:
        z, _ = extract_embeddings(load_model(path), dm, device=args.device)
        if not torch.isfinite(z).all():
            raise FloatingPointError(f"non-finite values in the embeddings of {path}")
        return z.cpu()

    cache: Dict[Path, Tensor] = {}
    rows: List[dict] = []
    by_seed: Dict[int, List[tuple]] = {}
    for raw in sorted(set(args.checkpoints)):
        path = Path(raw).resolve()
        m = CHECKPOINT.search(path.as_posix())
        if m is None:
            raise ValueError(f"unexpected checkpoint path {path}")
        untrained = path.is_relative_to(untrained_root)
        z = cache[path] if path in cache else embed(path)
        cache[path] = z
        ref_path = untrained_root / m["dataset"] / m["backbone"] / f"bgrl_seed{m['seed']}.pt"
        if ref_path not in cache:
            cache[ref_path] = embed(ref_path)
        row = {
            "checkpoint": str(raw),
            "backbone": m["backbone"],
            "method": "untrained" if untrained else m["method"],
            "seed": int(m["seed"]),
            **geometry(z, data.edge_index, pairs, args.pairs),
            "cka_to_untrained": linear_cka(z, cache[ref_path]),
        }
        rows.append(row)
        by_seed.setdefault(row["seed"], []).append((f"{row['backbone']}/{row['method']}", path))
        print(
            f"  {row['backbone']}/{row['method']}/seed{row['seed']}: "
            f"CKA-untrained {row['cka_to_untrained']:.3f} | rank {row['eff_rank']:.1f} | "
            f"edge-pair cos {row['edge_cos'] - row['pair_cos']:.3f} | "
            f"uniformity {row['uniformity']:.3f}",
            flush=True,
        )

    cka = {}
    if not args.no_cka_matrix:
        for seed, items in sorted(by_seed.items()):
            items.sort()
            matrix = [[linear_cka(cache[a], cache[b]) for _, b in items] for _, a in items]
            cka[str(seed)] = {"names": [n for n, _ in items], "matrix": matrix}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "dataset": args.dataset,
        "hyperparameters": {k: v for k, v in vars(args).items() if k != "checkpoints"},
        "rows": rows,
        "cka_matrices": cka,
        "provenance": {
            "timestamp_utc": timestamp,
            "git": _git_info(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
        },
    }
    path = out / f"metrics__{timestamp}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved to {path}", flush=True)


if __name__ == "__main__":
    main()
