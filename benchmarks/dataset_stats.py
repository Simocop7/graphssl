"""Statistics of the node-classification datasets of the factor study, for the paper's table.

Loads every dataset the way the runners do (``node_datasets.load_node_dataset``) and writes one
row per dataset: nodes, undirected edges, features, classes, edge homophily of the labels, and
the sizes of the standard split. A dataset whose files are not under ``--data`` is skipped.

Usage::

    python benchmarks/dataset_stats.py --out benchmarks/paper/analysis/datasets.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from node_datasets import ARXIV_YEAR, FULL_BATCH_DATASETS, load_node_dataset
from torch_geometric.utils import homophily, to_undirected

DATASETS = [*FULL_BATCH_DATASETS, "ogbn-arxiv", ARXIV_YEAR]


def stats(name: str, data_dir: str) -> dict:
    data, num_classes, train_idx, val_idx, test_idx = load_node_dataset(name, data_dir)
    edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    no_loops = edge_index[:, edge_index[0] != edge_index[1]]
    labelled = data.y >= 0  # arxiv-year marks nodes without a year with -1
    keep = labelled[no_loops[0]] & labelled[no_loops[1]]
    return {
        "dataset": name,
        "nodes": data.num_nodes,
        "edges": no_loops.size(1) // 2,
        "features": data.num_features,
        "classes": num_classes,
        "edge_homophily": round(float(homophily(no_loops[:, keep], data.y, method="edge")), 3),
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", default="data")
    p.add_argument("--out", type=Path, default=Path("benchmarks/paper/analysis/datasets.csv"))
    args = p.parse_args()
    torch.manual_seed(0)
    rows = []
    for name in DATASETS:
        try:
            rows.append(stats(name, args.data))
        except (FileNotFoundError, ImportError, RuntimeError) as e:
            print(f"skip {name}: {e}")
            continue
        print(rows[-1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
