"""Same configuration, same seeds, two runs: how far apart are the per-seed results?

Pairs two sets of ``run_benchmark.py`` result files by (dataset, encoder, model, seed) and
reports the per-seed differences of linear-probe and kNN test accuracy. With a rerun on other
hardware of code that computes the same thing, this measures run-to-run nondeterminism: the
seed fixes the initialisation and the data order, not the floating-point result.

The latest file per (dataset, encoder, model) is used on each side, as in render_tables.py.

Usage::

    python benchmarks/compare_reruns.py benchmarks/results benchmarks/paper/e1 \\
        --out benchmarks/paper/analysis/reruns.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = ["test_acc_linear", "test_acc_knn"]


def latest_runs(root: Path) -> dict:
    """{(dataset, encoder, model): result} with the newest file of each key under ``root``."""
    runs: dict = {}
    for path in sorted(root.glob("**/*.json")):
        result = json.loads(path.read_text())
        if "per_seed" not in result or "linear_probe" not in result.get("hyperparameters", {}):
            continue  # not a run_benchmark.py file, or a 0.1.0-probe result
        key = (result["dataset"], result["hyperparameters"]["encoder"], result["model"])
        stamp = result["provenance"]["timestamp_utc"]
        if key not in runs or stamp > runs[key]["provenance"]["timestamp_utc"]:
            runs[key] = result
    return runs


def describe(result: dict) -> str:
    prov = result["provenance"]
    return f"{prov['git']['commit'][:7]} torch {prov['versions']['torch']} {prov['versions']['platform'][:24]}"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("first", type=Path, help="directory of the reference runs")
    p.add_argument("second", type=Path, help="directory of the reruns")
    p.add_argument("--out", type=Path, default=None, help="CSV of every paired seed")
    args = p.parse_args()

    first, second = latest_runs(args.first), latest_runs(args.second)
    rows = []
    for key in sorted(first.keys() & second.keys()):
        a = {r["seed"]: r for r in first[key]["per_seed"]}
        b = {r["seed"]: r for r in second[key]["per_seed"]}
        for seed in sorted(a.keys() & b.keys()):
            row = dict(zip(["dataset", "encoder", "model"], key, strict=True)) | {"seed": seed}
            for metric in METRICS:
                row[f"{metric}_first"] = 100 * a[seed][metric]
                row[f"{metric}_second"] = 100 * b[seed][metric]
                row[f"{metric}_diff"] = row[f"{metric}_second"] - row[f"{metric}_first"]
            row["first"], row["second"] = describe(first[key]), describe(second[key])
            rows.append(row)
    if not rows:
        raise SystemExit("no (dataset, encoder, model, seed) in common")
    df = pd.DataFrame(rows)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)

    summary = []
    for (dataset, encoder), g in df.groupby(["dataset", "encoder"]):
        for metric in METRICS:
            d = g[f"{metric}_diff"]
            summary.append(
                {
                    "dataset": dataset,
                    "encoder": encoder,
                    "metric": metric.removeprefix("test_acc_"),
                    "pairs": len(d),
                    "mean |diff|": d.abs().mean(),
                    "max |diff|": d.abs().max(),
                    "mean diff": d.mean(),
                    "identical": int((d.abs() < 1e-9).sum()),
                }
            )
    print(pd.DataFrame(summary).round(2).to_string(index=False))
    print("\nreference:", ", ".join(sorted(set(df["first"]))))
    print("rerun:    ", ", ".join(sorted(set(df["second"]))))


if __name__ == "__main__":
    main()
