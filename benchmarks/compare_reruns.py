"""Same configuration, same seeds, two runs: how far apart are the per-seed results?

Pairs two sets of ``run_benchmark.py`` result files by (dataset, encoder, model, seed) and
reports the per-seed differences of linear-probe and kNN test accuracy. With a rerun on other
hardware of code that computes the same thing, this measures how far a seed pins a result
down: models are initialised on the CPU, so the initial weights match, but augmentations draw
from the generator of the device the graph is on, and floating-point results differ.

For the SSL models the summary also compares the per-seed differences with what two
independent seeds would give (2 / sqrt(pi) times the seed standard deviation), and checks
whether the 10-seed means and the method ranking survive the rerun.

The latest file per (dataset, encoder, model) is used on each side, as in render_tables.py.

Usage::

    python benchmarks/compare_reruns.py benchmarks/results benchmarks/paper/e1 \\
        --out benchmarks/paper/analysis/reruns.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

METRICS = ["test_acc_linear", "test_acc_knn"]
SUPERVISED = {"supervised", "supervised_reg"}


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
    for (dataset, encoder), g in df[~df.model.isin(SUPERVISED)].groupby(["dataset", "encoder"]):
        for metric in METRICS:
            first, second = g[f"{metric}_first"], g[f"{metric}_second"]
            d = g[f"{metric}_diff"]
            # Spread between seeds of one model, pooled over models, in the reference runs.
            seed_std = float(np.sqrt(g.groupby("model")[f"{metric}_first"].var(ddof=1).mean()))
            independent = 2 / np.sqrt(np.pi) * seed_std
            dev_first = first - g.groupby("model")[f"{metric}_first"].transform("mean")
            dev_second = second - g.groupby("model")[f"{metric}_second"].transform("mean")
            means = g.groupby("model")[[f"{metric}_first", f"{metric}_second"]].mean()
            summary.append(
                {
                    "dataset": dataset,
                    "encoder": encoder,
                    "metric": metric.removeprefix("test_acc_"),
                    "pairs": len(d),
                    "mean |diff|": d.abs().mean(),
                    "max |diff|": d.abs().max(),
                    "identical": int((d.abs() < 1e-9).sum()),
                    "seed std": seed_std,
                    "independent seeds |diff|": independent,
                    "ratio": d.abs().mean() / independent,
                    "same-seed corr": float(np.corrcoef(dev_first, dev_second)[0, 1]),
                    "max |mean diff|": float((means.iloc[:, 1] - means.iloc[:, 0]).abs().max()),
                    "kendall tau of means": float(
                        kendalltau(means.iloc[:, 0], means.iloc[:, 1]).statistic
                    ),
                }
            )
    summary_df = pd.DataFrame(summary)
    if args.out is not None:
        summary_df.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    print(summary_df.round(2).to_string(index=False))
    print("\nreference:", ", ".join(sorted(set(df["first"]))))
    print("rerun:    ", ", ".join(sorted(set(df["second"]))))


if __name__ == "__main__":
    main()
