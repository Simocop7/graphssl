"""How much of a graph-SSL result is the method? Factor analysis of compare_probes.py output.

Reads the JSON files written by ``compare_probes.py`` for checkpoints saved by
``run_benchmark.py --save-models DIR`` (paths ``DIR/<dataset>/<encoder>/<model>_seed<N>.pt``)
and reports, per dataset:

- every (backbone, method, probe) cell on the public split: mean test accuracy with a 95%
  percentile-bootstrap interval over seeds, and the gain over the untrained encoder of the
  same backbone, paired by seed;
- Kendall's tau between method rankings: GCN vs GIN under each probe, and each probe vs the
  converged GraphSSL probe under each backbone;
- the share of variance (eta squared, sums of squares of a balanced design) explained by
  method, backbone, probe, split and their interaction with the method, the rest being seed
  noise. Computed on accuracy and on the gain over the untrained encoder.

Untrained encoders are the checkpoints under ``--untrained-root`` (``run_benchmark.py
--epochs 0 --save-models``); supervised references are left out of rankings and variance.

Usage::

    python benchmarks/analyze_factors.py benchmarks/paper/probes/*/probes__*.json \\
        --untrained-root benchmarks/paper/models_untrained --out benchmarks/paper/analysis
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

CHECKPOINT = re.compile(
    r"(?P<dataset>[^/]+)/(?P<backbone>[^/]+)/(?P<method>[a-z_]+)_seed(?P<seed>\d+)\.pt$"
)
SUPERVISED = {"supervised", "supervised_reg"}
REFERENCE_PROBE = "graphssl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("probe_files", nargs="+", help="probes__*.json files from compare_probes.py")
    p.add_argument(
        "--untrained-root", required=True, help="--save-models DIR of the --epochs 0 run"
    )
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--out", default="benchmarks/paper/analysis")
    return p.parse_args()


def load_long_table(files: list[str], untrained_root: str) -> pd.DataFrame:
    """One row per (checkpoint, probe, split): dataset, backbone, method, seed, accuracy."""
    untrained_root = str(Path(untrained_root).resolve())
    records = []
    for file in files:
        result = json.loads(Path(file).read_text())
        for row in result["rows"]:
            if row.get("checkpoint") is None:
                continue  # an --untrained row of compare_probes: use the per-seed ones instead
            path = Path(row["checkpoint"]).resolve()
            m = CHECKPOINT.search(path.as_posix())
            if m is None:
                raise ValueError(f"unexpected checkpoint path {path}")
            untrained = path.as_posix().startswith(untrained_root)
            base = {
                "dataset": m["dataset"],
                "backbone": m["backbone"],
                "method": "untrained" if untrained else m["method"],
                "seed": int(m["seed"]),
                "eff_rank": row["eff_rank"],
            }
            records.append({**base, "probe": "knn", "split": "public", "acc": row["test_acc_knn"]})
            for probe, res in row["protocols"].items():
                records.append(
                    {**base, "probe": probe, "split": "public", "acc": res["test_acc"]["mean"]}
                )
            for split in row.get("random_splits", []):
                name = f"random{split['split_seed']}"
                records.append(
                    {**base, "probe": "knn", "split": name, "acc": split["test_acc_knn"]}
                )
                for probe, res in split["protocols"].items():
                    records.append(
                        {**base, "probe": probe, "split": name, "acc": res["test_acc"]["mean"]}
                    )
    df = pd.DataFrame.from_records(records)
    df["acc"] *= 100
    return df


def add_gain(df: pd.DataFrame) -> pd.DataFrame:
    """Gain over the untrained encoder of the same dataset, backbone, probe, split and seed."""
    keys = ["dataset", "backbone", "probe", "split", "seed"]
    base = df[df.method == "untrained"].set_index(keys)["acc"].rename("untrained_acc")
    df = df.join(base, on=keys)
    df["gain"] = df["acc"] - df["untrained_acc"]
    return df


def bootstrap_ci(values: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cell_table(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    public = df[df.split == "public"]
    for key, g in public.groupby(["dataset", "backbone", "probe", "method"]):
        lo, hi = bootstrap_ci(g["acc"].to_numpy(), n_boot, rng)
        glo, ghi = bootstrap_ci(g["gain"].dropna().to_numpy(), n_boot, rng)
        rows.append(
            dict(zip(["dataset", "backbone", "probe", "method"], key, strict=True))
            | {
                "n_seeds": len(g),
                "acc": g["acc"].mean(),
                "acc_lo": lo,
                "acc_hi": hi,
                "gain": g["gain"].mean(),
                "gain_lo": glo,
                "gain_hi": ghi,
                "eff_rank": g["eff_rank"].mean(),
            }
        )
    return pd.DataFrame(rows)


def rankings(cells: pd.DataFrame) -> pd.DataFrame:
    """Kendall's tau between method orders: across backbones, and across probes."""
    ssl = cells[~cells.method.isin(SUPERVISED | {"untrained"})]
    rows = []
    for (dataset, probe), g in ssl.groupby(["dataset", "probe"]):
        wide = g.pivot(index="method", columns="backbone", values="acc").dropna()
        if wide.shape[1] == 2 and len(wide) > 2:
            tau = kendalltau(wide.iloc[:, 0], wide.iloc[:, 1]).statistic
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"{wide.columns[0]} vs {wide.columns[1]}",
                    "within": probe,
                    "tau": tau,
                }
            )
    for (dataset, backbone), g in ssl.groupby(["dataset", "backbone"]):
        wide = g.pivot(index="method", columns="probe", values="acc").dropna()
        if REFERENCE_PROBE not in wide:
            continue
        for probe in wide.columns.drop(REFERENCE_PROBE):
            tau = kendalltau(wide[probe], wide[REFERENCE_PROBE]).statistic
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"{probe} vs {REFERENCE_PROBE}",
                    "within": backbone,
                    "tau": tau,
                }
            )
    return pd.DataFrame(rows)


def eta_squared(
    df: pd.DataFrame, value: str, factors: list[str], interactions: list[tuple[str, str]]
) -> dict:
    """Share of the total sum of squares per factor (balanced design); the rest is residual."""
    d = df.dropna(subset=[value])
    y = d[value]
    total = ((y - y.mean()) ** 2).sum()
    shares, explained = {}, 0.0
    for f in factors:
        means = d.groupby(f)[value].transform("mean")
        ss = ((means - y.mean()) ** 2).sum()
        shares[f] = ss / total
        explained += ss
    for a, b in interactions:
        cell = d.groupby([a, b])[value].transform("mean")
        ma = d.groupby(a)[value].transform("mean")
        mb = d.groupby(b)[value].transform("mean")
        ss = ((cell - ma - mb + y.mean()) ** 2).sum()
        shares[f"{a} x {b}"] = ss / total
        explained += ss
    shares["residual (seed)"] = max(0.0, 1.0 - explained / total)
    return shares


def variance_tables(df: pd.DataFrame) -> pd.DataFrame:
    ssl = df[~df.method.isin(SUPERVISED | {"untrained"})]
    rows = []
    for dataset, g in ssl.groupby("dataset"):
        public = g[g.split == "public"]
        for value in ("acc", "gain"):
            shares = eta_squared(
                public,
                value,
                ["method", "backbone", "probe"],
                [("method", "backbone"), ("method", "probe")],
            )
            rows.append(
                {"dataset": dataset, "design": "public split, all probes", "value": value, **shares}
            )
            ref = g[g.probe == REFERENCE_PROBE]
            if ref.split.nunique() > 1:
                shares = eta_squared(
                    ref,
                    value,
                    ["method", "backbone", "split"],
                    [("method", "backbone"), ("method", "split")],
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "design": f"{REFERENCE_PROBE} probe, all splits",
                        "value": value,
                        **shares,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = add_gain(load_long_table(args.probe_files, args.untrained_root))
    df.to_csv(out / "long_table.csv", index=False)
    cells = cell_table(df, args.bootstrap)
    cells.to_csv(out / "cells.csv", index=False)
    taus = rankings(cells)
    taus.to_csv(out / "kendall_tau.csv", index=False)
    variance = variance_tables(df)
    variance.to_csv(out / "variance_shares.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(cells.round(2).to_string(index=False))
    print("\nKendall tau\n" + taus.round(2).to_string(index=False))
    print("\nShare of variance\n" + variance.round(3).to_string(index=False))
    print(f"\nCSV files in {out}")


if __name__ == "__main__":
    main()
