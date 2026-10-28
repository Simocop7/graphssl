"""How much of a graph-SSL result is the method? Factor analysis of compare_probes.py output.

Reads the JSON files written by ``compare_probes.py`` for checkpoints saved by
``run_benchmark.py --save-models DIR`` (paths ``DIR/<dataset>/<encoder>/<model>_seed<N>.pt``)
and reports, per dataset:

- every (backbone, method, probe) cell on the public split: mean test accuracy with a 95%
  percentile-bootstrap interval over seeds, and the gain over the untrained encoder of the
  same backbone, paired by seed;
- Kendall's tau between method rankings, with a seed-bootstrap 95% interval: GCN vs GIN under
  each probe, and each probe vs the
  converged GraphSSL probe under each backbone;
- the share of variance (eta squared, sums of squares of a balanced design) explained by
  method, backbone, probe, split and their interaction with the method, the rest being seed
  noise. Computed on accuracy and on the gain over the untrained encoder;
  ``variance_shares_ci.csv`` adds a 95% seed-bootstrap interval to each share.

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
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

CHECKPOINT = re.compile(
    r"(?P<dataset>[^/]+)/(?P<backbone>[^/]+)/(?P<method>[a-z_]+)_seed(?P<seed>\d+)\.pt$"
)
SUPERVISED = {"supervised", "supervised_reg"}
REFERENCE_PROBE = "graphssl"
# Protocols used in published code (variance decomposition); the *_std and *_weakl2 runs of
# compare_probes.py are diagnostic controls and are left out of it.
PRACTICE_PROBES = ["adam100", "adam5000", "liblinear", "graphssl"]


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
    p.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="methods left out of every table (sensitivity analyses, e.g. graphdino)",
    )
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
                # The probed dataset, not the training one: arxiv-year scores the models
                # trained on ogbn-arxiv against other labels.
                "dataset": result.get("dataset", m["dataset"]),
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
    # A checkpoint scored by several runs (e.g. extra protocols added later) keeps its last value.
    df = df.drop_duplicates(
        subset=["dataset", "backbone", "method", "seed", "probe", "split"], keep="last"
    )
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


def tau_with_ci(
    a: pd.DataFrame, b: pd.DataFrame, n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Kendall's tau between the method orders of two conditions, with a seed-bootstrap 95% CI.

    ``a`` and ``b`` are (method x seed) accuracy tables. Each bootstrap replicate resamples the
    seeds of every method (jointly in the two conditions, which share seed indices) and
    recomputes the method means and tau.
    """
    methods = a.index.intersection(b.index)
    a, b = a.loc[methods].to_numpy(), b.loc[methods].to_numpy()
    point = kendalltau(a.mean(1), b.mean(1)).statistic
    n_seeds = a.shape[1]
    taus = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_seeds, size=(len(methods), n_seeds))
        rows = np.arange(len(methods))[:, None]
        taus[i] = kendalltau(a[rows, idx].mean(1), b[rows, idx].mean(1)).statistic
    return float(point), float(np.nanpercentile(taus, 2.5)), float(np.nanpercentile(taus, 97.5))


def rankings(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Kendall's tau between method orders on the public split: every pair of backbones under
    each probe, and each probe vs the converged GraphSSL probe under each backbone."""
    rng = np.random.default_rng(0)
    ssl = df[~df.method.isin(SUPERVISED | {"untrained"}) & (df.split == "public")]

    def table(g: pd.DataFrame) -> pd.DataFrame:
        return g.pivot_table(index="method", columns="seed", values="acc").dropna()

    rows = []
    for (dataset, probe), g in ssl.groupby(["dataset", "probe"]):
        for a, b in combinations(sorted(g.backbone.unique()), 2):
            tau, lo, hi = tau_with_ci(
                table(g[g.backbone == a]), table(g[g.backbone == b]), n_boot, rng
            )
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"{a} vs {b}",
                    "within": probe,
                    "tau": tau,
                    "tau_lo": lo,
                    "tau_hi": hi,
                }
            )
    for (dataset, backbone), g in ssl.groupby(["dataset", "backbone"]):
        if REFERENCE_PROBE not in set(g.probe):
            continue
        ref = table(g[g.probe == REFERENCE_PROBE])
        for probe in sorted(set(g.probe) - {REFERENCE_PROBE}):
            tau, lo, hi = tau_with_ci(table(g[g.probe == probe]), ref, n_boot, rng)
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"{probe} vs {REFERENCE_PROBE}",
                    "within": backbone,
                    "tau": tau,
                    "tau_lo": lo,
                    "tau_hi": hi,
                }
            )
    return pd.DataFrame(rows)


def gains_by_probe(cells: pd.DataFrame) -> pd.DataFrame:
    """Gain over the untrained encoder per method under each probe, and whether its sign
    differs from the sign under the converged GraphSSL probe."""
    ssl = cells[~cells.method.isin(SUPERVISED | {"untrained"})]
    wide = ssl.pivot_table(
        index=["dataset", "backbone", "method"], columns="probe", values="gain"
    ).reset_index()
    probes = [c for c in wide.columns if c not in {"dataset", "backbone", "method"}]
    if REFERENCE_PROBE in probes:
        for p in probes:
            if p != REFERENCE_PROBE:
                wide[f"sign_flip_{p}"] = np.sign(wide[p]) != np.sign(wide[REFERENCE_PROBE])
    return wide


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


def coded_eta_squared(
    y: np.ndarray,
    codes: dict[str, np.ndarray],
    factors: list[str],
    interactions: list[tuple[str, str]],
    idx: np.ndarray,
) -> np.ndarray:
    """eta_squared on the rows ``idx`` of integer-coded factors, fast enough for a bootstrap:
    the shares of ``factors``, then of ``interactions``, then the residual."""
    yy = y[idx]
    grand = yy.mean()
    total = ((yy - grand) ** 2).sum()

    def group_means(c: np.ndarray) -> np.ndarray:
        return (np.bincount(c, weights=yy) / np.maximum(np.bincount(c), 1))[c]

    main = {f: group_means(codes[f][idx]) for f in factors}
    ss = [((main[f] - grand) ** 2).sum() for f in factors]
    for a, b in interactions:
        joint = codes[a][idx] * (codes[b].max() + 1) + codes[b][idx]
        ss.append(((group_means(joint) - main[a] - main[b] + grand) ** 2).sum())
    shares = np.array(ss) / total
    return np.append(shares, max(0.0, 1.0 - shares.sum()))


def drop_repeated_splits(ref: pd.DataFrame) -> pd.DataFrame:
    """Rows of every random split that repeats the public one dropped. Datasets without a public
    split (Amazon, Coauthor) use random split 0 as their standard split, so it is scored twice;
    the split factor must count it once."""
    acc = ref.pivot_table(index=["backbone", "method", "seed"], columns="split", values="acc")
    if "public" not in acc:
        return ref
    repeated = [
        s for s in acc.columns if s != "public" and np.allclose(acc[s], acc["public"], atol=1e-9)
    ]
    return ref[~ref.split.isin(repeated)]


def variance_designs(g: pd.DataFrame) -> list[tuple[str, pd.DataFrame, list[str], list]]:
    """The designs decomposed for one dataset's SSL rows: (name, rows, factors, interactions)."""
    public = g[(g.split == "public") & g.probe.isin([*PRACTICE_PROBES, "knn"])]
    by_probe = (["method", "backbone", "probe"], [("method", "backbone"), ("method", "probe")])
    designs = [
        ("public split, all probes", public, *by_probe),
        ("public split, linear probes", public[public.probe != "knn"], *by_probe),
    ]
    ref = drop_repeated_splits(g[g.probe == REFERENCE_PROBE])
    if ref.split.nunique() > 1:
        designs.append(
            (
                f"{REFERENCE_PROBE} probe, all splits",
                ref,
                ["method", "backbone", "split"],
                [("method", "backbone"), ("method", "split")],
            )
        )
    return designs


def variance_tables(df: pd.DataFrame) -> pd.DataFrame:
    ssl = df[~df.method.isin(SUPERVISED | {"untrained"})]
    rows = []
    for dataset, g in ssl.groupby("dataset"):
        designs = variance_designs(g)
        for value in ("acc", "gain"):
            for design, d, factors, interactions in designs:
                shares = eta_squared(d, value, factors, interactions)
                rows.append({"dataset": dataset, "design": design, "value": value, **shares})
    return pd.DataFrame(rows)


def variance_intervals(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """95% percentile-bootstrap intervals of the variance shares, one row per component.

    Each replicate resamples, with replacement, the seeds of every method x backbone cell and
    keeps all rows of a drawn seed together (its scores under every probe or split come from
    the same checkpoint); probes and splits themselves are not resampled. The point estimate
    is the share of variance_shares.csv. Shares cannot be negative and resampling adds noise
    to the cell means, so the interval of a share near zero sits above it (Cora's method x
    split: 0.12% [0.14, 0.25]), and a drawn seed counted twice leaves the residual a little
    low: read such intervals as an upper bound, not a range around the point.
    """
    rng = np.random.default_rng(0)
    ssl = df[~df.method.isin(SUPERVISED | {"untrained"})]
    unit = ["method", "backbone"]
    rows = []
    for dataset, g in ssl.groupby("dataset"):
        designs = variance_designs(g)
        for value in ("acc", "gain"):
            for design, d, factors, interactions in designs:
                d = d.dropna(subset=[value]).sort_values([*unit, "seed"], kind="stable")
                block_sizes = d.groupby([*unit, "seed"]).size()
                seeds_per_unit = d.groupby(unit)["seed"].nunique()
                if block_sizes.nunique() != 1 or seeds_per_unit.nunique() != 1:
                    raise ValueError(f"{dataset} / {design} / {value}: unbalanced design")
                n_units, n_seeds = len(seeds_per_unit), int(seeds_per_unit.iloc[0])
                blocks = np.arange(len(d)).reshape(n_units, n_seeds, int(block_sizes.iloc[0]))
                y = d[value].to_numpy(dtype=float)
                codes = {f: d[f].astype("category").cat.codes.to_numpy() for f in factors}
                names = [*factors, *(f"{a} x {b}" for a, b in interactions), "residual (seed)"]
                point = coded_eta_squared(y, codes, factors, interactions, blocks.ravel())
                draws = np.empty((n_boot, len(names)))
                for i in range(n_boot):
                    pick = rng.integers(0, n_seeds, size=(n_units, n_seeds))
                    idx = blocks[np.arange(n_units)[:, None], pick].ravel()
                    draws[i] = coded_eta_squared(y, codes, factors, interactions, idx)
                lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
                for k, name in enumerate(names):
                    rows.append(
                        {
                            "dataset": dataset,
                            "design": design,
                            "value": value,
                            "component": name,
                            "share": point[k],
                            "lo": lo[k],
                            "hi": hi[k],
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = add_gain(load_long_table(args.probe_files, args.untrained_root))
    df = df[~df.method.isin(args.exclude)]
    df.to_csv(out / "long_table.csv", index=False)
    cells = cell_table(df, args.bootstrap)
    cells.to_csv(out / "cells.csv", index=False)
    taus = rankings(df, min(args.bootstrap, 2000))
    taus.to_csv(out / "kendall_tau.csv", index=False)
    gains = gains_by_probe(cells)
    gains.to_csv(out / "gains_by_probe.csv", index=False)
    variance = variance_tables(df)
    variance.to_csv(out / "variance_shares.csv", index=False)
    intervals = variance_intervals(df, args.bootstrap)
    intervals.to_csv(out / "variance_shares_ci.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(cells.round(2).to_string(index=False))
    print("\nKendall tau\n" + taus.round(2).to_string(index=False))
    print("\nShare of variance\n" + variance.round(3).to_string(index=False))
    linear = intervals[intervals.design == "public split, linear probes"]
    print("\nShare of variance, linear probes, 95% seed-bootstrap interval\n")
    print(linear.drop(columns="design").round(3).to_string(index=False))
    print(f"\nCSV files in {out}")


if __name__ == "__main__":
    main()
