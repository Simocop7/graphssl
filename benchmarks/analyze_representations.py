"""What training changes in the representation, and whether a label-free metric predicts the gain.

Reads the JSON files of ``representation_metrics.py`` and ``compare_probes.py`` (with its
``--raw-features`` row) and implements "M in detail" of ``benchmarks/paper/PLAN.md``. Writes to
``--out``:

``metrics_cells.csv``       every metric per (dataset, backbone, method): seed mean and 95%
                            percentile-bootstrap interval over seeds
``metrics_variance.csv``    share of each metric's variance explained by method, backbone and
                            method x backbone (SSL methods; the rest is seed), with 95%
                            seed-bootstrap intervals
``cka_grouping.csv``        mean linear CKA between two trained SSL models that share the
                            backbone (other method), the method (other backbone), or neither
``graph_contribution.csv``  raw features -> untrained encoder -> trained methods, per dataset
                            and backbone, standard split, with the labels' edge homophily
``prediction_groups.csv``   per (dataset, backbone): Spearman rho between each candidate metric
                            and the gain over the untrained encoder, selection regret, random regret
``prediction.csv``          the test fixed in advance: per candidate and set, the mean rho with a
                            95% seed-bootstrap interval, groups with rho > 0, regret vs random,
                            verdict. Target: GraphSSL probe (decision) and kNN (sensitivity)

Usage::

    python benchmarks/analyze_representations.py \\
        --metrics benchmarks/paper/metrics/*/metrics__*.json \\
        --probes benchmarks/paper/probes/*/probes__*.json \\
        --untrained-root benchmarks/paper/models_untrained --out benchmarks/paper/analysis
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_factors import (
    SUPERVISED,
    add_gain,
    bootstrap_ci,
    coded_eta_squared,
    load_long_table,
)
from compare_probes import RAW_FEATURES
from scipy.stats import rankdata

SSL_METHODS = ["dgi", "graphcl", "vicreg", "barlow_twins", "bgrl", "afgrl", "graphdino"]
# Candidates of the prediction test, in PLAN.md's order; higher is predicted to mean more gain.
CANDIDATES = {
    "eff_rank": "effective rank (RankMe)",
    "distance": "distance from untrained (1 - CKA)",
    "smoothing_gain": "smoothing gained",
    "neg_uniformity": "uniformity (negated)",
}
METRICS = [*CANDIDATES, "smoothing", "feature_std_mean", "pair_cos", "uniformity"]
# Sets of the prediction test (PLAN.md): decisions may use the development set only.
DEVELOPMENT = ["Cora", "CiteSeer"]
TEST = ["PubMed", "AmazonComputers", "AmazonPhoto", "CoauthorCS", "ogbn-arxiv"]
SEPARATE = ["arxiv-year"]  # smoothing is predicted not to help there
# Label sets scored on another dataset's models: their representation metrics are those.
METRICS_FROM = {"arxiv-year": "ogbn-arxiv"}
REFERENCE_PROBE = "graphssl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--metrics", nargs="+", required=True)
    p.add_argument("--probes", nargs="+", required=True)
    p.add_argument("--untrained-root", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--data-dir", default="data", help="for the labels' edge homophily")
    p.add_argument("--no-homophily", action="store_true", help="skip loading the datasets")
    p.add_argument("--out", default="benchmarks/paper/analysis")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_metrics(files: list[str]) -> tuple[pd.DataFrame, dict]:
    """One row per (dataset, backbone, method, seed) with the metrics, and the CKA matrices."""
    rows, matrices = [], {}
    for file in files:
        result = json.loads(Path(file).read_text())
        for r in result["rows"]:
            rows.append({"dataset": result["dataset"], **r})
        for seed, m in result.get("cka_matrices", {}).items():
            matrices[(result["dataset"], int(seed))] = m
    df = pd.DataFrame(rows).drop_duplicates(
        subset=["dataset", "backbone", "method", "seed"], keep="last"
    )
    df["smoothing"] = df["edge_cos"] - df["pair_cos"]
    df["distance"] = 1.0 - df["cka_to_untrained"]
    df["neg_uniformity"] = -df["uniformity"]
    keys = ["dataset", "backbone", "seed"]
    base = df[df.method == "untrained"].set_index(keys)["smoothing"].rename("smoothing_untrained")
    df = df.join(base, on=keys)
    df["smoothing_gain"] = df["smoothing"] - df["smoothing_untrained"]
    return df, matrices


def load_raw_features(files: list[str]) -> pd.DataFrame:
    """Accuracy of the --raw-features rows: one row per (dataset, probe, split)."""
    records = []
    for file in files:
        result = json.loads(Path(file).read_text())
        for row in result["rows"]:
            if row["name"] != RAW_FEATURES:
                continue
            d = result["dataset"]
            records.append(
                {"dataset": d, "probe": "knn", "split": "public", "acc": row["test_acc_knn"]}
            )
            for probe, res in row["protocols"].items():
                records.append(
                    {
                        "dataset": d,
                        "probe": probe,
                        "split": "public",
                        "acc": res["test_acc"]["mean"],
                    }
                )
            for split in row.get("random_splits", []):
                name = f"random{split['split_seed']}"
                records.append(
                    {"dataset": d, "probe": "knn", "split": name, "acc": split["test_acc_knn"]}
                )
                for probe, res in split["protocols"].items():
                    records.append(
                        {
                            "dataset": d,
                            "probe": probe,
                            "split": name,
                            "acc": res["test_acc"]["mean"],
                        }
                    )
    df = pd.DataFrame.from_records(records, columns=["dataset", "probe", "split", "acc"])
    df = df.drop_duplicates(subset=["dataset", "probe", "split"], keep="last")
    df["acc"] *= 100
    return df


def edge_homophily(datasets: list[str], data_dir: str) -> dict:
    from node_datasets import load_node_dataset
    from torch_geometric.utils import homophily

    out = {}
    for name in datasets:
        data, *_ = load_node_dataset(name, data_dir)
        out[name] = float(homophily(data.edge_index, data.y.view(-1), method="edge"))
    return out


# ---------------------------------------------------------------------------
# Part 1: what training changes
# ---------------------------------------------------------------------------


def metric_cells(m: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for key, g in m.groupby(["dataset", "backbone", "method"]):
        row = dict(zip(["dataset", "backbone", "method"], key, strict=True)) | {"n_seeds": len(g)}
        for metric in METRICS:
            values = g[metric].dropna().to_numpy()
            lo, hi = bootstrap_ci(values, n_boot, rng)
            row |= {
                metric: values.mean() if len(values) else np.nan,
                f"{metric}_lo": lo,
                f"{metric}_hi": hi,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def metric_variance(m: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """eta squared of each metric over the SSL methods: method, backbone, method x backbone."""
    rng = np.random.default_rng(0)
    factors, interactions = ["method", "backbone"], [("method", "backbone")]
    names = [*factors, "method x backbone", "residual (seed)"]
    rows = []
    for dataset, g in m[m.method.isin(SSL_METHODS)].groupby("dataset"):
        if g.backbone.nunique() < 2:
            continue
        for metric in METRICS:
            d = g.dropna(subset=[metric]).sort_values(["method", "backbone", "seed"], kind="stable")
            per_cell = d.groupby(["method", "backbone"]).size()
            y = d[metric].to_numpy(dtype=float)
            codes = {f: d[f].astype("category").cat.codes.to_numpy() for f in factors}
            point = coded_eta_squared(y, codes, factors, interactions, np.arange(len(d)))
            lo = hi = np.full(len(names), np.nan)
            if per_cell.nunique() == 1:  # balanced: resample the seeds of every cell
                n_cells, n_seeds = len(per_cell), int(per_cell.iloc[0])
                blocks = np.arange(len(d)).reshape(n_cells, n_seeds)
                draws = np.empty((n_boot, len(names)))
                for i in range(n_boot):
                    pick = rng.integers(0, n_seeds, size=(n_cells, n_seeds))
                    idx = blocks[np.arange(n_cells)[:, None], pick].ravel()
                    draws[i] = coded_eta_squared(y, codes, factors, interactions, idx)
                lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
            for k, name in enumerate(names):
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "component": name,
                        "share": point[k],
                        "lo": lo[k],
                        "hi": hi[k],
                    }
                )
    return pd.DataFrame(rows)


def mean_cka_matrices(matrices: dict) -> dict:
    """{dataset: DataFrame}: the CKA matrix averaged over the seeds that share its models."""
    by_dataset: dict = {}
    for (dataset, _seed), m in matrices.items():
        by_dataset.setdefault(dataset, []).append(m)
    out = {}
    for dataset, ms in by_dataset.items():
        names = max((m["names"] for m in ms), key=len)
        same = [np.asarray(m["matrix"]) for m in ms if m["names"] == names]
        out[dataset] = pd.DataFrame(np.mean(same, axis=0), index=names, columns=names)
    return out


def cka_grouping(matrices: dict, n_boot: int) -> pd.DataFrame:
    """Mean CKA between trained SSL models, by what the two share; per dataset over seeds."""
    rng = np.random.default_rng(0)
    per_seed: dict = {}
    for (dataset, _seed), m in matrices.items():
        names = [n.split("/") for n in m["names"]]
        mat = np.asarray(m["matrix"])
        groups: dict = {"same backbone": [], "same method": [], "neither": []}
        for i, j in combinations(range(len(names)), 2):
            (bi, mi), (bj, mj) = names[i], names[j]
            if mi not in SSL_METHODS or mj not in SSL_METHODS:
                continue
            if bi == bj:
                groups["same backbone"].append(mat[i, j])
            elif mi == mj:
                groups["same method"].append(mat[i, j])
            else:
                groups["neither"].append(mat[i, j])
        for kind, values in groups.items():
            if values:
                per_seed.setdefault((dataset, kind), []).append(float(np.mean(values)))
    rows = []
    for (dataset, kind), values in sorted(per_seed.items()):
        lo, hi = bootstrap_ci(np.asarray(values), n_boot, rng)
        rows.append(
            {
                "dataset": dataset,
                "pair": kind,
                "cka": np.mean(values),
                "lo": lo,
                "hi": hi,
                "n_seeds": len(values),
            }
        )
    return pd.DataFrame(rows)


def graph_contribution(
    gains: pd.DataFrame, raw: pd.DataFrame, m: pd.DataFrame, homophily: dict
) -> pd.DataFrame:
    """Raw features -> untrained encoder -> trained SSL methods, standard split."""
    public = gains[gains.split == "public"]
    rows = []
    for (dataset, backbone, probe), g in public.groupby(["dataset", "backbone", "probe"]):
        if probe not in (REFERENCE_PROBE, "knn"):
            continue
        untrained = g[g.method == "untrained"]["acc"].mean()
        ssl = g[g.method.isin(SSL_METHODS)].groupby("method")["acc"].mean()
        r = raw[(raw.dataset == dataset) & (raw.probe == probe) & (raw.split == "public")]
        raw_acc = r["acc"].iloc[0] if len(r) else np.nan
        mm = m[(m.dataset == METRICS_FROM.get(dataset, dataset)) & (m.backbone == backbone)]
        rows.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "probe": probe,
                "edge_homophily": homophily.get(dataset, np.nan),
                "raw_features": raw_acc,
                "untrained": untrained,
                "ssl_mean": ssl.mean(),
                "ssl_best": ssl.max(),
                "graph_adds": untrained - raw_acc,
                "ssl_adds_mean": ssl.mean() - untrained,
                "ssl_adds_best": ssl.max() - untrained,
                "smoothing_untrained": mm[mm.method == "untrained"]["smoothing"].mean(),
                "smoothing_ssl_mean": mm[mm.method.isin(SSL_METHODS)]["smoothing"].mean(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 2: the prediction test (fixed in advance in PLAN.md)
# ---------------------------------------------------------------------------


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def regrets(gain: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """(regret of picking the top-scored method, expected regret of a random pick), in points."""
    best = gain.max()
    return float(best - gain[int(np.argmax(score))]), float(best - gain.mean())


def paired_arrays(gains: pd.DataFrame, m: pd.DataFrame, probe: str) -> dict:
    """{(dataset, backbone): (methods, gain [methods x seeds], {candidate: [methods x seeds]})},
    gain and metrics paired by seed (the same checkpoint)."""
    target = gains[
        (gains.probe == probe) & (gains.split == "public") & gains.method.isin(SSL_METHODS)
    ]
    target = target[["dataset", "backbone", "method", "seed", "gain"]].dropna()
    target["metrics_dataset"] = target["dataset"].map(lambda d: METRICS_FROM.get(d, d))
    metrics = m[m.method.isin(SSL_METHODS)][["dataset", "backbone", "method", "seed", *CANDIDATES]]
    metrics = metrics.rename(columns={"dataset": "metrics_dataset"})
    j = target.merge(metrics, on=["metrics_dataset", "backbone", "method", "seed"], how="inner")
    out = {}
    for (dataset, backbone), g in j.groupby(["dataset", "backbone"]):
        # Seeds every method has, so each cell has the same number of draws.
        common = set.intersection(*(set(s) for s in g.groupby("method")["seed"].unique()))
        g = g[g.seed.isin(common)]
        methods = sorted(g.method.unique())

        def table(value: str, g: pd.DataFrame = g, methods: list = methods) -> np.ndarray:
            return (
                g.pivot_table(index="method", columns="seed", values=value).loc[methods].to_numpy()
            )

        out[(dataset, backbone)] = (methods, table("gain"), {c: table(c) for c in CANDIDATES})
    return out


def set_of(dataset: str) -> str:
    if dataset in DEVELOPMENT:
        return "development"
    if dataset in TEST:
        return "test"
    if dataset in SEPARATE:
        return dataset
    return "other"


def prediction(
    gains: pd.DataFrame, m: pd.DataFrame, n_boot: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    group_rows, summary_rows = [], []
    for probe in (REFERENCE_PROBE, "knn"):
        arrays = paired_arrays(gains, m, probe)
        if not arrays:
            continue
        keys = sorted(arrays)
        # Point estimates per group, from the seed means.
        point = {}
        for key in keys:
            methods, gain, cand = arrays[key]
            g = gain.mean(1)
            for c in CANDIDATES:
                s = cand[c].mean(1)
                rho = spearman(s, g)
                reg, rand = regrets(g, s)
                point[(key, c)] = rho
                group_rows.append(
                    {
                        "target": probe,
                        "dataset": key[0],
                        "backbone": key[1],
                        "set": set_of(key[0]),
                        "candidate": c,
                        "n_methods": len(methods),
                        "n_seeds": gain.shape[1],
                        "rho": rho,
                        "regret": reg,
                        "random_regret": rand,
                        "top_by_metric": methods[int(np.argmax(s))],
                        "top_by_gain": methods[int(np.argmax(g))],
                    }
                )
        # Seed bootstrap: resample the seeds of every (group, method) cell, gain and metric
        # together, and recompute every rho.
        draws = {(key, c): np.empty(n_boot) for key in keys for c in CANDIDATES}
        for i in range(n_boot):
            for key in keys:
                _, gain, cand = arrays[key]
                n_m, n_s = gain.shape
                pick = rng.integers(0, n_s, size=(n_m, n_s))
                rows_ = np.arange(n_m)[:, None]
                g = gain[rows_, pick].mean(1)
                for c in CANDIDATES:
                    draws[(key, c)][i] = spearman(cand[c][rows_, pick].mean(1), g)
        groups = pd.DataFrame([r for r in group_rows if r["target"] == probe])
        for set_name, gs in groups.groupby("set"):
            for c in CANDIDATES:
                gc = gs[gs.candidate == c]
                set_keys = [(d, b) for d, b in zip(gc.dataset, gc.backbone, strict=True)]
                boot = np.nanmean(np.stack([draws[(k, c)] for k in set_keys]), axis=0)
                lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                mean_rho = float(np.nanmean([point[(k, c)] for k in set_keys]))
                regret, random_regret = gc["regret"].mean(), gc["random_regret"].mean()
                if set_name == "development":
                    verdict = "development set (no verdict)"
                elif lo > 0 and regret < random_regret:
                    verdict = "predicts"
                else:
                    verdict = "does not predict"
                summary_rows.append(
                    {
                        "target": probe,
                        "set": set_name,
                        "candidate": c,
                        "datasets": " ".join(sorted(set(gc.dataset))),
                        "n_groups": len(gc),
                        "mean_rho": mean_rho,
                        "rho_lo": float(lo),
                        "rho_hi": float(hi),
                        "groups_rho_positive": int((gc["rho"] > 0).sum()),
                        "mean_regret": regret,
                        "mean_random_regret": random_regret,
                        "verdict": verdict,
                    }
                )
    return pd.DataFrame(group_rows), pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m, matrices = load_metrics(args.metrics)
    gains = add_gain(load_long_table(args.probes, args.untrained_root))
    gains = gains[~gains.method.isin(SUPERVISED)]
    raw = load_raw_features(args.probes)
    datasets = sorted(set(gains.dataset))
    homophily = {} if args.no_homophily else edge_homophily(datasets, args.data_dir)

    cells = metric_cells(m, args.bootstrap)
    cells.to_csv(out / "metrics_cells.csv", index=False)
    variance = metric_variance(m, args.bootstrap)
    variance.to_csv(out / "metrics_variance.csv", index=False)
    grouping = cka_grouping(matrices, args.bootstrap)
    grouping.to_csv(out / "cka_grouping.csv", index=False)
    for dataset, matrix in mean_cka_matrices(matrices).items():
        matrix.to_csv(out / f"cka_matrix__{dataset}.csv")
    contribution = graph_contribution(gains, raw, m, homophily)
    contribution.to_csv(out / "graph_contribution.csv", index=False)
    groups, summary = prediction(gains, m, args.bootstrap)
    groups.to_csv(out / "prediction_groups.csv", index=False)
    summary.to_csv(out / "prediction.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("Graph contribution (standard split)\n" + contribution.round(2).to_string(index=False))
    print("\nCKA grouping\n" + grouping.round(3).to_string(index=False))
    shown = variance[variance.metric.isin(CANDIDATES) | (variance.metric == "smoothing")]
    print("\nShare of variance of each metric\n" + shown.round(3).to_string(index=False))
    print("\nPrediction test\n" + summary.round(3).to_string(index=False))
    print(f"\nCSV files in {out}")


if __name__ == "__main__":
    main()
