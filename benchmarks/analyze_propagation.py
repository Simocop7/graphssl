"""E6 of the factor study (benchmarks/paper/PLAN.md, "E6: a parameter-free reference").

Compares every saved model with representations that need no training: the raw features, the
features propagated over the GCN-normalized adjacency (A^1X, A^2X: SGC's features) and the
first 256 principal components of A^2X. Each representation is scored by the regularized probe
that is better on validation (ours or liblinear; ties: ours), the "valsel" probe of the rule;
ours and liblinear are also reported alone.

Inputs: the probe files of the six datasets (``benchmarks/paper/probes/<D>/``: models, untrained
encoders, raw features) and of E6 (``benchmarks/paper/e6/probes/<D>/``: input-feature rows, and
for WikiCS also its models). The newest scoring of each checkpoint wins, as in the other
analyses. Writes to ``benchmarks/paper/analysis/e6/``:

``inputs.csv``        accuracy of the input-feature rows per probe (standard split, and the
                      mean over the random splits under our probe)
``cells.csv``         seed-mean test accuracy of every (dataset, backbone, method), per probe
``reference.csv``     per dataset and probe: A^2X against the GCN objectives (D = mean over the
                      objectives of objective − A^2X, 95% seed bootstrap), how many objectives
                      and how many models of any backbone are above A^2X
``architecture.csv``  untrained encoder minus raw features, per backbone and probe
``verdict.txt``       the decision rule of E6 on WikiCS

and the table ``benchmarks/paper/figures/tab_propagation.tex``.

Usage::

    python benchmarks/analyze_propagation.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

P = Path("benchmarks/paper")
SIX = ["Cora", "CiteSeer", "PubMed", "AmazonComputers", "AmazonPhoto", "CoauthorCS"]
DATASETS = [*SIX, "ogbn-arxiv", "arxiv-year", "WikiCS"]
NAMES = {
    "AmazonComputers": "Amazon Computers",
    "AmazonPhoto": "Amazon Photo",
    "CoauthorCS": "Coauthor CS",
}
INPUT_ROWS = {
    "raw features": "raw",
    "propagated A^1X": "A1X",
    "propagated A^2X": "A2X",
    "PCA-256 of A^2X": "PCA",
}
SUPERVISED = {"supervised", "supervised_reg"}
RULE_DATASET = "WikiCS"
RULE_THRESHOLD = 1.0
N_BOOT = 10_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", default=str(P / "analysis" / "e6"))
    p.add_argument("--table", default=str(P / "figures" / "tab_propagation.tex"))
    return p.parse_args()


def _fits(protocol: dict) -> tuple[float, float]:
    """(val, test) accuracy in %, averaged over a protocol's fits."""
    fits = protocol.get("fits") or []
    if fits:
        return (
            100 * float(np.mean([f.get("val_acc", np.nan) for f in fits])),
            100 * float(np.mean([f["test_acc"] for f in fits])),
        )
    t = protocol["test_acc"]
    return float("nan"), 100 * (t["mean"] if isinstance(t, dict) else t)


def load_rows() -> pd.DataFrame:
    """One row per (scored representation, probe): newest scoring per checkpoint or input row."""
    files = []
    for d in DATASETS:
        files += glob.glob(str(P / "probes" / d / "probes__*.json"))
        files += glob.glob(str(P / "e6" / "probes" / d / "probes__*.json"))
    latest: dict = {}
    for f in sorted(files, key=os.path.basename):  # timestamped names: newest last
        doc = json.load(open(f))
        dataset = doc["dataset"]
        for r in doc["rows"]:
            ckpt = r.get("checkpoint")
            if ckpt is None:
                if r["name"] not in INPUT_ROWS:
                    continue
                key = (dataset, INPUT_ROWS[r["name"]])
                meta = {"backbone": "-", "method": INPUT_ROWS[r["name"]], "seed": -1}
            else:
                m = re.search(r"/(models[^/]*)/[^/]+/([^/]+)/(.+)_seed(\d+)\.pt$", ckpt)
                if m is None:
                    continue
                untrained = "untrained" in m.group(1)
                if untrained and m.group(3) != "bgrl":
                    continue
                method = "untrained" if untrained else m.group(3)
                key = (dataset, m.group(2), method, int(m.group(4)))
                meta = {"backbone": m.group(2), "method": method, "seed": int(m.group(4))}
            entry = latest.setdefault(key, {"dataset": dataset, **meta, "probes": {}})
            for name, proto in r["protocols"].items():
                entry["probes"][name] = _fits(proto)
            splits = [
                100 * s["protocols"]["graphssl"]["test_acc"]["mean"]
                if isinstance(s["protocols"]["graphssl"]["test_acc"], dict)
                else 100 * s["protocols"]["graphssl"]["test_acc"]
                for s in r.get("random_splits", [])
                if "graphssl" in s.get("protocols", {})
            ]
            if splits:
                entry["probes"]["graphssl_random"] = (float("nan"), float(np.mean(splits)))
    out = []
    for e in latest.values():
        probes = e.pop("probes")
        ours, lib = probes.get("graphssl"), probes.get("liblinear")
        if ours is not None:
            probes["valsel"] = lib if (lib is not None and lib[0] > ours[0]) else ours
        for name, (val, test) in probes.items():
            out.append({**e, "probe": name, "val": val, "test": test})
    return pd.DataFrame(out)


def bootstrap_d(models: pd.DataFrame, ref: float, rng: np.random.Generator) -> tuple:
    """D = mean over objectives of (seed-mean accuracy − ref), with a seed-bootstrap 95% CI."""
    per = {m: g.test.to_numpy() for m, g in models.groupby("method")}
    point = float(np.mean([v.mean() for v in per.values()]) - ref)
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        draws[i] = np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in per.values()])
    lo, hi = np.percentile(draws - ref, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_rows()
    inputs = df[df.seed == -1].pivot_table(
        index=["dataset", "method"], columns="probe", values="test"
    )
    inputs.round(2).to_csv(out / "inputs.csv")

    models = df[df.seed >= 0]
    cells = (
        models.groupby(["dataset", "backbone", "method", "probe"])
        .test.agg(["mean", "std", "count"])
        .reset_index()
    )
    cells.round(3).to_csv(out / "cells.csv", index=False)

    rng = np.random.default_rng(0)
    rows = []
    for (dataset, probe), g in models.groupby(["dataset", "probe"]):
        ref_rows = df[(df.dataset == dataset) & (df.method == "A2X") & (df.probe == probe)]
        if ref_rows.empty:
            continue
        a2x = float(ref_rows.test.iloc[0])
        ssl = g[~g.method.isin(SUPERVISED | {"untrained"})]
        gcn = ssl[ssl.backbone == "gcn"]
        if gcn.empty:
            continue
        d, lo, hi = bootstrap_d(gcn, a2x, rng)
        means = gcn.groupby("method").test.mean()
        every = ssl.groupby(["backbone", "method"]).test.mean()
        untrained = g[(g.method == "untrained") & (g.backbone == "gcn")].test.mean()
        rows.append(
            {
                "dataset": dataset,
                "probe": probe,
                "raw": _input(df, dataset, "raw", probe),
                "A1X": _input(df, dataset, "A1X", probe),
                "A2X": a2x,
                "PCA": _input(df, dataset, "PCA", probe),
                "untrained_gcn": untrained,
                "gcn_min": means.min(),
                "gcn_mean": means.mean(),
                "gcn_max": means.max(),
                "D": d,
                "D_lo": lo,
                "D_hi": hi,
                "gcn_above_A2X": int((means > a2x).sum()),
                "gcn_objectives": int(len(means)),
                "models_above_A2X": int((every > a2x).sum()),
                "models": int(len(every)),
                "best_model": "/".join(every.idxmax()),
                "best_model_acc": every.max(),
            }
        )
    ref = pd.DataFrame(rows)
    ref.round(3).to_csv(out / "reference.csv", index=False)

    arch = []
    for (dataset, backbone, probe), g in models[models.method == "untrained"].groupby(
        ["dataset", "backbone", "probe"]
    ):
        raw = _input(df, dataset, "raw", probe)
        arch.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "probe": probe,
                "untrained": g.test.mean(),
                "raw": raw,
                "untrained_minus_raw": g.test.mean() - raw,
            }
        )
    pd.DataFrame(arch).round(3).to_csv(out / "architecture.csv", index=False)

    verdict = _verdict(ref)
    (out / "verdict.txt").write_text(verdict + "\n")
    print(verdict)
    Path(args.table).write_text(_table(ref))
    rule_table = Path(args.table).with_name("tab_e6_wikics.tex")
    rule_table.write_text(_rule_table(df, cells))
    print(f"wrote {out}/, {args.table} and {rule_table}")


def _input(df: pd.DataFrame, dataset: str, row: str, probe: str) -> float:
    r = df[(df.dataset == dataset) & (df.method == row) & (df.probe == probe)]
    return float(r.test.iloc[0]) if len(r) else float("nan")


def _verdict(ref: pd.DataFrame) -> str:
    r = ref[(ref.dataset == RULE_DATASET) & (ref.probe == "valsel")]
    if r.empty:
        return f"E6 rule: no {RULE_DATASET} results yet"
    r = r.iloc[0]
    fails = r.D >= RULE_THRESHOLD and r.D_lo > 0
    lines = [
        f"E6 rule on {RULE_DATASET} (valsel probe, standard split, GCN, "
        f"{r.gcn_objectives} objectives): D = {r.D:+.2f} [{r.D_lo:+.2f}, {r.D_hi:+.2f}] points "
        f"(A^2X {r.A2X:.2f}, objectives {r.gcn_min:.2f}-{r.gcn_max:.2f}, mean {r.gcn_mean:.2f})",
        "VERDICT: "
        + (
            "C10 is stated for the six datasets only; WikiCS is the counterexample"
            if fails
            else "C10 stands; WikiCS is the out-of-sample test it passed"
        ),
        f"GCN objectives above A^2X: {r.gcn_above_A2X} of {r.gcn_objectives}; models of any "
        f"backbone above it: {r.models_above_A2X} of {r.models}",
    ]
    for probe in ["graphssl", "liblinear", "graphssl_random"]:
        s = ref[(ref.dataset == RULE_DATASET) & (ref.probe == probe)]
        if len(s):
            s = s.iloc[0]
            lines.append(f"  under {probe}: D = {s.D:+.2f} [{s.D_lo:+.2f}, {s.D_hi:+.2f}]")
    return "\n".join(lines)


RULE_ROWS = [
    ("raw", "Raw features"),
    ("A1X", r"$\hat AX$"),
    ("A2X", r"$\hat A^2X$"),
    ("PCA", r"PCA-256 of $\hat A^2X$"),
    None,
    ("untrained", "Untrained GCN"),
    ("dgi", "DGI"),
    ("graphcl", "GraphCL"),
    ("bgrl", "BGRL"),
    ("afgrl", "AFGRL"),
    ("vicreg", "VICReg"),
    ("barlow_twins", "Barlow Twins"),
    ("graphdino", "GraphDINO"),
    ("graphmae", "GraphMAE"),
    None,
    ("supervised", r"Supervised$^\dagger$"),
    ("supervised_reg", r"Supervised, std.\ recipe$^\dagger$"),
]


def _rule_table(df: pd.DataFrame, cells: pd.DataFrame) -> str:
    """LaTeX body: every quantity the E6 rule reports on WikiCS, one row per representation
    (input rows, and the GCN models as seed means), under each regularized probe."""
    w = cells[(cells.dataset == RULE_DATASET) & (cells.backbone == "gcn")]
    a2x = _input(df, RULE_DATASET, "A2X", "valsel")

    def value(key: str, probe: str) -> tuple[float, float]:
        if key in INPUT_ROWS.values():
            return _input(df, RULE_DATASET, key, probe), float("nan")
        r = w[(w.method == key) & (w.probe == probe)]
        return (float(r["mean"].iloc[0]), float(r["std"].iloc[0])) if len(r) else (np.nan, np.nan)

    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Standard split} & 10 random splits \\",
        r"\cmidrule(lr){2-5}\cmidrule(l){6-6}",
        r"Representation & Ours & liblinear & Better on val. & $-\,\hat A^2X$ & Ours \\",
        r"\midrule",
    ]
    for row in RULE_ROWS:
        if row is None:
            lines.append(r"\midrule")
            continue
        key, name = row
        ours, lib, (sel, sd), rand = (
            value(key, "graphssl")[0],
            value(key, "liblinear")[0],
            value(key, "valsel"),
            value(key, "graphssl_random")[0],
        )
        if pd.isna(sel):
            continue
        shown = f"{sel:.1f}" if pd.isna(sd) else f"{sel:.1f} $\\pm$ {sd:.1f}"
        delta = "--" if key == "A2X" else f"${sel - a2x:+.1f}$"
        # the supervised encoders saw labels of the standard split, test nodes of a random one
        split = "--" if key in SUPERVISED else f"{rand:.1f}"
        lines.append(f"{name} & {ours:.1f} & {lib:.1f} & {shown} & {delta} & {split} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def _table(ref: pd.DataFrame) -> str:
    """LaTeX body: one row per dataset, the valsel probe (our probe where liblinear is absent)."""
    lines = [
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{No training} & \multicolumn{3}{c}{GCN} & Any backbone \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-8}",
        r"Dataset & Raw & $\hat A^2X$ & PCA-256 & Untrained & Objectives & $D$ [95\% CI] "
        r"& Above $\hat A^2X$ \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        r = ref[(ref.dataset == dataset) & (ref.probe == "valsel")]
        if r.empty:
            continue
        r = r.iloc[0]
        if dataset == RULE_DATASET:
            lines.append(r"\midrule")

        def f(x: float) -> str:
            return "--" if pd.isna(x) else f"{x:.1f}"

        lines.append(
            f"{NAMES.get(dataset, dataset)} & {f(r.raw)} & {f(r.A2X)} & {f(r.PCA)} & "
            f"{f(r.untrained_gcn)} & {r.gcn_mean:.1f} ({r.gcn_min:.1f}--{r.gcn_max:.1f}) & "
            f"${r.D:+.1f}$ $[{r.D_lo:+.1f}, {r.D_hi:+.1f}]$ & "
            f"{r.models_above_A2X} of {r.models} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
