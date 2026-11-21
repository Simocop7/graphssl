"""E7 (normalization statistics at evaluation), E8 (row-normalized features) and the checks of
C10, with the rules fixed before the runs (benchmarks/paper/PLAN.md, "E7", "E8").

Reads the probe files of E7 (``e7/probes_running/<D>``, ``e7/probes_graph/<D>``), of E8
(``e8/probes/<D>``), of E1/E2 (``probes/<D>``), the E6 reference (``analysis/e6``), and the E3
and E5 results. Writes ``analysis/e7/``, ``analysis/e8/``, ``analysis/c10/`` and the tables
``figures/tab_e7.tex`` and ``figures/tab_c10_checks.tex``.

    python benchmarks/analyze_e7_e8.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_factors import eta_squared  # noqa: E402

P = Path("benchmarks/paper")
SIX = ["Cora", "CiteSeer", "PubMed", "AmazonComputers", "AmazonPhoto", "CoauthorCS"]
PLANETOID = SIX[:3]
MANY_LABELS = SIX[3:]
ARXIV = ["ogbn-arxiv", "arxiv-year"]
NAMES = {
    "Cora": "Cora",
    "CiteSeer": "CiteSeer",
    "PubMed": "PubMed",
    "AmazonComputers": "Amazon Computers",
    "AmazonPhoto": "Amazon Photo",
    "CoauthorCS": "Coauthor CS",
    "ogbn-arxiv": "ogbn-arxiv",
    "arxiv-year": "arxiv-year",
}
BACKBONES = ["gcn", "gin", "transformer"]
BB_NAMES = {"gcn": "GCN", "gin": "GIN", "transformer": "Transformer"}
OBJECTIVES = ["dgi", "graphcl", "bgrl", "afgrl", "vicreg", "barlow_twins", "graphdino", "graphmae"]
INPUT_ROWS = {
    "raw features": "raw",
    "propagated A^1X": "A1X",
    "propagated A^2X": "A2X",
    "PCA-256 of A^2X": "PCA",
}
THRESHOLD = 1.0
N_BOOT = 10_000


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _test(protocol: dict) -> tuple[float, float]:
    """(val, test) accuracy in %, averaged over a protocol's fits."""
    fits = protocol.get("fits") or []
    if fits:
        return (
            100 * float(np.mean([f.get("val_acc", np.nan) for f in fits])),
            100 * float(np.mean([f["test_acc"] for f in fits])),
        )
    t = protocol["test_acc"]
    return float("nan"), 100 * (t["mean"] if isinstance(t, dict) else t)


def load_probes(pattern_dirs: list[str], datasets: list[str]) -> pd.DataFrame:
    """One row per (representation, probe), newest scoring per checkpoint or input row. Probes:
    every protocol of the files, ``knn`` and ``valsel`` (ours or liblinear, the better on
    validation; ours on ties). Columns also carry the effective rank and the embedding scale."""
    files = []
    for d in datasets:
        for root in pattern_dirs:
            files += glob.glob(str(P / root / d / "probes__*.json"))
    latest: dict = {}
    for f in sorted(files, key=os.path.basename):
        doc = json.load(open(f))
        dataset = doc["dataset"]
        for r in doc["rows"]:
            ckpt = r.get("checkpoint")
            if ckpt in (None, "None"):
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
            probes = {name: _test(proto) for name, proto in r["protocols"].items()}
            probes["knn"] = (float("nan"), 100 * float(r["test_acc_knn"]))
            ours, lib = probes.get("graphssl"), probes.get("liblinear")
            if ours is not None:
                probes["valsel"] = lib if (lib is not None and lib[0] > ours[0]) else ours
            latest[key] = {
                "dataset": dataset,
                **meta,
                "rank": float(r.get("eff_rank", np.nan)),
                "scale": float(r.get("feature_std_mean", np.nan)),
                "probes": probes,
            }
    out = []
    for e in latest.values():
        probes = e.pop("probes")
        for name, (val, test) in probes.items():
            out.append({**e, "probe": name, "val": val, "test": test})
    return pd.DataFrame(out)


def wide(df: pd.DataFrame, probe: str) -> pd.DataFrame:
    """Test accuracy of the models as [dataset, backbone, method, seed] -> value."""
    d = df[(df.probe == probe) & (df.seed >= 0)]
    return d.set_index(["dataset", "backbone", "method", "seed"]).test


def gains(acc: pd.Series) -> pd.DataFrame:
    """Gain of every objective over the untrained encoder of the same seed and backbone."""
    a = acc.unstack("method")
    objectives = [m for m in OBJECTIVES if m in a.columns]
    g = a[objectives].sub(a["untrained"], axis=0)
    return g.stack().rename("gain").reset_index()


def boot_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    draws = values[rng.integers(0, len(values), (N_BOOT, len(values)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def bootstrap_d(per: dict, ref: float, rng: np.random.Generator) -> tuple[float, float, float]:
    """D = mean over objectives of (seed-mean accuracy - ref), seeds resampled per objective."""
    point = float(np.mean([v.mean() for v in per.values()]) - ref)
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        draws[i] = np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in per.values()])
    lo, hi = np.percentile(draws - ref, [2.5, 97.5])
    return point, float(lo), float(hi)


def a2x_valsel() -> pd.Series:
    ref = pd.read_csv(P / "analysis" / "e6" / "reference.csv")
    return ref[ref.probe == "valsel"].set_index("dataset").A2X


# ---------------------------------------------------------------------------
# E7
# ---------------------------------------------------------------------------


def e7(out: Path, rng: np.random.Generator) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    # ogbn-arxiv and arxiv-year (descriptive, on the Mac GPU) join the per-cell tables when scored.
    run = load_probes(["e7/probes_running"], SIX + ARXIV)
    graph = load_probes(["e7/probes_graph"], SIX + ARXIV)
    if run.empty or graph.empty:
        print("E7: no probe files yet")
        return {}
    e2 = load_probes(["probes"], SIX + ARXIV)

    # Reproduction: the running rows of E7 against E2 (same checkpoints, same probes). The six
    # datasets ran on E2's device and threads; ogbn-arxiv on another device (a device check).
    rep, rep_arxiv = [], []
    for probe in ["adam100", "adam5000", "liblinear", "graphssl", "knn"]:
        a_all, b_all = wide(run, probe), wide(e2, probe)
        arx = a_all[a_all.index.get_level_values("dataset").isin(ARXIV)]
        common_x = arx.index.intersection(b_all.index)
        if len(common_x):
            dx = (arx.loc[common_x] - b_all.loc[common_x]).abs()
            rep_arxiv.append({"probe": probe, "compared": len(common_x),
                              "mean_abs_diff": float(dx.mean()), "max_abs_diff": float(dx.max())})  # fmt: skip
        a = a_all[a_all.index.get_level_values("dataset").isin(SIX)]
        b = b_all
        common = a.index.intersection(b.index)
        diff = (a.loc[common] - b.loc[common]).abs()
        rep.append(
            {
                "probe": probe,
                "compared": len(common),
                "identical": int((diff < 1e-9).sum()),
                "max_abs_diff": float(diff.max()) if len(diff) else float("nan"),
            }
        )
    pd.DataFrame(rep).to_csv(out / "reproduction.csv", index=False)
    pd.DataFrame(rep_arxiv).to_csv(out / "reproduction_arxiv_device.csv", index=False)

    # Untrained encoders and mean gains, per condition. "mixed" (descriptive, not in a rule):
    # the objectives with the statistics they hold against the untrained encoder with graph
    # statistics, i.e. only the reference changes.
    mixed = pd.concat([run[run.method != "untrained"], graph[graph.method == "untrained"]])
    rows, cells = [], []
    for cond, df in (("running", run), ("graph", graph), ("mixed", mixed)):
        for probe in ["adam100", "adam5000", "liblinear", "graphssl", "valsel", "knn"]:
            acc = wide(df, probe)
            if acc.empty:
                continue
            g = gains(acc)
            for (dataset, backbone), gg in g.groupby(["dataset", "backbone"]):
                u = acc.xs((dataset, backbone, "untrained"), level=[0, 1, 2])
                cells.append(
                    {
                        "condition": cond,
                        "probe": probe,
                        "dataset": dataset,
                        "backbone": backbone,
                        "untrained": float(u.mean()),
                        "objectives": float(
                            acc.xs((dataset, backbone), level=[0, 1])
                            .drop("untrained", level="method")
                            .groupby(level="method")
                            .mean()
                            .mean()
                        ),
                        "mean_gain": float(gg.groupby("method").gain.mean().mean()),
                    }
                )
        meta = df[(df.method == "untrained") & (df.probe == "graphssl")]
        for (dataset, backbone), m in meta.groupby(["dataset", "backbone"]):
            rows.append(
                {
                    "condition": cond,
                    "dataset": dataset,
                    "backbone": backbone,
                    "untrained_rank": m["rank"].mean(),
                    "untrained_scale": m["scale"].mean(),
                    "trained_scale": df[
                        (df.dataset == dataset)
                        & (df.backbone == backbone)
                        & df.method.isin(OBJECTIVES)
                        & (df.probe == "graphssl")
                    ]["scale"].mean(),
                }
            )
    cells_df = pd.DataFrame(cells)
    cells_df.round(3).to_csv(out / "cells.csv", index=False)
    meta_df = pd.DataFrame(rows)
    meta_df.round(4).to_csv(out / "untrained.csv", index=False)

    def cell(cond, probe, dataset, backbone, col="mean_gain"):
        c = cells_df[
            (cells_df.condition == cond)
            & (cells_df.probe == probe)
            & (cells_df.dataset == dataset)
            & (cells_df.backbone == backbone)
        ]
        return float(c[col].iloc[0]) if len(c) else float("nan")

    lines = ["E7 rules (benchmarks/paper/PLAN.md), fixed at f0fb47d before the runs", ""]

    # R1: C3, small GCN gains, ours, standard split.
    r1 = []
    for dataset in MANY_LABELS:
        gr = gains(wide(run, "graphssl")).query("dataset == @dataset and backbone == 'gcn'")
        gg = gains(wide(graph, "graphssl")).query("dataset == @dataset and backbone == 'gcn'")
        per_seed = (gg.groupby("seed").gain.mean() - gr.groupby("seed").gain.mean()).to_numpy()
        point, lo, hi = boot_mean(per_seed, rng)
        r1.append(
            {
                "dataset": dataset,
                "gain_running": gr.groupby("method").gain.mean().mean(),
                "gain_graph": gg.groupby("method").gain.mean().mean(),
                "change": point,
                "lo": lo,
                "hi": hi,
                "dependent": bool(point >= THRESHOLD and lo > 0),
            }
        )
    r1_df = pd.DataFrame(r1)
    r1_df.round(3).to_csv(out / "rule1_c3.csv", index=False)
    lines.append("R1 (C3): mean GCN gain, ours, graph minus running statistics")
    for r in r1:
        lines.append(
            f"  {NAMES[r['dataset']]}: running {r['gain_running']:+.2f}, graph "
            f"{r['gain_graph']:+.2f}, change {r['change']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]"
        )
    r1_dep = any(r["dependent"] for r in r1)
    lines.append(
        "  VERDICT: C3 "
        + (
            "is reported as dependent on the normalization statistics"
            if r1_dep
            else "stands under both"
        )
    )

    # R2: C10, D with the objectives under graph statistics (valsel).
    a2x = a2x_valsel()
    r2 = []
    for dataset in SIX:
        acc = wide(graph, "valsel").xs((dataset, "gcn"), level=[0, 1])
        per = {m: acc.xs(m, level="method").to_numpy() for m in OBJECTIVES}
        d, lo, hi = bootstrap_d(per, float(a2x[dataset]), rng)
        accr = wide(run, "valsel").xs((dataset, "gcn"), level=[0, 1])
        per_r = {m: accr.xs(m, level="method").to_numpy() for m in OBJECTIVES}
        d_r = float(np.mean([v.mean() for v in per_r.values()]) - a2x[dataset])
        means = {m: v.mean() for m, v in per.items()}
        r2.append(
            {
                "dataset": dataset,
                "D_running": d_r,
                "D_graph": d,
                "lo": lo,
                "hi": hi,
                "above": int(sum(v > a2x[dataset] for v in means.values())),
                "breaks": bool(d >= THRESHOLD and lo > 0),
            }
        )
    r2_df = pd.DataFrame(r2)
    r2_df.round(3).to_csv(out / "rule2_c10.csv", index=False)
    lines += ["", "R2 (C10): D (valsel) with the GCN objectives under graph statistics"]
    for r in r2:
        lines.append(
            f"  {NAMES[r['dataset']]}: D running {r['D_running']:+.2f}, graph {r['D_graph']:+.2f} "
            f"[{r['lo']:+.2f}, {r['hi']:+.2f}], objectives above A2X {r['above']} of 8"
        )
    r2_break = any(r["breaks"] for r in r2)
    lines.append(
        "  VERDICT: C10 "
        + ("is stated with the models' own statistics" if r2_break else "stands under both")
    )

    # R3: the probe's two biases.
    infl = []
    for cond in ("running", "graph", "mixed"):
        for dataset in SIX:
            for bb in BACKBONES:
                infl.append(
                    {
                        "condition": cond,
                        "dataset": dataset,
                        "backbone": bb,
                        "gain_adam100": cell(cond, "adam100", dataset, bb),
                        "gain_ours": cell(cond, "graphssl", dataset, bb),
                        "inflation": cell(cond, "adam100", dataset, bb)
                        - cell(cond, "graphssl", dataset, bb),
                    }
                )
    infl_df = pd.DataFrame(infl)
    infl_df.round(3).to_csv(out / "rule3_inflation.csv", index=False)
    many = infl_df[infl_df.dataset.isin(MANY_LABELS)]
    few = infl_df[infl_df.dataset.isin(PLANETOID)]
    i_run = many[many.condition == "running"].inflation.mean()
    i_graph = many[many.condition == "graph"].inflation.mean()
    defl_run = int((few[few.condition == "running"].inflation < 0).sum())
    defl_graph = int((few[few.condition == "graph"].inflation < 0).sum())
    infl_cells_graph = int((many[many.condition == "graph"].inflation > 0).sum())
    i_mixed = many[many.condition == "mixed"].inflation.mean()
    lines += [
        "",
        "R3 (C4/C5): inflation = mean gain under adam100 minus under ours",
        f"  many labels (9 cells): I_run {i_run:+.2f}, I_graph {i_graph:+.2f} "
        f"(cells inflated under graph statistics: {infl_cells_graph} of 9); "
        f"descriptive, only the reference recomputed: {i_mixed:+.2f}",
        "  VERDICT (inflation): "
        + (
            "attributed mainly to the untrained encoder's initial normalization statistics"
            if i_graph < i_run / 2
            else "a property of the probe under both"
        ),
        f"  20 labels per class (9 cells): deflated in {defl_run} (running), {defl_graph} (graph)",
        "  VERDICT (deflation): "
        + (
            "the overfitting bias stands"
            if defl_graph >= 6
            else "reported as dependent on the statistics"
        ),
    ]

    # Descriptive: kNN with a GCN on the Amazon graphs, objectives significantly above / below
    # the untrained encoder (95% bootstrap over seeds of the paired gain), per condition.
    sig = []
    for cond, df in (("running", run), ("graph", graph)):
        g = gains(wide(df, "knn"))
        for dataset in ["AmazonComputers", "AmazonPhoto"]:
            gd = g[(g.dataset == dataset) & (g.backbone == "gcn")]
            above = below = 0
            for _, gm in gd.groupby("method"):
                _, lo, hi = boot_mean(gm.gain.to_numpy(), rng)
                above += int(lo > 0)
                below += int(hi < 0)
            sig.append({"condition": cond, "dataset": dataset, "above": above, "below": below})
    sig_df = pd.DataFrame(sig)
    sig_df.to_csv(out / "knn_amazon_gcn.csv", index=False)
    lines += ["", "Descriptive: kNN, GCN, Amazon: objectives significantly above / below untrained"]
    for _, r in sig_df.iterrows():
        lines.append(f"  {r.condition} {NAMES[r.dataset]}: above {r.above}, below {r.below}")

    # Descriptive: does the ranking of the objectives change? Kendall tau, running vs graph.
    from scipy.stats import kendalltau

    taus = []
    for probe in ("graphssl", "knn"):
        a, b = wide(run, probe), wide(graph, probe)
        for dataset in SIX:
            for bb in BACKBONES:
                ma = a.xs((dataset, bb), level=[0, 1]).groupby(level="method").mean()[OBJECTIVES]
                mb = b.xs((dataset, bb), level=[0, 1]).groupby(level="method").mean()[OBJECTIVES]
                taus.append(
                    {
                        "probe": probe,
                        "dataset": dataset,
                        "backbone": bb,
                        "tau": kendalltau(ma, mb).statistic,
                    }
                )
    taus_df = pd.DataFrame(taus)
    taus_df.round(3).to_csv(out / "tau_running_graph.csv", index=False)
    lines += [
        "",
        "Descriptive: Kendall tau between the rankings under running and graph statistics",
    ]
    for probe, g in taus_df.groupby("probe"):
        lines.append(
            f"  {probe}: median {g.tau.median():.2f}, min {g.tau.min():.2f} over {len(g)} cells"
        )

    # Descriptive: shares of the variance of the gain, ours, standard split.
    shares = []
    for cond, df in (("running", run), ("graph", graph)):
        g = gains(wide(df, "graphssl"))
        for dataset, gd in g.groupby("dataset"):
            s = eta_squared(gd, "gain", ["method", "backbone"], [("method", "backbone")])
            shares.append({"condition": cond, "dataset": dataset, **s})
    shares_df = pd.DataFrame(shares)
    shares_df.round(4).to_csv(out / "shares_gain.csv", index=False)
    lines += ["", "Descriptive: backbone share of the variance of the gain (ours, standard split)"]
    for dataset in SIX:
        s = shares_df[shares_df.dataset == dataset].set_index("condition")
        lines.append(
            f"  {NAMES[dataset]}: backbone {100 * s.loc['running', 'backbone']:.0f}% -> "
            f"{100 * s.loc['graph', 'backbone']:.0f}%, objective "
            f"{100 * s.loc['running', 'method']:.0f}% -> {100 * s.loc['graph', 'method']:.0f}%"
        )
    lines += ["", "Descriptive: spread of the untrained backbones (ours), running -> graph"]
    for dataset in SIX:
        sp = []
        for cond in ("running", "graph"):
            u = [cell(cond, "graphssl", dataset, bb, "untrained") for bb in BACKBONES]
            sp.append(max(u) - min(u))
        lines.append(f"  {NAMES[dataset]}: {sp[0]:.1f} -> {sp[1]:.1f}")
    lines += ["", "Reproduction of E2 by the running rows (same checkpoints, same probes):"]
    for r in rep:
        lines.append(
            f"  {r['probe']}: {r['identical']} of {r['compared']} identical, "
            f"max difference {r['max_abs_diff']:.4f}"
        )
    verdict = "\n".join(lines)
    (out / "verdict.txt").write_text(verdict + "\n")
    print(verdict)
    return {
        "cells": cells_df,
        "meta": meta_df,
        "r1": r1_df,
        "r2": r2_df,
        "inflation": infl_df,
        "shares": shares_df,
    }


def e7_table(res: dict, path: Path) -> None:
    """Appendix table: the untrained encoder and the mean gain of the eight objectives under
    running and graph statistics, three probes, every dataset and backbone."""
    cells, meta = res["cells"], res["meta"]

    def c(cond, probe, d, bb, col):
        x = cells[
            (cells.condition == cond)
            & (cells.probe == probe)
            & (cells.dataset == d)
            & (cells.backbone == bb)
        ][col]
        return float(x.iloc[0])

    def rank(cond, d, bb):
        x = meta[(meta.condition == cond) & (meta.dataset == d) & (meta.backbone == bb)]
        return float(x.untrained_rank.iloc[0])

    lines = [
        "% Generated by benchmarks/analyze_e7_e8.py from the E7 probe files. Do not edit.",
        r"\begin{tabular}{@{}llrrrrrrrrrr@{}}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{Untrained encoder} & \multicolumn{6}{c}{Mean gain of the eight objectives} \\",
        r"\cmidrule(lr){3-6}\cmidrule(l){7-12}",
        r" & & \multicolumn{2}{c}{Ours} & \multicolumn{2}{c}{100 steps} & \multicolumn{2}{c}{Ours} & \multicolumn{2}{c}{100 steps} & \multicolumn{2}{c}{$k$NN} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(l){11-12}",
        r"Dataset & Backbone & run. & graph & run. & graph & run. & graph & run. & graph & run. & graph \\",
        r"\midrule",
    ]
    for i, d in enumerate(SIX):
        for j, bb in enumerate(BACKBONES):
            name = NAMES[d] if j == 0 else ""
            vals = [
                c("running", "graphssl", d, bb, "untrained"),
                c("graph", "graphssl", d, bb, "untrained"),
                c("running", "adam100", d, bb, "untrained"),
                c("graph", "adam100", d, bb, "untrained"),
            ]
            g = [
                c(cond, probe, d, bb, "mean_gain")
                for probe in ("graphssl", "adam100", "knn")
                for cond in ("running", "graph")
            ]
            cells_txt = [f"{v:.1f}" for v in vals] + [f"{v:+.1f}".replace("-", "$-$") for v in g]
            lines.append(f"{name} & {BB_NAMES[bb]} & " + " & ".join(cells_txt) + r" \\")
        if i < len(SIX) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# E8
# ---------------------------------------------------------------------------


def e8(out: Path, rng: np.random.Generator) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    rn = load_probes(["e8/probes"], PLANETOID)
    if rn.empty:
        print("E8: no probe files yet")
        return pd.DataFrame()
    e2 = load_probes(["probes"], PLANETOID)
    a2x_raw = a2x_valsel()
    rows, lines = [], ["E8 rule (benchmarks/paper/PLAN.md), fixed at f0fb47d before the runs", ""]
    for dataset in PLANETOID:
        inputs = rn[(rn.dataset == dataset) & (rn.seed < 0) & (rn.probe == "valsel")]
        a2x = float(inputs[inputs.method == "A2X"].test.iloc[0])
        raw_rn = float(inputs[inputs.method == "raw"].test.iloc[0])
        acc = wide(rn, "valsel").xs((dataset, "gcn"), level=[0, 1])
        per = {m: acc.xs(m, level="method").to_numpy() for m in OBJECTIVES}
        d, lo, hi = bootstrap_d(per, a2x, rng)
        best_ref = max(a2x, float(a2x_raw[dataset]))
        acc1 = wide(e2, "valsel").xs((dataset, "gcn"), level=[0, 1])
        per1 = {
            m: acc1.xs(m, level="method").loc[lambda s: s.index < 5].to_numpy() for m in OBJECTIVES
        }
        d1, lo1, hi1 = bootstrap_d(per1, float(a2x_raw[dataset]), rng)
        means = {m: v.mean() for m, v in per.items()}
        untrained = float(acc.xs("untrained", level="method").mean())
        row = {
            "dataset": dataset,
            "A2X_rownorm": a2x,
            "A2X_raw": float(a2x_raw[dataset]),
            "raw_rownorm": raw_rn,
            "untrained_gcn": untrained,
            "objectives_mean": float(np.mean(list(means.values()))),
            "D_rn": d,
            "lo": lo,
            "hi": hi,
            "D_vs_better_A2X": float(np.mean(list(means.values())) - best_ref),
            "D_e1_seeds0_4": d1,
            "D_e1_lo": lo1,
            "D_e1_hi": hi1,
            "above": int(sum(v > a2x for v in means.values())),
            "breaks": bool(d >= THRESHOLD and lo > 0),
            **{f"obj_{m}": v - a2x for m, v in means.items()},
        }
        rows.append(row)
        lines.append(
            f"{NAMES[dataset]}: A2X (row-normalized) {a2x:.1f} (raw {a2x_raw[dataset]:.1f}), "
            f"objectives {row['objectives_mean']:.1f}, D_rn {d:+.2f} [{lo:+.2f}, {hi:+.2f}], "
            f"{row['above']} of 8 above; E1 seeds 0-4 (raw) D {d1:+.2f}; untrained GCN {untrained:.1f}"
        )
    df = pd.DataFrame(rows)
    broken = bool(df.breaks.any())
    lines.append(
        "VERDICT: C10 "
        + ("is stated for raw features only" if broken else "holds with either preprocessing")
    )
    df.round(3).to_csv(out / "e8.csv", index=False)
    verdict = "\n".join(lines)
    (out / "verdict.txt").write_text(verdict + "\n")
    print(verdict)
    return df


# ---------------------------------------------------------------------------
# C10 checks on results in hand (descriptive) + E7 / E8
# ---------------------------------------------------------------------------


def e3_tuned(rng: np.random.Generator) -> pd.DataFrame:
    """GCN objectives at E3's shared cell and at their tuned cell (our probe), per seed."""
    choice = pd.read_csv(P / "analysis" / "e3" / "e3.csv")
    choice = choice[choice.backbone == "gcn"]
    runs: dict = {}
    for f in sorted(glob.glob(str(P / "e3" / "*" / "*" / "*.json")), key=os.path.basename):
        r = json.load(open(f))
        if r["hyperparameters"].get("encoder") != "gcn":
            continue
        cell_name = Path(f).parent.parent.name
        runs[(cell_name, r["dataset"], r["model"])] = {
            s["seed"]: 100 * s["test_acc_linear"]
            for s in r["per_seed"]
            if s.get("test_acc_linear") is not None
        }
    a2x = a2x_valsel()
    rows = []
    for dataset, g in choice.groupby("dataset"):
        shared = {m: np.array(list(runs[("adamw_lr5e-4", dataset, m)].values())) for m in g.method}
        tuned = {
            m: np.array(list(runs[(c, dataset, m)].values()))
            for m, c in zip(g.method, g.cell, strict=True)
        }
        ds, _, _ = bootstrap_d(shared, float(a2x[dataset]), rng)
        dt, lo, hi = bootstrap_d(tuned, float(a2x[dataset]), rng)
        best = max(v.mean() for v in tuned.values()) - a2x[dataset]
        rows.append(
            {
                "dataset": dataset,
                "D_shared": ds,
                "D_tuned": dt,
                "lo": lo,
                "hi": hi,
                "best_tuned": best,
            }
        )
    return pd.DataFrame(rows)


def e5_budget(rng: np.random.Generator) -> pd.DataFrame:
    """GCN objectives at 300 steps and at the best validation step of 3,000-step runs (ours)."""
    from analyze_e5 import arrays
    from analyze_e5 import load as load_e5

    runs = load_e5(Path("."))
    a2x = a2x_valsel()
    rows = []
    for dataset in SIX:
        short, stop = {}, {}
        for m in OBJECTIVES:
            a = arrays(runs, dataset, "gcn", m)
            if a is None:
                break
            s = int(np.argmax(a["val"].mean(axis=0)))
            short[m], stop[m] = a["g300_test"], a["test"][:, s]
        if len(stop) < len(OBJECTIVES):
            continue
        d3, _, _ = bootstrap_d(short, float(a2x[dataset]), rng)
        ds, lo, hi = bootstrap_d(stop, float(a2x[dataset]), rng)
        rows.append({"dataset": dataset, "D_300": d3, "D_stop": ds, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def c10_checks(
    out: Path, e7res: dict, e8df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    e2 = load_probes(["probes"], SIX)
    a2x = a2x_valsel()
    ref = pd.read_csv(P / "analysis" / "e6" / "reference.csv")
    ref = ref[ref.probe == "valsel"].set_index("dataset")
    tuned = e3_tuned(rng).set_index("dataset")
    budget = e5_budget(rng).set_index("dataset")
    rows = []
    for dataset in SIX:
        acc = wide(e2, "valsel").xs((dataset, "gcn"), level=[0, 1])
        per = {m: acc.xs(m, level="method").to_numpy() for m in OBJECTIVES}
        no_dino = {m: v for m, v in per.items() if m != "graphdino"}
        d_nd, lo_nd, hi_nd = bootstrap_d(no_dino, float(a2x[dataset]), rng)
        best = max(v.mean() for v in per.values()) - a2x[dataset]
        sup = (
            acc.xs("supervised_reg", level="method").mean() - a2x[dataset]
            if ("supervised_reg" in acc.index.get_level_values("method"))
            else np.nan
        )
        row = {
            "dataset": dataset,
            "D": float(ref.loc[dataset, "D"]),
            "D_lo": float(ref.loc[dataset, "D_lo"]),
            "D_hi": float(ref.loc[dataset, "D_hi"]),
            "D_no_graphdino": d_nd,
            "D_no_graphdino_lo": lo_nd,
            "D_no_graphdino_hi": hi_nd,
            "best_objective": best,
            "supervised_std_recipe": float(sup),
        }
        if dataset in tuned.index:
            row.update(
                D_e3_shared=tuned.loc[dataset, "D_shared"],
                D_e3_tuned=tuned.loc[dataset, "D_tuned"],
                D_e3_tuned_lo=tuned.loc[dataset, "lo"],
                D_e3_tuned_hi=tuned.loc[dataset, "hi"],
                best_e3_tuned=tuned.loc[dataset, "best_tuned"],
            )
        if dataset in budget.index:
            row.update(
                D_e5_300=budget.loc[dataset, "D_300"],
                D_e5_stop=budget.loc[dataset, "D_stop"],
                D_e5_stop_lo=budget.loc[dataset, "lo"],
                D_e5_stop_hi=budget.loc[dataset, "hi"],
            )
        if e7res:
            r2 = e7res["r2"].set_index("dataset")
            row.update(
                D_e7_graph=r2.loc[dataset, "D_graph"],
                D_e7_graph_lo=r2.loc[dataset, "lo"],
                D_e7_graph_hi=r2.loc[dataset, "hi"],
            )
        if not e8df.empty and dataset in set(e8df.dataset):
            r8 = e8df.set_index("dataset")
            row.update(
                D_e8_rownorm=r8.loc[dataset, "D_rn"],
                D_e8_rownorm_lo=r8.loc[dataset, "lo"],
                D_e8_rownorm_hi=r8.loc[dataset, "hi"],
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.round(3).to_csv(out / "checks.csv", index=False)
    return df


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:+.1f}".replace("-", "$-$")


def _ci(lo: float, hi: float) -> str:
    if lo is None or np.isnan(lo):
        return ""
    return r"{\tiny [" + _fmt(lo) + ", " + _fmt(hi) + "]}"


def c10_table(df: pd.DataFrame, path: Path) -> None:
    cols = [
        ("Shared\\\\protocol", "D", "D_lo", "D_hi"),
        ("Without\\\\GraphDINO", "D_no_graphdino", "D_no_graphdino_lo", "D_no_graphdino_hi"),
        ("Graph statistics (E7)", "D_e7_graph", "D_e7_graph_lo", "D_e7_graph_hi"),
        ("Row-normalized $X$ (E8)", "D_e8_rownorm", "D_e8_rownorm_lo", "D_e8_rownorm_hi"),
        ("Tuned (E3)", "D_e3_tuned", "D_e3_tuned_lo", "D_e3_tuned_hi"),
        ("3{,}000 steps (E5)", "D_e5_stop", "D_e5_stop_lo", "D_e5_stop_hi"),
    ]
    cols = [c for c in cols if c[1] in df.columns]
    lines = [
        "% Generated by benchmarks/analyze_e7_e8.py from the result files. Do not edit.",
        r"\begin{tabular}{@{}l" + "r" * (len(cols) + 2) + "@{}}",
        r"\toprule",
        r" & \multicolumn{" + str(len(cols)) + r"}{c}{$D$: average GCN objective minus $\hat A^2X$}"
        r" & \multicolumn{2}{c}{Minus $\hat A^2X$} \\",
        r"\cmidrule(lr){2-"
        + str(len(cols) + 1)
        + r"}\cmidrule(l){"
        + str(len(cols) + 2)
        + "-"
        + str(len(cols) + 3)
        + "}",
        "Dataset & "
        + " & ".join(
            r"\begin{tabular}[b]{@{}c@{}}" + c[0].replace(" (", r"\\(") + r"\end{tabular}"
            for c in cols
        )
        + r" & \begin{tabular}[b]{@{}c@{}}Best\\objective\end{tabular}"
        + r" & \begin{tabular}[b]{@{}c@{}}Supervised\\GCN\end{tabular} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        vals = []
        for _, k, _lo, _hi in cols:
            v = r.get(k, np.nan)
            vals.append(_fmt(v))  # intervals: analysis/c10/checks.csv and the caption
        vals += [_fmt(r["best_objective"]), _fmt(r["supervised_std_recipe"])]
        lines.append(NAMES[r["dataset"]] + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    rng = np.random.default_rng(0)
    e7res = e7(P / "analysis" / "e7", rng)
    if e7res:
        e7_table(e7res, P / "figures" / "tab_e7.tex")
    e8df = e8(P / "analysis" / "e8", rng)
    checks = c10_checks(P / "analysis" / "c10", e7res, e8df, rng)
    c10_table(checks, P / "figures" / "tab_c10_checks.tex")
    print(checks.round(2).to_string())


if __name__ == "__main__":
    main()
