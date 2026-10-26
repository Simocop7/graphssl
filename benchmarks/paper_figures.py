"""Figures of the factor-study paper, rendered from result files only (never retyped).

fig1  Linear-probe test accuracy of each method (mean and 95% bootstrap interval over seeds)
      per dataset and backbone, against the untrained encoder of the same backbone (band) and
      the standard supervised recipe (line). From run_benchmark.py JSONs.
fig2  Gain over the untrained encoder under each probe protocol, per method, dataset and
      backbone. From analyze_factors.py's cells.csv.
fig3  Share of variance explained by each factor (linear probes, public split). From
      analyze_factors.py's variance_shares.csv.
fig4  What training changes: linear CKA between every pair of models (Cora), and the smoothing
      of the untrained vs trained encoders per dataset and backbone; with study_metrics,
      study_cka and study_predict (the label-free prediction test). From
      analyze_representations.py.
tab_*.tex   LaTeX tables for paper_tmlr/.
study_*.md  The same results as Markdown tables, included by docs/study.md.

Usage::

    python benchmarks/paper_figures.py --results benchmarks/results \\
        --untrained benchmarks/ablations/untrained_encoder \\
        --analysis benchmarks/paper/analysis --out benchmarks/paper/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METHODS = ["dgi", "graphcl", "bgrl", "afgrl", "vicreg", "barlow_twins", "graphdino"]
NAMES = {
    "dgi": "DGI",
    "graphcl": "GraphCL",
    "bgrl": "BGRL",
    "afgrl": "AFGRL",
    "vicreg": "VICReg",
    "barlow_twins": "Barlow Twins",
    "graphdino": "GraphDINO",
    "untrained": "Untrained",
    "supervised_reg": "Supervised (std. recipe)",
}
DATASETS = ["Cora", "CiteSeer", "PubMed"]
BACKBONES = ["gcn", "gin"]
# Validated categorical palette (light surface), fixed order; ink and chrome colors.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# Figure 2: the protocols of public codebases, the two controls, and the converged probe.
PROBES = ["adam100", "adam5000", "adam100_std", "graphssl_weakl2", "liblinear", "graphssl"]
PROBE_NAMES = {
    "adam100": "100 Adam steps, raw (DGI, PyG-SSL)",
    "adam5000": "5,000 Adam steps, raw (PyGCL)",
    "adam100_std": "100 Adam steps, standardized",
    "adam5000_std": "5,000 Adam steps, standardized",
    "graphssl_weakl2": "L-BFGS, standardized, L2 1e-6",
    "liblinear": "liblinear, L2-normalized, C on val. (BGRL)",
    "graphssl": "L-BFGS, standardized, L2 on val. (GraphSSL)",
    "knn": "kNN, k = 5",
}
# Short protocol names for the LaTeX probe-ablation table.
PROBE_LABELS = {
    "adam100": "100 Adam steps",
    "adam100_std": "100 Adam steps",
    "adam5000": "5{,}000 Adam steps",
    "adam5000_std": "5{,}000 Adam steps",
    "graphssl_weakl2": "Converged, no selection",
    "liblinear": "liblinear",
    "graphssl": "Converged",
}
# Probe ablation table: what each protocol does, in the order of the columns.
ABLATION = [
    ("adam100", "raw", "none (100 steps)"),
    ("adam100_std", "standardized", "none (100 steps)"),
    ("adam5000", "raw", "none (5,000 steps, best val.)"),
    ("adam5000_std", "standardized", "none (5,000 steps, best val.)"),
    ("graphssl_weakl2", "standardized", "L2 $10^{-6}$, converged"),
    ("liblinear", "L2-normalized", "$C$ on val., converged"),
    ("graphssl", "standardized", "L2 on val., converged"),
]

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
    }
)


def latest(root: Path, encoder_key: str = "encoder") -> dict:
    """{(dataset, encoder, model): result}, newest file per key, current-probe results only."""
    runs: dict = {}
    for path in sorted(root.glob("**/*.json")):
        r = json.loads(path.read_text())
        hp = r.get("hyperparameters", {})
        if "per_seed" not in r or "linear_probe" not in hp:
            continue
        key = (r["dataset"], hp[encoder_key], r["model"])
        if (
            key not in runs
            or r["provenance"]["timestamp_utc"] > runs[key]["provenance"]["timestamp_utc"]
        ):
            runs[key] = r
    return runs


def mean_ci(values: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def accuracies(result: dict) -> np.ndarray:
    return 100 * np.array([s["test_acc_linear"] for s in result["per_seed"]])


def fig1(results: Path, untrained: Path, out: Path) -> None:
    runs, base = latest(results), latest(untrained)
    datasets = [d for d in DATASETS if any(k[0] == d for k in runs)]
    fig, axes = plt.subplots(
        len(datasets), 2, figsize=(6.8, 1.75 * len(datasets) + 0.5), sharey=True
    )
    axes = np.atleast_2d(axes)
    for i, dataset in enumerate(datasets):
        lo_x, hi_x = 100.0, 0.0
        for j, backbone in enumerate(BACKBONES):
            ax = axes[i, j]
            ys = np.arange(len(METHODS))[::-1]
            for y, method in zip(ys, METHODS, strict=True):
                r = runs.get((dataset, backbone, method))
                if r is None:
                    continue
                m, lo, hi = mean_ci(accuracies(r))
                ax.plot([lo, hi], [y, y], color=SERIES[0], lw=2, solid_capstyle="round")
                ax.plot(m, y, "o", ms=4.5, color=SERIES[0], mec="white", mew=1)
                lo_x, hi_x = min(lo_x, lo), max(hi_x, hi)
            u = base.get((dataset, backbone, "bgrl"))
            if u is not None:
                m, lo, hi = mean_ci(accuracies(u))
                ax.axvspan(lo, hi, color=MUTED, alpha=0.18, lw=0)
                ax.axvline(m, color=MUTED, lw=1)
                ax.text(m, len(METHODS) - 0.35, " untrained", color=INK_2, fontsize=7, va="bottom")
                lo_x = min(lo_x, lo)
            s = runs.get((dataset, backbone, "supervised_reg"))
            if s is not None:
                m = float(accuracies(s).mean())
                ax.axvline(m, color=SERIES[1], lw=1, ls=(0, (3, 2)))
                ax.text(m, -0.9, " supervised", color=INK_2, fontsize=7, va="bottom")
                hi_x = max(hi_x, m)
            ax.set_yticks(ys, [NAMES[m] for m in METHODS])
            ax.grid(axis="y", visible=False)
            ax.set_ylim(-1, len(METHODS))
            if i == 0:
                ax.set_title(backbone.upper(), color=INK, fontsize=9, fontweight="bold")
            if j == 0:
                ax.set_ylabel(dataset, color=INK, fontsize=9, fontweight="bold")
        for ax in axes[i]:
            ax.set_xlim(np.floor(lo_x - 2), np.ceil(hi_x + 2))
    fig.supxlabel(
        "Linear-probe test accuracy (%), mean and 95% CI over 10 seeds", fontsize=8, color=INK_2
    )
    fig.tight_layout()
    save(fig, out / "fig1_backbone_vs_method")


def fig2(analysis: Path, out: Path) -> None:
    cells = pd.read_csv(analysis / "cells.csv")
    cells = cells[cells.method.isin(METHODS)]
    panels = sorted({(d, b) for d, b in zip(cells.dataset, cells.backbone, strict=True)})
    fig, axes = plt.subplots(1, len(panels), figsize=(1.9 * len(panels) + 1.2, 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    probes = [p for p in PROBES if p in set(cells.probe)]
    offsets = np.linspace(-0.3, 0.3, len(probes))
    for ax, (dataset, backbone) in zip(axes, panels, strict=True):
        g = cells[(cells.dataset == dataset) & (cells.backbone == backbone)]
        ys = np.arange(len(METHODS))[::-1]
        for k, probe in enumerate(probes):
            gp = g[g.probe == probe].set_index("method")
            for y, method in zip(ys, METHODS, strict=True):
                if method not in gp.index:
                    continue
                row = gp.loc[method]
                yy = y + offsets[k]
                ax.plot([row.gain_lo, row.gain_hi], [yy, yy], color=SERIES[k], lw=1.2)
                ax.plot(row.gain, yy, "o", ms=3.5, color=SERIES[k], mec="white", mew=0.6)
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.set_yticks(ys, [NAMES[m] for m in METHODS])
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{dataset}, {backbone.upper()}", color=INK, fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Gain over untrained (points)")
    handles = [
        plt.Line2D([], [], marker="o", color=SERIES[k], lw=1.2, ms=4) for k in range(len(probes))
    ]
    fig.legend(
        handles,
        [PROBE_NAMES[p] for p in probes],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, out / "fig2_probe_protocols")


def fig3(analysis: Path, out: Path) -> None:
    v = pd.read_csv(analysis / "variance_shares.csv")
    v = v[v.design == "public split, linear probes"].reset_index(drop=True)
    factor_cols = [c for c in v.columns if c not in {"dataset", "design", "value"}]
    v = v.dropna(axis=1, how="all")
    factor_cols = [c for c in factor_cols if c in v.columns]
    labels = [
        f"{r.dataset}, {'accuracy' if r.value == 'acc' else 'gain over untrained'}"
        for r in v.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(6.8, 0.32 * len(v) + 1.2))
    left = np.zeros(len(v))
    for k, col in enumerate(factor_cols):
        share = v[col].fillna(0).to_numpy()
        color = MUTED if col.startswith("residual") else SERIES[k % len(SERIES)]
        ax.barh(
            np.arange(len(v)),
            share,
            left=left,
            height=0.6,
            color=color,
            edgecolor="white",
            lw=1,
            label=col,
        )
        left += share
    ax.set_yticks(np.arange(len(v)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of variance (eta squared)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, out / "fig3_variance_shares")


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    # No creation date in the files, so regenerating unchanged figures changes no byte.
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


def table_probe_ablation(analysis: Path, out: Path) -> None:
    """LaTeX rows: mean accuracy of the 7 trained methods, of the untrained encoder, and the
    gain, under each probe protocol, per dataset and backbone (public split)."""
    cells = pd.read_csv(analysis / "cells.csv")
    cells = cells[cells.split == "public"] if "split" in cells else cells
    t = cells.pivot_table(index=["dataset", "backbone", "method"], columns="probe", values="acc")
    trained = t[t.index.get_level_values("method").isin(METHODS)].groupby(level=[0, 1]).mean()
    untr = t.xs("untrained", level="method")
    panels = [(d, b) for d in DATASETS for b in BACKBONES if (d, b) in trained.index]
    lines = [
        "% Generated by benchmarks/paper_figures.py from analyze_factors.py output. Do not edit.",
        r"\begin{tabular}{lll" + "rrr" * len(panels) + "}",
        r"\toprule",
        " & ".join(
            ["Protocol", "Features", "Regularization"]
            + [rf"\multicolumn{{3}}{{c}}{{{d}, {b.upper()}}}" for d, b in panels]
        )
        + r" \\",
        " & ".join(["", "", ""] + ["SSL", "Untr.", r"$\Delta$"] * len(panels)) + r" \\",
        r"\midrule",
    ]
    for probe, features, reg in ABLATION:
        if probe not in trained.columns:
            continue
        cells_ = []
        for d, b in panels:
            a, u = trained.loc[(d, b), probe], untr.loc[(d, b), probe]
            cells_ += [f"{a:.1f}", f"{u:.1f}", f"{a - u:+.1f}"]
        lines.append(" & ".join([PROBE_LABELS[probe], features, reg, *cells_]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_probe_ablation.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'tab_probe_ablation.tex'}")


def table_main(results: Path, untrained: Path, out: Path) -> None:
    """LaTeX rows of the main table: linear probe mean ± std and gain over untrained."""
    runs, base = latest(results), latest(untrained)
    datasets = [d for d in DATASETS if any(k[0] == d for k in runs)]
    lines = [
        "% Generated by benchmarks/paper_figures.py from run_benchmark.py JSONs. Do not edit.",
        r"\begin{tabular}{ll" + "rr" * len(datasets) + "}",
        r"\toprule",
        " & ".join(["Backbone", "Method"] + [rf"\multicolumn{{2}}{{c}}{{{d}}}" for d in datasets])
        + r" \\",
        " & ".join(["", ""] + ["Acc.", r"$\Delta$"] * len(datasets)) + r" \\",
        r"\midrule",
    ]
    for backbone in BACKBONES:
        for method in ["untrained", *METHODS, "supervised_reg"]:
            cells = []
            for d in datasets:
                r = (
                    base.get((d, backbone, "bgrl"))
                    if method == "untrained"
                    else runs.get((d, backbone, method))
                )
                u = base.get((d, backbone, "bgrl"))
                if r is None:
                    cells += ["--", "--"]
                    continue
                acc = accuracies(r)
                cells.append(f"{acc.mean():.1f} $\\pm$ {acc.std(ddof=1):.1f}")
                if method == "untrained" or u is None:
                    cells.append("")
                else:
                    cells.append(f"{acc.mean() - accuracies(u).mean():+.1f}")
            name = (
                NAMES[method]
                if method not in {"untrained", "supervised_reg"}
                else rf"\textit{{{NAMES[method]}}}"
            )
            lines.append(" & ".join([backbone.upper(), name, *cells]) + r" \\")
        lines.append(r"\midrule" if backbone != BACKBONES[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_main.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'tab_main.tex'}")


def md_table(header: list[str], rows: list[list[str]], note: str = "") -> str:
    lines = [
        "<!-- Generated by benchmarks/paper_figures.py from the result files. Do not edit. -->",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    if note:
        lines += ["", note]
    return "\n".join(lines) + "\n"


def markdown_tables(analysis: Path, out: Path) -> None:
    """The study's results as Markdown tables, for docs/study.md (pymdownx.snippets)."""
    out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(analysis / "cells.csv")
    t = cells.pivot_table(index=["dataset", "backbone", "method"], columns="probe", values="acc")
    trained = t[t.index.get_level_values("method").isin(METHODS)].groupby(level=[0, 1]).mean()
    untr = t.xs("untrained", level="method")
    panels = [(d, b) for d in DATASETS for b in BACKBONES if (d, b) in trained.index]

    rows = []
    for probe, _features, reg in ABLATION:
        if probe not in trained.columns:
            continue
        cells_ = []
        for d, b in panels:
            a, u = trained.loc[(d, b), probe], untr.loc[(d, b), probe]
            cells_.append(f"{a:.1f} / {u:.1f} / **{a - u:+.1f}**")
        reg = reg.replace("$10^{-6}$", "1e-6").replace("$C$", "C")
        rows.append([PROBE_NAMES[probe], reg, *cells_])
    (out / "study_probes.md").write_text(
        md_table(
            ["Protocol", "Regularization", *[f"{d}, {b.upper()}" for d, b in panels]],
            rows,
            "Each cell: mean linear-probe test accuracy (%) of the seven SSL methods / of the "
            "untrained encoder / **gain**. Public split, 10 seeds.",
        )
    )

    gains = pd.read_csv(analysis / "gains_by_probe.csv")
    probes = [p for p, _, _ in ABLATION if p in gains.columns]
    rows = []
    for d, b in panels:
        g = gains[(gains.dataset == d) & (gains.backbone == b)].set_index("method")
        for method in METHODS:
            if method in g.index:
                rows.append(
                    [f"{d}, {b.upper()}", NAMES[method]]
                    + [f"{g.loc[method, p]:+.1f}" for p in probes]
                )
    (out / "study_gains.md").write_text(
        md_table(
            ["Data", "Method", *[PROBE_NAMES[p] for p in probes]],
            rows,
            "Gain over the untrained encoder of the same backbone (points), mean over 10 seeds.",
        )
    )

    v = pd.read_csv(analysis / "variance_shares.csv")
    v = v[v.design == "public split, linear probes"]
    factors = ["method", "backbone", "method x backbone", "probe", "method x probe"]
    rows = [
        [r["dataset"], "accuracy" if r["value"] == "acc" else "gain over untrained"]
        + [f"{100 * r[f]:.0f}%" for f in [*factors, "residual (seed)"]]
        for _, r in v.iterrows()
    ]
    (out / "study_variance.md").write_text(
        md_table(
            ["Data", "Quantity", *factors, "seeds (residual)"],
            rows,
            "Share of the variance (eta squared, balanced design) over the seven SSL methods, "
            "two backbones, the four linear-probe protocols of published code and 10 seeds, public "
            "split.",
        )
    )
    if (analysis / "variance_shares_ci.csv").exists():
        ci = pd.read_csv(analysis / "variance_shares_ci.csv")
        ci = ci[ci.design == "public split, linear probes"].set_index(
            ["dataset", "value", "component"]
        )
        rows = [
            [r["dataset"], "accuracy" if r["value"] == "acc" else "gain over untrained"]
            + [
                "{:.0f}% [{:.0f}, {:.0f}]".format(
                    *(100 * ci.loc[(r["dataset"], r["value"], f), ["share", "lo", "hi"]])
                )
                for f in [*factors, "residual (seed)"]
            ]
            for _, r in v.iterrows()
        ]
        (out / "study_variance_ci.md").write_text(
            md_table(
                ["Data", "Quantity", *factors, "seeds (residual)"],
                rows,
                "The shares of the table above with a 95% bootstrap interval: the 10 seeds of "
                "every method and backbone resampled, each seed's scores under the four probes "
                "kept together. The interval of a share near zero lies above it: read it as an "
                "upper bound.",
            )
        )

    tau = pd.read_csv(analysis / "kendall_tau.csv")

    def tau_cell(dataset: str, comparison: str, within: str) -> str:
        r = tau[(tau.dataset == dataset) & (tau.comparison == comparison) & (tau.within == within)]
        if r.empty:
            return "--"
        r = r.iloc[0]
        return f"{r.tau:.2f} [{r.tau_lo:.2f}, {r.tau_hi:.2f}]"

    cols = [
        ("gcn vs gin", "graphssl", "GCN vs GIN"),
        ("adam100 vs graphssl", "gcn", "100 steps vs converged, GCN"),
        ("adam100 vs graphssl", "gin", "100 steps vs converged, GIN"),
        ("knn vs graphssl", "gcn", "kNN vs converged, GCN"),
        ("knn vs graphssl", "gin", "kNN vs converged, GIN"),
    ]
    rows = [
        [d, *[tau_cell(d, c, w) for c, w, _ in cols]] for d in DATASETS if d in set(tau.dataset)
    ]
    (out / "study_ranking.md").write_text(
        md_table(
            ["Data", *[label for _, _, label in cols]],
            rows,
            "Kendall's tau between the orders of the seven methods, with a 95% seed-bootstrap "
            "interval. 1 = same order.",
        )
    )

    reruns = analysis / "reruns_summary.csv"
    if reruns.exists():
        r = pd.read_csv(reruns)
        r = r[r.metric == "linear"]
        rows = [
            [
                x.dataset,
                x.encoder.upper(),
                f"{x['mean |diff|']:.2f}",
                f"{x['max |diff|']:.1f}",
                f"{x['independent seeds |diff|']:.2f}",
                f"{x['same-seed corr']:.2f}",
                f"{x['max |mean diff|']:.2f}",
                f"{x['kendall tau of means']:.2f}",
            ]
            for _, x in r.iterrows()
        ]
        (out / "study_reruns.md").write_text(
            md_table(
                [
                    "Data",
                    "Backbone",
                    "Mean per-seed difference",
                    "Max per-seed difference",
                    "Two independent seeds (expected)",
                    "Same-seed correlation",
                    "Max difference of 10-seed means",
                    "Kendall tau of means",
                ],
                rows,
                "The seven SSL methods rerun with the same seeds and configuration on another "
                "machine (NVIDIA A2 with PyTorch 2.11 vs a laptop CPU with PyTorch 2.14); "
                "linear-probe test accuracy, absolute differences in points.",
            )
        )
    for md in sorted(out.glob("study_*.md")):
        md_to_tex(md, out / md.name.replace("study_", "tab_").replace(".md", ".tex"))
    print(f"wrote {out}/study_*.md and their tab_*.tex")


def md_to_tex(md: Path, tex: Path) -> None:
    """A generated Markdown table as LaTeX rows (booktabs), for paper_tmlr/body.tex."""

    def cell(text: str) -> str:
        text = text.strip()
        for a, b in (
            ("\\", r"\textbackslash{}"),
            ("%", r"\%"),
            ("_", r"\_"),
            ("&", r"\&"),
            ("#", r"\#"),
        ):
            text = text.replace(a, b)
        text = text.replace(" x ", r" $\times$ ").replace("tau", r"$\tau$")
        parts = text.split("**")
        return "".join(rf"\textbf{{{p}}}" if i % 2 else p for i, p in enumerate(parts))

    rows = [
        [cell(c) for c in line.strip().strip("|").split("|")]
        for line in md.read_text().splitlines()
        if line.startswith("|") and not line.startswith("|---")
    ]
    header, body = rows[0], rows[1:]
    n_text = next((i for i, c in enumerate(body[0]) if any(ch.isdigit() for ch in c)), 1)
    spec = "l" * n_text + "r" * (len(header) - n_text)
    lines = [
        f"% Generated by benchmarks/paper_figures.py from {md.name}. Do not edit.",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        *[" & ".join(r) + r" \\" for r in body],
        r"\bottomrule",
        r"\end{tabular}",
    ]
    tex.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# What training changes (analyze_representations.py): fig4 and three tables
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    "Cora",
    "CiteSeer",
    "PubMed",
    "AmazonComputers",
    "AmazonPhoto",
    "CoauthorCS",
    "ogbn-arxiv",
    "arxiv-year",
]
DATASET_NAMES = {
    "AmazonComputers": "Amazon Computers",
    "AmazonPhoto": "Amazon Photo",
    "CoauthorCS": "Coauthor CS",
}
ALL_BACKBONES = ["gcn", "gin", "transformer"]
BACKBONE_NAMES = {"gcn": "GCN", "gin": "GIN", "transformer": "Transformer"}
CANDIDATE_NAMES = {
    "eff_rank": "Effective rank (RankMe)",
    "distance": "Distance from untrained (1 - CKA)",
    "smoothing_gain": "Smoothing gained",
    "neg_uniformity": "Uniformity (negated)",
}
# Single-hue sequential ramp (blue, light to dark) for the CKA heatmap.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def fig4(analysis: Path, out: Path) -> None:
    """(a) Linear CKA between every pair of models on Cora, mean over seeds; (b) smoothing
    (cosine across edges minus cosine of random pairs) of the untrained encoder vs the trained
    methods, per dataset and backbone."""
    from matplotlib.colors import LinearSegmentedColormap

    cells = pd.read_csv(analysis / "metrics_cells.csv")
    matrix_file = analysis / "cka_matrix__Cora.csv"
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    if matrix_file.exists():
        m = pd.read_csv(matrix_file, index_col=0)
        order = [
            f"{b}/{meth}"
            for b in ALL_BACKBONES
            for meth in ["untrained", *METHODS]
            if f"{b}/{meth}" in m.index
        ]
        m = m.loc[order, order]
        cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
        im = ax_a.imshow(m.to_numpy(), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        labels = [NAMES[o.split("/")[1]] for o in order]
        ax_a.set_xticks(range(len(order)), labels, rotation=90, fontsize=6)
        ax_a.set_yticks(range(len(order)), labels, fontsize=6)
        ax_a.grid(False)
        # Surface-colored gaps between the backbone blocks, and the block names.
        start = 0
        for b in ALL_BACKBONES:
            n = sum(o.startswith(f"{b}/") for o in order)
            if n == 0:
                continue
            if start:
                ax_a.axhline(start - 0.5, color="white", lw=2)
                ax_a.axvline(start - 0.5, color="white", lw=2)
            ax_a.text(
                start + n / 2 - 0.5, -1.0, BACKBONE_NAMES[b], ha="center", va="bottom", color=INK_2
            )
            start += n
        cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.03)
        cbar.set_label("Linear CKA", color=INK_2)
        cbar.outline.set_visible(False)
        ax_a.set_title("(a) Cora: similarity between models", loc="left", color=INK, pad=14)

    datasets = [d for d in ALL_DATASETS if d in set(cells.dataset)]
    backbones = [b for b in ALL_BACKBONES if b in set(cells.backbone)]
    offsets = np.linspace(-0.22, 0.22, len(backbones)) if len(backbones) > 1 else [0.0]
    for k, b in enumerate(backbones):
        color = SERIES[k]
        for i, d in enumerate(datasets):
            g = cells[(cells.dataset == d) & (cells.backbone == b)]
            ssl = g[g.method.isin(METHODS)]["smoothing"]
            untr = g[g.method == "untrained"]["smoothing"]
            if ssl.empty or untr.empty:
                continue
            y = i + offsets[k]
            ax_b.plot([ssl.min(), ssl.max()], [y, y], color=color, lw=2, solid_capstyle="round")
            ax_b.plot(ssl.mean(), y, "o", ms=6, color=color, mec="white", mew=1)
            ax_b.plot(untr.mean(), y, "o", ms=6, mfc="white", mec=color, mew=1.5)
    ax_b.set_yticks(range(len(datasets)), [DATASET_NAMES.get(d, d) for d in datasets])
    ax_b.invert_yaxis()
    ax_b.axvline(0, color=MUTED, lw=0.8)
    ax_b.set_xlabel("Smoothing: cosine across edges minus cosine of random pairs")
    ax_b.set_title("(b) Smoothing, untrained vs trained", loc="left", color=INK)
    handles = [
        plt.Line2D(
            [], [], color=SERIES[k], lw=2, marker="o", ms=5, mec="white", label=BACKBONE_NAMES[b]
        )
        for k, b in enumerate(backbones)
    ]
    handles.append(
        plt.Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=INK_2, label="Untrained")
    )
    ax_b.legend(
        handles=handles, frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    fig.tight_layout()
    save(fig, out / "fig4_representations")


def representation_tables(analysis: Path, out: Path) -> None:
    """study_metrics.md, study_cka.md and study_predict.md (and their tab_*.tex)."""
    out.mkdir(parents=True, exist_ok=True)
    contribution = pd.read_csv(analysis / "graph_contribution.csv")
    cells = pd.read_csv(analysis / "metrics_cells.csv")
    c = contribution[contribution.probe == "graphssl"]
    rows = []
    for d in ALL_DATASETS:
        for b in ALL_BACKBONES:
            r = c[(c.dataset == d) & (c.backbone == b)]
            if r.empty:
                continue
            r = r.iloc[0]
            source = "ogbn-arxiv" if d == "arxiv-year" else d
            ssl = cells[
                (cells.dataset == source) & (cells.backbone == b) & cells.method.isin(METHODS)
            ]
            rows.append(
                [
                    DATASET_NAMES.get(d, d),
                    BACKBONE_NAMES[b],
                    f"{r.edge_homophily:.2f}",
                    f"{r.raw_features:.1f}",
                    f"{r.untrained:.1f}",
                    f"{r.ssl_mean:.1f} ({r.ssl_best:.1f})",
                    f"**{r.graph_adds:+.1f}**",
                    f"**{r.ssl_adds_mean:+.1f}**",
                    f"{r.smoothing_untrained:.2f} / {r.smoothing_ssl_mean:.2f}",
                    f"{ssl['distance'].mean():.2f}",
                ]
            )
    (out / "study_metrics.md").write_text(
        md_table(
            [
                "Data",
                "Backbone",
                "Homophily",
                "Raw features",
                "Untrained",
                "SSL mean (best)",
                "Graph adds",
                "Training adds",
                "Smoothing, untrained / trained",
                "Distance from untrained",
            ],
            rows,
            "Linear-probe test accuracy (%, GraphSSL probe, standard split): the input features "
            "alone, the untrained encoder, and the seven SSL methods (mean and best). Graph adds "
            "= untrained - raw features; training adds = SSL mean - untrained. Smoothing = cosine "
            "across edges minus cosine of random node pairs; distance = 1 - linear CKA to the "
            "untrained encoder of the same seed (mean over methods). Homophily: edge homophily "
            "of the labels. arxiv-year uses the ogbn-arxiv models.",
        )
    )

    grouping = pd.read_csv(analysis / "cka_grouping.csv")
    rows = []
    for d in ALL_DATASETS:
        g = grouping[grouping.dataset == d].set_index("pair")
        if g.empty:
            continue
        rows.append(
            [DATASET_NAMES.get(d, d)]
            + [
                f"{g.loc[k, 'cka']:.2f} [{g.loc[k, 'lo']:.2f}, {g.loc[k, 'hi']:.2f}]"
                if k in g.index
                else "-"
                for k in ("same backbone", "same method", "neither")
            ]
        )
    (out / "study_cka.md").write_text(
        md_table(
            ["Data", "Same backbone", "Same method", "Neither"],
            rows,
            "Mean linear CKA between two trained SSL models that share the backbone (different "
            "methods), the method (different backbones), or neither, with a 95% bootstrap "
            "interval over seeds. 1 = the same representation up to rotation and scale.",
        )
    )

    summary = pd.read_csv(analysis / "prediction.csv")
    s = summary[summary.target == "graphssl"]
    sets = [x for x in ("development", "test", "arxiv-year") if x in set(s.set)]
    rows = []
    for cand, name in CANDIDATE_NAMES.items():
        row = [name]
        for x in sets:
            r = s[(s.set == x) & (s.candidate == cand)]
            if r.empty:
                row += ["-", "-"]
                continue
            r = r.iloc[0]
            row += [
                f"{r.mean_rho:+.2f} [{r.rho_lo:+.2f}, {r.rho_hi:+.2f}] ({r.groups_rho_positive}/{r.n_groups})",
                f"{r.mean_regret:.1f} / {r.mean_random_regret:.1f}",
            ]
        rows.append(row)
    header = ["Candidate"]
    for x in sets:
        header += [f"{x.capitalize()}: correlation", f"{x.capitalize()}: regret / random"]
    (out / "study_predict.md").write_text(
        md_table(
            header,
            rows,
            "Does a label-free metric of the embeddings rank the seven methods by their gain "
            "over the untrained encoder? Spearman correlation within each dataset and backbone, "
            "averaged (95% seed-bootstrap interval; groups with a positive correlation / all); "
            "regret = points of gain lost by picking the method the metric ranks first, against "
            "the expected loss of a random pick. The test was fixed before any metric or gain "
            "of a test dataset was computed (benchmarks/paper/PLAN.md).",
        )
    )
    for name in ("study_metrics.md", "study_cka.md", "study_predict.md"):
        md_to_tex(out / name, out / name.replace("study_", "tab_").replace(".md", ".tex"))
    print(f"wrote {out}/study_metrics.md, study_cka.md, study_predict.md and their tab_*.tex")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    p.add_argument("--untrained", type=Path, default=Path("benchmarks/ablations/untrained_encoder"))
    p.add_argument("--analysis", type=Path, default=Path("benchmarks/paper/analysis"))
    p.add_argument("--out", type=Path, default=Path("benchmarks/paper/figures"))
    args = p.parse_args()
    fig1(args.results, args.untrained, args.out)
    table_main(args.results, args.untrained, args.out)
    if (args.analysis / "cells.csv").exists():
        fig2(args.analysis, args.out)
        table_probe_ablation(args.analysis, args.out)
    if (args.analysis / "variance_shares.csv").exists():
        fig3(args.analysis, args.out)
        markdown_tables(args.analysis, args.out)
    if (args.analysis / "graph_contribution.csv").exists():
        fig4(args.analysis, args.out)
        representation_tables(args.analysis, args.out)


if __name__ == "__main__":
    main()
