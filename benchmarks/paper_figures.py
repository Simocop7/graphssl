"""Figures of the factor-study paper, rendered from result files only (never retyped).

fig1  Linear-probe test accuracy of each method (mean and 95% bootstrap interval over seeds)
      per dataset and backbone, against the untrained encoder of the same backbone (band) and
      the standard supervised recipe (line). From run_benchmark.py JSONs.
fig2  Gain over the untrained encoder under each probe protocol, per method, dataset and
      backbone. From analyze_factors.py's cells.csv.
fig3  Share of variance explained by each factor. From analyze_factors.py's variance_shares.csv.

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
# Validated categorical palette (light surface), fixed order; ink and chrome colours.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
PROBES = ["adam100", "adam5000", "liblinear", "graphssl", "knn"]
PROBE_NAMES = {
    "adam100": "100 Adam steps (DGI, PyG-SSL)",
    "adam5000": "5,000 Adam steps (PyGCL)",
    "liblinear": "liblinear, L2-normalised (BGRL)",
    "graphssl": "converged L-BFGS (GraphSSL)",
    "knn": "kNN, k = 5",
}

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
    factor_cols = [c for c in v.columns if c not in {"dataset", "design", "value"}]
    v = v.dropna(axis=1, how="all")
    factor_cols = [c for c in factor_cols if c in v.columns]
    labels = [f"{r.dataset} · {r.design} · {r.value}" for r in v.itertuples()]
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
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


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
    if (args.analysis / "cells.csv").exists():
        fig2(args.analysis, args.out)
    if (args.analysis / "variance_shares.csv").exists():
        fig3(args.analysis, args.out)


if __name__ == "__main__":
    main()
