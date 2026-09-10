"""Render Markdown/LaTeX benchmark tables from the JSON files run_benchmark.py saves.

Reads every ``benchmarks/results/<Dataset>/<model>__<timestamp>.json``, keeps
only the latest run per (dataset, model) pair, and prints one table per
dataset — paste straight into README.md / paper.tex instead of retyping
numbers by hand (the exact failure mode this script exists to avoid: today
the same benchmark numbers are copied manually into README.md, CLAUDE.md and
paper.tex, and they *will* eventually drift out of sync).

Usage::

    python benchmarks/render_tables.py                  # Markdown, all datasets found
    python benchmarks/render_tables.py --latex           # also print LaTeX
    python benchmarks/render_tables.py --dataset Cora    # filter to one dataset
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_DISPLAY_NAMES = {
    "dgi": "DGI",
    "graphcl": "GraphCL",
    "vicreg": "VICReg",
    "barlow_twins": "Barlow Twins",
    "bgrl": "BGRL",
    "afgrl": "AFGRL",
    "graphdino": "GraphDINO",
    "supervised": "Supervised*",
}
# Fixed display order (paradigm-grouped) rather than alphabetical/discovery order.
MODEL_ORDER = [
    "dgi",
    "graphcl",
    "bgrl",
    "afgrl",
    "vicreg",
    "barlow_twins",
    "graphdino",
    "supervised",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-dir", default="benchmarks/results")
    p.add_argument("--dataset", nargs="+", default=None, help="filter to specific datasets")
    p.add_argument("--latex", action="store_true", help="also print a LaTeX booktabs table")
    return p.parse_args()


def load_latest_results(results_dir: Path, datasets: list[str] | None) -> dict:
    """Returns {dataset: {model: result_dict}}, keeping only the latest file per pair."""
    latest: dict[str, dict[str, tuple[str, dict]]] = defaultdict(dict)
    for path in sorted(results_dir.glob("*/*.json")):
        dataset = path.parent.name
        if datasets and dataset not in datasets:
            continue
        data = json.loads(path.read_text())
        model = data["model"]
        timestamp = data["provenance"]["timestamp_utc"]
        # Filenames sort chronologically (UTC timestamp in the name), so the
        # last one seen per (dataset, model) via sorted glob is the latest —
        # but compare timestamps explicitly rather than relying on glob order.
        if model not in latest[dataset] or timestamp > latest[dataset][model][0]:
            latest[dataset][model] = (timestamp, data)
    return {ds: {m: d for m, (_, d) in models.items()} for ds, models in latest.items()}


def fmt_pct(agg: dict) -> str:
    return f"{agg['mean'] * 100:.2f} ± {agg['std'] * 100:.2f}"


def render_markdown(dataset: str, models: dict) -> str:
    lines = [
        f"### {dataset}",
        "",
        "| Model | Linear probe (test) | KNN k={k} (test) | Seeds | Epochs |".format(
            k=next(iter(models.values()))["hyperparameters"]["knn_k"]
        ),
        "|---|---|---|---|---|",
    ]
    for model in MODEL_ORDER:
        if model not in models:
            continue
        d = models[model]
        n_seeds = len(d["seeds"])
        epochs = d["hyperparameters"]["epochs"]
        lines.append(
            f"| {MODEL_DISPLAY_NAMES[model]} | {fmt_pct(d['aggregate']['test_acc_linear'])} | "
            f"{fmt_pct(d['aggregate']['test_acc_knn'])} | {n_seeds} | {epochs} |"
        )
    if "supervised" in models:
        lines.append("")
        lines.append(
            "*Supervised uses train-split labels during pretraining (not an SSL method) — "
            "included as a reference point, not a like-for-like comparison.*"
        )
    return "\n".join(lines)


def render_latex(dataset: str, models: dict) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Test Accuracy (\%) on {dataset}}}",
        rf"\label{{tab:{dataset.lower()}-comparison}}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Linear Probe} & \textbf{$k$NN} \\ \midrule",
    ]
    for model in MODEL_ORDER:
        if model not in models:
            continue
        d = models[model]
        lin, knn = d["aggregate"]["test_acc_linear"], d["aggregate"]["test_acc_knn"]
        lines.append(
            rf"{MODEL_DISPLAY_NAMES[model]} & "
            rf"{lin['mean'] * 100:.2f} $\pm$ {lin['std'] * 100:.2f} & "
            rf"{knn['mean'] * 100:.2f} $\pm$ {knn['std'] * 100:.2f} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results_dir = REPO_ROOT / args.results_dir
    if not results_dir.exists():
        print(f"No results found at {results_dir} — run benchmarks/run_benchmark.py first.")
        return

    by_dataset = load_latest_results(results_dir, args.dataset)
    if not by_dataset:
        print(f"No result JSON files found under {results_dir}.")
        return

    for dataset in sorted(by_dataset):
        print(render_markdown(dataset, by_dataset[dataset]))
        print()
        if args.latex:
            print(render_latex(dataset, by_dataset[dataset]))
            print()


if __name__ == "__main__":
    main()
