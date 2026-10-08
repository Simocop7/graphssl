"""Render Markdown/LaTeX benchmark tables from the JSON files run_benchmark.py saves.

Reads every ``benchmarks/results/<Dataset>/<model>__<timestamp>.json``, keeps
only the latest run per (dataset, encoder, model), and prints one table per
dataset and encoder — paste straight into README.md / paper.tex instead of retyping
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
    "supervised_reg": "Supervised, std. recipe*",
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
    "supervised_reg",
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
    """Returns {(dataset, encoder): {model: result_dict}}, keeping the latest file of each.

    The encoder is part of the key: a run with another backbone is another table, not a
    newer version of the same rows.
    """
    latest: dict[tuple[str, str], dict[str, tuple[str, dict]]] = defaultdict(dict)
    for path in sorted(results_dir.glob("*/*.json")):
        dataset = path.parent.name
        if datasets and dataset not in datasets:
            continue
        data = json.loads(path.read_text())
        model = data["model"]
        key = (dataset, data["hyperparameters"]["encoder"])
        timestamp = data["provenance"]["timestamp_utc"]
        # Filenames sort chronologically (UTC timestamp in the name), so the
        # last one seen per key via sorted glob is the latest —
        # but compare timestamps explicitly rather than relying on glob order.
        if model not in latest[key] or timestamp > latest[key][model][0]:
            latest[key][model] = (timestamp, data)
    return {key: {m: d for m, (_, d) in models.items()} for key, models in latest.items()}


def fmt_pct(agg: dict) -> str:
    return f"{agg['mean'] * 100:.2f} ± {agg['std'] * 100:.2f}"


def describe_probe(settings: dict | None) -> str:
    """The linear-probe protocol of a result, from its ``hyperparameters.linear_probe``."""
    if settings is None:  # saved before the probe's settings were recorded
        return "0.1.0 probe (100 Adam steps on raw features, no regularisation)"
    features = "standardized" if settings["standardize"] else "raw"
    grid = ", ".join(f"{v:g}" for v in settings["weight_decays"])
    return (
        f"L2-regularised logistic regression on {features} features, fitted with L-BFGS "
        f"(up to {settings['max_iter']} iterations), L2 strength selected on validation "
        f"among {{{grid}}}"
    )


def probe_note(results: dict[str, dict]) -> str:
    """One line on the linear probe behind a table's rows ({row label: result}).

    Rows evaluated with different protocols are not comparable in the linear column: the
    note then lists which rows used which, instead of presenting them as one table.
    """
    by_protocol: dict[str, list[str]] = defaultdict(list)
    unconverged = total = 0
    for label, d in results.items():
        by_protocol[describe_probe(d["hyperparameters"].get("linear_probe"))].append(label)
        flags = [r["probe_converged"] for r in d["per_seed"] if "probe_converged" in r]
        unconverged += flags.count(False)
        total += len(flags)
    if len(by_protocol) == 1:
        note = f"*Linear probe: {next(iter(by_protocol))}.*"
    else:
        parts = "; ".join(f"{', '.join(rows)}: {desc}" for desc, rows in by_protocol.items())
        note = (
            "**Mixed linear-probe protocols: the linear column is not comparable across "
            f"these rows.** {parts}."
        )
    if unconverged:
        note += f" *{unconverged} of {total} probe fits did not converge.*"
    return note


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
    lines.append("")
    lines.append(
        probe_note({MODEL_DISPLAY_NAMES[m]: models[m] for m in MODEL_ORDER if m in models})
    )
    if "supervised" in models or "supervised_reg" in models:
        lines.append("")
        lines.append(
            "*Supervised uses train-split labels during pretraining (not an SSL method) — "
            "included as a reference point, not a like-for-like comparison. "
            "'Supervised' keeps the shared protocol (no dropout, last checkpoint); "
            "'std. recipe' uses dropout 0.5, Adam lr 0.01 + L2 5e-4 and the "
            "best-validation checkpoint (Kipf & Welling, 2017).*"
        )
    return "\n".join(lines)


def render_latex(dataset: str, models: dict) -> str:
    slug = "-".join(dataset.lower().replace("(", "").replace(")", "").split())
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Test Accuracy (\%) on {dataset}}}",
        rf"\label{{tab:{slug}-comparison}}",
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
    note = probe_note({MODEL_DISPLAY_NAMES[m]: models[m] for m in MODEL_ORDER if m in models})
    lines.append("% " + note.replace("*", ""))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results_dir = REPO_ROOT / args.results_dir
    if not results_dir.exists():
        print(f"No results found at {results_dir} — run benchmarks/run_benchmark.py first.")
        return

    tables = load_latest_results(results_dir, args.dataset)
    if not tables:
        print(f"No result JSON files found under {results_dir}.")
        return

    encoders_per_dataset = defaultdict(set)
    for dataset, encoder in tables:
        encoders_per_dataset[dataset].add(encoder)
    for dataset, encoder in sorted(tables):
        # Name the encoder only when a dataset has results with more than one.
        several = len(encoders_per_dataset[dataset]) > 1
        title = f"{dataset} ({encoder.upper()} encoder)" if several else dataset
        print(render_markdown(title, tables[(dataset, encoder)]))
        print()
        if args.latex:
            print(render_latex(title, tables[(dataset, encoder)]))
            print()


if __name__ == "__main__":
    main()
