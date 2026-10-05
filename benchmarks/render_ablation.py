"""Render ablation tables from run_benchmark.py JSONs saved outside benchmarks/results/.

Unlike render_tables.py, which reports the latest run per (dataset, model), this keeps
every configuration: one row per (model, encoder, training steps, starting EMA momentum,
``--set`` overrides),
one table per dataset, with the effective rank of the embeddings and, for teacher-student
models, the numbers of the other encoder (target for BGRL/AFGRL, student for GraphDINO).
When a configuration was run more than once, the latest run is shown.

Usage::

    python benchmarks/render_ablation.py benchmarks/ablations/teacher_student
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HEADER = (
    "| Model | Encoder | Steps | EMA τ start | Overrides | Linear | kNN | Rank "
    "| Alt. encoder | Alt. linear | Alt. rank | Seeds |"
)


def overrides_label(result: dict) -> str:
    """The run's --set overrides as ``key=value`` pairs, or an empty string."""
    overrides = result["hyperparameters"].get("overrides", {})
    return ", ".join(f"{k}={json.dumps(v)}" for k, v in sorted(overrides.items()))


def ema_start(result: dict) -> float | None:
    """Starting EMA momentum of the teacher, or None for models without one."""
    cfg = result.get("model_config", {})
    if result["model"] == "graphdino":
        return cfg.get("ema_tau_base")
    if result["model"] in ("bgrl", "afgrl"):
        return cfg.get("ema_tau")
    return None


def fmt_pct(agg: dict | None) -> str:
    return "—" if agg is None else f"{agg['mean'] * 100:.2f} ± {agg['std'] * 100:.2f}"


def fmt_rank(agg: dict | None) -> str:
    return "—" if agg is None else f"{agg['mean']:.1f} ± {agg['std']:.1f}"


def load(results_dir: Path) -> dict:
    """{dataset: {config key: result}}, keeping the latest run per configuration."""
    latest: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for path in sorted(results_dir.glob("*/*.json")):
        d = json.loads(path.read_text())
        hp = d["hyperparameters"]
        key = (d["model"], hp["encoder"], hp["epochs"], ema_start(d), overrides_label(d))
        prev = latest[d["dataset"]].get(key)
        if prev is None or d["provenance"]["timestamp_utc"] > prev["provenance"]["timestamp_utc"]:
            latest[d["dataset"]][key] = d
    return latest


def render(dataset: str, configs: dict) -> str:
    n_columns = HEADER.count("|") - 1
    lines = [f"### {dataset}", "", HEADER, "|" + "---|" * n_columns]
    for key in sorted(
        configs, key=lambda k: (k[0], k[1], k[2], -1 if k[3] is None else k[3], k[4])
    ):
        model, encoder, steps, tau, overrides = key
        d = configs[key]
        agg = d["aggregate"]
        lines.append(
            f"| {model} | {encoder} | {steps} | {'—' if tau is None else tau} "
            f"| {overrides or '—'} "
            f"| {fmt_pct(agg.get('test_acc_linear'))} | {fmt_pct(agg.get('test_acc_knn'))} "
            f"| {fmt_rank(agg.get('eff_rank'))} | {d.get('alt_encoder', '—')} "
            f"| {fmt_pct(agg.get('test_acc_linear_alt'))} | {fmt_rank(agg.get('eff_rank_alt'))} "
            f"| {len(d['seeds'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("results_dir", help="directory passed as --out-dir to run_benchmark.py")
    args = p.parse_args()

    results = load(Path(args.results_dir))
    if not results:
        print(f"No result JSON files found under {args.results_dir}.")
        return
    for dataset in sorted(results):
        print(render(dataset, results[dataset]))
        print()


if __name__ == "__main__":
    main()
