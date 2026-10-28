"""E4: training trajectories, read with the questions and definitions fixed in PLAN.md
("E4 in detail") before the runs.

Reads every ``run_benchmark.py --trajectory`` JSON under ``ROOT/benchmarks/paper/e4`` (one
directory per optimizer cell) and writes to ``--out``:

summary.md         Q1 ranking stability, Q2 shape of each trajectory, Q3 collapse and when,
                   Q4 the optimizer cells side by side.
early_stop_tau.csv Q1's early-stopping tau per cell, dataset and backbone, with its interval.
shapes.csv         Q2/Q3 per cell, dataset, backbone and method.
traj_<cell>.png    test accuracy, effective rank and pair_cos against the step, per dataset
                   and backbone (seed means; one line per method, step 0 = untrained).

Usage::

    python benchmarks/analyze_trajectories.py --root ../graphssl-runs-e4 \\
        --out benchmarks/paper/analysis/trajectories
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from paper_figures import METHODS_V2, NAMES, SERIES, plt
from scipy.stats import kendalltau

CELLS = ["adamw_lr5e-4", "adamw_lr1e-2", "sgd_lr1e-2"]
CELL_NAMES = {"adamw_lr5e-4": "AdamW 5e-4", "adamw_lr1e-2": "AdamW 1e-2", "sgd_lr1e-2": "SGD 1e-2"}
DATASETS = ["Cora", "CiteSeer"]
BACKBONES = ["gcn", "gin", "transformer"]
METRICS = ["test_acc_linear", "val_acc_linear", "test_acc_knn", "eff_rank", "pair_cos"]
# Fixed in PLAN.md before the runs.
DECLINE_POINTS = 1.0
LEARN_POINTS = 1.0
RANK_COLLAPSE = 10.0
PAIR_COS_COLLAPSE = 0.95
BOOTSTRAP = 2000
# Means over a diverged run's NaN seeds are expected: they mark the divergence.
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)


# The objectives with a run in every (cell, dataset, backbone) of E4, set in main(): a method
# whose runs are incomplete is left out and named, never a whole group dropped in silence.
METHODS: list[str] = []


def complete_methods(runs: dict) -> list[str]:
    keys = {(c, d, b) for c, d, b, _ in runs}
    return [m for m in METHODS_V2 if all((c, d, b, m) in runs for c, d, b in keys)]


def load(root: Path) -> dict:
    """{(cell, dataset, backbone, method): {"steps": [...], metric: array [seeds, steps]}}."""
    runs: dict = {}
    for path in sorted((root / "benchmarks" / "paper" / "e4").glob("*/*/*.json")):
        r = json.loads(path.read_text())
        key = (path.parts[-3], r["dataset"], r["hyperparameters"]["encoder"], r["model"])
        steps = r["hyperparameters"]["trajectory_steps"]
        run = {"steps": steps}
        for m in METRICS:
            run[m] = np.array(
                [
                    [np.nan if p.get(m) is None else p[m] for p in s["trajectory"]]
                    for s in r["per_seed"]
                ],
                dtype=float,
            )
        runs[key] = run  # the newest file of a key wins (sorted names end in a timestamp)
    return runs


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple:
    """Mean and 95% interval of the mean over seeds (resampled)."""
    idx = rng.integers(len(values), size=(BOOTSTRAP, len(values)))
    means = np.nanmean(values[idx], axis=1)
    return float(np.nanmean(values)), *np.nanpercentile(means, [2.5, 97.5])


def first_step(steps: list, mask: np.ndarray) -> int | None:
    hit = np.flatnonzero(mask)
    return steps[hit[0]] if len(hit) else None


def shape(run: dict, rng: np.random.Generator) -> dict:
    """Q2 and Q3 for one (cell, dataset, backbone, method)."""
    steps = run["steps"]
    test, val = 100 * run["test_acc_linear"], 100 * run["val_acc_linear"]
    diverged = np.isnan(test).any(axis=0)
    mean_val = np.where(diverged, -np.inf, np.nanmean(val, axis=0))
    best = int(np.argmax(mean_val))
    decline = bootstrap_ci(test[:, best] - test[:, -1], rng)
    mean_test = np.nanmean(test, axis=0)
    if np.isfinite(test[:, -1]).all() and decline[0] >= DECLINE_POINTS and decline[1] > 0:
        kind = "late decline"
    elif np.nanmax(mean_test) < mean_test[0] + LEARN_POINTS:
        kind = "never learns"
    else:
        kind = "rises and holds"
    if diverged[-1]:
        kind = "diverges"
    rank, pair = np.nanmean(run["eff_rank"], axis=0), np.nanmean(run["pair_cos"], axis=0)
    return {
        "kind": kind,
        "best_val_step": steps[best],
        "test_at_best": float(np.nanmean(test[:, best])),
        "test_at_end": float(mean_test[-1]),
        "test_at_0": float(mean_test[0]),
        "decline": decline,
        "diverged_at": first_step(steps, diverged),
        "rank_0": float(rank[0]),
        "rank_end": float(rank[-1]),
        "rank_collapse_at": first_step(steps, rank < RANK_COLLAPSE),
        "pair_cos_0": float(pair[0]),
        "pair_cos_end": float(pair[-1]),
        "pair_collapse_at": first_step(steps, pair > PAIR_COS_COLLAPSE),
    }


def tau(a: list, b: list) -> float:
    """Kendall tau of two rankings; a diverged method (NaN accuracy) ranks last."""
    return float(kendalltau(np.nan_to_num(a, nan=-1.0), np.nan_to_num(b, nan=-1.0)).statistic)


def ranking_stability(runs: dict, cell: str, dataset: str, backbone: str, rng) -> dict | None:
    """Q1: tau of the ranking at every step and with early stopping, against step 300."""
    group = [runs.get((cell, dataset, backbone, m)) for m in METHODS]
    if any(g is None for g in group):
        return None
    steps = group[0]["steps"]

    def curves(sample: bool) -> tuple:
        tests, stopped = [], []
        for g in group:
            test, val = g["test_acc_linear"], g["val_acc_linear"]
            if sample:
                idx = rng.integers(len(test), size=len(test))
                test, val = test[idx], val[idx]
            mean_test = np.nanmean(test, axis=0)
            with np.errstate(invalid="ignore"):
                mean_val = np.where(np.isnan(test).any(axis=0), -np.inf, np.nanmean(val, axis=0))
            tests.append(mean_test)
            stopped.append(mean_test[int(np.argmax(mean_val))])
        return np.array(tests), stopped

    tests, stopped = curves(False)
    end = list(tests[:, -1])
    per_step = [tau(list(tests[:, i]), end) for i in range(len(steps))]
    boot = []
    for _ in range(BOOTSTRAP // 4):
        t, s = curves(True)
        boot.append(tau(s, list(t[:, -1])))
    return {
        "steps": steps,
        "tau_per_step": per_step,
        "tau_early_stop": tau(stopped, end),
        "tau_early_stop_ci": tuple(np.nanpercentile(boot, [2.5, 97.5])),
    }


def fmt_step(s) -> str:
    return "-" if s is None else str(s)


def write_summary(runs: dict, out: Path, rng: np.random.Generator) -> None:
    lines = [
        "# E4: training trajectories",
        "",
        "Generated by `benchmarks/analyze_trajectories.py`; definitions in "
        "`benchmarks/paper/PLAN.md` (E4 in detail). GraphSSL probe, test accuracy in %, seed "
        "means; [95% seed-bootstrap interval]. Step 0 is the untrained encoder.",
        "",
        "## Q1. Is the ranking at step 300 the ranking elsewhere?",
        "",
        "Kendall tau between the method ranking at a step and at step 300; last column: every "
        "method stopped at its best-validation step.",
        "",
    ]
    taus, shapes = [], []
    for cell in CELLS:
        rows = []
        for d in DATASETS:
            for b in BACKBONES:
                st = ranking_stability(runs, cell, d, b, rng)
                if st is None:
                    continue
                taus.append(
                    {"cell": cell, "dataset": d, "backbone": b, "tau": st["tau_early_stop"],
                     "tau_lo": st["tau_early_stop_ci"][0], "tau_hi": st["tau_early_stop_ci"][1]}
                )  # fmt: skip
                if not rows:
                    header = [str(s) for s in st["steps"]]
                    rows.append(f"| {CELL_NAMES[cell]} | " + " | ".join(header) + " | early stop |")
                    rows.append("|---" * (len(header) + 2) + "|")
                lo, hi = st["tau_early_stop_ci"]
                rows.append(
                    f"| {d} {b} | "
                    + " | ".join(f"{t:+.2f}" for t in st["tau_per_step"])
                    + f" | {st['tau_early_stop']:+.2f} [{lo:+.2f}, {hi:+.2f}] |"
                )
        lines += [*rows, ""] if rows else []
    lines += [
        "## Q2–Q4. Shape, collapse and optimizer, per method",
        "",
        "Best val. = step with the best mean validation accuracy; decline = test accuracy there "
        "minus at step 300. Rank and pair_cos at step 0 → 300; 'at' = first step below rank "
        f"{RANK_COLLAPSE:g} / above pair_cos {PAIR_COS_COLLAPSE:g}.",
        "",
    ]
    for d in DATASETS:
        for b in BACKBONES:
            rows = []
            for m in METHODS:
                for cell in CELLS:
                    run = runs.get((cell, d, b, m))
                    if run is None:
                        continue
                    s = shape(run, rng)
                    mean, lo, hi = s["decline"]
                    shapes.append(
                        {"cell": cell, "dataset": d, "backbone": b, "method": m,
                         "kind": s["kind"], "best_val_step": s["best_val_step"],
                         "test_at_0": s["test_at_0"], "test_at_best": s["test_at_best"],
                         "test_at_end": s["test_at_end"], "decline": mean,
                         "rank_collapse_at": s["rank_collapse_at"],
                         "pair_collapse_at": s["pair_collapse_at"],
                         "diverged_at": s["diverged_at"]}
                    )  # fmt: skip
                    rows.append(
                        f"| {NAMES[m]} | {CELL_NAMES[cell]} | {s['kind']} | {s['test_at_0']:.1f} | "
                        f"{s['test_at_best']:.1f} @ {s['best_val_step']} | {s['test_at_end']:.1f} | "
                        f"{mean:+.1f} [{lo:+.1f}, {hi:+.1f}] | {s['rank_0']:.0f} → "
                        f"{s['rank_end']:.0f} (at {fmt_step(s['rank_collapse_at'])}) | "
                        f"{s['pair_cos_0']:.2f} → {s['pair_cos_end']:.2f} (at "
                        f"{fmt_step(s['pair_collapse_at'])}) | {fmt_step(s['diverged_at'])} |"
                    )
            if rows:
                lines += [
                    f"### {d}, {b}",
                    "",
                    "| Method | Cell | Shape | Step 0 | Best val. | Step 300 | Decline | Rank | "
                    "pair_cos | Diverged at |",
                    "|---|---|---|---|---|---|---|---|---|---|",
                    *rows,
                    "",
                ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    for name, table in (("early_stop_tau.csv", taus), ("shapes.csv", shapes)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)


def plot(runs: dict, out: Path) -> None:
    panels = [("test_acc_linear", "Linear probe, test %", 100), ("eff_rank", "Effective rank", 1),
              ("pair_cos", "pair_cos", 1)]  # fmt: skip
    for cell in CELLS:
        groups = [(d, b) for d in DATASETS for b in BACKBONES
                  if any((cell, d, b, m) in runs for m in METHODS)]  # fmt: skip
        if not groups:
            continue
        fig, axes = plt.subplots(
            len(panels), len(groups), figsize=(2.1 * len(groups), 5.4), squeeze=False
        )
        for j, (d, b) in enumerate(groups):
            for i, (metric, label, scale) in enumerate(panels):
                ax = axes[i][j]
                for m in METHODS:
                    run = runs.get((cell, d, b, m))
                    if run is None:
                        continue
                    x = np.array(run["steps"])
                    with np.errstate(invalid="ignore"):
                        y = scale * np.nanmean(run[metric], axis=0)
                    ax.plot(x + 1, y, color=SERIES[METHODS_V2.index(m)], lw=1.2, label=NAMES[m])
                ax.set_xscale("log")
                ax.set_xticks([1, 11, 101, 301], ["0", "10", "100", "300"])
                if i == 0:
                    ax.set_title(f"{d}, {b}", fontsize=8)
                if j == 0:
                    ax.set_ylabel(label)
                if i == len(panels) - 1:
                    ax.set_xlabel("step")
        axes[0][0].legend(fontsize=6, frameon=False, loc="lower right")
        fig.suptitle(CELL_NAMES[cell], fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"traj_{cell}.png", dpi=160, metadata={"Software": None})
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", default=".", help="tree whose benchmarks/paper/e4 holds the runs")
    p.add_argument("--out", default="benchmarks/paper/analysis/trajectories")
    args = p.parse_args()
    runs = load(Path(args.root))
    METHODS[:] = complete_methods(runs)
    left_out = sorted({k[3] for k in runs} - set(METHODS))
    print(
        f"objectives: {', '.join(METHODS)}"
        + (f"; incomplete, left out: {left_out}" if left_out else "")
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    write_summary(runs, out, rng)
    plot(runs, out)
    print(f"{len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
