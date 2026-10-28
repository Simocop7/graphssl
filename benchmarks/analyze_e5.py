"""E5: does a ten times longer budget change the gain over the untrained encoder?

Reads the ``run_benchmark.py`` JSONs of ``ROOT/benchmarks/paper/e5/steps300`` and ``steps3000``
and applies the rule fixed in PLAN.md ("GraphMAE and E5") before any E5 run:

- per (dataset, backbone, objective), regularized probe, standard split, gains over the
  untrained encoder of the same seed: G300 (the 300-step run), G3000 (the 3,000-step run at its
  end) and Gstop (the 3,000-step run at the evaluation step, among 300 ... 3,000, with the best
  mean validation accuracy over seeds);
- Δ = Gstop − G300, paired by seed; the mean of Δ over objectives with a 95% seed-bootstrap
  interval (seeds resampled within each objective, the stopping step chosen again in every
  replicate);
- verdict for claim C3 on Amazon Computers, Amazon Photo and Coauthor CS: "budget-dependent" if
  on any of them Δ ≥ 1.0 point with an interval above 0, otherwise "stands".

The untrained encoder of a seed is step 0 of that seed's 3,000-step run (same device as the
300-step run; every objective starts from the same encoder for a seed). Writes ``summary.md``
and ``e5.csv`` to ``--out``.

Usage::

    python benchmarks/analyze_e5.py --out benchmarks/paper/analysis/e5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from paper_figures import METHODS_V2, NAMES

BUDGETS = (300, 3000)
CLAIM_DATASETS = ["AmazonComputers", "AmazonPhoto", "CoauthorCS"]
THRESHOLD = 1.0  # points, fixed in PLAN.md
BOOTSTRAP = 5000


def load(root: Path) -> dict:
    """{(steps, dataset, backbone, method): {seed: per-seed record}}, newest file per key."""
    runs: dict = {}
    for steps in BUDGETS:
        for path in sorted(
            (root / "benchmarks" / "paper" / "e5" / f"steps{steps}").glob("*/*.json")
        ):
            r = json.loads(path.read_text())
            key = (steps, r["dataset"], r["hyperparameters"]["encoder"], r["model"])
            runs[key] = {s["seed"]: s for s in r["per_seed"]}  # names end in a timestamp
    return runs


def arrays(runs: dict, dataset: str, backbone: str, method: str) -> dict | None:
    """Per-seed accuracies (%) on the seeds both budgets have: untrained, 300-step test, and the
    3,000-step run's validation / test at each evaluated step >= 300."""
    short, long = (
        runs.get((300, dataset, backbone, method)),
        runs.get((3000, dataset, backbone, method)),
    )
    if short is None or long is None:
        return None
    seeds = sorted(set(short) & set(long))
    points = [{p["step"]: p for p in long[s]["trajectory"]} for s in seeds]
    steps = sorted(st for st in points[0] if st >= 300)
    return {
        "seeds": seeds,
        "steps": steps,
        "untrained": np.array([100 * p[0]["test_acc_linear"] for p in points]),
        "g300_test": np.array([100 * short[s]["test_acc_linear"] for s in seeds]),
        "val": np.array([[100 * p[st]["val_acc_linear"] for st in steps] for p in points]),
        "test": np.array([[100 * p[st]["test_acc_linear"] for st in steps] for p in points]),
    }


def gains(a: dict, idx: np.ndarray | None = None) -> tuple[float, float, float, int]:
    """(G300, G3000, Gstop, stopping step) on the seeds ``idx`` (all when None)."""
    i = np.arange(len(a["seeds"])) if idx is None else idx
    untrained = a["untrained"][i]
    stop = int(np.argmax(a["val"][i].mean(axis=0)))
    return (
        float((a["g300_test"][i] - untrained).mean()),
        float((a["test"][i, -1] - untrained).mean()),
        float((a["test"][i, stop] - untrained).mean()),
        a["steps"][stop],
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, default=Path("benchmarks/paper/analysis/e5"))
    args = p.parse_args()
    runs = load(args.root)
    rng = np.random.default_rng(0)
    rows, groups = [], {}
    for dataset, backbone in sorted({(k[1], k[2]) for k in runs}):
        per_method = {}
        for m in METHODS_V2:
            a = arrays(runs, dataset, backbone, m)
            if a is None:
                continue
            per_method[m] = a
            g300, g3000, gstop, step = gains(a)
            rows.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "method": m,
                    "seeds": len(a["seeds"]),
                    "g300": g300,
                    "g3000": g3000,
                    "gstop": gstop,
                    "stop_step": step,
                    "delta": gstop - g300,
                }  # fmt: skip
            )
        if not per_method:
            continue
        boot = []
        for _ in range(BOOTSTRAP):
            deltas = []
            for a in per_method.values():
                idx = rng.integers(len(a["seeds"]), size=len(a["seeds"]))
                g300, _, gstop, _ = gains(a, idx)
                deltas.append(gstop - g300)
            boot.append(np.mean(deltas))
        mean = float(
            np.mean(
                [r["delta"] for r in rows if (r["dataset"], r["backbone"]) == (dataset, backbone)]
            )
        )
        lo, hi = np.percentile(boot, [2.5, 97.5])
        groups[(dataset, backbone)] = (mean, float(lo), float(hi), sorted(per_method))

    claim = [(d, *groups[(d, "gcn")][:3]) for d in CLAIM_DATASETS if (d, "gcn") in groups]
    dependent = [d for d, mean, lo, _ in claim if mean >= THRESHOLD and lo > 0]
    complete = len(claim) == len(CLAIM_DATASETS)
    verdict = (
        f"budget-dependent ({', '.join(dependent)})"
        if dependent
        else ("stands" if complete else "incomplete: not every claim dataset has both budgets")
    )
    lines = [
        "# E5: training budget",
        "",
        "Generated by `benchmarks/analyze_e5.py`; rule fixed in `benchmarks/paper/PLAN.md` "
        '("GraphMAE and E5") before any E5 run. Regularized probe, standard split, gains over '
        "the untrained encoder of the same seed, in points. Gstop: the 3,000-step run at its "
        "best mean-validation evaluation step (300 ... 3,000). Δ = Gstop − G300; interval: 95% "
        "seed bootstrap, the stopping step chosen again in every replicate.",
        "",
        f"**Verdict for claim C3** (GCN, Amazon Computers / Photo, Coauthor CS; Δ ≥ "
        f"{THRESHOLD:g} with an interval above 0 on any of them): **{verdict}**",
        "",
        "| Data | Backbone | Mean Δ over objectives | Objectives |",
        "|---|---|---|---|",
        *[
            f"| {d} | {b} | {mean:+.2f} [{lo:+.2f}, {hi:+.2f}] | {len(ms)} |"
            for (d, b), (mean, lo, hi, ms) in groups.items()
        ],
        "",
        "| Data | Backbone | Objective | Seeds | G300 | G3000 | Gstop | Stop step | Δ |",
        "|---|---|---|---|---|---|---|---|---|",
        *[
            f"| {r['dataset']} | {r['backbone']} | {NAMES[r['method']]} | {r['seeds']} | "
            f"{r['g300']:+.1f} | {r['g3000']:+.1f} | {r['gstop']:+.1f} | {r['stop_step']} | "
            f"{r['delta']:+.1f} |"
            for r in rows
        ],
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    if rows:
        with open(args.out / "e5.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} (dataset, backbone, objective) cells -> {args.out}; C3: {verdict}")


if __name__ == "__main__":
    main()
