"""E3: does tuning each method's optimizer and learning rate change the conclusions of E1?

Reads every ``run_benchmark.py`` JSON under ``benchmarks/paper/e3/<cell>/<dataset>/`` (one
directory per optimizer x learning-rate cell) and compares, per dataset and backbone, the shared
untuned cell (AdamW 5e-4, E3's own run of it) with every method at its own best cell, chosen on
**validation** accuracy (PLAN.md, Analysis). A cell where any seed diverged cannot be chosen.
Cora / CiteSeer: every backbone (VM GPU); Amazon Photo / Computers, Coauthor CS: GCN only, on the
device of their E1 runs (PLAN.md "E3 on the Amazon and Coauthor graphs", added 29 Oct, with the
decision rule for claim C3 evaluated here). Writes to ``--out``:

summary.md   ranking (Kendall tau, shared vs tuned), the mean lift over the objectives, the
             decision rule for C3, the lift of every method with its cell, the spread between
             methods, how often the chosen cell is at the edge of the grid, the diverged cells
             and the device of every dataset.
e3.csv       one row per (dataset, backbone, method).
ranking.csv  one row per (dataset, backbone).

A (dataset, backbone) group is analysed only when every cell of the grid holds the same
objectives; an incomplete group is named in the summary, never dropped in silence. A cell outside
the ten (PLAN.md grid-edge rule: one step beyond an edge, for one objective on one dataset) is
used in that objective's selection only, and listed in the summary.

Intervals: seed bootstrap, the selection of the cell included in every replicate (each method's
seeds are resampled together across cells, which share seed indices).

Usage::

    python benchmarks/analyze_e3.py --out benchmarks/paper/analysis/e3
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from paper_figures import METHODS_V2, NAMES
from paper_manifest import E3_GRID
from scipy.stats import kendalltau

SHARED = "adamw_lr5e-4"
GRID = {f"{optimizer}_lr{lr}" for optimizer, lr in E3_GRID}  # every group needs these ten
DATASETS = ["Cora", "CiteSeer", "AmazonComputers", "AmazonPhoto", "CoauthorCS"]
BACKBONES = ["gcn", "gin", "transformer"]
BOOTSTRAP = 2000
# PLAN.md, fixed 29 Oct before any of its runs: if on any of these datasets the mean lift over the
# objectives (GCN, tuned - shared) is >= THRESHOLD points with its 95% interval above 0, claim C3
# is reported as protocol-dependent, with the tuned gains; otherwise it stands.
RULE_BACKBONE = "gcn"
RULE_DATASETS = ["AmazonComputers", "AmazonPhoto", "CoauthorCS"]
THRESHOLD = 1.0


# The objectives with a run in every cell of every complete group, set in main(): a method whose
# runs are incomplete is left out and named, never a whole group dropped in silence.
METHODS: list[str] = []


def groups(runs: dict) -> tuple[list[tuple], list[tuple]]:
    """(complete, incomplete) (dataset, backbone) groups: complete = every cell of the grid,
    with the same objectives in each."""
    cells = GRID
    complete, incomplete = [], []
    for d, b in sorted({(k[1], k[2]) for k in runs}):
        per_cell = {
            frozenset(m for c2, d2, b2, m in runs if (c2, d2, b2) == (c, d, b)) for c in cells
        }
        (complete if len(per_cell) == 1 and all(per_cell) else incomplete).append((d, b))
    return complete, incomplete


def complete_methods(runs: dict, complete: list[tuple]) -> list[str]:
    return [m for m in METHODS_V2 if all((c, d, b, m) in runs for c in GRID for d, b in complete)]


def cell_key(cell: str) -> tuple:
    optimizer, lr = cell.split("_lr")
    return optimizer, float(lr)


def at_edge(cell: str, cells: list) -> bool:
    """Whether a cell's learning rate is the lowest or highest of its optimizer's grid."""
    optimizer, lr = cell_key(cell)
    lrs = [r for o, r in map(cell_key, cells) if o == optimizer]
    return lr in (min(lrs), max(lrs))


def load(root: Path) -> tuple[dict, dict]:
    """({(cell, dataset, backbone, method): {"test": [seeds], "val": [seeds]}}, NaN = diverged;
    {dataset: devices its runs record})."""
    runs: dict = {}
    devices: dict = {}
    for path in sorted((root / "benchmarks" / "paper" / "e3").glob("*/*/*.json")):
        r = json.loads(path.read_text())
        key = (path.parts[-3], r["dataset"], r["hyperparameters"]["encoder"], r["model"])
        runs[key] = {  # the newest file of a key wins (sorted names end in a timestamp)
            split: np.array(
                [
                    np.nan if s.get(f"{split}_acc_linear") is None else s[f"{split}_acc_linear"]
                    for s in r["per_seed"]
                ],
                dtype=float,
            )
            * 100
            for split in ("test", "val")
        }
        prov = r["provenance"]
        platform = prov["versions"]["platform"].split("-")[0]
        devices.setdefault(r["dataset"], set()).add(
            prov.get("device") or f"{platform}, device not recorded"
        )
    return runs, devices


def select(runs: dict, cells: list, dataset: str, backbone: str, method: str, idx=None) -> tuple:
    """(test accuracy of the shared cell, chosen cell, its test accuracy) for one method."""
    best_cell, best_val = None, -np.inf
    for cell in cells:
        run = runs.get((cell, dataset, backbone, method))
        if run is None:
            continue
        test, val = (
            (run["test"], run["val"]) if idx is None else (run["test"][idx], run["val"][idx])
        )
        if np.isnan(test).any():
            continue  # a diverged seed: the cell cannot be chosen
        if val.mean() > best_val:
            best_cell, best_val = cell, val.mean()
    shared = runs[(SHARED, dataset, backbone, method)]["test"]
    chosen = runs[(best_cell, dataset, backbone, method)]["test"]
    if idx is not None:
        shared, chosen = shared[idx], chosen[idx]
    return float(shared.mean()), best_cell, float(chosen.mean())


def compare(runs: dict, cells: list, dataset: str, backbone: str, rng) -> dict:
    point = {m: select(runs, cells, dataset, backbone, m) for m in METHODS}
    n_seeds = len(runs[(SHARED, dataset, backbone, METHODS[0])]["test"])
    taus, lifts, spreads, mean_lifts = [], {m: [] for m in METHODS}, [], []
    for _ in range(BOOTSTRAP):
        rep = {
            m: select(runs, cells, dataset, backbone, m, rng.integers(n_seeds, size=n_seeds))
            for m in METHODS
        }
        shared, tuned = [rep[m][0] for m in METHODS], [rep[m][2] for m in METHODS]
        taus.append(kendalltau(shared, tuned).statistic)
        spreads.append((max(shared) - min(shared), max(tuned) - min(tuned)))
        for m in METHODS:
            lifts[m].append(rep[m][2] - rep[m][0])
        mean_lifts.append(np.mean([rep[m][2] - rep[m][0] for m in METHODS]))
    shared, tuned = [point[m][0] for m in METHODS], [point[m][2] for m in METHODS]
    return {
        "point": point,
        "tau": (kendalltau(shared, tuned).statistic, *np.nanpercentile(taus, [2.5, 97.5])),
        "lift_ci": {m: tuple(np.percentile(lifts[m], [2.5, 97.5])) for m in METHODS},
        "spread": (max(shared) - min(shared), max(tuned) - min(tuned)),
        "spread_change_ci": tuple(np.percentile([t - s for s, t in spreads], [2.5, 97.5])),
        "mean_lift": (
            float(np.mean([t - s for s, t in zip(shared, tuned, strict=True)])),
            *np.percentile(mean_lifts, [2.5, 97.5]),
        ),
        # against the cells this method ran on this dataset and backbone: an extension cell of
        # another (method, dataset) must not move the edge (DGI's AdamW 5e-5 on Amazon Photo hid
        # DGI at AdamW 1e-4 on Amazon Computers until 31 Oct)
        "edge": sum(
            at_edge(point[m][1], [c for c in cells if (c, dataset, backbone, m) in runs])
            for m in METHODS
        ),
    }


def verdict(mean_lift: tuple) -> str:
    lift, lo, _ = mean_lift
    return "protocol-dependent" if lift >= THRESHOLD and lo > 0 else "stands"


def write(runs: dict, devices: dict, complete: list, incomplete: list, out: Path) -> None:
    rng = np.random.default_rng(0)
    cells = sorted({k[0] for k in runs}, key=cell_key)
    lines = [
        "# E3: optimizer x learning rate",
        "",
        "Generated by `benchmarks/analyze_e3.py`. GraphSSL probe, test accuracy in %, means over "
        f"5 seeds; [95% seed-bootstrap interval, the choice of cell included]. *Shared*: E3's "
        f"own {SHARED} cell, on the dataset's own device (see Devices), never E1. *Tuned*: every "
        "method at the cell with its best mean **validation** accuracy; a cell with a diverged "
        "seed is never chosen. *Mean lift*: tuned - shared, averaged over the objectives. "
        "*Edge*: methods whose chosen cell is the lowest or highest learning rate of its "
        "optimizer.",
        "",
        "## Ranking and spread, shared vs tuned",
        "",
        "| Data | Backbone | Kendall tau | Spread shared | Spread tuned | Change | Mean lift "
        "| Edge |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows, ranking, per_group = [], [], {}
    for d in DATASETS:
        for b in BACKBONES:
            if (d, b) not in complete:
                continue
            res = per_group[(d, b)] = compare(runs, cells, d, b, rng)
            t, lo, hi = res["tau"]
            s, u = res["spread"]
            clo, chi = res["spread_change_ci"]
            ml, mlo, mhi = res["mean_lift"]
            lines.append(
                f"| {d} | {b} | {t:+.2f} [{lo:+.2f}, {hi:+.2f}] | {s:.1f} | {u:.1f} | "
                f"{u - s:+.1f} [{clo:+.1f}, {chi:+.1f}] | {ml:+.1f} [{mlo:+.1f}, {mhi:+.1f}] | "
                f"{res['edge']} of {len(METHODS)} |"
            )
            ranking.append(
                {
                    "dataset": d,
                    "backbone": b,
                    "tau": t,
                    "tau_lo": lo,
                    "tau_hi": hi,
                    "spread_shared": s,
                    "spread_tuned": u,
                    "change_lo": clo,
                    "change_hi": chi,
                    "mean_lift": ml,
                    "mean_lift_lo": mlo,
                    "mean_lift_hi": mhi,
                    "edge_choices": res["edge"],
                    "n_methods": len(METHODS),
                }
            )
    lines += [
        "",
        "## Decision rule for C3 (PLAN.md, fixed 29 Oct before the runs)",
        "",
        f"{RULE_BACKBONE.upper()}: C3 is reported as protocol-dependent if on any of these "
        f"datasets the mean lift is >= {THRESHOLD:.1f} point with its interval above 0.",
        "",
        "| Data | Mean lift | Verdict |",
        "|---|---|---|",
    ]
    verdicts = []
    for d in RULE_DATASETS:
        res = per_group.get((d, RULE_BACKBONE))
        if res is None:
            verdicts.append("pending")
            lines.append(f"| {d} | - | pending (runs incomplete) |")
            continue
        ml, mlo, mhi = res["mean_lift"]
        verdicts.append(verdict(res["mean_lift"]))
        lines.append(f"| {d} | {ml:+.1f} [{mlo:+.1f}, {mhi:+.1f}] | {verdicts[-1]} |")
    overall = (
        "pending"
        if "pending" in verdicts and "protocol-dependent" not in verdicts
        else ("protocol-dependent" if "protocol-dependent" in verdicts else "stands")
    )
    lines += ["", f"Overall: **C3 {overall}**."]
    for (d, b), res in per_group.items():
        lines += [
            "",
            f"## {d}, {b}",
            "",
            "| Method | Shared | Tuned | Cell | Lift |",
            "|---|---|---|---|---|",
        ]
        for m in METHODS:
            shared, cell, tuned = res["point"][m]
            lo, hi = res["lift_ci"][m]
            lines.append(
                f"| {NAMES[m]} | {shared:.1f} | {tuned:.1f} | {cell} | "
                f"{tuned - shared:+.1f} [{lo:+.1f}, {hi:+.1f}] |"
            )
            rows.append(
                {
                    "dataset": d,
                    "backbone": b,
                    "method": m,
                    "shared": shared,
                    "tuned": tuned,
                    "cell": cell,
                    "lift_lo": lo,
                    "lift_hi": hi,
                }
            )
    diverged = [
        (cell, d, b, m, int(np.isnan(run["test"]).sum()), len(run["test"]))
        for (cell, d, b, m), run in sorted(runs.items(), key=lambda kv: cell_key(kv[0][0]))
        if np.isnan(run["test"]).any()
    ]
    lines += ["", "## Diverged cells", "", "| Cell | Data | Backbone | Method | Seeds |"]
    lines += ["|---|---|---|---|---|"]
    lines += [f"| {c} | {d} | {b} | {NAMES[m]} | {n} of {k} |" for c, d, b, m, n, k in diverged]
    extensions = sorted((c, d, b, m) for c, d, b, m in runs if c not in GRID)
    if extensions:
        lines += ["", "## Grid extensions (PLAN.md edge rule)", ""]
        lines += [f"- {c}: {d}, {b}, {NAMES[m]}" for c, d, b, m in extensions]
    lines += ["", "## Devices", "", "| Data | Devices of its runs |", "|---|---|"]
    lines += [f"| {d} | {', '.join(sorted(devices[d]))} |" for d in DATASETS if d in devices]
    if incomplete:
        lines += ["", "## Incomplete groups, not analysed", ""]
        lines += [f"- {d}, {b}" for d, b in incomplete]
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    for name, table in (("e3.csv", rows), ("ranking.csv", ranking)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--root", type=Path, default=Path("."), help="tree whose benchmarks/paper/e3 holds the runs"
    )
    p.add_argument("--out", type=Path, default=Path("benchmarks/paper/analysis/e3"))
    args = p.parse_args()
    runs, devices = load(args.root)
    complete, incomplete = groups(runs)
    METHODS[:] = complete_methods(runs, complete)
    left_out = sorted({k[3] for k in runs} - set(METHODS))
    print(
        f"objectives: {', '.join(METHODS)}"
        + (f"; incomplete, left out: {left_out}" if left_out else "")
        + (f"; incomplete groups, not analysed: {incomplete}" if incomplete else "")
    )
    write(runs, devices, complete, incomplete, args.out)
    print(f"{len(runs)} runs -> {args.out}")


if __name__ == "__main__":
    main()
