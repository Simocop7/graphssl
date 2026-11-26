"""E9, E10, E11 of the factor study: the rules fixed in benchmarks/paper/PLAN.md, applied.

    python benchmarks/analyze_c10_edges.py e9   [--root benchmarks/paper]   # label budgets
    python benchmarks/analyze_c10_edges.py e10  [--root ...]                # heterophilous graphs
    python benchmarks/analyze_c10_edges.py e11  [--root ...]                # joint tuning

Each writes ``<root>/analysis/<exp>/cells.csv`` (every D with its interval) and
``verdict.txt``. Shared definitions (PLAN.md, "E9, E10, E11"):

- valsel: for each representation and fit, the probe with the best validation score among the
  listed ones (ties: the first listed);
- bootstrap: 2,000 replicates, NumPy seed 0, percentile 95% interval; the seed index (E9: the
  seed-draw index) is resampled jointly for every objective and the reference;
- a cell *favors pretraining* when D >= 1.0 point and its lower bound > 0.

Scores are fractions in the result files and points (x 100) here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OBJECTIVES = ["dgi", "graphcl", "vicreg", "barlow_twins", "bgrl", "afgrl", "graphdino", "graphmae"]
BACKBONES = ["gcn", "gin", "transformer"]
E9_DATASETS = [
    "Cora", "CiteSeer", "PubMed", "AmazonComputers", "AmazonPhoto", "CoauthorCS", "WikiCS",
]  # fmt: skip
E9_BUDGETS = [1, 2, 5, 10, 20]
E10_DATASETS = ["RomanEmpire", "AmazonRatings", "Minesweeper"]
E11_DATASETS = ["Cora", "CiteSeer", "PubMed", "AmazonComputers", "AmazonPhoto", "CoauthorCS"]
A2X, PCA, RAW, CONCAT = "propagated A^2X", "PCA-256 of A^2X", "raw features", "concat X..A^2X"
N_BOOT, THRESHOLD = 2000, 1.0
CHECKPOINT = re.compile(r"(?P<backbone>[a-z]+)/(?P<objective>[a-z_]+)_seed(?P<seed>\d+)\.pt$")


def latest(directory: Path) -> dict:
    files = sorted(directory.glob("probes__*.json"))
    if not files:
        raise FileNotFoundError(f"no probe file in {directory}")
    return json.loads(files[-1].read_text())


def parse(row: dict) -> tuple[str, str, int] | None:
    """(backbone, objective, seed) of a checkpoint row; None for an input-feature row."""
    if not row.get("checkpoint"):
        return None
    m = CHECKPOINT.search(row["checkpoint"])
    if m is None:
        raise ValueError(f"unexpected checkpoint name {row['checkpoint']}")
    objective = m["objective"]
    if "untrained" in row["checkpoint"].split("/")[-4]:
        objective = "untrained"
    return m["backbone"], objective, int(m["seed"])


def valsel(fits: dict, probes: list[str]) -> float:
    """Test score of the probe with the best validation score (ties: the first listed)."""
    best = max(range(len(probes)), key=lambda i: (fits[probes[i]]["val_acc"], -i))
    return 100 * fits[probes[best]]["test_acc"]


def bootstrap_d(models: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    """D = mean over objectives and indices of (model - reference), with a joint index bootstrap.

    ``models``: [objectives, indices]; ``reference``: [indices] (constant for a fixed row).
    """
    diff = models - reference[None, :]
    rng = np.random.default_rng(0)
    n = diff.shape[1]
    reps = np.array([diff[:, rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(diff.mean()), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def favors(d: float, lo: float) -> bool:
    return d >= THRESHOLD and lo > 0


def write(root: Path, exp: str, rows: list[dict], verdict: list[str]) -> None:
    out = root / "analysis" / exp
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cells.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "verdict.txt").write_text("\n".join(verdict) + "\n")
    print("\n".join(verdict))


# ---------------------------------------------------------------------------- E9


def e9(root: Path) -> None:
    probes = ["graphssl", "graphssl_alln", "liblinear"]
    rows, verdict = [], []
    favorable: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for dataset in E9_DATASETS:
        result = latest(root / "e9" / dataset)
        # scores[(val_mode, k)][(backbone, objective)][index] and refs[(val_mode, k)][name][index]
        scores: dict = defaultdict(lambda: defaultdict(dict))
        refs: dict = defaultdict(lambda: defaultdict(dict))
        for row in result["rows"]:
            key = parse(row)
            for fit in row["label_budgets"]:
                cell = (fit["val"], fit["k"])
                value = valsel(fit, probes)
                if key is None:
                    refs[cell][row["name"]][fit["draw"]] = value
                else:
                    backbone, objective, seed = key
                    assert fit["draw"] == seed, (row["name"], fit["draw"])
                    scores[cell][(backbone, objective)][seed] = value
        for (mode, k), per_model in sorted(scores.items()):
            for backbone in BACKBONES:
                models = np.array(
                    [[per_model[(backbone, o)][i] for i in range(10)] for o in OBJECTIVES]
                )
                untrained = np.array([per_model[(backbone, "untrained")][i] for i in range(10)])
                for ref_name in (A2X, PCA, RAW, "untrained"):
                    if ref_name == "untrained":
                        ref = untrained
                    elif ref_name in refs[(mode, k)]:
                        ref = np.array([refs[(mode, k)][ref_name][i] for i in range(10)])
                    else:
                        continue
                    d, lo, hi = bootstrap_d(models, ref)
                    fav = favors(d, lo)
                    rows.append(
                        {
                            "dataset": dataset, "k": k, "val": mode, "backbone": backbone,
                            "reference": ref_name, "D": round(d, 3), "lo": round(lo, 3),
                            "hi": round(hi, 3), "favors": fav,
                            "models_above_ref": int((models.mean(1) > ref.mean()).sum()),
                            "reference_mean": round(float(ref.mean()), 3),
                            "objectives_mean": round(float(models.mean()), 3),
                        }
                    )  # fmt: skip
                    if fav and mode == "standard" and backbone == "gcn":
                        favorable[(ref_name, k, mode)].append(dataset)
    budgets = [k for k in E9_BUDGETS if len(favorable[(A2X, k, "standard")]) >= 4]
    verdict.append(
        "E9 rule (GCN, valsel over ours / ours with every node's statistics / liblinear,"
    )
    verdict.append(
        "standard validation; a cell favors pretraining when D >= 1.0 with lower bound > 0):"
    )
    for k in E9_BUDGETS:
        verdict.append(
            f"  K = {k:2d}: graphs favoring pretraining over A^2X: "
            f"{len(favorable[(A2X, k, 'standard')])} of 7 {favorable[(A2X, k, 'standard')]}"
        )
    if not budgets:
        verdict.append("VERDICT: no budget has 4 of 7 graphs: C10 extends to 1-20 labels per class")
    else:
        verdict.append(
            f"VERDICT: with K in {budgets} labels per class the average pretrained GCN exceeds "
            "A^2X on most graphs; C10 is stated for the standard budgets"
        )
        for k in budgets:
            n = len(favorable[(PCA, k, "standard")])
            what = "pretraining" if n >= 4 else "the dimension (PCA-256 of A^2X recovers it)"
            verdict.append(
                f"  attribution at K = {k}: {n} of 7 graphs still favor pretraining over PCA-256 "
                f"of A^2X -> the advantage is attributed to {what}"
            )
    write(root, "e9", rows, verdict)


# ---------------------------------------------------------------------------- E10


def split0_table(result: dict, probes: list[str]) -> tuple[dict, dict, dict]:
    """(models[(backbone, objective)][seed], inputs[name], input validation[name]) on split 0."""
    models: dict = defaultdict(dict)
    inputs, inputs_val = {}, {}
    for row in result["rows"]:
        fits = {p: row["protocols"][p]["fits"][0] for p in probes}
        key = parse(row)
        if key is None:
            inputs[row["name"]] = valsel(fits, probes)
            inputs_val[row["name"]] = max(100 * fits[p]["val_acc"] for p in probes)
        else:
            models[(key[0], key[1])][key[2]] = valsel(fits, probes)
    return models, inputs, inputs_val


def matrix(models: dict, backbone: str, seeds: int = 10) -> np.ndarray:
    return np.array([[models[(backbone, o)][s] for s in range(seeds)] for o in OBJECTIVES])


def e10(root: Path) -> None:
    probes = ["graphssl", "liblinear"]
    rows, verdict = [], []
    q1, q2 = [], defaultdict(list)
    for dataset in E10_DATASETS:
        result = latest(root / "e10" / "probes" / dataset)
        models, inputs, inputs_val = split0_table(result, probes)
        r_star = max((RAW, A2X, CONCAT), key=lambda n: inputs_val[n])
        for backbone in BACKBONES:
            m = matrix(models, backbone)
            for ref_name in (A2X, r_star, "untrained"):
                if ref_name == "untrained":
                    ref = np.array([models[(backbone, "untrained")][s] for s in range(10)])
                else:
                    ref = np.full(10, inputs[ref_name])
                d, lo, hi = bootstrap_d(m, ref)
                fav = favors(d, lo)
                rows.append(
                    {
                        "dataset": dataset, "metric": result["metric"], "backbone": backbone,
                        "reference": ref_name, "is_r_star": ref_name == r_star, "D": round(d, 3),
                        "lo": round(lo, 3), "hi": round(hi, 3), "favors": fav,
                        "models_above_ref": int((m.mean(1) > ref.mean()).sum()),
                        "reference_mean": round(float(ref.mean()), 3),
                        "objectives_mean": round(float(m.mean()), 3),
                        "best_objective": OBJECTIVES[int(m.mean(1).argmax())],
                    }
                )  # fmt: skip
                if ref_name == A2X and backbone == "gcn" and fav:
                    q1.append(dataset)
                if ref_name == r_star and fav:
                    q2[backbone].append(dataset)
        verdict.append(
            f"{dataset} ({result['metric']}): R* = {r_star} "
            f"(validation {inputs_val[r_star]:.2f}, test {inputs[r_star]:.2f}); "
            + ", ".join(f"{n} {inputs[n]:.2f}" for n in (RAW, "propagated A^1X", A2X, CONCAT))
        )
    verdict.append(f"Q1 (GCN objectives vs A^2X): {len(q1)} of 3 graphs favor pretraining {q1}")
    verdict.append(
        "VERDICT Q1: C10 does not extend to heterophilous graphs"
        if len(q1) >= 2
        else "VERDICT Q1: C10 also holds on these heterophilous graphs"
    )
    winners = [b for b in BACKBONES if len(q2[b]) >= 2]
    for b in BACKBONES:
        verdict.append(
            f"Q2 ({b} objectives vs R*): {len(q2[b])} of 3 graphs favor pretraining {q2[b]}"
        )
    verdict.append(
        f"VERDICT Q2: pretraining adds beyond the best parameter-free filter with {winners}"
        if winners
        else "VERDICT Q2: no backbone's average objective exceeds a linear probe on [X, AX, A^2X]"
    )
    write(root, "e10", rows, verdict)


# ---------------------------------------------------------------------------- E11


def e11(root: Path) -> None:
    probes = ["graphssl", "liblinear"]
    rows, verdict, favorable = [], [], []
    for dataset in E11_DATASETS:
        result = latest(root / "e11" / "probes" / dataset)
        models, inputs, inputs_val = split0_table(result, probes)
        m = matrix(models, "gcn", seeds=5)
        k_best = max(("propagated A^1X", A2X, "propagated A^3X"), key=lambda n: inputs_val[n])
        for ref_name in (A2X, k_best):
            ref = np.full(5, inputs[ref_name])
            d, lo, hi = bootstrap_d(m, ref)
            fav = favors(d, lo)
            rows.append(
                {
                    "dataset": dataset, "reference": ref_name, "rule": ref_name == A2X,
                    "D": round(d, 3), "lo": round(lo, 3), "hi": round(hi, 3), "favors": fav,
                    "objectives_above_ref": int((m.mean(1) > ref.mean()).sum()),
                    "reference_mean": round(float(ref.mean()), 3),
                    "objectives_mean": round(float(m.mean()), 3),
                }
            )  # fmt: skip
            if ref_name == A2X and fav:
                favorable.append(dataset)
        verdict.append(
            f"{dataset}: D_tuned = {rows[-2]['D']:+.2f} [{rows[-2]['lo']:+.2f}, {rows[-2]['hi']:+.2f}] "
            f"vs A^2X {inputs[A2X]:.2f}; A^K X chosen on validation: {k_best}"
        )
    n = len(favorable)
    verdict.append(f"datasets favoring the tuned objectives over A^2X: {n} of 6 {favorable}")
    if n == 0:
        verdict.append("VERDICT: C10 holds under per-objective joint tuning")
    elif n <= 3:
        verdict.append(
            f"VERDICT: C10 stated for the shared protocol; tuned exceptions: {favorable}"
        )
    else:
        verdict.append(
            "VERDICT: C10 withdrawn as a general statement; kept for the shared protocol"
        )
    write(root, "e11", rows, verdict)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("experiment", choices=["e9", "e10", "e11"])
    p.add_argument("--root", default="benchmarks/paper", type=Path)
    a = p.parse_args()
    {"e9": e9, "e10": e10, "e11": e11}[a.experiment](a.root)


if __name__ == "__main__":
    main()
