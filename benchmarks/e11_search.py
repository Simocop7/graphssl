"""E11 of the factor study: a joint random search per objective (benchmarks/paper/PLAN.md).

Answers "the shared protocol is untuned" for claim C10 (with a GCN, the average objective does
not exceed logistic regression on A^2X). Every objective gets the same budget: the same
``N_CONFIGS`` configurations drawn once from its search space (fixed by ``SPACE_SEED`` and
written to ``configs.json`` before any run), plus the shared protocol itself as an anchor.
Selection uses validation labels only:

1. stage 1   every configuration, one seed (100), GCN; validation accuracy of the default probe
2. stage 2   the ``TOP`` best configurations of stage 1, two more seeds (101, 102); the one with
             the best mean validation accuracy over the three seeds is selected (ties: the lower
             configuration id)
3. final     the selected configuration with seeds 0-4 (never seen by the selection), models saved
4. probe     the final models and the input-feature rows scored as in E6 (our probe and
             liblinear on the standard split; the analysis picks per representation on validation)

The search covers the encoder's width, depth, normalization and dropout, AdamW's learning rate
and weight decay, the number of training steps, the augmentation strength (augmentation-based
objectives) and each objective's own hyperparameters. Views stay symmetric (one augmentation
distribution for both views) and the encoder stays a GCN.

Usage::

    python benchmarks/e11_search.py plan                     # writes benchmarks/paper/e11/configs.json
    python benchmarks/e11_search.py run --dataset PubMed --device cuda [--objectives graphcl afgrl]
    python benchmarks/e11_search.py probe --dataset PubMed
    python benchmarks/e11_search.py status

``run`` is resumable: a step whose JSON exists is skipped. Each step is one
``run_benchmark.py`` call; a failed call is logged as ``E11 STEP FAILED`` and the objective is
left for a rerun, the others go on.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
E11 = REPO_ROOT / "benchmarks" / "paper" / "e11"
MODELS = REPO_ROOT / "benchmarks" / "paper" / "models_e11"
CONFIGS = E11 / "configs.json"

OBJECTIVES = ["dgi", "graphcl", "vicreg", "barlow_twins", "bgrl", "afgrl", "graphdino", "graphmae"]
DATASETS = ["Cora", "CiteSeer", "PubMed", "AmazonComputers", "AmazonPhoto", "CoauthorCS"]
N_CONFIGS = 16
TOP = 3
SPACE_SEED = 20261126
# FINAL_SEEDS was 10 until 26 Nov 20:00 HKT although the rule (PLAN.md, committed with this file)
# says seeds 0-4: finals already run keep seeds 5-9, which the probe and the analysis ignore.
SEARCH_SEED, STAGE2_SEEDS, FINAL_SEEDS = 100, 2, 5
AUGMENTED = {"graphcl", "vicreg", "barlow_twins", "bgrl", "graphdino"}

# The shared protocol of E1: the anchor every search contains.
ANCHOR = {"hidden": 256, "layers": 2, "lr": 5e-4, "weight_decay": 1e-5, "epochs": 300, "set": {}}


def sample_config(objective: str, rng: np.random.Generator) -> dict:
    """One configuration of ``objective``'s search space."""

    def choice(values):
        return values[int(rng.integers(len(values)))]

    cfg = {
        "hidden": choice([128, 256, 512]),
        "layers": choice([1, 2, 3]),
        "lr": float(10 ** rng.uniform(-4, -2)),
        "weight_decay": float(10 ** rng.uniform(-6, -3)),
        "epochs": choice([100, 300, 1000]),
    }
    s: dict = {
        "encoder.norm_type": choice(["batch", "layer", "none"]),
        "encoder.drop": choice([0.0, 0.2]),
    }
    if objective in AUGMENTED:
        augment = [
            {"name": "edge_drop", "p": round(float(rng.uniform(0.0, 0.7)), 3)},
            {"name": "feat_mask", "p": round(float(rng.uniform(0.0, 0.5)), 3)},
        ]
        if objective == "graphdino":
            s["augment_teacher"] = augment
            s["augment_student"] = augment
        else:
            s["augment"] = augment
    if objective == "graphcl":
        s["tau"] = round(float(10 ** rng.uniform(-1, 0)), 3)
        s["proj_dim"] = choice([128, 256])
    elif objective == "vicreg":
        s["proj_dim"] = choice([256, 512, 1024])
        s["invariance"] = choice([10.0, 25.0, 50.0])
    elif objective == "barlow_twins":
        s["proj_dim"] = choice([256, 512, 1024])
        s["lambda_param"] = choice([None, 5e-3])
    elif objective == "bgrl":
        s["ema_tau"] = choice([0.9, 0.99, 0.999])
        s["pred_hidden"] = choice([256, 512])
        s["pred_norm"] = choice(["batch", "none"])
    elif objective == "afgrl":
        s["ema_tau"] = choice([0.9, 0.99])
        s["topk"] = choice([4, 8])
        s["num_centroids"] = choice([50, 100])
        s["pred_norm"] = choice(["batch", "none"])
    elif objective == "graphdino":
        s["ema_tau_base"] = choice([0.9, 0.996])
        s["head.teacher_temp"] = choice([0.04, 0.07])
    elif objective == "graphmae":
        s["mask_rate"] = round(float(rng.uniform(0.25, 0.75)), 3)
        s["replace_rate"] = choice([0.0, 0.05, 0.1])
        s["alpha"] = choice([1.0, 2.0, 3.0])
    cfg["set"] = s
    return cfg


def plan() -> None:
    if CONFIGS.exists():
        sys.exit(f"{CONFIGS} exists: the search space is fixed, delete it only deliberately")
    configs = {}
    for i, objective in enumerate(OBJECTIVES):
        rng = np.random.default_rng(SPACE_SEED + 1000 * i)
        configs[objective] = {"c00": ANCHOR} | {
            f"c{j:02d}": sample_config(objective, rng) for j in range(1, N_CONFIGS + 1)
        }
    E11.mkdir(parents=True, exist_ok=True)
    CONFIGS.write_text(json.dumps(configs, indent=1) + "\n")
    print(f"wrote {CONFIGS}: {len(OBJECTIVES)} objectives x {N_CONFIGS + 1} configurations")


def flags(cfg: dict) -> list[str]:
    out = [
        "--hidden", str(cfg["hidden"]),
        "--layers", str(cfg["layers"]),
        "--lr", repr(cfg["lr"]),
        "--weight-decay", repr(cfg["weight_decay"]),
        "--epochs", str(cfg["epochs"]),
    ]  # fmt: skip
    for key, value in cfg["set"].items():
        out += ["--set", f"{key}={json.dumps(value)}"]
    return out


def results(directory: Path) -> dict[int, dict]:
    """seed -> per-seed record of every run_benchmark JSON under ``directory``."""
    out: dict[int, dict] = {}
    for path in sorted(directory.glob("*/*.json")):
        for record in json.loads(path.read_text())["per_seed"]:
            out[int(record["seed"])] = record
    return out


def val(record: dict) -> float:
    v = record.get("val_acc_linear")
    return -math.inf if v is None else float(v)


def call(cmd: list[str], log: Path) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f"E11 STEP FAILED (exit {rc}): {' '.join(cmd[2:8])} ... log {log}", flush=True)
    return rc == 0


def runner(
    dataset: str, objective: str, cfg: dict, out: Path, seeds: int, first: int, args
) -> list[str]:
    return [
        sys.executable, "-u", "benchmarks/run_benchmark.py",
        "--dataset", dataset, "--model", objective, "--encoder", "gcn",
        "--seeds", str(seeds), "--first-seed", str(first),
        "--device", args.device, "--out-dir", str(out.relative_to(REPO_ROOT)),
        *flags(cfg),
    ]  # fmt: skip


def selection(dataset: str, objective: str, configs: dict) -> tuple[str | None, dict]:
    """(selected configuration id or None if stage 2 is incomplete, the scores behind it)."""
    base = E11 / "search" / dataset / objective
    stage1 = {cid: results(base / cid).get(SEARCH_SEED) for cid in configs}
    if any(r is None for r in stage1.values()):
        return None, {}
    ranked = sorted(configs, key=lambda cid: (-val(stage1[cid]), cid))[:TOP]
    means = {}
    for cid in ranked:
        rec = results(base / cid)
        seeds = range(SEARCH_SEED, SEARCH_SEED + 1 + STAGE2_SEEDS)
        if any(s not in rec for s in seeds):
            return None, {}
        means[cid] = sum(val(rec[s]) for s in seeds) / len(seeds)
    best = min(means, key=lambda cid: (-means[cid], cid))
    return best, {"stage1_val": {c: val(r) for c, r in stage1.items()}, "stage2_mean_val": means}


def run(args) -> None:
    configs = json.loads(CONFIGS.read_text())
    failed = 0
    for objective in args.objectives:
        base = E11 / "search" / args.dataset / objective
        log = E11 / "logs" / f"{args.dataset}_{objective}.log"
        # Stage 1: every configuration, seed 100.
        for cid, cfg in configs[objective].items():
            if SEARCH_SEED in results(base / cid):
                continue
            cmd = runner(args.dataset, objective, cfg, base / cid, 1, SEARCH_SEED, args)
            failed += not call(cmd, log)
        stage1 = {cid: results(base / cid).get(SEARCH_SEED) for cid in configs[objective]}
        if any(r is None for r in stage1.values()):
            print(f"[{args.dataset}/{objective}] stage 1 incomplete, skipped", flush=True)
            continue
        # Stage 2: the TOP best, seeds 101-102.
        ranked = sorted(stage1, key=lambda cid: (-val(stage1[cid]), cid))[:TOP]
        for cid in ranked:
            if all(
                s in results(base / cid)
                for s in range(SEARCH_SEED + 1, SEARCH_SEED + 1 + STAGE2_SEEDS)
            ):
                continue
            cmd = runner(args.dataset, objective, configs[objective][cid], base / cid,
                         STAGE2_SEEDS, SEARCH_SEED + 1, args)  # fmt: skip
            failed += not call(cmd, log)
        best, scores = selection(args.dataset, objective, configs[objective])
        if best is None:
            print(f"[{args.dataset}/{objective}] stage 2 incomplete, skipped", flush=True)
            continue
        sel = E11 / "selection" / args.dataset / f"{objective}.json"
        sel.parent.mkdir(parents=True, exist_ok=True)
        sel.write_text(
            json.dumps({"selected": best, "config": configs[objective][best], **scores}, indent=1)
            + "\n"
        )
        # Final: seeds 0-4, models saved.
        final = E11 / "final" / args.dataset / objective
        if len(results(final)) < FINAL_SEEDS:
            cmd = runner(
                args.dataset, objective, configs[objective][best], final, FINAL_SEEDS, 0, args
            )
            cmd += ["--save-models", str(MODELS.relative_to(REPO_ROOT))]
            failed += not call(cmd, log)
        print(
            f"[{args.dataset}/{objective}] selected {best}: {configs[objective][best]}", flush=True
        )
    print(f"E11 RUN DONE {args.dataset} {' '.join(args.objectives)}: {failed} failed", flush=True)


def probe(args) -> None:
    checkpoints = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in (MODELS / args.dataset / "gcn").glob("*.pt")
        if int(p.stem.rsplit("_seed", 1)[1]) < FINAL_SEEDS
    )
    expected = len(OBJECTIVES) * FINAL_SEEDS
    if len(checkpoints) != expected:
        sys.exit(f"{args.dataset}: {len(checkpoints)} final models, expected {expected}")
    cmd = [
        sys.executable, "-u", "benchmarks/compare_probes.py", "--dataset", args.dataset,
        "--raw-features", "--propagated", "1", "2", "3", "--protocols", "graphssl", "liblinear",
        "--checkpoints", *checkpoints, "--device", "cpu",
        "--out-dir", str((E11 / "probes" / args.dataset).relative_to(REPO_ROOT)),
    ]  # fmt: skip
    ok = call(cmd, E11 / "logs" / f"{args.dataset}_probe.log")
    print(f"E11 PROBE DONE {args.dataset}: {'0' if ok else '1'} failed", flush=True)


def status(_args) -> None:
    configs = json.loads(CONFIGS.read_text())
    for dataset in DATASETS:
        cells = []
        for objective in OBJECTIVES:
            base = E11 / "search" / dataset / objective
            n1 = sum(SEARCH_SEED in results(base / cid) for cid in configs[objective])
            n_final = len(results(E11 / "final" / dataset / objective))
            cells.append(f"{objective[:6]} {n1}/{len(configs[objective])} f{n_final}")
        print(f"{dataset:16s} " + " | ".join(cells))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True, choices=DATASETS)
    r.add_argument("--objectives", nargs="+", default=OBJECTIVES, choices=OBJECTIVES)
    r.add_argument("--device", default="cpu")
    pr = sub.add_parser("probe")
    pr.add_argument("--dataset", required=True, choices=DATASETS)
    sub.add_parser("status")
    args = p.parse_args()
    {"plan": lambda a: plan(), "run": run, "probe": probe, "status": status}[args.cmd](args)


if __name__ == "__main__":
    main()
