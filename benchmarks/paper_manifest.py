"""Every planned full-batch run of the paper, what is done, and how to resume the rest.

A run unit is one (dataset, backbone, model, configuration) of run_benchmark.py: it is done when
its result JSON exists (results are written once all seeds of a model have finished; the
seeds of a model cut short are rerun from the start, and on the same machine and thread count
they reproduce bit for bit). The queues are those launched on 26 Oct (benchmarks/paper/PLAN.md):

  A  E1 Amazon Photo / Computers with GCN and GIN
  B  E1 Cora / CiteSeer / PubMed with the Transformer
  T  E1 Amazon Photo / Computers with the Transformer (split off A on 26 Oct, to run in parallel
     with it once B or P frees a CPU slot)
  M  E1 Coauthor CS with every backbone on the Mac GPU (mps), in its own worktree
     graphssl-runs-mps (moved off A and B on 26 Oct: MPS runs a seed 3.5-4x faster than the
     Mac CPU and uses almost no CPU; the CPU runs A had made are in superseded/)
  P  E1 PubMed with GIN (the paper_runs.sh job in the main tree)
  V  E3 on the VM GPU, every cell (AdamW 5e-4 included), in the VM's tmux. Moved off queue B
     on 26 Oct: the VM's GPU runs a seed about 5x faster than the Mac, its CPU 5x slower.
     Its results are copied back into the main tree; ``--launch`` refuses it (Mac only).
  E4-gcn, E4-gin, E4-transformer
     E4, training trajectories (added 27 Oct): Cora / CiteSeer, 5 seeds, three optimizer cells,
     evaluated at 14 steps of the 300 (``run_benchmark.py --trajectory``). One stream per
     backbone on the Mac CPU, in the worktree graphssl-runs-e4.
  G-planetoid, G-amazon, G-cs, G-E4, G-V
     GraphMAE, the eighth objective (added 28 Oct, PLAN.md "GraphMAE and E5"), in the worktree
     graphssl-runs-graphmae at the commit that adds it: E1 on the Mac CPU (Planetoid, Amazon) and
     GPU (Coauthor CS), its E4 trajectories on the Mac CPU, its E3 cells on the VM (G-V).
  E5-mps, E5-V
     E5, the training budget (added 28 Oct): complete runs of 300 and 3,000 steps, 5 seeds, the
     eight objectives; the 3,000-step runs evaluated at E5_STEPS. GCN on Cora, Amazon Photo
     (Mac GPU) and Amazon Computers, Coauthor CS (VM GPU); GIN and Transformer on Cora as well.
     Per the cost rule of PLAN.md, an objective above 45 minutes per 3,000-step seed on its
     device runs seeds 0-2 (``E5_THREE_SEEDS``, filled in from measured costs). E5-mps moved to
     the worktree graphssl-runs-e5 on 28 Oct: at the GraphMAE commit every --trajectory run on a
     GPU crashed after step 0 (the trainer moves the graph to the device in place), so the
     3,000-step half is rerun at the commit that fixes it; the 300-step JSONs are copied in.
  E3X-amazon, E3X-cs
     E3 extended to GCN on Amazon Photo / Computers and Coauthor CS (added 29 Oct, PLAN.md
     "E3 on the Amazon and Coauthor graphs"): the same ten cells, 5 seeds, the eight
     objectives, each dataset on the device of its E1 runs (Amazon: Mac CPU, 3 threads; Coauthor
     CS: Mac GPU, 2 threads), in the worktree graphssl-runs-e3x. One dataset at a time, every
     cell of it, so each dataset's result is complete before the next one starts.

The launched script logs ``QUEUE STEP FAILED`` for every command that exits non-zero and ends
with ``QUEUE <name> DONE <date>, <n> failed``: "DONE" alone does not mean every step succeeded.

Usage::

    python benchmarks/paper_manifest.py status                 # done / remaining, every queue
    python benchmarks/paper_manifest.py resume --queue A       # print the commands that remain
    python benchmarks/paper_manifest.py resume --queue A --launch   # run them, detached
    python benchmarks/paper_manifest.py resume --queue all --launch # after a restart: everything

``--launch`` skips a queue while one of its run_benchmark.py processes is running (``--force``
starts a second copy: only for moving work between queues on purpose). Meant for after a
restart; while a queue is between two commands it can look idle for a moment.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

MAIN = Path(__file__).resolve().parent.parent
RUNS = MAIN.parent / "graphssl-runs"  # the worktree the v2 runs execute in
RUNS_MPS = MAIN.parent / "graphssl-runs-mps"  # the worktree of the Mac GPU runs
RUNS_E4 = MAIN.parent / "graphssl-runs-e4"  # the worktree of the trajectory runs
RUNS_G = MAIN.parent / "graphssl-runs-graphmae"  # GraphMAE and E5, from 28 Oct
RUNS_E5 = MAIN.parent / "graphssl-runs-e5"  # E5-mps from 28 Oct evening (trajectory fix)
RUNS_E3X = MAIN.parent / "graphssl-runs-e3x"  # E3 on Amazon / Coauthor CS, from 29 Oct
PYTHON = MAIN / ".venv" / "bin" / "python"
# The seven objectives of the queues launched on 26-27 Oct. Never add GraphMAE here: those
# queues' worktrees sit at commits without it, and --launch would start units they cannot run.
SSL = ["dgi", "graphcl", "vicreg", "barlow_twins", "bgrl", "afgrl", "graphdino"]
SSL8 = [*SSL, "graphmae"]
E1_MODELS = [*SSL, "supervised", "supervised_reg"]
E3_GRID = [
    ("adamw", "5e-4"),
    ("adamw", "1e-4"),
    ("adamw", "1e-3"),
    ("adamw", "5e-3"),
    ("adamw", "1e-2"),
    ("sgd", "1e-3"),
    ("sgd", "5e-3"),
    ("sgd", "1e-2"),
    ("sgd", "5e-2"),
    ("sgd", "1e-1"),
]

# E4: the shared protocol, the high-lr AdamW cell where E3's collapses are, and the SGD cell
# that changes E3's ranking most broadly without diverging.
E4_CELLS = [("adamw", "5e-4"), ("adamw", "1e-2"), ("sgd", "1e-2")]
E4_STEPS = ("1", "2", "5", "10", "20", "30", "50", "75", "100", "150", "200", "250", "300")
# E5: the 3,000-step runs are evaluated here (step 0, the untrained encoder, always is).
E5_STEPS = ("300", "600", "1000", "2000", "3000")
# (dataset, objective) pairs that run seeds 0-2 in E5 under PLAN.md's cost rule (> 45 min per
# 3,000-step seed on the dataset's E5 device), from the costs measured before launch.
# from logs/e5v_costs.csv (A2, 30 Oct): over 45 min per 3,000-step seed (71.4 and 49.4 min)
E5_THREE_SEEDS: set[tuple[str, str]] = {("CoauthorCS", "graphcl"), ("CoauthorCS", "afgrl")}


@dataclass(frozen=True)
class Unit:
    out: str  # results directory under benchmarks/paper
    dataset: str
    encoder: str
    model: str
    seeds: int = 10
    epochs: int = 300
    optimizer: str = "adamw"
    lr: str = "5e-4"
    save: str | None = None  # models directory under benchmarks/paper
    trajectory: tuple[str, ...] = ()  # steps evaluated during training (--trajectory)


@dataclass
class Queue:
    root: Path
    units: list[Unit] = field(default_factory=list)
    threads: int | None = 3
    device: str = "cpu"
    mac: bool = True  # --launch starts it here; the VM's queue is started by hand in tmux


def e1(datasets: list[str], encoder: str, models: list[str] = E1_MODELS) -> list[Unit]:
    trained = [Unit("e1", d, encoder, m, save="models") for d in datasets for m in models]
    untrained = [
        Unit("e1_untrained", d, encoder, "bgrl", epochs=0, save="models_untrained")
        for d in datasets
    ]
    return trained + untrained


def e4(encoder: str, models: list[str] = SSL) -> list[Unit]:
    return [
        Unit(f"e4/{opt}_lr{lr}", d, encoder, m, seeds=5, optimizer=opt, lr=lr, trajectory=E4_STEPS)
        for opt, lr in E4_CELLS
        for d in ("Cora", "CiteSeer")
        for m in models
    ]


def e3(encoder: str, models: list[str] = SSL) -> list[Unit]:
    return [
        Unit(f"e3/{opt}_lr{lr}", d, encoder, m, seeds=5, optimizer=opt, lr=lr)
        for opt, lr in E3_GRID
        for d in ("Cora", "CiteSeer")
        for m in models
    ]


def e3x(datasets: list[str]) -> list[Unit]:
    """E3's grid with GCN on other datasets, dataset by dataset (every cell of one first)."""
    return [
        Unit(f"e3/{opt}_lr{lr}", d, "gcn", m, seeds=5, optimizer=opt, lr=lr)
        for d in datasets
        for opt, lr in E3_GRID
        for m in SSL8
    ]


def e5(datasets: list[str], encoder: str) -> list[Unit]:
    """Both budgets of E5 for every objective, the 300-step runs first."""
    return [
        Unit(
            f"e5/steps{steps}",
            d,
            encoder,
            m,
            seeds=3 if (d, m) in E5_THREE_SEEDS else 5,
            epochs=steps,
            save=f"models_e5/steps{steps}",
            trajectory=E5_STEPS if steps == 3000 else (),
        )
        for steps in (300, 3000)
        for d in datasets
        for m in SSL8
    ]


AMAZON = ["AmazonPhoto", "AmazonComputers"]
PLANETOID = ["Cora", "CiteSeer", "PubMed"]
BACKBONES = ["gcn", "gin", "transformer"]
QUEUES = {
    "A": Queue(RUNS, e1(AMAZON, "gcn") + e1(AMAZON, "gin")),
    "B": Queue(RUNS, e1(["Cora", "CiteSeer", "PubMed"], "transformer")),
    "T": Queue(RUNS, e1(AMAZON, "transformer")),
    "M": Queue(
        RUNS_MPS,
        e1(["CoauthorCS"], "gcn") + e1(["CoauthorCS"], "gin") + e1(["CoauthorCS"], "transformer"),
        threads=2,
        device="mps",
    ),
    "P": Queue(MAIN, e1(["PubMed"], "gin"), threads=None),
    "V": Queue(MAIN, e3("gcn") + e3("gin") + e3("transformer"), device="cuda", mac=False),
    **{f"E4-{e}": Queue(RUNS_E4, e4(e), threads=2) for e in ("gcn", "gin", "transformer")},
    "G-planetoid": Queue(
        RUNS_G, [u for e in BACKBONES for u in e1(PLANETOID, e, ["graphmae"]) if u.epochs]
    ),
    "G-amazon": Queue(
        RUNS_G, [u for e in BACKBONES for u in e1(AMAZON, e, ["graphmae"]) if u.epochs]
    ),
    "G-cs": Queue(
        RUNS_G,
        [u for e in BACKBONES for u in e1(["CoauthorCS"], e, ["graphmae"]) if u.epochs],
        threads=2,
        device="mps",
    ),
    "G-E4": Queue(RUNS_G, [u for e in BACKBONES for u in e4(e, ["graphmae"])], threads=2),
    "G-V": Queue(
        MAIN, [u for e in BACKBONES for u in e3(e, ["graphmae"])], device="cuda", mac=False
    ),
    "E5-mps": Queue(
        RUNS_E5,
        e5(["Cora", "AmazonPhoto"], "gcn") + e5(["Cora"], "gin") + e5(["Cora"], "transformer"),
        threads=2,
        device="mps",
    ),
    "E5-V": Queue(MAIN, e5(["AmazonComputers", "CoauthorCS"], "gcn"), device="cuda", mac=False),
    "E3X-amazon": Queue(RUNS_E3X, e3x(["AmazonPhoto", "AmazonComputers"])),
    "E3X-cs": Queue(RUNS_E3X, e3x(["CoauthorCS"]), threads=2, device="mps"),
}


def results_root(root: Path) -> Path:
    """A queue's worktree or, once the worktree is removed, the main tree: every result was
    copied back and checked before the run worktrees were removed (31 Oct 2026)."""
    return root if root.exists() else MAIN


def done(root: Path, u: Unit) -> bool:
    """Whether a result JSON of this unit exists (same backbone, budget, optimizer, lr, seeds)."""
    base = results_root(root) / "benchmarks" / "paper" / u.out / u.dataset
    for path in base.glob(f"{u.model}__*.json"):
        r = json.loads(path.read_text())
        h = r["hyperparameters"]
        # supervised_reg records its own recipe (Adam, lr 0.01) in place of the shared one.
        recipe = u.model == "supervised_reg" or (
            h.get("optimizer", "adamw") == u.optimizer and float(h["lr"]) == float(u.lr)
        )
        if (
            h["encoder"] == u.encoder
            and h["epochs"] == u.epochs
            and recipe
            and len(r["seeds"]) == u.seeds
        ):
            return True
    return False


def commands(q: Queue, remaining: list[Unit]) -> list[str]:
    """run_benchmark.py commands for the remaining units, in queue order, models grouped."""
    groups: list[tuple[tuple, list[str]]] = []
    for u in remaining:
        key = (u.out, u.dataset, u.encoder, u.seeds, u.epochs, u.optimizer, u.lr, u.save,
               u.trajectory)  # fmt: skip
        if groups and groups[-1][0] == key:
            groups[-1][1].append(u.model)
        else:
            groups.append((key, [u.model]))
    lines = []
    for (out, dataset, encoder, seeds, epochs, optimizer, lr, save, trajectory), models in groups:
        cmd = [
            "python", "-u", "benchmarks/run_benchmark.py", "--dataset", dataset,
            "--model", *models, "--encoder", encoder, "--seeds", str(seeds),
            "--out-dir", f"benchmarks/paper/{out}", "--device", q.device,
        ]  # fmt: skip
        if epochs != 300:
            cmd += ["--epochs", str(epochs)]
        if (optimizer, lr) != ("adamw", "5e-4"):
            cmd += ["--optimizer", optimizer, "--lr", lr]
        if save:
            cmd += ["--save-models", f"benchmarks/paper/{save}"]
        if trajectory:
            cmd += ["--trajectory", *trajectory]
        lines.append(shlex.join(cmd))
    return lines


def running_for(q: Queue) -> list[str]:
    """run_benchmark.py processes in the queue's tree that run one of its units: a process
    belongs to a queue when one of its datasets, its backbone and its results directory do."""
    mine = {(u.dataset, u.encoder, f"benchmarks/paper/{u.out}") for u in q.units}
    out = subprocess.run(["pgrep", "-f", "run_benchmark.py"], capture_output=True, text=True)
    found = []
    for pid in out.stdout.split():
        cwd = subprocess.run(
            ["lsof", "-a", "-p", pid, "-d", "cwd", "-Fn"], capture_output=True, text=True
        ).stdout
        if f"n{q.root}" not in cwd.splitlines():
            continue
        argv = subprocess.run(
            ["ps", "-o", "command=", "-p", pid], capture_output=True, text=True
        ).stdout.split()

        def values(flag: str, argv: list[str] = argv) -> list[str]:
            if flag not in argv:
                return []
            rest = argv[argv.index(flag) + 1 :]
            return rest[: next((i for i, a in enumerate(rest) if a.startswith("--")), len(rest))]

        encoder = (values("--encoder") or ["gin"])[0]
        out_dir = (values("--out-dir") or ["benchmarks/results"])[0]
        if any((d, encoder, out_dir) in mine for d in values("--dataset")):
            found.append(pid)
    return found


def status() -> None:
    for name, q in QUEUES.items():
        rem = [u for u in q.units if not done(q.root, u)]
        where = q.root.name if q.root.exists() else f"{q.root.name} removed: main tree"
        print(f"queue {name}: {len(q.units) - len(rem)}/{len(q.units)} done ({where})")
        seen: dict = {}
        for u in q.units:
            key = (u.out.split("/")[0], u.dataset, u.encoder)
            d, n = seen.get(key, (0, 0))
            seen[key] = (d + done(q.root, u), n + 1)
        for (out, dataset, encoder), (d, n) in seen.items():
            mark = "done" if d == n else ("-" if d == 0 else "partly")
            print(f"  {out:13s} {dataset:16s} {encoder:11s} {d:3d}/{n:<3d} {mark}")


def resume(name: str, launch: bool, force: bool) -> None:
    q = QUEUES[name]
    rem = [u for u in q.units if not done(q.root, u)]
    lines = commands(q, rem)
    print(f"queue {name}: {len(rem)} of {len(q.units)} units remain, {len(lines)} commands")
    for line in lines:
        print("  " + line)
    if not launch or not lines:
        return
    if not q.root.exists():
        raise SystemExit(
            f"queue {name}: its worktree {q.root} was removed; recreate it with "
            f"`git worktree add --detach {q.root} <commit>` before launching"
        )
    if not q.mac:
        raise SystemExit(f"queue {name} runs on the VM: start its commands in the VM's tmux")
    busy = running_for(q)
    if busy and not force:
        raise SystemExit(
            f"queue {name} is still running (pids {' '.join(busy)}): not launching. "
            "Pass --force only to start a second copy on purpose."
        )
    logs = q.root / "benchmarks" / "paper" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    script = logs / f"resume_{name}.sh"
    steps = [
        f'{line} || {{ echo "QUEUE STEP FAILED (exit $?): {line}"; failed=$((failed + 1)); }}'
        for line in lines
    ]
    script.write_text(
        "set -u\nfailed=0\n"
        + "\n".join(steps)
        + f'\necho "QUEUE {name} DONE $(date), $failed failed"\n'
    )
    env = dict(os.environ, PATH=f"{PYTHON.parent}:{os.environ['PATH']}")
    if q.root != MAIN:
        env["PYTHONPATH"] = str(q.root / "src")
    if q.threads:
        env["OMP_NUM_THREADS"] = str(q.threads)
    log = open(logs / f"resume_{name}.log", "a")  # noqa: SIM115 - kept open by the child
    subprocess.Popen(
        ["nohup", "caffeinate", "-is", "bash", str(script)],
        cwd=q.root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"launched in {q.root}, log {logs / f'resume_{name}.log'}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("resume")
    r.add_argument("--queue", required=True, choices=[*sorted(QUEUES), "all"])
    r.add_argument("--launch", action="store_true")
    r.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.cmd == "status":
        status()
        return
    for name in sorted(QUEUES) if args.queue == "all" else [args.queue]:
        try:
            resume(name, args.launch, args.force)
        except SystemExit as e:  # one queue still running does not stop the others
            print(e)


if __name__ == "__main__":
    main()
