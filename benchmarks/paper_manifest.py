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
PYTHON = MAIN / ".venv" / "bin" / "python"
SSL = ["dgi", "graphcl", "vicreg", "barlow_twins", "bgrl", "afgrl", "graphdino"]
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


@dataclass
class Queue:
    root: Path
    units: list[Unit] = field(default_factory=list)
    threads: int | None = 3
    device: str = "cpu"
    mac: bool = True  # --launch starts it here; the VM's queue is started by hand in tmux


def e1(datasets: list[str], encoder: str) -> list[Unit]:
    trained = [Unit("e1", d, encoder, m, save="models") for d in datasets for m in E1_MODELS]
    untrained = [
        Unit("e1_untrained", d, encoder, "bgrl", epochs=0, save="models_untrained")
        for d in datasets
    ]
    return trained + untrained


def e3(encoder: str) -> list[Unit]:
    return [
        Unit(f"e3/{opt}_lr{lr}", d, encoder, m, seeds=5, optimizer=opt, lr=lr)
        for opt, lr in E3_GRID
        for d in ("Cora", "CiteSeer")
        for m in SSL
    ]


AMAZON = ["AmazonPhoto", "AmazonComputers"]
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
}


def done(root: Path, u: Unit) -> bool:
    """Whether a result JSON of this unit exists (same backbone, budget, optimizer, lr, seeds)."""
    for path in (root / "benchmarks" / "paper" / u.out / u.dataset).glob(f"{u.model}__*.json"):
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
        key = (u.out, u.dataset, u.encoder, u.seeds, u.epochs, u.optimizer, u.lr, u.save)
        if groups and groups[-1][0] == key:
            groups[-1][1].append(u.model)
        else:
            groups.append((key, [u.model]))
    lines = []
    for (out, dataset, encoder, seeds, epochs, optimizer, lr, save), models in groups:
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
        print(f"queue {name}: {len(q.units) - len(rem)}/{len(q.units)} done ({q.root.name})")
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
    script.write_text("set -u\n" + "\n".join(lines) + f'\necho "QUEUE {name} DONE $(date)"\n')
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
