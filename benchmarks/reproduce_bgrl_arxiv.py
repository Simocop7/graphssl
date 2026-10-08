"""BGRL on ogbn-arxiv with the protocol of the BGRL paper: does the library reproduce it?

Thakoor et al., "Large-Scale Representation Learning on Graphs via Bootstrapping"
(ICLR 2022), Table 5: 72.53 ± 0.09 validation / 71.64 ± 0.12 test accuracy over 20 seeds,
and 69.90 ± 0.11 / 68.94 ± 0.15 for the same encoder left untrained ("Random-Init").

Unlike run_benchmark.py (one shared, untuned protocol for every method), this follows the
paper's own setup (its Appendix F and Table 8):

- symmetrized graph, full-graph training, 10,000 steps;
- encoder: 3 GCN layers of 256 units, each followed by layer normalization and PReLU, with
  weight standardization;
- predictor: MLP with one hidden layer of 256 units;
- augmentation: edges dropped with probability 0.6 in both views, no feature masking;
- AdamW, weight decay 1e-5, learning rate 1e-2 with 1,000 linear warm-up steps and a cosine
  decay to zero; target momentum 0.99 -> 1.0 on a cosine schedule.

Known differences from the paper, all recorded in the output:

- Evaluation. The paper L2-normalizes the embeddings and trains a linear classifier for 100
  AdamW steps (lr 0.01), selecting its weight decay on validation. Implemented literally,
  that classifier is far from fitted (44% test on the untrained encoder, where the paper
  reports 68.9%; 68.8% after 5,000 steps). This script uses the library's linear probe
  (``LogRegEvaluator``: regularised logistic regression fitted to convergence, strength
  selected on validation), which gives 69.5% there.
- The library's predictor has batch normalization and ReLU after its hidden layer; the
  paper describes a plain MLP.
- The paper averages 20 seeds.

The untrained encoder is evaluated first: it checks the encoder and the evaluation
independently of training, and the gain of BGRL over it (paper: +2.63 validation, +2.70
test) does not depend on how strong the linear classifier is.

Usage::

    pip install "graphssl[benchmark]"
    # smoke test: a few steps, no claim
    python benchmarks/reproduce_bgrl_arxiv.py --steps 20 --eval-every 10 --seeds 1 \\
        --skip-linear-probe --out-dir benchmarks/reproductions/smoke
    # the real run, inside tmux
    python -u benchmarks/reproduce_bgrl_arxiv.py --seeds 5 2>&1 | tee -a reproduce_bgrl.log

One JSON per invocation goes to ``--out-dir`` (default
``benchmarks/reproductions/bgrl_ogbn_arxiv``).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch_geometric
from stress_ogbn_arxiv import git_info, load_arxiv
from torch.optim import AdamW, Optimizer

from graphssl.config.load import build_model
from graphssl.core.callback import Callback
from graphssl.data import DataModule
from graphssl.evaluation import KNNEvaluator, LogRegEvaluator, effective_rank, extract_embeddings
from graphssl.training import DINOTrainer
from graphssl.utils.schedulers import CosineDecayScheduler

# Table 5 of the paper, accuracy in %: (validation, test) as mean and std over 20 seeds.
PAPER = {
    "bgrl": {"val": (72.53, 0.09), "test": (71.64, 0.12)},
    "random_init": {"val": (69.90, 0.11), "test": (68.94, 0.15)},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", type=int, default=5, help="the paper averages 20")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="linear learning-rate warm-up (default: a tenth of --steps, the paper's 1,000)",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=1_000,
        help="kNN accuracy and effective rank every N steps during training (0: never)",
    )
    p.add_argument(
        "--skip-linear-probe",
        action="store_true",
        help="no linear probe before/after training (about a minute per fit): smoke tests",
    )
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--data-dir", default="data/ogbn-arxiv")
    p.add_argument("--out-dir", default="benchmarks/reproductions/bgrl_ogbn_arxiv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.warmup_steps is None:
        args.warmup_steps = args.steps // 10
    if not 0 <= args.warmup_steps < args.steps:
        p.error("--warmup-steps must be in [0, --steps)")
    return args


def make_config(steps: int) -> dict:
    """Table 8 of the paper, ogbn-arXiv column."""
    return {
        "name": "bgrl",
        "encoder": {
            "name": "gcn",
            "hidden_dim": 256,
            "num_layers": 3,
            "norm_type": "layer",
            "weight_standardization": True,
            "pool": False,
        },
        "augment": [{"name": "edge_drop", "p": 0.6}],  # p_e = 0.6, p_f = 0 in both views
        "pred_hidden": 256,
        "ema_tau": 0.99,
        "ema_tau_end": 1.0,
        "total_steps": steps,
    }


def evaluate(
    model: torch.nn.Module,
    dm: DataModule,
    num_classes: int,
    args: argparse.Namespace,
    linear_probe: bool,
) -> Dict[str, Any]:
    """kNN accuracy and effective rank of the full-graph embeddings, plus the linear probe."""
    z, y = extract_embeddings(model, dm, device=args.device)
    assert y is not None
    knn = KNNEvaluator(k=args.knn_k).evaluate(z, y, dm.train_idx, dm.val_idx, dm.test_idx)
    result: Dict[str, Any] = {
        "val_acc_knn": knn["val_acc"],
        "test_acc_knn": knn["test_acc"],
        "eff_rank": effective_rank(z),
    }
    if linear_probe:
        lin = LogRegEvaluator().evaluate(
            z.to(args.device), y, dm.train_idx, dm.val_idx, dm.test_idx, num_classes=num_classes
        )
        result.update(
            val_acc_linear=lin["val_acc"],
            test_acc_linear=lin["test_acc"],
            probe_weight_decay=lin["weight_decay"],
            probe_converged=lin["converged"],
        )
    return result


def fmt(ev: Dict[str, Any]) -> str:
    line = f"knn test {ev['test_acc_knn'] * 100:.2f} | rank {ev['eff_rank']:.1f}"
    if "test_acc_linear" in ev:
        note = "" if ev["probe_converged"] else ", NOT converged"
        line = (
            f"linear val {ev['val_acc_linear'] * 100:.2f} test {ev['test_acc_linear'] * 100:.2f} "
            f"(L2 {ev['probe_weight_decay']:g}{note}) | {line}"
        )
    return line


class Schedule(Callback):
    """Sets the learning rate before every step and evaluates every ``eval_every`` steps.

    Full-graph training: one epoch of the trainer is one gradient step.
    """

    def __init__(self, optimizer: Optimizer, args: argparse.Namespace, eval_fn) -> None:
        self.optimizer = optimizer
        self.lr = CosineDecayScheduler(1e-2, 0.0, args.steps, args.warmup_steps)
        self.eval_every = args.eval_every
        self.steps = args.steps
        self.eval_fn = eval_fn
        self.curve: List[dict] = []
        self._t0 = time.time()

    def on_epoch_start(self, trainer: Any, model: torch.nn.Module, epoch: int) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr.get(epoch)

    def on_epoch_end(
        self, trainer: Any, model: torch.nn.Module, epoch: int, metrics: Dict[str, Any]
    ) -> None:
        step = epoch + 1
        if self.eval_every > 0 and step % self.eval_every == 0 and step < self.steps:
            ev = {"step": step, "loss": metrics["loss"], **self.eval_fn()}
            self.curve.append(ev)
            model.train()  # the evaluation put the model in eval mode
            print(
                f"  step {step:>6}/{self.steps}: loss {metrics['loss']:.4f} | {fmt(ev)} "
                f"| {time.time() - self._t0:.0f}s",
                flush=True,
            )


def run_seed(seed: int, args: argparse.Namespace, dm: DataModule, num_classes: int) -> dict:
    assert dm.data is not None
    device = torch.device(args.device)
    torch.manual_seed(seed)
    model = build_model(make_config(args.steps), in_channels=dm.data.num_features)

    probe = not args.skip_linear_probe
    result: Dict[str, Any] = {"seed": seed}
    result["random_init"] = evaluate(model, dm, num_classes, args, probe)
    print(f"  untrained: {fmt(result['random_init'])}", flush=True)

    optimizer = AdamW(model.student_parameters(), lr=0.0, weight_decay=1e-5)
    schedule = Schedule(
        optimizer, args, eval_fn=lambda: evaluate(model, dm, num_classes, args, False)
    )
    t0 = time.time()
    trainer = DINOTrainer(grad_clip_norm=None, device=device, callbacks=[schedule])
    losses = trainer.train(model, dm.train_dataloader(), optimizer, num_epochs=args.steps)
    result["train_time_s"] = time.time() - t0
    result["final_loss"] = losses[-1]
    result["curve"] = schedule.curve
    result["bgrl"] = evaluate(model, dm, num_classes, args, probe)
    print(f"  trained ({result['train_time_s']:.0f}s): {fmt(result['bgrl'])}", flush=True)
    return result


def summarize(per_seed: List[dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{row: {val/test: {mean, std}}} of the linear-probe accuracy in %, with BGRL's gain."""

    def stats(values: List[float]) -> Dict[str, float]:
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return {"mean": statistics.mean(values), "std": std}

    summary: Dict[str, Dict[str, Dict[str, float]]] = {"random_init": {}, "bgrl": {}, "gain": {}}
    for split in ("val", "test"):
        key = f"{split}_acc_linear"
        untrained = [r["random_init"][key] * 100 for r in per_seed]
        trained = [r["bgrl"][key] * 100 for r in per_seed]
        summary["random_init"][split] = stats(untrained)
        summary["bgrl"][split] = stats(trained)
        summary["gain"][split] = stats([t - u for t, u in zip(trained, untrained, strict=True)])
    return summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, train_idx, val_idx, test_idx, num_classes = load_arxiv(args.data_dir)
    print(
        f"\nogbn-arxiv | nodes={data.num_nodes:,} edges={data.num_edges:,} | {args.device} "
        f"| {args.steps} steps, {args.seeds} seed(s)\n",
        flush=True,
    )
    dm = DataModule(
        data=data, is_graph_level=False, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    per_seed = []
    for seed in range(args.seeds):
        print(f"--- seed {seed} ---", flush=True)
        per_seed.append(run_seed(seed, args, dm, num_classes))

    summary = None if args.skip_linear_probe else summarize(per_seed)
    probe = LogRegEvaluator()
    report = {
        "dataset": "ogbn-arxiv",
        "model": "bgrl",
        "protocol": "Thakoor et al., ICLR 2022, Appendix F and Table 8",
        "paper": PAPER,
        "summary": summary,
        "model_config": make_config(args.steps),
        "hyperparameters": {
            **vars(args),
            "lr": 1e-2,
            "weight_decay": 1e-5,
            "linear_probe": {
                "standardize": probe.standardize,
                "weight_decays": list(probe.weight_decays),
                "max_iter": probe.max_iter,
            },
        },
        "differences_from_paper": [
            "evaluation: LogRegEvaluator, fitted to convergence (paper: 100 AdamW steps with "
            "a weight-decay search, on L2-normalized embeddings)",
            "predictor: Linear - BatchNorm - ReLU - Linear (paper: MLP with one hidden layer)",
            f"{args.seeds} seed(s) (paper: 20)",
        ],
        "per_seed": per_seed,
        "provenance": {
            "timestamp_utc": timestamp,
            "git": git_info(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        },
    }
    path = out_dir / f"bgrl__{timestamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")

    if summary is not None:
        print("\n| | Validation | Test | Paper validation | Paper test |\n|---|---|---|---|---|")
        for row, label in (("random_init", "Untrained encoder"), ("bgrl", "BGRL")):
            ours, paper = summary[row], PAPER[row]
            print(
                f"| {label} | {ours['val']['mean']:.2f} ± {ours['val']['std']:.2f} "
                f"| {ours['test']['mean']:.2f} ± {ours['test']['std']:.2f} "
                f"| {paper['val'][0]:.2f} ± {paper['val'][1]:.2f} "
                f"| {paper['test'][0]:.2f} ± {paper['test'][1]:.2f} |"
            )
        gain = summary["gain"]
        print(
            f"| BGRL − untrained | {gain['val']['mean']:+.2f} ± {gain['val']['std']:.2f} "
            f"| {gain['test']['mean']:+.2f} ± {gain['test']['std']:.2f} "
            f"| {PAPER['bgrl']['val'][0] - PAPER['random_init']['val'][0]:+.2f} "
            f"| {PAPER['bgrl']['test'][0] - PAPER['random_init']['test'][0]:+.2f} |"
        )
    print(f"\n{args.seeds} seed(s) (20 in the paper). Saved to {path}")


if __name__ == "__main__":
    main()
