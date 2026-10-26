"""The factor study on OGB graphs: the shared protocol of run_benchmark.py, in mini-batch.

run_benchmark.py trains full-batch, which does not fit ogbn-arxiv or ogbn-products for every
method (GraphCL's similarity matrix and AFGRL's positive mining are quadratic in the number of
nodes). This runner keeps everything else of that protocol and changes only what the graph
size forces:

- the same method configs (``run_benchmark.build_model_config``: augmentations, EMA, GraphDINO's
  reference DINO values), encoder (hidden 256, 2 layers, batch norm, no dropout) and optimizer
  (AdamW lr 5e-4 / wd 1e-5, or ``--optimizer sgd``);
- NeighborLoader training with every node as a seed node (SSL never sees labels), ``--fanout``
  neighbours per hop, ``--batch`` seeds per step, the loss on the seed nodes only; the EMA
  schedules span the whole run (``--epochs`` x steps per epoch, ``--max-steps`` caps an epoch);
- evaluation of the full-graph embeddings: kNN and effective rank always, the linear probe
  with ``--linear-probe`` only (compare_probes.py scores every saved model with every probe).

``--model bgrl --epochs 0`` saves the untrained encoder of each seed, as in paper_runs.sh.
A failing (model, seed) is recorded and the queue goes on; the exit code is 1 if any failed.

Usage::

    # smoke test, minutes on a CPU
    python benchmarks/run_ogb.py --model all --seeds 1 --epochs 1 --max-steps 3 \\
        --out-dir /tmp/ogb_smoke --save-models /tmp/ogb_smoke/models
    # the study, inside tmux on the GPU machine
    python -u benchmarks/run_ogb.py --dataset ogbn-arxiv --model all --encoder gcn --seeds 3 \\
        --epochs 20 --out-dir benchmarks/paper/e1 --save-models benchmarks/paper/models

Results: ``<out-dir>/<dataset>/<model>__<timestamp>.json`` in run_benchmark.py's format;
models: ``<save-models>/<dataset>/<encoder>/<model>_seed<N>.pt``.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_geometric
from node_datasets import OGB, load_node_dataset
from run_benchmark import SSL_MODELS, _git_info, _mean_std, build_model_config, make_optimizer
from stress_ogbn_arxiv import LimitedLoader, StressMonitor, evaluate, fmt_eval

from graphssl.config.load import build_model, save_model
from graphssl.data import DataModule
from graphssl.evaluation import LogRegEvaluator
from graphssl.training import DINOTrainer
from graphssl.utils.positive_miner import HAS_FAISS

REPO_ROOT = Path(__file__).resolve().parent.parent
AGGREGATED_METRICS = ["test_acc_knn", "eff_rank", "test_acc_linear"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default="ogbn-arxiv", choices=OGB)
    p.add_argument("--model", nargs="+", default=["all"], choices=[*SSL_MODELS, "all"])
    p.add_argument("--encoder", default="gcn", choices=["gin", "gcn", "transformer"])
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20, help="passes over every node (0: untrained)")
    p.add_argument("--max-steps", type=int, default=None, help="cap the mini-batches per epoch")
    p.add_argument("--batch", type=int, default=1024, help="seed nodes per mini-batch")
    p.add_argument("--fanout", type=int, default=10, help="neighbours sampled per hop")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--first-seed", type=int, default=0, help="run seeds first-seed ... + seeds-1")
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--linear-probe", action="store_true", help="also fit the linear probe")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="benchmarks/paper/e1")
    p.add_argument("--save-models", default=None, metavar="DIR")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if "all" in args.model:
        args.model = SSL_MODELS
    return args


def make_config(model_name: str, args: argparse.Namespace, total_steps: int) -> dict:
    """run_benchmark.py's config, with the EMA schedules spanning every mini-batch step."""
    encoder_cfg = {
        "name": args.encoder,
        "hidden_dim": args.hidden,
        "num_layers": args.layers,
        "norm_type": "batch",
        "pool": False,  # node-level task: no graph pooling
        "drop": 0.0,
    }
    # build_model_config reads epochs (GraphDINO's per-epoch warm-up and freeze), hidden and
    # ema_tau; total_steps is per optimizer step.
    cfg = build_model_config(
        model_name, encoder_cfg, argparse.Namespace(epochs=args.epochs, hidden=args.hidden)
    )
    if "total_steps" in cfg:
        cfg["total_steps"] = total_steps
    return cfg


def run_seed(
    model_name: str, args: argparse.Namespace, dm: DataModule, num_classes: int, seed: int
) -> dict:
    assert dm.data is not None
    device = torch.device(args.device)
    loader = LimitedLoader(
        dm.neighbor_loader(num_neighbors=[args.fanout] * args.layers, shuffle=True),
        args.max_steps,
    )
    config = make_config(model_name, args, total_steps=args.epochs * len(loader))
    result: dict = {"seed": seed, "status": "failed", "steps_per_epoch": len(loader)}
    monitor = None
    t0 = time.time()
    try:
        if model_name == "afgrl" and not HAS_FAISS:
            raise ImportError("faiss is not installed (pip install faiss-cpu)")
        torch.manual_seed(seed)
        model = build_model(config, in_channels=dm.data.num_features)
        if args.epochs > 0:
            monitor = StressMonitor(eval_fn=dict, eval_every=0, num_epochs=args.epochs)
            optimizer = make_optimizer(model.student_parameters(), args)
            trainer = DINOTrainer(grad_clip_norm=None, device=device, callbacks=[monitor])
            trainer.train(model, loader, optimizer, num_epochs=args.epochs)
        result["wall_time_s"] = time.time() - t0
        if args.save_models is not None:
            checkpoint = (
                Path(args.save_models) / args.dataset / args.encoder / f"{model_name}_seed{seed}.pt"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            save_model(model, checkpoint, config, in_channels=dm.data.num_features)
            result["checkpoint"] = str(checkpoint)
        model.to(device)
        result.update(evaluate(model, dm, num_classes, device, args.knn_k, args.linear_probe))
        result["status"] = "ok"
        print(f"  seed {seed}: {fmt_eval(result)} | {result['wall_time_s']:.0f}s", flush=True)
    except Exception as e:  # recorded; the next (model, seed) still runs
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        print(f"  seed {seed}: FAILED: {result['error']}", flush=True)
    finally:
        if monitor is not None:
            result["epochs"] = monitor.epochs
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result["config"] = config
    return result


def save_result(args: argparse.Namespace, model_name: str, per_seed: list[dict]) -> Path:
    out_dir = REPO_ROOT / args.out_dir / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{model_name}__{timestamp}.json"
    ok = [r for r in per_seed if r["status"] == "ok"]
    probe = LogRegEvaluator()
    hyperparameters = {
        "encoder": args.encoder,
        "hidden_dim": args.hidden,
        "num_layers": args.layers,
        "epochs": args.epochs,
        "steps_per_epoch": per_seed[0]["steps_per_epoch"],
        "max_steps": args.max_steps,
        "batch": args.batch,
        "fanout": args.fanout,
        "input_nodes": "all",
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "knn_k": args.knn_k,
        "linear_probe": {
            "standardize": probe.standardize,
            "weight_decays": list(probe.weight_decays),
            "max_iter": probe.max_iter,
        }
        if args.linear_probe
        else None,
    }
    if args.optimizer == "sgd":
        hyperparameters.update(momentum=args.momentum, nesterov=True)
    result = {
        "dataset": args.dataset,
        "model": model_name,
        "hyperparameters": hyperparameters,
        "model_config": per_seed[0]["config"],
        "seeds": [r["seed"] for r in per_seed],
        "per_seed": [{k: v for k, v in r.items() if k != "config"} for r in per_seed],
        "aggregate": {
            m: _mean_std([r[m] for r in ok]) for m in AGGREGATED_METRICS if ok and m in ok[0]
        },
        "provenance": {
            "timestamp_utc": timestamp,
            "git": _git_info(),
            "torch_threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            "versions": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_geometric": torch_geometric.__version__,
            },
        },
    }
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def main() -> None:
    args = parse_args()
    data, num_classes, train_idx, val_idx, test_idx = load_node_dataset(args.dataset, args.data_dir)
    print(
        f"\n=== {args.dataset} | nodes={data.num_nodes:,} edges={data.num_edges:,} "
        f"features={data.num_features} classes={num_classes} | {args.device} ===",
        flush=True,
    )
    dm = DataModule(
        data=data,
        is_graph_level=False,
        batch_size=args.batch,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        num_workers=args.workers,
    )
    failed = False
    for model_name in args.model:
        print(f"\n--- {args.dataset} / {model_name} / {args.encoder} ({args.seeds} seeds) ---")
        seeds = range(args.first_seed, args.first_seed + args.seeds)
        per_seed = [run_seed(model_name, args, dm, num_classes, seed) for seed in seeds]
        path = save_result(args, model_name, per_seed)
        failed |= any(r["status"] != "ok" for r in per_seed)
        knn = [r["test_acc_knn"] for r in per_seed if r["status"] == "ok"]
        mean = f"{100 * sum(knn) / len(knn):.2f}" if knn else "nan"
        print(f"  -> knn {mean} | saved to {path}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
