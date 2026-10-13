"""Score the same frozen embeddings with the linear-evaluation protocols of other codebases.

"Linear evaluation" is not one protocol: public graph-SSL codebases fit the classifier in
different ways, and their numbers are compared as if they did not. This script takes models
saved with ``save_model()``, extracts their full-graph embeddings once, and scores them with
four protocols:

``adam100``    The reference code of DGI (``PetarV-/DGI``, ``execute.py``) and the
               ``LogisticRegression`` evaluator of PyG-SSL (its defaults, and the values
               hard-coded in ``examples/example_BGRL.py``). A linear layer with a
               Xavier-uniform weight and a zero bias, 100 full-batch Adam steps (lr 0.01, no
               weight decay) on the raw embeddings. Test accuracy after the last step,
               averaged over random initialisations.
``adam5000``   ``LRTrainableEvaluator`` of PyGCL. The same layer (bias left at PyTorch's
               default) and optimizer for 5,000 steps; validation every 20 steps, test
               accuracy at the best validation accuracy.
``liblinear``  The BGRL reference implementation (``nerdslab/bgrl``,
               ``fit_logistic_regression_preset_splits``). Rows L2-normalised, one-vs-rest
               logistic regression with liblinear, C in 2^-10 ... 2^10 selected on
               validation. Needs scikit-learn. 21 x num_classes fits: slow on a large graph,
               so it is not run unless asked for.
``graphssl``   This library's ``LogRegEvaluator``: standardized features, L-BFGS to
               convergence from a zero initialisation, L2 strength selected on validation.

The first three are re-implementations written from the sources named above, as read on
2026-10-13; they are not imports of those packages.

Usage::

    # the seven models of the ogbn-arxiv stress test, and the same encoders untrained
    python -u benchmarks/compare_probes.py --dataset ogbn-arxiv --untrained \\
        --checkpoints benchmarks/stress/ogbn_arxiv_20261012/*.pt \\
        --out-dir benchmarks/probes/ogbn_arxiv_stress

    # one protocol, more repetitions
    python benchmarks/compare_probes.py --dataset ogbn-arxiv --protocols adam100 \\
        --repeats 50 --checkpoints benchmarks/reproductions/bgrl_ogbn_arxiv/*.pt

One JSON (every repetition of every protocol, kNN accuracy, effective rank, provenance) and
a ``summary.md`` table go to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch.optim import Adam

from graphssl.config.load import build_model, load_model
from graphssl.data import DataModule
from graphssl.evaluation import KNNEvaluator, LogRegEvaluator, effective_rank, extract_embeddings

REPO_ROOT = Path(__file__).resolve().parent.parent
PLANETOID = ["Cora", "CiteSeer", "PubMed"]
PROTOCOLS = ["adam100", "adam5000", "liblinear", "graphssl"]
DEFAULT_PROTOCOLS = ["adam100", "adam5000", "graphssl"]
# What each protocol stands for, printed under the table.
SOURCES = {
    "adam100": "DGI reference code, PyG-SSL: 100 Adam steps, test at the last step",
    "adam5000": "PyGCL: 5,000 Adam steps, test at the best validation",
    "liblinear": "BGRL reference implementation: liblinear on L2-normalised rows, C on validation",
    "graphssl": "GraphSSL: standardized features, L-BFGS to convergence, L2 on validation",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default="ogbn-arxiv", choices=["ogbn-arxiv", *PLANETOID])
    p.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="files written by save_model(); each row is named <parent dir>/<file stem>",
    )
    p.add_argument(
        "--untrained",
        action="store_true",
        help="also score each distinct architecture at initialisation (--untrained-seed)",
    )
    p.add_argument("--untrained-seed", type=int, default=0)
    p.add_argument("--protocols", nargs="+", default=DEFAULT_PROTOCOLS, choices=PROTOCOLS)
    p.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="random initialisations of adam100 (its sources use 20 to 50)",
    )
    p.add_argument("--repeats-long", type=int, default=3, help="initialisations of adam5000")
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="benchmarks/probes/run")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_graph(name: str, data_dir: str):
    """(data, train_idx, val_idx, test_idx, num_classes), as the benchmark scripts load it."""
    if name in PLANETOID:
        from torch_geometric.datasets import Planetoid

        dataset = Planetoid(root=f"{data_dir}/{name}", name=name)
        data = dataset[0]
        idx = [
            m.nonzero(as_tuple=True)[0] for m in (data.train_mask, data.val_mask, data.test_mask)
        ]
        return data, idx[0], idx[1], idx[2], dataset.num_classes

    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as e:
        raise ImportError('Install ogb: pip install "graphssl[benchmark]"') from e
    from torch_geometric.transforms import ToUndirected

    # ogb (<= 1.3.6) loads its processed file with a bare torch.load(); since PyTorch 2.6
    # that defaults to weights_only=True and rejects PyG's Data classes unless allow-listed.
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage

        add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

    dataset = PygNodePropPredDataset(name=name, root=f"{data_dir}/{name}")
    data = ToUndirected()(dataset[0])  # as in the stress test and the BGRL reproduction
    split = dataset.get_idx_split()
    return data, split["train"], split["valid"], split["test"], dataset.num_classes


# ---------------------------------------------------------------------------
# Protocols. Each returns {"val_acc", "test_acc", ...} for one fit.
# ---------------------------------------------------------------------------


def _accuracy(logits: Tensor, y: Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def adam_probe(
    z: Tensor,
    y: Tensor,
    train_idx: Tensor,
    val_idx: Tensor,
    test_idx: Tensor,
    num_classes: int,
    seed: int,
    steps: int,
    best_validation: bool,
) -> Dict[str, float]:
    """A linear layer trained full-batch with Adam (lr 0.01, no weight decay) on raw embeddings.

    ``best_validation=False`` is DGI / PyG-SSL: zero bias, accuracy after the last step.
    ``best_validation=True`` is PyGCL: default bias, validation every 20 steps, the test
    accuracy of the step with the best validation accuracy (the first one on ties).
    """
    torch.manual_seed(seed)
    clf = torch.nn.Linear(z.size(1), num_classes).to(z.device)
    torch.nn.init.xavier_uniform_(clf.weight)
    if not best_validation:
        torch.nn.init.zeros_(clf.bias)
    optimizer = Adam(clf.parameters(), lr=0.01, weight_decay=0.0)
    z_train, y_train = z[train_idx], y[train_idx]
    best_val, test_at_best = 0.0, None
    loss = torch.zeros(())
    for step in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(clf(z_train), y_train)
        loss.backward()
        optimizer.step()
        if best_validation and (step + 1) % 20 == 0:
            with torch.no_grad():
                val = _accuracy(clf(z[val_idx]), y[val_idx])
                if val > best_val:
                    best_val, test_at_best = val, _accuracy(clf(z[test_idx]), y[test_idx])
    with torch.no_grad():
        last_val = _accuracy(clf(z[val_idx]), y[val_idx])
        last_test = _accuracy(clf(z[test_idx]), y[test_idx])
    if best_validation and test_at_best is not None:
        return {"val_acc": best_val, "test_acc": test_at_best, "train_loss": loss.item()}
    return {"val_acc": last_val, "test_acc": last_test, "train_loss": loss.item()}


def liblinear_probe(
    z: Tensor, y: Tensor, train_idx: Tensor, val_idx: Tensor, test_idx: Tensor
) -> Dict[str, float]:
    """One-vs-rest liblinear on L2-normalised rows, C in 2^-10 ... 2^10 selected on validation."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import normalize
    except ImportError as e:
        raise ImportError("the liblinear protocol needs scikit-learn") from e

    x = normalize(z.cpu().numpy(), norm="l2")
    labels = y.cpu().numpy()
    tr, va, te = (i.cpu().numpy() for i in (train_idx, val_idx, test_idx))
    best_val, best_test, best_c = 0.0, 0.0, float("nan")
    for c in 2.0 ** np.arange(-10, 11):
        clf = OneVsRestClassifier(LogisticRegression(solver="liblinear", C=c), n_jobs=-1)
        clf.fit(x[tr], labels[tr])
        classes = clf.classes_
        val = float((classes[clf.predict_proba(x[va]).argmax(axis=1)] == labels[va]).mean())
        if val > best_val:
            test = float((classes[clf.predict_proba(x[te]).argmax(axis=1)] == labels[te]).mean())
            best_val, best_test, best_c = val, test, float(c)
    return {"val_acc": best_val, "test_acc": best_test, "C": best_c}


def run_protocol(
    name: str,
    z: Tensor,
    y: Tensor,
    train_idx: Tensor,
    val_idx: Tensor,
    test_idx: Tensor,
    num_classes: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Every fit of one protocol on one set of embeddings, with the time it took."""
    splits = (train_idx, val_idx, test_idx)
    t0 = time.time()
    fits: List[Dict[str, float]]
    if name == "adam100":
        fits = [
            adam_probe(z, y, *splits, num_classes, seed, steps=100, best_validation=False)
            for seed in range(args.repeats)
        ]
    elif name == "adam5000":
        fits = [
            adam_probe(z, y, *splits, num_classes, seed, steps=5000, best_validation=True)
            for seed in range(args.repeats_long)
        ]
    elif name == "liblinear":
        fits = [liblinear_probe(z, y, *splits)]
    else:
        lin = LogRegEvaluator().evaluate(z, y, *splits, num_classes=num_classes)
        fits = [
            {
                "val_acc": lin["val_acc"],
                "test_acc": lin["test_acc"],
                "weight_decay": lin["weight_decay"],
                "converged": lin["converged"],
            }
        ]
    test = [f["test_acc"] for f in fits]
    return {
        "fits": fits,
        "test_acc": {
            "mean": statistics.mean(test),
            "std": statistics.stdev(test) if len(test) > 1 else 0.0,
            "min": min(test),
            "max": max(test),
        },
        "time_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def row_name(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def collect_models(args: argparse.Namespace) -> List[tuple]:
    """(row name, zero-argument builder of the model), trained rows first."""
    rows: List[tuple] = []
    untrained: Dict[str, tuple] = {}
    for raw in args.checkpoints:
        path = Path(raw)
        rows.append((row_name(path), lambda path=path: load_model(path)))
        if args.untrained:
            checkpoint = torch.load(path, map_location="cpu")
            encoder = checkpoint["config"]["encoder"]
            key = json.dumps(encoder, sort_keys=True)
            if key not in untrained:
                label = f"untrained {encoder['name']}-{encoder['num_layers']}L ({row_name(path)})"

                def build(checkpoint: dict = checkpoint) -> torch.nn.Module:
                    torch.manual_seed(args.untrained_seed)
                    return build_model(
                        checkpoint["config"], checkpoint["in_channels"], checkpoint["num_classes"]
                    )

                untrained[key] = (label, build)
    return [*untrained.values(), *rows]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _git_info() -> dict:
    def _run(cmd: List[str]) -> Optional[str]:
        try:
            return subprocess.run(
                cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def _cell(result: Dict[str, Any]) -> str:
    acc = result["test_acc"]
    if len(result["fits"]) == 1:
        return f"{100 * acc['mean']:.2f}"
    return f"{100 * acc['mean']:.2f} ± {100 * acc['std']:.2f} ({100 * acc['min']:.1f}–{100 * acc['max']:.1f})"


def render_summary(rows: List[dict], protocols: List[str], args: argparse.Namespace) -> str:
    lines = [
        "| Embeddings | " + " | ".join(protocols) + " | kNN | Rank |",
        "|---|" + "---|" * (len(protocols) + 2),
    ]
    for r in rows:
        cells = " | ".join(_cell(r["protocols"][p]) for p in protocols)
        lines.append(
            f"| {r['name']} | {cells} | {100 * r['test_acc_knn']:.2f} | {r['eff_rank']:.1f} |"
        )
    lines += [
        "",
        f"Test accuracy (%) on {args.dataset}. A protocol fitted more than once reports the "
        "mean ± sample std over its random initialisations and, in brackets, the range.",
        "",
    ]
    lines += [f"- `{p}`: {SOURCES[p]}." for p in protocols]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    data, train_idx, val_idx, test_idx, num_classes = load_graph(args.dataset, args.data_dir)
    dm = DataModule(
        data=data, is_graph_level=False, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
    )
    print(
        f"{args.dataset} | nodes={data.num_nodes:,} train={len(train_idx):,} "
        f"classes={num_classes} | {device}",
        flush=True,
    )

    rows: List[dict] = []
    build: Callable[[], torch.nn.Module]
    for name, build in collect_models(args):
        z, y = extract_embeddings(build(), dm, device=str(device))
        assert y is not None
        if not torch.isfinite(z).all():
            raise FloatingPointError(f"non-finite values in the embeddings of {name}")
        y = y.view(-1).long()  # OGB labels are [N, 1]
        knn = KNNEvaluator(k=args.knn_k).evaluate(z, y, train_idx, val_idx, test_idx)
        row: Dict[str, Any] = {
            "name": name,
            "dim": z.size(1),
            "test_acc_knn": knn["test_acc"],
            "eff_rank": effective_rank(z),
            "protocols": {},
        }
        z_dev, y_dev = z.to(device), y.to(device)
        idx = [i.to(device) for i in (train_idx, val_idx, test_idx)]
        for protocol in args.protocols:
            row["protocols"][protocol] = run_protocol(
                protocol, z_dev, y_dev, *idx, num_classes, args
            )
        rows.append(row)
        cells = " | ".join(f"{p} {_cell(row['protocols'][p])}" for p in args.protocols)
        print(f"  {name}: {cells} | knn {100 * knn['test_acc']:.2f}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "dataset": args.dataset,
        "protocols": {p: SOURCES[p] for p in args.protocols},
        "hyperparameters": {**vars(args), "linear_probe_graphssl": "LogRegEvaluator() defaults"},
        "rows": rows,
        "provenance": {
            "timestamp_utc": timestamp,
            "git": _git_info(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
    }
    path = out_dir / f"probes__{timestamp}.json"
    path.write_text(json.dumps(result, indent=2))
    summary = render_summary(rows, args.protocols, args)
    (out_dir / "summary.md").write_text(summary)
    print(f"\n{summary}\nsaved to {path}", flush=True)


if __name__ == "__main__":
    main()
