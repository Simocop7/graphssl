# Evaluating and Reusing a Model

A self-supervised model is judged by what its encoder is good for afterwards. This page
covers the two ways of finding out (a probe on frozen embeddings, and fine-tuning), the
reference every number needs (the same encoder untrained), and how to save a model and
start from it later.

The code below is one script. It pretrains BGRL on Cora with a GCN encoder and runs on a
CPU in about a minute.

```python
import torch
from torch.optim import AdamW
from torch_geometric.datasets import Planetoid

from graphssl.config import build_model, load_model, save_model
from graphssl.core import pretrained_encoder
from graphssl.data import DataModule
from graphssl.evaluation import KNNEvaluator, LogRegEvaluator, effective_rank, extract_embeddings
from graphssl.training import DINOTrainer

dataset = Planetoid(root="data/Cora", name="Cora")
data = dataset[0]
train_idx, val_idx, test_idx = (
    mask.nonzero(as_tuple=True)[0] for mask in (data.train_mask, data.val_mask, data.test_mask)
)
dm = DataModule(
    data=data, is_graph_level=False, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
)

config = {
    "name": "bgrl",
    "encoder": {"name": "gcn", "hidden_dim": 256, "num_layers": 2, "pool": False},
    "augment": [{"name": "edge_drop", "p": 0.5}, {"name": "feat_mask", "p": 0.2}],
    "pred_hidden": 256, "ema_tau": 0.99, "ema_tau_end": 1.0, "total_steps": 300,
}
model = build_model(config, in_channels=data.num_features)
```

## Frozen evaluation, against the untrained encoder

```python
def evaluate(model):
    z, y = extract_embeddings(model, dm)  # one row per node, on the CPU
    linear = LogRegEvaluator().evaluate(
        z, y, train_idx, val_idx, test_idx, num_classes=dataset.num_classes
    )
    knn = KNNEvaluator(k=5).evaluate(z, y, train_idx, val_idx, test_idx)
    return {"linear": linear["test_acc"], "knn": knn["test_acc"], "rank": effective_rank(z)}


untrained = evaluate(model)  # the encoder as initialized, before any training

trainer = DINOTrainer(device="cpu")
optimizer = AdamW(model.student_parameters(), lr=5e-4, weight_decay=1e-5)
trainer.train(model, dm.train_dataloader(), optimizer, num_epochs=300)

trained = evaluate(model)
```

One run of this script gave:

| | Linear probe | kNN | Effective rank |
|---|---|---|---|
| Untrained encoder | 71.6 | 57.7 | 189 |
| After 300 steps of BGRL | 79.8 | 73.1 | 172 |

**Measure the untrained encoder every time.** A graph encoder with random weights already
mixes each node's features with its neighbors', and a probe on top of that is far from
chance: 72% here, where guessing gives 14%. What pretraining added is 8 points, not 80. The
benchmark tables report this row for the same reason, and it differs a lot between
backbones (an untrained GIN gives 42% on Cora).

What the three numbers are:

- **`LogRegEvaluator`** is the linear probe: an L2-regularized logistic regression on the
  embeddings, standardized with the training rows' statistics and fitted to convergence
  with L-BFGS from a zero initialization. It has no seed and no learning rate, so the same
  embeddings always give the same result. The L2 strength is the one with the best
  validation accuracy among `weight_decay` (by default 10 down to 1e-6). `evaluate()`
  returns `val_acc`, `test_acc`, the selected `weight_decay` and `converged`. With 140
  training nodes it takes a few seconds; with the 91k of ogbn-arxiv, minutes.
- **`KNNEvaluator`** is a cosine k-nearest-neighbor vote. It needs no fit and reads the
  geometry of the embedding space directly, so it can disagree with the linear probe.
- **`effective_rank`** says how many directions the embeddings use (1 = a single direction,
  256 = all of them here). A low rank after training means a dimensional collapse that the
  accuracy alone may hide. A higher rank is not better by itself.

"Linear evaluation" is not one protocol across codebases, and the choice changes how much
pretraining appears to add. What matters most is **regularization**: with a hundred or so
labeled nodes, an unregularised classifier overfits the embeddings of a trained encoder,
while the raw embeddings of an untrained one are small enough that a short optimization
barely fits them. The bias falls on one side, and the gain of pretraining shrinks or even
changes sign. Standardizing the embeddings and choosing the L2 strength on validation, as
`LogRegEvaluator` does, removes it ([Evaluation Study](study.md#the-probe-decides-how-large-the-gain-looks);
on ogbn-arxiv the same probes go the other way:
[Linear evaluation is not one protocol](benchmarks.md#linear-evaluation-is-not-one-protocol)).
When you compare with a number from a paper, check how its probe was fitted.

For a **regression** target use `RidgeEvaluator`: a closed-form ridge regression on the
standardized embeddings, with the L2 strength selected on the validation MAE.

```python
from graphssl.evaluation import RidgeEvaluator

# z: [N, D] frozen embeddings; targets: [N] or [N, T] floats
result = RidgeEvaluator().evaluate(z, targets, train_idx, val_idx, test_idx)
result["test_mae"], result["test_rmse"], result["weight_decay"]
```

## A probe during training

To follow the probe while the model trains, give the trainer a `LinearEvalCallback` in
place of the plain `DINOTrainer(device="cpu")` above. It runs the probe every few epochs and
keeps the results. The default evaluator fits one classifier per L2 strength; give it a
single one to keep each evaluation short.

```python
from graphssl.training import LinearEvalCallback

probe = LinearEvalCallback(
    dm,
    num_classes=dataset.num_classes,
    every_n_epochs=50,
    evaluator=LogRegEvaluator(weight_decay=1e-2),  # a single fit per evaluation
)
trainer = DINOTrainer(device="cpu", callbacks=[probe])
trainer.train(model, dm.train_dataloader(), optimizer, num_epochs=300)
probe.results_history  # one dict per evaluation: val_acc, test_acc, epoch, ...
```

Watching the curve matters with self-supervision: there is no validation loss that tells
when to stop, and several methods get worse when trained for longer (see the
[sensitivity section](benchmarks.md#sensitivity-budget-teacher-and-backbone) of the
benchmarks).

## Saving and loading

```python
save_model(model, "bgrl_cora.pt", config, in_channels=data.num_features)
model = load_model("bgrl_cora.pt")  # rebuilt from the saved config, weights loaded
```

The file holds the config next to the weights, so `load_model()` needs nothing else. A
supervised model also takes `num_classes=` in `save_model()`.

## Pre-train, then fine-tune

`pretrained_encoder(model)` is the encoder a model embeds with: the online encoder of BGRL
and AFGRL, the teacher of GraphDINO, the only encoder of the others. Fine-tuning puts a task
head on a copy of it and trains both on the labels.

```python
supervised = build_model(
    {"name": "supervised", "encoder": config["encoder"]},
    in_channels=data.num_features,
    num_classes=dataset.num_classes,
)
supervised.encoder.load_state_dict(pretrained_encoder(model).state_dict())

optimizer = AdamW(supervised.student_parameters(), lr=5e-4, weight_decay=1e-5)
trainer.train(supervised, dm.train_dataloader(), optimizer, num_epochs=100)

supervised.eval()
with torch.no_grad():
    prediction = supervised.head(supervised(data)).argmax(dim=-1)
test_acc = (prediction[test_idx] == data.y[test_idx]).float().mean().item()
```

For a regression target add `"task": "regression"` to the supervised config and pass the
number of targets as `num_classes`.

The reference for a fine-tuned model is **the same training from a random initialization**:
build the supervised model, skip the `load_state_dict` line, and train it identically. Give
both the same total budget and tune the fine-tuning learning rate for both, or the comparison
favors the pretrained model. On ZINC, with a frozen probe none of the seven objectives can be
told apart from the untrained encoder, while after fine-tuning three of them reach a lower
error than the from-scratch reference under one untuned protocol in which neither training has converged
([results](benchmarks.md#molecules-zinc-graph-level-regression)). The two evaluations answer
different questions, so run the one that matches how the encoder will be used.

## How many runs

A seed fixes the initial weights and the order of the data, not the result: on a GPU two
runs with the same seed end at different numbers, and on ZINC the difference is as large as
the spread between seeds. The same seed on another device draws other augmentations, so with
a GIN encoder a rerun elsewhere is close to a new seed
([Evaluation Study](study.md#a-seed-does-not-pin-a-result-down)). Report a mean and a
standard deviation over several seeds (the benchmarks here use 10), and do not read a
difference smaller than that deviation as an effect.
