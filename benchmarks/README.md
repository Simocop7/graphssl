# Benchmarks

Unified, reproducible benchmark runner — one command sweeps any combination of
datasets/models/seeds with the same encoder and training budget, instead of
maintaining a separate script per dataset.

```bash
# Cross-method comparison: all 7 SSL models on Cora, same encoder/budget
python benchmarks/run_benchmark.py --dataset Cora --model all --seeds 10

# One model, multiple datasets
python benchmarks/run_benchmark.py --dataset Cora CiteSeer PubMed --model bgrl

# Render the results (paste straight into README.md / paper.tex)
python benchmarks/render_tables.py                     # one table per dataset and encoder
python benchmarks/render_tables.py --latex
python benchmarks/render_tables.py --summary           # methods x datasets, per encoder
python benchmarks/render_tables.py --summary --latex
```

Each run writes `results/<Dataset>/<model>__<UTC-timestamp>.json` — per-seed
metrics, aggregate mean/std, the exact hyperparameters and full model config
(`model_config`, the dict passed to `build_model()`), the git commit (and whether tracked
files had uncommitted changes), and package versions. Old files are never overwritten, so
results accumulate as a history; `render_tables.py` always uses the most recent file per
(dataset, encoder, model): a run with `--encoder gcn` is another table, it does not replace
the GIN rows. `--summary` prints the layout of `../docs/benchmarks.md` (one row per method,
one column per dataset, best SSL method in bold, the supervised heads apart) and adds the
untrained encoder as the first row when `ablations/untrained_encoder` has the `--epochs 0`
runs.

Besides accuracy, every result records the **effective rank** of the evaluated embeddings
(`eff_rank`, from `graphssl.evaluation.effective_rank`: 1 = everything along one direction,
`hidden_dim` = variance spread evenly), which exposes dimensional collapse. For the
teacher-student models it also evaluates the **other encoder** of the same trained model
(`alt_encoder`: target for BGRL/AFGRL, student for GraphDINO), saved with an `_alt` suffix.

## Ablations

Ablations must not land in `results/`, or `render_tables.py` would report them as the
benchmark. Point `--out-dir` elsewhere and render them with `render_ablation.py`, which keeps
one row per (model, encoder, steps, starting EMA momentum, `--set` overrides) and, when a
configuration was run more than once, shows its latest run:

```bash
python benchmarks/run_benchmark.py --dataset Cora --model bgrl --epochs 1000 --ema-tau 0.9 \
    --out-dir benchmarks/ablations/my_ablation
python benchmarks/render_ablation.py benchmarks/ablations/my_ablation
```

`--ema-tau` sets the teacher's starting EMA momentum (BGRL/AFGRL anneal it to 1.0,
GraphDINO to 0.996); the schedule always spans the whole `--epochs` budget.

For GraphDINO the runner pins reference DINO's values (EMA momentum 0.996 from the first
step, teacher temperature 0.04) rather than the library defaults, so the benchmark row stays
untuned like the others. `--ema-tau 0.9 --set head.teacher_temp=0.07` runs the library
defaults.

`--set KEY=VALUE` (repeatable) overrides any field of the model config, with dotted keys for
nested configs and values parsed as JSON:

```bash
python benchmarks/run_benchmark.py --dataset Cora --model graphdino \
    --set head.norm_last_layer=true --set head.teacher_temp=0.07 \
    --out-dir benchmarks/ablations/my_ablation
```

Keys are checked against the model's config dataclasses, so a typo stops the run instead of
being silently ignored; the overrides are saved under `hyperparameters.overrides` and shown
as a column by `render_ablation.py`. `ablation_graphdino.sh` uses them to test GraphDINO's
stability levers (2.5 to 3.5 h on an A2, depending on how the GPU is shared).
`ablation_graphdino_ema.sh` is its follow-up (1.5 to 2 h 45 min):
the faster EMA teacher (`--ema-tau 0.9`) with GCN, on PubMed and combined with the softer
teacher. Both write to `ablations/graphdino_stability`.

`--epochs 0` skips training and evaluates the **untrained encoder**. Use it as the reference
when comparing backbones: an untrained GCN already separates the Planetoid classes far better
than an untrained GIN, so a higher number with GCN is not by itself evidence that a method
learned more.

`ablation_teacher_student.sh` is the grid behind the teacher-student question (why BGRL,
AFGRL and GraphDINO trail the top group): budget 300/1000/3000 steps × default vs faster
EMA teacher, a Barlow Twins control, BGRL with GCN, and AFGRL on Cora. Between 4 and 7 h on
an A2, depending on how the GPU is shared:

```bash
bash benchmarks/ablation_teacher_student.sh 2>&1 | tee -a ablation.log
python benchmarks/render_ablation.py benchmarks/ablations/teacher_student
```

`ablations/predictor` holds BGRL and AFGRL with the predictor of the public BGRL
implementation (Linear → PReLU → Linear) instead of the library's default (Linear →
BatchNorm → ReLU → Linear): the benchmark protocol, 10 seeds, both backbones. Its baseline is
the benchmark itself, i.e. the BGRL and AFGRL rows of `results/`, same seeds:

```bash
for E in gcn gin; do
  python benchmarks/run_benchmark.py --dataset Cora CiteSeer PubMed --seeds 10 \
      --model bgrl afgrl --encoder $E --set pred_norm=none --set pred_activation=prelu \
      --out-dir benchmarks/ablations/predictor
done
python benchmarks/render_ablation.py benchmarks/ablations/predictor
```

## Molecules (ZINC)

`run_zinc.py` runs every SSL method on ZINC-12k, a graph-level regression task on molecules:
the encoder is pre-trained on the training molecules without labels, frozen, and a ridge
regression on its graph embeddings predicts the target. It reports test MAE (lower is
better) and the effective rank, next to the same encoder left untrained. `--model
supervised` trains the encoder end to end on the labels as a reference.

```bash
python benchmarks/run_zinc.py --model bgrl --epochs 2 --seeds 1 --out-dir benchmarks/zinc/smoke
python -u benchmarks/run_zinc.py --model all supervised --seeds 10 --finetune-epochs 100 \
    --out-dir benchmarks/zinc_seeds10 2>&1 | tee -a zinc.log
```

`--finetune-epochs N` adds the second evaluation: after pre-training, the encoder and a new
linear head are trained on the labels for N epochs (best-validation epoch). Its reference is
the supervised row, the same training from a random initialisation.

One shared, untuned protocol (GINE with 4 layers of 128 units, 100 epochs, 20% of the atoms
masked and 20% of the bonds dropped): like the citation benchmark it compares the objectives
at equal budget. Results go to `--out-dir` (one JSON per model and `summary.md`). The
committed results are the 10-seed run in `benchmarks/zinc_seeds10/` (commit `acae6df`, about
13 hours on an NVIDIA A2); `benchmarks/zinc/` keeps the first run, with 3 seeds. Use 10: with
3 the standard deviations of the fine-tuned column came out two to three and a half times
smaller, and the same seed does not give the same number twice (see `../docs/benchmarks.md`).

## Probe protocols (`compare_probes.py`)

`compare_probes.py` scores the embeddings of saved models (checkpoints written by
`save_model()`: the stress test and the reproduction script keep theirs) with the
linear-evaluation protocols of other codebases, re-implemented from their sources:

- `adam100`: the reference code of DGI and PyG-SSL's evaluator. 100 full-batch Adam steps on
  the raw embeddings, test accuracy after the last step, averaged over `--repeats` random
  initialisations.
- `adam5000`: PyGCL. 5,000 Adam steps, test accuracy at the best validation accuracy.
- `liblinear`: the BGRL reference implementation. One-vs-rest liblinear on L2-normalised
  rows, C selected on validation. Needs scikit-learn; 21 × `num_classes` fits per model, so
  it is not run unless named in `--protocols`.
- `graphssl`: `LogRegEvaluator`, the probe of every other table here.

```bash
python -u benchmarks/compare_probes.py --dataset ogbn-arxiv --untrained \
    --checkpoints benchmarks/stress/ogbn_arxiv_20261012/*.pt \
                  benchmarks/reproductions/bgrl_ogbn_arxiv/*.pt \
                  benchmarks/reproductions/bgrl_ogbn_arxiv_library_predictor/*.pt \
    --out-dir benchmarks/probes/ogbn_arxiv
```

`--untrained` adds each distinct architecture at initialisation; `--dataset` also takes
Cora, CiteSeer and PubMed. One JSON with every fit and a `summary.md` go to `--out-dir`.
The committed run (2026-10-13, commit `7204b92`, about 40 minutes on an idle A2) is in
`benchmarks/probes/ogbn_arxiv/`; its tables and what they say are in
`../docs/benchmarks.md` ("Linear evaluation is not one protocol"). In short: the protocols
agree on well-conditioned embeddings and disagree on weak ones, where the 100-step protocol
is far from fitted, so it makes self-supervision look about three times more useful against
an untrained encoder than a converged probe does.

## Reproductions

`reproduce_bgrl_arxiv.py` runs BGRL on ogbn-arxiv with the protocol of the BGRL paper
(3 GCN layers with layer normalization and weight standardization, full-graph training,
10,000 steps) instead of the shared benchmark protocol, and compares with the paper's
Table 5. It also evaluates the encoder left untrained, which checks the encoder and the
evaluation independently of training. One seed takes about four hours on the A2. Results go
to `benchmarks/reproductions/bgrl_ogbn_arxiv/`. The predictor is the reference
implementation's (Linear → PReLU → Linear); `--predictor library` uses the library's default
(Linear → BatchNorm → ReLU → Linear).

```bash
python -u benchmarks/reproduce_bgrl_arxiv.py --seeds 5 2>&1 | tee -a reproduce_bgrl.log
```

The committed run (2026-10-10, commit `acae6df`, 3 seeds, reference predictor) gives 72.31 ±
0.14 validation / 71.32 ± 0.10 test accuracy against 72.53 ± 0.09 / 71.64 ± 0.12 in the
paper, and 70.23 ± 0.18 / 69.53 ± 0.03 for the untrained encoder (paper: 69.90 ± 0.11 /
68.94 ± 0.15). The table and its reading are in
[`docs/benchmarks.md`](../docs/benchmarks.md). The script prints that table at the end of a
run; the trained model of every seed is saved next to the JSON (`bgrl_seed<N>.pt`,
git-ignored).

The same run with the library's default predictor (`--predictor library --out-dir
benchmarks/reproductions/bgrl_ogbn_arxiv_library_predictor`, same commit and seeds) ends
below the untrained encoder: 68.97 ± 0.87 / 67.91 ± 0.78, against 70.24 ± 0.20 / 69.52 ±
0.03 untrained. With the layer-normalized encoder of this protocol the predictor must not
have a batch normalization.

## Stress test (ogbn-arxiv)

`stress_ogbn_arxiv.py` runs every SSL method through the **mini-batch** path on ogbn-arxiv
(169k nodes, 2.3M edges), which the full-batch citation benchmark never touches:
`NeighborLoader` training, the trainer's callback hooks with an evaluation (kNN, effective
rank) running during training, full-graph extraction and linear-probe / kNN evaluation at
that scale, and mini-batch extraction checked against the full-graph pass. It is a robustness test, not a benchmark
(one seed, no tuning). A model that fails does not stop the run: its error is saved and the
next model starts.

```bash
# smoke test, a few minutes
python benchmarks/stress_ogbn_arxiv.py --epochs 2 --max-steps 5 --eval-every 1 \
    --skip-linear-probe --out-dir benchmarks/stress/smoke
# the real run, inside tmux
python -u benchmarks/stress_ogbn_arxiv.py --epochs 50 2>&1 | tee -a stress_arxiv.log
```

One JSON per model goes to `benchmarks/stress/ogbn_arxiv/` (status, per-epoch loss / time /
peak GPU memory / sampled-subgraph size, kNN accuracy and effective rank during training,
the linear probe before and after it, the untrained encoder as reference), next to a
`summary.md` table. The linear probe is fitted only twice per model because one fit takes
minutes on 91k training nodes; `--skip-linear-probe` drops it for smoke tests.

The committed run (2026-10-12, commit `acae6df`, 100 epochs, one seed) is in
`benchmarks/stress/ogbn_arxiv_20261012/`. Test accuracy (%); the untrained GIN-2L encoder
scores 55.2 with the linear probe and 53.3 with kNN:

| Model | Status | Epochs | s/epoch | Peak GPU (GiB) | Untrained linear | Linear | kNN | Rank |
|---|---|---|---|---|---|---|---|---|
| dgi | ok | 100 | 26.8 | 0.93 | 55.22 | 44.15 | 52.87 | 1.5 |
| graphcl | ok | 100 | 27.4 | 0.90 | 55.20 | 63.64 | 54.33 | 38.3 |
| vicreg | ok | 100 | 27.4 | 0.89 | 55.17 | 62.03 | 54.17 | 36.8 |
| barlow_twins | ok | 100 | 27.4 | 0.89 | 55.20 | 62.72 | 54.38 | 68.6 |
| bgrl | ok | 100 | 34.8 | 1.10 | 55.19 | 63.40 | 59.87 | 49.2 |
| afgrl | ok | 100 | 138.1 | 2.07 | 55.17 | 58.37 | 58.71 | 23.1 |
| graphdino | ok | 100 | 18.1 | 1.08 | 55.22 | 62.44 | 56.18 | 19.0 |

- **Every model runs to the end.** Mini-batch extraction matches the full-graph pass
  (relative difference at most 1.2e-06), every probe converges, and no model needs more
  than 2.1 GiB of GPU memory. A model takes 0.6 to 1.4 hours on an NVIDIA A2, AFGRL 4.1:
  it mines its positives over every node of each batch.
- **`ok` means that a model ran, not that it learned.** DGI is `ok` and ends 11 points
  below the untrained encoder with the linear probe. Its kNN accuracy peaks at epoch 20
  (58.0) and declines to 52.9, with the effective rank down to 1.5. The cause is not
  identified; in mini-batch training stop DGI early.
- **The two metrics disagree.** With the linear probe GraphCL, BGRL, Barlow Twins, GraphDINO
  and VICReg are within 1.6 points of each other (62.0–63.6), 7 to 8 above the untrained
  encoder. With kNN only BGRL (59.9) and AFGRL (58.7) gain much; GraphCL, VICReg and Barlow
  Twins end at 54.2–54.4, one point above the untrained encoder, and Barlow Twins was at
  58.1 after 10 epochs.
- One seed and no tuning: this is a robustness test. An untrained 3-layer GCN reaches 69.5%
  on the same graph (see Reproductions), above every row here: the backbone sets the level.

Each trained model is saved next to its JSON (`<model>.pt`, not committed; `load_model()`
reads it). The first run (2026-10-06, commit `aaa280f`) stays in
`benchmarks/stress/ogbn_arxiv/` as history: its `Linear` columns come from the 0.1.0 probe,
which on that graph gives 45–54% on the same embeddings depending on its random seed, and
its DGI still encoded the real and the corrupted graph in two passes (kNN 33.7).

This intentionally does not replace `examples/*.py`, which stay as minimal
single-file demos for learning the API — this is for producing citable,
comparable numbers across the model zoo. See `../CLAUDE.md` (Benchmarks
section) for the methodology and current status.

**Linear probe.** `LogRegEvaluator`'s default protocol: standardized features, L2-regularised
logistic regression fitted to convergence, L2 strength selected on the validation split.
Each JSON records the settings (`hyperparameters.linear_probe`) and, per seed, the selected
strength and whether the fit converged (`probe_weight_decay`, `probe_converged`). A JSON
without `linear_probe` was evaluated with the 0.1.0 probe (100 Adam steps on raw features).
Both renderers print the protocol under each table and say so when its rows mix the two.
`run_benchmark.py` fits the probe on the CPU with 4 threads: with 60–140 training nodes
that takes 1–5 s, against 20–40 s with 48 threads or on a busy GPU.

**Current results:** all 7 SSL models and the two supervised references × Cora / CiteSeer /
PubMed × 10 seeds, with the GIN and the GCN backbone, are committed under `results/` and
rendered in `../docs/benchmarks.md` (runs of 2026-10-09 with the current probe: GIN at
commit `49e92ee`, GCN at `acae6df`). The untrained encoders are in
`ablations/untrained_encoder`. The older GIN files (0.1.0 probe) stay in `results/` as
history. `ablations/gcn_bn_momentum_0.01` keeps the GCN runs made before the encoder's
BatchNorm momentum went from 0.01 to 0.1: same training, but the statistics used at
evaluation had not converged after 300 steps, so their kNN, effective-rank and
supervised-head numbers are too low (the linear probe is within ±0.5). The ablation
directories `teacher_student` and `graphdino_stability` hold two runs of every
configuration: the first with the 0.1.0 probe, and the rerun of 2026-10-10/11 with the
current one (commit `acae6df`), which is the one `render_ablation.py` shows.
`ablations/predictor` is from the same commit, and so is `ablations/graphdino_defaults`:
GraphDINO with the library defaults in the benchmark protocol (10 seeds, three datasets,
both backbones), next to which the benchmark row keeps reference DINO's values.

Budget for a full sweep of one backbone: about 3.5 h on an NVIDIA A2. Most models take
~10–60 s per seed, but on PubMed GraphCL (O(N²) NT-Xent) takes ~7–10 min per seed and AFGRL
(dense N×N kNN + per-step k-means) ~4 min.

**Supervised references** (not part of `all`; `--model all supervised supervised_reg` runs
everything): both train the same encoder full-batch on the public split's `train_mask` only.
- `--model supervised` keeps the shared SSL protocol unchanged (no dropout, AdamW, last
  checkpoint). With 60–140 labels and no regularization it overfits, so it understates what
  the labels are worth.
- `--model supervised_reg` uses a recipe adapted from Kipf & Welling: dropout 0.5, Adam lr
  0.01 with L2 5e-4, checkpoint with the best validation accuracy. With the GCN backbone the
  encoder also has no normalization layer, as in their model (`SUPERVISED_REG_NORM` in
  `run_benchmark.py`; `--set encoder.norm_type=batch` puts it back). The recipe is recorded
  in the JSON's `hyperparameters` (`norm_type` included), the selected epoch per seed as
  `best_epoch`.
  - GCN: the head reaches 81.2 / 68.6 / 78.4, 0.3 / 1.7 / 0.6 below Kipf & Welling. With batch norm
    the same recipe gives 77.9 / 65.0 / 75.8 (`ablations/supervised_norm`).
  - GIN keeps batch norm: without a normalization layer its head falls to 57.2 / 49.2 / 71.6
    (same directory). With batch norm the recipe helps on Cora and CiteSeer and hurts on
    PubMed against the plain supervised row.
  - With GCN `best_epoch` is among the first six for most seeds on Cora and CiteSeer:
    validation accuracy peaks after a few Adam steps. The accuracy at that epoch is the one
    reported, so this is not a sign of a broken run.

Both get the same linear probe + kNN evaluation on their encoder embeddings; the accuracy of
the trained head itself is saved as `test_acc_head`.

**Known gaps:**
- `afgrl` is skipped automatically when `faiss-cpu` isn't installed (same
  behavior as `tests/test_afgrl.py`).
- `run_benchmark.py` covers the Planetoid datasets (Cora/CiteSeer/PubMed) only. ZINC has its
  own runner (`run_zinc.py`), ogbn-arxiv a stress test and a reproduction script (above).
  OGB graph-level datasets (ogbg-molhiv, ogbg-molpcba) are not wired up.
