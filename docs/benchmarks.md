# Benchmarks

## Citation networks: cross-method comparison

All 7 SSL methods, trained and evaluated under **one shared protocol**, so that differences
come from the objective rather than from per-method tuning. The whole comparison is run
twice, with two backbones: **GCN**, the encoder the node-level SSL literature uses, and
**GIN**, which is where the objectives differ most.

| Setting | Value |
|---|---|
| Encoder | GCN or GIN, 2 layers, `hidden_dim=256`, batch norm, no dropout |
| Training | full-batch, 300 steps, AdamW (lr `5e-4`, weight decay `1e-5`) |
| Augmentation | `edge_drop` p=0.5 + `feat_mask` p=0.2, both views (GraphCL, BGRL, VICReg, Barlow Twins, GraphDINO); DGI uses node shuffling, AFGRL none |
| EMA (BGRL, AFGRL) | cosine schedule 0.99 → 1.0 over training |
| GraphDINO | reference DINO values: EMA momentum 0.996, teacher temperature 0.04 — not the library defaults, see [Sensitivity](#sensitivity-budget-teacher-and-backbone) |
| Split | public Planetoid train/val/test |
| Evaluation | frozen embeddings → linear probe (`LogRegEvaluator`: L2-regularised logistic regression on standardized features, fitted to convergence, L2 strength selected on validation) and kNN (k=5, cosine) |
| Seeds | 10; mean ± sample std |
| Hardware | NVIDIA A2 (16 GB), PyTorch 2.11 + CUDA 12.8, PyG 2.8 |
| Untrained encoder | the same encoder at initialization (0 steps), evaluated the same way |
| Supervised references (†) | same encoder, trained on the public split's training labels only: *Supervised* keeps the protocol above; *std. recipe* uses dropout 0.5, Adam lr 0.01 + L2 5e-4 and the best-validation checkpoint (adapted from Kipf & Welling) |

No per-method or per-dataset hyperparameter search is performed: the goal is to compare
objectives at equal budget, not to reproduce each paper's best reported number. Test accuracy
(%), best SSL method per column in bold. The supervised rows (†) use labels during training:
they are reference points, not ranked. Every table below is generated from the saved JSON
files by `python benchmarks/render_tables.py --summary`.

### GCN encoder

Linear probe:

| Method | Cora | CiteSeer | PubMed |
|---|---|---|---|
| *Untrained encoder* | 73.17 ± 0.99 | 57.69 ± 1.47 | 74.07 ± 1.60 |
| DGI | 78.12 ± 1.22 | 65.07 ± 1.44 | 78.60 ± 1.35 |
| GraphCL | **82.46 ± 1.08** | **69.59 ± 1.00** | 81.28 ± 0.77 |
| BGRL | 80.82 ± 1.23 | 67.05 ± 0.89 | 79.85 ± 1.10 |
| AFGRL | 79.18 ± 1.26 | 62.63 ± 2.01 | 78.43 ± 1.00 |
| VICReg | 81.06 ± 1.10 | 66.78 ± 1.45 | **81.59 ± 0.70** |
| Barlow Twins | 81.65 ± 0.93 | 68.95 ± 1.33 | 81.24 ± 0.72 |
| GraphDINO | 77.53 ± 1.22 | 60.16 ± 1.40 | 76.97 ± 2.08 |
| *Supervised* † | 72.67 ± 0.85 | 56.35 ± 1.52 | 74.12 ± 1.01 |
| *Supervised, std. recipe* † | 78.89 ± 1.42 | 64.84 ± 3.84 | 75.45 ± 1.12 |

kNN (k=5):

| Method | Cora | CiteSeer | PubMed |
|---|---|---|---|
| *Untrained encoder* | 60.28 ± 2.65 | 48.52 ± 2.33 | 68.31 ± 1.47 |
| DGI | 73.51 ± 1.93 | 59.30 ± 1.93 | 72.63 ± 1.21 |
| GraphCL | 75.32 ± 0.98 | 63.52 ± 1.11 | 75.00 ± 1.04 |
| BGRL | 74.71 ± 1.19 | 60.89 ± 1.58 | 71.93 ± 1.31 |
| AFGRL | 72.90 ± 1.73 | 55.29 ± 1.89 | 71.02 ± 1.15 |
| VICReg | 72.53 ± 0.96 | 62.08 ± 1.28 | **75.07 ± 0.72** |
| Barlow Twins | **75.56 ± 1.35** | **63.58 ± 1.42** | 75.03 ± 0.86 |
| GraphDINO | 68.98 ± 1.40 | 51.39 ± 2.30 | 70.20 ± 1.39 |
| *Supervised* † | 71.59 ± 1.02 | 56.53 ± 0.91 | 73.56 ± 1.25 |
| *Supervised, std. recipe* † | 77.04 ± 1.24 | 65.36 ± 2.88 | 75.43 ± 1.54 |

### GIN encoder

Linear probe:

| Method | Cora | CiteSeer | PubMed |
|---|---|---|---|
| *Untrained encoder* | 41.70 ± 2.44 | 36.06 ± 1.57 | 56.60 ± 1.50 |
| DGI | 65.43 ± 3.03 | 49.18 ± 2.86 | 66.90 ± 3.87 |
| GraphCL | **81.31 ± 1.14** | 65.17 ± 1.44 | **80.36 ± 0.82** |
| BGRL | 63.77 ± 1.60 | 47.68 ± 2.30 | 68.52 ± 2.57 |
| AFGRL | 70.75 ± 1.75 | 49.32 ± 3.02 | 74.51 ± 1.11 |
| VICReg | 77.36 ± 1.88 | 63.52 ± 2.10 | 79.08 ± 1.27 |
| Barlow Twins | 79.33 ± 0.75 | **65.60 ± 1.62** | 77.67 ± 1.49 |
| GraphDINO | 62.53 ± 1.91 | 43.34 ± 3.27 | 70.38 ± 3.23 |
| *Supervised* † | 71.46 ± 1.83 | 51.21 ± 3.73 | 71.49 ± 1.70 |
| *Supervised, std. recipe* † | 75.83 ± 2.33 | 59.32 ± 4.51 | 71.96 ± 1.90 |

kNN (k=5):

| Method | Cora | CiteSeer | PubMed |
|---|---|---|---|
| *Untrained encoder* | 27.13 ± 1.79 | 27.00 ± 1.85 | 32.65 ± 2.86 |
| DGI | 65.02 ± 2.50 | 45.79 ± 4.00 | 66.43 ± 2.61 |
| GraphCL | 73.88 ± 1.40 | 60.55 ± 1.91 | 77.87 ± 1.26 |
| BGRL | 59.48 ± 3.75 | 42.90 ± 2.21 | 66.85 ± 2.75 |
| AFGRL | 64.80 ± 2.13 | 44.20 ± 3.61 | 70.63 ± 2.11 |
| VICReg | 72.71 ± 1.72 | 58.37 ± 1.71 | **78.49 ± 1.51** |
| Barlow Twins | **75.52 ± 1.06** | **63.14 ± 1.67** | 77.15 ± 1.52 |
| GraphDINO | 55.35 ± 1.80 | 37.03 ± 1.75 | 66.34 ± 3.69 |
| *Supervised* † | 74.11 ± 0.91 | 54.74 ± 1.39 | 74.62 ± 1.59 |
| *Supervised, std. recipe* † | 76.98 ± 1.90 | 61.41 ± 3.46 | 72.63 ± 2.05 |

### Supervised references: accuracy of their own head

The † rows above evaluate the supervised encoders like every other row, with a probe on
their frozen embeddings. This is the accuracy of the head they were trained with:

| Encoder, method | Cora | CiteSeer | PubMed |
|---|---|---|---|
| GCN, *Supervised* † | 72.33 ± 1.08 | 57.15 ± 1.06 | 73.84 ± 0.89 |
| GCN, *Supervised, std. recipe* † | 77.98 ± 2.08 | 65.37 ± 1.66 | 74.64 ± 1.68 |
| GIN, *Supervised* † | 73.18 ± 1.23 | 51.79 ± 2.08 | 74.45 ± 1.63 |
| GIN, *Supervised, std. recipe* † | 76.53 ± 2.07 | 61.79 ± 3.89 | 72.66 ± 2.17 |

### Reading the results

- **With GCN the objectives are close, and every one of them learns.** All seven methods are
  above the untrained GCN on all three datasets, by 2.5 to 11.9 points. GraphCL, Barlow
  Twins, VICReg and BGRL are within 1.6 points of each other on Cora, 2.8 on CiteSeer and
  1.7 on PubMed. AFGRL, DGI and GraphDINO are 3 to 9 points below the best method of each
  column.
- **With GIN they separate.** GraphCL, Barlow Twins and VICReg lose 1 to 4 points against
  their GCN numbers (GraphCL 81.3 / 65.2 / 80.4 against 82.5 / 69.6 / 81.3). BGRL, AFGRL,
  GraphDINO and DGI fall 6 to 22 points below the best method of each column. BGRL is the
  clearest case: 63.8 / 47.7 / 68.5 with GIN, 80.8 / 67.1 / 79.9 with GCN, and the effective
  rank of its embeddings is 8–9 of 256 dimensions with GIN against 148–190 with GCN. So
  "teacher-student methods trail" is a statement about the GIN table, not about the
  methods.
- **The untrained encoder is the reference to read both tables against.** An untrained GCN
  already gives 73.2 / 57.7 / 74.1, an untrained GIN 41.7 / 36.1 / 56.6. Much of what the
  GCN table shows is the backbone; what an objective adds is the difference from that row.
- **The supervised references stay below the best SSL probe.** The head of the standard
  recipe reaches 78.0 / 65.4 / 74.6 with GCN and 76.5 / 61.8 / 72.7 with GIN, against a best
  SSL linear probe of 82.5 / 69.6 / 81.6 and 81.3 / 65.6 / 80.4. These references are
  untuned too: the 2-layer GCN of Kipf & Welling reports 81.5 / 70.3 / 79.0, and with GCN
  the standard recipe's best-validation epoch is one of the first three on CiteSeer for all
  ten seeds. Under the shared protocol (no regularization) the supervised GCN's embeddings
  are no better for a linear probe than the untrained ones. Read them as what this encoder
  gets from 60–140 labels without tuning, not as an upper bound.
- **Quadratic cost is visible.** On PubMed (≈19.7k nodes) one 300-step GIN run takes ≈440 s
  for GraphCL (pairwise NT-Xent) and ≈250 s for AFGRL (dense kNN + per-step k-means), versus
  ≈35–55 s for the other methods. Wall times are indicative only: some runs shared the GPU.

!!! note "What changed since the first version of this page"
    The tables were rerun on 2026-10-09. The linear probe is now a regularised logistic
    regression fitted to convergence; the 0.1.0 probe (a linear layer trained for 100 Adam
    steps on raw features) gave results that depended on its random seed. With the same GIN
    protocol the linear column moved by −3.2 to +6.0 points (GraphCL on CiteSeer 59.2 →
    65.2) and kNN by at most 1, except for DGI, whose training changed as well (it now
    encodes the real and the corrupted graph in one forward pass). The GCN tables are new. They were produced after a fix to
    the GCN encoder's BatchNorm momentum (0.01 → 0.1): with the old value the statistics
    used at evaluation had not converged after 300 steps, which lowered kNN by up to 9
    points for DGI and the supervised head from 73.8 to 49.7 on PubMed, while leaving the
    linear probe within ±0.5. The runs with the old momentum are kept in
    `benchmarks/ablations/gcn_bn_momentum_0.01`.

## Sensitivity: budget, teacher and backbone

The benchmark fixes one protocol for every method. These ablations vary it for the
teacher-student methods: 5 seeds, linear-probe test accuracy (%), same optimizer and
augmentations as above. They are separate from the benchmark, whose rows are unchanged.
The settings were compared on test accuracy, so the best row of a table is an optimistic
estimate for that setting.

!!! warning "These ablations still use the 0.1.0 linear probe"
    The numbers in this section were measured before the benchmark above was rerun: 5 seeds,
    the 0.1.0 probe (100 Adam steps on raw features). Compare them with each other, not
    with the tables above, whose linear column comes from the current probe. They have not
    been rerun yet.

### GraphDINO: reference DINO values degrade with training

GIN backbone, 300 / 1000 / 3000 full-batch steps:

| GraphDINO setting | Cora 300 | Cora 1000 | Cora 3000 | CiteSeer 300 | CiteSeer 1000 | CiteSeer 3000 |
|---|---|---|---|---|---|---|
| Reference DINO values (benchmark row) | 63.34 ± 1.25 | 49.96 ± 1.13 | 38.88 ± 2.33 | 42.24 ± 2.39 | 33.32 ± 1.68 | 28.80 ± 1.25 |
| `norm_last_layer=True` | — | — | 37.44 ± 1.76 | — | — | 27.90 ± 1.49 |
| `teacher_temp=0.07` | — | — | 54.24 ± 2.31 | — | — | 36.94 ± 2.09 |
| `teacher_temp=0.07` + `norm_last_layer=True` | 64.92 ± 2.34 | 65.08 ± 2.27 | 46.04 ± 2.23 | 44.54 ± 2.14 | 42.80 ± 1.86 | 32.44 ± 2.45 |
| `ema_tau_base=0.9` | 66.52 ± 2.61 | 62.94 ± 3.00 | 62.86 ± 3.05 | 43.70 ± 3.36 | 43.98 ± 4.78 | 42.74 ± 4.27 |
| **`ema_tau_base=0.9` + `teacher_temp=0.07` (library defaults)** | 73.50 ± 2.92 | 68.06 ± 2.74 | 66.98 ± 2.99 | 47.76 ± 4.56 | 45.64 ± 4.68 | 44.96 ± 3.80 |

- With the reference values (EMA momentum 0.996 from the first step, teacher temperature
  0.04) accuracy drops as training continues. At 3000 steps it is at or below the untrained
  encoder (40.40 on Cora, 36.14 on CiteSeer).
- A faster teacher (`ema_tau_base=0.9`, annealed to 0.996 over the run) removes most of the
  decline. A softer teacher (`teacher_temp=0.07`, warmed up from 0.04) only slows it on its
  own. Together they are the best setting in every cell, including PubMed at 300 steps
  (75.56 ± 0.92, against 70.68 ± 2.02 with the reference values and 73.76 ± 1.62 with the
  faster teacher alone). `norm_last_layer` has no measurable effect.
- These two values are the **library defaults** (`GraphDINOConfig.ema_tau_base`,
  `HeadConfig.teacher_temp`). The benchmark row keeps the reference values, like every other
  untuned row of the table.
- What remains: a milder decline from 300 to 3000 steps (Cora 73.50 → 66.98) and a larger
  spread across seeds (std up to 4.7 on CiteSeer).

### Training budget

A longer budget alone doesn't change the picture with GIN. From 300 to 1000 to 3000 steps
Barlow Twins is flat on Cora (78.04 / 78.44 / 77.62) and BGRL improves but stays below
(64.50 / 71.24 / 73.48). GraphDINO with the reference values declines, as shown above.

### Backbone

The effect of the backbone used to be an ablation of this section. It is now part of the
benchmark: see the GCN and GIN tables above, each with its untrained encoder, on 10 seeds
and with the current probe. The earlier 5-seed table is superseded: with the 0.1.0 probe it
had BGRL + GCN at 57.8 on CiteSeer, barely above the untrained GCN, where the current probe
gives 67.1 against 57.7.

```bash
bash benchmarks/ablation_teacher_student.sh    # budget, EMA momentum, GCN for BGRL / Barlow Twins
bash benchmarks/ablation_graphdino.sh          # GraphDINO: teacher temperature, norm_last_layer
bash benchmarks/ablation_graphdino_ema.sh      # GraphDINO: faster teacher, PubMed, untrained encoders
python benchmarks/render_ablation.py benchmarks/ablations/teacher_student
python benchmarks/render_ablation.py benchmarks/ablations/graphdino_stability
```

## Reproduce

```bash
D="--dataset Cora CiteSeer PubMed --seeds 10"
python benchmarks/run_benchmark.py $D --model all supervised supervised_reg                 # GIN
python benchmarks/run_benchmark.py $D --model all supervised supervised_reg --encoder gcn
# the untrained encoders: an ablation directory, never benchmarks/results
for E in gin gcn; do
  python benchmarks/run_benchmark.py $D --model bgrl --epochs 0 --encoder $E \
      --out-dir benchmarks/ablations/untrained_encoder
done
python benchmarks/render_tables.py --summary           # the tables of this page
python benchmarks/render_tables.py --summary --latex   # their rows for paper.tex
python benchmarks/render_tables.py                     # one table per dataset and encoder
```

Each run is saved under `benchmarks/results/<Dataset>/<model>__<UTC-timestamp>.json` with
per-seed metrics (including training wall time), aggregate mean/std, the exact
hyperparameters, the git commit (plus whether tracked files had uncommitted changes) and
package versions. Old files are never overwritten, so results accumulate as a reproducible
history; `render_tables.py` uses the latest file per (dataset, encoder, model). See
[`benchmarks/README.md`](https://github.com/Simocop7/graphssl/blob/main/benchmarks/README.md)
for the details. The supervised references aren't part of `--model all`: name them next to
it, as above.

!!! note "Provenance of the committed results"
    The tables above come from runs of 2026-10-09 on a clean tree: GIN and the untrained
    encoders at commit `49e92ee`, GCN at `acae6df`. The earlier GIN files (0.1.0 probe,
    commits `e295451`, `dae32cb`, `20cce09`, `558af8e`) are still in `benchmarks/results/`
    as history; `render_tables.py` reports the latest run of each row. The 20 files recorded
    at `e295451` say `"dirty": true`: a false positive of the runner at the time, which
    counted untracked files as modifications (fixed in `dae32cb`).

## Comparison with the original BGRL numbers

The BGRL paper's small-dataset evaluation (Appendix C, Table 7 of Thakoor et al.) reports
83.83 ± 1.61 / 72.32 ± 0.89 / 86.03 ± 0.33 on Cora / CiteSeer / PubMed. BGRL in the GCN
table above reaches 80.82 ± 1.23 / 67.05 ± 0.89 / 79.85 ± 1.10. The two are not like for
like: the paper averages 20 random splits rather than the public split, uses a tuned
configuration and evaluates with a scikit-learn logistic regression, where this is the
shared untuned protocol at 300 steps
([`paper.tex`](https://github.com/Simocop7/graphssl/blob/main/paper.tex) lists the
differences). With GIN the same protocol gives 63.77 / 47.68 / 68.52: most of the distance
from the paper in the first version of this page was the backbone.

## Molecules: ZINC (graph-level regression)

`benchmarks/run_zinc.py` runs the same seven objectives on ZINC-12k (10k / 1k / 1k
molecules), again with one shared, untuned protocol: GINE with 4 layers of 128 units, batch
norm, mean readout, 100 epochs, batch 256, AdamW (lr `1e-3`, weight decay `1e-5`), 20% of
the atoms masked and 20% of the bonds dropped in each view. Two evaluations, test MAE (lower
is better), 3 seeds:

- **Frozen encoder**: ridge regression on the graph embeddings, L2 strength selected on
  validation. The same encoder untrained gives 0.593 ± 0.010.
- **Fine-tuned**: the pre-trained encoder and a new linear head are trained on the labels
  for 100 more epochs (best-validation epoch). Its reference is the † row: the same
  training from a random initialisation.

| Method | Frozen encoder | Fine-tuned | Rank |
|---|---|---|---|
| DGI | 0.932 ± 0.071 | 0.359 ± 0.018 | 9.2 ± 3.3 |
| GraphCL | 0.571 ± 0.006 | 0.341 ± 0.007 | 18.0 ± 0.2 |
| VICReg | 0.591 ± 0.010 | **0.335 ± 0.009** | 18.5 ± 0.2 |
| Barlow Twins | 0.574 ± 0.005 | 0.354 ± 0.012 | 15.7 ± 0.3 |
| BGRL | 0.700 ± 0.082 | 0.377 ± 0.007 | 5.6 ± 4.9 |
| AFGRL | 0.671 ± 0.046 | 0.379 ± 0.007 | 5.0 ± 1.3 |
| GraphDINO | 0.578 ± 0.009 | 0.361 ± 0.009 | 15.6 ± 0.3 |
| *Supervised* † | 0.396 ± 0.028 | 0.364 ± 0.007 | 7.5 ± 0.7 |

- **Frozen, self-supervision buys almost nothing here.** GraphCL, Barlow Twins and
  GraphDINO are 0.015–0.02 below the untrained encoder and VICReg equals it. BGRL and AFGRL
  collapse (effective rank about 5 of 128) and DGI is far worse than no training. None comes
  near the supervised encoder's embeddings (0.396).
- **Fine-tuned, two objectives help.** Pre-training with VICReg (−0.029) or GraphCL (−0.023)
  beats the same training from scratch by about three of its standard deviations. Barlow
  Twins, DGI and GraphDINO are within one. BGRL and AFGRL end up worse than no pre-training.
- Three seeds and no tuning: the differences among the first five rows of the fine-tuned
  column are small.

```bash
python benchmarks/run_zinc.py --model all supervised --seeds 3 --finetune-epochs 100
```

Results are in `benchmarks/zinc/` (one JSON per model and `summary.md`, commit `49e92ee`).

## In progress

- **ogbn-arxiv**: `benchmarks/reproduce_bgrl_arxiv.py` runs BGRL with the protocol of its
  paper (full-graph, 10,000 steps). Results will be added here once the three-seed run from
  committed code is complete.
- The sensitivity ablations above have to be rerun with the current probe.
- OGB graph-level benchmarks (ogbg-molhiv, ogbg-molpcba) are not wired up yet.
