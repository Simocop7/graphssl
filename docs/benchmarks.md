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
| Supervised references (†) | same encoder, trained on the public split's training labels only: *Supervised* keeps the protocol above; *std. recipe* uses dropout 0.5, Adam lr 0.01 + L2 5e-4 and the best-validation checkpoint (adapted from Kipf & Welling) and, with the GCN backbone, no normalization layer in the encoder, as in their model |

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
| *Supervised, std. recipe* † | 80.90 ± 0.76 | 67.72 ± 1.67 | 78.48 ± 0.92 |

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
| *Supervised, std. recipe* † | 79.55 ± 0.83 | 67.68 ± 1.48 | 78.41 ± 1.10 |

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
| GCN, *Supervised, std. recipe* † | 81.21 ± 0.70 | 68.62 ± 1.78 | 78.37 ± 1.20 |
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
- **The supervised references stay below the best SSL probe, with GCN by little.** The head
  of the standard recipe reaches 81.2 / 68.6 / 78.4 with GCN and 76.5 / 61.8 / 72.7 with GIN, against a
  best SSL linear probe of 82.5 / 69.6 / 81.6 and 81.3 / 65.6 / 80.4. The best SSL probe is the
  best of seven methods on the test split, so its margin over the reference (1.2 / 1.0 / 3.2
  with GCN) is an optimistic estimate.
- **With GCN the standard recipe is Kipf & Welling's model, and it lands near their
  numbers**: 0.3 / 1.7 / 0.6 below the 81.5 / 70.3 / 79.0 they report. It has dropout 0.5 and no
  normalization layer. With the batch normalization of the shared encoder the same recipe
  gives 77.9 / 65.0 / 75.8, 2.5 to 3.6 points less (`benchmarks/ablations/supervised_norm`, same
  seeds); the cause is not identified. The GIN encoder is the opposite case: without its
  normalization layer it falls to 57.2 / 49.2 / 71.6, so the GIN reference keeps batch norm.
  With GCN the best-validation epoch is early, within the first six epochs for 9 of 10
  seeds on Cora and within the first four for 8 of 10 on CiteSeer: validation accuracy
  peaks after a few Adam steps, at the accuracy reported.
- **Under the shared protocol the supervised encoder is not a strong reference.** Without
  regularization the supervised GCN's embeddings are no better for a linear probe than the
  untrained ones. Read the plain *Supervised* row as what this encoder gets from 60–140
  labels with the SSL protocol unchanged, not as an upper bound.
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

The benchmark fixes one protocol for every method. These ablations vary it: 5 seeds (10 for
the predictor), linear-probe test accuracy (%) unless stated, same optimizer, augmentations
and probe as above. They are separate from the benchmark, whose rows are unchanged. The
settings were compared on test accuracy, so the best row of a table is an optimistic
estimate for that setting. The untrained encoders over the same 5 seeds, as the reference:
GIN 41.52 ± 2.83 on Cora, 35.98 ± 1.93 on CiteSeer and 56.62 ± 1.78 on PubMed; GCN 73.02 ±
1.33, 57.42 ± 1.42 and 73.86 ± 2.04.

!!! note "Rerun with the current probe"
    Every number of this section was measured again on 2026-10-10/11 (commit `acae6df`,
    clean tree) with the probe of the benchmark. The first version of the section used the
    0.1.0 probe; those files are still in the ablation directories as history, and
    `render_ablation.py` reports the latest run of each configuration. For a trained GIN
    the linear column moved by at most 3 points, for a GCN by up to 9.6. What that changed
    in the reading is said where it applies.

### GraphDINO: reference DINO values degrade with training

GIN backbone, 300 / 1000 / 3000 full-batch steps, best value of each column in bold:

| GraphDINO setting | Cora 300 | Cora 1000 | Cora 3000 | CiteSeer 300 | CiteSeer 1000 | CiteSeer 3000 |
|---|---|---|---|---|---|---|
| Reference DINO values (benchmark row) | 63.52 ± 1.81 | 50.74 ± 1.53 | 39.60 ± 2.97 | 43.70 ± 2.53 | 34.06 ± 1.16 | 29.22 ± 1.04 |
| `norm_last_layer=True` | — | — | 38.10 ± 1.15 | — | — | 28.12 ± 1.30 |
| `teacher_temp=0.07` | — | — | 54.66 ± 2.68 | — | — | 37.94 ± 1.44 |
| `teacher_temp=0.07` + `norm_last_layer=True` | 64.34 ± 1.76 | 66.20 ± 3.09 | 46.60 ± 3.46 | 44.76 ± 3.69 | **45.72 ± 2.21** | 32.02 ± 1.61 |
| `ema_tau_base=0.9` | 65.78 ± 2.97 | 63.28 ± 4.17 | 61.56 ± 3.81 | 43.88 ± 2.75 | 43.82 ± 2.88 | 43.42 ± 2.60 |
| `ema_tau_base=0.9` + `teacher_temp=0.07` (library defaults) | **72.94 ± 3.77** | **68.02 ± 2.06** | **67.82 ± 2.46** | **48.02 ± 4.56** | 45.06 ± 4.81 | **46.10 ± 4.21** |

- With the reference values (EMA momentum 0.996 from the first step, teacher temperature
  0.04) accuracy drops as training continues. At 3000 steps it is at or below the untrained
  encoder (41.52 on Cora, 35.98 on CiteSeer).
- A faster teacher (`ema_tau_base=0.9`, annealed to 0.996 over the run) removes most of the
  decline. A softer teacher (`teacher_temp=0.07`, warmed up from 0.04) only slows it on its
  own. `norm_last_layer` alone has no measurable effect.
- Together, the faster and the softer teacher are the best setting in five of the six
  columns and tied in the sixth (CiteSeer at 1000 steps: 45.06 ± 4.81 against 45.72 ±
  2.21). The same on PubMed at 300 steps: 75.08 ± 0.94, against 70.50 ± 2.20 with the
  reference values and 72.88 ± 2.16 with the faster teacher alone.
- These two values are the **library defaults** (`GraphDINOConfig.ema_tau_base`,
  `HeadConfig.teacher_temp`). The benchmark row keeps the reference values, like every other
  untuned row of the table.

The same settings with the GCN backbone, at 300 and 3000 steps:

| GraphDINO setting | Cora 300 | Cora 3000 | CiteSeer 300 | CiteSeer 3000 |
|---|---|---|---|---|
| Reference DINO values (benchmark row) | 77.18 ± 1.14 | 71.86 ± 3.31 | 60.54 ± 1.21 | 47.84 ± 1.43 |
| `teacher_temp=0.07` + `norm_last_layer=True` | 77.52 ± 0.74 | 73.12 ± 1.99 | **62.36 ± 0.80** | 53.04 ± 1.76 |
| `ema_tau_base=0.9` | 75.80 ± 2.63 | 74.08 ± 1.53 | 55.66 ± 2.94 | 51.40 ± 2.76 |
| `ema_tau_base=0.9` + `teacher_temp=0.07` (library defaults) | **79.48 ± 1.69** | **77.28 ± 1.49** | 59.28 ± 2.90 | **55.04 ± 4.00** |

- **With GCN the defaults pay off in long runs, not in short ones.** At 3000 steps they are
  the best setting on both datasets, 5.4 and 7.2 points above the reference values. At 300
  steps they gain 2.3 points on Cora and nothing on CiteSeer, where the reference values
  (60.54 ± 1.21) do as well as the defaults (59.28 ± 2.90) and the softer teacher with
  `norm_last_layer` does better (62.36 ± 0.80).
- **Against the untrained GCN, GraphDINO adds little.** On CiteSeer three of the four
  settings are 1.9 to 4.9 points above it at 300 steps and all four are below it at 3000.
  On Cora the defaults are still 4.3 points above it at 3000 steps; the reference values
  are 1.2 below.
- What remains with the defaults: accuracy still declines from 300 to 3000 steps (Cora 72.94
  → 67.82 with GIN and 79.48 → 77.28 with GCN, CiteSeer 59.28 → 55.04 with GCN), and the
  spread across seeds is large (std up to 4.8 on CiteSeer).

**The library defaults in the benchmark protocol, 10 seeds.** The tables above compare
settings over 5 seeds on Cora and CiteSeer. This is the same comparison on the benchmark's
own terms: 300 steps, three datasets, both backbones, the benchmark's 10 seeds
(`benchmarks/ablations/graphdino_defaults`, commit `acae6df`). Linear probe:

| Backbone, setting | Cora | CiteSeer | PubMed |
|---|---|---|---|
| GCN, reference values (benchmark row) | 77.53 ± 1.22 | 60.16 ± 1.40 | 76.97 ± 2.08 |
| GCN, library defaults | 79.77 ± 1.41 | 59.62 ± 2.45 | 81.13 ± 0.88 |
| GIN, reference values (benchmark row) | 62.53 ± 1.91 | 43.34 ± 3.27 | 70.38 ± 3.23 |
| GIN, library defaults | 72.28 ± 2.61 | 48.23 ± 3.93 | 75.04 ± 1.21 |

- **With GIN the defaults gain 9.8, 4.9 and 4.7 points**, and as much in kNN accuracy
  (55.35 / 37.03 / 66.34 → 66.05 ± 3.63 / 42.57 ± 6.30 / 69.91 ± 1.64).
  GraphDINO is still 5 to 17 points below the best method of each column.
- **With GCN they gain 2.2 points on Cora and 4.2 on PubMed, and nothing on CiteSeer**
  (59.62 ± 2.45 against 60.16 ± 1.40). On PubMed GraphDINO reaches the leading
  group (81.13 against 81.24–81.59 for the three best methods); on CiteSeer it stays
  10 points below the best one and 1.9 above the untrained GCN.
- The two values were chosen on Cora and CiteSeer. PubMed took no part in the choice and
  gains 4.7 points with GIN and 4.2 with GCN.
- The effective rank goes down while both accuracies go up (Cora: 116 → 74 with GIN,
  193 → 171 with GCN): a higher rank is not better by itself.

### Training budget

From 300 to 1000 to 3000 steps, with the same constant learning rate:

| Method, encoder | Cora 300 | Cora 1000 | Cora 3000 | CiteSeer 300 | CiteSeer 1000 | CiteSeer 3000 |
|---|---|---|---|---|---|---|
| Barlow Twins, GIN | 79.20 ± 0.98 | 80.06 ± 1.53 | 78.12 ± 1.53 | 65.30 ± 1.40 | 65.10 ± 2.02 | 64.72 ± 2.05 |
| BGRL, GIN | 65.42 ± 2.06 | 70.70 ± 2.00 | 72.86 ± 0.83 | 46.38 ± 2.65 | 50.80 ± 3.19 | 49.26 ± 1.88 |
| BGRL, GIN, EMA from 0.9 | 64.78 ± 4.44 | 74.88 ± 1.01 | 67.70 ± 3.50 | 49.76 ± 3.11 | 55.28 ± 2.06 | 50.58 ± 3.01 |
| Barlow Twins, GCN | 81.86 ± 1.23 | — | 79.30 ± 0.65 | 68.92 ± 1.43 | — | 66.92 ± 1.19 |
| BGRL, GCN | 81.02 ± 1.41 | — | 76.98 ± 1.01 | 67.44 ± 0.63 | — | 60.00 ± 1.59 |

- **A longer budget helps only BGRL with GIN**, the row that starts lowest: +7.4 points on
  Cora at 3000 steps, +4.4 on CiteSeer at 1000 and +2.9 at 3000. It stays below Barlow
  Twins (72.86 against 78.12 on Cora, 49.26 against 64.72 on CiteSeer). Its effective rank
  grows from 9 to 36 of 256 dimensions on Cora.
- **Barlow Twins gains nothing from it.** With GIN its linear probe stays within 2 points,
  with no trend, while its kNN accuracy on Cora falls from 76.02 to 73.30 to 69.82. With GCN
  it loses 2.6 points on Cora and 2.0 on CiteSeer between 300 and 3000 steps.
- **With GCN, BGRL loses more**: 4.0 points on Cora and 7.4 on CiteSeer, where it ends 2.6
  points above the untrained GCN. Its effective rank does not fall (172 → 186 on Cora, 190 →
  181 on CiteSeer), so this is not a dimensional collapse.
- **A faster teacher** (EMA from 0.9 instead of 0.99) makes BGRL better at 1000 steps and
  worse at 3000. For AFGRL on Cora it gives 73.78 ± 1.30 against 71.50 ± 1.78 with GIN at
  300 steps, and no difference with GCN (79.10 against 79.24).
- GraphDINO with the reference values declines, as shown above.

Among the methods tested here (Barlow Twins, BGRL, GraphDINO, and AFGRL up to 1000 steps on
Cora), more steps improve only BGRL with GIN. Why accuracy falls with GCN is not identified.

### Predictor of BGRL and AFGRL

The default predictor of BGRL and AFGRL is Linear → BatchNorm → ReLU → Linear. The public
implementation of BGRL uses Linear → PReLU → Linear (`pred_norm: none`,
`pred_activation: prelu`). The benchmark rows (*default*) against the same runs with that
predictor (*reference*), 10 seeds each:

| Encoder | Method | Predictor | Cora linear | Cora kNN | CiteSeer linear | CiteSeer kNN | PubMed linear | PubMed kNN |
|---|---|---|---|---|---|---|---|---|
| GCN | BGRL | default | 80.82 ± 1.23 | 74.71 ± 1.19 | 67.05 ± 0.89 | 60.89 ± 1.58 | 79.85 ± 1.10 | 71.93 ± 1.31 |
| GCN | BGRL | reference | 81.30 ± 1.42 | 74.21 ± 1.29 | 66.57 ± 0.93 | 60.08 ± 2.22 | 79.71 ± 0.88 | 72.94 ± 1.42 |
| GCN | AFGRL | default | 79.18 ± 1.26 | 72.90 ± 1.73 | 62.63 ± 2.01 | 55.29 ± 1.89 | 78.43 ± 1.00 | 71.02 ± 1.15 |
| GCN | AFGRL | reference | 78.91 ± 0.94 | 70.98 ± 1.29 | 61.28 ± 1.60 | 53.98 ± 2.33 | 78.76 ± 1.00 | 72.94 ± 1.55 |
| GIN | BGRL | default | 63.77 ± 1.60 | 59.48 ± 3.75 | 47.68 ± 2.30 | 42.90 ± 2.21 | 68.52 ± 2.57 | 66.85 ± 2.75 |
| GIN | BGRL | reference | 63.28 ± 2.92 | 49.58 ± 2.85 | 53.32 ± 1.80 | 36.97 ± 3.73 | 64.92 ± 2.68 | 56.00 ± 4.15 |
| GIN | AFGRL | default | 70.75 ± 1.75 | 64.80 ± 2.13 | 49.32 ± 3.02 | 44.20 ± 3.61 | 74.51 ± 1.11 | 70.63 ± 2.11 |
| GIN | AFGRL | reference | 70.36 ± 3.03 | 67.83 ± 3.13 | 53.61 ± 2.63 | 48.24 ± 2.93 | 74.41 ± 1.38 | 68.48 ± 2.94 |

- **With GCN the predictor makes no difference**: the linear probe changes by at most 1.4
  points, less than one standard deviation.
- **With GIN the two metrics disagree.** The linear probe gains on CiteSeer (BGRL +5.6,
  AFGRL +4.3), loses on PubMed for BGRL (−3.6) and is unchanged elsewhere. BGRL's kNN
  accuracy falls on all three datasets, by 5.9 to 10.9 points and in every one of the 30
  seeds; AFGRL's goes up on Cora and CiteSeer (+3.0, +4.0) and down on PubMed (−2.2). The
  effective rank is lower with the reference predictor in every GIN cell: AFGRL 48 / 42 /
  41 → 21 / 19 / 12, BGRL 9 / 8 / 8 → 4 / 6 / 6 (Cora / CiteSeer / PubMed).
- **The default stays as it is.** With these batch-normalized encoders the reference
  predictor is no better with GCN, and with GIN it lowers the effective rank and, for BGRL,
  kNN accuracy. With a layer-normalized encoder it is the other way round: on ogbn-arxiv,
  with the protocol of the BGRL paper, the default predictor leaves BGRL below the
  untrained encoder (see
  [the ogbn-arxiv section](#the-predictor-decides-the-outcome)). Set `pred_norm: none`
  there.

The GIN benchmark rows are from commit `49e92ee` and the reference-predictor runs from
`acae6df`; between the two, the GIN encoder and the default predictor did not change.

### Backbone

The effect of the backbone used to be an ablation of this section. It is now part of the
benchmark: see the GCN and GIN tables above, each with its untrained encoder, on 10 seeds.
The first version of that ablation, with the 0.1.0 probe, had BGRL + GCN at 57.8 on
CiteSeer, barely above the untrained GCN; with the current probe the same five seeds give
67.44 ± 0.63 against 57.42 ± 1.42.

```bash
bash benchmarks/ablation_teacher_student.sh    # budget, EMA momentum, GCN for BGRL / Barlow Twins
bash benchmarks/ablation_graphdino.sh          # GraphDINO: teacher temperature, norm_last_layer
bash benchmarks/ablation_graphdino_ema.sh      # GraphDINO: faster teacher, PubMed, untrained encoders
for E in gcn gin; do                           # BGRL / AFGRL with the reference predictor
  python benchmarks/run_benchmark.py --dataset Cora CiteSeer PubMed --seeds 10 \
      --model bgrl afgrl --encoder $E --set pred_norm=none --set pred_activation=prelu \
      --out-dir benchmarks/ablations/predictor
done
python benchmarks/render_ablation.py benchmarks/ablations/teacher_student
python benchmarks/render_ablation.py benchmarks/ablations/graphdino_stability
python benchmarks/render_ablation.py benchmarks/ablations/predictor
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
# the standard-recipe supervised reference with the other normalization (ablation)
S="--model supervised_reg --out-dir benchmarks/ablations/supervised_norm"
python benchmarks/run_benchmark.py $D $S --encoder gcn --set encoder.norm_type=batch
python benchmarks/run_benchmark.py $D $S --encoder gin --set encoder.norm_type=none
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
    encoders at commit `49e92ee`, GCN at `acae6df`; the standard-recipe supervised reference
    with GCN is from 2026-10-13 at `fe85eb4`, the commit that made it Kipf & Welling's model
    (its earlier rows, with batch normalization and without the dropout that the GCN
    encoder then discarded, gave a head accuracy of 78.0 / 65.4 / 74.6). The earlier GIN files (0.1.0 probe,
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

Because the protocols differ, this comparison cannot tell a difference of configuration from
a defect of the implementation. The next section can.

## ogbn-arxiv: BGRL with the protocol of its paper

The tables above compare the objectives under one shared protocol. This section asks a
different question: given the configuration of its own paper, does the library's BGRL reach
the number published there? `benchmarks/reproduce_bgrl_arxiv.py` follows Thakoor et al.
(ICLR 2022, Table 8 and Appendix F) on ogbn-arxiv (169,343 nodes, 40 classes, standard
split):

| Setting | Value |
|---|---|
| Graph, training | symmetrized, full-graph, 10,000 steps |
| Encoder | GCN, 3 layers of 256 units, layer norm, weight standardization, PReLU |
| Predictor | Linear → PReLU → Linear, 256 hidden units (`pred_norm: none`, `pred_activation: prelu`) |
| Augmentation | `edge_drop` p=0.6 in both views, no feature masking |
| Optimizer | AdamW, weight decay `1e-5`, lr `1e-2` with 1,000 warm-up steps, then cosine decay to 0 |
| EMA | cosine schedule 0.99 → 1.0 |
| Evaluation | frozen embeddings → the same linear probe as above (`LogRegEvaluator`) and kNN (k=5, cosine) |
| Seeds | 3 (20 in the paper); mean ± sample std |
| Hardware | NVIDIA A2 (16 GB), 3.9 to 8.1 h per seed depending on how the GPU was shared |

Linear-probe accuracy (%). *Untrained encoder* is the same encoder at initialization
("Random-Init" in the paper); the last row is the difference between the two, per seed:

| | Validation | Test | Paper validation | Paper test |
|---|---|---|---|---|
| Untrained encoder | 70.23 ± 0.18 | 69.53 ± 0.03 | 69.90 ± 0.11 | 68.94 ± 0.15 |
| BGRL | 72.31 ± 0.14 | 71.32 ± 0.10 | 72.53 ± 0.09 | 71.64 ± 0.12 |
| BGRL − untrained | +2.08 ± 0.30 | +1.79 ± 0.08 | +2.63 | +2.70 |

- **The untrained encoder is within 0.33 / 0.59 points of the paper's** (validation / test).
  This checks the GCN encoder and the evaluation independently of any training.
- **Trained BGRL is 0.22 / 0.32 below the paper.** The three seeds agree (test 71.22 / 71.42
  / 71.33), so both differences are small but larger than the seed-to-seed standard
  deviation: close to the paper, not an exact match.
- **The gain over the untrained encoder is +2.08 / +1.79, against +2.63 / +2.70 in the
  paper**: smaller because our untrained encoder scores higher than the paper's and our
  trained one lower.
- **kNN agrees with the probe.** With no fitted classifier, test accuracy goes from 66.76 ±
  0.06 to 68.38 ± 0.18. Most of it comes early: for seed 0, 66.70 untrained, 67.76 after
  1,000 steps, 68.17 after 2,000, then between 68.1 and 68.3 to the end. The effective rank
  of the embeddings grows from 120 to 168 of 256 dimensions.
- **What differs from the paper.** The evaluation: the paper trains a linear classifier for
  100 AdamW steps on L2-normalized embeddings, here the library probe is fitted to
  convergence (hence the untrained-encoder row and the gain, which depend less on the
  classifier). The number of seeds. And the layer normalization, applied per node here and
  over the whole graph in the public PyTorch implementation of BGRL (`nerdslab/bgrl`), from
  which the predictor's layout is taken.
- **One method, one dataset.** The other six objectives have not been run with the
  protocols of their papers yet.

### The predictor decides the outcome

The run above uses the predictor of the public BGRL implementation. The default predictor
of BGRL and AFGRL in this library is Linear → BatchNorm → ReLU → Linear. The same protocol,
seeds and code with it (`--predictor library`), linear-probe accuracy and gain over the
untrained encoder (%), then kNN test accuracy, and the effective rank and final training
loss of each seed:

| Predictor | Validation | Test | Gain, validation | Gain, test | kNN test | Rank | Final loss |
|---|---|---|---|---|---|---|---|
| Linear → PReLU → Linear (reference implementation) | 72.31 ± 0.14 | 71.32 ± 0.10 | +2.08 ± 0.30 | +1.79 ± 0.08 | 68.38 ± 0.18 | 167 / 169 / 167 | 0.2103 / 0.2080 / 0.2084 |
| Linear → BatchNorm → ReLU → Linear (library default) | 68.97 ± 0.87 | 67.91 ± 0.78 | −1.26 ± 0.75 | −1.61 ± 0.78 | 64.63 ± 2.14 | 158 / 39 / 144 | 0.0042 / 0.0002 / 0.0065 |

- **With the default predictor BGRL ends below the untrained encoder in all three seeds**:
  test accuracy 67.40 / 67.51 / 68.81 against 69.5 untrained, and 3.4 points below the
  reference predictor on average. kNN agrees: 64.63 against 66.75 untrained.
- **The runs also differ from each other.** The training loss falls to almost zero, where
  it settles at 0.21 with the reference predictor. One seed loses most of its effective
  rank (39 of 256) and 4.3 points of kNN accuracy; another keeps the kNN accuracy of the
  untrained encoder and loses 0.7 points of linear accuracy. The standard deviation over
  seeds is 0.78, against 0.10.
- **The mechanism is not identified.** What is measured is an interaction: the failure
  appears with the layer-normalized encoder of this protocol, while with the
  batch-normalized encoders of the citation benchmark the reference predictor brings no
  overall gain (see [Predictor of BGRL and AFGRL](#predictor-of-bgrl-and-afgrl)). The
  library default is therefore unchanged. With a layer-normalized encoder, set
  `pred_norm: none`.

```bash
python -u benchmarks/reproduce_bgrl_arxiv.py --seeds 3
python -u benchmarks/reproduce_bgrl_arxiv.py --seeds 3 --predictor library \
    --out-dir benchmarks/reproductions/bgrl_ogbn_arxiv_library_predictor
```

The results are in `benchmarks/reproductions/bgrl_ogbn_arxiv/` (reference predictor) and
`benchmarks/reproductions/bgrl_ogbn_arxiv_library_predictor/` (default predictor): one JSON
each with the per-seed metrics and the kNN / effective-rank curve, both from commit
`acae6df` on a clean tree. The script also saves the trained model of every seed next to
the JSON (`bgrl_seed<N>.pt`, not committed).

## Molecules: ZINC (graph-level regression)

`benchmarks/run_zinc.py` runs the same seven objectives on ZINC-12k (10k / 1k / 1k
molecules), again with one shared, untuned protocol: GINE with 4 layers of 128 units, batch
norm, mean readout, 100 epochs, batch 256, AdamW (lr `1e-3`, weight decay `1e-5`), 20% of
the atoms masked and 20% of the bonds dropped in each view. Two evaluations, test MAE (lower
is better), 10 seeds:

- **Frozen encoder**: ridge regression on the graph embeddings, L2 strength selected on
  validation. The same encoder untrained gives 0.589 ± 0.011.
- **Fine-tuned**: the pre-trained encoder and a new linear head are trained on the labels
  for 100 more epochs (best-validation epoch). Its reference is the † row: the same
  training from a random initialisation.

| Method | Frozen encoder | Fine-tuned | Rank |
|---|---|---|---|
| DGI | 0.903 ± 0.182 | 0.363 ± 0.025 | 11.9 ± 3.9 |
| GraphCL | 0.578 ± 0.006 | 0.343 ± 0.024 | 18.0 ± 0.3 |
| VICReg | 0.585 ± 0.008 | **0.335 ± 0.027** | 18.9 ± 0.4 |
| Barlow Twins | 0.585 ± 0.010 | 0.363 ± 0.010 | 16.0 ± 0.4 |
| BGRL | 0.692 ± 0.060 | 0.369 ± 0.007 | 4.0 ± 1.9 |
| AFGRL | 0.641 ± 0.039 | 0.382 ± 0.013 | 8.9 ± 2.4 |
| GraphDINO | 0.579 ± 0.009 | 0.345 ± 0.017 | 16.0 ± 0.7 |
| *Supervised* † | 0.402 ± 0.020 | 0.369 ± 0.005 | 8.1 ± 0.8 |

- **Frozen, self-supervision buys nothing here.** GraphCL, GraphDINO, VICReg and Barlow
  Twins are 0.004 to 0.011 below the untrained encoder: within one of its standard
  deviations. BGRL and AFGRL are worse than it, with a low effective rank (4.0 and 8.9
  of 128), and DGI is far worse. None comes near the supervised encoder's embeddings
  (0.402).
- **Fine-tuned, three objectives help.** Pre-training with VICReg (−0.034), GraphCL
  (−0.025) or GraphDINO (−0.024) beats the same training from scratch (0.369 ± 0.005).
  VICReg is below the mean of the from-scratch runs in all ten seeds, GraphCL in nine,
  GraphDINO in eight and level with it in the other two. Barlow Twins (−0.006) and DGI
  (−0.005) are within their own standard deviation of it, BGRL equals it and AFGRL is
  worse (+0.013).
- **What the table does not show.**
    - Pre-training and fine-tuning use the same 10,000 molecules. The setting in which
      pre-training is expected to pay, many unlabeled molecules and few labels, is not
      tested.
    - Neither training has converged. The best fine-tuning epoch is the 80th or later in
      59 of the 70 runs, and between the 78th and the 97th for training from scratch. The
      gain is measured at a fixed, short budget.
    - One untuned configuration per method.

**Ten seeds, and why not three.** The first version of this table had 3 seeds
(`benchmarks/zinc/`, commit `49e92ee`). Its fine-tuned standard deviations for VICReg,
GraphCL and GraphDINO were 0.007–0.009, against 0.017–0.027 here, and it had GraphDINO
within one standard deviation of training from scratch. Seeds 0–2 are common to the two runs and
do not repeat. Each starts from the same encoder (its untrained MAE is equal to five
decimals in the two runs), yet the fine-tuned MAE of VICReg is 0.329 / 0.330 / 0.345 in
the first run and 0.354 / 0.294 / 0.313 in the second. A seed fixes the initialisation and
the order of the data, not the result: the runners do not ask for deterministic GPU
algorithms. Every number of this page is a draw from a distribution, and a rerun is another
draw.

```bash
python benchmarks/run_zinc.py --model all supervised --seeds 10 --finetune-epochs 100 \
    --out-dir benchmarks/zinc_seeds10
```

Results are in `benchmarks/zinc_seeds10/` (one JSON per model and `summary.md`, commit
`acae6df`, about 13 hours on an NVIDIA A2). The 3-seed run stays in `benchmarks/zinc/`.

## In progress

- OGB graph-level benchmarks (ogbg-molhiv, ogbg-molpcba) are not wired up yet.
