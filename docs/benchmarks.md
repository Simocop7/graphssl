# Benchmarks

## Citation networks: cross-method comparison

All 7 SSL methods, trained and evaluated under **one shared protocol**, so that differences
come from the objective rather than from per-method tuning:

| Setting | Value |
|---|---|
| Encoder | GIN, 2 layers, `hidden_dim=256`, batch norm, no dropout |
| Training | full-batch, 300 steps, AdamW (lr `5e-4`, weight decay `1e-5`) |
| Augmentation | `edge_drop` p=0.5 + `feat_mask` p=0.2, both views (GraphCL, BGRL, VICReg, Barlow Twins, GraphDINO); DGI uses node shuffling, AFGRL none |
| EMA (BGRL, AFGRL) | cosine schedule 0.99 → 1.0 over training |
| GraphDINO | reference DINO values: EMA momentum 0.996, teacher temperature 0.04 — not the library defaults, see [Sensitivity](#sensitivity-budget-teacher-and-backbone) |
| Split | public Planetoid train/val/test |
| Evaluation | frozen embeddings → linear probe (the 0.1.0 one: a linear layer trained with Adam for 100 steps, see the warning below) and kNN (k=5, cosine) |
| Seeds | 10; mean ± sample std |
| Hardware | NVIDIA A2 (16 GB), PyTorch 2.11 + CUDA 12.8, PyG 2.8 |
| Supervised references (†) | same encoder, trained on the public split's training labels only: *Supervised* keeps the protocol above; *std. recipe* uses dropout 0.5, Adam lr 0.01 + L2 5e-4 and the best-validation checkpoint (adapted from Kipf & Welling) |

No per-method or per-dataset hyperparameter search is performed: the goal is to compare
objectives at equal budget, not to reproduce each paper's best reported number. Test accuracy
(%), best SSL method per column in bold. The supervised rows (†) use labels during training:
they are reference points, not ranked.

!!! warning "The linear-probe numbers on this page predate the current probe"
    Every linear-probe number on this page was produced with the 0.1.0 probe: a randomly
    initialised linear layer trained for 100 Adam steps on raw features, with no
    regularisation. `LogRegEvaluator` is now an L2-regularised logistic regression fitted to
    convergence, with the L2 strength selected on the validation split, and the tables have
    not been rerun yet. The kNN and effective-rank columns are not affected.

    A spot check on the benchmark's own encoders (BGRL and Barlow Twins, 2 seeds per dataset)
    moved test accuracy by −2.6 to +2.4 points. The untrained-encoder references move more:
    the untrained GIN on PubMed goes from 44.1 to 57.0.

### Linear probe

| Method | Cora | CiteSeer | PubMed |
|---|---|---|---|
| DGI | 68.62 ± 2.77 | 51.10 ± 2.26 | 66.16 ± 2.83 |
| GraphCL | 78.06 ± 1.48 | 59.22 ± 2.50 | **80.11 ± 1.24** |
| BGRL | 64.18 ± 3.04 | 47.59 ± 2.22 | 68.67 ± 2.07 |
| AFGRL | 70.06 ± 2.64 | 47.50 ± 2.56 | 73.48 ± 1.27 |
| VICReg | 77.45 ± 1.78 | 61.44 ± 2.10 | 79.07 ± 1.62 |
| Barlow Twins | **78.13 ± 1.00** | **63.00 ± 1.31** | 77.82 ± 1.03 |
| GraphDINO | 62.12 ± 1.98 | 42.12 ± 2.58 | 70.97 ± 2.85 |
| *Supervised* † | 73.10 ± 1.54 | 51.81 ± 2.71 | 74.16 ± 1.72 |
| *Supervised, std. recipe* † | 76.10 ± 1.65 | 60.97 ± 3.42 | 71.36 ± 1.27 |

### kNN (k=5)

| Method | Cora | CiteSeer | PubMed |
|---|---|---|---|
| DGI | 65.38 ± 2.86 | 46.10 ± 3.63 | 64.47 ± 3.02 |
| GraphCL | 73.70 ± 1.39 | 60.69 ± 2.16 | 77.97 ± 1.33 |
| BGRL | 59.87 ± 3.53 | 43.07 ± 1.29 | 66.97 ± 2.86 |
| AFGRL | 65.13 ± 2.55 | 44.09 ± 2.81 | 70.63 ± 2.31 |
| VICReg | 73.08 ± 1.39 | 58.30 ± 2.50 | **78.28 ± 1.79** |
| Barlow Twins | **75.17 ± 1.06** | **62.23 ± 1.92** | 76.84 ± 1.33 |
| GraphDINO | 55.30 ± 1.56 | 36.92 ± 1.73 | 66.13 ± 3.79 |
| *Supervised* † | 74.14 ± 0.91 | 54.67 ± 1.38 | 74.63 ± 1.60 |
| *Supervised, std. recipe* † | 76.97 ± 1.51 | 62.04 ± 2.88 | 71.23 ± 1.03 |

### Reading the results

- **Redundancy reduction and contrastive learning lead.** Barlow Twins, VICReg and GraphCL
  form the top group on all three datasets, with seed std ≤ 2.5 points.
- **Teacher-student methods trail under this budget.** BGRL, AFGRL and GraphDINO sit below
  the top group everywhere. Within the family, AFGRL — which uses no augmentation at all —
  beats BGRL on Cora (+5.9) and PubMed (+4.8) and ties on CiteSeer. The
  [sensitivity ablations](#sensitivity-budget-teacher-and-backbone) below look at why: a
  longer budget alone doesn't close the gap, GraphDINO's reference hyperparameters fit short
  full-batch training badly, and the backbone matters a lot. One implementation cause is
  ruled out: GraphDINO's center was estimated from probabilities instead of logits until
  `20cce09`, and rerunning it after the fix moved every number by less than one std.
- **The supervised references are weak with this backbone.** Accuracy of the trained head
  (Cora / CiteSeer / PubMed): 73.2 / 51.7 / 74.4 under the shared protocol, 76.8 / 62.4 /
  71.4 with the standard recipe, which helps on Cora and CiteSeer and hurts on PubMed. Both
  stay below the best SSL linear probe on every dataset. That describes a GIN trained on
  60–140 labels, not SSL beating supervision in general: the 2-layer GCN of Kipf & Welling
  reports 81.5 / 70.3 / 79.0 with a similar recipe. The backbone effect hasn't been isolated
  here yet.
- **Quadratic cost is visible.** On PubMed (≈19.7k nodes) one 300-step run takes ≈440 s for
  GraphCL (pairwise NT-Xent) and ≈500 s for AFGRL (dense kNN + per-step k-means), versus
  ≈35–60 s for the other methods. Wall times are indicative only: some runs shared the GPU.

## Sensitivity: budget, teacher and backbone

The benchmark fixes one protocol for every method. These ablations vary it for the
teacher-student methods: 5 seeds, linear-probe test accuracy (%), same optimizer and
augmentations as above. They are separate from the benchmark, whose rows are unchanged.
The settings were compared on test accuracy, so the best row of a table is an optimistic
estimate for that setting.

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

### Backbone, and what an untrained encoder already gives

| 300 steps | GIN, Cora | GIN, CiteSeer | GCN, Cora | GCN, CiteSeer |
|---|---|---|---|---|
| Untrained encoder (0 steps) | 40.40 ± 2.15 | 36.14 ± 2.73 | 71.72 ± 1.67 | 56.96 ± 1.54 |
| Barlow Twins | 78.04 ± 1.33 | 63.00 ± 1.14 | 79.58 ± 0.61 | 61.30 ± 1.47 |
| BGRL | 64.50 ± 2.39 | 46.44 ± 1.45 | 78.24 ± 1.93 | 57.82 ± 2.72 |
| AFGRL | 70.58 ± 1.96 | — | 76.84 ± 1.49 | — |
| GraphDINO, reference values | 63.34 ± 1.25 | 42.24 ± 2.39 | 73.28 ± 0.23 | 53.04 ± 2.97 |
| GraphDINO, library defaults | 73.50 ± 2.92 | 47.76 ± 4.56 | 78.30 ± 2.19 | 54.48 ± 1.23 |

- **With GCN the teacher-student gap nearly closes on Cora.** BGRL goes from 13.5 points
  below Barlow Twins to 1.3 below, AFGRL from 7.5 to 2.7. The effective rank of BGRL's and
  AFGRL's embeddings goes from 9–48 with GIN to 171–189 with GCN.
- **But most of that is the backbone itself.** An untrained GCN already scores 71.72 on
  Cora and 56.96 on CiteSeer (72.18 on PubMed), against 40.40 and 36.14 (44.44) for an
  untrained GIN. Over that reference, with GCN, Barlow Twins adds +7.9 / +4.3 (Cora /
  CiteSeer), BGRL +6.5 / +0.9, and GraphDINO stays *below* it on CiteSeer with either
  setting (−3.9 with the reference values, −2.5 with the library defaults). With GIN every
  method is far above the untrained encoder (+22.9 to +37.6 on Cora).
- **A longer budget alone doesn't close the gap.** From 300 to 3000 steps with GIN, Barlow
  Twins is flat on Cora (78.04 / 78.44 / 77.62) and BGRL improves but stays below (64.50 /
  71.24 / 73.48). With GCN longer training lowers both (Barlow Twins 79.58 → 78.38, BGRL
  78.24 → 74.84).

So the benchmark's ranking measures how well each objective trains a GIN from scratch in 300
steps. It should not be read against GCN-based numbers from the literature without the
untrained-encoder reference.

```bash
bash benchmarks/ablation_teacher_student.sh    # budget, EMA momentum, GCN for BGRL / Barlow Twins
bash benchmarks/ablation_graphdino.sh          # GraphDINO: teacher temperature, norm_last_layer
bash benchmarks/ablation_graphdino_ema.sh      # GraphDINO: faster teacher, PubMed, untrained encoders
python benchmarks/render_ablation.py benchmarks/ablations/teacher_student
python benchmarks/render_ablation.py benchmarks/ablations/graphdino_stability
```

## Reproduce

```bash
python benchmarks/run_benchmark.py --dataset Cora CiteSeer PubMed --model all --seeds 10
python benchmarks/render_tables.py            # Markdown tables from the saved JSON files
python benchmarks/render_tables.py --latex    # also LaTeX (booktabs)
```

Each run is saved under `benchmarks/results/<Dataset>/<model>__<UTC-timestamp>.json` with
per-seed metrics (including training wall time), aggregate mean/std, the exact
hyperparameters, the git commit (plus whether tracked files had uncommitted changes) and
package versions. Old files are never overwritten, so results accumulate as a reproducible
history; `render_tables.py` uses the latest file per (dataset, model). See
[`benchmarks/README.md`](https://github.com/Simocop7/graphssl/blob/main/benchmarks/README.md)
for the known gaps (only Planetoid datasets are wired up so far). The supervised references
run with `--model supervised supervised_reg` (they aren't part of `--model all`).

!!! note "Provenance of the committed results"
    The 20 files recorded at commit `e295451` report `"dirty": true`. That is a false
    positive of the runner at the time — it counted untracked files (logs, the results
    themselves) as modifications; fixed in `dae32cb`. The code that ran is exactly
    `e295451`. GraphCL on PubMed was run at `dae32cb`, whose chunked NT-Xent is numerically
    identical to the unchunked one used for Cora and CiteSeer. GraphDINO (after its center
    fix) and *Supervised* were run at `20cce09`, *Supervised, std. recipe* at `558af8e`, all
    with `"dirty": false`.

## Comparison with the original BGRL numbers

The paper
([`paper.tex`](https://github.com/Simocop7/graphssl/blob/main/paper.tex), Results section)
places our BGRL numbers next to the original BGRL paper's small-dataset evaluation (Appendix C,
Table 7 of Thakoor et al.) and lists the protocol differences explicitly: random vs. public
splits, GCN vs. GIN encoder, tuned scikit-learn logistic regression vs. an untuned linear
probe. The gap is consistent with those differences; it is not a like-for-like comparison.

## In progress

ogbn-arxiv and OGB graph-level benchmarks (ogbg-molhiv, ogbg-molpcba) need larger compute and
are next. `examples/ogbn_arxiv_bgrl.py` and `examples/zinc_bgrl.py` already implement the
training scripts; `run_benchmark.py` still has to be extended to those datasets.
