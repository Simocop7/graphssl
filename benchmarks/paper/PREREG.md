# Analyses fixed in advance

The rules of the seven analyses of the study that were fixed before the results they apply to,
copied verbatim, by script, from the planning file as it stood at the commit that fixed each one.
Those commits belong to a development history that also holds private notes and is not part of
this repository. This is the authors' own record, not an external registration. What had been seen
by then is listed in the appendix of the paper ("Analyses fixed in advance"). In this public copy,
five personal notes about who asked for an analysis are replaced by "[omitted]"; the rules are otherwise
unchanged.

## Label-free prediction test

Commit `0bd9039`, 2026-10-26 04:05 UTC.

> **Part 2: label-free prediction of the gain. Fixed on 2026-10-26, before any result on the
> test datasets was looked at; do not change it after.**
> - **Target:** the gain over the untrained encoder, GraphSSL probe, standard split (Planetoid
>   public, ogbn-arxiv official, random split 0 elsewhere), per (dataset, backbone, method),
>   averaged over seeds.
> - **Candidates, a closed list. Each is tested alone, with no combinations.** Every metric is
>   averaged over seeds, and higher is predicted to mean a larger gain.
>   1. effective rank (RankMe, Garrido et al. 2023)
>   2. distance from the untrained encoder, 1 − linear CKA, same seed
>   3. smoothing gained, (edge_cos − pair_cos) of the model minus that of the untrained encoder,
>      same seed
>   4. uniformity, negated (−uniformity; Wang & Isola 2020)
> - **Statistic:**
>   - Spearman ρ between metric and gain across the 7 methods, within each (dataset, backbone)
>   - primary: the mean ρ over the test groups, with a 95% seed-bootstrap interval (seeds
>     resampled within each cell, everything recomputed)
>   - secondary: the number of groups with ρ > 0, and the selection regret (the gain of the
>     best method minus that of the method the metric ranks first, in points) against the
>     expected regret of a uniformly random choice
> - **Development set:** Cora, CiteSeer (all backbones). Decisions about the code and the
>   presentation may use it.
> - **Test set, homophilous:** PubMed, Amazon Computers, Amazon Photo, Coauthor CS, ogbn-arxiv.
>   arxiv-year is reported separately, because smoothing is predicted *not* to help there.
> - **"Predicts"** means the interval of the test mean ρ excludes 0 *and* the mean regret is
>   below random. Any other outcome is reported as it is, with no further metrics tried.
> - **Sensitivity, not part of the decision:** the same with the gain measured by kNN.
>
> **Out of scope** (go in Limitations):
> - heterogeneous graphs (ogbn-mag)
> - graph-level tasks in the main text (ZINC stays in the appendix)
> - training budget beyond the existing ablation
> - tuning beyond optimizer × learning rate
> - methods or backbones beyond those listed

## Definitions of the trajectories (E4)

Commit `2c83ccc`, 2026-10-27 05:03 UTC.

> ## E4 in detail (added 2026-10-27)
>
> **Why.** [omitted] asked (27 Oct) where models collapse, what changes in the embeddings,
> whether a model collapses at once or starts well and then loses it, and whether the optimizer
> changes the trajectory. It also answers an objection to the whole study: every E1/E3 number is
> an endpoint after 300 steps, and a reviewer can ask whether the ranking is a property of that
> stopping point. E4 measures the stopping point as a factor *within* the existing budget (it
> does not extend it: that stays out of scope).
>
> **How.** `run_benchmark.py --trajectory`: the final evaluation (GraphSSL probe with L2 picked on
> validation, kNN, effective rank), the label-free geometry of M (pair_cos, edge_cos, uniformity,
> feature scale) and the linear CKA to the step-0 embeddings, at each listed step; the loss of
> every step. The evaluations run on a forked RNG: the step-300 results are bit-identical to the
> same run without `--trajectory` (checked on Cora, DGI + GIN; without the fork they were not,
> because `LogRegEvaluator` builds an `nn.Linear`, whose default init draws random numbers, and
> zeroes it). Queues `E4-gcn`, `E4-gin`, `E4-transformer` of `paper_manifest.py`, worktree
> `graphssl-runs-e4`, 2 threads each. E4's AdamW 5e-4 cell is its own run (Mac CPU, 2 threads,
> 5 seeds): compare it with E4's other cells, not with E1 or E3.
>
> **What had been seen before these definitions were fixed:** E3's endpoints, and one pilot
> trajectory (DGI + GIN, Cora, seed 0: loss ≈ 0 by step 75, test accuracy peaks at step 100,
> 69.7, ends at 66.1; pair_cos falls 0.85 → 0.54 by step 30, then back to 0.84).
>
> **Questions and definitions, fixed before the E4 runs.** Per (cell, dataset, backbone, method),
> curves averaged over seeds, intervals by seed bootstrap. Descriptive: the thresholds are fixed
> now so that the summary is not tuned to the data, and every curve is shown.
> 1. *Is the ranking at step 300 the ranking elsewhere?* Kendall τ between the method ranking
>    at step t and at step 300, per (cell, dataset, backbone), against t. And the ranking with
>    each method stopped at its own best-**validation** step against the one at 300 (would early
>    stopping on validation change the conclusions of E1?).
> 2. *Shape of a trajectory* (test accuracy, GraphSSL probe):
>    - *late decline*: test accuracy at the best-validation step exceeds the one at 300 by
>      at least 1 point, with a seed-bootstrap 95% interval that excludes 0;
>    - *never learns*: the best test accuracy over all steps is less than 1 point above step 0;
>    - otherwise *rises and holds*.
> 3. *Collapse, and when* (first step at which it holds, seed-averaged):
>    - *divergence*: non-finite embeddings in any seed;
>    - *dimensional*: effective rank below 10;
>    - *directional*: pair_cos above 0.95.
>    Reported with the step-0 value, since an untrained GIN is already at pair_cos 0.85.
> 4. *Optimizer:* per method, the shape class, the peak step and the decline in each cell; and
>    the E3 claim re-read at each method's best-validation step instead of at step 300.

## Training budget (E5)

Commit `63396cc`, 2026-10-28 06:33 UTC.

> **E5: training budget** (fixed now, before any run).
> - *Question.* Does a ten times longer budget change the gain over the untrained encoder where
>   the paper says it is small (GCN on the Amazon and Coauthor graphs), and does the grouping of
>   representations by backbone survive it?
> - *Design.* One device per dataset for the whole of E5. Complete runs at **300 and 3,000
>   steps** (separate runs: BGRL, AFGRL and GraphDINO anneal their teacher momentum over the run
>   length, so step 300 of a long run is a different model), shared protocol otherwise (AdamW
>   5e-4, constant learning rate), seeds 0–4, the eight objectives. The 3,000-step runs are also
>   evaluated at steps 300, 600, 1,000, 2,000 and 3,000 (`--trajectory`, forked RNG). Models
>   saved at the end of both budgets.
>   - primary: GCN on Amazon Computers, Amazon Photo, Coauthor CS; Cora as the citation-graph
>     comparison
>   - grouping: GIN and Transformer on Cora as well, for the cross-backbone CKA control
>   - cost rule (fixed now, independent of results): an objective whose measured cost exceeds 45
>     minutes per 3,000-step seed on its device runs with seeds 0–2 instead of 0–4
> - *Quantities.* Per (dataset, objective), regularized probe, standard split, gains over the
>   untrained encoder of the same seed: G300 (300-step run), G3000 (3,000-step run, end),
>   Gstop (3,000-step run at its best mean-validation evaluation step). Δ = Gstop − G300, mean
>   over objectives, 95% seed-bootstrap interval (seeds resampled, paired).
> - *Decision rule for claim C3* ("with a GCN the average objective adds 0.2 to 1.4 points on
>   the Amazon and Coauthor graphs"). If on any of the three datasets Δ ≥ 1.0 point with an
>   interval that excludes 0, the paper reports the small gains as budget-dependent and states
>   them per budget. Otherwise C3 stands, and E5 is cited as the test it passed. Per-objective Δ
>   is reported either way; the rule uses the mean over objectives only.
> - *Grouping (descriptive, rule fixed now).* On Cora at 3,000 steps, the backbone grouping
>   "holds" if the mean CKA of same-backbone, other-objective pairs across seeds exceeds that of
>   same-objective, other-backbone pairs by at least 0.10 (at 300 steps: 0.47 vs 0.29). Also
>   reported: CKA to the untrained encoder at 300 vs 3,000 steps.
> - *Cost, measured 28 Oct on the Mac GPU* (60 steps of each, GCN, probes included): Amazon
>   Photo AFGRL 41 s, GraphCL 42 s, GraphMAE 50 s per 60 steps, so every objective is under 45
>   min per 3,000-step seed there: no three-seed exception on E5-mps. Coauthor CS on the Mac GPU
>   would need about 110 min per GraphCL seed: E5-V's costs are measured on the A2 before its
>   launch, and the exceptions written into `E5_THREE_SEEDS` then. Total E5: about 1.5–2 days of
>   GPU over the two machines (the 6–10 h said on 28 Oct was too low).
> - *Seen before fixing this rule:* E1 and E4 (300 steps); the teacher-student ablation of
>   CLAUDE.md (BGRL and Barlow Twins with GCN on Cora/CiteSeer at 3,000 steps, both lower than at
>   300). No run on the Amazon or Coauthor graphs beyond 300 steps.

## Optimizer and learning rate on the Amazon and Coauthor graphs (E3)

Commit `9263d36`, 2026-10-29 09:03 UTC.

> ## E3 on the Amazon and Coauthor graphs (added 2026-10-29, fixed before any of its runs)
>
> - *Why.* C3's small GCN gains (the average objective adds 0.2–1.4 points on Amazon Computers,
>   Amazon Photo, Coauthor CS) are the claim most exposed to "the protocol is untuned" (O1), and
>   E3 tuned only Cora / CiteSeer, where with a GCN tuning lifts the average objective by +0.9
>   [0.2, 1.1] (Cora) and +3.5 [2.9, 4.0] (CiteSeer). E5 tests the budget, not the learning rate.
>   [omitted]
> - *Design.* E3's ten cells unchanged, its AdamW 5e-4 cell rerun as this extension's own shared
>   cell, 300 steps, seeds 0–4, the eight objectives, **GCN only**, on the three datasets. Each
>   dataset on the device of its E1 runs: Amazon Photo then Amazon Computers on the Mac CPU (3
>   threads), Coauthor CS on the Mac GPU (2 threads, after E5-mps). Worktree
>   `graphssl-runs-e3x`, queues `E3X-amazon` and `E3X-cs` (`paper_manifest.py`), results in
>   `e3/<cell>/<dataset>/` beside E3's. Selection as in E3 (`analyze_e3.py`): each objective at
>   the cell with its best mean validation accuracy, a cell with a diverged seed never chosen.
> - *Decision rule for C3.* Mean lift = tuned − shared, averaged over the objectives, 95% seed
>   bootstrap with the selection inside every replicate. If on any of the three datasets the mean
>   lift is ≥ 1.0 point with its interval above 0, C3 is reported as protocol-dependent: the
>   paper gives the gains at the shared cell and at each objective's tuned cell side by side, and
>   the abstract says the small GCN gains hold at the shared protocol. Otherwise C3 stands and
>   this extension is cited as the test it passed. Per-objective lifts and Kendall τ are reported
>   either way. `analyze_e3.py` prints the verdict.
> - *Grid edge (conditional, fixed now).* If a dataset's verdict is "stands" and an objective's
>   chosen cell is the lowest or highest learning rate of its optimizer, that objective gets one
>   more cell beyond that edge on that dataset, in the grid's 1–5 pattern (AdamW 5e-5 or 5e-2, SGD
>   5e-4 or 5e-1), same device and seeds, and the rule is re-evaluated with it. Not needed for a
>   "protocol-dependent" verdict, which a wider grid would only strengthen.
> - *Not added, on purpose.* GIN and the Transformer: their gains on these datasets are already
>   1–14 points and tuning only adds to them, so it cannot weaken a claim, and E3 already measures
>   their ranking under tuning. PubMed: gains of 5.6 points are not at risk. Tuning beyond
>   optimizer × learning rate (augmentations, width, depth) and the joint learning rate × budget:
>   Limitations, with the BGRL paper's tuned gains on these datasets (+1.1 to +3.9) as the outside
>   reference. **ogbn-products is cut** (never started): ogbn-arxiv answers O4, and no claim waits
>   for it.
> - *Cost*, from E1's per-seed times on the same devices (300 steps, the eight objectives, E1 ran
>   beside other queues): Amazon Photo ~960 s per seed → ~13 h; Amazon Computers ~2,300 s → ~32 h;
>   Coauthor CS ~3,000 s on the Mac GPU → ~42 h.
> - *Free check, not a factor.* The Amazon shared cell runs on E1's device and thread count, so
>   its seeds 0–4 should repeat E1's: reported as a replication point.

## Parameter-free reference on WikiCS (E6)

Commit `6cf6db9`, 2026-11-13 17:16 UTC.

> ## E6: a parameter-free reference (added 2026-11-14, fixed before any WikiCS run)
>
> - *Why.* A post hoc check on 13 Nov ([omitted]; analysis of saved models only) found two things the paper does not show.
>   (1) The raw-features baseline depends on the probe: our probe standardizes the features,
>   which hurts the sparse bag-of-words of the Planetoid graphs (raw features, Cora / CiteSeer /
>   PubMed, ours vs liblinear: 52.6 / 47.5 / 67.0 vs 60.2 / 62.4 / 72.9, both already in the
>   probe files), so "what the architecture adds" is probe-dependent. (2) Logistic regression on
>   the features propagated twice over the GCN-normalized adjacency (A^2X, SGC's features, no
>   parameters) matches the trained GCN objectives: with each representation scored by the
>   regularized probe that is better on validation (ours or liblinear), it is above all eight
>   GCN objectives on Amazon Computers, Amazon Photo and CiteSeer, seven of eight on Cora and
>   Coauthor CS, five of eight on PubMed; 15 of the 144 SSL models (any backbone) are above it.
>   Both are post hoc and are reported as such. FastGCL (Wang et al., arXiv 2205.00905) reported
>   untrained SGC level with BGRL on the Amazon and Coauthor graphs. [omitted]
> - *Claim C10 (post hoc, six datasets).* With a GCN under the shared protocol, the average
>   objective does not exceed logistic regression on A^2X: mean over the eight objectives of
>   (objective − A^2X), probe chosen on validation per representation, standard split: Cora
>   −2.9, CiteSeer −6.6, PubMed −0.9, Amazon Computers −1.2, Amazon Photo −0.6, Coauthor CS −0.9.
> - *Out-of-sample test.* WikiCS (Mernyei & Cangea 2020; 11,701 nodes, dense 300-d features, 10
>   classes, edge homophily 0.65), never used in this study. Its public split 0 (580 / 1,769 /
>   5,847) is the standard split; the 10 random 10/10/80 splits of E2 for the split check. E1
>   there: eight objectives, the untrained encoder and the two supervised references, GCN, GIN and
>   the Transformer, 10 seeds, shared protocol, models saved; all on the VM's A2 (one device),
>   from a clean worktree at the commit that adds this section. Scored on the Mac CPU with the E2
>   protocols, plus the input-feature rows (raw, A^1X, A^2X, PCA-256 of A^2X).
> - *Probe of the rule.* For each representation (every model seed, the untrained encoder, the
>   input-feature rows), the regularized probe, ours or liblinear, with the higher validation
>   accuracy on the standard split (ties: ours). K = 2 (the depth of the encoders) is fixed: A^1X
>   and the PCA are reported, never chosen.
> - *Decision rule for C10.* D = mean over the eight GCN objectives of (seed-mean test accuracy −
>   A^2X test accuracy), 95% seed bootstrap (seeds resampled within each objective; A^2X has no
>   seed). If D ≥ 1.0 point with an interval that excludes 0, the paper states C10 for the six
>   datasets only and reports WikiCS as the counterexample, with D. Otherwise C10 stands and
>   WikiCS is cited as the out-of-sample test it passed. Reported either way: D per objective,
>   the number of GCN objectives above A^2X, the models of any backbone above it, D under ours
>   and under liblinear separately, and on the random splits (our probe).
> - *Not part of the rule, reported:* raw features vs the untrained encoders under each probe on
>   dense features (whether the probe-dependence of (1) is specific to bag-of-words).
> - *WikiCS enters nothing else*: not the variance shares, the rankings, the label-free test or
>   any other claim's numbers. The six-dataset analyses stay byte-identical (results in
>   `benchmarks/paper/e6/`, models in `models_e6*`).
> - *Seen before fixing this rule:* everything of the six datasets and ogbn-arxiv; on WikiCS,
>   FastGCL's Table 1 (another protocol: untrained GCN 77.97, untrained SGC 77.44, DGI 77.31,
>   GRACE 78.50, BGRL 79.98, AFGRL 77.62; i.e. objectives above the untrained encoders by up to
>   2.5 points, the largest of their five datasets) and a 3-step smoke run of GraphCL and DGI
>   with a GCN (seed 0, Mac CPU: 78.1 and 78.0, our probe). No input-feature row and no trained
>   model on WikiCS was scored.
> - *Also in E6, no rule (descriptive):* the input-feature rows of the six datasets, ogbn-arxiv
>   and arxiv-year under the E2 protocols (`paper_e6.sh baselines`); the split-half noise
>   ceiling of Kendall's τ (whether the disagreement between backbones exceeds what two halves
>   of the seeds of one backbone show).
> - *Cost.* Preflight on the A2 first (every command, 3 steps). Expected a few hours of GPU for
>   training, about an hour of Mac CPU for the scores.

*Note to the E6 rule, not part of it (recorded in the planning file on 14 Nov, 01:37 HKT, at
commit `16e660f`, before any WikiCS model was scored).* The six-dataset numbers in *Why* and
*Claim C10* above came from a scratch script that keyed probe rows by name, so the untrained BGRL
rows replaced the trained BGRL. Keyed by checkpoint (`analyze_propagation.py`), D is Cora −1.9,
CiteSeer −5.5, PubMed −0.2, Amazon Computers −1.0, Amazon Photo −0.6 and Coauthor CS −0.7, and 16
of the 144 models of any backbone are above A^2X (not 15). The counts of GCN objectives above
A^2X are unchanged, and the rule on WikiCS does not use these numbers.

## Normalization statistics at evaluation (E7)

Commit `f0fb47d`, 2026-11-19 14:46 UTC.

> ## E7: normalization statistics at evaluation (added 2026-11-19, fixed before its runs)
>
> - *Why.* A review of the draft on 19 Nov ([omitted]) found that the untrained encoder is evaluated with the
>   initial running statistics of its BatchNorm layers (mean 0, variance 1: in eval mode the layers
>   leave their input unchanged), while every trained model holds statistics accumulated over its
>   training passes. The reference therefore differs from a trained model in two ways, its
>   weights and its normalization statistics, and part of a measured gain can come from the
>   second. The embedding scale the paper uses to explain the 100-step probe's inflation (5–290×
>   smaller untrained embeddings) is this difference.
> - *Definition.* "Graph statistics": the running statistics of every BatchNorm layer are reset
>   and recomputed as the cumulative average over one forward pass of the model's own `forward()`
>   on the clean full graph, with only the BatchNorm layers in train mode (dropout and every other
>   layer in eval mode), no gradient, no label; then the usual eval-mode extraction. Implemented as
>   `extract_embeddings(..., norm_stats="graph")` (on a copy of the model) and
>   `compare_probes.py --norm-stats graph`. "Running statistics": the evaluation used so far
>   (`--norm-stats running`, the default).
> - *Scope.* Every saved E1 model of the six datasets except the supervised references: the eight
>   objectives and the untrained encoder, three backbones, 10 seeds (270 checkpoints per dataset).
>   The supervised references are not ranked and enter no claim tested here (the GCN of the
>   standard recipe has no normalization layer). E2's probes on the standard split: adam100 (20
>   initializations), adam5000, liblinear, ours, and kNN; no random splits, no `_std` / `weakl2`
>   controls. **Both conditions are scored by the same script on the same device and thread count
>   (Mac CPU, OMP_NUM_THREADS=3, the setting of E2)**, so every comparison is paired, and the
>   running rows double as a check that E2 reproduces (preflight, 19 Nov: two Cora checkpoints
>   identical to E2 to the digit under adam100, liblinear, ours and kNN). ogbn-arxiv and arxiv-year
>   (standard split): the 81 checkpoints with adam100, ours and kNN, both conditions on one device
>   (the A2 if reachable, otherwise the Mac GPU); they are descriptive and enter no verdict below.
>   WikiCS is not included (E6: WikiCS enters nothing else). Results in `benchmarks/paper/e7/`.
> - *What stays as it is.* Every analysis fixed before (label-free test, E4, E5, E3 on Amazon and
>   Coauthor, E6) and every main table use running statistics, as when they were fixed. E7 is
>   reported next to them, and the text says where a claim depends on the statistics.
> - *Rule R1 (C3, small GCN gains).* On Amazon Computers, Amazon Photo and Coauthor CS, the mean
>   gain of the eight GCN objectives over the untrained GCN, our probe, standard split, with the
>   objectives and the untrained encoder both under graph statistics, minus the same mean gain under
>   running statistics; 95% bootstrap over the 10 seeds (a seed's models resampled together, gains
>   paired by seed). If on any of the three it is ≥ 1.0 point with its interval above 0, C3 is
>   reported as dependent on the normalization statistics, with both gains side by side. Otherwise
>   C3 stands under both.
> - *Rule R2 (C10).* E6's D (valsel probe: ours or liblinear, whichever is better on validation, per
>   representation; mean over the eight GCN objectives of the seed-mean test accuracy minus that of
>   Â²X; 95% seed bootstrap) with the objectives under graph statistics, on each of the six
>   datasets. If D ≥ 1.0 with its interval above 0 on any of them, C10 is stated "with the models'
>   own statistics" and that dataset is reported. Otherwise C10 stands under both.
> - *Rule R3 (C4 and C5, the probe's two biases).* Inflation of a cell = mean gain of the eight
>   objectives under adam100 minus their mean gain under our probe (standard split, both the
>   objectives and the untrained encoder under the same statistics). I_run and I_graph = the mean
>   inflation over the 9 cells with many labels (Amazon Computers, Amazon Photo, Coauthor CS × three
>   backbones) under each condition. If I_graph < I_run / 2, the paper attributes the inflation
>   mainly to the untrained encoder's initial normalization statistics, not to the probe alone,
>   in the abstract and in the probe section. Otherwise the inflation is reported as a property of
>   the probe under both. Deflation: if under graph statistics adam100 still deflates the mean gain
>   (inflation < 0) in at least 6 of the 9 cells with 20 labels per class (as under running
>   statistics), the overfitting bias stands; otherwise it is reported as dependent on the
>   statistics.
> - *Descriptive, no rule:* the share of the variance of the gain explained by the backbone (our
>   probe, standard split, both conditions); the spread of the untrained backbones; the untrained
>   GIN's effective rank on the Amazon graphs; the gains under kNN; ogbn-arxiv and arxiv-year.
> - *Seen before fixing this rule:* everything of E1–E6 under running statistics; a scratch
>   diagnostic on 19 Nov of the **untrained encoders only** (Mac CPU, 4 threads; seeds 0–2 on Cora
>   and CiteSeer, 0–1 on Amazon Computers and Coauthor CS; adam100 with 5 initializations),
>   running → graph statistics, ours / kNN / adam100: Cora GCN 72.4 → 71.9 / 59.6 → 65.6 / 71.0 →
>   69.9, GIN 40.7 → 50.2 / 27.3 → 47.2 / 40.1 → 50.9, Transformer 49.4 → 55.4 / 29.7 → 46.1 /
>   46.3 → 49.1; CiteSeer GCN 58.4 → 58.9 / 50.2 → 50.8 / 58.0 → 54.0, GIN 36.6 → 40.3 / 27.2 →
>   32.6 / 36.6 → 39.6, Transformer 45.2 → 45.2 / 33.2 → 38.5 / 40.9 → 40.0; Amazon Computers GCN
>   87.3 → 87.4 / 87.0 → 85.0 / 80.7 → 87.3, GIN 79.7 → 77.7 / 76.9 → 78.6 / 67.4 → 76.5 (rank 4.8
>   → 62.5), Transformer 76.3 → 79.4 / 50.8 → 61.8 / 74.8 → 76.7; Coauthor CS GCN 91.1 → 90.8 / 89.5
>   → 89.8 / 82.5 → 90.4, GIN 80.7 → 80.0 / 69.1 → 78.7 / 57.9 → 77.5, Transformer 86.5 → 86.5 /
>   65.4 → 83.4 / 83.6 → 84.4; and the preflight's graph-statistics row of the untrained GCN, Cora
>   seed 0. No trained model under graph statistics, and nothing on PubMed, Amazon Photo or
>   ogbn-arxiv under graph statistics.
> - *Cost.* From E2's probe times: about 7.7 h of single-stream Mac CPU for both conditions on the
>   six datasets, in two streams; ogbn-arxiv a few hours on the A2 (about twice that on the Mac
>   GPU). Preflight first (every dataset, both conditions, two checkpoints).

## Row-normalized features (E8)

Commit `f0fb47d`, 2026-11-19 14:46 UTC.

> ## E8: row-normalized features (added 2026-11-19, fixed before its runs)
>
> - *Why.* The study feeds every encoder the node features as PyG distributes them. DGI, GRACE and
>   the PyGCL examples divide each Planetoid node's features by their sum (PyG's
>   `NormalizeFeatures`). C10 compares the GCN objectives with Â²X computed from the same raw
>   features; a reviewer can ask whether it holds with the usual preprocessing.
> - *Design.* Cora, CiteSeer and PubMed (the bag-of-words graphs where row normalization is the
>   convention), GCN only (C10 concerns the GCN), the eight objectives and the untrained encoder,
>   seeds 0–4, the shared protocol otherwise, row-normalized features (`run_benchmark.py
>   --features rownorm`), models saved in `models_e8*/`, results in `benchmarks/paper/e8/`. Mac
>   CPU, 3 threads (E1's device for these datasets), one stream. Scored with `compare_probes.py
>   --features rownorm` with ours and liblinear on the standard split, kNN, and the input rows of
>   the row-normalized features (raw, Â¹X, Â²X, PCA-256 of Â²X).
> - *Rule (C10).* D_rn = mean over the eight GCN objectives of (seed-mean test accuracy over seeds
>   0–4 − test accuracy of Â²X of the row-normalized features), valsel probe as in E6, 95% seed
>   bootstrap. If D_rn ≥ 1.0 with its interval above 0 on any of the three datasets, C10 is stated
>   for raw features only and that dataset is reported. Otherwise C10 holds with either
>   preprocessing. Reported either way: D_rn per objective; the objectives against the better of
>   the two Â²X (raw or row-normalized); E1's D on seeds 0–4; the untrained GCN.
> - *Seen before fixing this rule:* nothing with row-normalized features under the shared protocol.
>   Known from elsewhere: the GraphMAE validation of 28 Oct (CLAUDE.md; its reference recipe, not
>   the shared protocol, row-normalized features, Cora 81.9 / 80.7 at 1,500 epochs) and SGC's
>   published accuracies on Â²X of row-normalized features (Wu et al. 2019: 81.0 / 71.9 / 78.9).
> - *Cost.* From E1's per-seed times on the Mac CPU: about 3.1 h of training (PubMed's GraphCL and
>   AFGRL most of it), a few minutes of scoring. Preflight first (every objective, 3 steps, each
>   dataset).
