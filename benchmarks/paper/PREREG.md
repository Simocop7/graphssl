# Analyses fixed in advance

The rules of the five analyses of the study that were fixed before the results they apply to,
copied verbatim, by script, from the planning file as it stood at the commit that fixed each one.
Those commits belong to a development history that also holds private notes and is not part of
this repository. This is the authors' own record, not an external registration. What had been seen
by then is listed in the appendix of the paper ("Analyses fixed in advance"). In this public copy,
four personal notes about who asked for an analysis are replaced by "[omitted]"; the rules are otherwise
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
