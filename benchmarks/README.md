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
python benchmarks/render_tables.py
python benchmarks/render_tables.py --latex
```

Each run writes `results/<Dataset>/<model>__<UTC-timestamp>.json` — per-seed
metrics, aggregate mean/std, the exact hyperparameters and full model config
(`model_config`, the dict passed to `build_model()`), the git commit (and whether tracked
files had uncommitted changes), and package versions. Old files are never overwritten, so
results accumulate as a history; `render_tables.py` always uses the most recent file per
(dataset, model) pair.

Besides accuracy, every result records the **effective rank** of the evaluated embeddings
(`eff_rank`, from `graphssl.evaluation.effective_rank`: 1 = everything along one direction,
`hidden_dim` = variance spread evenly), which exposes dimensional collapse. For the
teacher-student models it also evaluates the **other encoder** of the same trained model
(`alt_encoder`: target for BGRL/AFGRL, student for GraphDINO), saved with an `_alt` suffix.

## Ablations

Ablations must not land in `results/`, or `render_tables.py` would report them as the
benchmark. Point `--out-dir` elsewhere and render them with `render_ablation.py`, which keeps
one row per (model, encoder, steps, starting EMA momentum):

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
stability levers (~3.5 h on an A2). `ablation_graphdino_ema.sh` is its follow-up (~2 h 45 min):
the faster EMA teacher (`--ema-tau 0.9`) with GCN, on PubMed and combined with the softer
teacher. Both write to `ablations/graphdino_stability`.

`--epochs 0` skips training and evaluates the **untrained encoder**. Use it as the reference
when comparing backbones: an untrained GCN already separates the Planetoid classes far better
than an untrained GIN, so a higher number with GCN is not by itself evidence that a method
learned more.

`ablation_teacher_student.sh` is the grid behind the teacher-student question (why BGRL,
AFGRL and GraphDINO trail the top group): budget 300/1000/3000 steps × default vs faster
EMA teacher, a Barlow Twins control, BGRL with GCN, and AFGRL on Cora. About 4 h on an A2:

```bash
bash benchmarks/ablation_teacher_student.sh 2>&1 | tee -a ablation.log
python benchmarks/render_ablation.py benchmarks/ablations/teacher_student
```

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

The committed run (2026-10-06, commit `aaa280f`) predates the current linear probe: its
`Linear` columns come from the 0.1.0 probe, which on that graph gives 45–54% on the same
embeddings depending on its random seed. Read its kNN column; the linear numbers cannot rank
the methods. Its `ok` status means that a model ran to the end, not that it learned: DGI is
`ok` with embeddings worse than the untrained encoder's.

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

**Current results:** all 7 SSL models × Cora/CiteSeer/PubMed × 10 seeds are committed under
`results/` and rendered in `../docs/benchmarks.md`. They were evaluated with the 0.1.0 probe
and have not been rerun with the current one yet. Budget for a full sweep: most models take
~10–60 s per seed on an NVIDIA A2, but on PubMed GraphCL (O(N²) NT-Xent) and AFGRL (dense
N×N kNN + per-step k-means) take ~7–9 min per seed, so ~75–85 min each for 10 seeds.

**Supervised references** (not part of `all`): both train the same encoder full-batch on the
public split's `train_mask` only.
- `--model supervised` keeps the shared SSL protocol unchanged (no dropout, AdamW, last
  checkpoint). With 120–140 labels and no regularization it overfits, so it understates what
  the labels are worth.
- `--model supervised_reg` uses a recipe adapted from Kipf & Welling: dropout 0.5, Adam lr
  0.01 with L2 5e-4, checkpoint with the best validation accuracy. Its recipe is recorded in
  the JSON's `hyperparameters`, the selected epoch per seed as `best_epoch`. With the GIN
  backbone it helps on Cora and CiteSeer and hurts on PubMed (see `../docs/benchmarks.md`).

Both get the same linear probe + kNN evaluation on their encoder embeddings; the accuracy of
the trained head itself is saved as `test_acc_head`.

**Known gaps:**
- `afgrl` is skipped automatically when `faiss-cpu` isn't installed (same
  behavior as `tests/test_afgrl.py`).
- Only Planetoid (Cora/CiteSeer/PubMed) is wired up so far — ogbn-arxiv/ZINC
  support is the next extension (add an entry next to `PLANETOID_DATASETS`).
