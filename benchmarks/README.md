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
stability levers (~3.5 h on an A2).

`ablation_teacher_student.sh` is the grid behind the teacher-student question (why BGRL,
AFGRL and GraphDINO trail the top group): budget 300/1000/3000 steps × default vs faster
EMA teacher, a Barlow Twins control, BGRL with GCN, and AFGRL on Cora. About 4 h on an A2:

```bash
bash benchmarks/ablation_teacher_student.sh 2>&1 | tee -a ablation.log
python benchmarks/render_ablation.py benchmarks/ablations/teacher_student
```

This intentionally does not replace `examples/*.py`, which stay as minimal
single-file demos for learning the API — this is for producing citable,
comparable numbers across the model zoo. See `../CLAUDE.md` (Benchmarks
section) for the methodology and current status.

**Current results:** all 7 SSL models × Cora/CiteSeer/PubMed × 10 seeds are committed under
`results/` and rendered in `../docs/benchmarks.md`. Budget for a full sweep: most models take
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
