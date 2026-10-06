# GraphDINO

**Adapted from Caron et al. (DINO), ICCV 2021.** Brings ViT-style self-distillation with no
labels to graphs: a student network learns to match a teacher's soft cluster assignments
(prototypes) across multiple augmented views, with the teacher itself just an EMA of the
student.

## How it works

- Student encoder + student head; teacher encoder + teacher head (EMA, gradient-free).
- `n_views` total augmented views; the first `n_global_views` go to the teacher, all views go
  to the student.
- **`DINOLoss`**: cross-entropy over every pair `(teacher_i, student_j)` with `i ≠ j`, averaged
  over the number of terms.
    - Student output: `log_softmax(logits / student_temp)`, computed inside `DINOHead`.
    - Teacher output: `softmax((logits − center) / teacher_temp_eff)`, computed inside
      `DINOHead`.
- **Teacher temperature warmup**: `teacher_temp_eff` grows linearly from `warmup_teacher_temp`
  to `teacher_temp` over `warmup_teacher_temp_epochs` epochs (`DINOHead.set_epoch(epoch)`).
- **Center update (EMA)**: `center = c_mom * center + (1 − c_mom) * mean(teacher_logits)`,
  over the **raw** teacher logits (`DINOHead.prototype_logits`, before centering and softmax),
  as in the reference DINO implementation. Handled by `DINOHead.update_center()`, called from
  `post_step()` — this is what keeps the teacher's output distribution from collapsing to a
  single prototype.
- **EMA teacher**: momentum grows on a cosine schedule from `ema_tau_base` to `ema_tau` over
  `total_steps`.
- **`freeze_last_layer_epochs`**: during the first N epochs, gradients of the prototype layer
  (`student_head.proto`) are zeroed out after backward, in `post_backward()` — a stabilization
  trick carried over from the original DINO recipe.

## Hyperparameters

| Field | Default | Notes |
|---|---|---|
| `student_temp` | `0.1` | |
| `teacher_temp` | `0.07` | final teacher temperature (reference DINO: `0.04`) |
| `warmup_teacher_temp` | `0.04` | starting teacher temperature (must be ≤ `teacher_temp`) |
| `warmup_teacher_temp_epochs` | `30` | epochs of linear warmup from `warmup_teacher_temp` to `teacher_temp`; `0` uses `teacher_temp` from the start |
| `center_momentum` | `0.9` | |
| `norm_last_layer` | `False` | fix the prototypes' weight-norm scale at 1 (logits become cosines in [-1, 1]), as DINO's `norm_last_layer`; off = trainable scale |
| `ema_tau` | `0.996` | final teacher EMA momentum |
| `ema_tau_base` | `0.9` | starting teacher EMA momentum, annealed to `ema_tau` on a cosine schedule over `total_steps` (reference DINO starts at `0.996`); never above `ema_tau` |
| `total_steps` | `0` | length of the EMA schedule in optimizer steps — **set it to your number of training steps**; `0` keeps the momentum fixed at `ema_tau_base` |
| `freeze_last_layer_epochs` | `1` | |
| `n_views` / `n_global_views` | `2` / `2` | |

!!! note "Why the defaults differ from reference DINO"
    With reference DINO's values (EMA momentum `0.996` from the first step, teacher
    temperature `0.04`) GraphDINO's accuracy *drops* as full-batch training continues: in the
    citation-network ablations (GIN, 5 seeds, linear probe) Cora goes 63.3 → 50.0 → 38.9 at
    300 → 1000 → 3000 steps, ending below the untrained encoder. Diagnostics tie it to the
    teacher's output sharpening until every node gets a hard prototype assignment that no
    longer tracks the classes.

    The defaults are the setting that held up best there: a faster teacher
    (`ema_tau_base=0.9`, annealed to `ema_tau=0.996`) with a softer one (`teacher_temp=0.07`,
    warmed up from `0.04`). Cora: 73.5 → 68.1 → 67.0. `norm_last_layer=True`, gradient
    clipping and more prototypes did not help. Tables and caveats:
    [Benchmarks → Sensitivity](../benchmarks.md#sensitivity-budget-teacher-and-backbone).

    - The ablations always set `total_steps` to the length of the run. Without it the
      momentum stays at `0.9`, which is only spot-checked at 300 steps.
    - To get reference DINO back: `ema_tau_base=0.996`, `head.teacher_temp=0.04`.

## Operation order

```
forward → loss → backward → clip_grad → post_backward (freeze last layer)
→ optimizer.step → post_step (EMA teacher + center update)
```

## Config example

```python
config = {
    "encoder": {"name": "gin", "hidden_dim": 128, "num_layers": 2},
    "head": {
        "name": "dino", "proj_hidden": 128, "bottleneck_dim": 32,
        "n_prototypes": 128,
    },
    "augment_teacher": [{"name": "edge_drop", "p": 0.2}, {"name": "feat_mask", "p": 0.1}],
    "augment_student": [{"name": "edge_drop", "p": 0.3}, {"name": "feat_mask", "p": 0.2}],
    "total_steps": num_epochs * steps_per_epoch,  # length of the EMA schedule
    "freeze_last_layer_epochs": 1,
}
model = GraphDINO(config, in_channels=dataset.num_features)
```

## Reference

`GraphDINOConfig` — see [Config API](../api/config.md#graphssl.config.schema.GraphDINOConfig).
