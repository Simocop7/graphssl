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
| `teacher_temp` | `0.04` | final teacher temperature |
| `warmup_teacher_temp` | `0.04` | starting teacher temperature (must be ≤ `teacher_temp`) |
| `warmup_teacher_temp_epochs` | `0` | the warmup only matters when `teacher_temp` > `warmup_teacher_temp` (e.g. `0.07`) |
| `center_momentum` | `0.9` | |
| `norm_last_layer` | `False` | fix the prototypes' weight-norm scale at 1 (logits become cosines in [-1, 1]), as DINO's `norm_last_layer`; off = trainable scale |
| `ema_tau` | `0.996` | |
| `freeze_last_layer_epochs` | `1` | |
| `n_views` / `n_global_views` | `2` / `2` | |

!!! warning "Longer training can hurt with the default config"
    In the citation-network ablation GraphDINO's accuracy *drops* as training gets longer
    (Cora: 62% at 300 steps, 39% at 3000). The teacher's output sharpens until every node
    gets a hard prototype assignment that no longer tracks the classes. Gradient clipping and
    more prototypes don't help; `norm_last_layer=True` helps partly and a softer teacher
    (`teacher_temp=0.07`, warmed up from `0.04`) stopped the decline in a first diagnostic.
    `benchmarks/ablation_graphdino.sh` is measuring both before the defaults change.

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
        "n_prototypes": 128, "warmup_teacher_temp_epochs": 30,
    },
    "augment_teacher": [{"name": "edge_drop", "p": 0.2}, {"name": "feat_mask", "p": 0.1}],
    "augment_student": [{"name": "edge_drop", "p": 0.3}, {"name": "feat_mask", "p": 0.2}],
    "ema_tau": 0.996,
    "freeze_last_layer_epochs": 1,
}
model = GraphDINO(config, in_channels=dataset.num_features)
```

## Reference

`GraphDINOConfig` — see [Config API](../api/config.md#graphssl.config.schema.GraphDINOConfig).
