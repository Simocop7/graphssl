#!/usr/bin/env bash
# Why do the teacher-student methods (BGRL, AFGRL, GraphDINO) trail the top SSL group on the
# citation benchmark? This grid tests the leading hypothesis: a budget too short for the EMA
# teacher to catch up (300 steps), and a teacher momentum tuned for far longer training.
#
#   - training budget 300 / 1000 / 3000 steps (the EMA schedule spans the whole run);
#   - starting EMA momentum: each model's default vs 0.9 (a faster-moving teacher);
#   - Barlow Twins as the control (no EMA teacher): if it gains as much from a longer
#     budget, the budget alone doesn't explain the gap;
#   - BGRL and the Barlow Twins control with GCN, the original BGRL paper's backbone;
#   - AFGRL is ~10x costlier per step (k-means every step): Cora only, up to 1000 steps.
#
# Every JSON also records the embeddings' effective rank (dimensional collapse) and the
# other encoder's accuracy (BGRL/AFGRL target, GraphDINO student).
#
# Run from the repo root on the GPU VM, inside tmux (~4 h on an NVIDIA A2):
#   bash benchmarks/ablation_teacher_student.sh 2>&1 | tee -a ablation.log
# Then:
#   python benchmarks/render_ablation.py benchmarks/ablations/teacher_student
set -euo pipefail

OUT=benchmarks/ablations/teacher_student
SEEDS=5

run() {
    python -u benchmarks/run_benchmark.py --seeds "$SEEDS" --out-dir "$OUT" "$@"
}

for STEPS in 300 1000 3000; do
    run --dataset Cora CiteSeer --model barlow_twins bgrl graphdino --epochs "$STEPS"
    run --dataset Cora CiteSeer --model bgrl graphdino --epochs "$STEPS" --ema-tau 0.9
done

for STEPS in 300 3000; do
    run --dataset Cora CiteSeer --model barlow_twins bgrl --encoder gcn --epochs "$STEPS"
done

for STEPS in 300 1000; do
    run --dataset Cora --model afgrl --epochs "$STEPS"
    run --dataset Cora --model afgrl --epochs "$STEPS" --ema-tau 0.9
done
