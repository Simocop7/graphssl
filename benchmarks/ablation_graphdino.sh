#!/usr/bin/env bash
# Why does GraphDINO get *worse* the longer it trains (teacher_student ablation: Cora 62 -> 39
# from 300 to 3000 steps at the default EMA)? Local single-seed diagnostics tied the decline
# to the teacher over-sharpening: its output entropy falls towards 0, so every node gets a
# hard prototype assignment that stops tracking the classes. Two levers from the reference
# DINO recipe are tested here with 5 seeds:
#
#   - head.norm_last_layer=true: prototype scale fixed at 1, logits are cosines in [-1, 1];
#   - head.teacher_temp=0.07: a softer teacher (warmed up from 0.04 over 30 epochs by the
#     runner), the value DINO recommends with a warmup.
#
# Default vs both levers at 300/1000/3000 steps, each lever alone at 3000 steps (where the
# decline is largest), and both settings with the GCN backbone. The last block adds AFGRL
# with GCN to the teacher_student ablation, completing its backbone comparison.
#
# Run from the repo root on the GPU VM, inside tmux (~3.5 h on an NVIDIA A2):
#   bash benchmarks/ablation_graphdino.sh 2>&1 | tee -a ablation_graphdino.log
# Then:
#   python benchmarks/render_ablation.py benchmarks/ablations/graphdino_stability
#   python benchmarks/render_ablation.py benchmarks/ablations/teacher_student
set -euo pipefail

SEEDS=5
FIX=(--set head.norm_last_layer=true --set head.teacher_temp=0.07)

run() {
    python -u benchmarks/run_benchmark.py --seeds "$SEEDS" \
        --out-dir benchmarks/ablations/graphdino_stability "$@"
}

for STEPS in 300 1000 3000; do
    run --dataset Cora CiteSeer --model graphdino --epochs "$STEPS"
    run --dataset Cora CiteSeer --model graphdino --epochs "$STEPS" "${FIX[@]}"
done

run --dataset Cora CiteSeer --model graphdino --epochs 3000 --set head.norm_last_layer=true
run --dataset Cora CiteSeer --model graphdino --epochs 3000 --set head.teacher_temp=0.07

for STEPS in 300 3000; do
    run --dataset Cora CiteSeer --model graphdino --encoder gcn --epochs "$STEPS"
    run --dataset Cora CiteSeer --model graphdino --encoder gcn --epochs "$STEPS" "${FIX[@]}"
done

for TAU in 0.99 0.9; do
    python -u benchmarks/run_benchmark.py --seeds "$SEEDS" \
        --out-dir benchmarks/ablations/teacher_student \
        --dataset Cora --model afgrl --encoder gcn --epochs 300 --ema-tau "$TAU"
done
