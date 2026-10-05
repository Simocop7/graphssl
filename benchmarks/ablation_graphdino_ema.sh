#!/usr/bin/env bash
# Follow-up to ablation_graphdino.sh, to settle GraphDINO's defaults.
#
# What the two earlier grids showed (GIN, 5 seeds, linear probe, Cora at 300/1000/3000 steps):
#   - default (EMA 0.996 fixed, teacher_temp 0.04):       63.3 -> 50.0 -> 38.9
#   - head.norm_last_layer=true alone:                    no effect (37.4 at 3000)
#   - head.teacher_temp=0.07 alone:                       slows the decline (54.2 at 3000)
#   - --ema-tau 0.9 (teacher_student ablation):           66.5 -> 62.9 -> 62.9, no decline
# So the faster teacher is the only setting that removes the decline, but it was only run
# with GIN on Cora/CiteSeer and never together with the softer teacher. This grid fills
# those gaps:
#
#   - --ema-tau 0.9 + head.teacher_temp=0.07 with GIN at 300/1000/3000 (compare with the
#     --ema-tau 0.9 rows in benchmarks/ablations/teacher_student, not rerun here);
#   - --ema-tau 0.9, with and without the softer teacher, with the GCN backbone;
#   - PubMed at the benchmark budget (300 steps): default vs both candidates, since the
#     benchmark row would be rerun on all three datasets;
#   - the untrained encoder (0 steps) for both backbones: an untrained GCN already scores
#     far above an untrained GIN on these datasets (about 72 vs 40 on Cora in a local CPU
#     check), so the GCN rows only mean something relative to this reference.
#
# The 0- and 300-step runs come first: they are cheap and already tell whether a candidate
# costs accuracy at the benchmark budget.
#
# Run from the repo root on the GPU VM, inside tmux (~2 h 45 min on a shared NVIDIA A2):
#   bash benchmarks/ablation_graphdino_ema.sh 2>&1 | tee -a ablation_graphdino_ema.log
# Then:
#   python benchmarks/render_ablation.py benchmarks/ablations/graphdino_stability
set -euo pipefail

SEEDS=5
TAU=(--ema-tau 0.9)
TEMP=(--set head.teacher_temp=0.07)

run() {
    python -u benchmarks/run_benchmark.py --seeds "$SEEDS" \
        --out-dir benchmarks/ablations/graphdino_stability --model graphdino "$@"
}

run --dataset Cora CiteSeer PubMed --epochs 0
run --dataset Cora CiteSeer PubMed --encoder gcn --epochs 0

run --dataset Cora CiteSeer --epochs 300 "${TAU[@]}" "${TEMP[@]}"
run --dataset Cora CiteSeer --encoder gcn --epochs 300 "${TAU[@]}"
run --dataset Cora CiteSeer --encoder gcn --epochs 300 "${TAU[@]}" "${TEMP[@]}"

run --dataset PubMed --epochs 300
run --dataset PubMed --epochs 300 "${TAU[@]}"
run --dataset PubMed --epochs 300 "${TAU[@]}" "${TEMP[@]}"

run --dataset Cora CiteSeer --epochs 1000 "${TAU[@]}" "${TEMP[@]}"
run --dataset Cora CiteSeer --epochs 3000 "${TAU[@]}" "${TEMP[@]}"
run --dataset Cora CiteSeer --encoder gcn --epochs 3000 "${TAU[@]}"
run --dataset Cora CiteSeer --encoder gcn --epochs 3000 "${TAU[@]}" "${TEMP[@]}"
