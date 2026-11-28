#!/usr/bin/env bash
# E10 of the factor study (benchmarks/paper/PLAN.md, "E10: heterophilous graphs"):
#
#   bash benchmarks/paper_e10.sh train DATASET ENCODER [MODEL ...]  # 10 seeds per model, models
#                                                   #   saved; with no MODEL: the eight objectives
#                                                   #   and the untrained encoder
#   bash benchmarks/paper_e10.sh probe DATASET      # every saved model of DATASET and the input
#                                                   #   rows: ours + liblinear on split 0, ours on
#                                                   #   the 10 fixed splits; then ours with graph
#                                                   #   statistics for every encoder
#
# DEVICE (default cuda) and SEEDS (default 10) come from the environment. A (dataset, encoder,
# model) whose JSON exists is skipped, so the script resumes after a crash. Failed steps are
# logged as "QUEUE STEP FAILED" and counted in the last line.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=benchmarks/paper
E10=$OUT/e10
OBJECTIVES=(dgi graphcl vicreg barlow_twins bgrl afgrl graphdino graphmae)
failed=0
run() {
  echo "[$(date '+%F %T')] $*"
  "$@" || { echo "QUEUE STEP FAILED (exit $?): $*"; failed=$((failed + 1)); }
}

step=${1:?usage: paper_e10.sh train DATASET ENCODER [MODEL ...] | probe DATASET}
case "$step" in
  train)
    D=${2:?dataset}
    E=${3:?encoder}
    shift 3
    models=("$@")
    [ ${#models[@]} -eq 0 ] && models=("${OBJECTIVES[@]}" untrained)
    for M in "${models[@]}"; do
      if [ "$M" = untrained ]; then
        dir=$E10/e1_untrained/$E/$D
        compgen -G "$dir/bgrl__*.json" > /dev/null && { echo "skip $D $E untrained"; continue; }
        run python -u benchmarks/run_benchmark.py --dataset "$D" --model bgrl --epochs 0 \
          --encoder "$E" --seeds "${SEEDS:-10}" --device "${DEVICE:-cuda}" \
          --out-dir "$E10/e1_untrained/$E" --save-models "$OUT/models_e10_untrained"
      else
        dir=$E10/e1/$E/$D
        compgen -G "$dir/${M}__*.json" > /dev/null && { echo "skip $D $E $M"; continue; }
        run python -u benchmarks/run_benchmark.py --dataset "$D" --model "$M" \
          --encoder "$E" --seeds "${SEEDS:-10}" --device "${DEVICE:-cuda}" \
          --out-dir "$E10/e1/$E" --save-models "$OUT/models_e10"
      fi
    done
    ;;
  probe)
    D=${2:?dataset}
    ckpts=("$OUT"/models_e10/"$D"/*/*.pt "$OUT"/models_e10_untrained/"$D"/*/*.pt)
    run python -u benchmarks/compare_probes.py --dataset "$D" \
      --raw-features --propagated 1 2 --concat-propagated \
      --protocols graphssl liblinear --liblinear-jobs "${LIBLINEAR_JOBS:-16}" \
      --fixed-splits 10 --split-protocols graphssl \
      --checkpoints "${ckpts[@]}" --device cpu --out-dir "$E10/probes/$D"
    run python -u benchmarks/compare_probes.py --dataset "$D" --norm-stats graph \
      --protocols graphssl --checkpoints "${ckpts[@]}" --device cpu \
      --out-dir "$E10/probes_graph/$D"
    ;;
  *)
    echo "unknown step: $step" >&2
    exit 2
    ;;
esac
echo "DONE $step ${*:2} $(date '+%F %T'), $failed failed"
[ "$failed" -eq 0 ]
