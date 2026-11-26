#!/usr/bin/env bash
# E9 of the factor study (benchmarks/paper/PLAN.md, "E9: label budgets"): the saved SSL models of
# E1 (six datasets) and E6 (WikiCS), their untrained encoders and the input rows, refitted with
# 1, 2, 5, 10 and 20 labels per class. No training.
#
#   bash benchmarks/paper_e9.sh DATASET      # running statistics, then the untrained encoders
#                                            #   again with graph statistics
#
# Failed steps are logged as "QUEUE STEP FAILED" and counted in the last line.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=benchmarks/paper
E9=$OUT/e9
OBJECTIVES=(dgi graphcl vicreg barlow_twins bgrl afgrl graphdino graphmae)
BUDGET=(--label-budgets 1 2 5 10 20 --budget-draws 10 --budget-val both --protocols)
failed=0
run() {
  echo "[$(date '+%F %T')] $*"
  "$@" || { echo "QUEUE STEP FAILED (exit $?): $*"; failed=$((failed + 1)); }
}

D=${1:?usage: paper_e9.sh DATASET}
if [ "$D" = WikiCS ]; then
  trained=$OUT/models_e6
  untrained=$OUT/models_e6_untrained
else
  trained=$OUT/models
  untrained=$OUT/models_untrained
fi
ckpts=()
for M in "${OBJECTIVES[@]}"; do ckpts+=("$trained/$D"/*/"${M}"_seed*.pt); done
run python -u benchmarks/compare_probes.py --dataset "$D" "${BUDGET[@]}" \
  --raw-features --propagated 2 --pca-dim 256 \
  --checkpoints "${ckpts[@]}" "$untrained/$D"/*/*.pt --device cpu --out-dir "$E9/$D"
run python -u benchmarks/compare_probes.py --dataset "$D" "${BUDGET[@]}" --norm-stats graph \
  --checkpoints "$untrained/$D"/*/*.pt --device cpu --out-dir "$E9/${D}_graph"
echo "DONE e9 $D $(date '+%F %T'), $failed failed"
[ "$failed" -eq 0 ]
