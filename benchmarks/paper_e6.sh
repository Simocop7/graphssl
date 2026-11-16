#!/usr/bin/env bash
# E6 of the factor study (benchmarks/paper/PLAN.md, "E6: a parameter-free reference"):
#
#   bash benchmarks/paper_e6.sh baselines  # the input-feature rows (raw, A^1X, A^2X, PCA-256 of
#                                          #   A^2X) of the six datasets and ogbn-arxiv /
#                                          #   arxiv-year, under the E2 protocols (Mac CPU)
#   bash benchmarks/paper_e6.sh train      # E1 on WikiCS: 8 objectives + 2 supervised
#                                          #   references + untrained, ENCODERS, 10 seeds (VM GPU:
#                                          #   pass --device cuda)
#   bash benchmarks/paper_e6.sh probe      # E2 on WikiCS: every saved model and the input-feature
#                                          #   rows, standard split + 10 random splits (Mac CPU)
#
# Extra arguments go to every python call. Results go to benchmarks/paper/e6/, models to
# benchmarks/paper/models_e6*/ (git-ignored): nothing here enters the analyses of the six
# datasets, which read benchmarks/paper/{e1,probes,metrics}.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=benchmarks/paper
E6=$OUT/e6
read -r -a ENCODERS <<< "${ENCODERS:-gcn gin transformer}"
INPUT_ROWS=(--raw-features --propagated 1 2 --pca-dim 256)
step=${1:?usage: paper_e6.sh baselines|train|probe [extra args]}
shift

case "$step" in
  baselines)
    for D in ${DATASETS:-Cora CiteSeer PubMed AmazonComputers AmazonPhoto CoauthorCS}; do
      python -u benchmarks/compare_probes.py --dataset "$D" "${INPUT_ROWS[@]}" \
        --protocols adam100 adam5000 liblinear graphssl --random-splits 10 \
        --out-dir "$E6/probes/$D" "$@"
    done
    if [ "${OGB:-1}" = 1 ]; then  # liblinear: 40 x 21 fits on 91k rows, not run (as in E2)
      python -u benchmarks/compare_probes.py --dataset ogbn-arxiv "${INPUT_ROWS[@]}" \
        --protocols adam100 adam5000 graphssl --out-dir "$E6/probes/ogbn-arxiv" "$@"
      python -u benchmarks/compare_probes.py --dataset arxiv-year "${INPUT_ROWS[@]}" \
        --protocols adam100 adam5000 graphssl --random-splits 5 \
        --out-dir "$E6/probes/arxiv-year" "$@"
    fi
    ;;
  train)
    for E in "${ENCODERS[@]}"; do
      python -u benchmarks/run_benchmark.py --dataset WikiCS \
        --model all supervised supervised_reg --encoder "$E" --seeds "${SEEDS:-10}" \
        --out-dir "$E6/e1" --save-models "$OUT/models_e6" "$@"
      python -u benchmarks/run_benchmark.py --dataset WikiCS \
        --model bgrl --epochs 0 --encoder "$E" --seeds "${SEEDS:-10}" \
        --out-dir "$E6/e1_untrained" --save-models "$OUT/models_e6_untrained" "$@"
    done
    ;;
  probe)
    python -u benchmarks/compare_probes.py --dataset WikiCS "${INPUT_ROWS[@]}" \
      --protocols adam100 adam5000 liblinear graphssl --random-splits 10 \
      --checkpoints "$OUT"/models_e6/WikiCS/*/*.pt "$OUT"/models_e6_untrained/WikiCS/*/*.pt \
      --out-dir "$E6/probes/WikiCS" "$@"
    ;;
  *)
    echo "unknown step: $step" >&2
    exit 2
    ;;
esac
