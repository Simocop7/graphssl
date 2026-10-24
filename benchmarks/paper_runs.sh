#!/usr/bin/env bash
# Runs behind the factor-study paper: train once, then score the same embeddings many ways.
#
#   bash benchmarks/paper_runs.sh train     # E1: 7 SSL methods + 2 supervised references,
#                                           #     GCN and GIN, Cora and CiteSeer, 10 seeds,
#                                           #     plus the untrained encoders; models saved
#   bash benchmarks/paper_runs.sh probe     # E2 + E3: every saved model scored with 4 probe
#                                           #     protocols + kNN on the public split, and with
#                                           #     the GraphSSL probe + kNN on 10 random splits
#   bash benchmarks/paper_runs.sh analyze   # CIs, gains over untrained, Kendall tau, variance shares
#
# Results: benchmarks/paper/{e1,e1_untrained,probes,analysis} (JSON / CSV, committed);
# models:  benchmarks/paper/{models,models_untrained} (*.pt, git-ignored).
# Extra arguments go to every python call, e.g. `bash benchmarks/paper_runs.sh train --device cpu`.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=benchmarks/paper
# Cora and CiteSeer by default (~2 h on a laptop CPU). PubMed is ~90% of the cost
# (GraphCL and AFGRL): add it with DATASETS="Cora CiteSeer PubMed", ideally on a GPU.
read -r -a DATASETS <<< "${DATASETS:-Cora CiteSeer}"
step=${1:?usage: paper_runs.sh train|probe|analyze [extra args]}
shift

case "$step" in
  train)
    for E in gcn gin; do
      python -u benchmarks/run_benchmark.py --dataset "${DATASETS[@]}" \
        --model all supervised supervised_reg --encoder "$E" --seeds 10 \
        --out-dir "$OUT/e1" --save-models "$OUT/models" "$@"
      python -u benchmarks/run_benchmark.py --dataset "${DATASETS[@]}" \
        --model bgrl --epochs 0 --encoder "$E" --seeds 10 \
        --out-dir "$OUT/e1_untrained" --save-models "$OUT/models_untrained" "$@"
    done
    ;;
  probe)
    for D in "${DATASETS[@]}"; do
      python -u benchmarks/compare_probes.py --dataset "$D" \
        --protocols adam100 adam5000 liblinear graphssl --random-splits 10 \
        --checkpoints "$OUT"/models/"$D"/*/*.pt "$OUT"/models_untrained/"$D"/*/*.pt \
        --out-dir "$OUT/probes/$D" "$@"
    done
    ;;
  analyze)
    python benchmarks/analyze_factors.py "$OUT"/probes/*/probes__*.json \
      --untrained-root "$OUT/models_untrained" --out "$OUT/analysis" "$@"
    ;;
  *)
    echo "unknown step: $step (train | probe | analyze)" >&2
    exit 2
    ;;
esac
