#!/usr/bin/env bash
# The runs added to the factor study on 2026-10-26 (benchmarks/paper/PLAN.md). Same layout as
# paper_runs.sh, which still does `analyze` and `figures` over everything:
#
#   bash benchmarks/paper_v2.sh e1      # E1 on the full-batch datasets: 7 SSL methods + 2
#                                       #   supervised references + untrained, 10 seeds, models
#                                       #   saved. DATASETS (default: Amazon/Coauthor) x ENCODERS
#                                       #   (default: gcn gin transformer)
#   bash benchmarks/paper_v2.sh e3      # E3: optimizer x learning rate, Cora + CiteSeer, 5 seeds,
#                                       #   ENCODERS; every cell, AdamW 5e-4 included (E3 runs on
#                                       #   the VM GPU, E1 on the Mac CPU: one device per study).
#                                       #   CELLS (opt:lr ...) runs a part of the grid
#   bash benchmarks/paper_v2.sh ogb     # E1 on OGB (GPU): EPOCHS (required), SEEDS (default 3),
#                                       #   OGB_DATASET (default ogbn-arxiv), ENCODERS
#   bash benchmarks/paper_v2.sh probe   # E2 for DATASETS (+ the raw-features row); arxiv-year
#                                       #   reuses the ogbn-arxiv models
#   bash benchmarks/paper_v2.sh metrics # M: representation metrics of every saved model, DATASETS
#   bash benchmarks/paper_v2.sh represent  # analyze_representations.py over everything, then
#                                       #   paper_figures.py (fig4, study_metrics/cka/predict)
#
# Extra arguments go to every python call (`--device cpu`). A dataset's runs all go on one
# machine: full-batch ones on the Mac (OMP_NUM_THREADS=3), OGB ones on the VM. E3 is its own
# study and runs whole on the VM GPU (`--device cuda`).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=benchmarks/paper
read -r -a DATASETS <<< "${DATASETS:-AmazonComputers AmazonPhoto CoauthorCS}"
read -r -a ENCODERS <<< "${ENCODERS:-gcn gin transformer}"
step=${1:?usage: paper_v2.sh e1|e3|ogb|probe|metrics [extra args]}
shift

case "$step" in
  e1)
    for E in "${ENCODERS[@]}"; do
      python -u benchmarks/run_benchmark.py --dataset "${DATASETS[@]}" \
        --model all supervised supervised_reg --encoder "$E" --seeds 10 \
        --out-dir "$OUT/e1" --save-models "$OUT/models" "$@"
      python -u benchmarks/run_benchmark.py --dataset "${DATASETS[@]}" \
        --model bgrl --epochs 0 --encoder "$E" --seeds 10 \
        --out-dir "$OUT/e1_untrained" --save-models "$OUT/models_untrained" "$@"
    done
    ;;
  e3)
    read -r -a CELLS <<< "${CELLS:-adamw:5e-4 adamw:1e-4 adamw:1e-3 adamw:5e-3 adamw:1e-2 \
      sgd:1e-3 sgd:5e-3 sgd:1e-2 sgd:5e-2 sgd:1e-1}"
    for E in "${ENCODERS[@]}"; do
      for cell in "${CELLS[@]}"; do
        opt=${cell%%:*} lr=${cell##*:}
        python -u benchmarks/run_benchmark.py --dataset Cora CiteSeer --model all \
          --encoder "$E" --seeds 5 --optimizer "$opt" --lr "$lr" \
          --out-dir "$OUT/e3/${opt}_lr${lr}" "$@"
      done
    done
    ;;
  ogb)
    : "${EPOCHS:?set EPOCHS (passes over every node), fixed after the pilot}"
    D=${OGB_DATASET:-ogbn-arxiv}
    for E in "${ENCODERS[@]}"; do
      python -u benchmarks/run_ogb.py --dataset "$D" --model all --encoder "$E" \
        --seeds "${SEEDS:-3}" --epochs "$EPOCHS" \
        --out-dir "$OUT/e1" --save-models "$OUT/models" "$@"
      python -u benchmarks/run_ogb.py --dataset "$D" --model bgrl --epochs 0 --encoder "$E" \
        --seeds "${SEEDS:-3}" --out-dir "$OUT/e1_untrained" --save-models "$OUT/models_untrained" "$@"
    done
    ;;
  probe)
    for D in "${DATASETS[@]}"; do
      M=$D extra=(--protocols adam100 adam5000 liblinear graphssl --random-splits 10)
      case "$D" in
        ogbn-*) extra=(--protocols adam100 adam5000 graphssl) ;;  # liblinear: 40 x 21 fits
        arxiv-year) M=ogbn-arxiv extra=(--protocols adam100 adam5000 graphssl --random-splits 5) ;;
      esac
      python -u benchmarks/compare_probes.py --dataset "$D" "${extra[@]}" --raw-features \
        --checkpoints "$OUT"/models/"$M"/*/*.pt "$OUT"/models_untrained/"$M"/*/*.pt \
        --out-dir "$OUT/probes/$D" "$@"
      python -u benchmarks/compare_probes.py --dataset "$D" \
        --protocols adam100_std adam5000_std graphssl_weakl2 \
        --checkpoints "$OUT"/models/"$M"/*/*.pt "$OUT"/models_untrained/"$M"/*/*.pt \
        --out-dir "$OUT/probes/${D}_std" "$@"
    done
    ;;
  metrics)
    for D in "${DATASETS[@]}"; do
      python -u benchmarks/representation_metrics.py --dataset "$D" \
        --checkpoints "$OUT"/models/"$D"/*/*.pt "$OUT"/models_untrained/"$D"/*/*.pt \
        --untrained-root "$OUT/models_untrained" --out "$OUT/metrics/$D" "$@"
    done
    ;;
  represent)
    python benchmarks/analyze_representations.py --metrics "$OUT"/metrics/*/metrics__*.json \
      --probes "$OUT"/probes/*/probes__*.json --untrained-root "$OUT/models_untrained" \
      --out "$OUT/analysis" "$@"
    python benchmarks/paper_figures.py --analysis "$OUT/analysis" --out "$OUT/figures"
    ;;
  *)
    echo "unknown step: $step (e1 | e3 | ogb | probe | metrics | represent)" >&2
    exit 2
    ;;
esac
