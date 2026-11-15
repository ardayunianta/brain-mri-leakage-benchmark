#!/bin/zsh
# Fourth follow-up (9 Nov): the GAIN baseline (RESULTS.md §9) and the held-out-source test (RESULTS.md §8).
#   1. GAIN, seed-42 learning-rate grid (default rate into results/models_grouped, trials into results/lr_tuning);
#   2. held-out source: 6 models at their tuned rates, 5 folds x 3 seeds, into results/heldout_source;
#   3. GAIN: the edge extension if needed, then seeds 43 and 44 at the chosen rate (code/lr_select.py --gain-next);
#   4. held-out source for GAIN at its chosen rate.
# Safe to re-run after an interruption: the notebook skips finished runs.
#   zsh code/run_followup4.sh > results/followup4.log 2>&1
cd "$(dirname "$0")/.."
while pgrep -f "kaggle/run_local.py kaggle/" > /dev/null; do sleep 60; done
COMMON=(MODELS_DATA_ROOT=data/brain-tumor-mri-dataset MODELS_AUDIT_DIR=results/audit MODELS_TPAL_DIR=results/tpal_rerun MODELS_WORKERS=0)
NB=kaggle/models_grouped_kaggle.ipynb
run() {  # run <MODELS> <SEEDS> <OUT> <CKPT> <PROTOCOL> <executed copy>
  env "${COMMON[@]}" MODELS="$1" MODELS_SEEDS="$2" MODELS_OUT="$3" MODELS_CKPT_DIR="$4" MODELS_PROTOCOLS="$5" \
    caffeinate -i .venv/bin/python kaggle/run_local.py $NB "$6"
}
G=efficientnet_b0_gain
HELDOUT=efficientnet_b0_lr1e-3,efficientnet_b0_aux_lesion_lr3e-3,deit_small_cls,deit_small_cls_lesion,bipal_tpal_cls,bipal_tpal_cls_lesion

echo "1. GAIN grid: $(date)"
run $G 42 results/models_grouped checkpoints/models_grouped grouped_5fold results/gain_default_s42_executed.ipynb
run ${G}_lr1e-4,${G}_lr1e-3 42 results/lr_tuning checkpoints/lr_tuning grouped_5fold results/gain_grid_executed.ipynb

echo "2. held-out source: $(date)"
run $HELDOUT 42,43,44 results/heldout_source checkpoints/heldout_source heldout_source results/heldout_source_executed.ipynb

echo "3. GAIN selection: $(date)"
NEXT=$(.venv/bin/python code/lr_select.py --gain-next); echo "$NEXT"
if [[ "$NEXT" == extend* ]]; then
  run ${G}_lr${NEXT#extend } 42 results/lr_tuning checkpoints/lr_tuning grouped_5fold results/gain_extension_executed.ipynb
  NEXT=$(.venv/bin/python code/lr_select.py --gain-next); echo "$NEXT"
fi
[[ "$NEXT" == seeds* ]] || { echo "unexpected: $NEXT"; exit 1; }
GN=${NEXT#seeds }
if [[ "$GN" == "$G" ]]; then
  run $GN 43,44 results/models_grouped checkpoints/models_grouped grouped_5fold results/gain_seeds_executed.ipynb
else
  run $GN 43,44 results/lr_tuning checkpoints/lr_tuning grouped_5fold results/gain_seeds_executed.ipynb
fi

echo "4. held-out source, GAIN ($GN): $(date)"
run $GN 42,43,44 results/heldout_source checkpoints/heldout_source heldout_source results/heldout_source_gain_executed.ipynb
echo "done: $(date)"
.venv/bin/python code/lr_select.py
