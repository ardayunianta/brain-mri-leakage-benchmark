#!/bin/zsh
# Learning-rate tuning, after phase 1 (code/run_lr_grid.sh):
#   1. the one-point extension for the phase-1 edge winners (seed 42; EXTEND in code/lr_select.py);
#   2. the selection over all points, then seeds 43 and 44 for every winner that is not the default.
# Safe to re-run after an interruption: both notebooks skip finished runs.
#   zsh code/run_lr_phase2.sh > results/lr_phase2.log 2>&1
cd "$(dirname "$0")/.."
while pgrep -f "kaggle/run_local.py kaggle/" > /dev/null; do sleep 60; done
COMMON_M=(MODELS_DATA_ROOT=data/brain-tumor-mri-dataset MODELS_AUDIT_DIR=results/audit MODELS_TPAL_DIR=results/tpal_rerun
          MODELS_OUT=results/lr_tuning MODELS_CKPT_DIR=checkpoints/lr_tuning MODELS_WORKERS=0)
COMMON_T=(TPAL_PROTOCOLS=grouped_5fold TPAL_DATA_ROOT=data/brain-tumor-mri-dataset TPAL_AUDIT_DIR=results/audit
          TPAL_OUT=results/tpal_lr TPAL_WORKERS=0)

echo "extension start: $(date)"
EXT=$(.venv/bin/python code/lr_select.py --print-extension)
M=$(echo "$EXT" | sed -n 's/^MODELS=//p'); T=$(echo "$EXT" | sed -n 's/^TPAL_LR=//p')
env "${COMMON_M[@]}" MODELS="$M" MODELS_SEEDS=42 \
  caffeinate -i .venv/bin/python kaggle/run_local.py kaggle/models_grouped_kaggle.ipynb results/lr_tuning_extension_executed.ipynb
env "${COMMON_T[@]}" TPAL_LR="$T" TPAL_SEEDS=42 \
  caffeinate -i .venv/bin/python kaggle/run_local.py kaggle/tpal_grouped_rerun_kaggle.ipynb results/tpal_lr_${T}_executed.ipynb

echo "selection after the extension: $(date)"
SEL=$(.venv/bin/python code/lr_select.py)
echo "$SEL"
M=$(echo "$SEL" | sed -n 's/^MODELS=//p')
T=$(echo "$SEL" | sed -n 's/^T-PAL notebook: TPAL_LR=\([^ ]*\).*/\1/p')
if [[ -z "$M" && -z "$T" ]]; then echo "nothing to run on seeds 43/44"; exit 0; fi
if [[ -n "$M" ]]; then
  env "${COMMON_M[@]}" MODELS="$M" MODELS_SEEDS=43,44 \
    caffeinate -i .venv/bin/python kaggle/run_local.py kaggle/models_grouped_kaggle.ipynb results/lr_tuning_phase2_executed.ipynb
fi
if [[ -n "$T" ]]; then
  env "${COMMON_T[@]}" TPAL_LR="$T" TPAL_SEEDS=43,44 \
    caffeinate -i .venv/bin/python kaggle/run_local.py kaggle/tpal_grouped_rerun_kaggle.ipynb results/tpal_lr_${T}_s43_44_executed.ipynb
fi
echo "phase 2 done: $(date)"
