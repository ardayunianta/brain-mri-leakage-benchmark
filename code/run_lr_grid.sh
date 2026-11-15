#!/bin/zsh
# Learning-rate tuning, phase 1: the seed-42 grid for the headline models and T-PAL (code/lr_select.py).
# Waits for any running models-notebook job first (only one GPU job at a time), then runs the trials.
# Safe to re-run after an interruption: both notebooks skip finished runs.
#   zsh code/run_lr_grid.sh > results/lr_grid.log 2>&1
cd "$(dirname "$0")/.."
while pgrep -f "kaggle/run_local.py kaggle/models_grouped_kaggle.ipynb" > /dev/null; do sleep 60; done
echo "start: $(date)"
MODELS=$(.venv/bin/python code/lr_select.py --print-grid | sed -n 's/^MODELS=//p')
MODELS="$MODELS" MODELS_SEEDS=42 MODELS_DATA_ROOT=data/brain-tumor-mri-dataset MODELS_AUDIT_DIR=results/audit \
  MODELS_TPAL_DIR=results/tpal_rerun MODELS_OUT=results/lr_tuning MODELS_CKPT_DIR=checkpoints/lr_tuning MODELS_WORKERS=0 \
  caffeinate -i .venv/bin/python kaggle/run_local.py kaggle/models_grouped_kaggle.ipynb results/lr_tuning_grid_executed.ipynb
for LR in 3e-4 3e-3; do
  TPAL_LR=$LR TPAL_PROTOCOLS=grouped_5fold TPAL_SEEDS=42 TPAL_DATA_ROOT=data/brain-tumor-mri-dataset TPAL_AUDIT_DIR=results/audit \
    TPAL_OUT=results/tpal_lr TPAL_WORKERS=0 \
    caffeinate -i .venv/bin/python kaggle/run_local.py kaggle/tpal_grouped_rerun_kaggle.ipynb results/tpal_lr_${LR}_executed.ipynb
done
echo "grid done: $(date)"
.venv/bin/python code/lr_select.py
