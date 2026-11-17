#!/bin/zsh
# CROSS pilot (15 runs: erm, cross_p, cross_pe x 5 folds, seed 42), held-out-source protocol.
#   zsh code/run_cross_pilot.sh > results/cross_pilot/pilot.log 2>&1
cd "$(dirname "$0")/.."
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python code/cross_pilot.py --arms erm,cross_p,cross_pe --folds 0,1,2,3,4 --seeds 42
