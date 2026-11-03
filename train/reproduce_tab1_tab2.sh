#!/usr/bin/env bash
# Tabs. 1-2: SO3brep on the five benchmarks, 5 seeds each, scored on original and rotated test parts.
#   nohup bash train/reproduce_tab1_tab2.sh > tab1_tab2.log 2>&1 &
source "$(dirname "$0")/_common.sh"
for cfg in configs/benchmark/*.yaml; do train "$cfg" "$@"; done
python -m eval.tables runs
