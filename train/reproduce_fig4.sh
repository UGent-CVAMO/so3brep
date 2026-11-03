#!/usr/bin/env bash
# Fig. 4: ablations on Fusion 360, 5 seeds each.
source "$(dirname "$0")/_common.sh"
for cfg in configs/ablations/*.yaml; do train "$cfg" "$@"; done
python -m eval.tables runs
