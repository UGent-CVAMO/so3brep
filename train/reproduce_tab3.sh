#!/usr/bin/env bash
# Tab. 3: CG, attention, concatenation and invariant message functions on identical inputs,
# full training sets and 100-part subsets of Fusion 360 and MFTRCAD, 5 seeds each.
source "$(dirname "$0")/_common.sh"
for cfg in configs/message_functions/*.yaml; do train "$cfg" "$@"; done
python -m eval.tables runs
