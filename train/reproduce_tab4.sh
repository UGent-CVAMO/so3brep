#!/usr/bin/env bash
# Tab. 4: parameters, inference and training-step time and memory of SO3brep on Fusion 360 (batch 64).
source "$(dirname "$0")/_common.sh"
python -m eval.cost --dataset fusion360 --root "$SO3BREP_DATA/fusion360" "$@"
