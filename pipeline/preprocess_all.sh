#!/usr/bin/env bash
# STEP -> graphs for all paper datasets: aligned graphs for every split and
# rotated graphs for the test split.
#
# Usage: bash pipeline/preprocess_all.sh --data_dir DIR [--procs N] [--root NAME=PATH ...]
#   DIR/<dataset>/ must hold step/ and train.txt, val.txt, test.txt
#   (fusion360 also seg/). --root overrides the folder of one dataset.
set -euo pipefail
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"   # run from any folder

DATA_DIR=""
PROCS=8
declare -A ROOTS
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --procs) PROCS="$2"; shift 2 ;;
        --root) ROOTS["${2%%=*}"]="${2#*=}"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ -n "$DATA_DIR" ]] || { echo "--data_dir is required" >&2; exit 1; }

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for DS in mfcad++ fusion360 solidletters mftrcad bendfm; do
    ROOT="${ROOTS[$DS]:-$DATA_DIR/$DS}"
    for ROTATE in "" "--rotate"; do
        SPLIT=all; [[ -n "$ROTATE" ]] && SPLIT=test
        nice python -m pipeline.preprocess --root "$ROOT" --dataset "$DS" \
            --split "$SPLIT" $ROTATE --workers "$PROCS" --skip_existing
    done
done
