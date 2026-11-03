#!/usr/bin/env bash
# Fig. 3: SO3brep on MFCAD++ with 10 ... 1000 training parts, 5 random subsets each.
source "$(dirname "$0")/_common.sh"
for n in 10 25 50 100 250 1000; do
  train configs/data_efficiency/mfcad.yaml --set name=mfcad_${n} data.train_subset=${n} "$@"
done
python -m eval.tables runs
