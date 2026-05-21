#!/usr/bin/env bash
# Train MambaIR on 4KLSDB.
#
# Usage:
#   bash scripts/train_mambair.sh --scale 4 --data data/4KLSDB

set -euo pipefail
SCALE=4
DATA=""
NPROC=${NPROC:-1}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scale) SCALE="$2"; shift 2;;
        --data)  DATA="$2";  shift 2;;
        --nproc) NPROC="$2"; shift 2;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

[[ -z "$DATA" ]] && { echo "ERROR: --data is required"; exit 1; }

OPT="options/train/train_MambaIR_classicSR_x${SCALE}_4KLSDB.yml"
cd models/mambair

if [[ "$NPROC" -gt 1 ]]; then
    torchrun --nproc_per_node=$NPROC basicsr/train.py -opt $OPT --launcher pytorch \
        --override "datasets.train.dataroot_gt=../../$DATA/train/HR datasets.train.dataroot_lq=../../$DATA/train/LR_x${SCALE}"
else
    python basicsr/train.py -opt $OPT \
        --override "datasets.train.dataroot_gt=../../$DATA/train/HR datasets.train.dataroot_lq=../../$DATA/train/LR_x${SCALE}"
fi
