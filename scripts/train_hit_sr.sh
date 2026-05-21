#!/usr/bin/env bash
# Train HiT-SR on 4KLSDB.
#
# Usage:
#   bash scripts/train_hit_sr.sh --scale 4 --data data/4KLSDB
#
# This wrapper points at models/hit_sr/options/Train/train_HiT_SRF_x${SCALE}_4KLSDB.yml.

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
[[ "$SCALE" =~ ^(4|8|16)$ ]] || { echo "ERROR: --scale must be 4|8|16"; exit 1; }

OPT="options/Train/train_HiT_SRF_x${SCALE}_4KLSDB.yml"
cd models/hit_sr

if [[ "$NPROC" -gt 1 ]]; then
    torchrun --nproc_per_node=$NPROC basicsr/train.py -opt $OPT --launcher pytorch \
        --override "datasets.train.dataroot_gt=../../$DATA/train/HR datasets.train.dataroot_lq=../../$DATA/train/LR_x${SCALE}"
else
    python basicsr/train.py -opt $OPT \
        --override "datasets.train.dataroot_gt=../../$DATA/train/HR datasets.train.dataroot_lq=../../$DATA/train/LR_x${SCALE}"
fi
