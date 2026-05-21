#!/usr/bin/env bash
# Train SwinIR on 4KLSDB (classical SR).
#
# Usage:
#   bash scripts/train_swinir.sh --scale 4 --data data/4KLSDB

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

OPT="options/swinir/train_swinir_sr_classical_x${SCALE}_4KLSDB.json"
cd models/swinir

if [[ "$NPROC" -gt 1 ]]; then
    torchrun --nproc_per_node=$NPROC main_train_psnr.py --opt $OPT --dist True \
        --data_root "../../$DATA"
else
    python main_train_psnr.py --opt $OPT --data_root "../../$DATA"
fi
