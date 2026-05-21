#!/usr/bin/env bash
# Fine-tune OSEDiff on 4KLSDB (real-world blind SR).
#
# Usage:
#   bash scripts/train_osediff.sh --scale 4 --data data/4KLSDB

set -euo pipefail
SCALE=4
DATA=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scale) SCALE="$2"; shift 2;;
        --data)  DATA="$2";  shift 2;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

[[ -z "$DATA" ]] && { echo "ERROR: --data is required"; exit 1; }

cd models/osediff

accelerate launch --num_processes 8 --mixed_precision fp16 \
    train_osediff.py \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-2-base \
        --train_data_dir   "../../$DATA/train/HR" \
        --output_dir       experiments/osediff_4klsdb_x${SCALE} \
        --resolution       512 \
        --upscale          $SCALE \
        --use_4klsdb_degradation True \
        --train_batch_size 4 \
        --max_train_steps  200000 \
        --learning_rate    5e-5 \
        --gradient_accumulation_steps 2
