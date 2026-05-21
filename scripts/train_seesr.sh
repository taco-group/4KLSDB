#!/usr/bin/env bash
# Fine-tune SeeSR on 4KLSDB (real-world SR with semantics).
#
# Usage:
#   bash scripts/train_seesr.sh --scale 4 --data data/4KLSDB

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

cd models/seesr

accelerate launch --num_processes 8 --mixed_precision fp16 \
    train_seesr.py \
        --pretrained_model_name_or_path preset/models/stable-diffusion-2-base \
        --controlnet_model_name_or_path preset/models/seesr \
        --train_data_dir "../../$DATA/train/HR" \
        --output_dir     experiences/seesr_4klsdb_x${SCALE} \
        --resolution     512 \
        --upscale        $SCALE \
        --use_4klsdb_degradation True \
        --train_batch_size 4 \
        --max_train_steps  150000 \
        --learning_rate    5e-5 \
        --checkpointing_steps 5000
