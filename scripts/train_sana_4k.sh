#!/usr/bin/env bash
# Fine-tune Sana on 4KLSDB for native 4K (4096×4096) T2I generation.
#
# Usage:
#   bash scripts/train_sana_4k.sh --resolution 4096 --data data/4KLSDB
#
# Pre-compute Gemma-2 caption embeddings first (see models/sana/README.md):
#   cd models/sana/diffusion/data/datasets
#   python embed_pro.py --data_root <path> --output_dir <emb_dir>

set -euo pipefail
RES=4096
DATA=""
EMB=""
CFG="configs/sana_config/4096ms/Sana_1600M_4Kms.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resolution) RES="$2"; shift 2;;
        --data)       DATA="$2"; shift 2;;
        --emb)        EMB="$2";  shift 2;;
        --config)     CFG="$2";  shift 2;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

[[ -z "$DATA" ]] && { echo "ERROR: --data is required"; exit 1; }
[[ -z "$EMB"  ]] && EMB="$DATA/embeddings_gemma2_2b"

cd models/sana

bash train_scripts/train.sh \
    --config_path  "$CFG" \
    --work_dir     "output_4klsdb_${RES}" \
    --data.data_dir "[../../$DATA/train]" \
    --train.train_batch_size 1 \
    --train.gradient_accumulation_steps 4 \
    --model.image_size $RES \
    --data.caption_emb_root "../../$EMB"
