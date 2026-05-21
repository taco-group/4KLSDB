#!/usr/bin/env bash
# One-click inference for the 4KLSDB-fine-tuned Sana model.
# Generates native 4096×4096 images from a text prompt.
#
# Usage:
#   bash scripts/inference_sana_4k.sh \
#       --prompt "A serene mountain lake at sunrise, 4K, photorealistic" \
#       --resolution 4096 \
#       --output results/sana_4k.png \
#       [--ckpt release_ckpts/sana/epoch_7_step_160000.pth] \
#       [--steps 20] [--cfg 4.5] [--seed 42]

set -euo pipefail

PROMPT=""
RES=4096
OUTPUT="results/sana_4k.png"
CKPT=""
STEPS=20
CFG=4.5
SEED=42
CKPT_ROOT="${CKPT_ROOT:-release_ckpts}"

print_help() { sed -n '2,14p' "$0" | sed 's/^# *//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt)     PROMPT="$2"; shift 2;;
        --resolution) RES="$2";    shift 2;;
        --output)     OUTPUT="$2"; shift 2;;
        --ckpt)       CKPT="$2";   shift 2;;
        --steps)      STEPS="$2";  shift 2;;
        --cfg)        CFG="$2";    shift 2;;
        --seed)       SEED="$2";   shift 2;;
        -h|--help)    print_help;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

[[ -z "$PROMPT" ]] && { echo "ERROR: --prompt is required"; exit 1; }
[[ -z "$CKPT" ]] && CKPT="${CKPT_ROOT}/sana/4KLSDB_sana_${RES}.pth"
[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT"; echo "  Run 'bash scripts/download_all_ckpts.sh' first."; exit 1; }

mkdir -p "$(dirname "$OUTPUT")"

echo "==> Prompt:     \"$PROMPT\""
echo "==> Resolution: ${RES}×${RES}"
echo "==> Ckpt:       $CKPT"
echo "==> Steps:      $STEPS  CFG: $CFG  Seed: $SEED"
echo "==> Output:     $OUTPUT"
echo

cd models/sana
python scripts/inference.py \
    --model_path "../../$CKPT" \
    --image_size $RES \
    --txt "$PROMPT" \
    --output_dir "$(dirname "../../$OUTPUT")" \
    --output_name "$(basename "../../$OUTPUT")" \
    --num_inference_steps $STEPS \
    --cfg_scale $CFG \
    --seed $SEED

echo
echo "✓ Done. 4K image saved to $OUTPUT"
