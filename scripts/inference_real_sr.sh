#!/usr/bin/env bash
# One-click inference for real-world SR models (OSEDiff / SeeSR)
# fine-tuned on 4KLSDB.
#
# Usage:
#   bash scripts/inference_real_sr.sh \
#       --model  seesr            # seesr | osediff
#       --scale  4                # 4 | 8 | 16
#       --input  data/4KLSDB/test/LR_real_x4
#       --output results/seesr_x4
#       [--ckpt-dir release_ckpts/seesr/x4]

set -euo pipefail

MODEL="seesr"
SCALE=4
INPUT=""
OUTPUT=""
CKPT_DIR=""
CKPT_ROOT="${CKPT_ROOT:-release_ckpts}"

print_help() { sed -n '2,12p' "$0" | sed 's/^# *//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL="$2"; shift 2;;
        --scale)    SCALE="$2"; shift 2;;
        --input)    INPUT="$2"; shift 2;;
        --output)   OUTPUT="$2"; shift 2;;
        --ckpt-dir) CKPT_DIR="$2"; shift 2;;
        -h|--help)  print_help;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

[[ -z "$INPUT"  ]] && { echo "ERROR: --input is required";  exit 1; }
[[ -z "$OUTPUT" ]] && { echo "ERROR: --output is required"; exit 1; }
[[ "$MODEL" =~ ^(seesr|osediff)$ ]] || { echo "ERROR: --model must be seesr|osediff"; exit 1; }
[[ "$SCALE" =~ ^(4|8|16)$ ]] || { echo "ERROR: --scale must be 4|8|16"; exit 1; }

[[ -z "$CKPT_DIR" ]] && CKPT_DIR="${CKPT_ROOT}/${MODEL}/x${SCALE}"
[[ -d "$CKPT_DIR" ]] || { echo "ERROR: checkpoint dir not found: $CKPT_DIR"; echo "  Run 'bash scripts/download_all_ckpts.sh' first."; exit 1; }

mkdir -p "$OUTPUT"

echo "==> model:    $MODEL"
echo "==> scale:    x$SCALE"
echo "==> ckpt_dir: $CKPT_DIR"
echo "==> input:    $INPUT"
echo "==> output:   $OUTPUT"
echo

case "$MODEL" in
    seesr)
        cd models/seesr
        python test_seesr.py \
            --pretrained_model_path "../../$CKPT_DIR" \
            --image_path  "../../$INPUT" \
            --output_dir  "../../$OUTPUT" \
            --upscale     "$SCALE" \
            --process_size 512 \
            --guidance_scale 5.5
        ;;
    osediff)
        cd models/osediff
        python test_osediff.py \
            --pretrained_model_path "../../$CKPT_DIR" \
            --image_path  "../../$INPUT" \
            --output_dir  "../../$OUTPUT" \
            --upscale     "$SCALE"
        ;;
esac

echo
echo "✓ Done. Results in $OUTPUT"
