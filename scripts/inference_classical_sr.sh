#!/usr/bin/env bash
# One-click inference for classical SR models (HiT-SR / SwinIR / MambaIR)
# fine-tuned on 4KLSDB.
#
# Usage:
#   bash scripts/inference_classical_sr.sh \
#       --model  hit_sr           # hit_sr | swinir | mambair
#       --scale  4                # 4 | 8 | 16
#       --input  data/4KLSDB/test/LR_x4
#       --output results/hit_sr_x4
#       [--ckpt release_ckpts/hit_sr/4KLSDB_x4.pth]   # optional override

set -euo pipefail

# ---- defaults ----
MODEL="hit_sr"
SCALE=4
INPUT=""
OUTPUT=""
CKPT=""
CKPT_ROOT="${CKPT_ROOT:-release_ckpts}"

# ---- arg parsing ----
print_help() {
    sed -n '2,12p' "$0" | sed 's/^# *//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)  MODEL="$2"; shift 2;;
        --scale)  SCALE="$2"; shift 2;;
        --input)  INPUT="$2"; shift 2;;
        --output) OUTPUT="$2"; shift 2;;
        --ckpt)   CKPT="$2";  shift 2;;
        -h|--help) print_help;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

[[ -z "$INPUT"  ]] && { echo "ERROR: --input is required";  exit 1; }
[[ -z "$OUTPUT" ]] && { echo "ERROR: --output is required"; exit 1; }
[[ "$MODEL" =~ ^(hit_sr|swinir|mambair)$ ]] || { echo "ERROR: --model must be hit_sr|swinir|mambair"; exit 1; }
[[ "$SCALE" =~ ^(4|8|16)$ ]] || { echo "ERROR: --scale must be 4|8|16"; exit 1; }

if [[ -z "$CKPT" ]]; then
    CKPT="${CKPT_ROOT}/${MODEL}/4KLSDB_x${SCALE}.pth"
fi
[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT"; echo "  Run 'bash scripts/download_all_ckpts.sh' first."; exit 1; }

mkdir -p "$OUTPUT"

echo "==> model:  $MODEL"
echo "==> scale:  x$SCALE"
echo "==> ckpt:   $CKPT"
echo "==> input:  $INPUT"
echo "==> output: $OUTPUT"
echo

case "$MODEL" in
    hit_sr)
        # Pin to upstream HiT-SR API
        cd models/hit_sr
        python basicsr/test.py \
            -opt options/Test/test_HiT_SRF_x${SCALE}.yml \
            --pretrained_model "../../$CKPT" \
            --input "../../$INPUT" \
            --output "../../$OUTPUT"
        ;;
    swinir)
        cd models/swinir
        python main_test_swinir.py \
            --task classical_sr --scale $SCALE \
            --model_path "../../$CKPT" \
            --folder_lq "../../$INPUT" \
            --save_dir "../../$OUTPUT"
        ;;
    mambair)
        cd models/mambair
        python basicsr/test.py \
            -opt options/test/test_MambaIR_classicSR_x${SCALE}.yml \
            --pretrained_model "../../$CKPT" \
            --input "../../$INPUT" \
            --output "../../$OUTPUT"
        ;;
esac

echo
echo "✓ Done. Results in $OUTPUT"
