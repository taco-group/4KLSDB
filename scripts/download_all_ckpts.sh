#!/usr/bin/env bash
# Download every 4KLSDB-fine-tuned checkpoint into ./release_ckpts/<model>/
# Requires: pip install -U "huggingface_hub[cli]"

set -euo pipefail

DEST="${1:-release_ckpts}"
HF_ORG="taco-group"

mkdir -p "$DEST"

echo "==> Destination: $DEST"
echo "==> HF org:      $HF_ORG"
echo

# (repo_id, sub-folder name on disk)
MODELS=(
    "${HF_ORG}/4KLSDB-HiT-SR   hit_sr"
    "${HF_ORG}/4KLSDB-SwinIR   swinir"
    "${HF_ORG}/4KLSDB-MambaIR  mambair"
    "${HF_ORG}/4KLSDB-OSEDiff  osediff"
    "${HF_ORG}/4KLSDB-SeeSR    seesr"
    "${HF_ORG}/4KLSDB-Sana     sana"
)

for entry in "${MODELS[@]}"; do
    set -- $entry
    REPO=$1
    SUB=$2
    echo "==> Downloading $REPO → $DEST/$SUB"
    huggingface-cli download "$REPO" --local-dir "$DEST/$SUB" --local-dir-use-symlinks False
    echo
done

echo "✓ All checkpoints downloaded to $DEST/"
echo "  hit_sr/   swinir/   mambair/   osediff/   seesr/   sana/"
