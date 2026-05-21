#!/usr/bin/env bash
# Download every 4KLSDB-fine-tuned checkpoint into ./release_ckpts/<model>/
# All ckpts live inside the dataset repo, under ckpts/<model>/.
# Requires: pip install -U "huggingface_hub[cli]"

set -euo pipefail

DEST="${1:-release_ckpts}"
REPO="SingleBicycle/4KLSDB"

mkdir -p "$DEST"

echo "==> Destination: $DEST"
echo "==> Source:      hf://$REPO/ckpts/"
echo

MODELS=(hit_sr swinir mambair osediff seesr sana)

for m in "${MODELS[@]}"; do
    echo "==> Downloading ckpts/$m"
    huggingface-cli download "$REPO" --repo-type dataset \
        --include "ckpts/$m/*" \
        --local-dir "$DEST"
    # Flatten: $DEST/ckpts/<m> -> $DEST/<m>
    if [[ -d "$DEST/ckpts/$m" ]]; then
        mkdir -p "$DEST"
        rsync -a --remove-source-files "$DEST/ckpts/$m/" "$DEST/$m/" 2>/dev/null \
            || cp -r "$DEST/ckpts/$m/." "$DEST/$m/"
    fi
done
rm -rf "$DEST/ckpts"

echo
echo "✓ All checkpoints downloaded to $DEST/"
echo "  $(ls "$DEST" | tr '\n' ' ')"
