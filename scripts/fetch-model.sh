#!/bin/bash
# Downloads the bundled on-device model into the app's resources.
# Default: Qwen2.5-1.5B-Instruct abliterated, Q4_K_M (~986 MB), Apache-2.0.
# Override with MODEL_URL=... ./Scripts/fetch-model.sh (e.g. the 0.5B Q4_K_M for a smaller IPA).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_URL="${MODEL_URL:-https://huggingface.co/mradermacher/Qwen2-VL-2B-Instruct-abliterated-GGUF/resolve/main/Qwen2-VL-2B-Instruct-abliterated.Q4_K_M.gguf}"
MMPROJ_URL="${MMPROJ_URL:-https://huggingface.co/mradermacher/Qwen2-VL-2B-Instruct-abliterated-GGUF/resolve/main/Qwen2-VL-2B-Instruct-abliterated.mmproj-fp16.gguf}"
MIN_BYTES="${MIN_BYTES:-600000000}"

mkdir -p ios/IPhoneAgent/LocalModel/Resources
OUT=ios/IPhoneAgent/LocalModel/Resources/model.gguf

if [ -f "$OUT" ] && [ "$(wc -c < "$OUT")" -gt "$MIN_BYTES" ]; then
  echo "model already present ($(wc -c < "$OUT") bytes)"
  exit 0
fi

echo "downloading $MODEL_URL"
curl -L --fail --retry 3 -o "$OUT" "$MODEL_URL"
SIZE=$(wc -c < "$OUT")
echo "downloaded $SIZE bytes"
if [ "$SIZE" -lt "$MIN_BYTES" ]; then
  echo "ERROR: model file looks truncated ($SIZE bytes)" >&2
  exit 1
fi

# the vision projector: without it the model answers text only
PROJ=ios/IPhoneAgent/LocalModel/Resources/mmproj.gguf
if [ ! -f "$PROJ" ] || [ "$(wc -c < "$PROJ")" -lt 100000000 ]; then
  echo "downloading $MMPROJ_URL"
  curl -L --fail --retry 3 -o "$PROJ" "$MMPROJ_URL"
fi
echo "projector: $(wc -c < "$PROJ") bytes"
