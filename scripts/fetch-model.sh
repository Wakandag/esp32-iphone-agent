#!/bin/bash
# Downloads the bundled on-device model into the app's resources.
# Default: Qwen2.5-1.5B-Instruct abliterated, Q4_K_M (~986 MB), Apache-2.0.
# Override with MODEL_URL=... ./Scripts/fetch-model.sh (e.g. the 0.5B Q4_K_M for a smaller IPA).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_URL="${MODEL_URL:-https://huggingface.co/Vali/Qwen2.5-1.5B-Instruct-abliterated-Q4_K_M-GGUF/resolve/main/qwen2.5-1.5b-instruct-abliterated-q4_k_m.gguf}"
MIN_BYTES="${MIN_BYTES:-300000000}"

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
