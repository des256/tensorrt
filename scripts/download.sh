#!/bin/bash
# Download a HuggingFace model to data/<model>/source/.
#
# Usage: ./scripts/download.sh <model_id> [output_dir]
#
# Examples:
#   ./scripts/download.sh meta-llama/Llama-3.2-1B-Instruct
#   ./scripts/download.sh nvidia/parakeet-tdt-0.6b data/parakeet/source

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_id> [output_dir]"
    echo "  model_id:   HuggingFace model ID (e.g. meta-llama/Llama-3.2-1B-Instruct)"
    echo "  output_dir: Where to save (default: data/<basename>/source)"
    exit 1
fi

MODEL_ID="$1"
MODEL_NAME="$(basename "$MODEL_ID")"
OUTPUT_DIR="${2:-data/${MODEL_NAME}/source}"

echo "Downloading ${MODEL_ID} → ${OUTPUT_DIR}"
huggingface-cli download "$MODEL_ID" --local-dir "$OUTPUT_DIR"
echo "Done: ${OUTPUT_DIR}"
