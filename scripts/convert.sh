#!/bin/bash
# Convert a HuggingFace model to intermediate format for engine building.
#
# Usage: ./scripts/convert.sh <model> --type llm|standard [--quant fp16|int4|int8]
#
# For LLMs:  Runs TRT-LLM checkpoint conversion.
#   Input:  data/<model>/source/
#   Output: data/<model>/trtllm_ckpt/<quant>/
#
# For standard models: Exports to ONNX.
#   Input:  data/<model>/source/
#   Output: data/<model>/onnx/
#
# This script is intended to run inside the Docker container where
# TRT-LLM and Python dependencies are available.
#
# Examples:
#   ./scripts/convert.sh llama3 --type llm --quant int4
#   ./scripts/convert.sh parakeet --type standard

set -euo pipefail

MODEL=""
TYPE=""
QUANT="fp16"

while [ $# -gt 0 ]; do
    case "$1" in
        --type)  TYPE="$2";  shift 2 ;;
        --quant) QUANT="$2"; shift 2 ;;
        -*)      echo "Unknown flag: $1"; exit 1 ;;
        *)       MODEL="$1"; shift ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$TYPE" ]; then
    echo "Usage: $0 <model> --type llm|standard [--quant fp16|int4|int8]"
    exit 1
fi

SOURCE_DIR="data/${MODEL}/source"
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: source directory not found: ${SOURCE_DIR}"
    echo "Run ./scripts/download.sh first."
    exit 1
fi

case "$TYPE" in
    llm)
        OUTPUT_DIR="data/${MODEL}/trtllm_ckpt/${QUANT}"
        mkdir -p "$OUTPUT_DIR"
        echo "Converting LLM checkpoint: ${SOURCE_DIR} → ${OUTPUT_DIR} (quant=${QUANT})"

        # Dispatch to TRT-LLM's checkpoint conversion.
        # This uses the convert_checkpoint.py script from TRT-LLM examples.
        # Adjust the script path based on the model architecture.
        python3 -c "
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import QuantConfig, QuantAlgo

quant_map = {
    'fp16': None,
    'int4': QuantAlgo.W4A16_AWQ,
    'int8': QuantAlgo.W8A16,
}
quant = quant_map.get('${QUANT}')
quant_config = QuantConfig(quant_algo=quant) if quant else None

llm = LLM(model='${SOURCE_DIR}', quant_config=quant_config)
llm.save('${OUTPUT_DIR}')
print('Checkpoint conversion complete.')
"
        ;;
    standard)
        OUTPUT_DIR="data/${MODEL}/onnx"
        mkdir -p "$OUTPUT_DIR"
        echo "Exporting ONNX: ${SOURCE_DIR} → ${OUTPUT_DIR}"

        python3 -c "
from optimum.exporters.onnx import main_export
main_export(model_name_or_path='${SOURCE_DIR}', output='${OUTPUT_DIR}')
print('ONNX export complete.')
"
        ;;
    *)
        echo "Error: --type must be 'llm' or 'standard'"
        exit 1
        ;;
esac

echo "Done: ${OUTPUT_DIR}"
