#!/bin/bash
# Build a TensorRT engine from converted model artifacts.
#
# Usage: ./scripts/build_engine.sh <model> --type llm|standard --platform desktop|jetson [--quant fp16|int4]
#
# For LLMs:
#   Input:  data/<model>/trtllm_ckpt/<quant>/
#   Output: data/<model>/engine/<platform>/
#   Tool:   trtllm-build
#
# For standard models:
#   Input:  data/<model>/onnx/model.onnx
#   Output: data/<model>/engine/<platform>/model.engine
#   Tool:   trtexec
#
# This script is intended to run inside the Docker container on the
# TARGET platform (desktop or Jetson).
#
# Examples:
#   ./scripts/build_engine.sh llama3 --type llm --platform desktop --quant int4
#   ./scripts/build_engine.sh parakeet --type standard --platform jetson

set -euo pipefail

MODEL=""
TYPE=""
PLATFORM=""
QUANT="fp16"

while [ $# -gt 0 ]; do
    case "$1" in
        --type)     TYPE="$2";     shift 2 ;;
        --platform) PLATFORM="$2"; shift 2 ;;
        --quant)    QUANT="$2";    shift 2 ;;
        -*)         echo "Unknown flag: $1"; exit 1 ;;
        *)          MODEL="$1";    shift ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$TYPE" ] || [ -z "$PLATFORM" ]; then
    echo "Usage: $0 <model> --type llm|standard --platform desktop|jetson [--quant fp16|int4]"
    exit 1
fi

OUTPUT_DIR="data/${MODEL}/engine/${PLATFORM}"
mkdir -p "$OUTPUT_DIR"

case "$TYPE" in
    llm)
        CKPT_DIR="data/${MODEL}/trtllm_ckpt/${QUANT}"
        if [ ! -d "$CKPT_DIR" ]; then
            echo "Error: checkpoint not found: ${CKPT_DIR}"
            echo "Run ./scripts/convert.sh first."
            exit 1
        fi

        echo "Building TRT-LLM engine: ${CKPT_DIR} → ${OUTPUT_DIR}"
        trtllm-build \
            --checkpoint_dir "$CKPT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --gemm_plugin auto
        ;;
    standard)
        ONNX_DIR="data/${MODEL}/onnx"
        # Find the ONNX model file.
        ONNX_FILE=$(find "$ONNX_DIR" -name "*.onnx" -type f | head -1)
        if [ -z "$ONNX_FILE" ]; then
            echo "Error: no .onnx file found in ${ONNX_DIR}"
            echo "Run ./scripts/convert.sh first."
            exit 1
        fi

        ENGINE_FILE="${OUTPUT_DIR}/model.engine"
        echo "Building TensorRT engine: ${ONNX_FILE} → ${ENGINE_FILE}"
        trtexec \
            --onnx="$ONNX_FILE" \
            --saveEngine="$ENGINE_FILE" \
            --fp16
        ;;
    *)
        echo "Error: --type must be 'llm' or 'standard'"
        exit 1
        ;;
esac

echo "Done: ${OUTPUT_DIR}"
