#!/usr/bin/env bash
#
# Generate TensorRT-LLM checkpoints for Llama 3.2 3B.
# Requires tensorrt_llm to be importable (run where TRT-LLM is installed).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TRTLLM_ROOT="${TRTLLM_ROOT:-$HOME/TensorRT-LLM}"
CONVERT="$TRTLLM_ROOT/examples/llama/convert_checkpoint.py"

MODEL_DIR="$ROOT_DIR/data/llama3-3b/source"
CKPT_DIR="$ROOT_DIR/data/llama3-3b/ckpt"

if [[ ! -f "$CONVERT" ]]; then
    echo "ERROR: $CONVERT not found. Set TRTLLM_ROOT." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# f16  —  FP16 weights, FP16 activations
# ---------------------------------------------------------------------------
echo "=== f16 ==="
python3 "$CONVERT" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$CKPT_DIR/f16" \
    --dtype float16

# ---------------------------------------------------------------------------
# q8f16  —  INT8 weight-only, FP16 activations  (W8A16)
# ---------------------------------------------------------------------------
echo "=== q8f16 ==="
python3 "$CONVERT" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$CKPT_DIR/q8f16" \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int8

# ---------------------------------------------------------------------------
# q8i8  —  INT8 weights + INT8 activations via SmoothQuant  (W8A8)
#
# SmoothQuant needs calibration data; uses cnn_dailymail by default.
# --per_channel --per_token gives the most accurate W8A8 variant.
# Needs the full model in memory (no --load_by_shard).
# ---------------------------------------------------------------------------
echo "=== q8i8 ==="
python3 "$CONVERT" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$CKPT_DIR/q8i8" \
    --dtype float16 \
    --smoothquant 0.5 \
    --per_channel \
    --per_token

# ---------------------------------------------------------------------------
# q4f16  —  INT4 weight-only, FP16 activations  (W4A16)
# ---------------------------------------------------------------------------
echo "=== q4f16 ==="
python3 "$CONVERT" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$CKPT_DIR/q4f16" \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int4

# ---------------------------------------------------------------------------
# q4i8  —  INT4 weights, INT8 KV-cache  (W4A16 + INT8 KV)
#
# TRT-LLM doesn't support true W4A8 (SmoothQuant is W8-only).
# The closest equivalent is INT4 weight-only with INT8 KV-cache,
# which quantises the memory-dominant KV tensors to INT8.
# ---------------------------------------------------------------------------
echo "=== q4i8 ==="
python3 "$CONVERT" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$CKPT_DIR/q4i8" \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int4 \
    --int8_kv_cache

echo "All checkpoints written to $CKPT_DIR"
