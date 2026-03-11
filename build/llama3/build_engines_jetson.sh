#!/usr/bin/env bash
#
# Build TensorRT-LLM engines for Llama 3.2 3B (Jetson profile).
# Targets: Jetson Orin NX (16 GB shared RAM, sm_87).
# Run directly on the Jetson (no docker).
# Requires: trtllm-build (pip install tensorrt-llm)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CKPT_DIR="$ROOT_DIR/data/llama3-3b/ckpt"
ENGINE_DIR="$ROOT_DIR/data/llama3-3b/engine/jetson"

# ── Jetson profile (conservative — 16 GB shared with CPU) ────────────────
MAX_BATCH_SIZE=1
MAX_INPUT_LEN=1024
MAX_SEQ_LEN=2048

# Common flags shared by every variant
COMMON=(
    --max_batch_size "$MAX_BATCH_SIZE"
    --max_input_len  "$MAX_INPUT_LEN"
    --max_seq_len    "$MAX_SEQ_LEN"
    --gpt_attention_plugin auto
    --gemm_plugin auto
    --paged_kv_cache enable
    --remove_input_padding enable
    --context_fmha enable
)

# ---------------------------------------------------------------------------
# f16  —  skipped: 6.8 GB weights exceed Jetson's 16 GB shared memory budget
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# q8f16  —  INT8 weight-only, FP16 activations  (W8A16)
# ---------------------------------------------------------------------------
echo "=== q8f16 ==="
trtllm-build \
    --checkpoint_dir "$CKPT_DIR/q8f16" \
    --output_dir     "$ENGINE_DIR/q8f16" \
    "${COMMON[@]}"

# ---------------------------------------------------------------------------
# q8i8  —  INT8 weights + INT8 activations via SmoothQuant  (W8A8)
# ---------------------------------------------------------------------------
echo "=== q8i8 ==="
trtllm-build \
    --checkpoint_dir "$CKPT_DIR/q8i8" \
    --output_dir     "$ENGINE_DIR/q8i8" \
    "${COMMON[@]}"

# ---------------------------------------------------------------------------
# q4f16  —  INT4 weight-only, FP16 activations  (W4A16)
# ---------------------------------------------------------------------------
echo "=== q4f16 ==="
trtllm-build \
    --checkpoint_dir "$CKPT_DIR/q4f16" \
    --output_dir     "$ENGINE_DIR/q4f16" \
    "${COMMON[@]}"

# ---------------------------------------------------------------------------
# q4i8  —  INT4 weights, INT8 KV-cache  (W4A16 + INT8 KV)
# ---------------------------------------------------------------------------
echo "=== q4i8 ==="
trtllm-build \
    --checkpoint_dir "$CKPT_DIR/q4i8" \
    --output_dir     "$ENGINE_DIR/q4i8" \
    "${COMMON[@]}"

echo "All engines written to $ENGINE_DIR"
