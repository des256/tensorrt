#!/usr/bin/env bash
#
# Build TensorRT engines for Moonshine Streaming Medium (Jetson profile).
# Targets: Jetson Orin NX (16 GB shared RAM, sm_87).
# Run directly on the Jetson (no docker).
# Requires: trtexec (apt install libnvinfer-bin)
#
# Moonshine Streaming is a HuggingFace encoder-decoder ASR model (not an LLM),
# so we use trtexec directly on ONNX checkpoints rather than trtllm-build.
#
# Each variant builds engines for both components:
#   encoder_model.onnx  — audio waveform → encoder hidden states
#   decoder_model.onnx  — decoder input IDs + encoder states → logits
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CKPT_DIR="$ROOT_DIR/data/moonshine/ckpt"
ENGINE_DIR="$ROOT_DIR/data/moonshine/engine/jetson"

if [[ ! -f "$CKPT_DIR/encoder_model.onnx" ]]; then
    echo "ERROR: $CKPT_DIR/encoder_model.onnx not found. Run build_ckpt.sh first." >&2
    exit 1
fi

# ── Encoder dynamic shape profiles ────────────────────────────────────────
# Input: input_values [batch, audio_len], attention_mask [batch, audio_len]
#
# audio_len must be a multiple of 80 (pad_to_multiple_of in preprocessor).
# At 16 kHz:  1600 samples ≈ 0.1 s,  80000 ≈ 5 s,  240000 ≈ 15 s
#
# Jetson profile: max 15 s (vs 30 s on desktop) to stay within shared-memory
# budget.
ENC_SHAPES_MIN="input_values:1x1600,attention_mask:1x1600"
ENC_SHAPES_OPT="input_values:1x80000,attention_mask:1x80000"
ENC_SHAPES_MAX="input_values:1x240000,attention_mask:1x240000"

# ── Decoder dynamic shape profiles ────────────────────────────────────────
# Called once per generated token with dec_len=1 (autoregressive decoding).
# Dynamic dims:
#   enc_len  — encoder sequence length, depends on audio duration:
#              enc_len = ((audio_len // 80) - 1) // 4 + 1
#              0.1 s → 5,   5 s → 250,   15 s → 750
#   past_len — self-attention KV cache length (0 at first token, grows to MAX_TOKENS=64)
#              k_self/v_self shape: [DEPTH=14, 1, NHEADS=10, past_len, HEAD_DIM=64]
DEC_SHAPES_MIN="input_ids:1x1,encoder_hidden_states:1x5x768,encoder_attention_mask:1x5,k_self:14x1x10x0x64,v_self:14x1x10x0x64"
DEC_SHAPES_OPT="input_ids:1x1,encoder_hidden_states:1x250x768,encoder_attention_mask:1x250,k_self:14x1x10x32x64,v_self:14x1x10x32x64"
DEC_SHAPES_MAX="input_ids:1x1,encoder_hidden_states:1x750x768,encoder_attention_mask:1x750,k_self:14x1x10x64x64,v_self:14x1x10x64x64"

# Common flags shared by every trtexec invocation
# Jetson: 2 GB workspace (16 GB shared with CPU, be conservative)
COMMON=(
    --memPoolSize=workspace:2048
)

# ---------------------------------------------------------------------------
# build_component  <variant>  <component>  <extra_trtexec_flags...>
#
# Builds a TensorRT engine for one ONNX component (encoder or decoder)
# into the specified variant output directory.
# ---------------------------------------------------------------------------
build_component() {
    local variant="$1"
    local component="$2"
    shift 2
    local extra_flags=("$@")

    local onnx="$CKPT_DIR/${component}_model.onnx"
    local out_dir="$ENGINE_DIR/$variant"
    mkdir -p "$out_dir"

    if [[ "$component" == "encoder" ]]; then
        local shapes=(
            --minShapes="$ENC_SHAPES_MIN"
            --optShapes="$ENC_SHAPES_OPT"
            --maxShapes="$ENC_SHAPES_MAX"
        )
    else
        local shapes=(
            --minShapes="$DEC_SHAPES_MIN"
            --optShapes="$DEC_SHAPES_OPT"
            --maxShapes="$DEC_SHAPES_MAX"
        )
    fi

    echo "  Building $component ..."
    trtexec \
        --onnx="$onnx" \
        --saveEngine="$out_dir/${component}_model.engine" \
        "${shapes[@]}" \
        "${COMMON[@]}" \
        "${extra_flags[@]}"
}

# ---------------------------------------------------------------------------
# build_variant  <variant_name>  <extra_trtexec_flags...>
# ---------------------------------------------------------------------------
build_variant() {
    local variant="$1"
    shift
    echo "=== $variant ==="
    build_component "$variant" encoder "$@"
    build_component "$variant" decoder "$@"
}

# ---------------------------------------------------------------------------
# f16  —  FP16 weights, FP16 activations
# ---------------------------------------------------------------------------
build_variant f16 --fp16

# ---------------------------------------------------------------------------
# q8f16  —  INT8 weights+activations, FP16 fallback  (W8A8)
#
# Uses builder-internal calibration (random data).
# ---------------------------------------------------------------------------
build_variant q8f16 --int8 --fp16

# ---------------------------------------------------------------------------
# q8i8  —  INT8 weights+activations, FP32 fallback  (W8A8)
#
# Uses builder-internal calibration (random data).
# ---------------------------------------------------------------------------
build_variant q8i8 --int8

# ---------------------------------------------------------------------------
# q4f16  —  INT4 weights, INT8 activations, FP16 fallback  (W4A8)
# ---------------------------------------------------------------------------
build_variant q4f16 --int8 --int4 --fp16

# ---------------------------------------------------------------------------
# q4i8  —  INT4 weights, INT8 activations, FP32 fallback  (W4A8)
# ---------------------------------------------------------------------------
build_variant q4i8 --int8 --int4

echo "All engines written to $ENGINE_DIR"
