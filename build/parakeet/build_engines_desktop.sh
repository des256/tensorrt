#!/usr/bin/env bash
#
# Build TensorRT engines for Parakeet TDT 0.6B (desktop profile).
# Targets: RTX 4080 Super (16 GB VRAM, sm_89).
# Requires: trtexec (apt install libnvinfer-bin)
#
# Parakeet is an ASR encoder-decoder (not an LLM), so we use trtexec
# directly on ONNX checkpoints rather than trtllm-build.
#
# Each variant builds engines for both components:
#   encoder-model.onnx        — Conformer encoder
#   decoder_joint-model.onnx  — LSTM decoder + joint network
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CKPT_DIR="$ROOT_DIR/data/parakeet/ckpt"
ENGINE_DIR="$ROOT_DIR/data/parakeet/engine/desktop"

if [[ ! -f "$CKPT_DIR/encoder-model.onnx" ]]; then
    echo "ERROR: $CKPT_DIR/encoder-model.onnx not found. Run build_ckpt.sh first." >&2
    exit 1
fi

# ── Encoder dynamic shape profiles ────────────────────────────────────────
# Input: audio_signal [batch, 128, time], length [batch]
# Output: outputs [batch, 1024, enc_time], encoded_lengths [batch]
#
# At 20 ms per mel frame: 100 frames ≈ 2 s, 200 frames ≈ 4 s, 1500 frames ≈ 30 s
ENC_SHAPES_MIN="audio_signal:1x128x1,length:1"
ENC_SHAPES_OPT="audio_signal:1x128x200,length:1"
ENC_SHAPES_MAX="audio_signal:1x128x1500,length:1"

# ── Decoder-joint dynamic shape profiles ──────────────────────────────────
# Called once per encoder time-step with a single frame.
# Inputs:  encoder_outputs [batch,1024,1], targets [batch,1],
#          target_length [batch], input_states_{1,2} [2,batch,640]
# All dynamic dims are batch=1 in practice.
DEC_SHAPES_MIN="encoder_outputs:1x1024x1,targets:1x1,target_length:1,input_states_1:2x1x640,input_states_2:2x1x640"
DEC_SHAPES_OPT="$DEC_SHAPES_MIN"
DEC_SHAPES_MAX="$DEC_SHAPES_MIN"

# Common flags shared by every trtexec invocation
COMMON=(
    --memPoolSize=workspace:4096
)

# ---------------------------------------------------------------------------
# build_component  <component_name>  <extra_trtexec_flags...>
#
# Builds a TensorRT engine for one ONNX component (encoder or decoder_joint)
# into the specified variant output directory.
# ---------------------------------------------------------------------------
build_component() {
    local variant="$1"
    local component="$2"
    shift 2
    local extra_flags=("$@")

    local onnx="$CKPT_DIR/${component}-model.onnx"
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
        --saveEngine="$out_dir/${component}-model.engine" \
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
    build_component "$variant" decoder_joint "$@"
}

# ---------------------------------------------------------------------------
# f16  —  FP16 weights, FP16 activations
# ---------------------------------------------------------------------------
build_variant f16 --fp16

# ---------------------------------------------------------------------------
# q8f16  —  INT8 weights, FP16 activations  (W8A16)
#
# --fp16 --int8: TensorRT selects INT8 where profitable, FP16 elsewhere.
# Uses builder-internal calibration (random data) — adequate for latency
# benchmarking but not for accuracy validation.
# ---------------------------------------------------------------------------
build_variant q8f16 --fp16 --int8

# ---------------------------------------------------------------------------
# q4f16  —  INT4 weights, FP16 activations  (W4A16)
# ---------------------------------------------------------------------------
build_variant q4f16 --fp16 --int4

echo "All engines written to $ENGINE_DIR"
