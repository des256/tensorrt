#!/usr/bin/env bash
#
# Generate TensorRT checkpoints (consolidated ONNX) for Moonshine Streaming.
#
# Consolidates the raw ONNX export (with external weight files produced by
# torch.onnx.export's dynamo backend) into self-contained .onnx files that
# trtexec can consume directly.
#
# Moonshine Streaming is a HuggingFace encoder-decoder ASR model, not an LLM,
# so TensorRT-LLM's convert_checkpoint.py does not apply.  The standard
# TensorRT path is:
#
#   HuggingFace  →  ONNX (this script)  →  trtexec  →  .engine
#
# Quantization (FP16, INT8) is controlled by trtexec flags at engine-build
# time, not at checkpoint time.  This script produces the single set of FP32
# ONNX files that all engine variants are built from.
#
# If a raw ONNX export already exists (from build_onnx.py), it is reused.
# Otherwise the HuggingFace model is exported fresh (requires transformers
# >=5.0.0 and torch >=2.6).
#
# Produces:  data/moonshine/ckpt/encoder_model.onnx
#            data/moonshine/ckpt/decoder_model.onnx
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SOURCE_DIR="$ROOT_DIR/data/moonshine/source"
RAW_DIR="$ROOT_DIR/data/moonshine/onnx/_raw"
CKPT_DIR="$ROOT_DIR/data/moonshine/ckpt"
PYTHON="${PYTHON:-/usr/bin/python3}"

mkdir -p "$CKPT_DIR"

# ---------------------------------------------------------------------------
# Moonshine Streaming exports as two components:
#   encoder_model.onnx  — audio waveform → encoder hidden states
#   decoder_model.onnx  — decoder input IDs + encoder states → logits
#
# The torch.onnx dynamo exporter stores weights in external .onnx.data files.
# We re-load each .onnx and re-save it as a single self-contained protobuf
# so trtexec can consume it without chasing external references.
# ---------------------------------------------------------------------------

# Decide whether to export fresh or reuse existing raw ONNX.
if [[ -f "$RAW_DIR/encoder_model.onnx" && -f "$RAW_DIR/decoder_model.onnx" ]]; then
    echo "=== Reusing existing raw ONNX from $RAW_DIR ==="
    EXPORT_DIR="$RAW_DIR"
else
    echo "=== No raw ONNX found — exporting HuggingFace → ONNX ==="
    if [[ ! -f "$SOURCE_DIR/model.safetensors" ]]; then
        echo "ERROR: Neither raw ONNX nor $SOURCE_DIR/model.safetensors found." >&2
        exit 1
    fi
    EXPORT_DIR=$(mktemp -d -t moonshine_ckpt_XXXX)
    trap 'rm -rf "$EXPORT_DIR"' EXIT

    "$PYTHON" - "$SOURCE_DIR" "$EXPORT_DIR" <<'PYEOF'
import sys
from pathlib import Path

import torch
from transformers import MoonshineStreamingForConditionalGeneration

source_dir = sys.argv[1]
export_dir = Path(sys.argv[2])

print(f"Loading model from {source_dir} ...")
# Eager attention avoids SymBool tracing issues with SDPA during export.
model = MoonshineStreamingForConditionalGeneration.from_pretrained(
    source_dir, attn_implementation="eager",
)
model.eval()


DEPTH = 14   # decoder_num_hidden_layers
NHEADS = 10  # decoder_num_attention_heads
HEAD_DIM = 64  # decoder_hidden_size // decoder_num_attention_heads


class DecoderWithKVCache(torch.nn.Module):
    """Wraps decoder + proj_out with explicit KV-cache tensors for ONNX."""

    def __init__(self, m):
        super().__init__()
        self.decoder = m.get_decoder()
        self.proj_out = m.proj_out
        self.depth = DEPTH

    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask,
                k_self, v_self):
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        sa_cache = DynamicCache()
        for i in range(self.depth):
            sa_cache.update(k_self[i], v_self[i], i)
        ca_cache = DynamicCache()
        past_kv = EncoderDecoderCache(sa_cache, ca_cache)

        dec_out = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            past_key_values=past_kv,
        )
        logits = self.proj_out(dec_out.last_hidden_state)

        new_sa = dec_out.past_key_values.self_attention_cache
        new_k = torch.stack([layer.keys for layer in new_sa.layers])
        new_v = torch.stack([layer.values for layer in new_sa.layers])
        return logits, new_k, new_v


# -- Encoder --
encoder = model.get_encoder()
dummy_audio = torch.randn(1, 16000)
dummy_mask = torch.ones(1, 16000, dtype=torch.long)

print("Exporting encoder ...")
torch.onnx.export(
    encoder,
    (dummy_audio, dummy_mask),
    str(export_dir / "encoder_model.onnx"),
    input_names=["input_values", "attention_mask"],
    output_names=["last_hidden_state", "encoder_attention_mask"],
    dynamic_axes={
        "input_values": {0: "batch", 1: "audio_len"},
        "attention_mask": {0: "batch", 1: "audio_len"},
        "last_hidden_state": {0: "batch", 1: "enc_len"},
        "encoder_attention_mask": {0: "batch", 1: "enc_len"},
    },
    opset_version=18,
)

# -- Decoder (with KV cache) --
with torch.no_grad():
    enc_out = encoder(input_values=dummy_audio, attention_mask=dummy_mask)
enc_hidden = enc_out.last_hidden_state
enc_attn_mask = torch.ones(1, enc_hidden.shape[1], dtype=torch.long)
dec_ids = torch.tensor([[1]], dtype=torch.long)
k_past = torch.zeros(DEPTH, 1, NHEADS, 1, HEAD_DIM)
v_past = torch.zeros(DEPTH, 1, NHEADS, 1, HEAD_DIM)

dec_wrapper = DecoderWithKVCache(model)
dec_wrapper.eval()

print("Exporting decoder (with KV cache) ...")
torch.onnx.export(
    dec_wrapper,
    (dec_ids, enc_hidden, enc_attn_mask, k_past, v_past),
    str(export_dir / "decoder_model.onnx"),
    input_names=["input_ids", "encoder_hidden_states", "encoder_attention_mask",
                 "k_self", "v_self"],
    output_names=["logits", "out_k_self", "out_v_self"],
    dynamic_axes={
        "input_ids":              {0: "batch", 1: "dec_len"},
        "encoder_hidden_states":  {0: "batch", 1: "enc_len"},
        "encoder_attention_mask": {0: "batch", 1: "enc_len"},
        "k_self":                 {3: "past_len"},
        "v_self":                 {3: "past_len"},
        "logits":                 {0: "batch", 1: "dec_len"},
        "out_k_self":             {3: "new_len"},
        "out_v_self":             {3: "new_len"},
    },
    opset_version=18,
)

del model, encoder, dec_wrapper
print("ONNX export complete.")
PYEOF
fi

# ---------------------------------------------------------------------------
# Consolidate: merge external data into self-contained .onnx files.
# ---------------------------------------------------------------------------
echo "=== Consolidating ONNX checkpoints ==="
"$PYTHON" - "$EXPORT_DIR" "$CKPT_DIR" <<'PYEOF'
import sys
from pathlib import Path

import onnx

export_dir = Path(sys.argv[1])
ckpt_dir   = Path(sys.argv[2])

PROTO_LIMIT = 2 * 1024**3  # 2 GB protobuf serialization ceiling

for onnx_file in sorted(export_dir.glob("*.onnx")):
    print(f"Consolidating {onnx_file.name} ...")
    proto = onnx.load(str(onnx_file), load_external_data=True)
    dst = ckpt_dir / onnx_file.name
    if proto.ByteSize() < PROTO_LIMIT:
        onnx.save(proto, str(dst))
    else:
        # Model exceeds 2 GB — must use external data so protobuf doesn't
        # silently corrupt the file.  trtexec handles this transparently.
        onnx.save_model(
            proto,
            str(dst),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=dst.stem + ".onnx_data",
        )
    del proto

print(f"Checkpoints written to {ckpt_dir}")
PYEOF

echo "All checkpoints written to $CKPT_DIR"
