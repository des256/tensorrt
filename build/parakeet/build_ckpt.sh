#!/usr/bin/env bash
#
# Generate TensorRT checkpoints (consolidated ONNX) for Parakeet TDT 0.6B.
#
# Consolidates the raw ONNX export (with scattered external weight files)
# into self-contained .onnx files that trtexec can consume directly.
#
# Parakeet is an ASR encoder-decoder, not an LLM, so TensorRT-LLM's
# convert_checkpoint.py does not apply.  The standard TensorRT path is:
#
#   NeMo  →  ONNX (this script)  →  trtexec  →  .engine
#
# Quantization (FP16, INT8) is controlled by trtexec flags at engine-build
# time, not at checkpoint time.  This script produces the single set of FP32
# ONNX files that all engine variants are built from.
#
# If a raw ONNX export already exists (from build_onnx.py), it is reused.
# Otherwise the NeMo model is exported fresh (requires nemo_toolkit[asr]).
#
# Produces:  data/parakeet/ckpt/encoder-model.onnx
#            data/parakeet/ckpt/decoder_joint-model.onnx
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

NEMO_FILE="$ROOT_DIR/data/parakeet/source/parakeet-tdt-0.6b-v3.nemo"
RAW_DIR="$ROOT_DIR/data/parakeet/onnx/_raw"
CKPT_DIR="$ROOT_DIR/data/parakeet/ckpt"
PYTHON="${PYTHON:-/usr/bin/python3}"

mkdir -p "$CKPT_DIR"

# ---------------------------------------------------------------------------
# NeMo TDT models export as two components:
#   encoder-model.onnx        — Conformer encoder
#   decoder_joint-model.onnx  — LSTM decoder + joint network
#
# The raw export scatters weights into external data files (the encoder
# alone produces 100+ sidecar files).  We re-load each .onnx and re-save
# it as a single self-contained protobuf so trtexec can consume it without
# chasing external references.
# ---------------------------------------------------------------------------

# Decide whether to export fresh or reuse existing raw ONNX.
if [[ -f "$RAW_DIR/encoder-model.onnx" && -f "$RAW_DIR/decoder_joint-model.onnx" ]]; then
    echo "=== Reusing existing raw ONNX from $RAW_DIR ==="
    EXPORT_DIR="$RAW_DIR"
else
    echo "=== No raw ONNX found — exporting NeMo → ONNX ==="
    if [[ ! -f "$NEMO_FILE" ]]; then
        echo "ERROR: Neither raw ONNX nor $NEMO_FILE found." >&2
        exit 1
    fi
    EXPORT_DIR=$(mktemp -d -t parakeet_ckpt_XXXX)
    trap 'rm -rf "$EXPORT_DIR"' EXIT

    "$PYTHON" - "$NEMO_FILE" "$EXPORT_DIR" <<'PYEOF'
import sys
from pathlib import Path

nemo_file  = sys.argv[1]
export_dir = Path(sys.argv[2])

# numba / coverage compatibility shim (same as build_onnx.py)
_cov = {k: sys.modules.pop(k) for k in list(sys.modules)
        if k == "coverage" or k.startswith("coverage.")}
sys.modules["coverage"] = None          # type: ignore[assignment]
import nemo.collections.asr as nemo_asr  # noqa: E402
sys.modules.pop("coverage", None)
sys.modules.update(_cov)

print(f"Loading {nemo_file} ...")
model = nemo_asr.models.ASRModel.restore_from(nemo_file)

print("Exporting ONNX (opset 17) ...")
model.export(str(export_dir / "model.onnx"), onnx_opset_version=17)
del model
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

for onnx_file in sorted(export_dir.glob("*.onnx")):
    print(f"Consolidating {onnx_file.name} ...")
    proto = onnx.load(str(onnx_file), load_external_data=True)
    onnx.save(proto, str(ckpt_dir / onnx_file.name))
    del proto

print(f"Checkpoints written to {ckpt_dir}")
PYEOF

echo "All checkpoints written to $CKPT_DIR"
