"""Build ONNX variants for Moonshine Streaming STT model.

Exports the HuggingFace encoder-decoder model to ONNX via torch.onnx.export,
then produces five quantized variants matching the Parakeet build script
structure:

  f16   — FP16 weights and compute
  q8f16 — INT8 dynamic quantization (weights INT8, activations FP16 at runtime)
  q8i8  — INT8 static quantization (weights + activations INT8, calibrated)
  q4f16 — INT4 block-wise weight quantization (activations FP16)
  q4i8  — INT4 block-wise weight quantization (activations INT8)

Moonshine Streaming is a HuggingFace encoder-decoder model (model_type
"moonshine_streaming").  The ONNX export produces two files:

  encoder_model.onnx  — audio waveform → encoder hidden states
  decoder_model.onnx  — decoder input IDs + encoder states → logits

Each quantization step is applied to every component independently.

Requires: transformers (>=5.0.0.dev0), torch (>=2.6), onnx, onnxruntime
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

MODEL_DIR = Path("data/moonshine")
ONNX_DIR = MODEL_DIR / "onnx"
SOURCE_DIR = MODEL_DIR / "source"

CALIB_SAMPLES = 200
# Sequence length for calibration inputs — symbolic dims beyond batch are
# resolved to this value.
CALIB_SEQ_LEN = 100


class SyntheticCalibrationReader(CalibrationDataReader):
    """Generates synthetic inputs for INT8 static quantization calibration.

    Inspects the ONNX model's input spec at init time and generates random data
    with matching shapes and dtypes.  Symbolic dimensions are resolved as:
      - first symbolic dim per input  → batch (size 1)
      - subsequent symbolic dims      → sequence / time (CALIB_SEQ_LEN)

    For production-quality calibration, replace with real audio data processed
    through the Moonshine preprocessor.
    """

    def __init__(
        self,
        onnx_path: str,
        num_samples: int = CALIB_SAMPLES,
        dim_overrides: dict[str, int] | None = None,
    ):
        session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.input_specs = []
        for inp in session.get_inputs():
            shape = self._resolve_shape(inp.shape, dim_overrides or {})
            self.input_specs.append((inp.name, inp.type, shape))
        self.num_samples = num_samples
        self.current = 0

    @staticmethod
    def _resolve_shape(
        raw_shape: list, dim_overrides: dict[str, int]
    ) -> list[int]:
        """First symbolic dim → batch=1, remaining symbolic dims → CALIB_SEQ_LEN.

        dim_overrides maps symbolic dim names (e.g. "dec_len") to explicit
        sizes, allowing model-specific calibration shapes.
        """
        shape: list[int] = []
        first_symbolic_seen = False
        for d in raw_shape:
            if isinstance(d, int) and d > 0:
                shape.append(d)
            elif isinstance(d, str):
                if d in dim_overrides:
                    shape.append(dim_overrides[d])
                elif not first_symbolic_seen:
                    shape.append(1)  # batch
                    first_symbolic_seen = True
                else:
                    shape.append(CALIB_SEQ_LEN)
            else:
                shape.append(1)
        return shape

    def get_next(self):
        if self.current >= self.num_samples:
            return None
        self.current += 1
        feed = {}
        for name, dtype, shape in self.input_specs:
            if "float" in dtype:
                feed[name] = np.random.randn(*shape).astype(np.float32)
            elif "int" in dtype:
                np_dtype = np.int64 if "64" in dtype else np.int32
                if len(shape) == 1:
                    # Length-like scalar per batch element → value = sequence length
                    feed[name] = np.full(shape, CALIB_SEQ_LEN, dtype=np_dtype)
                else:
                    # Token-index tensor → random indices
                    feed[name] = np.random.randint(0, 100, shape, dtype=np_dtype)
        return feed

    def rewind(self):
        self.current = 0


def save_onnx(model_proto, path: Path):
    """Save ONNX model, using external data format for models exceeding 2GB.

    onnx.save() silently produces corrupt files for large models instead of
    raising, so we check the serialized size up front.
    """
    PROTO_LIMIT = 2 * 1024**3
    if model_proto.ByteSize() < PROTO_LIMIT:
        onnx.save(model_proto, str(path))
    else:
        onnx.save_model(
            model_proto,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=path.stem + ".onnx_data",
        )


def consolidate_onnx(directory: Path):
    """Re-save ONNX models in a directory as single files (no external data).

    quantize_dynamic/quantize_static need use_external_data_format=True to
    handle >2GB raw inputs, but the quantized outputs are small enough to fit
    in a single protobuf.  This function loads each .onnx file (with its
    external data) and re-saves it as a self-contained file, then cleans up
    the leftover external-data fragments.
    """
    for onnx_file in sorted(directory.glob("*.onnx")):
        model = onnx.load(str(onnx_file))
        onnx.save(model, str(onnx_file))
        del model
    # Remove orphaned external-data files (everything that isn't .onnx)
    for f in directory.iterdir():
        if f.is_file() and not f.name.endswith(".onnx"):
            f.unlink()


# --- Step 1: Export HuggingFace model to ONNX ---
# Eager attention avoids SymBool tracing issues with SDPA during export.
import torch  # noqa: E402
from transformers import MoonshineStreamingForConditionalGeneration  # noqa: E402

raw_dir = ONNX_DIR / "_raw"
raw_dir.mkdir(parents=True, exist_ok=True)

print("Loading Moonshine Streaming model...")
hf_model = MoonshineStreamingForConditionalGeneration.from_pretrained(
    str(SOURCE_DIR),
    attn_implementation="eager",
)
hf_model.eval()


DEPTH = 14   # decoder_num_hidden_layers
NHEADS = 10  # decoder_num_attention_heads
HEAD_DIM = 64  # decoder_hidden_size // decoder_num_attention_heads


class DecoderWithKVCache(torch.nn.Module):
    """Wraps decoder + proj_out with explicit KV-cache tensors for ONNX."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.get_decoder()
        self.proj_out = model.proj_out
        self.depth = DEPTH

    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask,
                k_self, v_self):
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        sa_cache = DynamicCache()
        for i in range(self.depth):
            sa_cache.update(k_self[i], v_self[i], i)
        ca_cache = DynamicCache()  # empty — cross-attn populates it
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


# -- Encoder export --
encoder = hf_model.get_encoder()
dummy_audio = torch.randn(1, 16000)
dummy_mask = torch.ones(1, 16000, dtype=torch.long)

print("Exporting encoder...")
torch.onnx.export(
    encoder,
    (dummy_audio, dummy_mask),
    str(raw_dir / "encoder_model.onnx"),
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

# -- Decoder export (with KV cache) --
with torch.no_grad():
    enc_out = encoder(input_values=dummy_audio, attention_mask=dummy_mask)
enc_hidden = enc_out.last_hidden_state
enc_attn_mask = torch.ones(1, enc_hidden.shape[1], dtype=torch.long)
dec_ids = torch.tensor([[1]], dtype=torch.long)
# Trace with past_len=1 (zero-len works at runtime but tracing needs ≥1)
k_past = torch.zeros(DEPTH, 1, NHEADS, 1, HEAD_DIM)
v_past = torch.zeros(DEPTH, 1, NHEADS, 1, HEAD_DIM)

dec_wrapper = DecoderWithKVCache(hf_model)
dec_wrapper.eval()

print("Exporting decoder (with KV cache)...")
torch.onnx.export(
    dec_wrapper,
    (dec_ids, enc_hidden, enc_attn_mask, k_past, v_past),
    str(raw_dir / "decoder_model.onnx"),
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

del hf_model, encoder, dec_wrapper  # free memory before quantization

# Consolidate raw exports: merge external .onnx.data into self-contained
# .onnx files so the quantization tools can consume them.
print("Consolidating raw ONNX exports...")
for onnx_file in sorted(raw_dir.glob("*.onnx")):
    proto = onnx.load(str(onnx_file), load_external_data=True)
    save_onnx(proto, onnx_file)
    del proto
# Clean up leftover external-data files
for f in raw_dir.iterdir():
    if f.is_file() and not f.name.endswith(".onnx"):
        f.unlink()

onnx_files = sorted(raw_dir.glob("*.onnx"))
print(f"Exported: {[f.name for f in onnx_files]}")


# --- Step 2: FP16 conversion ---
print("Converting to f16...")
f16_dir = ONNX_DIR / "f16"
f16_dir.mkdir(parents=True, exist_ok=True)


def convert_type_proto_float_to_float16(type_proto):
    """Recursively convert TensorProto.FLOAT → FLOAT16 in an ONNX TypeProto."""
    if type_proto.HasField("tensor_type"):
        if type_proto.tensor_type.elem_type == onnx.TensorProto.FLOAT:
            type_proto.tensor_type.elem_type = onnx.TensorProto.FLOAT16
    elif type_proto.HasField("sequence_type"):
        convert_type_proto_float_to_float16(type_proto.sequence_type.elem_type)
    elif type_proto.HasField("map_type"):
        convert_type_proto_float_to_float16(type_proto.map_type.value_type)


for onnx_file in onnx_files:
    model = onnx.load(str(onnx_file), load_external_data=True)
    # Full FP16 conversion in-place (onnxconverter_common silently corrupts
    # models >2GB due to protobuf serialization limits).
    # 1) Convert initializer weights
    for tensor in model.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT:
            fp32_data = np.array(onnx.numpy_helper.to_array(tensor), dtype=np.float16)
            tensor.CopyFrom(onnx.numpy_helper.from_array(fp32_data, tensor.name))
    # 2) Convert graph input/output/value_info type annotations
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        convert_type_proto_float_to_float16(vi.type)
    # 3) Convert float attributes in nodes (e.g. Constant value tensors)
    #    and redirect Cast-to-float32 nodes to cast to float16 instead.
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.t and attr.t.data_type == onnx.TensorProto.FLOAT:
                data = np.array(onnx.numpy_helper.to_array(attr.t), dtype=np.float16)
                attr.t.CopyFrom(onnx.numpy_helper.from_array(data, attr.t.name))
            if node.op_type == "Cast" and attr.name == "to" and attr.i == onnx.TensorProto.FLOAT:
                attr.i = onnx.TensorProto.FLOAT16
    save_onnx(model, f16_dir / onnx_file.name)
    del model


# --- Step 3: INT8 dynamic quantization (q8f16) ---
# Weights quantized to INT8, activations quantized to INT8 on-the-fly per batch.
# No calibration data needed.
print("Quantizing q8f16 (INT8 dynamic)...")
q8f16_dir = ONNX_DIR / "q8f16"
q8f16_dir.mkdir(parents=True, exist_ok=True)

for onnx_file in onnx_files:
    quantize_dynamic(
        model_input=str(onnx_file),
        model_output=str(q8f16_dir / onnx_file.name),
        per_channel=True,
        weight_type=QuantType.QInt8,
        use_external_data_format=True,
    )

consolidate_onnx(q8f16_dir)

# --- Step 4: INT8 static quantization (q8i8) ---
# Both weights and activations quantized to INT8 using calibrated ranges.
# Uses synthetic inputs for calibration.
print(f"Calibrating and quantizing q8i8 (INT8 static, {CALIB_SAMPLES} samples)...")
q8i8_dir = ONNX_DIR / "q8i8"
q8i8_dir.mkdir(parents=True, exist_ok=True)

# Model-specific dim overrides for calibration:
# - Encoder: audio_len must be a multiple of 80 (pad_to_multiple_of in
#   preprocessor config), so use 16000 (1 second @ 16kHz).
# - Decoder: traced with dec_len=1 (single-token autoregressive decoding),
#   CumSum / position logic doesn't generalise to longer sequences.
ENCODER_DIM_OVERRIDES: dict[str, int] = {"audio_len": 16000}
DECODER_DIM_OVERRIDES: dict[str, int] = {"dec_len": 1, "past_len": 5}


def _overrides_for(name: str) -> dict[str, int] | None:
    if "encoder" in name:
        return ENCODER_DIM_OVERRIDES
    if "decoder" in name:
        return DECODER_DIM_OVERRIDES
    return None


for onnx_file in onnx_files:
    calib_reader = SyntheticCalibrationReader(
        str(onnx_file), dim_overrides=_overrides_for(onnx_file.name)
    )
    quantize_static(
        model_input=str(onnx_file),
        model_output=str(q8i8_dir / onnx_file.name),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        use_external_data_format=True,
    )

consolidate_onnx(q8i8_dir)


# --- Step 5: INT4 block-wise weight quantization (q4f16) ---
# ORT's standard quantizer doesn't support 4-bit packing; MatMulNBitsQuantizer
# does block-wise 4-bit weight quantization natively.
print("Quantizing q4f16 (INT4 weights)...")
q4f16_dir = ONNX_DIR / "q4f16"
q4f16_dir.mkdir(parents=True, exist_ok=True)

for onnx_file in onnx_files:
    q4 = MatMulNBitsQuantizer(
        str(onnx_file), bits=4, block_size=128, is_symmetric=True
    )
    q4.process()
    q4.model.save_model_to_file(str(q4f16_dir / onnx_file.name))
    del q4


# --- Step 6: INT4 weights + INT8 activation quantization (q4i8) ---
print("Quantizing q4i8 (INT4 weights + INT8 activations)...")
q4i8_dir = ONNX_DIR / "q4i8"
q4i8_dir.mkdir(parents=True, exist_ok=True)

for onnx_file in onnx_files:
    q4i8 = MatMulNBitsQuantizer(
        str(onnx_file), bits=4, block_size=128, is_symmetric=True, accuracy_level=4
    )
    q4i8.process()
    q4i8.model.save_model_to_file(str(q4i8_dir / onnx_file.name))
    del q4i8


print("ONNX exports ready.")
