"""Build ONNX variants for Parakeet TDT 0.6B STT model.

Exports the NeMo model to ONNX, then produces five quantized variants
matching the Llama3 build script structure:

  f16   — FP16 weights and compute
  q8f16 — INT8 dynamic quantization (weights INT8, activations FP16 at runtime)
  q8i8  — INT8 static quantization (weights + activations INT8, calibrated)
  q4f16 — INT4 block-wise weight quantization (activations FP16)
  q4i8  — INT4 block-wise weight quantization (activations INT8)

NeMo RNNT/TDT models export as multiple ONNX files (encoder + decoder_joint),
so each quantization step is applied to every component independently.

Requires: nemo_toolkit[asr], onnxruntime, onnxconverter-common
"""

import sys
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

MODEL_DIR = Path("data/parakeet")
ONNX_DIR = MODEL_DIR / "onnx"
NEMO_FILE = MODEL_DIR / "source" / "parakeet-tdt-0.6b-v3.nemo"

CALIB_SAMPLES = 200
# ~1 second of audio at 10ms stride — short enough for decoder LSTM comfort,
# long enough for representative encoder calibration ranges.
CALIB_SEQ_LEN = 100


class SyntheticCalibrationReader(CalibrationDataReader):
    """Generates synthetic inputs for INT8 static quantization calibration.

    Inspects the ONNX model's input spec at init time and generates random data
    with matching shapes and dtypes.  Symbolic dimensions are resolved as:
      - first symbolic dim per input  → batch (size 1)
      - subsequent symbolic dims      → sequence / time (CALIB_SEQ_LEN)

    For production-quality calibration, replace with real audio data processed
    through NeMo's AudioToMelSpectrogramPreprocessor.
    """

    def __init__(self, onnx_path: str, num_samples: int = CALIB_SAMPLES):
        session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.input_specs = []
        for inp in session.get_inputs():
            shape = self._resolve_shape(inp.shape)
            self.input_specs.append((inp.name, inp.type, shape))
        self.num_samples = num_samples
        self.current = 0

    @staticmethod
    def _resolve_shape(raw_shape: list) -> list[int]:
        """First symbolic dim → batch=1, remaining symbolic dims → CALIB_SEQ_LEN."""
        shape: list[int] = []
        first_symbolic_seen = False
        for d in raw_shape:
            if isinstance(d, int) and d > 0:
                shape.append(d)
            elif isinstance(d, str):
                if not first_symbolic_seen:
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




# --- Step 1: Load NeMo model and export base ONNX ---
# Workaround: numba 0.63 references coverage.types.Tracer and several other
# type aliases that were renamed/removed in coverage 7.4+.  Hide the system
# coverage from numba so its coverage_support falls back to coverage_available=False.
_cov_mods = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "coverage" or k.startswith("coverage.")}
sys.modules["coverage"] = None  # type: ignore[assignment] — sentinel blocks import

import nemo.collections.asr as nemo_asr  # noqa: E402 — heavy import, after lightweight deps

# Restore coverage for anything that needs it later
sys.modules.pop("coverage", None)
sys.modules.update(_cov_mods)

print("Loading NeMo model...")
model = nemo_asr.models.ASRModel.restore_from(str(NEMO_FILE))

raw_dir = ONNX_DIR / "_raw"
raw_dir.mkdir(parents=True, exist_ok=True)

print("Exporting base ONNX model...")
model.export(str(raw_dir / "model.onnx"), onnx_opset_version=17)

# NeMo RNNT/TDT models produce multiple files: encoder + decoder_joint
onnx_files = sorted(raw_dir.glob("*.onnx"))
print(f"Exported: {[f.name for f in onnx_files]}")
del model  # free GPU/CPU memory before quantization

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
# Uses synthetic mel-spectrogram-like inputs for calibration.
print(f"Calibrating and quantizing q8i8 (INT8 static, {CALIB_SAMPLES} samples)...")
q8i8_dir = ONNX_DIR / "q8i8"
q8i8_dir.mkdir(parents=True, exist_ok=True)

for onnx_file in onnx_files:
    calib_reader = SyntheticCalibrationReader(str(onnx_file))
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
