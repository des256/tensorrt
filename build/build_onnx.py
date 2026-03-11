from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import (
    AutoCalibrationConfig,
    AutoQuantizationConfig,
)
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

MODEL_DIR = "data/llama3-3b"
ONNX_DIR = Path(MODEL_DIR) / "onnx"
RAW_MODEL = str(ONNX_DIR / "_raw" / "model.onnx")
TASK = "text-generation-with-past"

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128

# --- Base ONNX exports ---
print("Exporting ONNX f16 model...")
main_export(
    model_name_or_path=MODEL_DIR + "/source",
    output=MODEL_DIR + "/onnx/f16",
    task=TASK,
    optimize="O3",
    dtype="fp16",
)

# Un-optimized export for quantization — O3 fuses operators into custom domains
# that the quantizer can't process. ORT applies optimizations at inference time anyway.
print("Exporting un-optimized ONNX model for quantization base...")
main_export(
    model_name_or_path=MODEL_DIR + "/source",
    output=str(ONNX_DIR / "_raw"),
    task=TASK,
)

# --- INT8 quantization via optimum ORTQuantizer ---
quantizer = ORTQuantizer.from_pretrained(str(ONNX_DIR / "_raw"))

# INT8 dynamic: weights INT8, activations quantized to INT8 on-the-fly per batch
qconfig_q8_dynamic = AutoQuantizationConfig.avx512_vnni(
    is_static=False, per_channel=True
)

print("Quantizing ONNX q8f16 model...")
quantizer.quantize(
    save_dir=MODEL_DIR + "/onnx/q8f16",
    quantization_config=qconfig_q8_dynamic,
    use_external_data_format=True,
)

# --- INT8 static quantization (needs calibration) ---
print("Collecting calibration ranges (200 samples from WikiText-2)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR + "/source")
tokenizer.pad_token = tokenizer.eos_token


def preprocess(examples):
    tokens = tokenizer(
        examples["text"], padding="max_length", max_length=128, truncation=True
    )
    batch_size = len(tokens["input_ids"])
    seq_len = len(tokens["input_ids"][0])
    tokens["position_ids"] = [list(range(seq_len))] * batch_size
    # Provide dummy KV cache inputs (past_seq_len=1 with zeros) for calibration.
    # HF datasets can't store zero-length arrays, so we use 1 dummy past token
    # and extend the attention mask to cover it.
    tokens["attention_mask"] = [[1] + row for row in tokens["attention_mask"]]
    # Shape without batch dim — the data reader adds it via [value] wrapping
    kv_zeros = np.zeros((NUM_KV_HEADS, 1, HEAD_DIM), dtype=np.float32)
    for i in range(NUM_LAYERS):
        tokens[f"past_key_values.{i}.key"] = [kv_zeros] * batch_size
        tokens[f"past_key_values.{i}.value"] = [kv_zeros] * batch_size
    return tokens


calibration_dataset = quantizer.get_calibration_dataset(
    "wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    dataset_split="train",
    num_samples=200,
    preprocess_function=preprocess,
    preprocess_batch=True,
)
calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)

qconfig_q8_static = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=True)
calibration_ranges = quantizer.fit(
    dataset=calibration_dataset,
    calibration_config=calibration_config,
    use_gpu=False,
    use_external_data_format=True,
)

print("Quantizing ONNX q8i8 model...")
quantizer.quantize(
    save_dir=MODEL_DIR + "/onnx/q8i8",
    quantization_config=qconfig_q8_static,
    calibration_tensors_range=calibration_ranges,
    use_external_data_format=True,
)

# --- INT4 quantization via ORT MatMulNBitsQuantizer ---
# ORT's standard quantizer doesn't support 4-bit packing; MatMulNBitsQuantizer
# does block-wise 4-bit weight quantization natively.

print("Quantizing ONNX q4f16 model...")
q4_model = MatMulNBitsQuantizer(RAW_MODEL, bits=4, block_size=128, is_symmetric=True)
q4_model.process()
q4f16_dir = ONNX_DIR / "q4f16"
q4f16_dir.mkdir(parents=True, exist_ok=True)
q4_model.model.save_model_to_file(
    str(q4f16_dir / "model.onnx"), use_external_data_format=True
)
# Copy tokenizer/config from raw export
for f in (ONNX_DIR / "_raw").iterdir():
    if not f.name.startswith("model.onnx") and not f.is_dir():
        (q4f16_dir / f.name).write_bytes(f.read_bytes())

print("Quantizing ONNX q4i8 model...")
q4i8_model = MatMulNBitsQuantizer(
    RAW_MODEL, bits=4, block_size=128, is_symmetric=True, accuracy_level=4
)
q4i8_model.process()
q4i8_dir = ONNX_DIR / "q4i8"
q4i8_dir.mkdir(parents=True, exist_ok=True)
q4i8_model.model.save_model_to_file(
    str(q4i8_dir / "model.onnx"), use_external_data_format=True
)
for f in (ONNX_DIR / "_raw").iterdir():
    if not f.name.startswith("model.onnx") and not f.is_dir():
        (q4i8_dir / f.name).write_bytes(f.read_bytes())

print("ONNX exports ready.")
