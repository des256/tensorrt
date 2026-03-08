# Model Pipeline Plan

Generic pipeline for downloading, converting, building, and benchmarking models across platforms (desktop x86_64, Jetson Orin NX aarch64).

## Architecture Overview

Every model goes through three stages to produce runnable engines. Each model is then benchmarked across **three runtime paths** to compare performance.

### Three Runtime Paths (Benchmarking)

| Path               | Runtime                        | Rust Crate | Feature Gate        | Models                          |
| ------------------ | ------------------------------ | ---------- | ------------------- | ------------------------------- |
| **TensorRT-LLM**   | TRT-LLM Executor C++ API       | `tensorrt` | `--features trtllm` | LLMs only                       |
| **Plain TensorRT** | TensorRT C/C++ API (`nvinfer`) | `tensorrt` | (always available)  | STT, TTS, classification        |
| **ONNX Runtime**   | ONNX Runtime C API             | `onnx`     | (always available)  | All models (benchmark baseline) |

Both TensorRT paths live in the single `tensorrt` crate. Plain TensorRT is always available (it only needs `libnvinfer`). The TRT-LLM Executor is feature-gated behind `--features trtllm` because it pulls in the heavier TRT-LLM dependencies.

The ONNX Runtime path stays permanently as the **benchmark baseline** — the whole point of the project is measuring how much TensorRT and TensorRT-LLM improve over it. The three paths run side-by-side in each experiment binary.

### Engine vs Runtime Matrix

| Model Type               | Optimized Engine                                            | Baseline                              |
| ------------------------ | ----------------------------------------------------------- | ------------------------------------- |
| LLM (autoregressive)     | TensorRT-LLM engine via `tensorrt` crate (`trtllm` feature) | ONNX model via `onnx` crate (CUDA EP) |
| STT, TTS, classification | Plain TensorRT engine via `tensorrt` crate                  | ONNX model via `onnx` crate (CUDA EP) |

## Pipeline Stages

### Stage 0: Download Source Model

Acquire the original model weights in their native format.

- Input: Model identifier (HuggingFace ID, NGC model, URL)
- Output: `data/<model>/source/`
- Tool: `huggingface-cli download`, NGC CLI, or custom script
- One-time, platform-independent

Source formats vary by model:

| Model         | Source Format                      | Origin         |
| ------------- | ---------------------------------- | -------------- |
| Llama 3.2     | SafeTensors (sharded)              | HuggingFace    |
| Parakeet      | NeMo checkpoint or ONNX            | NVIDIA NGC     |
| Pocket        | SafeTensors / ONNX                 | Model-specific |
| Future models | SafeTensors, PyTorch `.pt`, `.bin` | Varies         |

Where possible, acquire the real source (SafeTensors, NeMo) rather than a pre-exported ONNX. This gives full control over export settings and quantization.

For models currently only available as ONNX (Parakeet, Pocket), use those as-is until the real source is traced.

### Stage 1: Convert to Engine-Ready Formats (In Docker)

Produce portable intermediate artifacts for each runtime path. Runs on the desktop inside Docker.

**LLMs need two outputs:**

1. **TRT-LLM checkpoint** (for the TensorRT-LLM path):

   ```
   Source (SafeTensors) → TRT-LLM Python API → trtllm_ckpt/<quant>/
                                                 ├── rank0.safetensors
                                                 └── config.json
   ```

   - Quantization variant selected here: FP16, INT4 (W4A16), INT8 (W8A16)
   - Multiple variants can be produced from a single source
   - Tool: TRT-LLM Python API (e.g., `LLaMAForCausalLM.from_hugging_face(quant_config=...)`)

2. **ONNX model** (for the ONNX Runtime baseline):
   ```
   Source (SafeTensors) → optimum-cli / export script → onnx/
                                                         ├── model.onnx
                                                         ├── model.onnx.data
                                                         └── tokenizer.json
   ```

   - Tool: `optimum-cli export onnx` or custom export script

**Non-LLMs need two outputs:**

1. **ONNX as interchange** (for building plain TensorRT engines):

   ```
   Source (SafeTensors/NeMo) → export script → onnx/model.onnx
   ```

   - ONNX is used only as an interchange format for TensorRT's ONNX parser
   - Not an ONNX Runtime dependency — just how you feed a model graph to TensorRT's builder

2. **Same ONNX file** also serves as input for the ONNX Runtime baseline path. One ONNX export covers both uses.

### Stage 2: Build Engines (In Docker, On Target Platform)

Produce platform-locked engine files. Runs inside Docker on each target platform (desktop and Jetson).

**TensorRT-LLM engines (LLMs):**

```
trtllm_ckpt/<quant>/ → trtllm-build → engine/<platform>/
                                        ├── rank0.engine
                                        └── config.json
```

- Select quantization variant based on target memory budget
- Configure max_batch_size, max_input_len, max_seq_len for target

**Plain TensorRT engines (non-LLMs):**

```
onnx/model.onnx → trtexec → engine/<platform>/model.engine
```

- `trtexec --onnx=model.onnx --saveEngine=model.engine --fp16`
- Or `--int8 --calib=<calibration_data>` for INT8
- Can also use TensorRT Python API for dynamic shapes, multiple optimization profiles

**ONNX Runtime path needs no engine build** — it loads the `.onnx` file directly at runtime.

**Memory-aware variant selection:**

| Target             | Available GPU Memory | 3B LLM | 1B STT | 200M TTS |
| ------------------ | -------------------- | ------ | ------ | -------- |
| Desktop (RTX 4070) | 12 GB VRAM           | FP16   | FP16   | FP16     |
| Jetson Orin NX     | 16 GB shared         | INT4   | FP16   | FP16     |

The Jetson's constraint is primarily during engine building (TensorRT builder workspace), not inference. A 3B INT4 model uses ~3 GB at inference but needed ~6 GB builder workspace. Smaller models (Parakeet ~4.6 GB, Pocket ~190 MB) build fine in FP16 on the Jetson.

### Stage 3: Benchmark (Native Rust)

Each experiment binary runs all applicable runtime paths and reports metrics.

**LLM experiments run two paths:**

1. TensorRT-LLM engine via `tensorrt` crate → measure TTFT
2. ONNX model via `onnx` crate (CUDA EP) → measure TTFT

**Non-LLM experiments run two paths:**

1. Plain TensorRT engine via `tensorrt` crate → measure latency
2. ONNX model via `onnx` crate (CUDA EP) → measure latency

No Docker, no Python at runtime. Just Rust loading engine files and ONNX models.

## Docker Strategy

Both checkpoint conversion (Stage 1) and engine building (Stage 2) happen inside Docker. Only the final Rust benchmark binary runs natively on the host.

### Standardize TRT-LLM Version

A single TRT-LLM version is used across both platforms. This eliminates the config schema mismatches, rope type patching, tied embedding workarounds, and library hacks we hit with the Jetson's v0.11.0 vs the desktop's v1.3.0rc7.

### Dockerfiles

Two Dockerfiles, one per platform:

**`docker/Dockerfile.desktop`** (x86_64):

- Base: `tensorrt_llm/release:latest` or NVIDIA PyTorch image
- Includes: TRT-LLM, TensorRT (`trtexec`), Python, HuggingFace tools
- Mounts: `data/` for model I/O, HF cache

**`docker/Dockerfile.jetson`** (aarch64):

- Base: JetPack-compatible L4T image
- Includes: same TRT-LLM version as desktop, TensorRT from JetPack
- Must match: CUDA version, TensorRT version from JetPack R36.x

### What Runs Where

| Operation                             | Where            | Container                          |
| ------------------------------------- | ---------------- | ---------------------------------- |
| Download source model                 | Desktop          | Optional (can run on host)         |
| Convert checkpoint / export ONNX      | Desktop          | `docker/Dockerfile.desktop`        |
| Build desktop engines                 | Desktop          | `docker/Dockerfile.desktop`        |
| Build Jetson engines                  | Jetson           | `docker/Dockerfile.jetson`         |
| Transfer checkpoints + ONNX to Jetson | Desktop → Jetson | `rsync` / `push.sh`                |
| Run benchmarks                        | Both (native)    | No container — Rust binary on host |

## Directory Convention

```
data/<model>/
  source/                    # Original download (gitignored)
  onnx/                      # ONNX model (used as TensorRT build input AND ONNX Runtime baseline)
    model.onnx
    model.onnx.data          # (if weights are external)
    tokenizer.*
  trtllm_ckpt/               # TRT-LLM checkpoints, LLMs only (gitignored)
    fp16/
      rank0.safetensors
      config.json
    int4/
      rank0.safetensors
      config.json
  engine/                    # Built engines (gitignored, platform-specific)
    desktop/
      rank0.engine           # TRT-LLM engine (LLMs) or model.engine (non-LLMs)
      config.json
    jetson/
      rank0.engine
      config.json
```

Tokenizers live alongside their model's primary format. Engine directories mirror the structure but are platform-specific and never transferred.

## Rust Crates

### `tensorrt` — Unified TensorRT Crate

A single crate exposing both plain TensorRT and TRT-LLM functionality. Two layers, one crate.

```
tensorrt/
  src/
    lib.rs                        # Re-exports both modules
    runtime.rs                    # Plain TensorRT runtime (always compiled)
    executor.rs                   # TRT-LLM Executor (existing code, behind #[cfg(feature = "trtllm")])
    ffi/
      trt_runtime.cpp             # C++ stub wrapping nvinfer runtime API (new)
      trt_runtime.h
      trtllm_executor.cpp         # C++ stub wrapping TRT-LLM Executor API (existing)
      trtllm_executor.h
  build.rs                        # Always links libnvinfer + libcudart;
                                  # conditionally links TRT-LLM libs when trtllm feature is on
  Cargo.toml                      # features = { trtllm = [] }
```

**Plain TensorRT API** (always available, links only `libnvinfer` + `libcudart`):

- `tensorrt::Runtime::new()` → TensorRT runtime (`nvinfer::createInferRuntime`)
- `tensorrt::Engine::load(&runtime, path)` → deserialize `.engine` file
- `tensorrt::Context::new(&engine)` → execution context
- `context.set_input(name, &tensor)` → bind input tensor
- `context.execute()` → run inference
- `context.get_output(name)` → read output tensor

Used by: STT (Parakeet), TTS (Pocket), classification, any non-autoregressive model.

**TRT-LLM Executor API** (behind `--features trtllm`, additionally links `libtensorrt_llm` + `libnvinfer_plugin_tensorrt_llm`):

- `tensorrt::Executor::new(engine_dir, &config)` → TRT-LLM executor (existing)
- `executor.enqueue(tokens, max_new_tokens, sampling, ...)` → request_id
- `executor.await_response(req_id, timeout_ms)` → (output_tokens, is_final)
- `executor.shutdown()`

Used by: LLMs (Llama 3.2, future autoregressive models).

**Build script (`build.rs`) logic:**

```
always:
  link libnvinfer, libcudart

if cfg(feature = "trtllm"):
  require TRTLLM_ROOT env var
  compile trtllm_executor.cpp
  link libtensorrt_llm, libnvinfer_plugin_tensorrt_llm

always:
  compile trt_runtime.cpp
```

This means experiments that only use plain TensorRT (e.g., Parakeet) can build without `TRTLLM_ROOT` and without TRT-LLM installed. LLM experiments opt in with `--features trtllm`.

### `onnx` — ONNX Runtime FFI (Benchmark Baseline)

Wraps ONNX Runtime C API. Supports CPU, CUDA, and TensorRT execution providers.

- Links: `libonnxruntime`
- Comprehensive: sessions, values, metadata, type introspection

Stays as the permanent benchmark baseline. Every experiment uses this crate alongside `tensorrt` to produce comparison numbers. No changes needed.

## Scripts and Automation

```
scripts/
  download.sh           # Stage 0: download source model
  convert.sh            # Stage 1: convert to checkpoint/ONNX (runs in Docker)
  build_engine.sh       # Stage 2: build engine on current platform (runs in Docker)
docker/
  Dockerfile.desktop    # x86_64 build environment
  Dockerfile.jetson     # aarch64 build environment
```

Top-level orchestration (Makefile or shell wrapper):

```
make download      MODEL=llama3
make convert       MODEL=llama3 QUANT=int4
make engine        MODEL=llama3 PLATFORM=jetson
make bench         MODEL=llama3 PLATFORM=jetson
```

For adding a new model, the workflow is:

1. Add download config (model ID, source type)
2. Add conversion config (model type, quantization options, export method)
3. Run `make convert && make engine && make bench`
4. Write experiment binary under `experiments/<model>/`

## Work Items

| Item                                                                              | Dependency                     |
| --------------------------------------------------------------------------------- | ------------------------------ |
| Add plain TensorRT runtime to `tensorrt` crate (`trt_runtime.cpp` + `runtime.rs`) | None                           |
| Restructure `tensorrt` crate: move existing Executor code behind `trtllm` feature | None                           |
| Update `build.rs`: always link `libnvinfer`; conditionally link TRT-LLM           | None                           |
| Create `docker/Dockerfile.desktop`                                                | None                           |
| Create `docker/Dockerfile.jetson` (same TRT-LLM version)                          | JetPack compatibility research |
| Write `scripts/download.sh`                                                       | None                           |
| Write `scripts/convert.sh`                                                        | Dockerfiles                    |
| Write `scripts/build_engine.sh`                                                   | Dockerfiles                    |

## Lessons Learned (From Llama 3.2 Exercise)

These inform the design decisions above:

1. **Engines are platform-locked.** GPU architecture + TensorRT version must match exactly. Never transfer engines between platforms.
2. **Checkpoints and ONNX models are portable.** Always transfer these, build engines locally.
3. **TRT-LLM version mismatch is the #1 pain point.** Standardize to a single version across platforms.
4. **Memory constraints force quantization choices at build time.** The Jetson can't build FP16 engines for 3B+ models — the builder workspace alone exceeds available memory.
5. **Docker isolates the messy Python/C++ build environment.** Rust only ever sees clean `.engine` files and `.onnx` files.
6. **All manual patching (config stripping, rope type mapping, tied embeddings) was caused by version mismatch.** Fix the version, eliminate the patches.
