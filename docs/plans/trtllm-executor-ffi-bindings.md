# Plan: TensorRT-LLM Executor C/Rust FFI Bindings

**Context:** The `tensorrt/` crate is currently empty. The goal is to wrap TensorRT-LLM's C++ Executor API with a thin C stub and Rust FFI, so `experiments/llama3/` can later use it for LLM inference benchmarking against ONNX Runtime. TensorRT-LLM is not installed on this system.

**Type:** Feature
**Status:** COMPLETE

---

## Prerequisites (User Action Required)

TensorRT-LLM is **not installed**. The dev headers for TensorRT are also missing. Install:

```bash
# 1. TensorRT dev headers
sudo apt install libnvinfer-dev

# 2. TensorRT-LLM (build from source for C++ headers + library)
git clone https://github.com/NVIDIA/TensorRT-LLM.git ~/TensorRT-LLM
cd ~/TensorRT-LLM
git submodule update --init --recursive
python3 scripts/build_wheel.py --trt_root /usr
# After build, C++ headers are at ~/TensorRT-LLM/cpp/include/
# and libtensorrt_llm.so is in the build output

# 3. Set env var for the Rust build
export TRTLLM_ROOT=~/TensorRT-LLM
```

**Note on model conversion (later, not this task):** The existing `data/llama3/model.onnx` cannot be used directly by TRT-LLM. TRT-LLM requires HuggingFace checkpoint → `convert_checkpoint.py` → `trtllm-build` pipeline. This is a separate step for the inference experiment.

---

## File Structure

```
tensorrt/
  Cargo.toml                          # add cc build-dep, trtllm feature
  build.rs                            # compile C++ stub, link libs
  src/
    lib.rs                            # safe Rust wrapper (Executor, SamplingParams)
    ffi/
      mod.rs                          # re-export
      ffi.rs                          # extern "C" declarations
      trtllm_executor.h              # flat C header
      trtllm_executor.cpp            # C++ impl wrapping tle::Executor
```

---

## Tasks

### Task 1: C header — `tensorrt/src/ffi/trtllm_executor.h`
- [x] Done: 1 | Left: 0

Flat C API with opaque `TrtLlmExecutor*` handle:
- `trtllm_executor_create(engine_dir, model_type, kv_cache_fraction, max_beam_width, &handle)`
- `trtllm_executor_enqueue(handle, tokens, n, max_new, temp, top_k, top_p, rep_pen, end_id, pad_id, streaming, &req_id)`
- `trtllm_executor_await(handle, req_id, out_buf, capacity, &n_tokens, &is_final, timeout_ms)`
- `trtllm_executor_cancel(handle, req_id)`
- `trtllm_executor_shutdown(handle)`
- `trtllm_executor_destroy(handle)`
- `trtllm_get_last_error()` — thread-local error string

Status enum: `TRTLLM_OK = 0`, `TRTLLM_ERROR = 1`.

### Task 2: C++ implementation — `tensorrt/src/ffi/trtllm_executor.cpp`
- [x] Done: 1 | Left: 0

Wraps `tensorrt_llm::executor::Executor`. Each function:
- Catches all C++ exceptions → sets thread-local error string → returns `TRTLLM_ERROR`
- `create`: builds `KvCacheConfig` + `ExecutorConfig`, constructs `Executor`
- `enqueue`: builds `VecTokens`, `SamplingConfig`, `OutputConfig` (excludeInputFromOutput=true), `Request`
- `await`: calls `awaitResponses(requestId, timeout)`, copies beam-0 tokens to caller buffer
- `destroy`: `delete handle` (triggers `~Executor` which shuts down internally)

### Task 3: Rust FFI — `tensorrt/src/ffi/ffi.rs` + `mod.rs`
- [x] Done: 1 | Left: 0

- `#[repr(C)] enum TrtLlmStatus { Ok = 0, Error = 1 }`
- `#[repr(C)] struct TrtLlmExecutor { _private: [u8; 0] }` (opaque)
- `unsafe extern "C"` block with all 7 function declarations
- `mod.rs`: `mod ffi; pub use ffi::*;`

### Task 4: Safe Rust wrapper — `tensorrt/src/lib.rs`
- [x] Done: 1 | Left: 0

- `pub struct Executor` — owns `*mut ffi::TrtLlmExecutor`, `impl Drop`
- `unsafe impl Send + Sync` (TRT-LLM Executor is internally thread-safe)
- `pub enum ModelType { DecoderOnly, EncoderOnly, EncoderDecoder }`
- `pub struct SamplingParams { temperature, top_k, top_p, repetition_penalty }` with `Default`
- `pub struct ExecutorConfig { model_type, kv_cache_free_gpu_mem_fraction, max_beam_width }` with `Default`
- Methods: `new()`, `enqueue()`, `await_response() -> (Vec<i32>, bool)`, `cancel()`, `shutdown()`
- Error handling: panic with `trtllm_get_last_error()` message (matches onnx crate pattern)

### Task 5: Build system — `Cargo.toml` + `build.rs`
- [x] Done: 1 | Left: 0

**Cargo.toml:** add `cc = "1"` to `[build-dependencies]`, add `[features] trtllm = []`

**build.rs:**
- Feature-gated: only compiles/links when `--features trtllm`
- Reads `TRTLLM_ROOT` env var for include path (`$TRTLLM_ROOT/cpp/include`)
- Uses `cc` crate to compile `trtllm_executor.cpp` with C++17
- Links: `libtensorrt_llm` (from `$TRTLLM_ROOT/build/tensorrt_llm`), `libnvinfer`, `libcudart`, `libstdc++`
- Adds both x86_64 and aarch64 lib search paths (only matching arch has files)

Without `--features trtllm`, the crate compiles as an empty shell so the workspace builds on systems without TRT-LLM.

---

## Verification

1. `cargo build` (without feature) — workspace compiles, tensorrt crate is empty shell
2. With TRT-LLM installed: `TRTLLM_ROOT=~/TensorRT-LLM cargo build --features trtllm` — C++ compiles, links succeed
3. Inspect symbols: `nm -D target/debug/libtensorrt.rlib | grep trtllm` shows the extern functions
