// Flat C API wrapping TensorRT-LLM's Executor (tensorrt_llm::executor::Executor).
// This header is consumed by both the C++ implementation and Rust FFI (via cc crate).

#ifndef TRTLLM_EXECUTOR_H
#define TRTLLM_EXECUTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TRTLLM_OK = 0,
    TRTLLM_ERROR = 1,
} TrtLlmStatus;

typedef enum {
    TRTLLM_DECODER_ONLY = 0,
    TRTLLM_ENCODER_ONLY = 1,
    TRTLLM_ENCODER_DECODER = 2,
} TrtLlmModelType;

// Opaque handle to the underlying tle::Executor.
typedef struct TrtLlmExecutor TrtLlmExecutor;

// Returns a thread-local error string set by the last failing call.
// The pointer is valid until the next call from the same thread.
const char* trtllm_get_last_error(void);

// Create an executor from an engine directory.
//   engine_dir:          path to the TRT-LLM engine folder
//   model_type:          decoder-only / encoder-only / encoder-decoder
//   kv_cache_fraction:   fraction of free GPU memory for KV cache (0.0 = auto)
//   max_beam_width:      maximum beam width (1 = greedy)
//   out_handle:          receives the executor handle on success
TrtLlmStatus trtllm_executor_create(
    const char* engine_dir,
    TrtLlmModelType model_type,
    float kv_cache_fraction,
    int32_t max_beam_width,
    TrtLlmExecutor** out_handle);

// Enqueue an inference request.
//   handle:       executor handle
//   tokens:       input token ids
//   n_tokens:     number of input tokens
//   max_new:      maximum new tokens to generate
//   temperature:  sampling temperature (0.0 = use default)
//   top_k:        top-k sampling (0 = use default)
//   top_p:        top-p / nucleus sampling (0.0 = use default)
//   rep_penalty:  repetition penalty (0.0 = use default)
//   end_id:       end-of-sequence token id (-1 = none)
//   pad_id:       padding token id (-1 = none)
//   streaming:    whether to stream partial results
//   out_req_id:   receives the request id on success
TrtLlmStatus trtllm_executor_enqueue(
    TrtLlmExecutor* handle,
    const int32_t* tokens,
    size_t n_tokens,
    int32_t max_new,
    float temperature,
    int32_t top_k,
    float top_p,
    float rep_penalty,
    int32_t end_id,
    int32_t pad_id,
    int streaming,
    uint64_t* out_req_id);

// Await a response for a specific request.
//   handle:       executor handle
//   req_id:       request id returned by enqueue
//   out_tokens:   caller-allocated buffer for output token ids
//   capacity:     size of out_tokens buffer
//   out_n_tokens: receives the number of tokens written
//   out_is_final: receives 1 if this is the final response, 0 otherwise
//   timeout_ms:   maximum time to wait in milliseconds (0 = wait forever)
TrtLlmStatus trtllm_executor_await(
    TrtLlmExecutor* handle,
    uint64_t req_id,
    int32_t* out_tokens,
    size_t capacity,
    size_t* out_n_tokens,
    int* out_is_final,
    int64_t timeout_ms);

// Cancel a pending request.
TrtLlmStatus trtllm_executor_cancel(TrtLlmExecutor* handle, uint64_t req_id);

// Gracefully shut down the executor (blocks until all requests finish).
TrtLlmStatus trtllm_executor_shutdown(TrtLlmExecutor* handle);

// Destroy the executor and free all resources.
void trtllm_executor_destroy(TrtLlmExecutor* handle);

#ifdef __cplusplus
}
#endif

#endif // TRTLLM_EXECUTOR_H
