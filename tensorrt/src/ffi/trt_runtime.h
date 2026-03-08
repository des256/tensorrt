// Flat C API wrapping TensorRT's nvinfer runtime (IRuntime, ICudaEngine, IExecutionContext).
// This header is consumed by both the C++ implementation and Rust FFI (via cc crate).

#ifndef TRT_RUNTIME_H
#define TRT_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TRT_OK = 0,
    TRT_ERROR = 1,
} TrtStatus;

// Opaque handles.
typedef struct TrtRuntime TrtRuntime;
typedef struct TrtEngine TrtEngine;
typedef struct TrtContext TrtContext;

// Returns a thread-local error string set by the last failing call.
const char* trt_get_last_error(void);

// --- Runtime ---

// Create a TensorRT runtime (wraps nvinfer::createInferRuntime).
TrtStatus trt_runtime_create(TrtRuntime** out);

// Destroy the runtime.
void trt_runtime_destroy(TrtRuntime* rt);

// --- Engine ---

// Deserialize an engine from a file (reads the .engine / .plan file).
TrtStatus trt_engine_load(TrtRuntime* rt, const char* path, TrtEngine** out);

// Number of I/O tensors (inputs + outputs).
int32_t trt_engine_get_num_io_tensors(TrtEngine* engine);

// Name of the i-th I/O tensor (0-indexed). Returns NULL on out-of-range.
const char* trt_engine_get_io_tensor_name(TrtEngine* engine, int32_t index);

// I/O mode: 0 = input, 1 = output.
int32_t trt_engine_get_tensor_io_mode(TrtEngine* engine, const char* name);

// Data type (maps to nvinfer::DataType): 0=Float, 1=Half, 2=Int8, 3=Int32, 4=Bool.
int32_t trt_engine_get_tensor_dtype(TrtEngine* engine, const char* name);

// Tensor shape. Returns the number of dimensions. Writes up to `capacity`
// elements into `dims`. Use -1 for dynamic dimensions.
int32_t trt_engine_get_tensor_shape(TrtEngine* engine, const char* name,
                                     int64_t* dims, int32_t capacity);

// Destroy the engine.
void trt_engine_destroy(TrtEngine* engine);

// --- Execution context ---

// Create an execution context from an engine.
TrtStatus trt_context_create(TrtEngine* engine, TrtContext** out);

// Set the shape for a dynamic input tensor.
TrtStatus trt_context_set_input_shape(TrtContext* ctx, const char* name,
                                       const int64_t* dims, int32_t ndims);

// Bind a device pointer to a named tensor (input or output).
TrtStatus trt_context_set_tensor_address(TrtContext* ctx, const char* name, void* ptr);

// Enqueue inference on a CUDA stream. Pass cudaStream_t as void*.
TrtStatus trt_context_enqueue(TrtContext* ctx, void* stream);

// Destroy the execution context.
void trt_context_destroy(TrtContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // TRT_RUNTIME_H
