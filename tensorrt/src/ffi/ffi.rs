//! Raw FFI declarations for the TensorRT-LLM executor C API.

use std::ffi::c_char;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TrtLlmStatus {
    Ok = 0,
    Error = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrtLlmModelType {
    DecoderOnly = 0,
    EncoderOnly = 1,
    EncoderDecoder = 2,
}

/// Opaque handle to the C++ TrtLlmExecutor.
#[repr(C)]
pub struct TrtLlmExecutor {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn trtllm_get_last_error() -> *const c_char;

    pub fn trtllm_executor_create(
        engine_dir: *const c_char,
        model_type: TrtLlmModelType,
        kv_cache_fraction: f32,
        max_beam_width: i32,
        out_handle: *mut *mut TrtLlmExecutor,
    ) -> TrtLlmStatus;

    pub fn trtllm_executor_enqueue(
        handle: *mut TrtLlmExecutor,
        tokens: *const i32,
        n_tokens: usize,
        max_new: i32,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        rep_penalty: f32,
        end_id: i32,
        pad_id: i32,
        streaming: i32,
        out_req_id: *mut u64,
    ) -> TrtLlmStatus;

    pub fn trtllm_executor_await(
        handle: *mut TrtLlmExecutor,
        req_id: u64,
        out_tokens: *mut i32,
        capacity: usize,
        out_n_tokens: *mut usize,
        out_is_final: *mut i32,
        timeout_ms: i64,
    ) -> TrtLlmStatus;

    pub fn trtllm_executor_cancel(handle: *mut TrtLlmExecutor, req_id: u64) -> TrtLlmStatus;

    pub fn trtllm_executor_shutdown(handle: *mut TrtLlmExecutor) -> TrtLlmStatus;

    pub fn trtllm_executor_destroy(handle: *mut TrtLlmExecutor);
}
