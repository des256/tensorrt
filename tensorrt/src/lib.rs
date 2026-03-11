//! Unified TensorRT crate.
//!
//! Plain TensorRT runtime types (`Runtime`, `Engine`, `Context`) are always
//! available. TRT-LLM Executor types are behind the `trtllm` feature gate.

mod ffi;

mod runtime;
pub use runtime::*;

#[cfg(feature = "trt")]
mod executor;
#[cfg(feature = "trt")]
pub use executor::*;
