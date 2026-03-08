//! Unified TensorRT crate.
//!
//! Plain TensorRT runtime types (`Runtime`, `Engine`, `Context`) are always
//! available. TRT-LLM Executor types are behind the `trtllm` feature gate.

mod ffi;

mod runtime;
pub use runtime::*;

#[cfg(feature = "trtllm")]
mod executor;
#[cfg(feature = "trtllm")]
pub use executor::*;
