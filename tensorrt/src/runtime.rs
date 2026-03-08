//! Safe Rust wrapper around plain TensorRT runtime (nvinfer).
//!
//! Always compiled — no feature gate required. Only needs libnvinfer + libcudart.

use crate::ffi::trt_runtime_ffi as ffi;
use std::ffi::{CStr, CString};

/// TensorRT data types (matches nvinfer1::DataType enum values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Float32,
    Float16,
    Int8,
    Int32,
    Bool,
}

impl DataType {
    fn from_ffi(val: i32) -> Self {
        match val {
            0 => DataType::Float32,
            1 => DataType::Float16,
            2 => DataType::Int8,
            3 => DataType::Int32,
            4 => DataType::Bool,
            _ => DataType::Float32, // fallback
        }
    }

    /// Size in bytes of a single element.
    pub fn byte_size(self) -> usize {
        match self {
            DataType::Float32 | DataType::Int32 => 4,
            DataType::Float16 => 2,
            DataType::Int8 | DataType::Bool => 1,
        }
    }
}

/// Metadata about an engine I/O tensor.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub is_input: bool,
    pub dtype: DataType,
    /// Shape dimensions. -1 indicates a dynamic dimension.
    pub shape: Vec<i64>,
}

fn last_error() -> String {
    unsafe {
        let ptr = ffi::trt_get_last_error();
        if ptr.is_null() {
            "Unknown TensorRT error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

/// TensorRT runtime — the entry point for deserializing engines.
pub struct Runtime {
    handle: *mut ffi::TrtRuntime,
}

unsafe impl Send for Runtime {}

impl Runtime {
    pub fn new() -> Self {
        let mut handle: *mut ffi::TrtRuntime = std::ptr::null_mut();
        let status = unsafe { ffi::trt_runtime_create(&mut handle) };
        if status != ffi::TrtStatus::Ok {
            panic!("Failed to create TensorRT runtime: {}", last_error());
        }
        Self { handle }
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::trt_runtime_destroy(self.handle) };
        }
    }
}

/// A deserialized TensorRT engine (plan).
pub struct Engine {
    handle: *mut ffi::TrtEngine,
}

unsafe impl Send for Engine {}

impl Engine {
    /// Load (deserialize) an engine from a `.engine` / `.plan` file.
    pub fn load(runtime: &Runtime, path: &str) -> Self {
        let c_path = CString::new(path).unwrap_or_else(|e| panic!("Null byte in path: {e}"));
        let mut handle: *mut ffi::TrtEngine = std::ptr::null_mut();
        let status = unsafe { ffi::trt_engine_load(runtime.handle, c_path.as_ptr(), &mut handle) };
        if status != ffi::TrtStatus::Ok {
            panic!("Failed to load engine '{}': {}", path, last_error());
        }
        Self { handle }
    }

    /// Query all I/O tensor metadata.
    pub fn io_tensors(&self) -> Vec<TensorInfo> {
        let n = unsafe { ffi::trt_engine_get_num_io_tensors(self.handle) };
        let mut result = Vec::with_capacity(n as usize);

        for i in 0..n {
            let name_ptr = unsafe { ffi::trt_engine_get_io_tensor_name(self.handle, i) };
            if name_ptr.is_null() {
                continue;
            }
            let name = unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned();
            let c_name = CString::new(name.as_str()).unwrap();

            let io_mode =
                unsafe { ffi::trt_engine_get_tensor_io_mode(self.handle, c_name.as_ptr()) };
            let dtype_raw =
                unsafe { ffi::trt_engine_get_tensor_dtype(self.handle, c_name.as_ptr()) };

            let mut dims = [0i64; 16];
            let ndims = unsafe {
                ffi::trt_engine_get_tensor_shape(
                    self.handle,
                    c_name.as_ptr(),
                    dims.as_mut_ptr(),
                    16,
                )
            };

            result.push(TensorInfo {
                name,
                is_input: io_mode == 0,
                dtype: DataType::from_ffi(dtype_raw),
                shape: dims[..ndims as usize].to_vec(),
            });
        }

        result
    }

    /// Raw handle for creating contexts.
    pub(crate) fn raw(&self) -> *mut ffi::TrtEngine {
        self.handle
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::trt_engine_destroy(self.handle) };
        }
    }
}

/// An execution context bound to an engine.
pub struct Context {
    handle: *mut ffi::TrtContext,
}

unsafe impl Send for Context {}

impl Context {
    /// Create a new execution context from an engine.
    pub fn new(engine: &Engine) -> Self {
        let mut handle: *mut ffi::TrtContext = std::ptr::null_mut();
        let status = unsafe { ffi::trt_context_create(engine.raw(), &mut handle) };
        if status != ffi::TrtStatus::Ok {
            panic!("Failed to create execution context: {}", last_error());
        }
        Self { handle }
    }

    /// Set the concrete shape for a dynamic input tensor.
    pub fn set_input_shape(&self, name: &str, dims: &[i64]) {
        let c_name = CString::new(name).unwrap();
        let status = unsafe {
            ffi::trt_context_set_input_shape(
                self.handle,
                c_name.as_ptr(),
                dims.as_ptr(),
                dims.len() as i32,
            )
        };
        if status != ffi::TrtStatus::Ok {
            panic!(
                "Failed to set input shape for '{}': {}",
                name,
                last_error()
            );
        }
    }

    /// Bind a device (GPU) pointer to a named tensor.
    pub fn set_tensor_address(&self, name: &str, ptr: *mut std::ffi::c_void) {
        let c_name = CString::new(name).unwrap();
        let status =
            unsafe { ffi::trt_context_set_tensor_address(self.handle, c_name.as_ptr(), ptr) };
        if status != ffi::TrtStatus::Ok {
            panic!(
                "Failed to set tensor address for '{}': {}",
                name,
                last_error()
            );
        }
    }

    /// Enqueue inference on a CUDA stream.
    ///
    /// Pass `cudaStream_t` as a raw pointer. Use `std::ptr::null_mut()` for the
    /// default stream.
    pub fn enqueue(&self, stream: *mut std::ffi::c_void) {
        let status = unsafe { ffi::trt_context_enqueue(self.handle, stream) };
        if status != ffi::TrtStatus::Ok {
            panic!("Failed to enqueue inference: {}", last_error());
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::trt_context_destroy(self.handle) };
        }
    }
}
