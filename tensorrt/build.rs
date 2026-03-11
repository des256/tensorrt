fn main() {
    // --- Always: plain TensorRT runtime ---
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/ffi/trt_runtime.cpp")
        .include("/usr/local/cuda/include")
        .compile("trt_runtime_stub");

    // TensorRT
    println!("cargo:rustc-link-search=native=/usr/local/tensorrt/lib");
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=dylib=nvinfer");

    // CUDA runtime
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/aarch64-linux/lib");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/x86_64-linux/lib");
    println!("cargo:rustc-link-lib=dylib=cudart");

    // C++ standard library
    println!("cargo:rustc-link-lib=dylib=stdc++");

    println!("cargo:rerun-if-changed=src/ffi/trt_runtime.cpp");
    println!("cargo:rerun-if-changed=src/ffi/trt_runtime.h");

    let trtllm_root = std::env::var("TRTLLM_ROOT")
        .expect("TRTLLM_ROOT env var must be set when building with --features trtllm");

    let include_dir = format!("{trtllm_root}/cpp/include");
    let build_lib_dir = format!("{trtllm_root}/cpp/build/tensorrt_llm");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/ffi/trtllm_executor.cpp")
        .include(&include_dir)
        .include("/usr/local/cuda/include")
        .flag("-Wno-deprecated-declarations")
        .flag("-D_GLIBCXX_USE_CXX11_ABI=1")
        .compile("trtllm_executor_stub");

    // TRT-LLM specific libraries (nvinfer/cudart/stdc++ already linked above).
    println!("cargo:rustc-link-search=native={build_lib_dir}");
    println!("cargo:rustc-link-search=native={build_lib_dir}/plugins");
    println!("cargo:rustc-link-lib=dylib=tensorrt_llm");
    println!("cargo:rustc-link-lib=dylib=nvinfer_plugin_tensorrt_llm");

    println!("cargo:rerun-if-changed=src/ffi/trtllm_executor.cpp");
    println!("cargo:rerun-if-changed=src/ffi/trtllm_executor.h");
    println!("cargo:rerun-if-env-changed=TRTLLM_ROOT");
}
