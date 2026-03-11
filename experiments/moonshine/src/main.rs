use tokenizers::Tokenizer;

mod onnx;
#[cfg(feature = "trt")]
mod tensorrt;

// test ONNX with CPU provider
#[cfg(all(not(feature = "cuda"), not(feature = "trt")))]
fn run_onnx_cpu(model_folder_path: &str, audio: &[i16]) {
    println!("Running ONNX model on CPU: {}", model_folder_path);

    // TODO: warmup request (discard result)

    // TODO: perform test
    let latency = 0u64;
    let output = "TODO".to_string();

    println!("latency: {} ms", latency);
    println!("Output: {}", output);
}

// Test ONNX runtime on CUDA
#[cfg(all(feature = "cuda", not(feature = "trt")))]
fn run_onnx_cuda(model_folder_path: &str, audio: &[i16]) {
    println!("Running ONNX model on CUDA: {}", model_path);

    // TODO: warmup request (discard result)

    // TODO: perform test
    latency = 0;
    output = "TODO".to_string()

    println!("latency: {} ms", latency);
    println!("Output: {}", output);
}

// Test TensorRT
#[cfg(all(not(feature = "cuda"), feature = "trt"))]
fn run_tensorrt(engine_folder_path: &str, audio: &[i16]) {
    println!("Running ONNX model on CUDA: {}", model_path);

    // TODO: warmup request (discard result)

    // TODO: perform test
    latency = 0;
    output = "TODO".to_string()

    println!("latency: {} ms", latency);
    println!("Output: {}", output);
}

fn main() {
    // TODO: load audio from file
    let audio = vec![0i16; 16000];

    #[cfg(not(feature = "trt"))]
    let model_folder_paths = [
        "data/moonshine/onnx/f16",
        "data/moonshine/onnx/q8f16",
        "data/moonshine/onnx/q8i8",
        "data/moonshine/onnx/q4f16",
        "data/moonshine/onnx/q4i8",
    ];
    #[cfg(all(not(feature = "cuda"), not(feature = "trt")))]
    for model_folder_path in &model_folder_paths {
        run_onnx_cpu(model_folder_path, &audio);
    }
    #[cfg(all(feature = "cuda", not(feature = "trt")))]
    for model_folder_path in &model_folder_paths {
        run_onnx_cuda(model_folder_path, &audio);
    }
    #[cfg(all(not(feature = "cuda"), feature = "trt"))]
    {
        #[cfg(feature = "jetson")]
        let platform = "murdock";
        #[cfg(not(feature = "jetson"))]
        let platform = "genmei";

        let engine_variants = ["q8f16", "q8i8", "q4f16", "q4i8"];
        for engine_variant in &engine_variants {
            run_tensorrt(&format!(
                "data/moonshine/engine/{}/{}",
                platform, engine_variant
            ), &audio);
        }
    }
}
