use tokenizers::Tokenizer;

mod onnx;
#[cfg(feature = "trtllm")]
mod tensorrt;

fn run_onnx_cpu(model_path: &str, tokenizer_path: &str, prompt: &str) {
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .unwrap_or_else(|e| panic!("Failed to load tokenizer: {e}"));

    let onnx = onnx::Onnx::new(18);
    let session = onnx.create_session(
        onnx::Executor::Cpu,
        onnx::OptimizationLevel::EnableAll,
        4,
        model_path,
    );

    // warmup request (discard result)
    let _ = onnx::generate(&session, &tokenizer, "hi");

    // perform test
    let (output, ttft_ms) = onnx::generate(&session, &tokenizer, prompt);

    println!("[ONNX Runtime CPU]");
    println!("model: {}", model_path);
    println!("TTFT: {ttft_ms} ms");
    println!("Output: {output}");
}

#[allow(dead_code)]
fn run_onnx_cuda(model_path: &str, tokenizer_path: &str, prompt: &str) {
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .unwrap_or_else(|e| panic!("Failed to load tokenizer: {e}"));

    let onnx = onnx::Onnx::new(18);
    let session = onnx.create_session(
        onnx::Executor::Cuda(0),
        onnx::OptimizationLevel::EnableAll,
        4,
        model_path,
    );

    // warmup request (discard result)
    let _ = onnx::generate(&session, &tokenizer, "hi");

    // perform test
    let (output, ttft_ms) = onnx::generate(&session, &tokenizer, prompt);

    println!("[ONNX Runtime CPU]");
    println!("model: {}", model_path);
    println!("TTFT: {ttft_ms} ms");
    println!("Output: {output}");
}

#[cfg(feature = "trtllm")]
fn run_tensorrt(engine_path: &str, prompt: &str) {
    let engine_dir = std::path::Path::new(engine_path)
        .parent()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let tokenizer_path = "data/llama3-3b/source/tokenizer.json";

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .unwrap_or_else(|e| panic!("Failed to load tokenizer: {e}"));

    let config = tensorrt::ExecutorConfig {
        model_type: tensorrt::ModelType::DecoderOnly,
        kv_cache_free_gpu_mem_fraction: 0.5,
        max_beam_width: 1,
    };
    let executor = tensorrt::Executor::new(&engine_dir, &config);

    // Warmup request (discard result)
    let _ = tensorrt::generate(&executor, &tokenizer, "hi");

    let (output, ttft_ms) = tensorrt::generate(&executor, &tokenizer, prompt);

    println!("[TensorRT-LLM]");
    println!("model: {}", engine_path);
    println!("TTFT: {ttft_ms} ms");
    println!("Output: {output}");

    executor.shutdown();
}

fn main() {
    let prompt = "Please tell me something about cars";
    let tokenizer_path = "data/llama3-3b/source/tokenizer.json";

    /*
    let onnx_models = [
        "data/llama3-3b/onnx/f32/model.onnx",
        "data/llama3-3b/onnx/f16/model.onnx",
        "data/llama3-3b/onnx/q8f16/model_quantized.onnx",
        "data/llama3-3b/onnx/q8i8/model_quantized.onnx",
        "data/llama3-3b/onnx/q4f16/model.onnx",
        "data/llama3-3b/onnx/q4i8/model.onnx",
    ];

    for model in &onnx_models {
        if std::path::Path::new(model).exists() {
            run_onnx_cpu(model, tokenizer_path, prompt);
        } else {
            println!("Skipping (not found): {model}");
        }
    }

    for model in &onnx_models {
        if std::path::Path::new(model).exists() {
            run_onnx_cuda(model, tokenizer_path, prompt);
        } else {
            println!("Skipping (not found): {model}");
        }
    }
    */

    #[cfg(feature = "trtllm")]
    {
        let platform = std::env::var("PLATFORM").unwrap_or_else(|_| "desktop".to_string());
        let engine_variants = ["f16", "q8f16", "q8i8", "q4f16", "q4i8"];
        for variant in &engine_variants {
            let path = format!("data/llama3-3b/engine/{platform}/{variant}/rank0.engine");
            if std::path::Path::new(&path).exists() {
                run_tensorrt(&path, prompt);
            } else {
                println!("Skipping (not found): {path}");
            }
        }
    }
}
