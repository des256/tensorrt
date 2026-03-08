#[cfg(feature = "trtllm")]
mod tensorrt;

mod onnx;

fn main() {
    let prompt = "Please tell me something about cars";

    #[cfg(feature = "trtllm")]
    {
        use ::tensorrt::{Executor, ExecutorConfig, ModelType};
        use tokenizers::Tokenizer;

        let engine_dir = std::env::var("TRTLLM_ENGINE_DIR")
            .unwrap_or_else(|_| "data/llama3/engine".to_string());
        let tokenizer_path = "data/raw/llama3/tokenizer.json";

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .unwrap_or_else(|e| panic!("Failed to load tokenizer: {e}"));

        let config = ExecutorConfig {
            model_type: ModelType::DecoderOnly,
            kv_cache_free_gpu_mem_fraction: 0.5,
            max_beam_width: 1,
        };
        let executor = Executor::new(&engine_dir, &config);

        // Warmup request (discard result)
        let _ = tensorrt::generate(&executor, &tokenizer, "hi");

        let (output, ttft_ms) = tensorrt::generate(&executor, &tokenizer, prompt);

        println!("[TensorRT-LLM]");
        println!("TTFT: {ttft_ms} ms");
        println!("Output: {output}");

        executor.shutdown();
    }

    {
        use ::onnx::{Executor as OnnxExecutor, Onnx, OptimizationLevel};
        use tokenizers::Tokenizer;

        let model_path = "data/llama3/onnx/model.onnx";
        let tokenizer_path = "data/llama3/onnx/tokenizer.json";

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .unwrap_or_else(|e| panic!("Failed to load tokenizer: {e}"));

        let ort = Onnx::new(18);
        let session = ort.create_session(
            OnnxExecutor::Cuda(0),
            OptimizationLevel::EnableAll,
            1,
            model_path,
        );

        // Warmup request (discard result)
        let _ = onnx::generate(&session, &tokenizer, "hi");

        let (output, ttft_ms) = onnx::generate(&session, &tokenizer, prompt);

        println!("[ONNX Runtime]");
        println!("TTFT: {ttft_ms} ms");
        println!("Output: {output}");
    }
}
