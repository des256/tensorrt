#[cfg(feature = "trtllm")]
mod tensorrt;

fn main() {
    #[cfg(feature = "trtllm")]
    {
        use ::tensorrt::{Executor, ExecutorConfig, ModelType};
        use tokenizers::Tokenizer;

        let engine_dir = "data/llama3/engine";
        let tokenizer_path = "data/raw/llama3/tokenizer.json";

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .unwrap_or_else(|e| panic!("Failed to load tokenizer: {e}"));

        let config = ExecutorConfig {
            model_type: ModelType::DecoderOnly,
            kv_cache_free_gpu_mem_fraction: 0.5,
            max_beam_width: 1,
        };
        let executor = Executor::new(engine_dir, &config);

        let (output, ttft_ms) =
            tensorrt::generate(&executor, &tokenizer, "Please tell me something about cars");

        println!("TTFT: {ttft_ms} ms");
        println!("Output: {output}");

        executor.shutdown();
    }

    #[cfg(not(feature = "trtllm"))]
    {
        eprintln!("Build with --features trtllm to enable TensorRT-LLM inference");
    }
}
