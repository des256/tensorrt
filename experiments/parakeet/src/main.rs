mod featureextractor;
use featureextractor::*;

mod tokenizer;
use tokenizer::*;

mod onnx_impl;

#[cfg(feature = "trt")]
mod tensorrt_impl;

// test ONNX with CPU provider
#[cfg(all(not(feature = "cuda"), not(feature = "trt")))]
fn run_onnx_cpu<T: onnx::TensorElement + Default + onnx_impl::ToFromF32>(
    model_folder_path: &str,
    audio: &[i16],
) {
    println!("Running ONNX model on CPU: {}", model_folder_path);

    let mut parakeet = onnx_impl::Parakeet::<T>::new(model_folder_path, onnx::Executor::Cpu);
    // TODO: warmup request (discard result)

    // perform test
    let mut latency: Option<u64> = None;
    let mut last_partial = String::new();
    for chunk in audio.chunks(8000) {
        let (partial, new_latency) = parakeet.run_chunk(chunk);
        if latency.is_none() {
            latency = Some(new_latency);
        }
        last_partial = partial;
    }
    let latency = latency.unwrap();
    println!("latency: {} ms", latency);
    println!("Output: {}", last_partial);
}

// Test ONNX runtime on CUDA
#[cfg(all(feature = "cuda", not(feature = "trt")))]
fn run_onnx_cuda<T: onnx::TensorElement + Default + onnx_impl::ToFromF32>(
    model_folder_path: &str,
    audio: &[i16],
) {
    println!("Running ONNX model on CUDA: {}", model_folder_path);

    let mut parakeet = onnx_impl::Parakeet::<T>::new(model_folder_path, onnx::Executor::Cuda(0));
    // TODO: warmup request (discard result)

    // perform test
    let mut latency: Option<u64> = None;
    let mut last_partial = String::new();
    for chunk in audio.chunks(8000) {
        let (partial, new_latency) = parakeet.run_chunk(chunk);
        if latency.is_none() {
            latency = Some(new_latency);
        }
        last_partial = partial;
    }
    let latency = latency.unwrap();
    println!("latency: {} ms", latency);
    println!("Output: {}", last_partial);
}

// Test TensorRT
#[cfg(all(not(feature = "cuda"), feature = "trt"))]
fn run_tensorrt(engine_folder_path: &str, audio: &[i16]) {
    println!("Running ONNX model on CUDA: {}", model_path);

    // TODO: warmup request (discard result)

    // TODO: perform test
    let latency = 0;
    let output = "TODO".to_string();

    println!("latency: {} ms", latency);
    println!("Output: {}", output);
}

fn main() {
    let reader = hound::WavReader::open("test.wav").expect("failed to open test.wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16000, "expected 16 kHz sample rate");
    assert_eq!(spec.channels, 1, "expected mono audio");
    let audio: Vec<i16> = reader.into_samples::<i16>().map(|s| s.unwrap()).collect();

    //run_onnx_cpu("data/parakeet/onnx/f16", &audio);

    #[cfg(all(not(feature = "cuda"), not(feature = "trt")))]
    {
        run_onnx_cpu::<u16>("data/parakeet/onnx/f16", &audio);
        run_onnx_cpu::<f32>("data/parakeet/onnx/q8f16", &audio);
        run_onnx_cpu::<f32>("data/parakeet/onnx/q8i8", &audio);
        run_onnx_cpu::<f32>("data/parakeet/onnx/q4f16", &audio);
        run_onnx_cpu::<f32>("data/parakeet/onnx/q4i8", &audio);
    }
    #[cfg(all(feature = "cuda", not(feature = "trt")))]
    {
        run_onnx_cuda::<u16>("data/parakeet/onnx/f16", &audio);
        run_onnx_cuda::<f32>("data/parakeet/onnx/q8f16", &audio);
        run_onnx_cuda::<f32>("data/parakeet/onnx/q8i8", &audio);
        run_onnx_cuda::<f32>("data/parakeet/onnx/q4f16", &audio);
        run_onnx_cuda::<f32>("data/parakeet/onnx/q4i8", &audio);
    }
    #[cfg(all(not(feature = "cuda"), feature = "trt"))]
    {
        #[cfg(feature = "jetson")]
        let platform = "murdock";
        #[cfg(not(feature = "jetson"))]
        let platform = "genmei";

        let engine_variants = ["q8f16", "q8i8", "q4f16", "q4i8"];
        for engine_variant in &engine_variants {
            run_tensorrt(
                &format!("data/parakeet/engine/{}/{}", platform, engine_variant),
                &audio,
            );
        }
    }
}
