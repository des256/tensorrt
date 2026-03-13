mod onnx_impl;

#[cfg(feature = "trt")]
mod tensorrt_impl;

// test ONNX with CPU provider
#[cfg(all(not(feature = "cuda"), not(feature = "trt")))]
fn run_onnx_cpu<T: onnx::TensorElement + Default + onnx_impl::ToFromF32>(
    model_folder_path: &str,
    text: &str,
    audio_path: &str,
) {
    println!("Running ONNX model on CPU: {}", model_folder_path);

    let mut pocket = onnx_impl::Pocket::<T>::new(model_folder_path, onnx::Executor::Cpu);

    let mut audio: Vec<i16> = Vec::new();
    let start = std::time::Instant::now();

    pocket.start(text);
    let mut steps = 0u32;
    while let Some(chunk) = pocket.step() {
        if steps == 0 {
            println!("first-token latency: {} ms", start.elapsed().as_millis());
        }
        steps += 1;
        audio.extend_from_slice(chunk);
    }
    let total_ms = start.elapsed().as_millis();
    println!("{} steps in {} ms ({:.1} ms/step)", steps, total_ms,
             total_ms as f64 / steps.max(1) as f64);

    // Save audio as WAV
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(audio_path, spec).expect("create wav");
    for &s in &audio {
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();

    let duration_s = audio.len() as f64 / 24000.0;
    println!("{:.1}s audio ({} samples) saved to {}", duration_s, audio.len(), audio_path);
}

// test ONNX with CUDA provider
#[cfg(all(feature = "cuda", not(feature = "trt")))]
fn run_onnx_cuda<T: onnx::TensorElement + Default + onnx_impl::ToFromF32>(
    model_folder_path: &str,
    text: &str,
    audio_path: &str,
) {
    println!("Running ONNX model on CUDA: {}", model_folder_path);

    let mut pocket = onnx_impl::Pocket::<T>::new(model_folder_path, onnx::Executor::Cuda(0));

    let mut audio: Vec<i16> = Vec::new();

    let mut latency: Option<u64> = None;
    pocket.start(text);
    while let Some(chunk) = pocket.step() {
        if latency.is_none() {
            latency = Some(new_latency);
        }
        audio.extend_from_slice(chunk);
    }

    // TODO: save audio to file

    let latency = latency.unwrap();
    println!("latency: {} ms", latency);
    println!("audio saved to {}", audio_path);
}

// test TensorRT
#[cfg(all(not(feature = "cuda"), feature = "trt"))]
fn run_tensorrt(engine_folder_path: &str, text: &str, audio_path: &str) {
    println!("Running TensorRT engine: {}", engine_folder_path);

    let mut pocket = tensorrt_impl::Pocket::new(engine_folder_path);

    let mut audio: Vec<i16> = Vec::new();

    let mut latency: Option<u64> = None;
    pocket.start(text);
    while let Some(chunk) = pocket.step() {
        if latency.is_none() {
            latency = Some(new_latency);
        }
        audio.extend_from_slice(chunk);
    }

    // TODO: save audio to file

    let latency = latency.unwrap();
    println!("latency: {} ms", latency);
    println!("audio saved to {}", audio_path);
}

fn main() {
    let sentence = "After days in the shade, sunlight now cuts through the rough surface, sending shimmering rays dancing across the rocky bed of the river, and illuminating the patches of bright-green algae that carpeted the rocks of deeper, slower pools.";
    run_onnx_cpu::<u16>("data/pocket/onnx/f16", sentence, "cpu_f16.wav");
    /*
    #[cfg(all(not(feature = "cuda"), not(feature = "trt")))]
    {
        run_onnx_cpu::<u16>("data/pocket/onnx/f16", sentence, "cpu_f16.wav");
        run_onnx_cpu::<f32>("data/pocket/onnx/q8f16", sentence, "cpu_q8f16.wav");
        run_onnx_cpu::<f32>("data/pocket/onnx/q8i8", sentence, "cpu_q8i8.wav");
        run_onnx_cpu::<f32>("data/pocket/onnx/q4f16", sentence, "cpu_q4f16.wav");
        run_onnx_cpu::<f32>("data/pocket/onnx/q4i8", sentence, "cpu_q4i8.wav");
    }
    #[cfg(all(feature = "cuda", not(feature = "trt")))]
    {
        run_onnx_cuda::<u16>("data/pocket/onnx/f16", sentence, "cuda_f16.wav");
        run_onnx_cuda::<f32>("data/pocket/onnx/q8f16", sentence, "cuda_q8f16.wav");
        run_onnx_cuda::<f32>("data/pocket/onnx/q8i8", sentence, "cuda_q8i8.wav");
        run_onnx_cuda::<f32>("data/pocket/onnx/q4f16", sentence, "cuda_q4f16.wav");
        run_onnx_cuda::<f32>("data/pocket/onnx/q4i8", sentence, "cuda_q4i8.wav");
    }
    #[cfg(all(not(feature = "cuda"), feature = "trt"))]
    {
        #[cfg(feature = "jetson")]
        let platform = "jetson";
        #[cfg(not(feature = "jetson"))]
        let platform = "desktop";

        let engine_variants = ["f16", "q8f16", "q8i8", "q4f16", "q4i8"];
        for engine_variant in &engine_variants {
            run_tensorrt(
                &format!("data/pocket/engine/{}/{}", platform, engine_variant),
                sentence,
                &format!("tensorrt_{}.wav", engine_variant),
            );
        }
    }
    */
}
