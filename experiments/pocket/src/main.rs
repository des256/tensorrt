mod onnx_impl;

#[cfg(feature = "trt")]
mod tensorrt_impl;

/// Run voice_encoder.onnx on a WAV file, returning the flat f32 conditioning
/// tensor of shape [1, T_frames, 1024].
fn encode_voice(wav_path: &str) -> Vec<f32> {
    println!("Encoding voice from {}", wav_path);
    let reader = hound::WavReader::open(wav_path).unwrap_or_else(|e| {
        panic!("Failed to open voice WAV '{}': {}", wav_path, e);
    });
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "expected mono audio, got {} channels", spec.channels);
    assert_eq!(spec.sample_rate, 24000, "expected 24kHz, got {}Hz", spec.sample_rate);

    let mut audio: Vec<f32> = reader
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    // Pad to next multiple of 1920 (frame boundary at 12.5 Hz)
    let frame_size = 1920;
    let remainder = audio.len() % frame_size;
    if remainder != 0 {
        audio.resize(audio.len() + frame_size - remainder, 0.0);
    }

    let onnx = onnx::Onnx::new(18);
    let session = onnx.create_session(
        onnx::Executor::Cpu,
        onnx::OptimizationLevel::EnableAll,
        4,
        "data/pocket/onnx/_raw/voice_encoder.onnx",
    );

    let audio_len = audio.len();
    let input = onnx::Value::from_slice(&onnx, &[1, 1, audio_len], &audio);
    let outputs = session.run(&[("audio", &input)], &["conditioning"]);

    let shape = outputs[0].tensor_shape();
    println!("Voice conditioning: {:?}", shape);
    outputs[0].extract_as_f32()
}

// test ONNX with CPU provider
#[cfg(all(not(feature = "cuda"), not(feature = "trt")))]
fn run_onnx_cpu<T: onnx::TensorElement + Default + onnx_impl::ToFromF32>(
    model_folder_path: &str,
    voice: &[f32],
    text: &str,
    audio_path: &str,
) {
    println!("Running ONNX model on CPU: {}", model_folder_path);

    let mut pocket = onnx_impl::Pocket::<T>::new(model_folder_path, onnx::Executor::Cpu);

    let mut audio: Vec<i16> = Vec::new();
    let start = std::time::Instant::now();

    pocket.start(voice, text);
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
    voice: &[f32],
    text: &str,
    audio_path: &str,
) {
    println!("Running ONNX model on CUDA: {}", model_folder_path);

    let mut pocket = onnx_impl::Pocket::<T>::new(model_folder_path, onnx::Executor::Cuda(0));

    let mut audio: Vec<i16> = Vec::new();

    let mut latency: Option<u64> = None;
    pocket.start(voice, text);
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
fn run_tensorrt(engine_folder_path: &str, voice: &[f32], text: &str, audio_path: &str) {
    println!("Running TensorRT engine: {}", engine_folder_path);

    let mut pocket = tensorrt_impl::Pocket::new(engine_folder_path);

    let mut audio: Vec<i16> = Vec::new();

    let mut latency: Option<u64> = None;
    pocket.start(voice, text);
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
    let voice = encode_voice("data/pocket/voices/desmond-calm.wav");
    run_onnx_cpu::<u16>("data/pocket/onnx/f16", &voice, sentence, "cpu_f16.wav");
    /*
    #[cfg(all(not(feature = "cuda"), not(feature = "trt")))]
    {
        run_onnx_cpu::<u16>("data/pocket/onnx/f16", voice, sentence, "cpu_f16.wav");
        run_onnx_cpu::<f32>("data/pocket/onnx/q8f16", voice, sentence, "cpu_q8f16.wav");
        run_onnx_cpu::<f32>("data/pocket/onnx/q8i8", voice, sentence, "cpu_q8i8.wav");
        run_onnx_cpu::<f32>("data/pocket/onnx/q4f16", voice, sentence, "cpu_q4f16.wav");
        run_onnx_cpu::<f32>("data/pocket/onnx/q4i8", voice, sentence, "cpu_q4i8.wav");
    }
    #[cfg(all(feature = "cuda", not(feature = "trt")))]
    {
        run_onnx_cuda::<u16>("data/pocket/onnx/f16", voice, sentence, "cuda_f16.wav");
        run_onnx_cuda::<f32>("data/pocket/onnx/q8f16", voice, sentence, "cuda_q8f16.wav");
        run_onnx_cuda::<f32>("data/pocket/onnx/q8i8", voice, sentence, "cuda_q8i8.wav");
        run_onnx_cuda::<f32>("data/pocket/onnx/q4f16", voice, sentence, "cuda_q4f16.wav");
        run_onnx_cuda::<f32>("data/pocket/onnx/q4i8", voice, sentence, "cuda_q4i8.wav");
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
                &format!("data/pocket/engine/{}/{}", platform, engine_variant), voice, sentence, &format!("tensorrt_{}.wav", engine_variant),
            );
        }
    }
    */
}
