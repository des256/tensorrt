use std::{fs::File, io::Write, path::Path};

const SAMPLE_RATE: u32 = 24000;
const FRAME_SIZE: usize = 1920; // 24000 Hz / 12.5 Hz

fn read_wav(path: &str) -> Vec<f32> {
    let reader = hound::WavReader::open(path).unwrap_or_else(|e| {
        eprintln!("Failed to open WAV file '{}': {}", path, e);
        std::process::exit(1);
    });

    let spec = reader.spec();
    if spec.channels != 1 {
        eprintln!(
            "Expected mono audio (1 channel), got {} channels",
            spec.channels
        );
        std::process::exit(1);
    }
    if spec.sample_rate != SAMPLE_RATE {
        eprintln!(
            "Expected {}Hz sample rate, got {}Hz",
            SAMPLE_RATE, spec.sample_rate
        );
        std::process::exit(1);
    }

    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect(),
        (hound::SampleFormat::Int, 32) => reader
            .into_samples::<i32>()
            .map(|s| s.unwrap() as f32 / 2_147_483_648.0)
            .collect(),
        (hound::SampleFormat::Float, 32) => {
            reader.into_samples::<f32>().map(|s| s.unwrap()).collect()
        }
        (fmt, bits) => {
            eprintln!("Unsupported WAV format: {:?} {}bit", fmt, bits);
            std::process::exit(1);
        }
    }
}

fn pad_to_frame_boundary(audio: &mut Vec<f32>) {
    let remainder = audio.len() % FRAME_SIZE;
    if remainder != 0 {
        let pad = FRAME_SIZE - remainder;
        audio.extend(std::iter::repeat(0.0f32).take(pad));
    }
}

fn write_bin(path: &str, shape: &[u64], data: &[f32]) {
    let mut file = File::create(path).unwrap_or_else(|e| {
        eprintln!("Failed to create output file '{}': {}", path, e);
        std::process::exit(1);
    });

    let ndims = shape.len() as u32;
    file.write_all(&ndims.to_le_bytes()).unwrap();
    for &dim in shape {
        file.write_all(&dim.to_le_bytes()).unwrap();
    }
    for &val in data {
        file.write_all(&val.to_le_bytes()).unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input.wav> <output.bin>", args[0]);
        std::process::exit(1);
    }
    let input_wav = &args[1];
    let output_bin = &args[2];

    // 1. Read and validate WAV
    let mut audio = read_wav(input_wav);
    let original_len = audio.len();
    println!(
        "Read {} samples ({:.2}s) from {}",
        original_len,
        original_len as f64 / SAMPLE_RATE as f64,
        input_wav
    );

    // 2. Pad to frame boundary
    pad_to_frame_boundary(&mut audio);
    if audio.len() != original_len {
        println!(
            "Padded {} -> {} samples (next multiple of {})",
            original_len,
            audio.len(),
            FRAME_SIZE
        );
    }

    // 3. Load voice_encoder.onnx
    let model_path = "data/pocket/onnx/_raw/voice_encoder.onnx";
    if !Path::new(model_path).exists() {
        eprintln!("Model not found: {}", model_path);
        std::process::exit(1);
    }

    let onnx = onnx::Onnx::new(18);
    let session =
        onnx.create_session(onnx::Executor::Cpu, onnx::OptimizationLevel::EnableAll, 4, model_path);

    // Print model info
    let n_in = session.input_count();
    let n_out = session.output_count();
    for i in 0..n_in {
        println!(
            "  input[{}]: \"{}\" shape={:?}",
            i,
            session.input_name(i),
            session.input_shape(i)
        );
    }
    for i in 0..n_out {
        println!("  output[{}]: \"{}\"", i, session.output_name(i));
    }

    // 4. Create input tensor [1, 1, audio_len]
    let audio_len = audio.len();
    let input_tensor = onnx::Value::from_slice(&onnx, &[1, 1, audio_len], &audio);

    // 5. Run inference
    let input_name = session.input_name(0);
    let output_name = session.output_name(0);
    println!("Running voice encoder...");
    let start = std::time::Instant::now();
    let outputs = session.run(&[(&input_name, &input_tensor)], &[&output_name]);
    let elapsed = start.elapsed();
    println!("Inference completed in {:.1}ms", elapsed.as_secs_f64() * 1000.0);

    // 6. Extract output
    let output = &outputs[0];
    let shape = output.tensor_shape();
    let data = output.extract_as_f32();

    println!(
        "Output shape: {:?} ({} floats)",
        shape,
        data.len()
    );

    // 7. Write binary file
    let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
    write_bin(output_bin, &shape_u64, &data);

    let expected_size = 4 + shape_u64.len() * 8 + data.len() * 4;
    let actual_size = std::fs::metadata(output_bin).unwrap().len() as usize;
    println!(
        "Wrote {} ({} bytes, expected {})",
        output_bin, actual_size, expected_size
    );
    assert_eq!(actual_size, expected_size, "File size mismatch");
}
