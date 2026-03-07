mod onnx;

fn read_wav(path: &str) -> Vec<i16> {
    let data = std::fs::read(path).expect("Failed to read WAV file");
    assert!(&data[0..4] == b"RIFF", "Not a RIFF file");
    assert!(&data[8..12] == b"WAVE", "Not a WAVE file");

    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut num_channels = 0u16;
    let mut audio_data: &[u8] = &[];

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        match chunk_id {
            b"fmt " => {
                num_channels = u16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap());
                sample_rate = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
                bits_per_sample = u16::from_le_bytes(data[pos + 14..pos + 16].try_into().unwrap());
            }
            b"data" => {
                audio_data = &data[pos..pos + chunk_size];
            }
            _ => {}
        }
        pos += chunk_size;
    }

    assert_eq!(bits_per_sample, 16, "Only 16-bit WAV supported");
    assert_eq!(sample_rate, 16000, "Only 16kHz WAV supported");

    let samples: Vec<i16> = audio_data
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // mix to mono if stereo
    if num_channels == 2 {
        samples
            .chunks_exact(2)
            .map(|pair| ((pair[0] as i32 + pair[1] as i32) / 2) as i16)
            .collect()
    } else {
        samples
    }
}

fn main() {
    let samples = read_wav("test.wav");
    let duration_secs = samples.len() as f64 / 16000.0;
    println!("Audio: {:.2}s ({} samples)", duration_secs, samples.len());

    let result = onnx::transcribe(&samples);

    for partial in &result.partials {
        println!("[partial] {}", partial);
    }
    println!("[final] {}", result.final_text);

    println!("\nTotal processing time: {:.3}s", result.total_time.as_secs_f64());
    if let Some(first) = result.time_to_first_utterance {
        println!("Time to first utterance: {:.3}s", first.as_secs_f64());
    } else {
        println!("No utterances detected");
    }
}
