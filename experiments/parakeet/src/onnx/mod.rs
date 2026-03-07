use std::time::{Duration, Instant};

const PARAKEET_ENCODER_PATH: &str = "data/parakeet/encoder.onnx";
const PARAKEET_DECODER_PATH: &str = "data/parakeet/decoder_joint.onnx";
const PARAKEET_TOKENIZER_PATH: &str = "data/parakeet/tokenizer.model";

const HANN_WINDOW_SIZE: usize = 400; // number of samples in Hann window
const FFT_SIZE: usize = 512; // number of bins in FFT
const SPECTRUM_SIZE: usize = FFT_SIZE / 2 + 1; // number of bins in spectrum
const MEL_SIZE: usize = 128; // number of bands in mel filterbank
const ENCODER_WINDOW_SIZE: usize = 121; // size of encoder window
const ENCODER_CHUNK_SHIFT: usize = 112; // number of frames to shift between chunks
const ENCODER_LAYERS: usize = 24; // number of layers in encoder
const ENCODER_OUTPUT_DIM: usize = 1024; // encoder output vector size
const ENCODER_CHANNEL_DIM: usize = 70; // channel-related context dimension
const ENCODER_TIME_DIM: usize = 8; // time-related context dimension
const BLANK_ID: i64 = 1024; // token ID for blank token
const VOCAB_SIZE: usize = 1025; // 1024 tokens + 1 blank
const DECODER_STATE_DIM: usize = 640; // decoder context dimension
const MAX_SYMBOLS_PER_STEP: usize = 16; // maximum number of tokens to decode per step

const CHUNK_SAMPLES: usize = 16000; // 1 second chunks

mod featureextractor;
use featureextractor::*;

mod encoder;
use encoder::*;

mod decoder;
use decoder::*;

mod tokenizer;
use tokenizer::*;

pub struct TranscribeResult {
    pub partials: Vec<String>,
    pub final_text: String,
    pub time_to_first_utterance: Option<Duration>,
    pub total_time: Duration,
}

pub fn transcribe(samples: &[i16]) -> TranscribeResult {
    let rt = onnx::Onnx::new(18);

    let mut feature_extractor = FeatureExtractor::new();
    let mut encoder = Encoder::new(&rt, onnx::Executor::Cpu);
    let mut decoder = Decoder::new(&rt, onnx::Executor::Cpu);
    let tokenizer = Tokenizer::new();

    let mut features: Vec<f32> = Vec::new();
    let mut frames = 0usize;
    let mut accumulator = String::new();
    let mut partials = Vec::new();
    let mut first_utterance_time: Option<Duration> = None;
    let start = Instant::now();

    let mut offset = 0;
    while offset < samples.len() {
        let end = (offset + CHUNK_SAMPLES).min(samples.len());
        let chunk = &samples[offset..end];
        let is_last = end == samples.len();
        offset = end;

        // extract features
        let new_features = feature_extractor.extract_features(chunk);
        let new_frames = new_features.len() / MEL_SIZE;
        if new_frames > 0 {
            features.extend_from_slice(&new_features);
            frames += new_frames;
        }

        // on last chunk, pad to fill at least one encoder window
        if is_last && frames > 0 && frames < ENCODER_WINDOW_SIZE {
            let pad_frames = ENCODER_WINDOW_SIZE - frames;
            features.resize(features.len() + pad_frames * MEL_SIZE, 0.0);
            frames = ENCODER_WINDOW_SIZE;
        }

        // process full encoder windows
        while frames >= ENCODER_WINDOW_SIZE {
            let window = &features[..ENCODER_WINDOW_SIZE * MEL_SIZE];
            let mut transposed = vec![0.0f32; ENCODER_WINDOW_SIZE * MEL_SIZE];
            for frame in 0..ENCODER_WINDOW_SIZE {
                for bin in 0..MEL_SIZE {
                    transposed[bin * ENCODER_WINDOW_SIZE + frame] = window[frame * MEL_SIZE + bin];
                }
            }

            let shift = ENCODER_CHUNK_SHIFT.min(frames);
            features.drain(..shift * MEL_SIZE);
            frames -= shift;

            let encoder_frames = encoder.encode_window(&transposed);
            let tokens = decoder.decode(&encoder_frames);
            let text = tokenizer.tokenize(&tokens);

            if text.chars().any(|c| c.is_alphanumeric()) {
                accumulator.push_str(&text);

                if first_utterance_time.is_none() {
                    first_utterance_time = Some(start.elapsed());
                }

                partials.push(accumulator.clone());
            }
        }
    }

    let total_time = start.elapsed();

    TranscribeResult {
        partials,
        final_text: accumulator,
        time_to_first_utterance: first_utterance_time,
        total_time,
    }
}
