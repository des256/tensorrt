use {
    super::*,
    std::{sync::Arc, time::Instant},
};

const SAMPLES_PER_FRAME: usize = 320;
const WINDOW_SIZE: usize = 200;
const WINDOW_SHIFT: usize = 20;
const MEL_SIZE: usize = 128;
const ENCODER_OUTPUT_DIM: usize = 1024; // encoder output vector size
const BLANK_ID: i64 = 8192; // token ID for blank token (vocab_size index)
const VOCAB_SIZE: usize = 8193; // 8192 tokens + 1 blank
const NUM_DURATIONS: usize = 5; // TDT duration logit count
const TDT_DURATIONS: [usize; 5] = [0, 1, 2, 3, 4]; // TDT duration values
const DECODER_STATE_DIM: usize = 640; // decoder context dimension
const MAX_SYMBOLS_PER_STEP: usize = 16; // maximum number of tokens to decode per step

fn find_longest_common_substring(s1: &str, s2: &str) -> Option<(usize, usize, usize)> {
    let len1 = s1.len();
    let len2 = s2.len();
    let mut table = vec![vec![0; len2 + 1]; len1 + 1];
    let mut max_len = 0;
    let mut end_idx1 = 0;
    let mut end_idx2 = 0;
    let b1 = s1.as_bytes();
    let b2 = s2.as_bytes();
    for i in 1..=len1 {
        for j in 1..=len2 {
            if b1[i - 1] == b2[j - 1] {
                table[i][j] = table[i - 1][j - 1] + 1;
                if table[i][j] > max_len {
                    max_len = table[i][j];
                    end_idx1 = i - 1;
                    end_idx2 = j - 1;
                }
            }
        }
    }
    if max_len == 0 {
        None
    } else {
        Some((max_len, end_idx1, end_idx2))
    }
}

pub struct Parakeet {
    onnx: Arc<onnx::Onnx>,
    feature_extractor: FeatureExtractor,
    encoder_session: onnx::Session,
    decoder_session: onnx::Session,
    at_start: bool,
    accumulated_text: String,
    samples: Vec<i16>,
    decoder_state1: onnx::Value,
    decoder_state2: onnx::Value,
    tokenizer: Tokenizer,
    last_token: i64,
}

impl Parakeet {
    pub fn new(model_folder_path: &str, executor: onnx::Executor) -> Self {
        let onnx = onnx::Onnx::new(18);
        let feature_extractor = FeatureExtractor::new();
        let encoder_session = onnx.create_session(
            executor,
            onnx::OptimizationLevel::EnableAll,
            4,
            format!("{}/encoder-model.onnx", model_folder_path),
        );
        let decoder_session = onnx.create_session(
            executor,
            onnx::OptimizationLevel::EnableAll,
            4,
            format!("{}/decoder_joint-model.onnx", model_folder_path),
        );
        let decoder_state1 = onnx::Value::zeros::<u16>(&onnx, &[2, 1, DECODER_STATE_DIM as i64]);
        let decoder_state2 = onnx::Value::zeros::<u16>(&onnx, &[2, 1, DECODER_STATE_DIM as i64]);
        Self {
            onnx,
            feature_extractor,
            encoder_session,
            decoder_session,
            at_start: true,
            accumulated_text: String::new(),
            samples: Vec::new(),
            decoder_state1,
            decoder_state2,
            tokenizer: Tokenizer::new(),
            last_token: BLANK_ID,
        }
    }

    pub fn transcribe_window(&mut self, window: &[i16]) -> String {
        // reset decoder state for each independent window
        self.decoder_state1 =
            onnx::Value::zeros::<u16>(&self.onnx, &[2, 1, DECODER_STATE_DIM as i64]);
        self.decoder_state2 =
            onnx::Value::zeros::<u16>(&self.onnx, &[2, 1, DECODER_STATE_DIM as i64]);
        self.last_token = BLANK_ID;

        // extract features
        let features = self.feature_extractor.extract_features(window);

        // encode
        let num_frames = features.len() / MEL_SIZE;
        let mut transposed = vec![0.0f32; MEL_SIZE * num_frames];
        for frame in 0..num_frames {
            for bin in 0..MEL_SIZE {
                transposed[bin * num_frames + frame] = features[frame * MEL_SIZE + bin];
            }
        }
        let transposed_f16: Vec<u16> = transposed.iter().map(|&v| onnx::f32_to_f16(v)).collect();
        let audio_signal_tensor =
            onnx::Value::from_slice(&self.onnx, &[1, MEL_SIZE, num_frames], &transposed_f16);
        let length_tensor = onnx::Value::from_slice(&self.onnx, &[1], &[num_frames as i64]);
        let outputs = self.encoder_session.run(
            &[
                ("audio_signal", &audio_signal_tensor),
                ("length", &length_tensor),
            ],
            &["outputs", "encoded_lengths"],
        );
        let enc_output = outputs[0].extract_tensor::<u16>();
        let t_enc = enc_output.len() / ENCODER_OUTPUT_DIM;
        let mut encoded = vec![0u16; t_enc * ENCODER_OUTPUT_DIM];
        for frame in 0..t_enc {
            for dim in 0..ENCODER_OUTPUT_DIM {
                encoded[frame * ENCODER_OUTPUT_DIM + dim] = enc_output[dim * t_enc + frame];
            }
        }

        // decode
        let num_frames = encoded.len() / ENCODER_OUTPUT_DIM;
        let mut tokens = Vec::new();
        let mut time_index: usize = 0;
        let mut symbols_added: usize = 0;
        while time_index < num_frames {
            if symbols_added >= MAX_SYMBOLS_PER_STEP {
                time_index += 1;
                symbols_added = 0;
                continue;
            }
            let state1_data = self.decoder_state1.extract_tensor::<u16>().to_vec();
            let state2_data = self.decoder_state2.extract_tensor::<u16>().to_vec();
            let frame_start = time_index * ENCODER_OUTPUT_DIM;
            let frame_data = &encoded[frame_start..frame_start + ENCODER_OUTPUT_DIM];
            let encoder_outputs_tensor =
                onnx::Value::from_slice(&self.onnx, &[1, ENCODER_OUTPUT_DIM, 1], frame_data);
            let targets_tensor =
                onnx::Value::from_slice(&self.onnx, &[1, 1], &[self.last_token as i32]);
            let target_length_tensor = onnx::Value::from_slice(&self.onnx, &[1], &[1i32]);
            let mut outputs = self.decoder_session.run(
                &[
                    ("encoder_outputs", &encoder_outputs_tensor),
                    ("targets", &targets_tensor),
                    ("target_length", &target_length_tensor),
                    ("input_states_1", &self.decoder_state1),
                    ("input_states_2", &self.decoder_state2),
                ],
                &[
                    "outputs",
                    "prednet_lengths",
                    "output_states_1",
                    "output_states_2",
                ],
            );
            let logits = outputs[0].extract_as_f32();
            self.decoder_state1 = outputs.remove(2);
            self.decoder_state2 = outputs.remove(2);
            let token_logits = &logits[..VOCAB_SIZE];
            let predicted_token = token_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .unwrap_or(0);

            // predict duration: argmax over duration logits [VOCAB_SIZE..VOCAB_SIZE+NUM_DURATIONS)
            let duration_logits = &logits[VOCAB_SIZE..VOCAB_SIZE + NUM_DURATIONS];
            let duration_index = duration_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let mut duration = TDT_DURATIONS[duration_index];

            if predicted_token == BLANK_ID {
                self.decoder_state1 = onnx::Value::from_slice::<u16>(
                    &self.onnx,
                    &[2, 1, DECODER_STATE_DIM],
                    &state1_data,
                );
                self.decoder_state2 = onnx::Value::from_slice::<u16>(
                    &self.onnx,
                    &[2, 1, DECODER_STATE_DIM],
                    &state2_data,
                );
                if duration == 0 {
                    duration = 1;
                }
                time_index += duration;
                symbols_added = 0;
            } else {
                tokens.push(predicted_token);
                self.last_token = predicted_token;
                time_index += duration;
                if duration > 0 {
                    symbols_added = 0;
                } else {
                    symbols_added += 1;
                }
            }
        }
        let text = self.tokenizer.tokenize(&tokens);
        text
    }

    fn merge_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if self.accumulated_text.is_empty() {
            self.accumulated_text = text.to_string();
        } else if let Some((_, end1, end2)) =
            find_longest_common_substring(&self.accumulated_text, text)
        {
            self.accumulated_text = format!("{}{}", &self.accumulated_text[..end1], &text[end2..]);
        }
    }

    pub fn run_chunk(&mut self, audio: &[i16]) -> (String, u64) {
        self.samples.extend_from_slice(audio);
        let start = Instant::now();
        if self.at_start && self.samples.len() < SAMPLES_PER_FRAME * WINDOW_SIZE {
            let mut window = self.samples.clone();
            window.resize(SAMPLES_PER_FRAME * WINDOW_SIZE, 0);
            let text = self.transcribe_window(&window);
            self.merge_text(&text);
            (
                self.accumulated_text.clone(),
                start.elapsed().as_millis() as u64,
            )
        } else {
            self.at_start = false;
            while self.samples.len() >= SAMPLES_PER_FRAME * WINDOW_SIZE {
                let window = self.samples[..SAMPLES_PER_FRAME * WINDOW_SIZE].to_vec();
                self.samples.drain(..SAMPLES_PER_FRAME * WINDOW_SHIFT);
                let text = self.transcribe_window(&window);
                self.merge_text(&text);
            }
            if self.samples.len() < SAMPLES_PER_FRAME * WINDOW_SIZE {
                let mut window = self.samples.clone();
                window.resize(SAMPLES_PER_FRAME * WINDOW_SIZE, 0);
                let text = self.transcribe_window(&window);
                self.merge_text(&text);
            }
            (self.accumulated_text.clone(), 0)
        }
    }
}
