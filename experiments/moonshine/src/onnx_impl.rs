use {
    onnx::TensorElement,
    std::{marker::PhantomData, path::Path, sync::Arc, time::Instant},
};

const SAMPLES_PER_FRAME: usize = 320;
const WINDOW_SIZE: usize = 100;
const WINDOW_SHIFT: usize = 20;
const VOCAB_SIZE: usize = 32768;
const BOS_ID: i64 = 1;
const EOS_ID: i64 = 2;
const MAX_TOKENS: usize = 64;
const REPETITION_PENALTY: f32 = 1.2;
const NO_REPEAT_NGRAM: usize = 3;
const DEPTH: usize = 14;
const NHEADS: usize = 10;
const HEAD_DIM: usize = 64;

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

pub trait ToFromF32 {
    fn to_f32(self) -> f32;
    fn from_f32(value: f32) -> Self;
}

impl ToFromF32 for u16 {
    fn to_f32(self) -> f32 {
        onnx::f16_to_f32(self)
    }
    fn from_f32(value: f32) -> Self {
        onnx::f32_to_f16(value)
    }
}

impl ToFromF32 for f32 {
    fn to_f32(self) -> f32 {
        self
    }
    fn from_f32(value: f32) -> Self {
        value
    }
}

pub struct Moonshine<T: TensorElement + Default> {
    onnx: Arc<onnx::Onnx>,
    encoder_session: onnx::Session,
    decoder_session: onnx::Session,
    tokenizer: tokenizers::Tokenizer,
    at_start: bool,
    accumulated_text: String,
    samples: Vec<i16>,
    phantom: PhantomData<T>,
}

impl<T: TensorElement + Default + ToFromF32> Moonshine<T> {
    pub fn new(model_folder_path: &str, executor: onnx::Executor) -> Self {
        let onnx = onnx::Onnx::new(18);
        let encoder_session = onnx.create_session(
            executor,
            onnx::OptimizationLevel::EnableAll,
            4,
            format!("{}/encoder_model.onnx", model_folder_path),
        );
        let decoder_session = onnx.create_session(
            executor,
            onnx::OptimizationLevel::EnableAll,
            4,
            format!("{}/decoder_model.onnx", model_folder_path),
        );

        // Tokenizer lives in the source directory (two levels above the variant folder)
        let model_path = Path::new(model_folder_path);
        let tokenizer_path = model_path
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("source/tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .expect("failed to load tokenizer.json");

        Self {
            onnx,
            encoder_session,
            decoder_session,
            tokenizer,
            at_start: true,
            accumulated_text: String::new(),
            samples: Vec::new(),
            phantom: PhantomData,
        }
    }

    fn transcribe_window(&self, window: &[i16]) -> String {
        let audio_len = window.len();

        // Normalize i16 → float and convert to model precision T
        let audio: Vec<T> = window
            .iter()
            .map(|&s| T::from_f32(s as f32 / 32768.0))
            .collect();
        let attention_mask = vec![1i64; audio_len];

        let audio_tensor = onnx::Value::from_slice(&self.onnx, &[1, audio_len], &audio);
        let mask_tensor = onnx::Value::from_slice(&self.onnx, &[1, audio_len], &attention_mask);

        // Encoder: audio waveform → hidden states
        let mut enc_outputs = self.encoder_session.run(
            &[
                ("input_values", &audio_tensor),
                ("attention_mask", &mask_tensor),
            ],
            &["last_hidden_state", "encoder_attention_mask"],
        );
        let enc_hidden = enc_outputs.remove(0);
        let enc_shape = enc_hidden.tensor_shape(); // [1, enc_len, 768]
        let enc_len = enc_shape[1] as usize;

        // Decoder expects int64 attention mask; encoder outputs bool — create fresh
        let dec_enc_mask =
            onnx::Value::from_slice(&self.onnx, &[1, enc_len], &vec![1i64; enc_len]);

        // Self-attention KV cache: [DEPTH, 1, NHEADS, past_len, HEAD_DIM]
        // Starts empty (past_len = 0). Use T so the element type matches the
        // model precision (float16 for f16 variant, float32 for quantised).
        let mut k_self =
            onnx::Value::zeros::<T>(&self.onnx, &[DEPTH as i64, 1, NHEADS as i64, 0, HEAD_DIM as i64]);
        let mut v_self =
            onnx::Value::zeros::<T>(&self.onnx, &[DEPTH as i64, 1, NHEADS as i64, 0, HEAD_DIM as i64]);

        // Autoregressive decoding with KV cache
        let mut tokens: Vec<i64> = Vec::new();
        let mut current_token = BOS_ID;
        for _ in 0..MAX_TOKENS {
            let input_ids = onnx::Value::from_slice(&self.onnx, &[1, 1], &[current_token]);

            let mut dec_outputs = self.decoder_session.run(
                &[
                    ("input_ids", &input_ids),
                    ("encoder_hidden_states", &enc_hidden),
                    ("encoder_attention_mask", &dec_enc_mask),
                    ("k_self", &k_self),
                    ("v_self", &v_self),
                ],
                &["logits", "out_k_self", "out_v_self"],
            );

            // logits shape: [1, 1, VOCAB_SIZE]
            let mut logits = dec_outputs[0].extract_as_f32();
            let last_logits = &mut logits[..VOCAB_SIZE];

            // Update KV cache for next step
            k_self = dec_outputs.remove(1);
            v_self = dec_outputs.remove(1);

            // Repetition penalty on previously generated tokens
            for &tok in &tokens {
                let idx = tok as usize;
                if idx < VOCAB_SIZE {
                    if last_logits[idx] > 0.0 {
                        last_logits[idx] /= REPETITION_PENALTY;
                    } else {
                        last_logits[idx] *= REPETITION_PENALTY;
                    }
                }
            }

            // No-repeat n-gram blocking
            if NO_REPEAT_NGRAM >= 2 && tokens.len() >= NO_REPEAT_NGRAM - 1 {
                let prefix = &tokens[tokens.len() - (NO_REPEAT_NGRAM - 1)..];
                for w in tokens.windows(NO_REPEAT_NGRAM) {
                    if w[..NO_REPEAT_NGRAM - 1] == *prefix {
                        let blocked = w[NO_REPEAT_NGRAM - 1] as usize;
                        if blocked < VOCAB_SIZE {
                            last_logits[blocked] = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            // Greedy argmax
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .unwrap_or(0);

            if next_token == EOS_ID {
                break;
            }
            tokens.push(next_token);
            current_token = next_token;
        }

        // Decode token IDs → text
        let token_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        self.tokenizer.decode(&token_ids, true).unwrap_or_default()
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
            self.accumulated_text =
                format!("{}{}", &self.accumulated_text[..end1], &text[end2..]);
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
