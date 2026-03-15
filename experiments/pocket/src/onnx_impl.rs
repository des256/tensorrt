use {
    onnx::TensorElement,
    std::{marker::PhantomData, path::Path, sync::Arc},
};

// Model dimensions (from config/b6369a24.yaml)
const D_MODEL: usize = 1024;
const NUM_HEADS: usize = 16;
const NUM_LAYERS: usize = 6;
const HEAD_DIM: usize = D_MODEL / NUM_HEADS; // 64
const LDIM: usize = 32;
const MAX_GEN_STEPS: usize = 500;
const TEMPERATURE: f32 = 0.7;
const EOS_THRESHOLD: f32 = -4.0;
const FRAMES_AFTER_EOS: usize = 3;

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

/// Pocket TTS inference using ONNX Runtime (split architecture).
///
/// Uses the split model architecture matching Actor:
///   text_encoder    — token IDs → text embeddings
///   flow_lm_main    — transformer backbone with KV cache
///   flow_lm_flow    — flow matching ODE step
///   mimi_decoder    — latent frames → audio waveform
pub struct Pocket<T: TensorElement + Default> {
    onnx: Arc<onnx::Onnx>,
    text_encoder_session: onnx::Session,
    flow_lm_main_session: onnx::Session,
    flow_lm_flow_session: onnx::Session,
    mimi_decoder_session: onnx::Session,
    tokenizer: tokenizers::Tokenizer,

    // Generation state (set by start(), consumed by step())
    kv_cache: Option<onnx::Value>,
    cache_len: i64,
    prev_latent: Option<Vec<T>>, // None = BOS (NaN), Some = previous latent
    latents: Vec<Vec<f32>>,      // accumulated latent frames as f32
    audio_output: Option<Vec<i16>>,
    eos_step: Option<usize>,
    done: bool,
    rng: Xorshift64,

    phantom: PhantomData<T>,
}

impl<T: TensorElement + Default + ToFromF32> Pocket<T> {
    pub fn new(model_folder_path: &str, executor: onnx::Executor) -> Self {
        let onnx = onnx::Onnx::new(18);

        let text_encoder_session = onnx.create_session(
            executor,
            onnx::OptimizationLevel::EnableAll,
            4,
            format!("{}/text_encoder.onnx", model_folder_path),
        );
        let flow_lm_main_session = onnx.create_session(
            executor,
            onnx::OptimizationLevel::EnableAll,
            4,
            format!("{}/flow_lm_main.onnx", model_folder_path),
        );
        let flow_lm_flow_session = onnx.create_session(
            executor,
            onnx::OptimizationLevel::EnableAll,
            4,
            format!("{}/flow_lm_flow.onnx", model_folder_path),
        );
        let mimi_decoder_session = onnx.create_session(
            executor,
            onnx::OptimizationLevel::EnableAll,
            4,
            format!("{}/mimi_decoder.onnx", model_folder_path),
        );

        // Tokenizer lives in source/ (two levels above the variant folder)
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
            text_encoder_session,
            flow_lm_main_session,
            flow_lm_flow_session,
            mimi_decoder_session,
            tokenizer,
            kv_cache: None,
            cache_len: 0,
            prev_latent: None,
            latents: Vec::new(),
            audio_output: None,
            eos_step: None,
            done: true,
            rng: Xorshift64::new(42),
            phantom: PhantomData,
        }
    }

    /// Run a conditioning pass through flow_lm_main (no latent input).
    ///
    /// Feeds `text_embeddings` through the transformer with an empty sequence,
    /// updating the KV cache. Used for voice prefill and text conditioning.
    fn conditioning_pass(&mut self, text_embeddings: &onnx::Value) {
        let empty_seq: Vec<T> = Vec::new();
        let empty_seq_tensor =
            onnx::Value::from_slice(&self.onnx, &[1, 0, LDIM], &empty_seq);
        let cache_len_tensor =
            onnx::Value::from_slice(&self.onnx, &[1], &[self.cache_len]);
        let kv = self.kv_cache.as_ref().unwrap();

        let mut outputs = self.flow_lm_main_session.run(
            &[
                ("sequence", &empty_seq_tensor),
                ("text_embeddings", text_embeddings),
                ("kv_cache", kv),
                ("cache_len", &cache_len_tensor),
            ],
            &["conditioning", "eos_logit", "new_kv_cache", "new_cache_len"],
        );

        // Only need the updated KV cache from conditioning passes
        let _conditioning = outputs.remove(0);
        let _eos_logit = outputs.remove(0);
        let new_kv_cache = outputs.remove(0);
        let new_cache_len_val = outputs.remove(0);

        self.kv_cache = Some(new_kv_cache);
        self.cache_len = new_cache_len_val.extract_tensor::<i64>()[0];
    }

    /// Prepare for generation: tokenize text, run conditioning passes, init state.
    ///
    /// `voice` is a flat f32 slice of shape [1, T_frames, 1024] from the voice
    /// encoder. Pass an empty slice to generate without voice conditioning.
    pub fn start(&mut self, voice: &[f32], text: &str) {
        let text = prepare_text(text);
        let encoding = self.tokenizer.encode(text, false).expect("tokenize failed");
        let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let num_tokens = token_ids.len();

        // Run text_encoder: token_ids [1, T] → text_embeddings [1, T, 1024]
        let token_tensor =
            onnx::Value::from_slice(&self.onnx, &[1, num_tokens], &token_ids);
        let enc_outputs = self.text_encoder_session.run(
            &[("token_ids", &token_tensor)],
            &["text_embeddings"],
        );
        let text_embeddings_f32 = enc_outputs[0].extract_as_f32();

        // Init empty KV cache: [NUM_LAYERS, 2, 1, 0, NUM_HEADS, HEAD_DIM]
        self.kv_cache = Some(onnx::Value::zeros::<T>(
            &self.onnx,
            &[NUM_LAYERS as i64, 2, 1, 0, NUM_HEADS as i64, HEAD_DIM as i64],
        ));
        self.cache_len = 0;

        // Voice conditioning pass (if voice provided)
        if !voice.is_empty() {
            let t_frames = voice.len() / D_MODEL;
            let voice_t: Vec<T> = voice.iter().map(|&v| T::from_f32(v)).collect();
            let voice_tensor =
                onnx::Value::from_slice(&self.onnx, &[1, t_frames, D_MODEL], &voice_t);
            self.conditioning_pass(&voice_tensor);
        }

        // Text conditioning pass
        let text_t: Vec<T> = text_embeddings_f32.iter().map(|&v| T::from_f32(v)).collect();
        let text_tensor =
            onnx::Value::from_slice(&self.onnx, &[1, num_tokens, D_MODEL], &text_t);
        self.conditioning_pass(&text_tensor);

        self.prev_latent = None; // BOS
        self.latents = Vec::new();
        self.audio_output = None;
        self.eos_step = None;
        self.done = false;
    }

    /// Run one autoregressive step.
    ///
    /// Returns `Some(audio_chunk)` while generating (may be empty during latent
    /// accumulation), or `None` when generation is complete.
    pub fn step(&mut self) -> Option<&[i16]> {
        if self.done {
            return None;
        }

        // If we already decoded audio, return it once then signal done
        if self.audio_output.is_some() {
            self.done = true;
            return Some(self.audio_output.as_ref().unwrap());
        }

        // Build sequence tensor [1, 1, LDIM]
        let sequence_tensor = match &self.prev_latent {
            None => {
                // BOS: fill with NaN
                let nan_data: Vec<T> = vec![T::from_f32(f32::NAN); LDIM];
                onnx::Value::from_slice(&self.onnx, &[1, 1, LDIM], &nan_data)
            }
            Some(data) => onnx::Value::from_slice(&self.onnx, &[1, 1, LDIM], data),
        };

        // Empty text_embeddings for generation steps (conditioning is in KV cache)
        let empty_text: Vec<T> = Vec::new();
        let empty_text_tensor =
            onnx::Value::from_slice(&self.onnx, &[1, 0, D_MODEL], &empty_text);

        let cache_len_tensor =
            onnx::Value::from_slice(&self.onnx, &[1], &[self.cache_len]);
        let kv = self.kv_cache.as_ref().unwrap();

        // Run flow_lm_main: get conditioning vector + eos_logit + updated KV cache
        let mut main_outputs = self.flow_lm_main_session.run(
            &[
                ("sequence", &sequence_tensor),
                ("text_embeddings", &empty_text_tensor),
                ("kv_cache", kv),
                ("cache_len", &cache_len_tensor),
            ],
            &["conditioning", "eos_logit", "new_kv_cache", "new_cache_len"],
        );

        let conditioning = main_outputs.remove(0);
        let eos_logit_val = main_outputs.remove(0);
        let new_kv_cache = main_outputs.remove(0);
        let new_cache_len_val = main_outputs.remove(0);

        // EOS check: raw logit > -4.0
        let eos_logit = eos_logit_val.extract_as_f32()[0];
        let is_eos = eos_logit > EOS_THRESHOLD;

        // Update KV cache state
        self.kv_cache = Some(new_kv_cache);
        self.cache_len = new_cache_len_val.extract_tensor::<i64>()[0];

        // Flow matching: single LSD step (s=0, t=1)
        let temp_scale = TEMPERATURE.sqrt();
        let noise_f32: Vec<f32> = (0..LDIM / 2)
            .flat_map(|_| {
                let (n1, n2) = self.rng.normal_pair();
                [n1 * temp_scale, n2 * temp_scale]
            })
            .collect();

        let noise_t: Vec<T> = noise_f32.iter().map(|&v| T::from_f32(v)).collect();
        let noise_tensor = onnx::Value::from_slice(&self.onnx, &[1, LDIM], &noise_t);

        let s_tensor =
            onnx::Value::from_slice(&self.onnx, &[1, 1], &[T::from_f32(0.0)]);
        let t_tensor =
            onnx::Value::from_slice(&self.onnx, &[1, 1], &[T::from_f32(1.0)]);

        let flow_outputs = self.flow_lm_flow_session.run(
            &[
                ("c", &conditioning),
                ("s", &s_tensor),
                ("t", &t_tensor),
                ("x", &noise_tensor),
            ],
            &["flow_dir"],
        );

        let flow_dir_f32 = flow_outputs[0].extract_as_f32();

        // Compute next latent = noise + flow_dir
        let latent_f32: Vec<f32> = noise_f32
            .iter()
            .zip(flow_dir_f32.iter())
            .map(|(&n, &fd)| n + fd)
            .collect();

        // Store for next step
        let latent_t: Vec<T> = latent_f32.iter().map(|&v| T::from_f32(v)).collect();
        self.prev_latent = Some(latent_t);
        self.latents.push(latent_f32);

        // EOS handling: continue for FRAMES_AFTER_EOS after first EOS
        if is_eos && self.eos_step.is_none() {
            self.eos_step = Some(self.latents.len());
        }
        let should_stop = self.latents.len() >= MAX_GEN_STEPS
            || self.eos_step.is_some_and(|es| self.latents.len() >= es + FRAMES_AFTER_EOS);
        if should_stop {
            self.audio_output = Some(self.decode_audio());
            // Return empty slice this step; next step returns the audio
            return Some(&[]);
        }

        Some(&[])
    }

    /// Decode accumulated latents to audio using mimi_decoder.
    fn decode_audio(&self) -> Vec<i16> {
        let num_frames = self.latents.len();
        if num_frames == 0 {
            return Vec::new();
        }

        // Build latent tensor [1, LDIM, num_frames] in column-major order
        // latents[frame][dim] → tensor[0][dim][frame]
        let mut latent_data: Vec<T> = vec![T::default(); LDIM * num_frames];
        for (frame_idx, latent) in self.latents.iter().enumerate() {
            for (dim_idx, &val) in latent.iter().enumerate() {
                latent_data[dim_idx * num_frames + frame_idx] = T::from_f32(val);
            }
        }

        let latent_tensor =
            onnx::Value::from_slice(&self.onnx, &[1, LDIM, num_frames], &latent_data);

        let outputs = self.mimi_decoder_session.run(
            &[("latent", &latent_tensor)],
            &["audio"],
        );

        // audio: [1, 1, T_samples] as f32
        let audio_f32 = outputs[0].extract_as_f32();

        // Convert f32 audio → i16 PCM
        audio_f32
            .iter()
            .map(|&s| {
                let clamped = s.clamp(-1.0, 1.0);
                (clamped * 32767.0) as i16
            })
            .collect()
    }
}

/// Simple text normalization matching Python's prepare_text_prompt().
fn prepare_text(text: &str) -> String {
    let mut t = text
        .trim()
        .replace('\n', " ")
        .replace('\r', " ")
        .replace("  ", " ");

    if t.is_empty() {
        return t;
    }

    // Uppercase first letter
    let mut chars = t.chars();
    if let Some(first) = chars.next() {
        t = first.to_uppercase().to_string() + chars.as_str();
    }

    // Ensure ends with punctuation
    if t.ends_with(|c: char| c.is_alphanumeric()) {
        t.push('.');
    }

    // Pad short texts
    if t.split_whitespace().count() < 5 {
        t = format!("        {}", t);
    }

    t
}

/// Xorshift64 PRNG with Box-Muller normal generation.
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn next_f32_open(&mut self) -> f32 {
        // (0, 1) open interval
        ((self.next_u64() >> 40) as f32 + 0.5) / 16777216.0
    }

    fn normal_pair(&mut self) -> (f32, f32) {
        let u1 = self.next_f32_open();
        let u2 = self.next_f32_open();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }
}
