use std::{ffi::c_void, path::Path, time::Instant};
use tensorrt::{Context, Engine, Runtime};

// ── CUDA runtime FFI (cudart is already linked by the tensorrt crate) ────

unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemset(devPtr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
}

const CUDA_MEMCPY_H2D: i32 = 1;
const CUDA_MEMCPY_D2H: i32 = 2;

// ── Tiny GPU buffer helper ───────────────────────────────────────────────

struct GpuBuf {
    ptr: *mut c_void,
    size: usize,
}

impl GpuBuf {
    fn new(bytes: usize) -> Self {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { cudaMalloc(&mut ptr, bytes) };
        assert!(rc == 0, "cudaMalloc({bytes}) failed: {rc}");
        unsafe { cudaMemset(ptr, 0, bytes) };
        Self { ptr, size: bytes }
    }

    fn upload<T>(&self, data: &[T]) {
        let bytes = data.len() * size_of::<T>();
        assert!(
            bytes <= self.size,
            "upload overflow: {bytes} > {}",
            self.size
        );
        unsafe {
            cudaMemcpy(
                self.ptr,
                data.as_ptr() as *const c_void,
                bytes,
                CUDA_MEMCPY_H2D,
            );
        }
    }

    fn download<T>(&self, data: &mut [T]) {
        let bytes = data.len() * size_of::<T>();
        assert!(
            bytes <= self.size,
            "download overflow: {bytes} > {}",
            self.size
        );
        unsafe {
            cudaMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.ptr,
                bytes,
                CUDA_MEMCPY_D2H,
            );
        }
    }
}

impl Drop for GpuBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cudaFree(self.ptr) };
        }
    }
}

// ── Model constants (mirrored from onnx_impl) ───────────────────────────

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
const ENC_DIM: usize = 768;

/// Max audio length in samples (30 s at 16 kHz) — must match --maxShapes.
const MAX_AUDIO_LEN: usize = 480000;
/// Max encoder sequence length — must match --maxShapes.
const MAX_ENC_LEN: usize = 1500;

/// KV cache buffer size in bytes at maximum past_len.
const KV_MAX_BYTES: usize = DEPTH * 1 * NHEADS * MAX_TOKENS * HEAD_DIM * size_of::<f32>();

/// Compute encoder output sequence length from audio sample count.
///   enc_len = ((audio_len / 80) - 1) / 4 + 1
fn enc_len_from_audio_len(audio_len: usize) -> usize {
    ((audio_len / 80) - 1) / 4 + 1
}

fn find_longest_common_substring(s1: &str, s2: &str) -> Option<(usize, usize, usize)> {
    let (len1, len2) = (s1.len(), s2.len());
    let mut table = vec![vec![0usize; len2 + 1]; len1 + 1];
    let (mut max_len, mut end1, mut end2) = (0, 0, 0);
    let (b1, b2) = (s1.as_bytes(), s2.as_bytes());
    for i in 1..=len1 {
        for j in 1..=len2 {
            if b1[i - 1] == b2[j - 1] {
                table[i][j] = table[i - 1][j - 1] + 1;
                if table[i][j] > max_len {
                    max_len = table[i][j];
                    end1 = i - 1;
                    end2 = j - 1;
                }
            }
        }
    }
    if max_len == 0 {
        None
    } else {
        Some((max_len, end1, end2))
    }
}

// ── Moonshine TensorRT implementation ───────────────────────────────────

pub struct Moonshine {
    tokenizer: tokenizers::Tokenizer,

    // ── Streaming state ──
    at_start: bool,
    accumulated_text: String,
    samples: Vec<i16>,

    // ── CUDA stream ──
    stream: *mut c_void,

    // ── Encoder GPU buffers ──
    enc_input_values: GpuBuf, // [1, audio_len] f32
    enc_attn_mask: GpuBuf,    // [1, audio_len] i32  (TRT converts i64 → i32)
    enc_hidden: GpuBuf,       // [1, enc_len, 768] f32
    enc_out_mask: GpuBuf,     // [1, enc_len] — encoder output mask (not used by decoder)

    // ── Decoder GPU buffers ──
    dec_input_ids: GpuBuf, // [1, 1] i32
    dec_enc_mask: GpuBuf,  // [1, enc_len] i32
    dec_logits: GpuBuf,    // [1, 1, VOCAB_SIZE] f32

    // ── KV cache — double-buffered (A and B swap roles each step) ──
    kv_k_a: GpuBuf, // [14, 1, 10, past_len, 64] f32
    kv_v_a: GpuBuf,
    kv_k_b: GpuBuf,
    kv_v_b: GpuBuf,

    // ── TRT objects (contexts → engines → runtime drop order) ──
    encoder_ctx: Context,
    decoder_ctx: Context,
    _encoder_engine: Engine,
    _decoder_engine: Engine,
    _runtime: Runtime,
}

impl Moonshine {
    pub fn new(engine_folder_path: &str) -> Self {
        let runtime = Runtime::new();
        let encoder_engine = Engine::load(
            &runtime,
            &format!("{engine_folder_path}/encoder_model.engine"),
        );
        let decoder_engine = Engine::load(
            &runtime,
            &format!("{engine_folder_path}/decoder_model.engine"),
        );

        // Print I/O metadata for debugging
        for (label, eng) in [("encoder", &encoder_engine), ("decoder", &decoder_engine)] {
            println!("  {label} I/O tensors:");
            for t in eng.io_tensors() {
                let dir = if t.is_input { "IN " } else { "OUT" };
                println!("    {dir} {:30} {:?} {:?}", t.name, t.dtype, t.shape);
            }
        }

        let encoder_ctx = Context::new(&encoder_engine);
        let decoder_ctx = Context::new(&decoder_engine);

        let mut stream: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { cudaStreamCreate(&mut stream) };
        assert!(rc == 0, "cudaStreamCreate failed: {rc}");

        let f32s = size_of::<f32>();

        // ── Allocate encoder buffers ──
        let enc_input_values = GpuBuf::new(MAX_AUDIO_LEN * f32s);
        let enc_attn_mask = GpuBuf::new(MAX_AUDIO_LEN * size_of::<i64>());
        let enc_hidden = GpuBuf::new(MAX_ENC_LEN * ENC_DIM * f32s);
        let enc_out_mask = GpuBuf::new(MAX_ENC_LEN * size_of::<i64>());

        // ── Allocate decoder buffers ──
        let dec_input_ids = GpuBuf::new(size_of::<i64>());
        let dec_enc_mask = GpuBuf::new(MAX_ENC_LEN * size_of::<i64>());
        let dec_logits = GpuBuf::new(VOCAB_SIZE * f32s);

        // ── Allocate KV cache double buffer ──
        let kv_k_a = GpuBuf::new(KV_MAX_BYTES);
        let kv_v_a = GpuBuf::new(KV_MAX_BYTES);
        let kv_k_b = GpuBuf::new(KV_MAX_BYTES);
        let kv_v_b = GpuBuf::new(KV_MAX_BYTES);

        // Tokenizer: engine path is data/moonshine/engine/<platform>/<variant>,
        // tokenizer is at data/moonshine/source/tokenizer.json (3 levels up).
        let engine_path = Path::new(engine_folder_path);
        let tokenizer_path = engine_path
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("source/tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .expect("failed to load tokenizer.json");

        Self {
            tokenizer,
            at_start: true,
            accumulated_text: String::new(),
            samples: Vec::new(),
            stream,
            enc_input_values,
            enc_attn_mask,
            enc_hidden,
            enc_out_mask,
            dec_input_ids,
            dec_enc_mask,
            dec_logits,
            kv_k_a,
            kv_v_a,
            kv_k_b,
            kv_v_b,
            encoder_ctx,
            decoder_ctx,
            _encoder_engine: encoder_engine,
            _decoder_engine: decoder_engine,
            _runtime: runtime,
        }
    }

    fn transcribe_window(&mut self, window: &[i16]) -> String {
        let audio_len = window.len();
        let enc_len = enc_len_from_audio_len(audio_len);

        // ── Encoder ─────────────────────────────────────────────────────

        // Normalize i16 → f32
        let audio: Vec<f32> = window.iter().map(|&s| s as f32 / 32768.0).collect();
        let attn_mask = vec![1i64; audio_len];

        self.enc_input_values.upload(&audio);
        self.enc_attn_mask.upload(&attn_mask);

        self.encoder_ctx
            .set_input_shape("input_values", &[1, audio_len as i64]);
        self.encoder_ctx
            .set_input_shape("attention_mask", &[1, audio_len as i64]);

        self.encoder_ctx
            .set_tensor_address("input_values", self.enc_input_values.ptr);
        self.encoder_ctx
            .set_tensor_address("attention_mask", self.enc_attn_mask.ptr);
        self.encoder_ctx
            .set_tensor_address("last_hidden_state", self.enc_hidden.ptr);
        self.encoder_ctx
            .set_tensor_address("encoder_attention_mask", self.enc_out_mask.ptr);

        self.encoder_ctx.enqueue(self.stream);
        unsafe { cudaStreamSynchronize(self.stream) };

        // ── Decoder (autoregressive with KV cache) ──────────────────────

        // All-ones encoder attention mask for the decoder (i64)
        let dec_enc_mask = vec![1i64; enc_len];
        self.dec_enc_mask.upload(&dec_enc_mask);

        let mut tokens: Vec<i64> = Vec::new();
        let mut current_token = BOS_ID;
        let mut past_len: usize = 0;
        let mut kv_idx = false; // false → A is input / B is output

        for _ in 0..MAX_TOKENS {
            // Upload current token (i64)
            self.dec_input_ids.upload(&[current_token]);

            // Set all decoder input shapes (enc_len fixed, past_len grows)
            self.decoder_ctx.set_input_shape("input_ids", &[1, 1]);
            self.decoder_ctx.set_input_shape(
                "encoder_hidden_states",
                &[1, enc_len as i64, ENC_DIM as i64],
            );
            self.decoder_ctx
                .set_input_shape("encoder_attention_mask", &[1, enc_len as i64]);
            self.decoder_ctx.set_input_shape(
                "k_self",
                &[
                    DEPTH as i64,
                    1,
                    NHEADS as i64,
                    past_len as i64,
                    HEAD_DIM as i64,
                ],
            );
            self.decoder_ctx.set_input_shape(
                "v_self",
                &[
                    DEPTH as i64,
                    1,
                    NHEADS as i64,
                    past_len as i64,
                    HEAD_DIM as i64,
                ],
            );

            // Bind addresses — swap KV buffer roles based on kv_idx
            let (k_in, v_in, k_out, v_out) = if !kv_idx {
                (&self.kv_k_a, &self.kv_v_a, &self.kv_k_b, &self.kv_v_b)
            } else {
                (&self.kv_k_b, &self.kv_v_b, &self.kv_k_a, &self.kv_v_a)
            };

            self.decoder_ctx
                .set_tensor_address("input_ids", self.dec_input_ids.ptr);
            self.decoder_ctx
                .set_tensor_address("encoder_hidden_states", self.enc_hidden.ptr);
            self.decoder_ctx
                .set_tensor_address("encoder_attention_mask", self.dec_enc_mask.ptr);
            self.decoder_ctx.set_tensor_address("k_self", k_in.ptr);
            self.decoder_ctx.set_tensor_address("v_self", v_in.ptr);
            self.decoder_ctx
                .set_tensor_address("logits", self.dec_logits.ptr);
            self.decoder_ctx
                .set_tensor_address("out_k_self", k_out.ptr);
            self.decoder_ctx
                .set_tensor_address("out_v_self", v_out.ptr);

            // Infer
            self.decoder_ctx.enqueue(self.stream);
            unsafe { cudaStreamSynchronize(self.stream) };

            // Download logits to CPU for post-processing
            let mut logits = vec![0f32; VOCAB_SIZE];
            self.dec_logits.download(&mut logits);

            // Repetition penalty on previously generated tokens
            for &tok in &tokens {
                let idx = tok as usize;
                if idx < VOCAB_SIZE {
                    if logits[idx] > 0.0 {
                        logits[idx] /= REPETITION_PENALTY;
                    } else {
                        logits[idx] *= REPETITION_PENALTY;
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
                            logits[blocked] = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            // Greedy argmax
            let next_token = logits
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
            past_len += 1;
            kv_idx = !kv_idx;
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

    pub fn warmup(&mut self, audio: &[i16]) {
        // Run a full inference pass to trigger TRT JIT compilation
        for chunk in audio.chunks(8000) {
            self.run_chunk(chunk);
        }

        // Reset all streaming state so the real run starts clean
        self.at_start = true;
        self.accumulated_text.clear();
        self.samples.clear();
    }
}

impl Drop for Moonshine {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe { cudaStreamDestroy(self.stream) };
        }
    }
}
