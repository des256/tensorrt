use super::*;
use std::{ffi::c_void, time::Instant};
use tensorrt::{Context, Engine, Runtime};

// ── CUDA runtime FFI (cudart is already linked by the tensorrt crate) ────

unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: *mut c_void,
    ) -> i32;
    fn cudaMemset(devPtr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
}

// Parakeet CUDA kernels (compiled from src/kernels/argmax.cu)
unsafe extern "C" {
    fn tdt_argmax(
        logits: *const f32,
        result: *mut i32,
        stream: *mut c_void,
        vocab_size: i32,
        num_durations: i32,
    );
}

const CUDA_MEMCPY_H2D: i32 = 1;
const CUDA_MEMCPY_D2H: i32 = 2;
const CUDA_MEMCPY_D2D: i32 = 3;

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

    fn copy_from(&self, src: &GpuBuf, bytes: usize) {
        assert!(bytes <= self.size && bytes <= src.size);
        unsafe { cudaMemcpy(self.ptr, src.ptr, bytes, CUDA_MEMCPY_D2D) };
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
const WINDOW_SIZE: usize = 200;
const WINDOW_SHIFT: usize = 20;
const MEL_SIZE: usize = 128;
const ENCODER_OUTPUT_DIM: usize = 1024;
const BLANK_ID: i64 = 8192;
const VOCAB_SIZE: usize = 8193;
const NUM_DURATIONS: usize = 5;
const TDT_DURATIONS: [usize; 5] = [0, 1, 2, 3, 4];
const DECODER_STATE_DIM: usize = 640;
const MAX_SYMBOLS_PER_STEP: usize = 16;

/// Max encoder input frames — must match --maxShapes used in build_engines.
const MAX_ENC_INPUT_FRAMES: usize = 1500;

/// Logit vector length: VOCAB_SIZE + NUM_DURATIONS = 8198.
const LOGIT_DIM: usize = VOCAB_SIZE + NUM_DURATIONS;

// ── State dimension helpers ──────────────────────────────────────────────

const STATE_ELEMS: usize = 2 * 1 * DECODER_STATE_DIM; // [2, 1, 640]
const STATE_BYTES: usize = STATE_ELEMS * size_of::<f32>();

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

// ── Parakeet TensorRT implementation ─────────────────────────────────────

pub struct Parakeet {
    // CPU-side helpers
    feature_extractor: FeatureExtractor,
    tokenizer: Tokenizer,

    // ── Streaming state ──
    at_start: bool,
    accumulated_text: String,
    samples: Vec<i16>,
    last_token: i64,

    // ── CUDA stream (raw pointer — freed in Drop impl before fields) ──
    stream: *mut c_void,

    // ── GPU buffers (dropped before TRT objects) ──
    enc_audio_signal: GpuBuf,  // [1, 128, MAX_ENC_INPUT_FRAMES] f32
    enc_length: GpuBuf,        // [1] i32  (TRT converts ONNX int64→int32)
    enc_outputs: GpuBuf,       // [1, 1024, MAX_ENC_INPUT_FRAMES] f32
    enc_lengths: GpuBuf,       // [1] i32
    enc_transposed: GpuBuf,    // [MAX_ENC_INPUT_FRAMES, 1024] f32 — transposed encoder output
    dec_targets: GpuBuf,       // [1, 1] i32
    dec_tgt_len: GpuBuf,      // [1] i32
    dec_in_s1: GpuBuf,        // [2, 1, 640] f32
    dec_in_s2: GpuBuf,        // [2, 1, 640] f32
    dec_out_logits: GpuBuf,   // [1, 1, 1, 8198] f32
    dec_out_predlen: GpuBuf,  // [1] i32
    dec_out_s1: GpuBuf,       // [2, 1, 640] f32
    dec_out_s2: GpuBuf,       // [2, 1, 640] f32
    saved_s1: GpuBuf,         // [2, 1, 640] f32 — rollback snapshot
    saved_s2: GpuBuf,         // [2, 1, 640] f32
    argmax_result: GpuBuf,    // [3] i32 — {token, duration_idx, is_blank}

    // ── TRT objects (dropped last: contexts before engines before runtime) ──
    encoder_ctx: Context,
    decoder_ctx: Context,
    _encoder_engine: Engine,
    _decoder_engine: Engine,
    _runtime: Runtime,
}

impl Parakeet {
    pub fn new(engine_folder_path: &str) -> Self {
        let runtime = Runtime::new();
        let encoder_engine = Engine::load(
            &runtime,
            &format!("{engine_folder_path}/encoder-model.engine"),
        );
        let decoder_engine = Engine::load(
            &runtime,
            &format!("{engine_folder_path}/decoder_joint-model.engine"),
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
        let i32s = size_of::<i32>();

        // ── Allocate encoder buffers ──
        let enc_audio_signal = GpuBuf::new(1 * MEL_SIZE * MAX_ENC_INPUT_FRAMES * f32s);
        let enc_length = GpuBuf::new(1 * i32s);
        let enc_outputs = GpuBuf::new(1 * ENCODER_OUTPUT_DIM * MAX_ENC_INPUT_FRAMES * f32s);
        let enc_lengths = GpuBuf::new(1 * i32s);

        // ── Allocate transposed encoder output buffer ──
        let enc_transposed = GpuBuf::new(MAX_ENC_INPUT_FRAMES * ENCODER_OUTPUT_DIM * f32s);

        // ── Allocate decoder buffers ──
        let dec_targets = GpuBuf::new(1 * 1 * i32s);
        let dec_tgt_len = GpuBuf::new(1 * i32s);
        let dec_in_s1 = GpuBuf::new(STATE_BYTES);
        let dec_in_s2 = GpuBuf::new(STATE_BYTES);
        let dec_out_logits = GpuBuf::new(1 * 1 * 1 * LOGIT_DIM * f32s);
        let dec_out_predlen = GpuBuf::new(1 * i32s);
        let dec_out_s1 = GpuBuf::new(STATE_BYTES);
        let dec_out_s2 = GpuBuf::new(STATE_BYTES);
        let saved_s1 = GpuBuf::new(STATE_BYTES);
        let saved_s2 = GpuBuf::new(STATE_BYTES);
        let argmax_result = GpuBuf::new(3 * i32s);

        // ── Bind decoder addresses (shapes are fixed) ──
        decoder_ctx.set_input_shape("encoder_outputs", &[1, ENCODER_OUTPUT_DIM as i64, 1]);
        decoder_ctx.set_input_shape("targets", &[1, 1]);
        decoder_ctx.set_input_shape("target_length", &[1]);
        decoder_ctx.set_input_shape("input_states_1", &[2, 1, DECODER_STATE_DIM as i64]);
        decoder_ctx.set_input_shape("input_states_2", &[2, 1, DECODER_STATE_DIM as i64]);

        decoder_ctx.set_tensor_address("encoder_outputs", enc_transposed.ptr);
        decoder_ctx.set_tensor_address("targets", dec_targets.ptr);
        decoder_ctx.set_tensor_address("target_length", dec_tgt_len.ptr);
        decoder_ctx.set_tensor_address("input_states_1", dec_in_s1.ptr);
        decoder_ctx.set_tensor_address("input_states_2", dec_in_s2.ptr);
        decoder_ctx.set_tensor_address("outputs", dec_out_logits.ptr);
        decoder_ctx.set_tensor_address("prednet_lengths", dec_out_predlen.ptr);
        decoder_ctx.set_tensor_address("output_states_1", dec_out_s1.ptr);
        decoder_ctx.set_tensor_address("output_states_2", dec_out_s2.ptr);

        // Upload the constant target_length = 1
        dec_tgt_len.upload(&[1i32]);

        Self {
            feature_extractor: FeatureExtractor::new(),
            tokenizer: Tokenizer::new(),
            at_start: true,
            accumulated_text: String::new(),
            samples: Vec::new(),
            last_token: BLANK_ID,
            stream,
            enc_audio_signal,
            enc_length,
            enc_outputs,
            enc_lengths,
            enc_transposed,
            dec_targets,
            dec_tgt_len,
            dec_in_s1,
            dec_in_s2,
            dec_out_logits,
            dec_out_predlen,
            dec_out_s1,
            dec_out_s2,
            saved_s1,
            saved_s2,
            argmax_result,
            encoder_ctx,
            decoder_ctx,
            _encoder_engine: encoder_engine,
            _decoder_engine: decoder_engine,
            _runtime: runtime,
        }
    }

    // ── Run encoder on GPU, transpose output into enc_transposed, return enc_time

    fn run_encoder(&mut self, features: &[f32]) -> usize {
        let num_frames = features.len() / MEL_SIZE;

        // Transpose [num_frames, MEL_SIZE] → [1, MEL_SIZE, num_frames] (row-major)
        let mut transposed = vec![0f32; MEL_SIZE * num_frames];
        for frame in 0..num_frames {
            for bin in 0..MEL_SIZE {
                transposed[bin * num_frames + frame] = features[frame * MEL_SIZE + bin];
            }
        }

        // Upload inputs
        self.enc_audio_signal.upload(&transposed);
        self.enc_length.upload(&[num_frames as i32]);

        // Set dynamic input shapes
        self.encoder_ctx
            .set_input_shape("audio_signal", &[1, MEL_SIZE as i64, num_frames as i64]);
        self.encoder_ctx.set_input_shape("length", &[1]);

        // Bind addresses (must rebind after shape change)
        self.encoder_ctx
            .set_tensor_address("audio_signal", self.enc_audio_signal.ptr);
        self.encoder_ctx
            .set_tensor_address("length", self.enc_length.ptr);
        self.encoder_ctx
            .set_tensor_address("outputs", self.enc_outputs.ptr);
        self.encoder_ctx
            .set_tensor_address("encoded_lengths", self.enc_lengths.ptr);

        // Infer
        self.encoder_ctx.enqueue(self.stream);
        unsafe { cudaStreamSynchronize(self.stream) };

        // Read encoded_lengths to know the actual output time dimension
        let mut enc_len = [0i32; 1];
        self.enc_lengths.download(&mut enc_len);
        let enc_time = enc_len[0] as usize;

        // Download encoder output [1, 1024, enc_time], transpose to [enc_time, 1024],
        // and upload into enc_transposed so the decoder can index directly on GPU.
        let total = ENCODER_OUTPUT_DIM * enc_time;
        let mut enc_output = vec![0f32; total];
        unsafe {
            cudaMemcpy(
                enc_output.as_mut_ptr() as *mut c_void,
                self.enc_outputs.ptr,
                total * size_of::<f32>(),
                CUDA_MEMCPY_D2H,
            );
        }

        let mut encoded = vec![0f32; enc_time * ENCODER_OUTPUT_DIM];
        for frame in 0..enc_time {
            for dim in 0..ENCODER_OUTPUT_DIM {
                encoded[frame * ENCODER_OUTPUT_DIM + dim] = enc_output[dim * enc_time + frame];
            }
        }

        self.enc_transposed.upload(&encoded);
        enc_time
    }

    // ── Run one decoder step, argmax on GPU, return (token, duration_idx, is_blank)
    //
    // Everything is queued async on self.stream with a single sync at the end.

    fn run_decoder_step(&mut self, target_token: i32) -> (i64, usize, bool) {
        let token = [target_token];
        let mut result = [0i32; 3];

        unsafe {
            // 1. Upload target token (async H2D on stream)
            cudaMemcpyAsync(
                self.dec_targets.ptr,
                token.as_ptr() as *const c_void,
                size_of::<i32>(),
                CUDA_MEMCPY_H2D,
                self.stream,
            );

            // 2. Decoder inference
            self.decoder_ctx.enqueue(self.stream);

            // 3. Argmax kernel
            tdt_argmax(
                self.dec_out_logits.ptr as *const f32,
                self.argmax_result.ptr as *mut i32,
                self.stream,
                VOCAB_SIZE as i32,
                NUM_DURATIONS as i32,
            );

            // 4. Copy output states → input states (async D2D on stream)
            cudaMemcpyAsync(
                self.dec_in_s1.ptr,
                self.dec_out_s1.ptr,
                STATE_BYTES,
                CUDA_MEMCPY_D2D,
                self.stream,
            );
            cudaMemcpyAsync(
                self.dec_in_s2.ptr,
                self.dec_out_s2.ptr,
                STATE_BYTES,
                CUDA_MEMCPY_D2D,
                self.stream,
            );

            // 5. Download argmax result (async D2H on stream)
            cudaMemcpyAsync(
                result.as_mut_ptr() as *mut c_void,
                self.argmax_result.ptr,
                3 * size_of::<i32>(),
                CUDA_MEMCPY_D2H,
                self.stream,
            );

            // 6. Single sync — wait for everything above
            cudaStreamSynchronize(self.stream);
        }

        (result[0] as i64, result[1] as usize, result[2] != 0)
    }

    // ── Save / restore decoder states entirely on GPU (D2D) ──────────────

    fn save_decoder_states(&self) {
        unsafe {
            cudaMemcpyAsync(self.saved_s1.ptr, self.dec_in_s1.ptr, STATE_BYTES, CUDA_MEMCPY_D2D, self.stream);
            cudaMemcpyAsync(self.saved_s2.ptr, self.dec_in_s2.ptr, STATE_BYTES, CUDA_MEMCPY_D2D, self.stream);
        }
    }

    fn restore_decoder_states(&self) {
        unsafe {
            cudaMemcpyAsync(self.dec_in_s1.ptr, self.saved_s1.ptr, STATE_BYTES, CUDA_MEMCPY_D2D, self.stream);
            cudaMemcpyAsync(self.dec_in_s2.ptr, self.saved_s2.ptr, STATE_BYTES, CUDA_MEMCPY_D2D, self.stream);
        }
    }

    fn reset_decoder_states(&self) {
        let zeros = vec![0f32; STATE_ELEMS];
        self.dec_in_s1.upload(&zeros);
        self.dec_in_s2.upload(&zeros);
    }

    // ── Transcribe one window ────────────────────────────────────────────

    pub fn transcribe_window(&mut self, window: &[i16]) -> String {
        self.reset_decoder_states();
        self.last_token = BLANK_ID;

        let features = self.feature_extractor.extract_features(window);
        let num_frames = self.run_encoder(&features);

        let mut tokens = Vec::new();
        let mut time_index: usize = 0;
        let mut symbols_added: usize = 0;

        while time_index < num_frames {
            if symbols_added >= MAX_SYMBOLS_PER_STEP {
                time_index += 1;
                symbols_added = 0;
                continue;
            }

            self.save_decoder_states();

            // Point decoder's encoder_outputs directly into the transposed GPU buffer
            let offset = time_index * ENCODER_OUTPUT_DIM * size_of::<f32>();
            let frame_ptr = unsafe { self.enc_transposed.ptr.byte_add(offset) };
            self.decoder_ctx
                .set_tensor_address("encoder_outputs", frame_ptr);

            let (predicted_token, duration_index, is_blank) =
                self.run_decoder_step(self.last_token as i32);
            let mut duration = TDT_DURATIONS[duration_index];

            if is_blank {
                self.restore_decoder_states();
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

        self.tokenizer.tokenize(&tokens)
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

    pub fn warmup(&mut self, audio: &[i16]) {
        // Run a full inference pass to trigger TRT JIT compilation
        for chunk in audio.chunks(8000) {
            self.run_chunk(chunk);
        }

        // Reset all streaming state so the real run starts clean
        self.at_start = true;
        self.accumulated_text.clear();
        self.samples.clear();
        self.last_token = BLANK_ID;
        self.reset_decoder_states();
    }
}

impl Drop for Parakeet {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe { cudaStreamDestroy(self.stream) };
        }
    }
}
