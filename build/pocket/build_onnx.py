"""Build ONNX variants for Kyutai Pocket TTS model.

Exports the Pocket TTS safetensors model to ONNX, then produces five
quantized variants matching the repo structure:

  f16   — FP16 weights and compute
  q8f16 — INT8 dynamic quantization (weights INT8, activations FP16 at runtime)
  q8i8  — INT8 static quantization (weights + activations INT8, calibrated)
  q4f16 — INT4 block-wise weight quantization (activations FP16)
  q4i8  — INT4 block-wise weight quantization (activations INT8)

The model is split into four ONNX components:

  text_encoder.onnx  — token IDs → text embeddings [B, T_text, 1024]
  flow_lm.onnx       — autoregressive step: text_emb + prev_latent + KV cache
                        → next_latent [B, 32] + is_eos [B, 1] + updated KV cache
  mimi_decoder.onnx   — codec latent [B, 32, T_frames] → audio [B, 1, T_samples]
  voice_encoder.onnx  — reference audio [B, 1, T_samples] → speaker conditioning
                        [B, T_frames, 1024] for voice cloning

The flow_lm export wraps the transformer backbone with explicit KV-cache
tensors (no streaming state dicts), and unrolls the single-step LSD flow
matching inline.  The mimi_decoder runs in non-streaming mode (full-sequence
decode) since TensorRT cannot handle the stateful conv padding buffers
efficiently — the caller accumulates latent frames and decodes in one shot.
The voice_encoder runs the Mimi encoder pipeline (SEANet → transformer →
downsample → speaker projection) to produce conditioning embeddings from
reference audio for voice cloning.

Requires: pocket-tts, torch, onnx, onnxruntime, safetensors, sentencepiece
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

MODEL_DIR = Path("data/pocket")
ONNX_DIR = MODEL_DIR / "onnx"
SOURCE_DIR = MODEL_DIR / "source"

CALIB_SAMPLES = 200
CALIB_SEQ_LEN = 100

# ── Model dimensions from config/b6369a24.yaml ──────────────────────────
D_MODEL = 1024         # transformer d_model
D_FF = 4096            # transformer dim_feedforward (d_model * hidden_scale=4)
NUM_HEADS = 16         # transformer num_heads
NUM_LAYERS = 6         # transformer num_layers
HEAD_DIM = D_MODEL // NUM_HEADS  # 64
LDIM = 32              # quantizer.dimension (latent / codebook dim)
FLOW_DIM = 512         # flow.dim
FLOW_DEPTH = 6         # flow.depth (res_blocks)
TEXT_VOCAB = 4001      # n_bins + 1 (4000 + padding)
MAX_PERIOD = 10_000.0  # RoPE max_period


# ═══════════════════════════════════════════════════════════════════════════
#  Utility functions (shared with other build scripts in this repo)
# ═══════════════════════════════════════════════════════════════════════════

class SyntheticCalibrationReader(CalibrationDataReader):
    """Generates synthetic inputs for INT8 static quantization calibration."""

    def __init__(
        self,
        onnx_path: str,
        num_samples: int = CALIB_SAMPLES,
        dim_overrides: dict[str, int] | None = None,
    ):
        session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.input_specs = []
        for inp in session.get_inputs():
            shape = self._resolve_shape(inp.shape, dim_overrides or {})
            self.input_specs.append((inp.name, inp.type, shape))
        self.num_samples = num_samples
        self.current = 0

    @staticmethod
    def _resolve_shape(
        raw_shape: list, dim_overrides: dict[str, int]
    ) -> list[int]:
        shape: list[int] = []
        first_symbolic_seen = False
        for d in raw_shape:
            if isinstance(d, int) and d > 0:
                shape.append(d)
            elif isinstance(d, str):
                if d in dim_overrides:
                    shape.append(dim_overrides[d])
                elif not first_symbolic_seen:
                    shape.append(1)
                    first_symbolic_seen = True
                else:
                    shape.append(CALIB_SEQ_LEN)
            else:
                shape.append(1)
        return shape

    def get_next(self):
        if self.current >= self.num_samples:
            return None
        self.current += 1
        feed = {}
        for name, dtype, shape in self.input_specs:
            if "float" in dtype:
                feed[name] = np.random.randn(*shape).astype(np.float32)
            elif "int" in dtype:
                np_dtype = np.int64 if "64" in dtype else np.int32
                if len(shape) == 1:
                    feed[name] = np.full(shape, CALIB_SEQ_LEN, dtype=np_dtype)
                else:
                    feed[name] = np.random.randint(0, 100, shape, dtype=np_dtype)
        return feed

    def rewind(self):
        self.current = 0


def save_onnx(model_proto, path: Path):
    PROTO_LIMIT = 2 * 1024**3
    if model_proto.ByteSize() < PROTO_LIMIT:
        onnx.save(model_proto, str(path))
    else:
        onnx.save_model(
            model_proto,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=path.stem + ".onnx_data",
        )


def consolidate_onnx(directory: Path):
    for onnx_file in sorted(directory.glob("*.onnx")):
        model = onnx.load(str(onnx_file))
        onnx.save(model, str(onnx_file))
        del model
    for f in directory.iterdir():
        if f.is_file() and not f.name.endswith(".onnx"):
            f.unlink()



# ═══════════════════════════════════════════════════════════════════════════
#  Load model weights
# ═══════════════════════════════════════════════════════════════════════════

print("Loading Pocket TTS model...")

from pocket_tts.models.tts_model import TTSModel  # noqa: E402

tts_model = TTSModel.load_model()
tts_model.eval()
tts_model = tts_model.to(dtype=torch.float32)

flow_lm = tts_model.flow_lm
mimi = tts_model.mimi


# ═══════════════════════════════════════════════════════════════════════════
#  Component 1: Text Encoder (embedding lookup)
# ═══════════════════════════════════════════════════════════════════════════

class TextEncoderONNX(nn.Module):
    """Token IDs → text embeddings via the LUT conditioner."""

    def __init__(self, embed: nn.Embedding):
        super().__init__()
        self.embed = embed

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, T_text] int64
        return self.embed(token_ids)  # [B, T_text, 1024]


# ═══════════════════════════════════════════════════════════════════════════
#  Component 2: Flow LM (one autoregressive step)
# ═══════════════════════════════════════════════════════════════════════════

def apply_rope_onnx(q, k, offset, max_period=MAX_PERIOD):
    """RoPE for ONNX export — no in-place ops, deterministic."""
    B, T, H, D = q.shape
    Hk = k.shape[2]
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-torch.log(torch.tensor(max_period, dtype=torch.float32)) * 2.0 / D))
    ts = torch.arange(T, device=q.device, dtype=torch.float32) + offset.to(torch.float32)
    ts = ts.view(-1, 1, 1)

    q = q.view(B, T, H, D // 2, 2)
    k = k.view(B, T, Hk, D // 2, 2)

    qr, qi = q[..., 0].float(), q[..., 1].float()
    kr, ki = k[..., 0].float(), k[..., 1].float()

    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)

    qo = torch.stack([qr * rotr - qi * roti, qr * roti + qi * rotr], dim=-1)
    ko = torch.stack([kr * rotr - ki * roti, kr * roti + ki * rotr], dim=-1)

    return qo.view(B, T, H, D), ko.view(B, T, Hk, D)


class FlowLMStepONNX(nn.Module):
    """Single autoregressive step of the Flow LM with explicit KV cache.

    Inputs:
        text_embeddings: [B, T_cond, 1024] — text + audio conditioning prefix
        prev_latent:     [B, 1, LDIM]      — previous latent (NaN for BOS)
        kv_cache:        [NUM_LAYERS, 2, B, past_len, NUM_HEADS, HEAD_DIM]
        cache_len:       scalar int64       — number of valid entries in cache
        noise:           [B, LDIM]          — pre-generated Gaussian noise

    Outputs:
        next_latent:     [B, LDIM]
        is_eos:          [B, 1] bool
        new_kv_cache:    [NUM_LAYERS, 2, B, new_len, NUM_HEADS, HEAD_DIM]
        new_cache_len:   scalar int64
    """

    def __init__(self, flow_lm_model):
        super().__init__()
        self.bos_emb = flow_lm_model.bos_emb          # [LDIM]
        self.input_linear = flow_lm_model.input_linear  # Linear(LDIM, D_MODEL)
        self.out_norm = flow_lm_model.out_norm          # LayerNorm(D_MODEL)
        self.out_eos = flow_lm_model.out_eos            # Linear(D_MODEL, 1)

        # Transformer layer weights
        self.layers = flow_lm_model.transformer.layers

        # Flow net
        self.flow_net = flow_lm_model.flow_net

    def forward(
        self,
        text_embeddings: torch.Tensor,   # [B, T_cond, D_MODEL]
        prev_latent: torch.Tensor,        # [B, 1, LDIM]
        kv_cache: torch.Tensor,           # [NUM_LAYERS, 2, B, past_len, NUM_HEADS, HEAD_DIM]
        cache_len: torch.Tensor,          # scalar int64
        noise: torch.Tensor,              # [B, LDIM]
    ):
        B = prev_latent.shape[0]

        # Replace NaN (BOS marker) with learned bos_emb
        is_nan = torch.isnan(prev_latent)
        prev_latent = torch.where(is_nan, self.bos_emb.unsqueeze(0).unsqueeze(0), prev_latent)

        # Project latent to transformer dim
        input_proj = self.input_linear(prev_latent)  # [B, 1, D_MODEL]

        # Concatenate text conditioning prefix with the single latent token
        x = torch.cat([text_embeddings, input_proj], dim=1)  # [B, T_cond+1, D_MODEL]
        T_new = x.shape[1]

        # Run through transformer layers, updating KV cache
        new_kv_layers = []
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = kv_cache[layer_idx]  # [2, B, past_len, NUM_HEADS, HEAD_DIM]

            x, new_layer_kv = self._transformer_layer_forward(
                layer, x, layer_cache, cache_len
            )
            new_kv_layers.append(new_layer_kv)

        # Apply output norm
        x = self.out_norm(x.to(torch.float32))

        # Extract the last position (the latent token output)
        transformer_out = x[:, -1, :]  # [B, D_MODEL]

        # EOS prediction
        is_eos = self.out_eos(transformer_out) > -4.0  # [B, 1]

        # Flow matching: single LSD step (s=0, t=1)
        s_time = torch.zeros(B, 1, device=noise.device, dtype=noise.dtype)
        t_time = torch.ones(B, 1, device=noise.device, dtype=noise.dtype)
        flow_dir = self.flow_net(transformer_out, s_time, t_time, noise)
        next_latent = noise + flow_dir  # [B, LDIM]

        new_kv_cache = torch.stack(new_kv_layers)  # [NUM_LAYERS, 2, B, new_len, H, D]
        new_cache_len = cache_len + T_new

        return next_latent, is_eos, new_kv_cache, new_cache_len

    def _transformer_layer_forward(self, layer, x, layer_cache, cache_len):
        """Forward one transformer layer with explicit KV cache concat."""
        B, T, D = x.shape

        # ── Self-attention ──
        x_normed = layer.norm1(x)
        projected = layer.self_attn.in_proj(x_normed)  # [B, T, 3*D]
        packed = projected.view(B, T, 3, NUM_HEADS, HEAD_DIM)
        q, k, v = torch.unbind(packed, dim=2)  # each [B, T, H, D_head]

        # RoPE with position offset = cache_len
        q, k = apply_rope_onnx(q, k, cache_len)

        # Transpose to [B, H, T, D_head] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Concat with cached K, V
        # layer_cache: [2, B, past_len, H, D_head]
        past_k = layer_cache[0].transpose(1, 2)  # [B, H, past_len, D_head]
        past_v = layer_cache[1].transpose(1, 2)

        full_k = torch.cat([past_k, k], dim=2)   # [B, H, past+T, D_head]
        full_v = torch.cat([past_v, v], dim=2)

        # Causal attention (new tokens attend to all past + each other causally)
        attn_out = F.scaled_dot_product_attention(
            q, full_k, full_v, is_causal=False,
            attn_mask=self._causal_mask(T, full_k.shape[2], q.device),
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = layer.self_attn.out_proj(attn_out)
        x = x + attn_out

        # ── Feed-forward ──
        x_normed2 = layer.norm2(x)
        ff_out = layer.linear2(F.gelu(layer.linear1(x_normed2)))
        x = x + ff_out

        # Build updated cache: concat old + new K, V
        # Store in [2, B, new_len, H, D_head] format
        new_k = full_k.transpose(1, 2)  # [B, new_len, H, D_head]
        new_v = full_v.transpose(1, 2)
        new_layer_kv = torch.stack([new_k, new_v])  # [2, B, new_len, H, D_head]

        return x, new_layer_kv

    @staticmethod
    def _causal_mask(T_q: int, T_kv: int, device) -> torch.Tensor:
        """Causal mask: each of the T_q new tokens can see all past + itself."""
        # New tokens are at positions [T_kv - T_q, ..., T_kv - 1]
        # Token at position i can attend to positions [0, ..., T_kv - T_q + i]
        row_idx = torch.arange(T_q, device=device).unsqueeze(1)    # [T_q, 1]
        col_idx = torch.arange(T_kv, device=device).unsqueeze(0)   # [1, T_kv]
        # Each query at row i can see columns [0, T_kv - T_q + i]
        mask = col_idx <= (T_kv - T_q + row_idx)
        return mask


# ── Split architecture: FlowLMMain (transformer only) + FlowLMFlow (ODE) ──

def _transformer_layer_forward(layer, x, layer_cache, cache_len):
    """Forward one transformer layer with explicit KV cache concat."""
    B, T, D = x.shape

    x_normed = layer.norm1(x)
    projected = layer.self_attn.in_proj(x_normed)  # [B, T, 3*D]
    packed = projected.view(B, T, 3, NUM_HEADS, HEAD_DIM)
    q, k, v = torch.unbind(packed, dim=2)

    q, k = apply_rope_onnx(q, k, cache_len)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    past_k = layer_cache[0].transpose(1, 2)
    past_v = layer_cache[1].transpose(1, 2)
    full_k = torch.cat([past_k, k], dim=2)
    full_v = torch.cat([past_v, v], dim=2)

    T_q = T
    T_kv = full_k.shape[2]
    row_idx = torch.arange(T_q, device=x.device).unsqueeze(1)
    col_idx = torch.arange(T_kv, device=x.device).unsqueeze(0)
    mask = col_idx <= (T_kv - T_q + row_idx)

    attn_out = F.scaled_dot_product_attention(
        q, full_k, full_v, is_causal=False, attn_mask=mask,
    )

    attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
    attn_out = layer.self_attn.out_proj(attn_out)
    x = x + attn_out

    x_normed2 = layer.norm2(x)
    ff_out = layer.linear2(F.gelu(layer.linear1(x_normed2)))
    x = x + ff_out

    new_k = full_k.transpose(1, 2)
    new_v = full_v.transpose(1, 2)
    new_layer_kv = torch.stack([new_k, new_v])

    return x, new_layer_kv


class FlowLMMainONNX(nn.Module):
    """Transformer backbone with explicit KV cache — no flow matching.

    Split architecture counterpart to FlowLMStepONNX.  This outputs the raw
    conditioning vector and EOS logit; the caller runs FlowLMFlowONNX
    separately for the ODE flow-matching step.

    This supports conditioning-only passes (empty sequence, non-empty
    text_embeddings) which is required for voice cloning prefill.

    Inputs:
        sequence:        [B, T_seq, LDIM]  — latent input (NaN=BOS, empty for conditioning)
        text_embeddings: [B, T_cond, D_MODEL] — conditioning (voice/text, empty during gen)
        kv_cache:        [NUM_LAYERS, 2, B, past_len, NUM_HEADS, HEAD_DIM]
        cache_len:       scalar int64

    Outputs:
        conditioning:    [B, D_MODEL]  — transformer output at last position
        eos_logit:       [B, 1]        — raw EOS logit (caller thresholds at -4.0)
        new_kv_cache:    [NUM_LAYERS, 2, B, new_len, NUM_HEADS, HEAD_DIM]
        new_cache_len:   scalar int64
    """

    def __init__(self, flow_lm_model):
        super().__init__()
        self.bos_emb = flow_lm_model.bos_emb
        self.input_linear = flow_lm_model.input_linear
        self.out_norm = flow_lm_model.out_norm
        self.out_eos = flow_lm_model.out_eos
        self.layers = flow_lm_model.transformer.layers

    def forward(
        self,
        sequence: torch.Tensor,          # [B, T_seq, LDIM]
        text_embeddings: torch.Tensor,    # [B, T_cond, D_MODEL]
        kv_cache: torch.Tensor,           # [NUM_LAYERS, 2, B, past_len, NUM_HEADS, HEAD_DIM]
        cache_len: torch.Tensor,          # scalar int64
    ):
        # Replace NaN (BOS marker) with learned bos_emb
        is_nan = torch.isnan(sequence)
        sequence = torch.where(is_nan, self.bos_emb.unsqueeze(0).unsqueeze(0), sequence)

        # Project latent to transformer dim
        input_proj = self.input_linear(sequence)  # [B, T_seq, D_MODEL]

        # Concatenate: text conditioning prefix + projected latent
        x = torch.cat([text_embeddings, input_proj], dim=1)
        T_new = x.shape[1]

        # Transformer with KV cache
        new_kv_layers = []
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = kv_cache[layer_idx]
            x, new_layer_kv = _transformer_layer_forward(
                layer, x, layer_cache, cache_len
            )
            new_kv_layers.append(new_layer_kv)

        # Output norm + last position
        x = self.out_norm(x.to(torch.float32))
        conditioning = x[:, -1, :]  # [B, D_MODEL]

        # Raw EOS logit (not thresholded)
        eos_logit = self.out_eos(conditioning)  # [B, 1]

        new_kv_cache = torch.stack(new_kv_layers)
        new_cache_len = cache_len + T_new

        return conditioning, eos_logit, new_kv_cache, new_cache_len


class FlowLMFlowONNX(nn.Module):
    """Flow matching ODE step — separate from transformer backbone.

    Takes the conditioning vector from FlowLMMainONNX and produces the
    flow direction used to step from noise to the next latent.

    Inputs:
        c: [B, D_MODEL]  — conditioning vector
        s: [B, 1]        — start time (0.0 for single step)
        t: [B, 1]        — end time (1.0 for single step)
        x: [B, LDIM]     — current point (noise for first step)

    Outputs:
        flow_dir: [B, LDIM]  — flow direction
    """

    def __init__(self, flow_lm_model):
        super().__init__()
        self.flow_net = flow_lm_model.flow_net

    def forward(
        self,
        c: torch.Tensor,   # [B, D_MODEL]
        s: torch.Tensor,   # [B, 1]
        t: torch.Tensor,   # [B, 1]
        x: torch.Tensor,   # [B, LDIM]
    ) -> torch.Tensor:
        return self.flow_net(c, s, t, x)  # [B, LDIM]


# ═══════════════════════════════════════════════════════════════════════════
#  Non-streaming convolution helpers (shared by mimi_decoder & voice_encoder)
# ═══════════════════════════════════════════════════════════════════════════


def non_streaming_conv1d(conv_module, x):
    """Run StreamingConv1d without streaming state — pad and convolve.

    Matches the real StreamingConv1d.forward(model_state=None) behavior:
      - pad_mode="constant" → left-pad with zeros
      - pad_mode="replicate" → left-pad with copies of the first time-step
    """
    kernel = conv_module._effective_kernel_size
    stride = conv_module._stride
    pad_left = kernel - stride
    if pad_left > 0:
        if conv_module.pad_mode == "replicate":
            # Real model: init state["previous"] = zeros, then fills with x[..., :1]
            # when first=True, then concatenates. Net effect = replicate first element.
            pad = x[..., :1].expand(*x.shape[:-1], pad_left)
            x = torch.cat([pad, x], dim=-1)
        else:
            x = F.pad(x, (pad_left, 0), mode="constant", value=0.0)
    return conv_module.conv(x)


def non_streaming_convtr1d(convtr_module, x):
    """Run StreamingConvTranspose1d without streaming state."""
    return convtr_module.convtr(x)


def non_streaming_resblock(resblock, x):
    """Run SEANetResnetBlock without streaming state."""
    from pocket_tts.modules.conv import StreamingConv1d

    v = x
    for layer in resblock.block:
        if isinstance(layer, StreamingConv1d):
            v = non_streaming_conv1d(layer, v)
        else:
            v = layer(v)
    return x + v


def mimi_transformer_layer_forward(layer, x):
    """Mimi transformer layer without streaming state.

    Uses context-windowed causal attention matching the real model's
    context=250 windowing — each position attends to at most the previous
    `context` positions (not the entire sequence).
    """
    B, T, D = x.shape
    context = layer.self_attn.context  # typically 250

    x_normed = layer.norm1(x)
    projected = layer.self_attn.in_proj(x_normed)

    num_heads = layer.self_attn.num_heads
    head_dim = D // num_heads
    packed = projected.view(B, T, 3, num_heads, head_dim)
    q, k, v = torch.unbind(packed, dim=2)

    q, k = apply_rope_onnx(
        q, k, torch.zeros(1, dtype=torch.long, device=x.device),
        max_period=layer.self_attn.rope.max_period,
    )

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Build context-windowed causal mask: each query position i attends to
    # key positions max(0, i-context+1)..i only (not the full history).
    # delta[i,j] = i - j  (how far back key j is from query i)
    positions = torch.arange(T, device=x.device)
    delta = positions[:, None] - positions[None, :]  # [T, T]
    mask = (delta >= 0) & (delta < context)
    attn_mask = torch.zeros(T, T, dtype=q.dtype, device=x.device)
    attn_mask[~mask] = float("-inf")

    attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
    attn_out = layer.self_attn.out_proj(attn_out)
    x = x + layer.layer_scale_1(attn_out)

    x_normed2 = layer.norm2(x)
    ff_out = layer.linear2(F.gelu(layer.linear1(x_normed2)))
    x = x + layer.layer_scale_2(ff_out)

    return x


# ═══════════════════════════════════════════════════════════════════════════
#  Component 3: Mimi Decoder (non-streaming, full-sequence)
# ═══════════════════════════════════════════════════════════════════════════

class MimiDecoderONNX(nn.Module):
    """Decodes accumulated latent frames to audio in one shot (non-streaming).

    Input:  latent [B, LDIM, T_frames] — accumulated codec latent frames
    Output: audio  [B, 1, T_samples]   — raw audio at 24kHz

    This runs the mimi decoder pipeline without streaming state:
      denormalize → quantizer.output_proj → upsample → decoder_transformer → SEANet decoder
    """

    def __init__(self, flow_lm_model, mimi_model):
        super().__init__()
        self.emb_std = flow_lm_model.emb_std      # [LDIM]
        self.emb_mean = flow_lm_model.emb_mean     # [LDIM]
        self.quantizer = mimi_model.quantizer       # Conv1d(LDIM, 512, 1)

        # Upsample from frame_rate to encoder_frame_rate
        # frame_rate=12.5, encoder_frame_rate=24000/120=200
        # stride = 200/12.5 = 16
        self.upsample = mimi_model.upsample

        # Decoder transformer (non-streaming)
        self.decoder_transformer_input_proj = None
        self.decoder_transformer_layers = mimi_model.decoder_transformer.transformer.layers
        self.decoder_transformer_output_proj = mimi_model.decoder_transformer.output_projs[0]
        if mimi_model.decoder_transformer.input_proj is not None:
            self.decoder_transformer_input_proj = mimi_model.decoder_transformer.input_proj

        # SEANet decoder layers
        self.decoder = mimi_model.decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: [B, LDIM, T_frames]

        # Denormalize
        # emb_std/emb_mean are [LDIM], need to broadcast over [B, LDIM, T]
        x = latent * self.emb_std.unsqueeze(0).unsqueeze(-1) + self.emb_mean.unsqueeze(0).unsqueeze(-1)

        # Quantizer output projection: Conv1d(LDIM→512, k=1)
        x = self.quantizer(x)  # [B, 512, T_frames]

        # Upsample: ConvTranspose1d stride=16 (12.5Hz → 200Hz)
        x = self.upsample.convtr.convtr(x)  # [B, 512, T_frames*16]
        # Note: non-streaming, so no partial overlap-add state needed.
        # The ConvTranspose1d with groups=dimension is depthwise, kernel=32, stride=16.
        # In non-streaming mode the full output is correct (we just have
        # kernel_size - stride = 16 extra samples at the end that we trim later
        # if needed).

        # Decoder transformer (non-streaming, no KV cache)
        # ProjectedTransformer: transpose → optional input_proj → transformer → output_proj → transpose
        z = x.transpose(1, 2)  # [B, T, 512]
        if self.decoder_transformer_input_proj is not None:
            z = self.decoder_transformer_input_proj(z)
        for layer in self.decoder_transformer_layers:
            z = mimi_transformer_layer_forward(layer, z)
        z = self.decoder_transformer_output_proj(z)
        z = z.transpose(1, 2)  # [B, 512, T]

        # SEANet decoder (non-streaming convolutions)
        out = self._seanet_decoder_forward(z)  # [B, 1, T_audio]
        return out

    def _seanet_decoder_forward(self, z):
        """Run SEANet decoder without streaming state (standard convolutions)."""
        from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
        from pocket_tts.modules.seanet import SEANetResnetBlock

        for layer in self.decoder.model:
            if isinstance(layer, StreamingConv1d):
                z = non_streaming_conv1d(layer, z)
            elif isinstance(layer, StreamingConvTranspose1d):
                z = non_streaming_convtr1d(layer, z)
            elif isinstance(layer, SEANetResnetBlock):
                z = non_streaming_resblock(layer, z)
            else:
                z = layer(z)
        return z


# ═══════════════════════════════════════════════════════════════════════════
#  Component 4: Voice Encoder (reference audio → speaker conditioning)
# ═══════════════════════════════════════════════════════════════════════════

# Mimi encoder dimensions from config
SEANET_DIM = 512          # seanet.dimension
SEANET_RATIOS = [6, 5, 4] # seanet.ratios (reversed in encoder → [4,5,6])
SEANET_HOP = 4 * 5 * 6    # =120 (product of reversed ratios)
ENCODER_FRAME_RATE = 24000 / SEANET_HOP    # =200 Hz
DOWNSAMPLE_STRIDE = int(ENCODER_FRAME_RATE / 12.5)  # =16
FRAME_SIZE = 24000 // 12  # =1920 samples per frame at 12.5 Hz (matches hop*stride)


class VoiceEncoderONNX(nn.Module):
    """Encodes reference audio into speaker conditioning embeddings for voice cloning.

    Input:  audio [B, 1, T_samples] — reference audio at 24kHz
                                      (T_samples must be a multiple of 1920)
    Output: conditioning [B, T_frames, 1024] — speaker embeddings at 12.5 Hz

    Pipeline: SEANet encoder → encoder_transformer → downsample → speaker_proj

    The output conditioning tensor is what gets concatenated with text_embeddings
    and fed into the flow_lm model for voice-cloned generation.
    """

    def __init__(self, mimi_model, speaker_proj_weight):
        super().__init__()
        self.encoder = mimi_model.encoder
        self.encoder_transformer_layers = mimi_model.encoder_transformer.transformer.layers
        self.encoder_transformer_input_proj = mimi_model.encoder_transformer.input_proj
        self.encoder_transformer_output_proj = mimi_model.encoder_transformer.output_projs[0]
        self.downsample = mimi_model.downsample
        self.speaker_proj_weight = speaker_proj_weight  # nn.Parameter [1024, 512]

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [B, 1, T_samples]

        # SEANet encoder (non-streaming)
        x = self._seanet_encoder_forward(audio)  # [B, 512, T/120]

        # Encoder transformer (non-streaming, full-sequence causal attention)
        z = x.transpose(1, 2)  # [B, T', 512]
        if self.encoder_transformer_input_proj is not None:
            z = self.encoder_transformer_input_proj(z)
        for layer in self.encoder_transformer_layers:
            z = mimi_transformer_layer_forward(layer, z)
        z = self.encoder_transformer_output_proj(z)
        x = z.transpose(1, 2)  # [B, 512, T']

        # Downsample: stride=16 (200 Hz → 12.5 Hz)
        x = non_streaming_conv1d(self.downsample.conv, x)  # [B, 512, T'/16]

        # Speaker projection: [B, T'', 512] → [B, T'', 1024]
        latents = x.transpose(1, 2).to(torch.float32)
        conditioning = F.linear(latents, self.speaker_proj_weight)

        return conditioning

    def _seanet_encoder_forward(self, z):
        """Run SEANet encoder without streaming state."""
        from pocket_tts.modules.conv import StreamingConv1d
        from pocket_tts.modules.seanet import SEANetResnetBlock

        for layer in self.encoder.model:
            if isinstance(layer, StreamingConv1d):
                z = non_streaming_conv1d(layer, z)
            elif isinstance(layer, SEANetResnetBlock):
                z = non_streaming_resblock(layer, z)
            else:
                z = layer(z)
        return z


# ═══════════════════════════════════════════════════════════════════════════
#  Step 1: Export to ONNX
# ═══════════════════════════════════════════════════════════════════════════

raw_dir = ONNX_DIR / "_raw"
raw_dir.mkdir(parents=True, exist_ok=True)

# ── 1a: Text Encoder ────────────────────────────────────────────────────
print("Exporting text_encoder...")
text_enc = TextEncoderONNX(flow_lm.conditioner.embed)
text_enc.eval()

dummy_tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

torch.onnx.export(
    text_enc,
    (dummy_tokens,),
    str(raw_dir / "text_encoder.onnx"),
    input_names=["token_ids"],
    output_names=["text_embeddings"],
    dynamic_axes={
        "token_ids": {0: "batch", 1: "text_len"},
        "text_embeddings": {0: "batch", 1: "text_len"},
    },
    opset_version=18,
    dynamo=False,
)

# ── 1b: Flow LM (single autoregressive step) ───────────────────────────
print("Exporting flow_lm...")
flow_step = FlowLMStepONNX(flow_lm)
flow_step.eval()

B = 1
T_COND = 5   # dummy text conditioning length
PAST = 3     # dummy past KV cache length

dummy_text_emb = torch.randn(B, T_COND, D_MODEL)
dummy_prev_latent = torch.full((B, 1, LDIM), float("nan"))  # BOS
dummy_kv = torch.randn(NUM_LAYERS, 2, B, PAST, NUM_HEADS, HEAD_DIM)
dummy_cache_len = torch.tensor(PAST, dtype=torch.long)
dummy_noise = torch.randn(B, LDIM)

torch.onnx.export(
    flow_step,
    (dummy_text_emb, dummy_prev_latent, dummy_kv, dummy_cache_len, dummy_noise),
    str(raw_dir / "flow_lm.onnx"),
    input_names=[
        "text_embeddings", "prev_latent", "kv_cache", "cache_len", "noise",
    ],
    output_names=[
        "next_latent", "is_eos", "new_kv_cache", "new_cache_len",
    ],
    dynamic_axes={
        "text_embeddings": {0: "batch", 1: "cond_len"},
        "prev_latent":     {0: "batch"},
        "kv_cache":        {2: "batch", 3: "past_len"},
        "noise":           {0: "batch"},
        "next_latent":     {0: "batch"},
        "is_eos":          {0: "batch"},
        "new_kv_cache":    {2: "batch", 3: "new_len"},
    },
    opset_version=18,
    dynamo=False,
)

# ── 1c: Mimi Decoder ───────────────────────────────────────────────────
print("Exporting mimi_decoder...")
mimi_dec = MimiDecoderONNX(flow_lm, mimi)
mimi_dec.eval()

# 10 latent frames at 12.5 Hz = 0.8 seconds → ~19200 audio samples
dummy_latent = torch.randn(1, LDIM, 10)

torch.onnx.export(
    mimi_dec,
    (dummy_latent,),
    str(raw_dir / "mimi_decoder.onnx"),
    input_names=["latent"],
    output_names=["audio"],
    dynamic_axes={
        "latent": {0: "batch", 2: "num_frames"},
        "audio":  {0: "batch", 2: "audio_len"},
    },
    opset_version=18,
    dynamo=False,
)

# ── 1d: Voice Encoder (reference audio → speaker conditioning) ────────
print("Exporting voice_encoder...")
voice_enc = VoiceEncoderONNX(mimi, flow_lm.speaker_proj_weight)
voice_enc.eval()

# 2 seconds of audio at 24kHz = 48000 samples (multiple of 1920)
dummy_audio = torch.randn(1, 1, 48000)

torch.onnx.export(
    voice_enc,
    (dummy_audio,),
    str(raw_dir / "voice_encoder.onnx"),
    input_names=["audio"],
    output_names=["conditioning"],
    dynamic_axes={
        "audio": {0: "batch", 2: "audio_len"},
        "conditioning": {0: "batch", 1: "num_frames"},
    },
    opset_version=18,
    dynamo=False,
)

# ── 1e: Flow LM Main (transformer only, split architecture) ──────────
print("Exporting flow_lm_main...")
flow_main = FlowLMMainONNX(flow_lm)
flow_main.eval()

dummy_seq = torch.full((B, 1, LDIM), float("nan"))  # BOS
dummy_text_emb_main = torch.randn(B, T_COND, D_MODEL)
dummy_kv_main = torch.randn(NUM_LAYERS, 2, B, PAST, NUM_HEADS, HEAD_DIM)
dummy_cache_len_main = torch.tensor(PAST, dtype=torch.long)

torch.onnx.export(
    flow_main,
    (dummy_seq, dummy_text_emb_main, dummy_kv_main, dummy_cache_len_main),
    str(raw_dir / "flow_lm_main.onnx"),
    input_names=["sequence", "text_embeddings", "kv_cache", "cache_len"],
    output_names=["conditioning", "eos_logit", "new_kv_cache", "new_cache_len"],
    dynamic_axes={
        "sequence":        {0: "batch", 1: "seq_len"},
        "text_embeddings": {0: "batch", 1: "cond_len"},
        "kv_cache":        {2: "batch", 3: "past_len"},
        "conditioning":    {0: "batch"},
        "eos_logit":       {0: "batch"},
        "new_kv_cache":    {2: "batch", 3: "new_len"},
    },
    opset_version=18,
    dynamo=False,
)

# ── 1f: Flow LM Flow (ODE step, split architecture) ──────────────────
print("Exporting flow_lm_flow...")
flow_flow = FlowLMFlowONNX(flow_lm)
flow_flow.eval()

dummy_c = torch.randn(B, D_MODEL)
dummy_s = torch.zeros(B, 1)
dummy_t = torch.ones(B, 1)
dummy_x = torch.randn(B, LDIM)

torch.onnx.export(
    flow_flow,
    (dummy_c, dummy_s, dummy_t, dummy_x),
    str(raw_dir / "flow_lm_flow.onnx"),
    input_names=["c", "s", "t", "x"],
    output_names=["flow_dir"],
    dynamic_axes={
        "c":        {0: "batch"},
        "s":        {0: "batch"},
        "t":        {0: "batch"},
        "x":        {0: "batch"},
        "flow_dir": {0: "batch"},
    },
    opset_version=18,
    dynamo=False,
)

del tts_model, flow_lm, mimi, text_enc, flow_step, mimi_dec, voice_enc
del flow_main, flow_flow
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Consolidate raw exports
print("Consolidating raw ONNX exports...")
for onnx_file in sorted(raw_dir.glob("*.onnx")):
    proto = onnx.load(str(onnx_file), load_external_data=True)
    save_onnx(proto, onnx_file)
    del proto
for f in raw_dir.iterdir():
    if f.is_file() and not f.name.endswith(".onnx"):
        f.unlink()

onnx_files = sorted(raw_dir.glob("*.onnx"))
print(f"Exported: {[f.name for f in onnx_files]}")


# ═══════════════════════════════════════════════════════════════════════════
#  Step 2: FP16 conversion
# ═══════════════════════════════════════════════════════════════════════════

print("Converting to f16...")
f16_dir = ONNX_DIR / "f16"
f16_dir.mkdir(parents=True, exist_ok=True)

# Use ORT's float16 converter — it correctly handles mixed-precision graphs
# by inserting Cast nodes at type boundaries (Range inputs, RoPE, etc.).
from onnxruntime.transformers.float16 import convert_float_to_float16  # noqa: E402

for onnx_file in onnx_files:
    model = onnx.load(str(onnx_file), load_external_data=True)
    model_fp16 = convert_float_to_float16(model, keep_io_types=False)
    save_onnx(model_fp16, f16_dir / onnx_file.name)
    del model, model_fp16


# ═══════════════════════════════════════════════════════════════════════════
#  Step 3: INT8 dynamic quantization (q8f16)
# ═══════════════════════════════════════════════════════════════════════════

print("Quantizing q8f16 (INT8 dynamic)...")
q8f16_dir = ONNX_DIR / "q8f16"
q8f16_dir.mkdir(parents=True, exist_ok=True)

for onnx_file in onnx_files:
    quantize_dynamic(
        model_input=str(onnx_file),
        model_output=str(q8f16_dir / onnx_file.name),
        per_channel=True,
        weight_type=QuantType.QInt8,
        use_external_data_format=True,
    )

consolidate_onnx(q8f16_dir)


# ═══════════════════════════════════════════════════════════════════════════
#  Step 4: INT8 static quantization (q8i8)
# ═══════════════════════════════════════════════════════════════════════════

print(f"Calibrating and quantizing q8i8 (INT8 static, {CALIB_SAMPLES} samples)...")
q8i8_dir = ONNX_DIR / "q8i8"
q8i8_dir.mkdir(parents=True, exist_ok=True)

# Model-specific dim overrides for calibration
FLOW_LM_DIM_OVERRIDES: dict[str, int] = {
    "cond_len": 10,
    "past_len": 20,
    "seq_len": 1,  # flow_lm_main: sequence is 0 or 1 tokens
}
MIMI_DIM_OVERRIDES: dict[str, int] = {
    "num_frames": 10,
}
# voice_encoder needs audio_len to be a multiple of 1920 (frame_size)
# and large enough for the full SEANet→downsample chain: ≥ 3840
VOICE_ENCODER_DIM_OVERRIDES: dict[str, int] = {
    "audio_len": 19200,  # 10 frames at 12.5Hz
}


def _overrides_for(name: str) -> dict[str, int] | None:
    if "flow_lm" in name:
        return FLOW_LM_DIM_OVERRIDES
    if "voice" in name:
        return VOICE_ENCODER_DIM_OVERRIDES
    if "mimi" in name:
        return MIMI_DIM_OVERRIDES
    return None


for onnx_file in onnx_files:
    calib_reader = SyntheticCalibrationReader(
        str(onnx_file), dim_overrides=_overrides_for(onnx_file.name)
    )
    quantize_static(
        model_input=str(onnx_file),
        model_output=str(q8i8_dir / onnx_file.name),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        use_external_data_format=True,
    )

consolidate_onnx(q8i8_dir)


# ═══════════════════════════════════════════════════════════════════════════
#  Step 5: INT4 block-wise weight quantization (q4f16)
# ═══════════════════════════════════════════════════════════════════════════

print("Quantizing q4f16 (INT4 weights)...")
q4f16_dir = ONNX_DIR / "q4f16"
q4f16_dir.mkdir(parents=True, exist_ok=True)

for onnx_file in onnx_files:
    q4 = MatMulNBitsQuantizer(
        str(onnx_file), bits=4, block_size=128, is_symmetric=True
    )
    q4.process()
    q4.model.save_model_to_file(str(q4f16_dir / onnx_file.name))
    del q4


# ═══════════════════════════════════════════════════════════════════════════
#  Step 6: INT4 weights + INT8 activation quantization (q4i8)
# ═══════════════════════════════════════════════════════════════════════════

print("Quantizing q4i8 (INT4 weights + INT8 activations)...")
q4i8_dir = ONNX_DIR / "q4i8"
q4i8_dir.mkdir(parents=True, exist_ok=True)

for onnx_file in onnx_files:
    q4i8 = MatMulNBitsQuantizer(
        str(onnx_file), bits=4, block_size=128, is_symmetric=True, accuracy_level=4
    )
    q4i8.process()
    q4i8.model.save_model_to_file(str(q4i8_dir / onnx_file.name))
    del q4i8


print("ONNX exports ready.")
