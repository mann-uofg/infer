"""A from-scratch PyTorch forward pass for Llama-architecture models.

This is the ground truth the CUDA engine is checked against. It is written directly from the
architecture, not from the Hugging Face modeling code, and is tested against Hugging Face
transformers in test_reference.py.

Shapes use T for sequence length, H for hidden size, D for head dim, and the conventions of the
checkpoint: a Linear weight is [out_features, in_features] and computes y = x @ W.T.

The whole sequence is recomputed on every call (no KV cache). That keeps the reference simple;
it only has to be right, not fast.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file


@dataclass
class Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict | None
    tie_word_embeddings: bool

    @staticmethod
    def from_json(path: str | Path) -> "Config":
        c = json.loads(Path(path).read_text())
        if c.get("model_type") != "llama":
            raise ValueError(f"unsupported model_type {c.get('model_type')!r}")
        if c.get("attention_bias") or c.get("mlp_bias"):
            raise ValueError("biases are not supported")
        if c.get("hidden_act", "silu") != "silu":
            raise ValueError("only silu is supported")
        return Config(
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_layers=c["num_hidden_layers"],
            num_heads=c["num_attention_heads"],
            num_kv_heads=c.get("num_key_value_heads", c["num_attention_heads"]),
            head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=c.get("rope_theta", 10000.0),
            rope_scaling=c.get("rope_scaling"),
            tie_word_embeddings=c.get("tie_word_embeddings", False),
        )


def load_weights(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """Loads every tensor from model.safetensors, or from all shards listed in the index."""
    model_dir = Path(model_dir)
    single = model_dir / "model.safetensors"
    if single.exists():
        return load_file(single)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weights: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index["weight_map"].values())):
        weights.update(load_file(model_dir / shard))
    return weights


def rope_inv_freq(cfg: Config) -> torch.Tensor:
    """Inverse frequency for each of the D/2 rotation pairs, as float32.

    Pair i rotates by angle position * inv_freq[i]. Llama 3 rescales the low frequencies (long
    wavelengths) by `factor` so the model can use longer contexts than it was pretrained on,
    leaves the high frequencies alone, and blends linearly in between.
    """
    d = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    s = cfg.rope_scaling
    if s is None or s.get("rope_type", s.get("type")) in (None, "default"):
        return inv_freq
    if s.get("rope_type", s.get("type")) != "llama3":
        raise ValueError(f"unsupported rope scaling {s}")

    factor = s["factor"]
    low, high = s["low_freq_factor"], s["high_freq_factor"]
    old_ctx = s["original_max_position_embeddings"]
    low_freq_wavelen = old_ctx / low
    high_freq_wavelen = old_ctx / high

    wavelen = 2 * math.pi / inv_freq
    out = inv_freq.clone()
    long_wave = wavelen > low_freq_wavelen
    out[long_wave] = inv_freq[long_wave] / factor
    middle = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    smooth = (old_ctx / wavelen[middle] - low) / (high - low)
    out[middle] = (1 - smooth) * inv_freq[middle] / factor + smooth * inv_freq[middle]
    return out


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """y = x / sqrt(mean(x^2) + eps) * weight, over the last dimension.

    The reduction is done in float32 even for bf16 input: summing thousands of squares in bf16
    (8 bits of mantissa) loses most of the precision.
    """
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(dtype)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Rotates x [T, heads, D] by position.

    Hugging Face checkpoints pair dimension i with dimension i + D/2 ("rotate half"), not
    neighbouring dimensions as in the original Meta code; the conversion script permutes the
    q and k weights to make the two equivalent. The engine must use the same pairing.
    """
    angles = positions.float()[:, None] * inv_freq[None, :]  # [T, D/2], float32
    cos = torch.cos(angles)[:, None, :].to(x.dtype)
    sin = torch.sin(angles)[:, None, :].to(x.dtype)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal grouped-query attention.

    q is [T, Hq, D]; k and v are [T, Hkv, D] with Hq a multiple of Hkv. Query head h reads
    key/value head h // (Hq / Hkv), so each K/V head serves a group of query heads.
    Returns [T, Hq, D].
    """
    t, hq, d = q.shape
    group = hq // k.shape[1]
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(d)  # [Hq, T, T]
    future = torch.ones(t, t, dtype=torch.bool, device=q.device).triu(1)
    scores = scores.masked_fill(future, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.einsum("hts,shd->thd", probs, v)


class Llama:
    """Holds the weights and runs the forward pass. Not an nn.Module on purpose: there is
    nothing to train, and plain tensors keep every step visible."""

    def __init__(
        self,
        model_dir: str | Path,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        kv_dtype: torch.dtype | None = None,
    ):
        """kv_dtype rounds keys and values to that type after RoPE, the way a KV cache stored
        in that type would. The engine keeps its cache in bf16, so comparisons against it use
        kv_dtype=torch.bfloat16."""
        model_dir = Path(model_dir)
        self.cfg = Config.from_json(model_dir / "config.json")
        self.w = {k: v.to(device=device, dtype=dtype) for k, v in load_weights(model_dir).items()}
        if self.cfg.tie_word_embeddings:
            self.w["lm_head.weight"] = self.w["model.embed_tokens.weight"]
        self.inv_freq = rope_inv_freq(self.cfg).to(device)
        self.device = device
        self.dtype = dtype
        self.kv_dtype = kv_dtype

    def layer(self, i: int, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cfg, w = self.cfg, self.w
        p = f"model.layers.{i}."
        t = x.shape[0]

        # Attention block (pre-norm, residual).
        h = rms_norm(x, w[p + "input_layernorm.weight"], cfg.rms_norm_eps)
        q = (h @ w[p + "self_attn.q_proj.weight"].T).view(t, cfg.num_heads, cfg.head_dim)
        k = (h @ w[p + "self_attn.k_proj.weight"].T).view(t, cfg.num_kv_heads, cfg.head_dim)
        v = (h @ w[p + "self_attn.v_proj.weight"].T).view(t, cfg.num_kv_heads, cfg.head_dim)
        q = apply_rope(q, positions, self.inv_freq)
        k = apply_rope(k, positions, self.inv_freq)
        if self.kv_dtype is not None:
            k = k.to(self.kv_dtype).to(x.dtype)
            v = v.to(self.kv_dtype).to(x.dtype)
        a = attention(q, k, v).reshape(t, cfg.num_heads * cfg.head_dim)
        x = x + a @ w[p + "self_attn.o_proj.weight"].T

        # MLP block: SwiGLU, down(silu(gate(h)) * up(h)).
        h = rms_norm(x, w[p + "post_attention_layernorm.weight"], cfg.rms_norm_eps)
        gate = h @ w[p + "mlp.gate_proj.weight"].T
        up = h @ w[p + "mlp.up_proj.weight"].T
        x = x + (torch.nn.functional.silu(gate) * up) @ w[p + "mlp.down_proj.weight"].T
        return x

    @torch.no_grad()
    def forward(
        self, tokens: list[int] | torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Logits [T, vocab] for every position. With return_hidden, also the residual stream
        after the embedding and after each layer (num_layers + 1 tensors of [T, H])."""
        tokens = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
        positions = torch.arange(tokens.shape[0], device=self.device)
        x = self.w["model.embed_tokens.weight"][tokens]
        hidden = [x]
        for i in range(self.cfg.num_layers):
            x = self.layer(i, x, positions)
            hidden.append(x)
        x = rms_norm(x, self.w["model.norm.weight"], self.cfg.rms_norm_eps)
        logits = (x @ self.w["lm_head.weight"].T).float()
        return (logits, hidden) if return_hidden else logits

    @torch.no_grad()
    def generate_greedy(self, prompt: list[int], n: int) -> list[int]:
        """Appends n tokens, each the argmax of the last position's logits."""
        tokens = list(prompt)
        for _ in range(n):
            tokens.append(int(self.forward(tokens)[-1].argmax()))
        return tokens[len(prompt) :]
