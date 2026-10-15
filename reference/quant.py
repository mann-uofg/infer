"""Weight-only quantization, the same rule as the engine (src/kernels/quant.cu).

Each function returns the dequantized weights in float32, which is what the reference uses: the
engine's quantized kernels and this reference should then agree up to floating-point
summation order.

int8: one scale per row, scale = max|w| / 127, q = round(w / scale) in [-127, 127].
int4: one scale per `group` weights of a row (32, 64 or 128). The value m with the largest magnitude maps to -8,
      scale = m / -8, q = round(w / scale) clamped to [-8, 7]; the scale is stored in fp16.
Rounding is half-to-even in both (np.rint and CUDA's rintf).

This runs in numpy on the CPU on purpose. PyTorch on CUDA divides a tensor by a Python scalar
by multiplying with its reciprocal, which is sometimes one ulp off a true division; that
changed a few int8 scales and flipped the rounding of some weights relative to the kernel.
numpy float32 division is a true IEEE division, like '/' in the kernel.
"""

import numpy as np
import torch

INT4_GROUP = 32


def quantize_int8(w: torch.Tensor) -> torch.Tensor:
    x = w.float().cpu().numpy()
    amax = np.abs(x).max(axis=1, keepdims=True)
    scale = np.where(amax > 0, amax / np.float32(127.0), np.float32(1.0)).astype(np.float32)
    q = np.clip(np.rint(x / scale), -127, 127)
    return torch.from_numpy((q * scale).astype(np.float32)).to(w.device)


def quantize_int4(w: torch.Tensor, group: int = INT4_GROUP) -> torch.Tensor:
    rows, cols = w.shape
    if cols % group != 0:
        raise ValueError(f"cols must be a multiple of {group}")
    g = w.float().cpu().numpy().reshape(rows, cols // group, group)
    # argmax returns the first index on ties, like the kernel.
    idx = np.abs(g).argmax(axis=-1)[..., None]
    m = np.take_along_axis(g, idx, axis=-1)
    scale = np.where(m != 0, m / np.float32(-8.0), np.float32(1.0)).astype(np.float32)
    q = np.clip(np.rint(g / scale), -8, 7)
    stored = scale.astype(np.float16).astype(np.float32)
    return torch.from_numpy((q * stored).astype(np.float32).reshape(rows, cols)).to(w.device)


QUANTIZERS = {"int8": quantize_int8, "int4": quantize_int4}
