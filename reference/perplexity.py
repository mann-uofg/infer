"""Chunked perplexity, the same method as tools/perplexity.cpp and llama.cpp's llama-perplexity.

Tokens are cut into chunks of `ctx`; each chunk (first token replaced by BOS if given) is run
from scratch, and only its second half is scored: logits at positions ctx/2 .. ctx-2 predict
tokens ctx/2+1 .. ctx-1. Perplexity is exp of the mean negative log-likelihood.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


def chunked_nll(
    logits_of: Callable[[list[int]], torch.Tensor],
    tokens: list[int],
    ctx: int,
    bos: int | None = None,
    max_chunks: int | None = None,
) -> tuple[float, int]:
    """Sum of NLLs and number of scored tokens. logits_of(chunk) returns [ctx, vocab]."""
    n = len(tokens) // ctx
    if max_chunks:
        n = min(n, max_chunks)
    first = ctx // 2
    total = 0.0
    count = 0
    for c in range(n):
        chunk = list(tokens[c * ctx : (c + 1) * ctx])
        if bos is not None:
            chunk[0] = bos
        logits = logits_of(chunk).float()
        logprobs = torch.log_softmax(logits[first : ctx - 1], dim=-1)
        targets = torch.tensor(chunk[first + 1 : ctx], device=logprobs.device)
        total += float(-logprobs.gather(1, targets[:, None]).sum())
        count += ctx - 1 - first
    return total, count


def perplexity(total: float, count: int) -> float:
    return math.exp(total / count)
