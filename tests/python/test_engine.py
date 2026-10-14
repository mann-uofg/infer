"""End-to-end: the CUDA engine against the PyTorch reference.

Compares the residual stream after every layer (so a bug points at the layer that introduced
it), the logits, and greedy generation token for token. Needs a GPU and a built engine
(build/release/infer); skipped otherwise.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from reference.llama import Llama
from tests.python.engine_harness import BINARY, engine_dump, engine_generate

MODELS = Path(__file__).resolve().parents[2] / "models"

pytestmark = pytest.mark.skipif(
    not BINARY.exists() or not torch.cuda.is_available(), reason="needs the built engine and a GPU"
)


def model_dir(name: str) -> Path:
    d = MODELS / name
    if not (d / "config.json").exists():
        pytest.skip(f"{d} not present (python -m reference.random_model ...)")
    return d


def random_tokens(vocab: int, n: int, seed: int) -> list[int]:
    return np.random.default_rng(seed).integers(0, vocab, n).tolist()


# Largest allowed max|engine - reference| / max|reference|, per model.
#
# The reference rounds K and V to bf16 like the engine's cache does. Rounding is a step
# function, so where the two sides' fp32 values differ by an ulp next to a rounding boundary
# they round to neighbouring bf16 values, 2^-8 apart. On the tiny model that never happens in
# this test and the error is ~2e-6; on the 1B models it does and the error is ~1e-3 (measured;
# without the bf16 emulation it is ~7e-3). An indexing bug gives errors of order 1.
TOLERANCE = {"tiny": 1e-4, "random-1b": 3e-3, "llama-3.2-1b": 3e-3}


@pytest.fixture(scope="module", params=["tiny", "random-1b", "llama-3.2-1b"])
def model(request):
    d = model_dir(request.param)
    return d, Llama(d, device="cuda", kv_dtype=torch.bfloat16), TOLERANCE[request.param]


# Engine configurations under test: the naive baseline, the optimized decode path run
# eagerly, and the same path replayed as a CUDA graph (the default).
BACKENDS = {
    "naive": ("--backend", "naive"),
    "optimized": ("--backend", "optimized", "--graph", "off"),
    "optimized-graph": ("--backend", "optimized", "--graph", "on"),
}


@pytest.fixture(params=list(BACKENDS))
def backend(request):
    return BACKENDS[request.param]


def relative_error(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.abs(got - want).max() / np.abs(want).max())


def test_layer_by_layer(model, backend):
    d, ref, tol = model
    cfg = ref.cfg
    tokens = random_tokens(cfg.vocab_size, 23, seed=10)
    want_logits, want_hidden = ref.forward(tokens, return_hidden=True)
    want_logits = want_logits.cpu().numpy()
    hidden, logits, decode_logits = engine_dump(
        d, tokens, cfg.num_layers, cfg.hidden_size, cfg.vocab_size, *backend
    )

    # Both sides compute in fp32 from the same bf16 weights and round K/V to bf16; they differ
    # only in the order of floating-point additions (see TOLERANCE).
    for i, (got, want) in enumerate(zip(hidden, want_hidden)):
        err = relative_error(got, want.cpu().numpy())
        assert err < tol, f"residual stream after layer {i}: relative error {err:.2e}"
    err = relative_error(logits, want_logits)
    assert err < tol, f"prefill logits: relative error {err:.2e}"
    assert (logits.argmax(-1) == want_logits.argmax(-1)).all()

    # The decode path (one token at a time against the KV cache) must give the same logits.
    err = relative_error(decode_logits, want_logits)
    assert err < tol, f"decode logits: relative error {err:.2e}"
    assert (decode_logits.argmax(-1) == want_logits.argmax(-1)).all()


def test_greedy_generation_matches_reference(model, backend):
    d, ref, _ = model
    prompt = random_tokens(ref.cfg.vocab_size, 7, seed=11)
    assert engine_generate(d, prompt, 24, *backend) == ref.generate_greedy(prompt, 24)
