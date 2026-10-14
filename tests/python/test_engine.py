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


@pytest.fixture(scope="module", params=["tiny", "random-1b", "llama-3.2-1b"])
def model(request):
    d = model_dir(request.param)
    return d, Llama(d, device="cuda")


def relative_error(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.abs(got - want).max() / np.abs(want).max())


def test_layer_by_layer(model):
    d, ref = model
    cfg = ref.cfg
    tokens = random_tokens(cfg.vocab_size, 23, seed=10)
    want_logits, want_hidden = ref.forward(tokens, return_hidden=True)
    hidden, logits = engine_dump(d, tokens, cfg.num_layers, cfg.hidden_size, cfg.vocab_size)

    # Both sides compute in fp32 from the same bf16 weights; they differ only in the order of
    # floating-point additions, so the error should stay around 1e-6 relative.
    for i, (got, want) in enumerate(zip(hidden, want_hidden)):
        err = relative_error(got, want.cpu().numpy())
        assert err < 1e-4, f"residual stream after layer {i}: relative error {err:.2e}"
    err = relative_error(logits, want_logits.cpu().numpy())
    assert err < 1e-4, f"logits: relative error {err:.2e}"
    assert (logits.argmax(-1) == want_logits.argmax(-1).cpu().numpy()).all()


def test_greedy_generation_matches_reference(model):
    d, ref = model
    prompt = random_tokens(ref.cfg.vocab_size, 7, seed=11)
    assert engine_generate(d, prompt, 24) == ref.generate_greedy(prompt, 24)
