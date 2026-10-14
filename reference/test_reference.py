"""Checks the from-scratch reference against Hugging Face transformers.

Both run in float32 on the same bf16 checkpoint, so they should agree to float32 rounding.
The tiny random model always runs (CPU is enough, so this also runs in CI). Larger models run
when their directories exist under models/:

    models/random-1b   random weights, Llama-3.2-1B shapes
    models/llama-3.2-1b   the real checkpoint
"""

from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from reference.llama import Config, Llama, rope_inv_freq
from reference.random_model import make_random_model

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "reference" / "configs"
MODELS = REPO / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("tiny")
    make_random_model(CONFIGS / "tiny.json", out, seed=1)
    return out


def hf_model(model_dir: Path, device: str) -> LlamaForCausalLM:
    return LlamaForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    ).to(device).eval()


def random_tokens(vocab: int, n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=g).tolist()


@pytest.mark.parametrize("config", ["tiny.json", "llama-3.2-1b.json"])
def test_rope_inv_freq_matches_hf(config):
    path = CONFIGS / config
    ours = rope_inv_freq(Config.from_json(path))
    hf_cfg = LlamaConfig.from_json_file(str(path))
    theirs, _ = ROPE_INIT_FUNCTIONS["llama3"](hf_cfg, "cpu")
    torch.testing.assert_close(ours, theirs, rtol=0, atol=0)


def test_tiny_logits_and_hidden_states_match_hf(tiny_dir):
    ref = Llama(tiny_dir)
    hf = hf_model(tiny_dir, "cpu")
    tokens = random_tokens(ref.cfg.vocab_size, 37, seed=2)

    logits, hidden = ref.forward(tokens, return_hidden=True)
    with torch.no_grad():
        out = hf(torch.tensor([tokens]), output_hidden_states=True)

    # Hugging Face returns the embedding output, then the output of each layer except the
    # last, then the last layer's output after the final norm. Compare all but that last one.
    for i in range(ref.cfg.num_layers):
        torch.testing.assert_close(hidden[i], out.hidden_states[i][0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(logits, out.logits[0], rtol=1e-4, atol=1e-4)


def test_tiny_greedy_generation_matches_hf(tiny_dir):
    ref = Llama(tiny_dir)
    hf = hf_model(tiny_dir, "cpu")
    prompt = random_tokens(ref.cfg.vocab_size, 9, seed=3)

    ours = ref.generate_greedy(prompt, 16)
    with torch.no_grad():
        seq = hf.generate(
            torch.tensor([prompt]), max_new_tokens=16, do_sample=False, pad_token_id=0
        )
    assert ours == seq[0, len(prompt) :].tolist()


@pytest.mark.parametrize("name", ["random-1b", "llama-3.2-1b"])
def test_full_size_logits_match_hf(name):
    model_dir = MODELS / name
    if not (model_dir / "config.json").exists():
        pytest.skip(f"{model_dir} not present")
    cfg = Config.from_json(model_dir / "config.json")
    tokens = random_tokens(cfg.vocab_size, 64, seed=4)

    # Run one model at a time: two float32 copies of a 1B model don't fit next to each other
    # on a 16 GB card.
    hf = hf_model(model_dir, DEVICE)
    with torch.no_grad():
        expected = hf(torch.tensor([tokens], device=DEVICE)).logits[0].float().cpu()
    del hf
    torch.cuda.empty_cache()

    ref = Llama(model_dir, device=DEVICE)
    got = ref.forward(tokens).cpu()
    torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-3)
    assert torch.equal(got.argmax(-1), expected.argmax(-1))
