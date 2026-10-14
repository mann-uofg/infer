"""Writes a randomly initialised Llama checkpoint in Hugging Face format.

Random weights with the real model's shapes are enough to test correctness against the
reference (same weights in, same logits out) and to measure speed (which depends only on the
shapes). They are not enough for perplexity or for checking that text makes sense.

    python -m reference.random_model reference/configs/tiny.json models/tiny
    python -m reference.random_model reference/configs/llama-3.2-1b.json models/random-1b
"""

import argparse
import shutil
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def make_random_model(config: Path, out: Path, seed: int = 0) -> None:
    torch.manual_seed(seed)
    cfg = LlamaConfig.from_json_file(str(config))
    model = LlamaForCausalLM(cfg).to(torch.bfloat16)

    # Norm weights start at exactly 1.0, which would let a bug that skips the multiply by the
    # weight go unnoticed. Perturb them.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.add_(0.1 * torch.randn_like(p))

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    # Keep the original config file next to the weights so nothing is lost in re-serialisation.
    shutil.copy(config, out / "config.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    make_random_model(args.config, args.out, args.seed)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
