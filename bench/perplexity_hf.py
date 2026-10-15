"""Perplexity with Hugging Face transformers, by the same method as tools/perplexity, to check
the tool on the real checkpoint.

    python bench/perplexity_hf.py --model models/llama-3.2-1b \\
        --tokens models/llama-3.2-1b/wikitext2-test.bin --dtype float32
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reference.perplexity import chunked_nll, perplexity  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--bos", type=int, default=128000)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)
    ).cuda().eval()
    tokens = np.fromfile(args.tokens, dtype="<i4").tolist()

    @torch.no_grad()
    def logits_of(chunk):
        return model(torch.tensor([chunk], device="cuda")).logits[0]

    total, count = chunked_nll(logits_of, tokens, args.ctx, bos=args.bos)
    print(f"transformers {args.dtype}: tokens scored {count}, mean nll {total / count:.6f}, "
          f"perplexity {perplexity(total, count):.4f}")


if __name__ == "__main__":
    main()
