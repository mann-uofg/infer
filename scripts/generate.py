"""Text in, text out: tokenizes a prompt, runs the engine binary, detokenizes the result.

    python scripts/generate.py --model models/llama-3.2-1b "The capital of France is" \\
        --tokens 32 [--weights int4] [--temperature 0.6 --top-p 0.9 --seed 1]

Extra arguments after the prompt go to the engine (build/release/infer) unchanged. With
--compare, the same greedy continuation is also produced by Hugging Face transformers
(--compare-dtype float32 or bfloat16) and printed underneath. My engine computes in fp32 with
bf16 weights, so it follows float32 closely; bfloat16 drifts after a few tokens.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("prompt")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--compare-dtype", default="float32", choices=["float32", "bfloat16"])
    args, engine_args = ap.parse_known_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt)["input_ids"]  # starts with BOS
    cmd = [str(REPO / "build" / "release" / "infer"), "--model", str(args.model),
           "--tokens", ",".join(map(str, ids)), "--generate", str(args.tokens), *engine_args]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    new = [int(x) for x in out.stdout.split()]
    print(args.prompt + tok.decode(new))
    sys.stderr.write(out.stderr)

    if args.compare:
        import torch
        from transformers import AutoModelForCausalLM

        dtype = getattr(torch, args.compare_dtype)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).cuda()
        with torch.no_grad():
            seq = model.generate(torch.tensor([ids]).cuda(), max_new_tokens=args.tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        hf = seq[0, len(ids):].tolist()
        print(f"--- transformers ({args.compare_dtype}, greedy):")
        print(args.prompt + tok.decode(hf))
        same = next((i for i, (a, b) in enumerate(zip(new, hf)) if a != b), min(len(new), len(hf)))
        print(f"--- first {same} of {args.tokens} tokens identical")


if __name__ == "__main__":
    main()
