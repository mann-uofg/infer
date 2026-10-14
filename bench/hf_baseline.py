"""Hugging Face transformers baseline: the same prefill/decode measurement as bench_engine.

The model runs in bf16 on the GPU with a KV cache, in eager mode (one Python call per decode
step, PyTorch kernels); attention is either the eager implementation or PyTorch SDPA.
Greedy decoding, random prompt tokens, the same context lengths, decode steps and repeats as
bench_engine.

    python bench/hf_baseline.py --model models/random-1b --attn sdpa --json out.jsonl
"""

import argparse
import json
import statistics
import time

import torch
from transformers import DynamicCache, LlamaForCausalLM


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


@torch.no_grad()
def run_once(model, prompt, steps):
    cache = DynamicCache()
    out, ttft = timed(lambda: model(input_ids=prompt, past_key_values=cache, use_cache=True))
    nxt = out.logits[:, -1:].argmax(-1)

    def decode():
        tok = nxt
        for _ in range(steps):
            o = model(input_ids=tok, past_key_values=cache, use_cache=True)
            tok = o.logits[:, -1:].argmax(-1)
        return tok

    _, dt = timed(decode)
    return ttft, steps / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--attn", default="sdpa", choices=["eager", "sdpa"])
    ap.add_argument("--contexts", default="128,512,2048")
    ap.add_argument("--generate", type=int, default=128)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--json")
    args = ap.parse_args()

    model = LlamaForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    ).cuda().eval()
    vocab = model.config.vocab_size
    print(f"transformers, bf16, attention={args.attn}, {args.warmup} warmup + {args.runs} runs")
    print(f"{'context':>7} | {'TTFT ms':>10} | {'decode tok/s (range)':>22}")
    g = torch.Generator(device="cuda").manual_seed(1234)
    for ctx in map(int, args.contexts.split(",")):
        prompt = torch.randint(0, vocab, (1, ctx), device="cuda", generator=g)
        ttfts, tps = [], []
        for r in range(args.warmup + args.runs):
            t, s = run_once(model, prompt, args.generate)
            if r >= args.warmup:
                ttfts.append(t * 1e3)
                tps.append(s)
        print(f"{ctx:>7} | {statistics.median(ttfts):>10.2f} | "
              f"{statistics.median(tps):>7.1f} ({min(tps):5.1f}-{max(tps):5.1f})")
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps({
                    "engine": "transformers", "attn": args.attn, "context": ctx,
                    "generate": args.generate, "runs": args.runs,
                    "ttft_ms_median": statistics.median(ttfts),
                    "decode_tok_s_median": statistics.median(tps),
                    "decode_tok_s_min": min(tps), "decode_tok_s_max": max(tps),
                }) + "\n")


if __name__ == "__main__":
    main()
