"""Markdown tables for the README, built from the results files of one bench/run_all.sh run.

    python3 bench/summarize.py bench/results m5-random

Every number in the README tables comes from here, so it traces back to a results file.
"""

import json
import re
import sys
from pathlib import Path


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    results, prefix = Path(sys.argv[1]), sys.argv[2]
    peak = float(re.findall(r"read ([0-9.]+) GB/s",
                            (results / f"{prefix}-membw.txt").read_text())[-1])
    out = []

    # ---- Decode and prefill, all engines ----
    engines = [
        ("infer, naive kernels", f"{prefix}-engine-naive"),
        ("infer bf16, no CUDA graph", f"{prefix}-engine-bf16-nograph"),
        ("infer bf16", f"{prefix}-engine-bf16"),
        ("infer int8", f"{prefix}-engine-int8"),
        ("infer int4 g32, int8 LM head", f"{prefix}-engine-int4-int8head"),
        ("infer int4 g32", f"{prefix}-engine-int4"),
        ("infer int4 g128", f"{prefix}-engine-int4-g128"),
    ]
    contexts = [128, 512, 2048]
    out.append(f"Decode, batch 1, 128 new tokens after a prompt of the given length. "
               f"tok/s is the median of 5 runs; GB/s is weights + KV cache read per step; "
               f"peak = {peak:.1f} GB/s measured.\n")
    out.append("| engine | weights read / token | " +
               " | ".join(f"ctx {c} tok/s" for c in contexts) + " | % of peak (ctx 128) |")
    out.append("|---|---:|" + "---:|" * len(contexts) + "---:|")
    for name, f in engines:
        rows = {r["context"]: r for r in jsonl(results / f"{f}.jsonl")}
        if not rows:
            continue
        first = rows[contexts[0]]
        weights_gb = (first["step_bytes"] - 2 * 16 * (contexts[0] + 64.5) * 512 * 2) / 1e9
        cells = [f"{rows[c]['decode_tok_s_median']:.1f}" if c in rows else "" for c in contexts]
        pct = 100 * first["decode_gbps"] / peak
        out.append(f"| {name} | {weights_gb:.2f} GB | " + " | ".join(cells) + f" | {pct:.1f}% |")

    for label, q in [("llama.cpp BF16", "bf16"), ("llama.cpp Q8_0", "Q8_0"),
                     ("llama.cpp Q4_0 (Q6_K head)", "Q4_0")]:
        rows = jsonl(results / f"{prefix}-llamacpp-{q}.jsonl")
        tg = {r["n_depth"]: r for r in rows if r["n_gen"]}
        if tg:
            cells = [f"{tg[c]['avg_ts']:.1f}" if c in tg else "" for c in contexts]
            out.append(f"| {label} | | " + " | ".join(cells) + " | |")
    for attn in ["sdpa", "eager"]:
        rows = {r["context"]: r for r in jsonl(results / f"{prefix}-hf-{attn}.jsonl")}
        if rows:
            cells = [f"{rows[c]['decode_tok_s_median']:.1f}" if c in rows else "" for c in contexts]
            out.append(f"| transformers bf16 ({attn}) | | " + " | ".join(cells) + " | |")

    out.append("\nPrefill (prompt processing), tokens per second; time to first token in "
               "brackets for infer and transformers.\n")
    out.append("| engine | " + " | ".join(f"{c} tokens" for c in contexts) + " |")
    out.append("|---|" + "---:|" * len(contexts))
    for name, f in engines:
        rows = {r["context"]: r for r in jsonl(results / f"{f}.jsonl")}
        if rows and "nograph" not in f and "int8head" not in f and "g128" not in f:
            cells = [f"{rows[c]['prefill_tok_s']:.0f} ({rows[c]['ttft_ms_median']:.0f} ms)"
                     for c in contexts if c in rows]
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    for label, q in [("llama.cpp BF16", "bf16"), ("llama.cpp Q8_0", "Q8_0"),
                     ("llama.cpp Q4_0", "Q4_0")]:
        pp = {r["n_prompt"]: r for r in jsonl(results / f"{prefix}-llamacpp-{q}.jsonl")
              if r["n_prompt"]}
        if pp:
            out.append(f"| {label} | " +
                       " | ".join(f"{pp[c]['avg_ts']:.0f}" for c in contexts if c in pp) + " |")
    for attn in ["sdpa", "eager"]:
        rows = {r["context"]: r for r in jsonl(results / f"{prefix}-hf-{attn}.jsonl")}
        if rows:
            cells = [f"{c / (rows[c]['ttft_ms_median'] / 1e3):.0f} "
                     f"({rows[c]['ttft_ms_median']:.0f} ms)" for c in contexts if c in rows]
            out.append(f"| transformers bf16 ({attn}) | " + " | ".join(cells) + " |")

    # ---- Per-kernel before/after ----
    kernels = jsonl(results / f"{prefix}-kernels.jsonl")
    out.append("\nPer kernel, Llama-3.2-1B decode shapes, inputs rotated through 256 MB so "
               "nothing is served from L2. Times are per launch, medians of graph-batched "
               "launches.\n")
    out.append("| kernel | naive us | optimized us | speedup | optimized GB/s | % of peak |")
    out.append("|---|---:|---:|---:|---:|---:|")
    by = {}
    for r in kernels:
        by.setdefault(r["kernel"], {})[r["variant"]] = r
    for name, v in by.items():
        if "naive" not in v or "gbps" not in v["naive"]:
            continue
        best = v.get("optimized") or v.get("split16") or v.get("multiblock")
        if best is None:
            continue
        out.append(f"| {name} | {v['naive']['us']:.1f} | {best['us']:.1f} | "
                   f"{v['naive']['us'] / best['us']:.0f}x | {best['gbps']:.0f} | "
                   f"{100 * best['gbps'] / peak:.0f}% |")
    out.append("\nQuantized GEMVs (GB/s on the quantized bytes):\n")
    out.append("| kernel | bf16 us | int8 us | int4 us | int8 % of peak | int4 % of peak |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for name, v in by.items():
        if "int8" in v and "int4" in v:
            out.append(f"| {name} | {v['optimized']['us']:.1f} | {v['int8']['us']:.1f} | "
                       f"{v['int4']['us']:.1f} | {100 * v['int8']['gbps'] / peak:.0f}% | "
                       f"{100 * v['int4']['gbps'] / peak:.0f}% |")
    out.append("\nPrefill GEMMs (TFLOPS):\n")
    out.append("| shape | naive | tiled (mine, fp32) | cuBLAS fp32 | cuBLAS bf16 tensor cores |")
    out.append("|---|---:|---:|---:|---:|")
    for name, v in by.items():
        if "tiled" in v:
            cell = lambda k: f"{v[k]['tflops']:.1f}" if k in v else ""
            out.append(f"| {name} | {cell('naive')} | {cell('tiled')} | {cell('cublas-f32')} | "
                       f"{cell('cublas-bf16')} |")
    # ---- Perplexity (only for runs on the real checkpoint) ----
    ppl_rows = [
        ("transformers fp32", "hf-float32", r"perplexity ([0-9.]+)"),
        ("transformers bf16", "hf-bfloat16", r"perplexity ([0-9.]+)"),
        ("infer bf16", "bf16", r"perplexity ([0-9.]+ \+/- [0-9.]+)"),
        ("llama.cpp BF16", "llamacpp-bf16", r"PPL = ([0-9.]+ \+/- [0-9.]+)"),
        ("infer int8 (per row)", "int8", r"perplexity ([0-9.]+ \+/- [0-9.]+)"),
        ("llama.cpp Q8_0 (blocks of 32)", "llamacpp-Q8_0", r"PPL = ([0-9.]+ \+/- [0-9.]+)"),
        ("infer int4 g32, int8 LM head", "int4-g32-int8head", r"perplexity ([0-9.]+ \+/- [0-9.]+)"),
        ("llama.cpp Q4_0 (blocks of 32, Q6_K head)", "llamacpp-Q4_0", r"PPL = ([0-9.]+ \+/- [0-9.]+)"),
        ("infer int4 g32", "int4-g32", r"perplexity ([0-9.]+ \+/- [0-9.]+)"),
        ("infer int4 g64", "int4-g64", r"perplexity ([0-9.]+ \+/- [0-9.]+)"),
        ("infer int4 g128", "int4-g128", r"perplexity ([0-9.]+ \+/- [0-9.]+)"),
        ("infer int4 g128, int8 LM head", "int4-g128-int8head", r"perplexity ([0-9.]+ \+/- [0-9.]+)"),
    ]
    found = []
    for name, f, pattern in ppl_rows:
        path = results / f"{prefix}-ppl-{f}.txt"
        if path.exists():
            m = re.findall(pattern, path.read_text())
            if m:
                found.append(f"| {name} | {m[-1].replace('+/-', '±')} |")
    if found:
        out.append("\nPerplexity, WikiText-2 test (raw), chunks of 512 tokens scoring the second half "
                   "of each (llama.cpp's method), 564 chunks, 143,820 scored tokens:\n")
        out.append("| model | perplexity |")
        out.append("|---|---:|")
        out.extend(found)
    print("\n".join(out))


if __name__ == "__main__":
    main()
