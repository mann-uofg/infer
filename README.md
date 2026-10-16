# infer

An LLM inference engine written from scratch in C++ and CUDA for Llama-3.2-1B: my own kernels
for every op, checked layer by layer against a PyTorch reference, with decode running at
**93% of the GPU's measured memory bandwidth** on an RTX 5070 Ti, and 1.3-1.5x llama.cpp's
decode speed at the same perplexity.

## Results

Llama-3.2-1B, batch size 1, on an RTX 5070 Ti (Blackwell, 16 GB), driver 617.42, CUDA 13.3,
WSL2. Decode after a 128-token prompt; perplexity on WikiText-2 with llama.cpp's method.

| | infer | llama.cpp v0.6.0 | transformers 5.19 (bf16, SDPA) |
|---|---:|---:|---:|
| decode, bf16 | **317 tok/s** (93% of measured bandwidth) | 248 tok/s | 90 tok/s |
| decode, 8-bit weights | **584 tok/s** (int8) | 387 tok/s (Q8_0) | |
| decode, 4-bit weights | **733 tok/s** (int4 g32, int8 head) | 518 tok/s (Q4_0, Q6_K head) | |
| perplexity, bf16 | 10.189 | 10.190 | 10.196 (fp32: 10.189) |
| perplexity, 8-bit | 10.208 | 10.201 | |
| perplexity, 4-bit | 11.49 | 11.51 | |
| prefill, 2048 tokens | 9.4k tok/s | **30.9k tok/s** | 28.9k tok/s |

At the same quality, my decode is 1.3-1.5x llama.cpp's on this model and GPU. Prefill is where
it is not: llama.cpp and cuBLAS use tensor cores, my GEMM runs on CUDA cores in fp32, and they
are ~3x faster.

The bandwidth percentage is against **846.3 GB/s, the peak I measured on this card** with a
streaming read kernel (`bench/membw.cu`), not the spec sheet. A few big GEMVs measure slightly
above it, so that number is itself a little conservative.

Greedy text from the engine is token-for-token identical to transformers in fp32 (and to my
PyTorch reference) on the prompts I checked, e.g. *"The capital of France is Paris. It is the
most populous city in France and the country's capital and largest city."*

<details>
<summary>All numbers (from <code>bench/summarize.py</code>)</summary>

Decode, batch 1, 128 new tokens after a prompt of the given length. tok/s is the median of 5 runs; GB/s is weights + KV cache read per step; peak = 846.3 GB/s measured.

| engine | weights read / token | ctx 128 tok/s | ctx 512 tok/s | ctx 2048 tok/s | % of peak (ctx 128) |
|---|---:|---:|---:|---:|---:|
| infer, naive kernels | 2.47 GB | 26.3 | 17.8 | 7.8 | 7.7% |
| infer bf16, no CUDA graph | 2.47 GB | 293.8 | 291.5 | 280.1 | 86.0% |
| infer bf16 | 2.47 GB | 316.7 | 319.0 | 300.6 | 92.7% |
| infer int8 | 1.24 GB | 583.9 | 590.8 | 533.8 | 85.9% |
| infer int4 g32, int8 LM head | 0.81 GB | 732.5 | 716.4 | 668.4 | 70.7% |
| infer int4 g32 | 0.70 GB | 815.8 | 801.7 | 749.7 | 67.6% |
| infer int4 g128 | 0.64 GB | 910.6 | 851.7 | 763.8 | 69.3% |
| llama.cpp BF16 | | 247.7 | 249.9 | 247.2 | |
| llama.cpp Q8_0 | | 386.7 | 388.6 | 348.0 | |
| llama.cpp Q4_0 (Q6_K head) | | 517.5 | 510.3 | 507.7 | |
| transformers bf16 (sdpa) | | 90.2 | 83.7 | 83.2 | |
| transformers bf16 (eager) | | 61.6 | 56.6 | 62.8 | |

Prefill (prompt processing), tokens per second; time to first token in brackets for infer and transformers.

| engine | 128 tokens | 512 tokens | 2048 tokens |
|---|---:|---:|---:|
| infer, naive kernels | 180 (712 ms) | 192 (2662 ms) | 193 (10637 ms) |
| infer bf16 | 6420 (20 ms) | 9251 (55 ms) | 9381 (218 ms) |
| infer int8 | 7137 (18 ms) | 9891 (52 ms) | 9621 (213 ms) |
| infer int4 g32 | 6467 (20 ms) | 9353 (55 ms) | 8847 (231 ms) |
| llama.cpp BF16 | 19860 | 30588 | 30901 |
| llama.cpp Q8_0 | 19695 | 38052 | 36746 |
| llama.cpp Q4_0 | 21048 | 35679 | 35045 |
| transformers bf16 (sdpa) | 12002 (11 ms) | 27483 (19 ms) | 28932 (71 ms) |
| transformers bf16 (eager) | 6556 (20 ms) | 22131 (23 ms) | 12142 (169 ms) |

Per kernel, Llama-3.2-1B decode shapes, inputs rotated through 256 MB so nothing is served from L2. Times are per launch, medians of graph-batched launches.

| kernel | naive us | optimized us | speedup | optimized GB/s | % of peak |
|---|---:|---:|---:|---:|---:|
| gemv qkv 3072x2048 | 188.5 | 15.9 | 12x | 794 | 94% |
| gemv o 2048x2048 | 191.6 | 11.0 | 17x | 763 | 90% |
| gemv gate_up 16384x2048 | 244.5 | 83.2 | 3x | 808 | 95% |
| gemv down 2048x8192 | 761.8 | 41.4 | 18x | 812 | 96% |
| gemv lm_head 128256x2048 | 2105.0 | 631.5 | 3x | 833 | 98% |
| attention decode ctx=512 | 1580.8 | 5.5 | 289x | 196 | 23% |
| attention decode ctx=2048 | 6346.9 | 11.0 | 576x | 383 | 45% |
| argmax V=128256 | 19.7 | 1.5 | 13x | 350 | 41% |

Quantized GEMVs (GB/s on the quantized bytes):

| kernel | bf16 us | int8 us | int4 us | int8 % of peak | int4 % of peak |
|---|---:|---:|---:|---:|---:|
| gemv qkv 3072x2048 | 15.9 | 9.0 | 8.1 | 83% | 52% |
| gemv o 2048x2048 | 11.0 | 7.2 | 6.2 | 69% | 45% |
| gemv gate_up 16384x2048 | 83.2 | 40.9 | 24.5 | 97% | 91% |
| gemv down 2048x8192 | 41.4 | 22.1 | 14.2 | 90% | 79% |
| gemv lm_head 128256x2048 | 631.5 | 316.6 | 162.0 | 98% | 108% |

Prefill GEMMs (TFLOPS):

| shape | naive | tiled (mine, fp32) | cuBLAS fp32 | cuBLAS bf16 tensor cores |
|---|---:|---:|---:|---:|
| gemm qkv T=128 | 0.4 | 12.8 | 15.9 | 79.1 |
| gemm qkv T=2048 |  | 21.5 | 26.4 | 89.3 |
| gemm o T=2048 |  | 21.6 | 25.5 | 89.2 |
| gemm gate_up T=2048 |  | 22.9 | 27.9 | 93.4 |
| gemm down T=2048 |  | 22.9 | 27.6 | 93.8 |

Perplexity, WikiText-2 test (raw), chunks of 512 tokens scoring the second half of each (llama.cpp's method), 564 chunks, 143,820 scored tokens:

| model | perplexity |
|---|---:|
| transformers fp32 | 10.1894 |
| transformers bf16 | 10.1962 |
| infer bf16 | 10.1893 ± 0.0704 |
| llama.cpp BF16 | 10.1898 ± 0.07044 |
| infer int8 (per row) | 10.2080 ± 0.0706 |
| llama.cpp Q8_0 (blocks of 32) | 10.2006 ± 0.07053 |
| infer int4 g32, int8 LM head | 11.4946 ± 0.0808 |
| llama.cpp Q4_0 (blocks of 32, Q6_K head) | 11.5112 ± 0.08086 |
| infer int4 g32 | 11.6197 ± 0.0819 |
| infer int4 g64 | 12.0935 ± 0.0856 |
| infer int4 g128 | 12.8937 ± 0.0928 |
| infer int4 g128, int8 LM head | 12.6877 ± 0.0910 |

llama.cpp: `llama-bench -ngl 99 -fa 1 -r 5`, mean of 5 (decode is `tg128` at depth 128, 512
and 2048); `llama-perplexity -c 512` on the same `wiki.test.raw`. Its Q8_0 and Q4_0 use blocks
of 32 with fp16 scales and keep the tied embedding / LM head in Q8_0 and Q6_K; my int8 has one
scale per row and my int4 one per 32 weights (g32) or 128 (g128). Raw results, with the GPU
state at the start of each run, are in `bench/results/m5-real-*`; the same suite on random
weights with the same shapes is in `bench/results/m5-random-*`.
</details>

## Profiler evidence

Nsight Compute on one decode step of the real model at context 512 (`bench/profile_decode.sh`,
summaries in `bench/results/profiles/m5-real-*-ncu.md`, full reports kept locally):

| kernel | DRAM throughput, % of peak | SM throughput, % of peak | achieved occupancy | registers |
|---|---:|---:|---:|---:|
| naive matmul (all projections) | 13.7% | 3.3% | 17.5% | 39 |
| naive attention | 0.5% | 0.1% | 2.1% | 167 |
| GEMV bf16, gate/up + SwiGLU | 92.9% | 7.7% | 73% | 44 |
| GEMV bf16, o and down (+= residual) | 82.6% | 6.0% | 53% | 40 |
| GEMV bf16, qkv and LM head | 80.2% | 7.7% | 42% | 44 |
| split-KV decode attention | 29.1% | 21.7% | 15% | 51 |
| GEMV int4, gate/up + SwiGLU | 81.5% | 18.3% | 31% | 86 |
| GEMV int4, o and down | 58.7% | 10.4% | 17% | 86 |

How to read it: the optimized GEMVs keep DRAM 80-93% busy while the SMs sit at ~7%, which is
what a memory-bound kernel that is doing its job looks like. The naive kernels have both low:
they are neither moving bytes nor computing, they are waiting (uncoalesced loads, one thread
per row). The int4 GEMV spends more SM time unpacking and runs at 86 registers per thread,
which allows 2 blocks per SM; capping it at 85 registers for a third block was measured and made
no difference, so the remaining int4 gap is instruction count and staging the input vector
per block, not occupancy. Nsight Compute replays each kernel with cold caches, so its times are
not comparable with the benchmark's; the percentages are the point.

## Why bandwidth is the number that matters

At batch size 1, generating a token reads every weight once and does one multiply-add with
it: 2 flops per 2 bytes of bf16, one flop per byte. This GPU can do about 55 TFLOPS of fp32 on
its CUDA cores but move only ~850 GB/s, so anything below ~65 flops per byte is limited by
memory. Decode sits at 1, so the best possible decode speed is simply bandwidth divided by the
bytes per token, and the honest way to report a decode kernel is as a fraction of that.
Prefill reuses each weight once per prompt token, so at 2048 tokens it sits at hundreds of flops per
byte and is limited by arithmetic instead.

![roofline](docs/roofline.png)

Decode GEMVs sit at the far left, on the bandwidth roof; nothing but fewer bytes moves them,
which is why weight-only quantization speeds decode up even though the math stays fp32.
Prefill GEMMs sit far right, under the compute roof, where my CUDA-core GEMM reaches ~40% of
the fp32 peak and cuBLAS on bf16 tensor cores is 4x faster again.

## Architecture

```
 token ──► Embedding ──► x (fp32 residual stream)
              │
              │ ×16 layers
              │   GEMV(RMSNorm(x) fused, Wqkv)          q | k | v
              │   RoPE, write k and v to the bf16 KV cache
              │   split-KV attention (one block per KV head, 16 splits) + merge
              │   x += GEMV(attn, Wo)                   residual fused
              │   GEMV(RMSNorm(x) fused, Wgate|up)      silu(gate) * up fused
              │   x += GEMV(act, Wdown)                 residual fused
              ▼
         GEMV(RMSNorm(x) fused, LM head) ──► logits ──► argmax / sampling ──► next token
```

The whole decode step (argmax, 115 kernel launches, position update) is captured once as a
CUDA graph and replayed per token; the position and the token live in device memory so the same
graph is valid at every step. Prefill runs the same layers with a tiled GEMM and a
FlashAttention-style kernel.

| Piece | Where |
|---|---|
| PyTorch reference (written from the architecture, checked against transformers) | `reference/llama.py` |
| safetensors loader (mmap), JSON parser, config | `src/safetensors.cpp`, `src/json.cpp`, `src/config.cpp` |
| Engine: KV cache, prefill/decode, CUDA graph, sampling loop | `src/engine.cpp` |
| Naive kernels (the baseline) | `src/kernels/naive.cu` |
| Decode GEMV (bf16 / int8 / int4) with fused norm, residual, SwiGLU | `src/kernels/gemv.cu` |
| Split-KV decode attention, RoPE + cache write, argmax | `src/kernels/decode.cu` |
| Tiled GEMM, flash-style prefill attention, RMSNorm | `src/kernels/prefill.cu` |
| Weight-only quantization | `src/kernels/quant.cu` |
| Temperature / top-k / top-p sampling | `src/kernels/sampling.cu` |
| Perplexity | `tools/perplexity.cpp`, `src/kernels/eval.cu` |

## Correctness

- **Reference vs transformers:** `reference/llama.py` matches Hugging Face transformers
  exactly on a tiny random model (difference 0.0 in fp32) and within 1e-3 on logits for the
  1B shapes.
- **Every kernel vs a CPU implementation:** 95 GoogleTest cases with explicit tolerances,
  including ragged shapes, ties in argmax, empty attention splits and bit-exact quantization.
  compute-sanitizer memcheck, racecheck and initcheck are clean.
- **Engine vs reference, layer by layer:** the residual stream after every layer, the prefill
  logits and the decode-path logits at every position, for every backend (naive, optimized,
  CUDA graph, int8, int4), plus 24 tokens of greedy generation that must match token for
  token. Quantized runs compare against a reference using the same quantized weights.
- **Perplexity tool vs a Python implementation of the same method,** and on the real model
  against transformers (10.1893 vs 10.1894 in fp32) and llama.cpp (10.1898).

The KV cache is bf16, so the reference rounds K and V to bf16 too; the remaining differences
come from summation order (about 1e-3 relative on the 1B model, 1e-4 on the tiny one).

## Build, test, run

Linux (developed in WSL2 Ubuntu 24.04), CUDA 12.8+ (built with 13.3), CMake 3.25+, Ninja, a
C++20 compiler, and a GPU with compute capability 8.0 or newer.

```sh
cmake --preset release && cmake --build --preset release
ctest --preset release                 # C++ tests (GPU tests need a GPU)

scripts/setup_python.sh                # venv with torch + transformers
python -m pytest                       # reference + engine end-to-end tests

# The real model (gated: accept the licence on Hugging Face, then `hf auth login`):
hf download meta-llama/Llama-3.2-1B --local-dir models/llama-3.2-1b --exclude "original/*"
python scripts/generate.py --model models/llama-3.2-1b "The capital of France is" --tokens 32
python scripts/generate.py --model models/llama-3.2-1b "Once upon a time" --tokens 64 \
    --weights int4 --temperature 0.6 --top-p 0.9 --seed 1

# Or random weights with the same shapes (no download, enough for tests and speed):
python -m reference.random_model reference/configs/llama-3.2-1b.json models/random-1b

# Perplexity on WikiText-2:
python scripts/prepare_wikitext.py --text-out models/wiki.test.raw --tokenizer models/llama-3.2-1b
build/release/perplexity --model models/llama-3.2-1b --tokens models/llama-3.2-1b/wikitext2-test.bin \
    --bos 128000 --weights int4
```

Benchmarks (on an idle GPU):

```sh
build/release/membw                    # peak bandwidth
bench/llamacpp/build.sh                # llama.cpp v0.6.0 with CUDA
bench/llamacpp/convert.sh models/llama-3.2-1b ~/models/llama-3.2-1b-gguf
bench/run_all.sh models/llama-3.2-1b ~/models/llama-3.2-1b-gguf m5-real
python3 bench/summarize.py bench/results m5-real
```

## Design decisions

- **bf16 weights, fp32 activations and accumulation, bf16 KV cache.** At batch 1 the weights
  are 2.47 GB per token and the activations a few KB, so fp32 activations cost nothing
  measurable and keep the reference comparison tight.
- **Fusion where it saves memory traffic or launches:** RMSNorm is applied while the GEMV
  stages its input, residual adds happen in the GEMV epilogue, SwiGLU is computed from paired
  gate/up rows. 7 launches per layer instead of 10.
- **GQA-aware split-KV attention:** one block per KV head serves its 4 query heads, so K and V
  are read once per step; 16 splits fill the GPU at long context.
- **Weight-only quantization with dequant in registers:** int8 per row, int4 per 32 weights
  (the largest-magnitude value maps to -8, as in llama.cpp's Q4_0). Groups of 128 were the
  starting point; they cost 2.7 perplexity points on this 1B model against 1.4 for groups of
  32, for 12% fewer bytes. Nibbles are converted with the 2^23 exponent trick instead of
  int-to-float instructions.
- **Sampling on the GPU** (radix select for the top-k and top-p cut-offs), with per-position
  random numbers from a seed, so a sampled step is still one CUDA graph and runs reproduce.

## Limitations

- One model architecture (Llama), batch size 1, one GPU.
- Prefill runs on CUDA cores in fp32; tensor cores (as cuBLAS uses) would make it ~4x faster.
- int4 decode reaches ~68% of peak bandwidth: small int4 matrices are limited by staging the
  input vector per block and by instruction count, not by DRAM.
- Quality is measured as WikiText-2 perplexity only, not on downstream tasks.
- Developed and measured under WSL2; numbers on bare-metal Linux may differ slightly.
