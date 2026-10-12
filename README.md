# infer

A transformer inference engine written from scratch in C++ and CUDA. It runs one small
decoder-only LLM with hand-written kernels, checks every layer against a PyTorch reference,
and measures how close decode gets to the memory bandwidth limit of the GPU.

**Status:** early. The build scaffold is in place; nothing runs yet.

## Plan

| Milestone | Deliverable |
|---|---|
| M0 Setup + reference | Verified environment and measured peak bandwidth; CMake + CUDA build; from-scratch PyTorch forward pass matching Hugging Face logits |
| M1 Naive end-to-end | safetensors loader; a naive CUDA kernel for every op; greedy generation matching the reference token for token |
| M2 KV cache + profiling | Prefill/decode split; KV cache; baseline tokens/sec; Nsight profiles; roofline baseline |
| M3 Optimized kernels | Optimized GEMV; fused norm and activation kernels; flash-style attention; CUDA Graphs; per-kernel before/after table |
| M4 Quantization | INT8 and INT4 weight-only, with perplexity measured against BF16 |
| M5 Benchmarks | Comparison against llama.cpp and Hugging Face transformers |

## Build

Requires CMake 3.25+, Ninja and a C++20 compiler. The CUDA toolkit (12.8 or later) is needed
for the GPU targets; without it only the CPU-side targets are built.

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
```
