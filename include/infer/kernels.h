// Launchers for every GPU kernel in the engine.
//
// Conventions:
//   - Weights are bf16, stored as in the checkpoint: a projection W is [out, in], row-major,
//     and computes y = x @ W^T.
//   - Activations are fp32, row-major [T, width], one row per token.
//   - All accumulation is in fp32.
//   - Every launcher enqueues on `stream` and returns without synchronizing.
//
// Each op has a naive version (the simplest correct kernel, kept as the baseline for the
// before/after table) and, where it matters, optimized versions.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace infer::kernels {

using bf16 = __nv_bfloat16;

// out[t, :] = table[tokens[t], :]
void Embedding(const bf16* table, const int* tokens, int T, int H, float* out, cudaStream_t stream);

// out[t, :] = x[t, :] / sqrt(mean(x[t, :]^2) + eps) * w
// Naive: one thread per row.
void RmsNormNaive(const float* x, const bf16* w, int T, int H, float eps, float* out,
                  cudaStream_t stream);

// y[t, n] (+)= sum_k x[t, k] * W[n, k]. With accumulate, adds into y (used to fold the
// residual connection into the projection that feeds it).
// Naive: one thread per output element.
void MatmulNaive(const float* x, const bf16* W, int T, int N, int K, float* y, bool accumulate,
                 cudaStream_t stream);

// Rotates the q and k parts of each row of qkv in place. qkv rows are laid out as
// [q heads | k heads | v heads], each head head_dim wide; row t has position pos0 + t.
// Pairs dimension i with i + head_dim/2 (Hugging Face "rotate half" layout).
void RopeNaive(float* qkv, int T, int num_heads, int num_kv_heads, int head_dim,
               const float* inv_freq, int pos0, cudaStream_t stream);

// Copies the k and v parts of each qkv row (after RoPE) into the cache as bf16. Row t is
// position pos0 + t. Cache layout per layer: [num_kv_heads, max_seq, head_dim], so the keys of
// one head are contiguous in position order.
void KvCacheWrite(const float* qkv, int T, int num_heads, int num_kv_heads, int head_dim, int pos0,
                  int max_seq, bf16* k_cache, bf16* v_cache, cudaStream_t stream);

// Causal grouped-query attention for T queries at positions pos0..pos0+T-1, against the cache
// (which must already hold positions 0..pos0+T-1). q is read from the qkv buffer
// [T, qkv_dim]; out is [T, num_heads * head_dim].
// Naive: one thread per (query, head), two passes over the keys.
void AttentionNaive(const float* qkv, int T, int pos0, int num_heads, int num_kv_heads,
                    int head_dim, const bf16* k_cache, const bf16* v_cache, int max_seq, float* out,
                    cudaStream_t stream);

// out[t, i] = silu(gate_up[t, i]) * gate_up[t, I + i]
void SwiGlu(const float* gate_up, int T, int I, float* out, cudaStream_t stream);

// out[0] = index of the largest logit (lowest index on ties).
void Argmax(const float* logits, int V, int* out, cudaStream_t stream);

}  // namespace infer::kernels
