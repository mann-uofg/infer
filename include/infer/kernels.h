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

// ---- Optimized prefill kernels (many tokens at once) ----

// Same as RmsNormNaive, one block per row with vector loads. H must be a multiple of 4.
void RmsNorm(const float* x, const bf16* w, int T, int H, float eps, float* out,
             cudaStream_t stream);

// C[T, N] (+)= A[T, K] x W[N, K]^T, shared-memory tiled with register blocking.
// K must be a multiple of 16.
void Gemm(const float* A, const bf16* W, int T, int N, int K, float* C, bool accumulate,
          cudaStream_t stream);

// Same contract as AttentionNaive, FlashAttention-style: tiles of keys through shared memory
// with an online softmax, never materialising the T x T score matrix. head_dim must be 64.
void AttentionPrefill(const float* qkv, int T, int pos0, int num_heads, int num_kv_heads,
                      int head_dim, const bf16* k_cache, const bf16* v_cache, int max_seq,
                      float* out, cudaStream_t stream);

// ---- Optimized decode kernels (one token at a time) ----

enum class GemvOut {
  kStore,       // y = W x
  kAccumulate,  // y += W x (residual connection)
  kSwiGlu,      // W is [2N, K] gate rows then up rows; y[i] = silu(gate_i . x) * (up_i . x)
};

// Matrix-vector product, W [N, K] (or [2N, K] for kSwiGlu), x [K], y [N]. If norm_w is not
// null, x is RMS-normalised and scaled by norm_w first (fused RMSNorm). K must be a multiple
// of 8 and at most 12288.
void Gemv(const float* x, const bf16* W, int N, int K, float* y, GemvOut out, const bf16* norm_w,
          float eps, cudaStream_t stream);

// Decode kernels read the current position from device memory (*pos), so that one captured
// CUDA graph can be replayed at every step.

// Rotates q and k of the single qkv row by position *pos, leaves q in place and writes k and v
// to the cache at *pos (layout as KvCacheWrite).
void RopeKvDecode(float* qkv, int num_heads, int num_kv_heads, int head_dim, const float* inv_freq,
                  const int* pos, int max_seq, bf16* k_cache, bf16* v_cache, cudaStream_t stream);

// Workspace for AttentionDecode: num_heads * num_splits * (head_dim + 2) floats.
size_t AttentionDecodeWorkspace(int num_heads, int head_dim, int num_splits);

// Split-KV ("Flash-Decoding") attention for one query at position *pos against cache
// positions 0..*pos. The positions are cut into num_splits ranges processed by separate blocks
// (so a long context still fills the GPU), each producing a partial softmax; a second kernel
// combines them. One block serves all query heads that share a KV head, so each K/V row is
// read from memory once per step, not once per query head.
void AttentionDecode(const float* q, int num_heads, int num_kv_heads, int head_dim,
                     const bf16* k_cache, const bf16* v_cache, int max_seq, const int* pos,
                     int num_splits, float* workspace, float* out, cudaStream_t stream);

// *pos += 1.
void AdvancePosition(int* pos, cudaStream_t stream);

// *pos = value.
void SetPosition(int* pos, int value, cudaStream_t stream);

// history[*pos] = *token: records the token about to be run at the current position.
void StoreToken(const int* token, int* history, const int* pos, cudaStream_t stream);

// Argmax over many blocks. Each block finds its local best and merges it into *best with one
// 64-bit atomicMax on (order-preserving float bits << 32 | ~index), so ties go to the lowest
// index. *best must be 0 beforehand; ArgmaxFinish leaves it 0 again.
void ArgmaxPartial(const float* logits, int V, unsigned long long* best, cudaStream_t stream);

// *token = index held in *best; history[*pos] = *token; *best = 0.
void ArgmaxFinish(unsigned long long* best, int* token, int* history, const int* pos,
                  cudaStream_t stream);

// Samples a token from softmax(logits / temperature), restricted to the top_k largest logits
// (top_k <= 0: no limit) and then to the smallest set of those whose probability adds up to
// top_p (1: no limit). The random number is uniforms[*pos], in [0, 1). Writes *token and
// history[*pos]. One block; see sampling.cu for the radix-select method.
void Sample(const float* logits, int V, float temperature, int top_k, float top_p,
            const float* uniforms, const int* pos, int* token, int* history, cudaStream_t stream);

}  // namespace infer::kernels
