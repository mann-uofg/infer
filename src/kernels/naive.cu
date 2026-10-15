// The naive kernels: the simplest correct implementation of each op.
//
// These are deliberately not optimized. They are the baseline for the before/after table and
// a second opinion when an optimized kernel disagrees with the reference.

#include <cfloat>

#include "infer/cuda_util.h"
#include "infer/kernels.h"

namespace infer::kernels {

namespace {

constexpr int kThreads = 256;

int Blocks(long long n) { return static_cast<int>((n + kThreads - 1) / kThreads); }

__global__ void EmbeddingKernel(const bf16* table, int V, const int* tokens, int H, float* out) {
  const int t = blockIdx.x;
  const int token = tokens[t];
  float* dst = out + static_cast<size_t>(t) * H;
  if (token < 0 || token >= V) {
    for (int k = threadIdx.x; k < H; k += blockDim.x) dst[k] = 0.0f;
    return;
  }
  const bf16* row = table + static_cast<size_t>(token) * H;
  for (int k = threadIdx.x; k < H; k += blockDim.x) dst[k] = __bfloat162float(row[k]);
}

__global__ void RmsNormNaiveKernel(const float* x, const bf16* w, int T, int H, float eps,
                                   float* out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const float* row = x + static_cast<size_t>(t) * H;
  float sum_sq = 0.0f;
  for (int k = 0; k < H; ++k) sum_sq += row[k] * row[k];
  const float scale = rsqrtf(sum_sq / H + eps);
  for (int k = 0; k < H; ++k) {
    out[static_cast<size_t>(t) * H + k] = row[k] * scale * __bfloat162float(w[k]);
  }
}

// Consecutive threads take consecutive n, so at each step of the k loop the threads of a warp
// read 32 different rows of W, 2*K bytes apart: every load is its own memory transaction.
// That is the main thing the optimized GEMV fixes.
__global__ void MatmulNaiveKernel(const float* x, const bf16* W, int T, int N, int K, float* y,
                                  bool accumulate) {
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<long long>(T) * N) return;
  const int n = static_cast<int>(idx % N);
  const int t = static_cast<int>(idx / N);
  const float* xr = x + static_cast<size_t>(t) * K;
  const bf16* wr = W + static_cast<size_t>(n) * K;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += xr[k] * __bfloat162float(wr[k]);
  float* out = y + static_cast<size_t>(t) * N + n;
  *out = accumulate ? *out + acc : acc;
}

__global__ void RopeNaiveKernel(float* qkv, int T, int rotated_heads, int head_dim, int row_stride,
                                const float* inv_freq, int pos0) {
  const int half = head_dim / 2;
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<long long>(T) * rotated_heads * half) return;
  const int i = static_cast<int>(idx % half);
  const int head = static_cast<int>((idx / half) % rotated_heads);
  const int t = static_cast<int>(idx / (static_cast<long long>(half) * rotated_heads));

  // Same order of operations as the reference: angle in fp32, then accurate sin/cos.
  const float angle = static_cast<float>(pos0 + t) * inv_freq[i];
  float s;
  float c;
  sincosf(angle, &s, &c);
  float* p = qkv + static_cast<size_t>(t) * row_stride + static_cast<size_t>(head) * head_dim;
  const float x1 = p[i];
  const float x2 = p[i + half];
  p[i] = x1 * c - x2 * s;
  p[i + half] = x2 * c + x1 * s;
}

__global__ void KvCacheWriteKernel(const float* qkv, int T, int num_heads, int num_kv_heads,
                                   int head_dim, int pos0, int max_seq, bf16* k_cache,
                                   bf16* v_cache) {
  const int kv_dim = num_kv_heads * head_dim;
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<long long>(T) * kv_dim) return;
  const int t = static_cast<int>(idx / kv_dim);
  const int c = static_cast<int>(idx % kv_dim);  // channel within k (or v): head * hd + d
  const int head = c / head_dim;
  const int d = c % head_dim;
  const float* row = qkv + static_cast<size_t>(t) * (num_heads * head_dim + 2 * kv_dim);
  const size_t dst =
      (static_cast<size_t>(head) * max_seq + static_cast<size_t>(pos0 + t)) * head_dim + d;
  k_cache[dst] = __float2bfloat16(row[num_heads * head_dim + c]);
  v_cache[dst] = __float2bfloat16(row[num_heads * head_dim + kv_dim + c]);
}

template <int HD>
__global__ void AttentionNaiveKernel(const float* qkv, int T, int pos0, int num_heads,
                                     int num_kv_heads, const bf16* k_cache, const bf16* v_cache,
                                     int max_seq, float* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * num_heads) return;
  const int t = idx / num_heads;
  const int h = idx % num_heads;
  const int kvh = h / (num_heads / num_kv_heads);
  const int stride = (num_heads + 2 * num_kv_heads) * HD;
  const bf16* k_base = k_cache + static_cast<size_t>(kvh) * max_seq * HD;
  const bf16* v_base = v_cache + static_cast<size_t>(kvh) * max_seq * HD;
  const int last = pos0 + t;  // causal: attend to positions 0..last

  float q[HD];
  const float* qp = qkv + static_cast<size_t>(t) * stride + static_cast<size_t>(h) * HD;
  for (int d = 0; d < HD; ++d) q[d] = qp[d];
  const float scale = rsqrtf(static_cast<float>(HD));

  // Pass 1: the largest score, so the exponentials in pass 2 cannot overflow.
  float m = -FLT_MAX;
  for (int j = 0; j <= last; ++j) {
    const bf16* k = k_base + static_cast<size_t>(j) * HD;
    float s = 0.0f;
    for (int d = 0; d < HD; ++d) s += q[d] * __bfloat162float(k[d]);
    m = fmaxf(m, s * scale);
  }

  // Pass 2: softmax weights and the weighted sum of values.
  float acc[HD];
  for (int d = 0; d < HD; ++d) acc[d] = 0.0f;
  float denom = 0.0f;
  for (int j = 0; j <= last; ++j) {
    const bf16* k = k_base + static_cast<size_t>(j) * HD;
    const bf16* v = v_base + static_cast<size_t>(j) * HD;
    float s = 0.0f;
    for (int d = 0; d < HD; ++d) s += q[d] * __bfloat162float(k[d]);
    const float p = expf(s * scale - m);
    denom += p;
    for (int d = 0; d < HD; ++d) acc[d] += p * __bfloat162float(v[d]);
  }
  float* o = out + static_cast<size_t>(t) * num_heads * HD + static_cast<size_t>(h) * HD;
  for (int d = 0; d < HD; ++d) o[d] = acc[d] / denom;
}

__global__ void SwiGluKernel(const float* gate_up, int T, int I, float* out) {
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<long long>(T) * I) return;
  const int i = static_cast<int>(idx % I);
  const int t = static_cast<int>(idx / I);
  const float g = gate_up[static_cast<size_t>(t) * 2 * I + i];
  const float u = gate_up[static_cast<size_t>(t) * 2 * I + I + i];
  out[idx] = g / (1.0f + expf(-g)) * u;
}

// One block. Each thread scans a strided slice, then the block reduces in shared memory.
// Ties go to the lower index, like torch.argmax.
__global__ void ArgmaxKernel(const float* logits, int V, int* out) {
  __shared__ float best_val[1024];
  __shared__ int best_idx[1024];
  float bv = -FLT_MAX;
  int bi = 0x7fffffff;
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    const float v = logits[i];
    if (v > bv) {  // strided scan visits indices in increasing order, so > keeps the lowest
      bv = v;
      bi = i;
    }
  }
  best_val[threadIdx.x] = bv;
  best_idx[threadIdx.x] = bi;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float ov = best_val[threadIdx.x + s];
      const int oi = best_idx[threadIdx.x + s];
      if (ov > best_val[threadIdx.x] ||
          (ov == best_val[threadIdx.x] && oi < best_idx[threadIdx.x])) {
        best_val[threadIdx.x] = ov;
        best_idx[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) out[0] = best_idx[0];
}

}  // namespace

void Embedding(const bf16* table, int V, const int* tokens, int T, int H, float* out,
               cudaStream_t stream) {
  EmbeddingKernel<<<T, kThreads, 0, stream>>>(table, V, tokens, H, out);
  CUDA_CHECK_LAUNCH();
}

void RmsNormNaive(const float* x, const bf16* w, int T, int H, float eps, float* out,
                  cudaStream_t stream) {
  RmsNormNaiveKernel<<<Blocks(T), kThreads, 0, stream>>>(x, w, T, H, eps, out);
  CUDA_CHECK_LAUNCH();
}

void MatmulNaive(const float* x, const bf16* W, int T, int N, int K, float* y, bool accumulate,
                 cudaStream_t stream) {
  MatmulNaiveKernel<<<Blocks(static_cast<long long>(T) * N), kThreads, 0, stream>>>(x, W, T, N, K,
                                                                                    y, accumulate);
  CUDA_CHECK_LAUNCH();
}

void RopeNaive(float* qkv, int T, int num_heads, int num_kv_heads, int head_dim,
               const float* inv_freq, int pos0, cudaStream_t stream) {
  // q heads and k heads are adjacent at the start of each row, so they can be treated as one
  // run of num_heads + num_kv_heads heads.
  const int rotated = num_heads + num_kv_heads;
  const int stride = (num_heads + 2 * num_kv_heads) * head_dim;
  const long long n = static_cast<long long>(T) * rotated * (head_dim / 2);
  RopeNaiveKernel<<<Blocks(n), kThreads, 0, stream>>>(qkv, T, rotated, head_dim, stride, inv_freq,
                                                      pos0);
  CUDA_CHECK_LAUNCH();
}

void KvCacheWrite(const float* qkv, int T, int num_heads, int num_kv_heads, int head_dim, int pos0,
                  int max_seq, bf16* k_cache, bf16* v_cache, cudaStream_t stream) {
  const long long n = static_cast<long long>(T) * num_kv_heads * head_dim;
  KvCacheWriteKernel<<<Blocks(n), kThreads, 0, stream>>>(qkv, T, num_heads, num_kv_heads, head_dim,
                                                         pos0, max_seq, k_cache, v_cache);
  CUDA_CHECK_LAUNCH();
}

void AttentionNaive(const float* qkv, int T, int pos0, int num_heads, int num_kv_heads,
                    int head_dim, const bf16* k_cache, const bf16* v_cache, int max_seq, float* out,
                    cudaStream_t stream) {
  const int n = T * num_heads;
  // Few, register-heavy threads: a small block keeps more of them resident.
  constexpr int kAttnThreads = 64;
  const int blocks = (n + kAttnThreads - 1) / kAttnThreads;
  switch (head_dim) {
    case 64:
      AttentionNaiveKernel<64><<<blocks, kAttnThreads, 0, stream>>>(
          qkv, T, pos0, num_heads, num_kv_heads, k_cache, v_cache, max_seq, out);
      break;
    case 128:
      AttentionNaiveKernel<128><<<blocks, kAttnThreads, 0, stream>>>(
          qkv, T, pos0, num_heads, num_kv_heads, k_cache, v_cache, max_seq, out);
      break;
    default:
      throw CudaError("attention: unsupported head_dim " + std::to_string(head_dim));
  }
  CUDA_CHECK_LAUNCH();
}

void SwiGlu(const float* gate_up, int T, int I, float* out, cudaStream_t stream) {
  SwiGluKernel<<<Blocks(static_cast<long long>(T) * I), kThreads, 0, stream>>>(gate_up, T, I, out);
  CUDA_CHECK_LAUNCH();
}

void Argmax(const float* logits, int V, int* out, cudaStream_t stream) {
  ArgmaxKernel<<<1, 1024, 0, stream>>>(logits, V, out);
  CUDA_CHECK_LAUNCH();
}

}  // namespace infer::kernels
