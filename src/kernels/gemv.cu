// Matrix-vector product for decode: y = W x with W bf16 [N, K] and x fp32 [K].
//
// This is where decode time goes: every weight is read exactly once per token and used for
// one multiply-add, so the kernel's job is to stream W from DRAM as close to peak bandwidth
// as possible. How it does that:
//
//   - One warp per output row (two rows, or a gate/up pair, per warp). On each step of the
//     k loop, lane l loads the 16 bytes (8 bf16) at k = (step * 32 + l) * 8, so the warp reads
//     512 contiguous bytes of the row: fully coalesced 128-bit loads.
//   - The loop is unrolled so each thread has several 16-byte loads in flight. DRAM latency is
//     ~0.5-1 us; to keep ~850 GB/s busy the GPU needs ~0.5 MB in flight at all times.
//   - x is copied into shared memory once per block and read from there by all its warps.
//   - The per-lane partial sums are combined with warp shuffles; no shared memory, no atomics.
//
// Fusions (each saves a kernel launch and a round trip of the activation through memory):
//   - norm_w != nullptr: x is RMS-normalised and scaled by norm_w while it is staged in shared
//     memory. Every block recomputes the norm of the same 2048 floats, which is cheap compared
//     with a separate launch.
//   - kAccumulate: y += W x, which folds in the residual connection.
//   - kSwiGlu: W is [2N, K] (gate rows, then up rows) and y[i] = silu(gate_i . x) * (up_i . x),
//     so the 16384-wide gate/up output is never written to memory.

#include <cuda_bf16.h>

#include "infer/cuda_util.h"
#include "infer/kernels.h"

namespace infer::kernels {

namespace {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;

__device__ __forceinline__ float WarpSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// 8 bf16 packed in a uint4, dotted with 8 floats from shared memory.
__device__ __forceinline__ float Dot8(const uint4& w, const float* x) {
  const float4 xa = *reinterpret_cast<const float4*>(x);
  const float4 xb = *reinterpret_cast<const float4*>(x + 4);
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&w);
  const float2 w0 = __bfloat1622float2(p[0]);
  const float2 w1 = __bfloat1622float2(p[1]);
  const float2 w2 = __bfloat1622float2(p[2]);
  const float2 w3 = __bfloat1622float2(p[3]);
  float s = w0.x * xa.x;
  s = fmaf(w0.y, xa.y, s);
  s = fmaf(w1.x, xa.z, s);
  s = fmaf(w1.y, xa.w, s);
  s = fmaf(w2.x, xb.x, s);
  s = fmaf(w2.y, xb.y, s);
  s = fmaf(w3.x, xb.z, s);
  s = fmaf(w3.y, xb.w, s);
  return s;
}

// Streaming load: the weights are read once per token, so there is no point keeping them in
// L1. (They still go through L2.)
__device__ __forceinline__ uint4 LoadWeights(const bf16* p) {
  return __ldcs(reinterpret_cast<const uint4*>(p));
}

// Copies x into shared memory, optionally applying RMSNorm with weight norm_w.
__device__ void StageInput(const float* __restrict__ x, int K, const bf16* __restrict__ norm_w,
                           float eps, float* xs) {
  __shared__ float partial[kWarps];
  float sum_sq = 0.0f;
  for (int i = threadIdx.x * 4; i < K; i += kThreads * 4) {
    const float4 v = *reinterpret_cast<const float4*>(x + i);
    *reinterpret_cast<float4*>(xs + i) = v;
    sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  if (norm_w == nullptr) {
    __syncthreads();
    return;
  }
  sum_sq = WarpSum(sum_sq);
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum_sq;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) total += partial[w];
  const float scale = rsqrtf(total / K + eps);
  for (int i = threadIdx.x; i < K; i += kThreads) {
    xs[i] = xs[i] * scale * __bfloat162float(norm_w[i]);
  }
  __syncthreads();
}

// Each warp computes kRows consecutive outputs.
template <int kRows, GemvOut kOut>
__global__ void __launch_bounds__(kThreads)
    GemvKernel(const float* __restrict__ x, const bf16* __restrict__ W, int N, int K,
               float* __restrict__ y, const bf16* __restrict__ norm_w, float eps) {
  extern __shared__ float xs[];
  StageInput(x, K, norm_w, eps, xs);

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row0 = (blockIdx.x * kWarps + warp) * kRows;
  if (row0 >= N) return;

  const bf16* rows[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    rows[r] = W + static_cast<size_t>(min(row0 + r, N - 1)) * K;  // clamp: tail rows are unused
  }

  float acc[kRows] = {};
#pragma unroll 4
  for (int k = lane * 8; k < K; k += 32 * 8) {
    uint4 w[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) w[r] = LoadWeights(rows[r] + k);
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] += Dot8(w[r], xs + k);
  }

#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const float v = WarpSum(acc[r]);
    if (lane == 0 && row0 + r < N) {
      if constexpr (kOut == GemvOut::kAccumulate) {
        y[row0 + r] += v;
      } else {
        y[row0 + r] = v;
      }
    }
  }
}

// One output per warp from a pair of rows: gate row i and up row N + i.
__global__ void __launch_bounds__(kThreads)
    GemvSwiGluKernel(const float* __restrict__ x, const bf16* __restrict__ W, int N, int K,
                     float* __restrict__ y, const bf16* __restrict__ norm_w, float eps) {
  extern __shared__ float xs[];
  StageInput(x, K, norm_w, eps, xs);

  const int lane = threadIdx.x % 32;
  const int i = blockIdx.x * kWarps + threadIdx.x / 32;
  if (i >= N) return;
  const bf16* gate = W + static_cast<size_t>(i) * K;
  const bf16* up = W + static_cast<size_t>(N + i) * K;

  float g = 0.0f;
  float u = 0.0f;
#pragma unroll 4
  for (int k = lane * 8; k < K; k += 32 * 8) {
    const uint4 wg = LoadWeights(gate + k);
    const uint4 wu = LoadWeights(up + k);
    g += Dot8(wg, xs + k);
    u += Dot8(wu, xs + k);
  }
  g = WarpSum(g);
  u = WarpSum(u);
  if (lane == 0) y[i] = g / (1.0f + __expf(-g)) * u;
}

}  // namespace

void Gemv(const float* x, const bf16* W, int N, int K, float* y, GemvOut out, const bf16* norm_w,
          float eps, cudaStream_t stream) {
  if (K % 8 != 0) throw CudaError("gemv: K must be a multiple of 8");
  const size_t smem = static_cast<size_t>(K) * sizeof(float);
  if (smem > 48 * 1024) throw CudaError("gemv: K too large for the shared-memory input");
  constexpr int kRows = 2;
  switch (out) {
    case GemvOut::kStore: {
      const int blocks = (N + kWarps * kRows - 1) / (kWarps * kRows);
      GemvKernel<kRows, GemvOut::kStore>
          <<<blocks, kThreads, smem, stream>>>(x, W, N, K, y, norm_w, eps);
      break;
    }
    case GemvOut::kAccumulate: {
      const int blocks = (N + kWarps * kRows - 1) / (kWarps * kRows);
      GemvKernel<kRows, GemvOut::kAccumulate>
          <<<blocks, kThreads, smem, stream>>>(x, W, N, K, y, norm_w, eps);
      break;
    }
    case GemvOut::kSwiGlu: {
      const int blocks = (N + kWarps - 1) / kWarps;
      GemvSwiGluKernel<<<blocks, kThreads, smem, stream>>>(x, W, N, K, y, norm_w, eps);
      break;
    }
  }
  CUDA_CHECK_LAUNCH();
}

}  // namespace infer::kernels
