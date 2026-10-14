// Cross-entropy for perplexity: the negative log-likelihood of each target token.

#include <cfloat>

#include "infer/cuda_util.h"
#include "infer/kernels.h"

namespace infer::kernels {

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float WarpMax(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ float WarpSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// One block per row: nll = logsumexp(logits) - logits[target], with the max subtracted first
// so the exponentials cannot overflow.
__global__ void __launch_bounds__(kThreads)
    CrossEntropyKernel(const float* __restrict__ logits, int V, const int* __restrict__ targets,
                       float* __restrict__ nll) {
  const float* row = logits + static_cast<size_t>(blockIdx.x) * V;
  __shared__ float scratch[kThreads / 32];

  float m = -FLT_MAX;
  for (int i = threadIdx.x; i < V; i += kThreads) m = fmaxf(m, row[i]);
  m = WarpMax(m);
  if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = m;
  __syncthreads();
  m = -FLT_MAX;
  for (int w = 0; w < kThreads / 32; ++w) m = fmaxf(m, scratch[w]);
  __syncthreads();

  float sum = 0.0f;
  for (int i = threadIdx.x; i < V; i += kThreads) sum += expf(row[i] - m);
  sum = WarpSum(sum);
  if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int w = 0; w < kThreads / 32; ++w) total += scratch[w];
    nll[blockIdx.x] = logf(total) + m - row[targets[blockIdx.x]];
  }
}

}  // namespace

void CrossEntropy(const float* logits, int rows, int V, const int* targets, float* nll,
                  cudaStream_t stream) {
  CrossEntropyKernel<<<rows, kThreads, 0, stream>>>(logits, V, targets, nll);
  CUDA_CHECK_LAUNCH();
}

}  // namespace infer::kernels
