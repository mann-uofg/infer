// Sampling with temperature, top-k and top-p, entirely on the GPU, in one block.
//
// Sorting 128K logits every token would be the obvious way and the slow one. Instead the
// cut-offs are found with a radix select on the logits' bit patterns:
//
//   1. Map each logit to an unsigned key that sorts the same way (OrderedBits).
//   2. top-k: find the k-th largest key exactly in four passes, one byte at a time from the
//      top: histogram the current byte of every key that matches the prefix found so far,
//      walk the 256 bins from the top until k is reached, fix that byte, repeat.
//   3. top-p: the same walk, but each bin holds the probability mass of its keys instead of a
//      count, and it stops when the mass reaches p (of the top-k mass).
//   4. Keep logits at or above both cut-offs, and pick one in proportion to exp(logit / T):
//      a block-wide prefix sum over contiguous index ranges finds where u * total lands.
//
// Ties at a cut-off are all kept, so top-k can keep slightly more than k tokens when logits
// are exactly equal.
//
// The uniform random number comes from device memory (one per position, generated on the host
// from a seed), so a sampled decode step is still a fixed graph and runs are reproducible.

#include <cfloat>

#include "infer/cuda_util.h"
#include "infer/kernels.h"

namespace infer::kernels {

namespace {

constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ unsigned int OrderedBits(float f) {
  const unsigned int b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ float WarpSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float WarpMax(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

template <bool kMax>
__device__ float BlockReduce(float v, float* scratch) {
  v = kMax ? WarpMax(v) : WarpSum(v);
  __syncthreads();  // scratch may still be in use by a previous reduction
  if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = v;
  __syncthreads();
  float r = kMax ? -FLT_MAX : 0.0f;
  for (int w = 0; w < kWarps; ++w) r = kMax ? fmaxf(r, scratch[w]) : r + scratch[w];
  return r;
}

// Per-warp histograms, padded so the 32 copies of bin b fall in different banks when merged.
constexpr int kHistStride = 257;

// Finds the largest key K such that the weight of keys >= K is at least `target`, where an
// element's weight is 1 (counting, for top-k) or its probability (for top-p). Only elements
// with key >= floor take part.
//
// Each warp counts into its own histogram, and the 32 are summed at the end of each pass. With
// one shared histogram every element was an atomicAdd on one of a handful of bins (logits of
// similar size share their top bits), and those atomics serialised: 0.8-1.6 ms per token.
template <bool kByMass>
__device__ unsigned int RadixSelect(const float* logits, int V, float target, unsigned int floor,
                                    float max_logit, float inv_temp, float* warp_hist,
                                    float* hist) {
  unsigned int prefix = 0;
  unsigned int prefix_mask = 0;
  __shared__ unsigned int chosen;
  __shared__ float above;  // weight of keys strictly above the current prefix
  if (threadIdx.x == 0) above = 0.0f;
  float* mine = warp_hist + (threadIdx.x / 32) * kHistStride;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = threadIdx.x; b < kWarps * kHistStride; b += kThreads) warp_hist[b] = 0.0f;
    __syncthreads();
    for (int i = threadIdx.x; i < V; i += kThreads) {
      const unsigned int key = OrderedBits(logits[i]);
      if (key < floor || (key & prefix_mask) != prefix) continue;
      const float w = kByMass ? __expf((logits[i] - max_logit) * inv_temp) : 1.0f;
      atomicAdd(&mine[(key >> shift) & 0xffu], w);
    }
    __syncthreads();
    if (threadIdx.x < 256) {
      float sum = 0.0f;
      for (int w = 0; w < kWarps; ++w) sum += warp_hist[w * kHistStride + threadIdx.x];
      hist[threadIdx.x] = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float acc = above;
      int b = 255;
      for (; b > 0; --b) {
        if (acc + hist[b] >= target) break;
        acc += hist[b];
      }
      chosen = static_cast<unsigned int>(b);
      above = acc;
    }
    __syncthreads();
    prefix |= chosen << shift;
    prefix_mask |= 0xffu << shift;
  }
  return prefix;
}

__global__ void __launch_bounds__(kThreads)
    SampleKernel(const float* __restrict__ logits, int V, float temperature, int top_k, float top_p,
                 const float* __restrict__ uniforms, const int* __restrict__ pos,
                 int* __restrict__ token, int* __restrict__ history) {
  __shared__ float scratch[kWarps];
  __shared__ float hist[256];
  __shared__ float warp_hist[kWarps * kHistStride];
  __shared__ float prefix[kThreads];

  float m = -FLT_MAX;
  for (int i = threadIdx.x; i < V; i += kThreads) m = fmaxf(m, logits[i]);
  m = BlockReduce<true>(m, scratch);
  const float inv_temp = 1.0f / temperature;

  // top-k cut-off (k >= V or k <= 0 means no cut).
  unsigned int floor_key = 0;
  if (top_k > 0 && top_k < V) {
    floor_key =
        RadixSelect<false>(logits, V, static_cast<float>(top_k), 0, m, inv_temp, warp_hist, hist);
  }

  // Mass of the top-k set, then the top-p cut-off within it.
  float mass = 0.0f;
  for (int i = threadIdx.x; i < V; i += kThreads) {
    if (OrderedBits(logits[i]) >= floor_key) mass += __expf((logits[i] - m) * inv_temp);
  }
  mass = BlockReduce<false>(mass, scratch);
  if (top_p < 1.0f) {
    floor_key = RadixSelect<true>(logits, V, top_p * mass, floor_key, m, inv_temp, warp_hist, hist);
  }

  // Each thread takes a contiguous range of indices, so thread order is index order.
  const int chunk = (V + kThreads - 1) / kThreads;
  const int begin = min(V, static_cast<int>(threadIdx.x) * chunk);
  const int end = min(V, begin + chunk);
  float local = 0.0f;
  for (int i = begin; i < end; ++i) {
    if (OrderedBits(logits[i]) >= floor_key) local += __expf((logits[i] - m) * inv_temp);
  }
  prefix[threadIdx.x] = local;
  __syncthreads();
  // Inclusive scan (Hillis-Steele) over the 1024 per-thread sums.
  for (int off = 1; off < kThreads; off <<= 1) {
    const float add = threadIdx.x >= off ? prefix[threadIdx.x - off] : 0.0f;
    __syncthreads();
    prefix[threadIdx.x] += add;
    __syncthreads();
  }
  const float total = prefix[kThreads - 1];
  const float target = uniforms[*pos] * total;
  const float before = threadIdx.x == 0 ? 0.0f : prefix[threadIdx.x - 1];

  __shared__ int picked;
  if (threadIdx.x == 0) picked = -1;
  __syncthreads();
  if (local > 0.0f && target >= before && target < prefix[threadIdx.x]) {
    float acc = before;
    int last = -1;
    for (int i = begin; i < end; ++i) {
      if (OrderedBits(logits[i]) < floor_key) continue;
      last = i;
      acc += __expf((logits[i] - m) * inv_temp);
      if (target < acc) break;
    }
    picked = last;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int t = picked;
    if (t < 0) {
      // u * total rounded up to total: take the last kept token.
      for (int i = V - 1; i >= 0; --i) {
        if (OrderedBits(logits[i]) >= floor_key) {
          t = i;
          break;
        }
      }
    }
    *token = t;
    history[*pos] = t;
  }
}

}  // namespace

void Sample(const float* logits, int V, float temperature, int top_k, float top_p,
            const float* uniforms, const int* pos, int* token, int* history, cudaStream_t stream) {
  if (!(temperature > 0.0f)) throw CudaError("sample: temperature must be > 0 (use argmax)");
  if (!(top_p > 0.0f && top_p <= 1.0f)) throw CudaError("sample: top_p must be in (0, 1]");
  SampleKernel<<<1, kThreads, 0, stream>>>(logits, V, temperature, top_k, top_p, uniforms, pos,
                                           token, history);
  CUDA_CHECK_LAUNCH();
}

}  // namespace infer::kernels
