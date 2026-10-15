// Weight-only quantization, done once at load time on the GPU.
//
// int8, per output row (per channel): scale = max|w| / 127, q = round(w / scale) in
// [-127, 127]. One scale per row is enough because each row of a projection is one output
// feature and its weights have one typical magnitude.
//
// int4, per group of 32, 64 or 128 weights along a row: the value with the largest magnitude,
// m, maps to -8 (scale = m / -8), and q = round(w / scale) clamped to [-8, 7], stored as q + 8
// in a nibble. That spends all 16 levels on the side of the range where the extreme value is,
// the same trick as llama.cpp's Q4_0. Smaller groups follow the weights' local magnitude more
// closely at the cost of scale bytes: 2 bytes per group is 0.5 bits per weight at 32 (Q4_0's
// 4.5 bits in total) and 0.125 at 128. On Llama-3.2-1B the group size matters a lot for
// perplexity (see the README).
//
// Rounding is round-half-to-even (rintf), the same as torch.round, so the Python reference can
// reproduce the quantized weights bit for bit.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

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

// One block per row.
__global__ void QuantizeInt8Kernel(const bf16* __restrict__ w, int cols, int8_t* __restrict__ q,
                                   float* __restrict__ scales) {
  const bf16* row = w + static_cast<size_t>(blockIdx.x) * cols;
  float amax = 0.0f;
  for (int i = threadIdx.x; i < cols; i += kThreads) {
    amax = fmaxf(amax, fabsf(__bfloat162float(row[i])));
  }
  __shared__ float warp_max[kThreads / 32];
  amax = WarpMax(amax);
  if (threadIdx.x % 32 == 0) warp_max[threadIdx.x / 32] = amax;
  __syncthreads();
  amax = 0.0f;
  for (int i = 0; i < kThreads / 32; ++i) amax = fmaxf(amax, warp_max[i]);

  const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  if (threadIdx.x == 0) scales[blockIdx.x] = scale;
  int8_t* out = q + static_cast<size_t>(blockIdx.x) * cols;
  for (int i = threadIdx.x; i < cols; i += kThreads) {
    const float v = rintf(__bfloat162float(row[i]) / scale);
    out[i] = static_cast<int8_t>(fminf(fmaxf(v, -127.0f), 127.0f));
  }
}

// One thread per group: find the value with the largest magnitude (the first one on ties, like
// torch.argmax), derive the scale, write group / 2 bytes of nibbles and one fp16 scale.
__global__ void QuantizeInt4Kernel(const bf16* __restrict__ w, long long groups, int group,
                                   uint8_t* __restrict__ q, __half* __restrict__ scales) {
  const long long g = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= groups) return;
  const bf16* src = w + g * group;
  float best = __bfloat162float(src[0]);
  for (int i = 1; i < group; ++i) {
    const float v = __bfloat162float(src[i]);
    if (fabsf(v) > fabsf(best)) best = v;
  }
  const float scale = best != 0.0f ? best / -8.0f : 1.0f;
  scales[g] = __float2half(scale);
  uint8_t* dst = q + g * (group / 2);
  for (int i = 0; i < group; i += 2) {
    const float a = fminf(fmaxf(rintf(__bfloat162float(src[i]) / scale), -8.0f), 7.0f);
    const float b = fminf(fmaxf(rintf(__bfloat162float(src[i + 1]) / scale), -8.0f), 7.0f);
    dst[i / 2] = static_cast<uint8_t>(static_cast<unsigned int>(static_cast<int>(a) + 8) |
                                      (static_cast<unsigned int>(static_cast<int>(b) + 8) << 4));
  }
}

}  // namespace

void QuantizeInt8(const bf16* w, int rows, int cols, int8_t* q, float* scales,
                  cudaStream_t stream) {
  QuantizeInt8Kernel<<<rows, kThreads, 0, stream>>>(w, cols, q, scales);
  CUDA_CHECK_LAUNCH();
}

void QuantizeInt4(const bf16* w, int rows, int cols, int group, uint8_t* q, __half* scales,
                  cudaStream_t stream) {
  if (group != 32 && group != 64 && group != 128) {
    throw CudaError("int4: group must be 32, 64 or 128");
  }
  if (cols % group != 0) throw CudaError("int4: cols must be a multiple of the group size");
  const long long groups = static_cast<long long>(rows) * (cols / group);
  const long long blocks = (groups + kThreads - 1) / kThreads;
  QuantizeInt4Kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(w, groups, group,
                                                                                 q, scales);
  CUDA_CHECK_LAUNCH();
}

}  // namespace infer::kernels
