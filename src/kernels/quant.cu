// Weight-only quantization, done once at load time on the GPU.
//
// int8, per output row (per channel): scale = max|w| / 127, q = round(w / scale) in
// [-127, 127]. One scale per row is enough because each row of a projection is one output
// feature and its weights have one typical magnitude.
//
// int4, per group of 128 weights along a row: the value with the largest magnitude, m, maps to
// -8 (scale = m / -8), and q = round(w / scale) clamped to [-8, 7], stored as q + 8 in a
// nibble. That spends all 16 levels on the side of the range where the extreme value is, the
// same trick as llama.cpp's Q4_0 (which uses groups of 32). Smaller groups follow the weights'
// local magnitude more closely; 128 keeps the scales at 2 bytes per 128 weights (0.125 bits
// per weight on top of the 4).
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

// One warp per group of 128 weights: each lane handles 4 of them (two bytes of output).
__global__ void QuantizeInt4Kernel(const bf16* __restrict__ w, int rows, int cols,
                                   uint8_t* __restrict__ q, __half* __restrict__ scales) {
  const int groups_per_row = cols / kInt4Group;
  const long long g = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  if (g >= static_cast<long long>(rows) * groups_per_row) return;
  const int lane = threadIdx.x % 32;
  const bf16* src = w + g * kInt4Group + lane * 4;

  float v[4];
  for (int i = 0; i < 4; ++i) v[i] = __bfloat162float(src[i]);

  // The signed value with the largest magnitude in the group; on equal magnitudes the lowest
  // index wins (as torch.argmax does), which matters when m and -m both occur.
  float best = v[0];
  int best_idx = lane * 4;
  for (int i = 1; i < 4; ++i) {
    if (fabsf(v[i]) > fabsf(best)) {
      best = v[i];
      best_idx = lane * 4 + i;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float other = __shfl_xor_sync(0xffffffffu, best, o);
    const int other_idx = __shfl_xor_sync(0xffffffffu, best_idx, o);
    if (fabsf(other) > fabsf(best) || (fabsf(other) == fabsf(best) && other_idx < best_idx)) {
      best = other;
      best_idx = other_idx;
    }
  }
  const float scale = best != 0.0f ? best / -8.0f : 1.0f;
  if (lane == 0) scales[g] = __float2half(scale);

  uint8_t* dst = q + g * (kInt4Group / 2) + lane * 2;
  unsigned int nib[4];
  for (int i = 0; i < 4; ++i) {
    const float qv = fminf(fmaxf(rintf(v[i] / scale), -8.0f), 7.0f);
    nib[i] = static_cast<unsigned int>(static_cast<int>(qv) + 8);
  }
  dst[0] = static_cast<uint8_t>(nib[0] | (nib[1] << 4));
  dst[1] = static_cast<uint8_t>(nib[2] | (nib[3] << 4));
}

}  // namespace

void QuantizeInt8(const bf16* w, int rows, int cols, int8_t* q, float* scales,
                  cudaStream_t stream) {
  QuantizeInt8Kernel<<<rows, kThreads, 0, stream>>>(w, cols, q, scales);
  CUDA_CHECK_LAUNCH();
}

void QuantizeInt4(const bf16* w, int rows, int cols, uint8_t* q, __half* scales,
                  cudaStream_t stream) {
  if (cols % kInt4Group != 0) throw CudaError("int4: cols must be a multiple of 128");
  const long long groups = static_cast<long long>(rows) * (cols / kInt4Group);
  const int warps_per_block = kThreads / 32;
  const long long blocks = (groups + warps_per_block - 1) / warps_per_block;
  QuantizeInt4Kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(w, rows, cols, q,
                                                                                 scales);
  CUDA_CHECK_LAUNCH();
}

}  // namespace infer::kernels
