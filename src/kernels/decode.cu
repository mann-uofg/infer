// Decode-only kernels: fused RoPE + KV cache write, split-KV attention, and the position
// counter. All of them take the position from device memory so the whole decode step can be
// captured once as a CUDA graph and replayed.

#include <cuda_bf16.h>

#include <algorithm>
#include <cfloat>

#include "infer/cuda_util.h"
#include "infer/kernels.h"

namespace infer::kernels {

namespace {

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

// Threads [0, rot) rotate one (head, pair) of q or k; threads [rot, rot + kv_dim) copy one
// element of v. Rotated k goes to the cache; rotated q is written back in place.
__global__ void RopeKvDecodeKernel(float* qkv, int num_heads, int num_kv_heads, int head_dim,
                                   const float* inv_freq, const int* pos_ptr, int max_seq,
                                   bf16* k_cache, bf16* v_cache) {
  const int half = head_dim / 2;
  const int rot = (num_heads + num_kv_heads) * half;
  const int kv_dim = num_kv_heads * head_dim;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int pos = *pos_ptr;
  if (idx < rot) {
    const int head = idx / half;
    const int i = idx % half;
    const float angle = static_cast<float>(pos) * inv_freq[i];
    float s;
    float c;
    sincosf(angle, &s, &c);
    float* p = qkv + static_cast<size_t>(head) * head_dim;
    const float x1 = p[i];
    const float x2 = p[i + half];
    const float r1 = x1 * c - x2 * s;
    const float r2 = x2 * c + x1 * s;
    if (head < num_heads) {
      p[i] = r1;
      p[i + half] = r2;
    } else {
      const int kvh = head - num_heads;
      bf16* dst = k_cache + (static_cast<size_t>(kvh) * max_seq + pos) * head_dim;
      dst[i] = __float2bfloat16(r1);
      dst[i + half] = __float2bfloat16(r2);
    }
  } else if (idx < rot + kv_dim) {
    const int c = idx - rot;
    const int kvh = c / head_dim;
    const int d = c % head_dim;
    const float v = qkv[static_cast<size_t>(num_heads + num_kv_heads) * head_dim + c];
    v_cache[(static_cast<size_t>(kvh) * max_seq + pos) * head_dim + d] = __float2bfloat16(v);
  }
}

constexpr int kTile = 64;  // cache positions per shared-memory tile

// Grid (num_kv_heads, num_splits); block kGroup warps, warp h = query head kvh * kGroup + h,
// lane l owns output dimensions 2l and 2l + 1.
//
// For each tile of 64 positions: stage K and V in shared memory, compute the 64 scores of
// every head (each lane two of them), update the running max and sum (online softmax), and
// accumulate p * V. K rows are padded to 66 bf16 so that 32 lanes reading the same dimension
// of 32 different rows hit 32 different banks.
template <int kHeadDim, int kGroup>
__global__ void __launch_bounds__(kGroup * 32)
    AttentionDecodeSplitKernel(const float* __restrict__ q, const bf16* __restrict__ k_cache,
                               const bf16* __restrict__ v_cache, int max_seq,
                               const int* __restrict__ pos_ptr, int num_splits,
                               float* __restrict__ part_o, float* __restrict__ part_ml) {
  static_assert(kHeadDim == 64, "one lane owns two output dimensions");
  constexpr int kPad = kHeadDim + 2;
  __shared__ float qs[kGroup][kHeadDim];
  __shared__ __nv_bfloat162 ks[kTile][kPad / 2];
  __shared__ __nv_bfloat162 vs[kTile][kHeadDim / 2];
  __shared__ float probs[kGroup][kTile];

  const int kvh = blockIdx.x;
  const int split = blockIdx.y;
  const int h = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int qh = kvh * kGroup + h;
  const int num_heads = gridDim.x * kGroup;

  const int len = *pos_ptr + 1;
  const int chunk = (len + num_splits - 1) / num_splits;
  const int start = split * chunk;
  const int end = min(len, start + chunk);

  const float scale = rsqrtf(static_cast<float>(kHeadDim));
  qs[h][2 * lane] = q[qh * kHeadDim + 2 * lane] * scale;
  qs[h][2 * lane + 1] = q[qh * kHeadDim + 2 * lane + 1] * scale;

  const __nv_bfloat162* kbase = reinterpret_cast<const __nv_bfloat162*>(
      k_cache + static_cast<size_t>(kvh) * max_seq * kHeadDim);
  const __nv_bfloat162* vbase = reinterpret_cast<const __nv_bfloat162*>(
      v_cache + static_cast<size_t>(kvh) * max_seq * kHeadDim);

  float m = -FLT_MAX;
  float l = 0.0f;
  float o0 = 0.0f;
  float o1 = 0.0f;

  for (int t0 = start; t0 < end; t0 += kTile) {
    const int n = min(kTile, end - t0);
    __syncthreads();  // previous tile fully consumed (and qs written, on the first pass)
    // The n rows are contiguous in the cache: n * 32 bf16x2 words for K and for V.
    for (int w = threadIdx.x; w < n * (kHeadDim / 2); w += kGroup * 32) {
      const int j = w / (kHeadDim / 2);
      const int c = w % (kHeadDim / 2);
      ks[j][c] = kbase[static_cast<size_t>(t0) * (kHeadDim / 2) + w];
      vs[j][c] = vbase[static_cast<size_t>(t0) * (kHeadDim / 2) + w];
    }
    __syncthreads();

    // Scores for positions lane and lane + 32 of this tile.
    float s[2];
#pragma unroll
    for (int a = 0; a < 2; ++a) {
      const int j = lane + 32 * a;
      if (j < n) {
        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < kHeadDim / 2; ++c) {
          const float2 kk = __bfloat1622float2(ks[j][c]);
          acc = fmaf(qs[h][2 * c], kk.x, acc);
          acc = fmaf(qs[h][2 * c + 1], kk.y, acc);
        }
        s[a] = acc;
      } else {
        s[a] = -FLT_MAX;
      }
    }

    // Online softmax: rescale what has been accumulated so far to the new running max.
    const float m_new = fmaxf(m, WarpMax(fmaxf(s[0], s[1])));
    const float corr = __expf(m - m_new);
    const float p0 = lane < n ? __expf(s[0] - m_new) : 0.0f;
    const float p1 = lane + 32 < n ? __expf(s[1] - m_new) : 0.0f;
    probs[h][lane] = p0;
    probs[h][lane + 32] = p1;
    l = l * corr + WarpSum(p0 + p1);
    o0 *= corr;
    o1 *= corr;
    m = m_new;
    __syncwarp();

    for (int j = 0; j < n; ++j) {
      const float p = probs[h][j];
      const float2 v = __bfloat1622float2(vs[j][lane]);
      o0 = fmaf(p, v.x, o0);
      o1 = fmaf(p, v.y, o1);
    }
    __syncwarp();
  }

  // Unnormalised partial result for this split; empty splits leave l = 0.
  float* po = part_o + (static_cast<size_t>(qh) * num_splits + split) * kHeadDim;
  po[2 * lane] = o0;
  po[2 * lane + 1] = o1;
  if (lane == 0) {
    part_ml[(static_cast<size_t>(qh) * num_splits + split) * 2] = m;
    part_ml[(static_cast<size_t>(qh) * num_splits + split) * 2 + 1] = l;
  }
  (void)num_heads;
}

// One block per query head, one thread per output dimension. Rescales every split's partial
// sums to the global max and normalises.
__global__ void AttentionDecodeCombineKernel(const float* __restrict__ part_o,
                                             const float* __restrict__ part_ml, int num_splits,
                                             int head_dim, float* __restrict__ out) {
  const int qh = blockIdx.x;
  const int d = threadIdx.x;
  const float* ml = part_ml + static_cast<size_t>(qh) * num_splits * 2;
  float m = -FLT_MAX;
  for (int s = 0; s < num_splits; ++s) {
    if (ml[2 * s + 1] > 0.0f) m = fmaxf(m, ml[2 * s]);
  }
  float l = 0.0f;
  float o = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const float ls = ml[2 * s + 1];
    if (ls > 0.0f) {
      const float w = __expf(ml[2 * s] - m);
      l += ls * w;
      o += part_o[(static_cast<size_t>(qh) * num_splits + s) * head_dim + d] * w;
    }
  }
  out[static_cast<size_t>(qh) * head_dim + d] = o / l;
}

__global__ void AdvancePositionKernel(int* pos) { *pos += 1; }

__global__ void SetPositionKernel(int* pos, int value) { *pos = value; }

__global__ void StoreTokenKernel(const int* token, int* history, const int* pos) {
  history[*pos] = *token;
}

// Maps a float to an unsigned int that sorts the same way: flip the sign bit of positives,
// flip all bits of negatives.
__device__ __forceinline__ unsigned int OrderedBits(float f) {
  const unsigned int b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

constexpr int kArgmaxThreads = 256;

__global__ void __launch_bounds__(kArgmaxThreads)
    ArgmaxPartialKernel(const float* __restrict__ logits, int V, unsigned long long* best) {
  // Larger key = larger value, then lower index (the low half stores ~index).
  unsigned long long key = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += gridDim.x * blockDim.x) {
    const unsigned long long k =
        (static_cast<unsigned long long>(OrderedBits(logits[i])) << 32) | (0xffffffffu - i);
    key = k > key ? k : key;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const unsigned long long other = __shfl_xor_sync(0xffffffffu, key, o);
    key = other > key ? other : key;
  }
  __shared__ unsigned long long warp_best[kArgmaxThreads / 32];
  if (threadIdx.x % 32 == 0) warp_best[threadIdx.x / 32] = key;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < kArgmaxThreads / 32; ++w) key = warp_best[w] > key ? warp_best[w] : key;
    atomicMax(best, key);
  }
}

__global__ void ArgmaxFinishKernel(unsigned long long* best, int* token, int* history,
                                   const int* pos) {
  const int t = static_cast<int>(0xffffffffu - static_cast<unsigned int>(*best & 0xffffffffu));
  *token = t;
  history[*pos] = t;
  *best = 0;
}

}  // namespace

void RopeKvDecode(float* qkv, int num_heads, int num_kv_heads, int head_dim, const float* inv_freq,
                  const int* pos, int max_seq, bf16* k_cache, bf16* v_cache, cudaStream_t stream) {
  const int n = (num_heads + num_kv_heads) * (head_dim / 2) + num_kv_heads * head_dim;
  RopeKvDecodeKernel<<<(n + 255) / 256, 256, 0, stream>>>(qkv, num_heads, num_kv_heads, head_dim,
                                                          inv_freq, pos, max_seq, k_cache, v_cache);
  CUDA_CHECK_LAUNCH();
}

size_t AttentionDecodeWorkspace(int num_heads, int head_dim, int num_splits) {
  return static_cast<size_t>(num_heads) * num_splits * (head_dim + 2);
}

void AttentionDecode(const float* q, int num_heads, int num_kv_heads, int head_dim,
                     const bf16* k_cache, const bf16* v_cache, int max_seq, const int* pos,
                     int num_splits, float* workspace, float* out, cudaStream_t stream) {
  if (head_dim != 64) throw CudaError("attention decode: head_dim must be 64");
  float* part_o = workspace;
  float* part_ml = workspace + static_cast<size_t>(num_heads) * num_splits * head_dim;
  const dim3 grid(num_kv_heads, num_splits);
  switch (num_heads / num_kv_heads) {
    case 1:
      AttentionDecodeSplitKernel<64, 1>
          <<<grid, 32, 0, stream>>>(q, k_cache, v_cache, max_seq, pos, num_splits, part_o, part_ml);
      break;
    case 2:
      AttentionDecodeSplitKernel<64, 2>
          <<<grid, 64, 0, stream>>>(q, k_cache, v_cache, max_seq, pos, num_splits, part_o, part_ml);
      break;
    case 4:
      AttentionDecodeSplitKernel<64, 4><<<grid, 128, 0, stream>>>(q, k_cache, v_cache, max_seq, pos,
                                                                  num_splits, part_o, part_ml);
      break;
    case 8:
      AttentionDecodeSplitKernel<64, 8><<<grid, 256, 0, stream>>>(q, k_cache, v_cache, max_seq, pos,
                                                                  num_splits, part_o, part_ml);
      break;
    default:
      throw CudaError("attention decode: unsupported group size");
  }
  CUDA_CHECK_LAUNCH();
  AttentionDecodeCombineKernel<<<num_heads, head_dim, 0, stream>>>(part_o, part_ml, num_splits,
                                                                   head_dim, out);
  CUDA_CHECK_LAUNCH();
}

void AdvancePosition(int* pos, cudaStream_t stream) {
  AdvancePositionKernel<<<1, 1, 0, stream>>>(pos);
  CUDA_CHECK_LAUNCH();
}

void SetPosition(int* pos, int value, cudaStream_t stream) {
  SetPositionKernel<<<1, 1, 0, stream>>>(pos, value);
  CUDA_CHECK_LAUNCH();
}

void StoreToken(const int* token, int* history, const int* pos, cudaStream_t stream) {
  StoreTokenKernel<<<1, 1, 0, stream>>>(token, history, pos);
  CUDA_CHECK_LAUNCH();
}

void ArgmaxPartial(const float* logits, int V, unsigned long long* best, cudaStream_t stream) {
  // ~4 elements per thread: enough blocks to spread the 0.5 MB read over every SM.
  const int blocks = std::min(256, (V + kArgmaxThreads * 4 - 1) / (kArgmaxThreads * 4));
  ArgmaxPartialKernel<<<blocks, kArgmaxThreads, 0, stream>>>(logits, V, best);
  CUDA_CHECK_LAUNCH();
}

void ArgmaxFinish(unsigned long long* best, int* token, int* history, const int* pos,
                  cudaStream_t stream) {
  ArgmaxFinishKernel<<<1, 1, 0, stream>>>(best, token, history, pos);
  CUDA_CHECK_LAUNCH();
}

}  // namespace infer::kernels
