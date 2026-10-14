// Prefill kernels: many tokens at once.
//
// With T tokens every weight is reused T times, so prefill is limited by arithmetic, not by
// memory: a [T, K] x [K, N] product does 2*T*N*K flops while reading 2*N*K bytes of weights,
// about T flops per byte. Past roughly 60 flops per byte (the ratio of this GPU's fp32 peak to
// its bandwidth) the GPU's arithmetic is the limit, so the kernels here are about keeping
// operands in registers and shared memory and doing as many FMAs per load as possible.

#include <cuda_bf16.h>

#include <cfloat>

#include "infer/cuda_util.h"
#include "infer/kernels.h"

namespace infer::kernels {

namespace {

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

// ---- RMSNorm: one block per row ----

constexpr int kNormThreads = 256;

__global__ void __launch_bounds__(kNormThreads)
    RmsNormKernel(const float* __restrict__ x, const bf16* __restrict__ w, int H, float eps,
                  float* __restrict__ out) {
  const float* row = x + static_cast<size_t>(blockIdx.x) * H;
  float* dst = out + static_cast<size_t>(blockIdx.x) * H;
  float sum_sq = 0.0f;
  for (int i = threadIdx.x * 4; i < H; i += kNormThreads * 4) {
    const float4 v = *reinterpret_cast<const float4*>(row + i);
    sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  __shared__ float partial[kNormThreads / 32];
  sum_sq = WarpSum(sum_sq);
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum_sq;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int i = 0; i < kNormThreads / 32; ++i) total += partial[i];
  const float scale = rsqrtf(total / H + eps);
  for (int i = threadIdx.x * 4; i < H; i += kNormThreads * 4) {
    const float4 v = *reinterpret_cast<const float4*>(row + i);
    const __nv_bfloat162 w01 = *reinterpret_cast<const __nv_bfloat162*>(w + i);
    const __nv_bfloat162 w23 = *reinterpret_cast<const __nv_bfloat162*>(w + i + 2);
    const float2 a = __bfloat1622float2(w01);
    const float2 b = __bfloat1622float2(w23);
    *reinterpret_cast<float4*>(dst + i) =
        make_float4(v.x * scale * a.x, v.y * scale * a.y, v.z * scale * b.x, v.w * scale * b.y);
  }
}

// ---- GEMM: C[T, N] (+)= A[T, K] (fp32) x W[N, K]^T (bf16) ----
//
// Each block computes a BM x BN tile of C, each thread a TM x TN sub-tile held in registers.
// The k dimension is walked BK at a time: the block copies an A tile (BM x BK) and a W tile
// (BN x BK, converted to fp32) into shared memory, transposed so that for each k the values a
// thread needs (TM of A, TN of W) are contiguous and can be read with one 16-byte load each.
// Every element brought into shared memory is then used TN (or TM) times from registers.
// This is the 2D block-tiling scheme from Simon Boehm's SGEMM write-up (kernel 5), adapted
// to a bf16 right-hand side.
template <int BM, int BN, int BK, int TM, int TN, bool kAccumulate>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    GemmKernel(const float* __restrict__ A, const bf16* __restrict__ W, int T, int N, int K,
               float* __restrict__ C) {
  constexpr int kThreads = (BM / TM) * (BN / TN);
  static_assert(BM * BK / 4 == kThreads, "one float4 of A per thread per tile");
  static_assert(BN * BK / 4 == kThreads, "four bf16 of W per thread per tile");
  // +4 floats of padding per row spreads the transposed stores over more banks.
  __shared__ __align__(16) float As[BK][BM + 4];
  __shared__ __align__(16) float Ws[BK][BN + 4];

  const int tid = threadIdx.x;
  const int tx = tid % (BN / TN);  // which TN-wide column group of the tile
  const int ty = tid / (BN / TN);  // which TM-tall row group
  const int t0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;

  // Loader mapping: thread loads 4 consecutive k of one row of A and one row of W.
  const int lr = tid / (BK / 4);
  const int lk = (tid % (BK / 4)) * 4;

  float acc[TM][TN] = {};
  for (int k0 = 0; k0 < K; k0 += BK) {
    {
      const int t = t0 + lr;
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
      if (t < T) a = *reinterpret_cast<const float4*>(A + static_cast<size_t>(t) * K + k0 + lk);
      As[lk + 0][lr] = a.x;
      As[lk + 1][lr] = a.y;
      As[lk + 2][lr] = a.z;
      As[lk + 3][lr] = a.w;
      const int n = n0 + lr;
      float2 w01 = make_float2(0.f, 0.f);
      float2 w23 = make_float2(0.f, 0.f);
      if (n < N) {
        const uint2 raw = *reinterpret_cast<const uint2*>(W + static_cast<size_t>(n) * K + k0 + lk);
        w01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
        w23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
      }
      Ws[lk + 0][lr] = w01.x;
      Ws[lk + 1][lr] = w01.y;
      Ws[lk + 2][lr] = w23.x;
      Ws[lk + 3][lr] = w23.y;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float a[TM];
      float b[TN];
#pragma unroll
      for (int i = 0; i < TM; i += 4) {
        const float4 v = *reinterpret_cast<const float4*>(&As[k][ty * TM + i]);
        a[i] = v.x;
        a[i + 1] = v.y;
        a[i + 2] = v.z;
        a[i + 3] = v.w;
      }
#pragma unroll
      for (int j = 0; j < TN; j += 4) {
        const float4 v = *reinterpret_cast<const float4*>(&Ws[k][tx * TN + j]);
        b[j] = v.x;
        b[j + 1] = v.y;
        b[j + 2] = v.z;
        b[j + 3] = v.w;
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int t = t0 + ty * TM + i;
    if (t >= T) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int n = n0 + tx * TN + j;
      if (n >= N) continue;
      float* c = C + static_cast<size_t>(t) * N + n;
      if constexpr (kAccumulate) {
        *c += acc[i][j];
      } else {
        *c = acc[i][j];
      }
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
void LaunchGemm(const float* A, const bf16* W, int T, int N, int K, float* C, bool accumulate,
                cudaStream_t stream) {
  const dim3 grid((N + BN - 1) / BN, (T + BM - 1) / BM);
  constexpr int threads = (BM / TM) * (BN / TN);
  if (accumulate) {
    GemmKernel<BM, BN, BK, TM, TN, true><<<grid, threads, 0, stream>>>(A, W, T, N, K, C);
  } else {
    GemmKernel<BM, BN, BK, TM, TN, false><<<grid, threads, 0, stream>>>(A, W, T, N, K, C);
  }
  CUDA_CHECK_LAUNCH();
}

// ---- Prefill attention, FlashAttention style ----
//
// Grid (query tiles of 32, query heads); 4 warps, each owning 8 query rows; lane l owns output
// dimensions 2l and 2l + 1. Keys and values are read from the cache in tiles of 64 positions
// staged in shared memory. For each row the warp computes the 64 scores of the tile (two per
// lane), updates the row's running max and sum (online softmax: earlier partial sums are
// rescaled by exp(old max - new max)), and adds p * V. The T x T score matrix never exists
// anywhere; per row only the current tile's 64 scores do.
constexpr int kAttnRows = 32;
constexpr int kAttnWarps = 4;
constexpr int kRowsPerWarp = kAttnRows / kAttnWarps;
constexpr int kKeyTile = 64;

template <int kHeadDim>
__global__ void __launch_bounds__(kAttnWarps * 32)
    AttentionPrefillKernel(const float* __restrict__ qkv, int T, int pos0, int num_heads,
                           int num_kv_heads, const bf16* __restrict__ k_cache,
                           const bf16* __restrict__ v_cache, int max_seq, float* __restrict__ out) {
  static_assert(kHeadDim == 64, "one lane owns two output dimensions");
  __shared__ float qs[kAttnRows][kHeadDim];
  __shared__ __nv_bfloat162 ks[kKeyTile][kHeadDim / 2 + 1];  // +1 word: conflict-free columns
  __shared__ __nv_bfloat162 vs[kKeyTile][kHeadDim / 2];
  __shared__ float probs[kAttnWarps][kKeyTile];

  const int h = blockIdx.y;
  const int kvh = h / (num_heads / num_kv_heads);
  const int q0 = blockIdx.x * kAttnRows;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int stride = (num_heads + 2 * num_kv_heads) * kHeadDim;
  const float scale = rsqrtf(static_cast<float>(kHeadDim));

  for (int i = threadIdx.x; i < kAttnRows * kHeadDim; i += kAttnWarps * 32) {
    const int r = i / kHeadDim;
    const int d = i % kHeadDim;
    qs[r][d] =
        q0 + r < T ? qkv[static_cast<size_t>(q0 + r) * stride + h * kHeadDim + d] * scale : 0.0f;
  }

  float m[kRowsPerWarp];
  float l[kRowsPerWarp];
  float o0[kRowsPerWarp];
  float o1[kRowsPerWarp];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    m[r] = -FLT_MAX;
    l[r] = 0.0f;
    o0[r] = 0.0f;
    o1[r] = 0.0f;
  }

  const __nv_bfloat162* kbase = reinterpret_cast<const __nv_bfloat162*>(
      k_cache + static_cast<size_t>(kvh) * max_seq * kHeadDim);
  const __nv_bfloat162* vbase = reinterpret_cast<const __nv_bfloat162*>(
      v_cache + static_cast<size_t>(kvh) * max_seq * kHeadDim);
  // The last query of this tile sees keys 0..pos0 + its index.
  const int key_end = pos0 + min(T, q0 + kAttnRows);

  for (int t0 = 0; t0 < key_end; t0 += kKeyTile) {
    const int n = min(kKeyTile, key_end - t0);
    __syncthreads();
    for (int w = threadIdx.x; w < n * (kHeadDim / 2); w += kAttnWarps * 32) {
      const int j = w / (kHeadDim / 2);
      const int c = w % (kHeadDim / 2);
      ks[j][c] = kbase[static_cast<size_t>(t0) * (kHeadDim / 2) + w];
      vs[j][c] = vbase[static_cast<size_t>(t0) * (kHeadDim / 2) + w];
    }
    __syncthreads();

#pragma unroll 1
    for (int r = 0; r < kRowsPerWarp; ++r) {
      const int row = warp * kRowsPerWarp + r;
      const int limit = pos0 + q0 + row;  // last key position this query may see
      if (t0 > limit) continue;           // the whole tile is in this row's future

      float s[2];
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        const int j = lane + 32 * a;
        if (j < n && t0 + j <= limit) {
          float acc = 0.0f;
#pragma unroll
          for (int c = 0; c < kHeadDim / 2; ++c) {
            const float2 kk = __bfloat1622float2(ks[j][c]);
            acc = fmaf(qs[row][2 * c], kk.x, acc);
            acc = fmaf(qs[row][2 * c + 1], kk.y, acc);
          }
          s[a] = acc;
        } else {
          s[a] = -FLT_MAX;
        }
      }
      const float m_new = fmaxf(m[r], WarpMax(fmaxf(s[0], s[1])));
      const float corr = __expf(m[r] - m_new);
      const float p0 = s[0] > -FLT_MAX ? __expf(s[0] - m_new) : 0.0f;
      const float p1 = s[1] > -FLT_MAX ? __expf(s[1] - m_new) : 0.0f;
      probs[warp][lane] = p0;
      probs[warp][lane + 32] = p1;
      l[r] = l[r] * corr + WarpSum(p0 + p1);
      o0[r] *= corr;
      o1[r] *= corr;
      m[r] = m_new;
      __syncwarp();
      const int jn = min(n, limit - t0 + 1);
      for (int j = 0; j < jn; ++j) {
        const float p = probs[warp][j];
        const float2 v = __bfloat1622float2(vs[j][lane]);
        o0[r] = fmaf(p, v.x, o0[r]);
        o1[r] = fmaf(p, v.y, o1[r]);
      }
      __syncwarp();
    }
  }

#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    const int q = q0 + warp * kRowsPerWarp + r;
    if (q >= T) continue;
    float* dst = out + static_cast<size_t>(q) * num_heads * kHeadDim + h * kHeadDim;
    dst[2 * lane] = o0[r] / l[r];
    dst[2 * lane + 1] = o1[r] / l[r];
  }
}

}  // namespace

void RmsNorm(const float* x, const bf16* w, int T, int H, float eps, float* out,
             cudaStream_t stream) {
  if (H % 4 != 0) throw CudaError("rmsnorm: H must be a multiple of 4");
  RmsNormKernel<<<T, kNormThreads, 0, stream>>>(x, w, H, eps, out);
  CUDA_CHECK_LAUNCH();
}

void Gemm(const float* A, const bf16* W, int T, int N, int K, float* C, bool accumulate,
          cudaStream_t stream) {
  if (K % 16 != 0) throw CudaError("gemm: K must be a multiple of 16");
  // Big tiles reuse each loaded value more, but a short prompt would leave most SMs idle with
  // them, so fall back to 64 x 64 tiles when 128 x 128 would give fewer than ~2 waves.
  const long long big_tiles = static_cast<long long>((T + 127) / 128) * ((N + 127) / 128);
  if (big_tiles >= 140) {
    LaunchGemm<128, 128, 8, 8, 8>(A, W, T, N, K, C, accumulate, stream);
  } else {
    LaunchGemm<64, 64, 16, 4, 4>(A, W, T, N, K, C, accumulate, stream);
  }
}

void AttentionPrefill(const float* qkv, int T, int pos0, int num_heads, int num_kv_heads,
                      int head_dim, const bf16* k_cache, const bf16* v_cache, int max_seq,
                      float* out, cudaStream_t stream) {
  if (head_dim != 64) throw CudaError("attention prefill: head_dim must be 64");
  const dim3 grid((T + kAttnRows - 1) / kAttnRows, num_heads);
  AttentionPrefillKernel<64><<<grid, kAttnWarps * 32, 0, stream>>>(
      qkv, T, pos0, num_heads, num_kv_heads, k_cache, v_cache, max_seq, out);
  CUDA_CHECK_LAUNCH();
}

}  // namespace infer::kernels
