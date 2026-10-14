// Matrix-vector product for decode: y = W x, W [N, K] in bf16, int8 or int4, x fp32 [K].
//
// This is where decode time goes: every weight is read exactly once per token and used for
// one multiply-add, so the kernel's job is to stream W from DRAM as close to peak bandwidth
// as possible. How it does that:
//
//   - One warp per output row (two rows, or a gate/up pair, per warp). On each step of the
//     k loop, every lane loads 16 bytes of the row, so the warp reads 512 contiguous bytes:
//     fully coalesced 128-bit loads. 16 bytes is 8 bf16, 16 int8 or 32 int4 weights.
//   - The loop is unrolled so each thread has several 16-byte loads in flight. DRAM latency is
//     ~0.5-1 us; to keep ~850 GB/s busy the GPU needs ~0.5 MB in flight at all times.
//   - x is copied into shared memory once per block and read from there by all its warps.
//   - The per-lane partial sums are combined with warp shuffles; no shared memory, no atomics.
//
// Quantized weights are dequantized in registers right after the load: the math stays fp32
// and only the bytes read from DRAM shrink, which is exactly what a bandwidth-bound kernel
// wants. int8 has one scale per row, applied once to the finished dot product; int4 has one
// scale per 128 weights, applied to each lane's 32-weight partial sum.
//
// Fusions (each saves a kernel launch and a round trip of the activation through memory):
//   - norm_w != nullptr: x is RMS-normalised and scaled by norm_w while it is staged in shared
//     memory. Every block recomputes the norm of the same 2048 floats, which is cheap compared
//     with a separate launch.
//   - kAccumulate: y += W x, which folds in the residual connection.
//   - kSwiGlu: W is [2N, K] (gate rows, then up rows) and y[i] = silu(gate_i . x) * (up_i . x),
//     so the 16384-wide gate/up output is never written to memory.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "infer/cuda_util.h"
#include "infer/kernels.h"

namespace infer::kernels {

namespace {

constexpr int kMaxWarps = 8;  // warps per block

__device__ __forceinline__ float WarpSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Streaming load: the weights are read once per token, so there is no point keeping them in
// L1. (They still go through L2.)
__device__ __forceinline__ uint4 LoadStreaming(const void* p) {
  return __ldcs(reinterpret_cast<const uint4*>(p));
}

// ---- Weight formats. Each one says how many weights a 16-byte load covers, and how to dot
// ---- such a load with the matching slice of x.

struct Bf16Format {
  static constexpr int kPerLoad = 8;
  static constexpr bool kQuantized = false;
  struct Row {
    const bf16* w;
  };
  __device__ static Row MakeRow(const Linear& L, int row) {
    return {static_cast<const bf16*>(L.data) + static_cast<size_t>(row) * L.cols};
  }
  __device__ static uint4 Load(const Row& r, int k) { return LoadStreaming(r.w + k); }
  __device__ static float Dot(const Row&, const uint4& w, int, const float* x) {
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
  __device__ static float Finish(const Row&, float acc) { return acc; }
};

// Small integers to float without an int-to-float instruction (I2F runs at a fraction of the
// FMA rate and became the limit for the quantized kernels): put the integer u (0..255) in the
// low mantissa bits of 2^23, i.e. the bit pattern 0x4B000000 | u, which is the float
// 8388608 + u exactly, then subtract 8388608 + bias in one FADD. __byte_perm builds the bit
// pattern for byte i of a word in one instruction.
__device__ __forceinline__ float ByteToFloat(unsigned int word, unsigned int selector, float bias) {
  return __uint_as_float(__byte_perm(word, 0x4B000000u, selector)) - bias;
}
// Selector 0x754i puts byte i of the first operand under 0x4B (bytes 5, 6, 7 of the
// second operand are 0x00, 0x00, 0x4B).

struct Int8Format {
  static constexpr int kPerLoad = 16;
  static constexpr bool kQuantized = true;
  struct Row {
    const int8_t* w;
    float scale;
  };
  __device__ static Row MakeRow(const Linear& L, int row) {
    return {static_cast<const int8_t*>(L.data) + static_cast<size_t>(row) * L.cols,
            static_cast<const float*>(L.scales)[row]};
  }
  __device__ static uint4 Load(const Row& r, int k) { return LoadStreaming(r.w + k); }
  __device__ static float Dot(const Row&, const uint4& w, int, const float* x) {
    // Flipping the sign bit of each byte turns int8 q into unsigned q + 128.
    const unsigned int words[4] = {w.x ^ 0x80808080u, w.y ^ 0x80808080u, w.z ^ 0x80808080u,
                                   w.w ^ 0x80808080u};
    constexpr float kBias = 8388608.0f + 128.0f;
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 xv = *reinterpret_cast<const float4*>(x + 4 * i);
      s = fmaf(ByteToFloat(words[i], 0x7540u, kBias), xv.x, s);
      s = fmaf(ByteToFloat(words[i], 0x7541u, kBias), xv.y, s);
      s = fmaf(ByteToFloat(words[i], 0x7542u, kBias), xv.z, s);
      s = fmaf(ByteToFloat(words[i], 0x7543u, kBias), xv.w, s);
    }
    return s;
  }
  // The row scale is constant along the row, so it multiplies the finished sum once.
  __device__ static float Finish(const Row& r, float acc) { return acc * r.scale; }
};

struct Int4Format {
  static constexpr int kPerLoad = 32;
  static constexpr bool kQuantized = true;
  struct Row {
    const uint8_t* w;
    const __half* scales;
  };
  __device__ static Row MakeRow(const Linear& L, int row) {
    return {
        static_cast<const uint8_t*>(L.data) + static_cast<size_t>(row) * (L.cols / 2),
        static_cast<const __half*>(L.scales) + static_cast<size_t>(row) * (L.cols / kInt4Group)};
  }
  __device__ static uint4 Load(const Row& r, int k) { return LoadStreaming(r.w + k / 2); }
  // A lane's 32 weights never straddle a group (32 divides 128 and k is a multiple of 32),
  // so one scale applies to the whole partial sum.
  __device__ static float Dot(const Row& r, const uint4& w, int k, const float* x) {
    const unsigned int words[4] = {w.x, w.y, w.z, w.w};
    constexpr float kBias = 8388608.0f + 8.0f;  // stored nibbles are q + 8
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // Byte j of a word holds element 2j (low nibble) and 2j + 1 (high nibble). Split the
      // nibbles into two words of bytes, then convert byte by byte.
      const unsigned int lo = words[i] & 0x0F0F0F0Fu;
      const unsigned int hi = (words[i] >> 4) & 0x0F0F0F0Fu;
      const float4 xa = *reinterpret_cast<const float4*>(x + 8 * i);
      const float4 xb = *reinterpret_cast<const float4*>(x + 8 * i + 4);
      s = fmaf(ByteToFloat(lo, 0x7540u, kBias), xa.x, s);
      s = fmaf(ByteToFloat(hi, 0x7540u, kBias), xa.y, s);
      s = fmaf(ByteToFloat(lo, 0x7541u, kBias), xa.z, s);
      s = fmaf(ByteToFloat(hi, 0x7541u, kBias), xa.w, s);
      s = fmaf(ByteToFloat(lo, 0x7542u, kBias), xb.x, s);
      s = fmaf(ByteToFloat(hi, 0x7542u, kBias), xb.y, s);
      s = fmaf(ByteToFloat(lo, 0x7543u, kBias), xb.z, s);
      s = fmaf(ByteToFloat(hi, 0x7543u, kBias), xb.w, s);
    }
    return s * __half2float(r.scales[k / kInt4Group]);
  }
  __device__ static float Finish(const Row&, float acc) { return acc; }
};

// Copies x into shared memory, optionally applying RMSNorm with weight norm_w.
template <int kWarps>
__device__ void StageInput(const float* __restrict__ x, int K, const bf16* __restrict__ norm_w,
                           float eps, float* xs) {
  constexpr int kThreads = kWarps * 32;
  __shared__ float partial[kMaxWarps];
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

// Each warp reads kRows weight rows. For kStore/kAccumulate those are kRows consecutive
// outputs; for kSwiGlu they are kRows / 2 (gate, up) pairs, one output per pair.
template <class Fmt, int kWarps, int kRows, GemvOut kOut>
__global__ void __launch_bounds__(kWarps * 32)
    GemvKernel(const float* __restrict__ x, const Linear W, float* __restrict__ y,
               const bf16* __restrict__ norm_w, float eps) {
  extern __shared__ float xs[];
  constexpr bool kGlu = kOut == GemvOut::kSwiGlu;
  constexpr int kOutputs = kGlu ? kRows / 2 : kRows;
  constexpr int kStep = 32 * Fmt::kPerLoad;
  const int K = W.cols;
  const int n_out = kGlu ? W.rows / 2 : W.rows;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int out0 = (blockIdx.x * kWarps + warp) * kOutputs;

  typename Fmt::Row rows[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    // Clamp: rows past the end are computed and thrown away, which keeps the warp uniform.
    const int o = min(out0 + (kGlu ? r / 2 : r), n_out - 1);
    rows[r] = Fmt::MakeRow(W, kGlu && r % 2 == 1 ? n_out + o : o);
  }

  // The first loads do not depend on x, so issue them before staging it: their DRAM latency
  // then overlaps the copy of x and the norm instead of following it. With quantized weights a
  // block has little else to do, and this prologue was most of its time.
  const int k_first = lane * Fmt::kPerLoad;
  uint4 cur[kRows] = {};
  if (k_first < K) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) cur[r] = Fmt::Load(rows[r], k_first);
  }
  StageInput<kWarps>(x, K, norm_w, eps, xs);

  float acc[kRows] = {};
#pragma unroll 2
  for (int k = k_first; k < K; k += kStep) {
    // Load the next step while computing this one.
    uint4 next[kRows] = {};
    if (k + kStep < K) {
#pragma unroll
      for (int r = 0; r < kRows; ++r) next[r] = Fmt::Load(rows[r], k + kStep);
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] += Fmt::Dot(rows[r], cur[r], k, xs + k);
#pragma unroll
    for (int r = 0; r < kRows; ++r) cur[r] = next[r];
  }

  float v[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) v[r] = Fmt::Finish(rows[r], WarpSum(acc[r]));
  if (lane != 0) return;
#pragma unroll
  for (int o = 0; o < kOutputs; ++o) {
    if (out0 + o >= n_out) break;
    if constexpr (kGlu) {
      const float g = v[2 * o];
      const float u = v[2 * o + 1];
      y[out0 + o] = g / (1.0f + __expf(-g)) * u;
    } else if constexpr (kOut == GemvOut::kAccumulate) {
      y[out0 + o] += v[o];
    } else {
      y[out0 + o] = v[o];
    }
  }
}

template <class Fmt, int kWarps, int kRows>
void LaunchRows(const float* x, const Linear& W, float* y, GemvOut out, const bf16* norm_w,
                float eps, size_t smem, int blocks, cudaStream_t stream) {
  constexpr int threads = kWarps * 32;
  switch (out) {
    case GemvOut::kStore:
      GemvKernel<Fmt, kWarps, kRows, GemvOut::kStore>
          <<<blocks, threads, smem, stream>>>(x, W, y, norm_w, eps);
      break;
    case GemvOut::kAccumulate:
      GemvKernel<Fmt, kWarps, kRows, GemvOut::kAccumulate>
          <<<blocks, threads, smem, stream>>>(x, W, y, norm_w, eps);
      break;
    case GemvOut::kSwiGlu:
      if constexpr (kRows % 2 == 0) {
        GemvKernel<Fmt, kWarps, kRows, GemvOut::kSwiGlu>
            <<<blocks, threads, smem, stream>>>(x, W, y, norm_w, eps);
      }
      break;
  }
}

int SmCount() {
  static const int count = [] {
    int device = 0;
    int n = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device));
    return n;
  }();
  return count;
}

template <class Fmt>
void LaunchGemv(const float* x, const Linear& W, float* y, GemvOut out, const bf16* norm_w,
                float eps, cudaStream_t stream) {
  if (W.cols % Fmt::kPerLoad != 0) {
    throw CudaError("gemv: K must be a multiple of " + std::to_string(Fmt::kPerLoad));
  }
  const size_t smem = static_cast<size_t>(W.cols) * sizeof(float);
  if (smem > 48 * 1024) throw CudaError("gemv: K too large for the shared-memory input");

  // Rows per warp. Quantized formats take 4: a quantized row is few bytes (an int4 row of
  // 2048 is two loads per lane), so a warp needs several rows to keep enough bytes in flight;
  // measured, fewer rows were slower for them even when they gave more blocks. bf16 takes 2,
  // or 1 when 2 would leave fewer than two blocks per SM (the 2048-output projections).
  // (16 warps per block, to amortise staging x over more rows, was measured too and was slower.)
  const bool glu = out == GemvOut::kSwiGlu;
  const int n_out = glu ? W.rows / 2 : W.rows;
  auto blocks_for = [&](int rows) {
    const int outputs_per_block = kMaxWarps * (glu ? rows / 2 : rows);
    return (n_out + outputs_per_block - 1) / outputs_per_block;
  };
  if constexpr (Fmt::kQuantized) {
    LaunchRows<Fmt, kMaxWarps, 4>(x, W, y, out, norm_w, eps, smem, blocks_for(4), stream);
  } else if (glu || blocks_for(2) >= 2 * SmCount()) {
    LaunchRows<Fmt, kMaxWarps, 2>(x, W, y, out, norm_w, eps, smem, blocks_for(2), stream);
  } else {
    LaunchRows<Fmt, kMaxWarps, 1>(x, W, y, out, norm_w, eps, smem, blocks_for(1), stream);
  }
  CUDA_CHECK_LAUNCH();
}

}  // namespace

void Gemv(const float* x, const Linear& W, float* y, GemvOut out, const bf16* norm_w, float eps,
          cudaStream_t stream) {
  switch (W.format) {
    case WeightFormat::kBf16:
      LaunchGemv<Bf16Format>(x, W, y, out, norm_w, eps, stream);
      break;
    case WeightFormat::kInt8:
      LaunchGemv<Int8Format>(x, W, y, out, norm_w, eps, stream);
      break;
    case WeightFormat::kInt4:
      LaunchGemv<Int4Format>(x, W, y, out, norm_w, eps, stream);
      break;
  }
}

}  // namespace infer::kernels
