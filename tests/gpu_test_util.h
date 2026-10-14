// Helpers for GPU kernel tests: moving data to and from the device, random inputs, and CPU
// reference implementations that every kernel is compared against. The CPU versions
// accumulate in double so they are a strictly better answer than any fp32 kernel.

#pragma once

#include <cuda_bf16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "infer/cuda_util.h"
#include "infer/kernels.h"

namespace infer::test {

using bf16 = __nv_bfloat16;

template <typename T>
DeviceBuffer<T> Upload(const std::vector<T>& host) {
  DeviceBuffer<T> d(host.size());
  CUDA_CHECK(cudaMemcpy(d.get(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

template <typename T>
std::vector<T> Download(const DeviceBuffer<T>& d) {
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<T> host(d.size());
  CUDA_CHECK(cudaMemcpy(host.data(), d.get(), d.size() * sizeof(T), cudaMemcpyDeviceToHost));
  return host;
}

inline std::vector<float> RandomFloats(size_t n, float scale, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, scale);
  std::vector<float> v(n);
  for (float& x : v) x = dist(rng);
  return v;
}

inline std::vector<bf16> ToBf16(const std::vector<float>& v) {
  std::vector<bf16> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = __float2bfloat16(v[i]);
  return out;
}

// The values the GPU will actually see after rounding to bf16.
inline std::vector<float> RoundTripBf16(const std::vector<float>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = __bfloat162float(__float2bfloat16(v[i]));
  return out;
}

// |got - want| <= atol + rtol * |want| for every element; reports the worst element.
inline void ExpectClose(const std::vector<float>& got, const std::vector<float>& want, float atol,
                        float rtol) {
  ASSERT_EQ(got.size(), want.size());
  size_t worst = 0;
  double worst_excess = -1.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double err = std::abs(static_cast<double>(got[i]) - want[i]);
    const double excess = err - (atol + rtol * std::abs(want[i]));
    if (excess > worst_excess) {
      worst_excess = excess;
      worst = i;
    }
  }
  EXPECT_LE(worst_excess, 0.0) << "element " << worst << ": got " << got[worst] << ", want "
                               << want[worst];
}

inline kernels::Linear Bf16Linear(const bf16* w, int rows, int cols) {
  return kernels::Linear{kernels::WeightFormat::kBf16, w, nullptr, rows, cols};
}

// ---- CPU reference ops (same conventions as include/infer/kernels.h) ----

inline std::vector<float> CpuRmsNorm(const std::vector<float>& x, const std::vector<float>& w,
                                     int T, int H, float eps) {
  std::vector<float> out(x.size());
  for (int t = 0; t < T; ++t) {
    double ss = 0;
    for (int k = 0; k < H; ++k) ss += static_cast<double>(x[t * H + k]) * x[t * H + k];
    const double scale = 1.0 / std::sqrt(ss / H + eps);
    for (int k = 0; k < H; ++k) out[t * H + k] = static_cast<float>(x[t * H + k] * scale * w[k]);
  }
  return out;
}

inline std::vector<float> CpuMatmul(const std::vector<float>& x, const std::vector<float>& W, int T,
                                    int N, int K) {
  std::vector<float> y(static_cast<size_t>(T) * N);
  for (int t = 0; t < T; ++t) {
    for (int n = 0; n < N; ++n) {
      double acc = 0;
      for (int k = 0; k < K; ++k) {
        acc += static_cast<double>(x[static_cast<size_t>(t) * K + k]) *
               W[static_cast<size_t>(n) * K + k];
      }
      y[static_cast<size_t>(t) * N + n] = static_cast<float>(acc);
    }
  }
  return y;
}

// Rotates q and k heads of each [q | k | v] row, positions pos0 + t.
inline void CpuRope(std::vector<float>& qkv, int T, int nh, int nkv, int hd,
                    const std::vector<float>& inv_freq, int pos0) {
  const int stride = (nh + 2 * nkv) * hd;
  const int half = hd / 2;
  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < nh + nkv; ++h) {
      float* p = &qkv[static_cast<size_t>(t) * stride + static_cast<size_t>(h) * hd];
      for (int i = 0; i < half; ++i) {
        const double angle = static_cast<double>(static_cast<float>(pos0 + t) * inv_freq[i]);
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        const double x1 = p[i];
        const double x2 = p[i + half];
        p[i] = static_cast<float>(x1 * c - x2 * s);
        p[i + half] = static_cast<float>(x2 * c + x1 * s);
      }
    }
  }
}

// Causal GQA. q: [T, nh*hd] rows with stride q_stride; k, v: rows with stride kv_stride.
// Query t attends to keys 0..t + past (past = number of earlier cached positions).
inline std::vector<float> CpuAttention(const float* q, int q_stride, const float* k, const float* v,
                                       int kv_stride, int T, int past, int nh, int nkv, int hd) {
  std::vector<float> out(static_cast<size_t>(T) * nh * hd);
  const int group = nh / nkv;
  const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < nh; ++h) {
      const int kvh = h / group;
      const float* qr = q + static_cast<size_t>(t) * q_stride + static_cast<size_t>(h) * hd;
      const int n = past + t + 1;
      std::vector<double> s(n);
      double m = -1e300;
      for (int j = 0; j < n; ++j) {
        const float* kr = k + static_cast<size_t>(j) * kv_stride + static_cast<size_t>(kvh) * hd;
        double dot = 0;
        for (int d = 0; d < hd; ++d) dot += static_cast<double>(qr[d]) * kr[d];
        s[j] = dot * scale;
        m = std::max(m, s[j]);
      }
      double denom = 0;
      for (int j = 0; j < n; ++j) {
        s[j] = std::exp(s[j] - m);
        denom += s[j];
      }
      for (int d = 0; d < hd; ++d) {
        double acc = 0;
        for (int j = 0; j < n; ++j) {
          acc += s[j] * v[static_cast<size_t>(j) * kv_stride + static_cast<size_t>(kvh) * hd + d];
        }
        out[static_cast<size_t>(t) * nh * hd + static_cast<size_t>(h) * hd + d] =
            static_cast<float>(acc / denom);
      }
    }
  }
  return out;
}

inline std::vector<float> CpuSwiGlu(const std::vector<float>& gate_up, int T, int I) {
  std::vector<float> out(static_cast<size_t>(T) * I);
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < I; ++i) {
      const double g = gate_up[static_cast<size_t>(t) * 2 * I + i];
      const double u = gate_up[static_cast<size_t>(t) * 2 * I + I + i];
      out[static_cast<size_t>(t) * I + i] = static_cast<float>(g / (1 + std::exp(-g)) * u);
    }
  }
  return out;
}

}  // namespace infer::test
