// Quantization kernels against a CPU implementation of the same rule (bit-exact), and the
// quantized GEMV/GEMM against a CPU product with the dequantized weights.

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "gpu_test_util.h"
#include "infer/kernels.h"

namespace infer {
namespace {

using kernels::Linear;
using kernels::WeightFormat;
using test::bf16;
using test::Download;
using test::ExpectClose;
using test::RandomFloats;
using test::RoundTripBf16;
using test::ToBf16;
using test::Upload;

struct CpuInt8 {
  std::vector<int8_t> q;
  std::vector<float> scales;
  std::vector<float> dequant;
};

CpuInt8 QuantizeInt8Cpu(const std::vector<float>& w, int rows, int cols) {
  CpuInt8 r;
  r.q.resize(w.size());
  r.dequant.resize(w.size());
  for (int i = 0; i < rows; ++i) {
    float amax = 0;
    for (int k = 0; k < cols; ++k)
      amax = std::max(amax, std::fabs(w[static_cast<size_t>(i) * cols + k]));
    const float s = amax > 0 ? amax / 127.0f : 1.0f;
    r.scales.push_back(s);
    for (int k = 0; k < cols; ++k) {
      const size_t idx = static_cast<size_t>(i) * cols + k;
      const float q = std::fmin(std::fmax(std::rint(w[idx] / s), -127.0f), 127.0f);
      r.q[idx] = static_cast<int8_t>(q);
      r.dequant[idx] = q * s;
    }
  }
  return r;
}

struct CpuInt4 {
  std::vector<uint8_t> packed;
  std::vector<__half> scales;
  std::vector<float> dequant;
};

CpuInt4 QuantizeInt4Cpu(const std::vector<float>& w, int rows, int cols) {
  CpuInt4 r;
  r.packed.resize(w.size() / 2);
  r.dequant.resize(w.size());
  const int G = kernels::kInt4Group;
  for (int i = 0; i < rows; ++i) {
    for (int g = 0; g < cols / G; ++g) {
      const size_t base = static_cast<size_t>(i) * cols + static_cast<size_t>(g) * G;
      float best = w[base];
      for (int k = 1; k < G; ++k) {
        if (std::fabs(w[base + k]) > std::fabs(best)) best = w[base + k];
      }
      const float s = best != 0 ? best / -8.0f : 1.0f;
      const __half hs = __float2half(s);
      r.scales.push_back(hs);
      for (int k = 0; k < G; ++k) {
        const float q = std::fmin(std::fmax(std::rint(w[base + k] / s), -8.0f), 7.0f);
        const unsigned nib = static_cast<unsigned>(static_cast<int>(q) + 8);
        uint8_t& byte = r.packed[(base + k) / 2];
        byte =
            static_cast<uint8_t>(k % 2 == 0 ? (byte & 0xf0u) | nib : (byte & 0x0fu) | (nib << 4));
        r.dequant[base + k] = q * __half2float(hs);
      }
    }
  }
  return r;
}

// Weights with some exact ties in magnitude and opposite sign inside int4 groups, which is
// where a careless "largest magnitude" picks the wrong sign.
std::vector<float> TestWeights(int rows, int cols, uint32_t seed) {
  auto w = RoundTripBf16(RandomFloats(static_cast<size_t>(rows) * cols, 0.05f, seed));
  for (int i = 0; i < rows; i += 3) {
    w[static_cast<size_t>(i) * cols + 5] = 0.25f;
    w[static_cast<size_t>(i) * cols + 9] = -0.25f;
  }
  if (rows > 1) {
    for (int k = 0; k < cols; ++k) w[static_cast<size_t>(cols) + k] = 0.0f;  // an all-zero row
  }
  return w;
}

TEST(QuantTest, Int8MatchesCpuBitForBit) {
  const int rows = 37, cols = 512;
  const auto w = TestWeights(rows, cols, 80);
  auto d_w = Upload(ToBf16(w));
  DeviceBuffer<int8_t> q(w.size());
  DeviceBuffer<float> scales(rows);
  kernels::QuantizeInt8(d_w.get(), rows, cols, q.get(), scales.get(), nullptr);
  const CpuInt8 cpu = QuantizeInt8Cpu(w, rows, cols);
  EXPECT_EQ(Download(q), cpu.q);
  EXPECT_EQ(Download(scales), cpu.scales);
}

TEST(QuantTest, Int4MatchesCpuBitForBit) {
  const int rows = 37, cols = 512;
  const auto w = TestWeights(rows, cols, 81);
  auto d_w = Upload(ToBf16(w));
  DeviceBuffer<uint8_t> q(w.size() / 2);
  DeviceBuffer<__half> scales(static_cast<size_t>(rows) * cols / kernels::kInt4Group);
  kernels::QuantizeInt4(d_w.get(), rows, cols, q.get(), scales.get(), nullptr);
  const CpuInt4 cpu = QuantizeInt4Cpu(w, rows, cols);
  EXPECT_EQ(Download(q), cpu.packed);
  const auto gs = Download(scales);
  for (size_t i = 0; i < gs.size(); ++i) {
    ASSERT_EQ(__half_as_ushort(gs[i]), __half_as_ushort(cpu.scales[i])) << i;
  }
}

TEST(QuantTest, QuantizationErrorIsBounded) {
  // Round-to-nearest leaves at most half a step of error: scale / 2 per weight.
  const int rows = 64, cols = 1024;
  const auto w = TestWeights(rows, cols, 82);
  const CpuInt8 q8 = QuantizeInt8Cpu(w, rows, cols);
  const CpuInt4 q4 = QuantizeInt4Cpu(w, rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int k = 0; k < cols; ++k) {
      const size_t idx = static_cast<size_t>(i) * cols + k;
      EXPECT_LE(std::fabs(q8.dequant[idx] - w[idx]), q8.scales[static_cast<size_t>(i)] * 0.5001f);
      const float s4 = __half2float(q4.scales[idx / kernels::kInt4Group]);
      // The extreme value maps to -8, so the other side of the range only reaches +7: a value
      // more than 7.5 steps out on that side is clamped and loses up to a whole step. Everything
      // else is within half a step (plus a little for the fp16 rounding of the scale).
      const float bound = (w[idx] / s4 > 7.5f ? 1.0f : 0.5f) * std::fabs(s4);
      EXPECT_LE(std::fabs(q4.dequant[idx] - w[idx]), bound + std::fabs(w[idx]) * 1e-3f) << idx;
    }
  }
}

struct Quantized {
  Linear linear;
  std::vector<float> dequant;
  DeviceBuffer<std::byte> data;
  DeviceBuffer<std::byte> scales;
};

Quantized Quantize(const std::vector<float>& w, int rows, int cols, WeightFormat f) {
  Quantized r;
  auto d_w = Upload(ToBf16(w));
  r.linear = Linear{f, nullptr, nullptr, rows, cols};
  if (f == WeightFormat::kInt8) {
    r.data = DeviceBuffer<std::byte>(w.size());
    r.scales = DeviceBuffer<std::byte>(rows * sizeof(float));
    kernels::QuantizeInt8(d_w.get(), rows, cols, reinterpret_cast<int8_t*>(r.data.get()),
                          reinterpret_cast<float*>(r.scales.get()), nullptr);
    r.dequant = QuantizeInt8Cpu(w, rows, cols).dequant;
  } else {
    r.data = DeviceBuffer<std::byte>(w.size() / 2);
    r.scales = DeviceBuffer<std::byte>(w.size() / kernels::kInt4Group * sizeof(__half));
    kernels::QuantizeInt4(d_w.get(), rows, cols, reinterpret_cast<uint8_t*>(r.data.get()),
                          reinterpret_cast<__half*>(r.scales.get()), nullptr);
    r.dequant = QuantizeInt4Cpu(w, rows, cols).dequant;
  }
  r.linear.data = r.data.get();
  r.linear.scales = r.scales.get();
  return r;
}

class QuantGemvTest : public ::testing::TestWithParam<std::tuple<WeightFormat, int, int>> {};

TEST_P(QuantGemvTest, MatchesDequantizedReference) {
  const auto [fmt, N, K] = GetParam();
  const auto x = RandomFloats(K, 1.0f, 83);
  const auto nw = RoundTripBf16(RandomFloats(K, 1.0f, 84));
  auto d_x = Upload(x);
  auto d_nw = Upload(ToBf16(nw));
  const auto normed = test::CpuRmsNorm(x, nw, 1, K, 1e-5f);

  const Quantized q = Quantize(TestWeights(N, K, 85), N, K, fmt);
  DeviceBuffer<float> d_y(N);
  kernels::Gemv(d_x.get(), q.linear, d_y.get(), kernels::GemvOut::kStore, d_nw.get(), 1e-5f,
                nullptr);
  ExpectClose(Download(d_y), test::CpuMatmul(normed, q.dequant, 1, N, K), 1e-4f, 1e-4f);

  const auto y0 = RandomFloats(N, 1.0f, 86);
  d_y = Upload(y0);
  kernels::Gemv(d_x.get(), q.linear, d_y.get(), kernels::GemvOut::kAccumulate, nullptr, 0.0f,
                nullptr);
  auto want = test::CpuMatmul(x, q.dequant, 1, N, K);
  for (int i = 0; i < N; ++i) want[i] += y0[i];
  ExpectClose(Download(d_y), want, 1e-4f, 1e-4f);

  const Quantized gu = Quantize(TestWeights(2 * N, K, 87), 2 * N, K, fmt);
  kernels::Gemv(d_x.get(), gu.linear, d_y.get(), kernels::GemvOut::kSwiGlu, d_nw.get(), 1e-5f,
                nullptr);
  ExpectClose(Download(d_y),
              test::CpuSwiGlu(test::CpuMatmul(normed, gu.dequant, 1, 2 * N, K), 1, N), 1e-4f,
              1e-4f);
}

TEST_P(QuantGemvTest, GemmMatchesDequantizedReference) {
  const auto [fmt, N, K] = GetParam();
  for (int T : {3, 300}) {
    const auto x = RandomFloats(static_cast<size_t>(T) * K, 1.0f, 88);
    auto d_x = Upload(x);
    const Quantized q = Quantize(TestWeights(N, K, 89), N, K, fmt);
    DeviceBuffer<float> d_y(static_cast<size_t>(T) * N);
    kernels::Gemm(d_x.get(), q.linear, T, d_y.get(), false, nullptr);
    ExpectClose(Download(d_y), test::CpuMatmul(x, q.dequant, T, N, K), 1e-4f, 1e-4f);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, QuantGemvTest,
    ::testing::Values(std::make_tuple(WeightFormat::kInt8, 3072, 2048),
                      std::make_tuple(WeightFormat::kInt8, 2048, 8192),
                      std::make_tuple(WeightFormat::kInt8, 33, 1408),
                      std::make_tuple(WeightFormat::kInt4, 3072, 2048),
                      std::make_tuple(WeightFormat::kInt4, 2048, 8192),
                      std::make_tuple(WeightFormat::kInt4, 33, 1408)),
    [](const ::testing::TestParamInfo<std::tuple<WeightFormat, int, int>>& info) {
      // No structured binding here: its commas would split the macro arguments.
      const WeightFormat f = std::get<0>(info.param);
      return std::string(f == WeightFormat::kInt8 ? "int8" : "int4") + "_" +
             std::to_string(std::get<1>(info.param)) + "x" +
             std::to_string(std::get<2>(info.param));
    });

}  // namespace
}  // namespace infer
