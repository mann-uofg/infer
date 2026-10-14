// Every kernel against its CPU reference.
//
// Tolerances: kernels accumulate in fp32 and the reference in double. For a dot product of
// length K with inputs of size ~1, fp32 rounding error grows roughly like sqrt(K) * 2^-24
// relative to the terms, so 1e-4 relative plus a small absolute term is comfortably loose for
// K up to 8192 while still catching any real indexing bug (which gives O(1) errors).

#include <gtest/gtest.h>

#include "gpu_test_util.h"
#include "infer/kernels.h"

namespace infer {
namespace {

using test::bf16;
using test::Download;
using test::ExpectClose;
using test::RandomFloats;
using test::RoundTripBf16;
using test::ToBf16;
using test::Upload;

TEST(KernelTest, Embedding) {
  const int V = 50, H = 72, T = 5;
  const auto table = RoundTripBf16(RandomFloats(V * H, 1.0f, 1));
  const std::vector<int> tokens = {3, 0, 49, 3, 17};
  auto d_table = Upload(ToBf16(table));
  auto d_tokens = Upload(tokens);
  DeviceBuffer<float> d_out(T * H);
  kernels::Embedding(d_table.get(), d_tokens.get(), T, H, d_out.get(), nullptr);
  const auto out = Download(d_out);
  for (int t = 0; t < T; ++t) {
    for (int k = 0; k < H; ++k) ASSERT_EQ(out[t * H + k], table[tokens[t] * H + k]);
  }
}

TEST(KernelTest, RmsNormNaive) {
  const int T = 7, H = 2048;
  const auto x = RandomFloats(T * H, 3.0f, 2);
  const auto w = RoundTripBf16(RandomFloats(H, 1.0f, 3));
  auto d_x = Upload(x);
  auto d_w = Upload(ToBf16(w));
  DeviceBuffer<float> d_out(T * H);
  kernels::RmsNormNaive(d_x.get(), d_w.get(), T, H, 1e-5f, d_out.get(), nullptr);
  ExpectClose(Download(d_out), test::CpuRmsNorm(x, w, T, H, 1e-5f), 1e-5f, 1e-5f);
}

class MatmulTest : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(MatmulTest, Naive) {
  const auto [T, N, K] = GetParam();
  const auto x = RandomFloats(static_cast<size_t>(T) * K, 1.0f, 4);
  const auto W = RoundTripBf16(RandomFloats(static_cast<size_t>(N) * K, 0.05f, 5));
  auto d_x = Upload(x);
  auto d_W = Upload(ToBf16(W));
  DeviceBuffer<float> d_y(static_cast<size_t>(T) * N);
  kernels::MatmulNaive(d_x.get(), d_W.get(), T, N, K, d_y.get(), false, nullptr);
  const auto want = test::CpuMatmul(x, W, T, N, K);
  ExpectClose(Download(d_y), want, 1e-4f, 1e-4f);

  // accumulate=true adds into what is already there.
  kernels::MatmulNaive(d_x.get(), d_W.get(), T, N, K, d_y.get(), true, nullptr);
  std::vector<float> twice(want.size());
  for (size_t i = 0; i < want.size(); ++i) twice[i] = 2 * want[i];
  ExpectClose(Download(d_y), twice, 2e-4f, 1e-4f);
}

INSTANTIATE_TEST_SUITE_P(Shapes, MatmulTest,
                         ::testing::Values(std::make_tuple(1, 3072, 2048),  // decode qkv
                                           std::make_tuple(1, 2048, 8192),  // decode down
                                           std::make_tuple(9, 1408, 512),   // tiny model
                                           std::make_tuple(3, 37, 131)));   // odd sizes

TEST(KernelTest, RopeNaive) {
  const int T = 6, nh = 8, nkv = 2, hd = 64, pos0 = 1000;
  const int stride = (nh + 2 * nkv) * hd;
  auto qkv = RandomFloats(T * stride, 1.0f, 6);
  std::vector<float> inv_freq(hd / 2);
  for (int i = 0; i < hd / 2; ++i)
    inv_freq[i] = static_cast<float>(std::pow(500000.0, -2.0 * i / hd));
  auto d_qkv = Upload(qkv);
  auto d_freq = Upload(inv_freq);
  kernels::RopeNaive(d_qkv.get(), T, nh, nkv, hd, d_freq.get(), pos0, nullptr);
  test::CpuRope(qkv, T, nh, nkv, hd, inv_freq, pos0);
  // v heads are untouched; q and k agree to sin/cos precision at angles up to ~1000 rad.
  ExpectClose(Download(d_qkv), qkv, 1e-5f, 1e-5f);
}

class AttentionTest : public ::testing::TestWithParam<std::tuple<int, int, int, int>> {};

TEST_P(AttentionTest, Naive) {
  const auto [T, nh, nkv, hd] = GetParam();
  const int stride = (nh + 2 * nkv) * hd;
  const auto qkv = RandomFloats(static_cast<size_t>(T) * stride, 1.0f, 7);
  auto d_qkv = Upload(qkv);
  DeviceBuffer<float> d_out(static_cast<size_t>(T) * nh * hd);
  kernels::AttentionNaive(d_qkv.get(), T, nh, nkv, hd, d_out.get(), nullptr);
  const float* q = qkv.data();
  const float* k = q + nh * hd;
  const float* v = k + nkv * hd;
  const auto want = test::CpuAttention(q, stride, k, v, stride, T, 0, nh, nkv, hd);
  ExpectClose(Download(d_out), want, 1e-5f, 1e-4f);
}

INSTANTIATE_TEST_SUITE_P(Shapes, AttentionTest,
                         ::testing::Values(std::make_tuple(1, 32, 8, 64),
                                           std::make_tuple(37, 32, 8, 64),
                                           std::make_tuple(130, 8, 2, 64),
                                           std::make_tuple(20, 4, 4, 128)));

TEST(KernelTest, SwiGlu) {
  const int T = 3, I = 1408;
  const auto gu = RandomFloats(T * 2 * I, 2.0f, 8);
  auto d_gu = Upload(gu);
  DeviceBuffer<float> d_out(T * I);
  kernels::SwiGlu(d_gu.get(), T, I, d_out.get(), nullptr);
  ExpectClose(Download(d_out), test::CpuSwiGlu(gu, T, I), 1e-6f, 1e-5f);
}

TEST(KernelTest, ArgmaxPicksLowestIndexOnTies) {
  std::vector<float> logits = RandomFloats(128256, 1.0f, 9);
  logits[777] = 50.0f;
  logits[100000] = 50.0f;
  auto d_logits = Upload(logits);
  DeviceBuffer<int> d_out(1);
  kernels::Argmax(d_logits.get(), static_cast<int>(logits.size()), d_out.get(), nullptr);
  EXPECT_EQ(Download(d_out)[0], 777);

  logits[5] = 60.0f;
  d_logits = Upload(logits);
  kernels::Argmax(d_logits.get(), static_cast<int>(logits.size()), d_out.get(), nullptr);
  EXPECT_EQ(Download(d_out)[0], 5);
}

}  // namespace
}  // namespace infer
