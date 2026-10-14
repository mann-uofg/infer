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

// A KV cache holding `past` earlier positions plus T new ones, written by KvCacheWrite in two
// calls (so the pos0 offset is exercised), and the expected attention output for the T new
// queries. Shared by the naive and optimized attention tests.
struct AttentionCase {
  int T, past, nh, nkv, hd, max_seq, stride;
  DeviceBuffer<float> d_all;  // qkv rows for all past + T positions
  DeviceBuffer<bf16> k_cache, v_cache;
  std::vector<float> want;

  AttentionCase(int T_, int past_, int nh_, int nkv_, int hd_, uint32_t seed)
      : T(T_),
        past(past_),
        nh(nh_),
        nkv(nkv_),
        hd(hd_),
        max_seq(past_ + T_ + 5),
        stride((nh_ + 2 * nkv_) * hd_) {
    const int P = past + T;
    const auto all = RandomFloats(static_cast<size_t>(P) * stride, 1.0f, seed);
    d_all = Upload(all);
    k_cache = DeviceBuffer<bf16>(static_cast<size_t>(nkv) * max_seq * hd);
    v_cache = DeviceBuffer<bf16>(static_cast<size_t>(nkv) * max_seq * hd);
    if (past > 0) {
      kernels::KvCacheWrite(d_all.get(), past, nh, nkv, hd, 0, max_seq, k_cache.get(),
                            v_cache.get(), nullptr);
    }
    kernels::KvCacheWrite(new_rows(), T, nh, nkv, hd, past, max_seq, k_cache.get(), v_cache.get(),
                          nullptr);

    // The kernel sees k and v rounded to bf16 by the cache; q stays fp32.
    const auto rounded = RoundTripBf16(all);
    const float* q = all.data() + static_cast<size_t>(past) * stride;
    const float* k = rounded.data() + static_cast<size_t>(nh) * hd;
    const float* v = k + static_cast<size_t>(nkv) * hd;
    want = test::CpuAttention(q, stride, k, v, stride, T, past, nh, nkv, hd);
  }

  const float* new_rows() const { return d_all.get() + static_cast<size_t>(past) * stride; }
};

class AttentionTest : public ::testing::TestWithParam<std::tuple<int, int, int, int, int>> {};

TEST_P(AttentionTest, Naive) {
  const auto [T, past, nh, nkv, hd] = GetParam();
  AttentionCase c(T, past, nh, nkv, hd, 7);
  DeviceBuffer<float> d_out(static_cast<size_t>(T) * nh * hd);
  kernels::AttentionNaive(c.new_rows(), T, past, nh, nkv, hd, c.k_cache.get(), c.v_cache.get(),
                          c.max_seq, d_out.get(), nullptr);
  ExpectClose(Download(d_out), c.want, 1e-5f, 1e-4f);
}

// (T, past, num_heads, num_kv_heads, head_dim)
INSTANTIATE_TEST_SUITE_P(Shapes, AttentionTest,
                         ::testing::Values(std::make_tuple(1, 0, 32, 8, 64),    // first token
                                           std::make_tuple(1, 300, 32, 8, 64),  // decode
                                           std::make_tuple(37, 0, 32, 8, 64),   // prefill
                                           std::make_tuple(17, 50, 8, 2, 64),   // chunked
                                           std::make_tuple(130, 0, 8, 2, 64),
                                           std::make_tuple(20, 3, 4, 4, 128)));

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

// ---- Optimized decode kernels ----

namespace infer {
namespace {

using test::bf16;
using test::Download;
using test::ExpectClose;
using test::RandomFloats;
using test::RoundTripBf16;
using test::ToBf16;
using test::Upload;

class GemvTest : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(GemvTest, StoreAccumulateAndFusedNorm) {
  const auto [N, K] = GetParam();
  const auto x = RandomFloats(K, 1.0f, 20);
  const auto W = RoundTripBf16(RandomFloats(static_cast<size_t>(N) * K, 0.05f, 21));
  const auto nw = RoundTripBf16(RandomFloats(K, 1.0f, 22));
  const auto y0 = RandomFloats(N, 1.0f, 23);
  auto d_x = Upload(x);
  auto d_W = Upload(ToBf16(W));
  auto d_nw = Upload(ToBf16(nw));
  const auto Wx = test::CpuMatmul(x, W, 1, N, K);

  DeviceBuffer<float> d_y(N);
  kernels::Gemv(d_x.get(), test::Bf16Linear(d_W.get(), N, K), d_y.get(), kernels::GemvOut::kStore,
                nullptr, 0.0f, nullptr);
  ExpectClose(Download(d_y), Wx, 1e-4f, 1e-4f);

  d_y = Upload(y0);
  kernels::Gemv(d_x.get(), test::Bf16Linear(d_W.get(), N, K), d_y.get(),
                kernels::GemvOut::kAccumulate, nullptr, 0.0f, nullptr);
  std::vector<float> want(N);
  for (int i = 0; i < N; ++i) want[i] = y0[i] + Wx[i];
  ExpectClose(Download(d_y), want, 1e-4f, 1e-4f);

  kernels::Gemv(d_x.get(), test::Bf16Linear(d_W.get(), N, K), d_y.get(), kernels::GemvOut::kStore,
                d_nw.get(), 1e-5f, nullptr);
  const auto normed = test::CpuRmsNorm(x, nw, 1, K, 1e-5f);
  ExpectClose(Download(d_y), test::CpuMatmul(normed, W, 1, N, K), 1e-4f, 1e-4f);
}

TEST_P(GemvTest, SwiGlu) {
  const auto [N, K] = GetParam();
  const auto x = RandomFloats(K, 1.0f, 24);
  const auto W = RoundTripBf16(RandomFloats(static_cast<size_t>(2 * N) * K, 0.05f, 25));
  const auto nw = RoundTripBf16(RandomFloats(K, 1.0f, 26));
  auto d_x = Upload(x);
  auto d_W = Upload(ToBf16(W));
  auto d_nw = Upload(ToBf16(nw));
  DeviceBuffer<float> d_y(N);
  kernels::Gemv(d_x.get(), test::Bf16Linear(d_W.get(), 2 * N, K), d_y.get(),
                kernels::GemvOut::kSwiGlu, d_nw.get(), 1e-5f, nullptr);
  const auto normed = test::CpuRmsNorm(x, nw, 1, K, 1e-5f);
  const auto gu = test::CpuMatmul(normed, W, 1, 2 * N, K);
  ExpectClose(Download(d_y), test::CpuSwiGlu(gu, 1, N), 1e-4f, 1e-4f);
}

// (N, K): Llama-1B decode shapes, the tiny model's MLP (K not a multiple of 256), odd N.
INSTANTIATE_TEST_SUITE_P(Shapes, GemvTest,
                         ::testing::Values(std::make_tuple(3072, 2048), std::make_tuple(2048, 8192),
                                           std::make_tuple(512, 1408), std::make_tuple(1001, 512),
                                           std::make_tuple(17, 64)));

TEST(DecodeKernelTest, RopeKvDecodeMatchesNaiveRopeAndCacheWrite) {
  const int nh = 32, nkv = 8, hd = 64, max_seq = 40, pos = 37;
  const int stride = (nh + 2 * nkv) * hd;
  const auto row = RandomFloats(stride, 1.0f, 27);
  std::vector<float> inv_freq(hd / 2);
  for (int i = 0; i < hd / 2; ++i)
    inv_freq[i] = static_cast<float>(std::pow(500000.0, -2.0 * i / hd));
  auto d_freq = Upload(inv_freq);
  auto d_pos = Upload(std::vector<int>{pos});
  const size_t cache = static_cast<size_t>(nkv) * max_seq * hd;

  // Optimized: one fused kernel.
  auto d_a = Upload(row);
  DeviceBuffer<bf16> ka(cache), va(cache);
  CUDA_CHECK(cudaMemset(ka.get(), 0, ka.bytes()));
  CUDA_CHECK(cudaMemset(va.get(), 0, va.bytes()));
  kernels::RopeKvDecode(d_a.get(), nh, nkv, hd, d_freq.get(), d_pos.get(), max_seq, ka.get(),
                        va.get(), nullptr);

  // Naive: rope, then cache write.
  auto d_b = Upload(row);
  DeviceBuffer<bf16> kb(cache), vb(cache);
  CUDA_CHECK(cudaMemset(kb.get(), 0, kb.bytes()));
  CUDA_CHECK(cudaMemset(vb.get(), 0, vb.bytes()));
  kernels::RopeNaive(d_b.get(), 1, nh, nkv, hd, d_freq.get(), pos, nullptr);
  kernels::KvCacheWrite(d_b.get(), 1, nh, nkv, hd, pos, max_seq, kb.get(), vb.get(), nullptr);

  // q must match exactly (same arithmetic); k and v in the cache too.
  const auto qa = Download(d_a);
  const auto qb = Download(d_b);
  for (int i = 0; i < nh * hd; ++i) ASSERT_EQ(qa[i], qb[i]) << i;
  const auto kav = Download(ka), kbv = Download(kb), vav = Download(va), vbv = Download(vb);
  for (size_t i = 0; i < cache; ++i) {
    ASSERT_EQ(__bfloat162float(kav[i]), __bfloat162float(kbv[i])) << i;
    ASSERT_EQ(__bfloat162float(vav[i]), __bfloat162float(vbv[i])) << i;
  }
}

class AttentionDecodeTest : public ::testing::TestWithParam<std::tuple<int, int, int, int>> {};

TEST_P(AttentionDecodeTest, SplitKvMatchesReference) {
  const auto [past, nh, nkv, splits] = GetParam();
  const int hd = 64;
  AttentionCase c(1, past, nh, nkv, hd, 30);
  auto d_pos = Upload(std::vector<int>{past});
  DeviceBuffer<float> ws(kernels::AttentionDecodeWorkspace(nh, hd, splits));
  DeviceBuffer<float> d_out(static_cast<size_t>(nh) * hd);
  kernels::AttentionDecode(c.new_rows(), nh, nkv, hd, c.k_cache.get(), c.v_cache.get(), c.max_seq,
                           d_pos.get(), splits, ws.get(), d_out.get(), nullptr);
  ExpectClose(Download(d_out), c.want, 1e-5f, 1e-4f);
}

// (past positions, num_heads, num_kv_heads, num_splits). Covers: a single position, splits
// with no positions at all (past + 1 < splits), tiles that are partly filled, and long
// contexts.
INSTANTIATE_TEST_SUITE_P(
    Shapes, AttentionDecodeTest,
    ::testing::Values(std::make_tuple(0, 32, 8, 16), std::make_tuple(5, 32, 8, 16),
                      std::make_tuple(100, 32, 8, 1), std::make_tuple(300, 32, 8, 16),
                      std::make_tuple(2047, 32, 8, 16), std::make_tuple(777, 8, 2, 7),
                      std::make_tuple(64, 8, 8, 3), std::make_tuple(129, 16, 2, 4)));

TEST(DecodeKernelTest, ArgmaxPartialFindsMaxLowestIndexAndResets) {
  const int V = 128256;
  DeviceBuffer<unsigned long long> best(1);
  CUDA_CHECK(cudaMemset(best.get(), 0, best.bytes()));
  DeviceBuffer<int> token(1);
  DeviceBuffer<int> history(4);
  CUDA_CHECK(cudaMemset(history.get(), 0, history.bytes()));
  auto d_pos = Upload(std::vector<int>{2});

  auto run = [&](const std::vector<float>& logits) {
    auto d = Upload(logits);
    kernels::ArgmaxPartial(d.get(), static_cast<int>(logits.size()), best.get(), nullptr);
    kernels::ArgmaxFinish(best.get(), token.get(), history.get(), d_pos.get(), nullptr);
    const int t = Download(token)[0];
    EXPECT_EQ(Download(history)[2], t);
    EXPECT_EQ(Download(best)[0], 0ull) << "best must be reset for the next step";
    return t;
  };

  std::vector<float> logits = RandomFloats(V, 1.0f, 40);
  logits[90000] = 9.0f;
  logits[1234] = 9.0f;  // tie: the lower index wins
  EXPECT_EQ(run(logits), 1234);

  // All negative, max at the very end.
  for (float& x : logits) x = -std::abs(x) - 1.0f;
  logits[V - 1] = -0.5f;
  EXPECT_EQ(run(logits), V - 1);

  // Index 0, and a vocabulary that is not a multiple of the block size.
  std::vector<float> small = RandomFloats(4099, 1.0f, 41);
  small[0] = 100.0f;
  EXPECT_EQ(run(small), 0);
}

// ---- Optimized prefill kernels ----

TEST(PrefillKernelTest, RmsNormMatchesReference) {
  const int T = 9, H = 2048;
  const auto x = RandomFloats(T * H, 3.0f, 50);
  const auto w = RoundTripBf16(RandomFloats(H, 1.0f, 51));
  auto d_x = Upload(x);
  auto d_w = Upload(ToBf16(w));
  DeviceBuffer<float> d_out(T * H);
  kernels::RmsNorm(d_x.get(), d_w.get(), T, H, 1e-5f, d_out.get(), nullptr);
  ExpectClose(Download(d_out), test::CpuRmsNorm(x, w, T, H, 1e-5f), 1e-5f, 1e-5f);
}

class GemmTest : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(GemmTest, MatchesReference) {
  const auto [T, N, K] = GetParam();
  const auto x = RandomFloats(static_cast<size_t>(T) * K, 1.0f, 52);
  const auto W = RoundTripBf16(RandomFloats(static_cast<size_t>(N) * K, 0.05f, 53));
  const auto y0 = RandomFloats(static_cast<size_t>(T) * N, 1.0f, 54);
  auto d_x = Upload(x);
  auto d_W = Upload(ToBf16(W));
  const auto want = test::CpuMatmul(x, W, T, N, K);

  DeviceBuffer<float> d_y(static_cast<size_t>(T) * N);
  kernels::Gemm(d_x.get(), test::Bf16Linear(d_W.get(), N, K), T, d_y.get(), false, nullptr);
  ExpectClose(Download(d_y), want, 1e-4f, 1e-4f);

  d_y = Upload(y0);
  kernels::Gemm(d_x.get(), test::Bf16Linear(d_W.get(), N, K), T, d_y.get(), true, nullptr);
  std::vector<float> sum(want.size());
  for (size_t i = 0; i < want.size(); ++i) sum[i] = y0[i] + want[i];
  ExpectClose(Download(d_y), sum, 1e-4f, 1e-4f);
}

// Small tiles (few output tiles), big tiles (>= 140 of them), and ragged edges in T and N.
INSTANTIATE_TEST_SUITE_P(
    Shapes, GemmTest,
    ::testing::Values(std::make_tuple(2, 64, 64), std::make_tuple(37, 1001, 512),
                      std::make_tuple(128, 3072, 2048), std::make_tuple(300, 8192, 2048),
                      std::make_tuple(257, 2048, 1408), std::make_tuple(1, 17, 16)));

class AttentionPrefillTest : public ::testing::TestWithParam<std::tuple<int, int, int, int>> {};

TEST_P(AttentionPrefillTest, MatchesReference) {
  const auto [T, past, nh, nkv] = GetParam();
  AttentionCase c(T, past, nh, nkv, 64, 55);
  DeviceBuffer<float> d_out(static_cast<size_t>(T) * nh * 64);
  kernels::AttentionPrefill(c.new_rows(), T, past, nh, nkv, 64, c.k_cache.get(), c.v_cache.get(),
                            c.max_seq, d_out.get(), nullptr);
  ExpectClose(Download(d_out), c.want, 1e-5f, 1e-4f);
}

// (T, past, num_heads, num_kv_heads): a query tile that is partly empty, several key tiles,
// a prefill continuing after cached positions, and a single query.
INSTANTIATE_TEST_SUITE_P(
    Shapes, AttentionPrefillTest,
    ::testing::Values(std::make_tuple(5, 0, 32, 8), std::make_tuple(37, 0, 32, 8),
                      std::make_tuple(130, 0, 8, 2), std::make_tuple(70, 100, 8, 2),
                      std::make_tuple(1, 63, 4, 4), std::make_tuple(200, 3, 16, 4)));

}  // namespace
}  // namespace infer
