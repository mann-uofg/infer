// The sampling kernel against a CPU implementation of the same rule:
//   keep logits >= the k-th largest (ties kept), then the smallest top set whose probability
//   reaches top_p of what is left (ties at the cut kept), then pick, in index order, the first
//   kept token whose cumulative weight exp((l - max) / T) exceeds u * total.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "gpu_test_util.h"
#include "infer/kernels.h"

namespace infer {
namespace {

using test::Download;
using test::RandomFloats;
using test::Upload;

struct CpuSampler {
  std::vector<float> logits;
  float temperature;
  int top_k;
  float top_p;
  std::vector<bool> keep;
  std::vector<double> weight;  // exp((l - max) / T) for kept tokens, 0 otherwise

  CpuSampler(std::vector<float> l, float t, int k, float p)
      : logits(std::move(l)), temperature(t), top_k(k), top_p(p) {
    const int V = static_cast<int>(logits.size());
    std::vector<float> sorted = logits;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    const float m = sorted[0];
    float cut = sorted[V - 1];
    if (top_k > 0 && top_k < V) cut = sorted[static_cast<size_t>(top_k - 1)];
    double mass = 0;
    for (float x : sorted) {
      if (x >= cut) mass += std::exp((x - m) / temperature);
    }
    if (top_p < 1.0f) {
      double acc = 0;
      for (float x : sorted) {
        if (x < cut) break;
        acc += std::exp((x - m) / temperature);
        if (acc >= top_p * mass) {
          cut = x;
          break;
        }
      }
    }
    keep.resize(static_cast<size_t>(V));
    weight.resize(static_cast<size_t>(V));
    for (int i = 0; i < V; ++i) {
      keep[static_cast<size_t>(i)] = logits[static_cast<size_t>(i)] >= cut;
      weight[static_cast<size_t>(i)] =
          keep[static_cast<size_t>(i)]
              ? std::exp((logits[static_cast<size_t>(i)] - m) / temperature)
              : 0.0;
    }
  }

  double total() const {
    double t = 0;
    for (double w : weight) t += w;
    return t;
  }
};

// Runs the kernel once with uniform number u.
int GpuSample(const std::vector<float>& logits, float t, int k, float p, float u) {
  auto d_logits = Upload(logits);
  auto d_u = Upload(std::vector<float>{0.0f, u});
  auto d_pos = Upload(std::vector<int>{1});
  DeviceBuffer<int> token(1);
  DeviceBuffer<int> history(2);
  kernels::Sample(d_logits.get(), static_cast<int>(logits.size()), t, k, p, d_u.get(), d_pos.get(),
                  token.get(), history.get(), nullptr);
  const int tok = Download(token)[0];
  EXPECT_EQ(Download(history)[1], tok);
  return tok;
}

struct Params {
  int V;
  float temperature;
  int top_k;
  float top_p;
};

class SampleTest : public ::testing::TestWithParam<Params> {};

// u at the middle of each kept token's interval must select exactly that token.
TEST_P(SampleTest, PicksTheTokenWhoseIntervalContainsU) {
  const Params prm = GetParam();
  const auto logits = RandomFloats(static_cast<size_t>(prm.V), 2.0f, 70);
  const CpuSampler cpu(logits, prm.temperature, prm.top_k, prm.top_p);
  const double total = cpu.total();
  double before = 0;
  int checked = 0;
  for (int i = 0; i < prm.V; ++i) {
    const double w = cpu.weight[static_cast<size_t>(i)];
    if (w == 0) continue;
    // Skip slivers too thin to hit reliably in fp32 summation order.
    if (w / total > 1e-4) {
      const float u = static_cast<float>((before + w / 2) / total);
      ASSERT_EQ(GpuSample(logits, prm.temperature, prm.top_k, prm.top_p, u), i) << "u = " << u;
      if (++checked == 40) break;
    }
    before += w;
  }
  EXPECT_GT(checked, 0);
}

// Random u: the token is always one that the CPU rule keeps.
TEST_P(SampleTest, NeverPicksAFilteredToken) {
  const Params prm = GetParam();
  const auto logits = RandomFloats(static_cast<size_t>(prm.V), 2.0f, 71);
  const CpuSampler cpu(logits, prm.temperature, prm.top_k, prm.top_p);
  const auto us = RandomFloats(64, 1.0f, 72);
  for (float u : us) {
    const float uu = std::fmod(std::abs(u), 1.0f);
    const int t = GpuSample(logits, prm.temperature, prm.top_k, prm.top_p, uu);
    ASSERT_GE(t, 0);
    ASSERT_LT(t, prm.V);
    ASSERT_TRUE(cpu.keep[static_cast<size_t>(t)]) << "token " << t << " u " << uu;
  }
}

INSTANTIATE_TEST_SUITE_P(Settings, SampleTest,
                         ::testing::Values(Params{1000, 1.0f, 0, 1.0f},    // plain softmax
                                           Params{1000, 0.7f, 50, 1.0f},   // top-k
                                           Params{1000, 1.0f, 0, 0.9f},    // top-p
                                           Params{128256, 0.6f, 0, 0.9f},  // Llama defaults
                                           Params{128256, 1.0f, 40, 0.95f},
                                           Params{5000, 2.0f, 5000, 1.0f}),  // k = V: no cut
                         [](const ::testing::TestParamInfo<Params>& info) {
                           const Params& p = info.param;
                           return "V" + std::to_string(p.V) + "_T" +
                                  std::to_string(static_cast<int>(p.temperature * 100)) + "_k" +
                                  std::to_string(p.top_k) + "_p" +
                                  std::to_string(static_cast<int>(p.top_p * 100));
                         });

TEST(SampleTest, TopKOneIsArgmax) {
  auto logits = RandomFloats(128256, 1.0f, 73);
  logits[4242] = 10.0f;
  for (float u : {0.0f, 0.3f, 0.999f}) EXPECT_EQ(GpuSample(logits, 1.0f, 1, 1.0f, u), 4242);
}

TEST(SampleTest, SmallTopPKeepsOnlyTheMostLikelyToken) {
  auto logits = RandomFloats(1000, 1.0f, 74);
  logits[17] = 8.0f;  // ~half the probability mass on its own
  for (float u : {0.0f, 0.5f, 0.999f}) EXPECT_EQ(GpuSample(logits, 1.0f, 0, 0.01f, u), 17);
}

}  // namespace
}  // namespace infer
