#include "infer/config.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace infer {
namespace {

// reference/configs/llama-3.2-1b.json, trimmed to the fields the engine reads.
const char* kLlama32 = R"({
  "model_type": "llama", "attention_bias": false, "mlp_bias": false, "hidden_act": "silu",
  "vocab_size": 128256, "hidden_size": 2048, "intermediate_size": 8192,
  "num_hidden_layers": 16, "num_attention_heads": 32, "num_key_value_heads": 8,
  "head_dim": 64, "rms_norm_eps": 1e-05, "rope_theta": 500000.0, "tie_word_embeddings": true,
  "bos_token_id": 128000, "eos_token_id": 128001,
  "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                   "original_max_position_embeddings": 8192, "rope_type": "llama3"}
})";

TEST(ConfigTest, ReadsLlama32) {
  const ModelConfig c = ModelConfig::FromJsonText(kLlama32);
  EXPECT_EQ(c.vocab_size, 128256);
  EXPECT_EQ(c.hidden_size, 2048);
  EXPECT_EQ(c.intermediate_size, 8192);
  EXPECT_EQ(c.num_layers, 16);
  EXPECT_EQ(c.num_heads, 32);
  EXPECT_EQ(c.num_kv_heads, 8);
  EXPECT_EQ(c.head_dim, 64);
  EXPECT_FLOAT_EQ(c.rms_norm_eps, 1e-5f);
  EXPECT_TRUE(c.tie_word_embeddings);
  EXPECT_EQ(c.bos_token_id, 128000);
  EXPECT_EQ(c.eos_token_id, 128001);
  EXPECT_EQ(c.q_dim(), 2048);
  EXPECT_EQ(c.kv_dim(), 512);
  EXPECT_EQ(c.qkv_dim(), 3072);
}

TEST(ConfigTest, Llama3RopeFrequenciesMatchReference) {
  // Expected values printed by reference.llama.rope_inv_freq for the same config. Indices
  // cover all three bands: unchanged (0, 5), smoothed (10, 12, 14), divided by factor (16+).
  const ModelConfig c = ModelConfig::FromJsonText(kLlama32);
  ASSERT_EQ(c.rope_inv_freq.size(), 32u);
  const struct {
    size_t i;
    double want;
  } cases[] = {{0, 1.0},
               {5, 0.12868738174438477},
               {10, 0.016560440883040428},
               {12, 0.00729266507551074},
               {14, 0.0032114461064338684},
               {16, 0.000429556705057621},
               {20, 8.570255886297673e-06},
               {31, 9.418306490260875e-08}};
  for (const auto& tc : cases) {
    EXPECT_NEAR(c.rope_inv_freq[tc.i], tc.want, tc.want * 1e-6) << "pair " << tc.i;
  }
}

TEST(ConfigTest, DefaultRopeWithoutScaling) {
  std::string text = kLlama32;
  text.replace(text.find("\"rope_scaling\""), std::string::npos, "\"rope_scaling\": null}");
  const ModelConfig c = ModelConfig::FromJsonText(text);
  EXPECT_NEAR(c.rope_inv_freq[16], 1.0 / std::pow(500000.0, 0.5), 1e-9);
}

TEST(ConfigTest, RejectsUnsupportedModels) {
  auto with = [](const std::string& from, const std::string& to) {
    std::string t = kLlama32;
    t.replace(t.find(from), from.size(), to);
    return t;
  };
  EXPECT_THROW(ModelConfig::FromJsonText(with("\"llama\"", "\"qwen2\"")), std::runtime_error);
  EXPECT_THROW(
      ModelConfig::FromJsonText(with("\"attention_bias\": false", "\"attention_bias\": true")),
      std::runtime_error);
  EXPECT_THROW(
      ModelConfig::FromJsonText(with("\"num_key_value_heads\": 8", "\"num_key_value_heads\": 5")),
      std::runtime_error);
  EXPECT_THROW(ModelConfig::FromJsonText(with("\"llama3\"", "\"yarn\"")), std::runtime_error);
}

TEST(ConfigTest, RejectsHostileNumbers) {
  auto with = [](const std::string& from, const std::string& to) {
    std::string t = kLlama32;
    t.replace(t.find(from), from.size(), to);
    return t;
  };
  // 2^32 + 2048 would wrap to 2048 if narrowed to int without a check.
  EXPECT_THROW(
      ModelConfig::FromJsonText(with("\"hidden_size\": 2048", "\"hidden_size\": 4294969344")),
      std::runtime_error);
  EXPECT_THROW(ModelConfig::FromJsonText(with("\"vocab_size\": 128256", "\"vocab_size\": -1")),
               std::runtime_error);
  // heads x head_dim: 4096 x 1024 = 4M columns for q alone, beyond any sane width.
  std::string wide = with("\"num_attention_heads\": 32", "\"num_attention_heads\": 4096");
  wide.replace(wide.find("\"head_dim\": 64"), 14, "\"head_dim\": 1024");
  EXPECT_THROW(ModelConfig::FromJsonText(wide), std::runtime_error);
  EXPECT_THROW(ModelConfig::FromJsonText(with("\"rms_norm_eps\": 1e-05", "\"rms_norm_eps\": 0")),
               std::runtime_error);
  EXPECT_THROW(ModelConfig::FromJsonText(with("\"factor\": 32.0", "\"factor\": 0")),
               std::runtime_error);
  EXPECT_THROW(
      ModelConfig::FromJsonText(with("\"high_freq_factor\": 4.0", "\"high_freq_factor\": 1.0")),
      std::runtime_error);
}

}  // namespace
}  // namespace infer
