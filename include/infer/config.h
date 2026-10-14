// Model hyperparameters, read from a Hugging Face config.json.

#pragma once

#include <string>
#include <vector>

namespace infer {

struct ModelConfig {
  int vocab_size = 0;
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_layers = 0;
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  float rms_norm_eps = 1e-5f;
  bool tie_word_embeddings = false;
  int bos_token_id = -1;
  int eos_token_id = -1;

  // RoPE: pair i of every head rotates by position * rope_inv_freq[i]. Precomputed on the host
  // because Llama 3 rescales the frequencies in a way that is awkward to do per call.
  std::vector<float> rope_inv_freq;

  int q_dim() const { return num_heads * head_dim; }
  int kv_dim() const { return num_kv_heads * head_dim; }
  // Width of the fused q|k|v projection output.
  int qkv_dim() const { return q_dim() + 2 * kv_dim(); }

  static ModelConfig FromJsonFile(const std::string& path);
  static ModelConfig FromJsonText(const std::string& text);
};

}  // namespace infer
