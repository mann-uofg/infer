#include "infer/config.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "infer/json.h"

namespace infer {

namespace {

int GetInt(const Json& j, const std::string& key) { return static_cast<int>(j[key].AsInt()); }

// Same arithmetic as the reference (reference/llama.py: rope_inv_freq).
std::vector<float> ComputeRopeInvFreq(const Json& j, int head_dim) {
  const double theta = j.Has("rope_theta") ? j["rope_theta"].AsNumber() : 10000.0;
  std::vector<float> inv_freq;
  for (int i = 0; i < head_dim; i += 2) {
    inv_freq.push_back(
        static_cast<float>(1.0 / std::pow(theta, static_cast<double>(i) / head_dim)));
  }
  if (!j.Has("rope_scaling") || j["rope_scaling"].is_null()) return inv_freq;

  const Json& s = j["rope_scaling"];
  const std::string type =
      s.Has("rope_type") ? s["rope_type"].AsString() : (s.Has("type") ? s["type"].AsString() : "");
  if (type.empty() || type == "default") return inv_freq;
  if (type != "llama3") throw std::runtime_error("unsupported rope scaling type " + type);

  const double factor = s["factor"].AsNumber();
  const double low = s["low_freq_factor"].AsNumber();
  const double high = s["high_freq_factor"].AsNumber();
  const double old_ctx = s["original_max_position_embeddings"].AsNumber();
  const double low_freq_wavelen = old_ctx / low;
  const double high_freq_wavelen = old_ctx / high;
  const double kPi = 3.14159265358979323846;

  for (float& f : inv_freq) {
    const double freq = f;
    const double wavelen = 2 * kPi / freq;
    double out = freq;
    if (wavelen > low_freq_wavelen) {
      out = freq / factor;
    } else if (wavelen >= high_freq_wavelen) {
      const double smooth = (old_ctx / wavelen - low) / (high - low);
      out = (1 - smooth) * freq / factor + smooth * freq;
    }
    f = static_cast<float>(out);
  }
  return inv_freq;
}

}  // namespace

ModelConfig ModelConfig::FromJsonText(const std::string& text) {
  const Json j = Json::Parse(text);
  if (j["model_type"].AsString() != "llama") {
    throw std::runtime_error("unsupported model_type " + j["model_type"].AsString());
  }
  if ((j.Has("attention_bias") && j["attention_bias"].AsBool()) ||
      (j.Has("mlp_bias") && j["mlp_bias"].AsBool())) {
    throw std::runtime_error("biases are not supported");
  }
  if (j.Has("hidden_act") && j["hidden_act"].AsString() != "silu") {
    throw std::runtime_error("only silu is supported");
  }

  ModelConfig c;
  c.vocab_size = GetInt(j, "vocab_size");
  c.hidden_size = GetInt(j, "hidden_size");
  c.intermediate_size = GetInt(j, "intermediate_size");
  c.num_layers = GetInt(j, "num_hidden_layers");
  c.num_heads = GetInt(j, "num_attention_heads");
  c.num_kv_heads = j.Has("num_key_value_heads") ? GetInt(j, "num_key_value_heads") : c.num_heads;
  c.head_dim = j.Has("head_dim") && !j["head_dim"].is_null() ? GetInt(j, "head_dim")
                                                             : c.hidden_size / c.num_heads;
  c.rms_norm_eps = static_cast<float>(j["rms_norm_eps"].AsNumber());
  c.tie_word_embeddings = j.Has("tie_word_embeddings") && j["tie_word_embeddings"].AsBool();
  if (j.Has("bos_token_id") && j["bos_token_id"].is_number()) {
    c.bos_token_id = GetInt(j, "bos_token_id");
  }
  // Some checkpoints list several end-of-sequence ids; the first is the primary one.
  if (j.Has("eos_token_id") && j["eos_token_id"].is_number()) {
    c.eos_token_id = GetInt(j, "eos_token_id");
  } else if (j.Has("eos_token_id") && j["eos_token_id"].is_array() &&
             !j["eos_token_id"].AsArray().empty()) {
    c.eos_token_id = static_cast<int>(j["eos_token_id"].AsArray()[0].AsInt());
  }

  if (c.vocab_size <= 0 || c.hidden_size <= 0 || c.intermediate_size <= 0 || c.num_layers <= 0 ||
      c.num_heads <= 0 || c.num_kv_heads <= 0 || c.head_dim <= 0) {
    throw std::runtime_error("config has a non-positive dimension");
  }
  if (c.num_heads % c.num_kv_heads != 0) {
    throw std::runtime_error("num_attention_heads must be a multiple of num_key_value_heads");
  }
  if (c.head_dim % 2 != 0) throw std::runtime_error("head_dim must be even for RoPE");

  c.rope_inv_freq = ComputeRopeInvFreq(j, c.head_dim);
  return c;
}

ModelConfig ModelConfig::FromJsonFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return FromJsonText(ss.str());
}

}  // namespace infer
