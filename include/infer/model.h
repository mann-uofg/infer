// Model weights on the GPU.

#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "infer/config.h"
#include "infer/cuda_util.h"
#include "infer/kernels.h"

namespace infer {

using kernels::bf16;
using kernels::Linear;
using kernels::WeightFormat;

struct LayerWeights {
  const bf16* attn_norm;  // [H]
  Linear wqkv;            // [q_dim + 2*kv_dim, H]: q_proj, k_proj, v_proj stacked by rows
  Linear wo;              // [H, q_dim]
  const bf16* mlp_norm;   // [H]
  Linear w_gate_up;       // [2*I, H]: gate_proj then up_proj stacked by rows
  Linear w_down;          // [H, I]
};

inline WeightFormat ParseWeightFormat(const std::string& s) {
  if (s == "bf16") return WeightFormat::kBf16;
  if (s == "int8") return WeightFormat::kInt8;
  if (s == "int4") return WeightFormat::kInt4;
  throw std::invalid_argument("unknown weight format '" + s + "' (bf16, int8 or int4)");
}

inline const char* WeightFormatName(WeightFormat f) {
  switch (f) {
    case WeightFormat::kBf16:
      return "bf16";
    case WeightFormat::kInt8:
      return "int8";
    case WeightFormat::kInt4:
      return "int4";
  }
  return "?";
}

struct LoadOptions {
  // Format of the projection weights (q/k/v/o, gate/up/down).
  WeightFormat format = WeightFormat::kBf16;
  // Format of the LM head; empty means the same as `format`. The head is 21% of the bytes a
  // decode step reads for Llama-3.2-1B, so it matters for speed, and it is the layer most
  // sensitive to quantization. (llama.cpp's Q4_0 keeps it at 6 bits; int4 layers with an int8
  // head read almost exactly the same bytes per token.)
  std::optional<WeightFormat> lm_head_format;

  WeightFormat head() const { return lm_head_format.value_or(format); }
};

class Weights {
 public:
  // Reads the checkpoint in `dir` (config.json + safetensors) and uploads it, quantizing the
  // projections on the GPU if asked to. Each tensor starts on a 256-byte boundary so that
  // kernels can use 16-byte vector loads on any row.
  static Weights Load(const std::string& dir, const ModelConfig& cfg,
                      const LoadOptions& options = {});

  const bf16* embed = nullptr;           // [V, H], always bf16 (one row is read per token)
  const bf16* final_norm = nullptr;      // [H]
  Linear lm_head;                        // [V, H]; the embedding table itself when tied and bf16
  const float* rope_inv_freq = nullptr;  // [head_dim / 2]
  std::vector<LayerWeights> layers;
  LoadOptions options;

  // Bytes of weights a single decode step has to read: one embedding row, the norms, every
  // projection (with its scales) and the LM head.
  size_t decode_bytes() const { return decode_bytes_; }
  size_t device_bytes() const { return arena_.bytes(); }

 private:
  DeviceBuffer<std::byte> arena_;
  DeviceBuffer<float> inv_freq_;
  size_t decode_bytes_ = 0;
};

// The bf16 pointer of a projection; for code paths (the naive kernels) that only take bf16.
inline const bf16* Bf16Data(const Linear& l) {
  if (l.format != WeightFormat::kBf16) {
    throw std::invalid_argument("the naive kernels only support bf16 weights");
  }
  return static_cast<const bf16*>(l.data);
}

}  // namespace infer
