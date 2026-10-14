// Model weights on the GPU.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "infer/config.h"
#include "infer/cuda_util.h"
#include "infer/kernels.h"

namespace infer {

using kernels::bf16;

struct LayerWeights {
  const bf16* attn_norm;  // [H]
  const bf16* wqkv;       // [q_dim + 2*kv_dim, H]: q_proj, k_proj, v_proj stacked by rows
  const bf16* wo;         // [H, q_dim]
  const bf16* mlp_norm;   // [H]
  const bf16* w_gate_up;  // [2*I, H]: gate_proj then up_proj stacked by rows
  const bf16* w_down;     // [H, I]
};

class Weights {
 public:
  // Reads the checkpoint in `dir` (config.json + safetensors) and uploads it. All tensors go
  // into one device allocation, each starting on a 256-byte boundary so that kernels can use
  // 16-byte vector loads on any row.
  static Weights Load(const std::string& dir, const ModelConfig& cfg);

  const bf16* embed = nullptr;           // [V, H]
  const bf16* final_norm = nullptr;      // [H]
  const bf16* lm_head = nullptr;         // [V, H]; same pointer as embed when tied
  const float* rope_inv_freq = nullptr;  // [head_dim / 2]
  std::vector<LayerWeights> layers;

  // Bytes of weights a single decode step has to read: everything except the embedding table
  // (one row of it is read), counting the LM head once even when it is tied to the embedding.
  size_t decode_bytes() const { return decode_bytes_; }
  size_t device_bytes() const { return arena_.bytes(); }

 private:
  DeviceBuffer<std::byte> arena_;
  DeviceBuffer<float> inv_freq_;
  size_t decode_bytes_ = 0;
};

}  // namespace infer
