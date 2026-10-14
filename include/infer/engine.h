// Runs the model: the forward pass and greedy generation.

#pragma once

#include <vector>

#include "infer/config.h"
#include "infer/cuda_util.h"
#include "infer/model.h"

namespace infer {

enum class Backend { kNaive };

class Engine {
 public:
  // max_tokens bounds the sequence length; buffers are sized for it up front.
  Engine(const ModelConfig& cfg, const Weights& weights, int max_tokens, Backend backend);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Runs the whole sequence (positions 0..T-1) and returns logits on the host: [T, V] if
  // all_logits, else just the last row [V]. If hidden is non-null it receives the residual
  // stream after the embedding and after each layer (num_layers + 1 arrays of [T, H]).
  std::vector<float> Forward(const std::vector<int>& tokens, bool all_logits,
                             std::vector<std::vector<float>>* hidden = nullptr);

  // Appends n tokens, each the argmax of the previous step's logits.
  std::vector<int> GenerateGreedy(const std::vector<int>& prompt, int n);

 private:
  // Enqueues the forward pass; logits for the last row (or all rows) land in logits_.
  void ForwardDevice(int T, bool all_logits, std::vector<std::vector<float>>* hidden);

  const ModelConfig& cfg_;
  const Weights& w_;
  int max_tokens_;
  Backend backend_;
  cudaStream_t stream_ = nullptr;

  DeviceBuffer<int> tokens_;
  DeviceBuffer<float> x_;        // residual stream [T, H]
  DeviceBuffer<float> h_;        // normed input to a block [T, H]
  DeviceBuffer<float> qkv_;      // [T, qkv_dim]
  DeviceBuffer<float> attn_;     // [T, q_dim]
  DeviceBuffer<float> gate_up_;  // [T, 2I]
  DeviceBuffer<float> act_;      // [T, I]
  DeviceBuffer<float> logits_;   // [T, V]
  DeviceBuffer<int> next_token_;
};

}  // namespace infer
