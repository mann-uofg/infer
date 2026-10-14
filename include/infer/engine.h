// Runs the model: prefill, decode and greedy generation, with a KV cache.
//
// Prefill processes a run of T prompt tokens in one pass; every projection is a matrix-matrix
// product. Decode processes one token; every projection is a matrix-vector product that reads
// each weight once and does two flops with it, so decode speed is set by memory bandwidth.

#pragma once

#include <vector>

#include "infer/config.h"
#include "infer/cuda_util.h"
#include "infer/model.h"

namespace infer {

enum class Backend { kNaive };

struct EngineOptions {
  int max_seq = 2048;      // KV cache capacity, and the longest prefill
  int max_logit_rows = 1;  // how many rows of logits Prefill can return (perplexity wants T)
  Backend backend = Backend::kNaive;
};

class Engine {
 public:
  Engine(const ModelConfig& cfg, const Weights& weights, const EngineOptions& options);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Forgets the sequence; the next token goes to position 0.
  void Reset() {
    pos_ = 0;
    logit_rows_ = 0;
  }
  int position() const { return pos_; }

  // Runs tokens at positions position()..position()+T-1 and appends them to the cache. Leaves
  // logits for the last token (or for all T, if all_logits) on the device. If hidden is
  // non-null it receives the residual stream after the embedding and after each layer
  // (num_layers + 1 arrays of [T, H]).
  void Prefill(const std::vector<int>& tokens, bool all_logits = false,
               std::vector<std::vector<float>>* hidden = nullptr);

  // Runs one token at position(); same as Prefill with T = 1.
  void Decode(int token);

  // Copies the last `rows` rows of logits produced by the last Prefill/Decode to the host.
  std::vector<float> Logits(int rows = 1);

  // Runs n decode steps, each fed the argmax of the previous step's logits (the first one
  // takes the argmax of the last Prefill/Decode). Returns the n tokens fed in. The tokens stay
  // on the GPU between steps, so the host only waits once, at the end.
  std::vector<int> DecodeGreedy(int n);

  // Reset, prefill the prompt, then DecodeGreedy(n).
  std::vector<int> GenerateGreedy(const std::vector<int>& prompt, int n);

  cudaStream_t stream() const { return stream_; }

 private:
  // Enqueues the forward pass for T tokens at d_tokens (device), at positions
  // pos_..pos_+T-1.
  void Run(const int* d_tokens, int T, bool all_logits, std::vector<std::vector<float>>* hidden);
  const float* LastLogitsRow() const;
  void CheckTokens(const std::vector<int>& tokens) const;

  const ModelConfig& cfg_;
  const Weights& w_;
  EngineOptions opt_;
  cudaStream_t stream_ = nullptr;
  int pos_ = 0;
  int logit_rows_ = 0;  // rows written by the last Run

  // KV cache: for each layer, k then v, each [num_kv_heads, max_seq, head_dim] in bf16.
  DeviceBuffer<bf16> kv_cache_;

  DeviceBuffer<int> tokens_;
  DeviceBuffer<float> x_;        // residual stream [T, H]
  DeviceBuffer<float> h_;        // normed input to a block [T, H]
  DeviceBuffer<float> qkv_;      // [T, qkv_dim]
  DeviceBuffer<float> attn_;     // [T, q_dim]
  DeviceBuffer<float> gate_up_;  // [T, 2I]
  DeviceBuffer<float> act_;      // [T, I]
  DeviceBuffer<float> logits_;   // [max_logit_rows, V]
  DeviceBuffer<int> generated_;  // [max_seq] tokens produced by DecodeGreedy
};

}  // namespace infer
