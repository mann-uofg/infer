// Runs the model: prefill, decode and greedy generation, with a KV cache.
//
// Prefill processes a run of T prompt tokens in one pass; every projection is a matrix-matrix
// product. Decode processes one token; every projection is a matrix-vector product that reads
// each weight once and does two flops with it, so decode speed is set by memory bandwidth.

#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "infer/config.h"
#include "infer/cuda_util.h"
#include "infer/model.h"

namespace infer {

enum class Backend {
  kNaive,      // the naive kernels everywhere: the baseline
  kOptimized,  // fused GEMV decode path, split-KV attention, CUDA graph
};

inline Backend ParseBackend(const std::string& s) {
  if (s == "naive") return Backend::kNaive;
  if (s == "optimized") return Backend::kOptimized;
  throw std::invalid_argument("unknown backend '" + s + "' (naive or optimized)");
}

struct EngineOptions {
  int max_seq = 2048;      // KV cache capacity, and the longest prefill
  int max_logit_rows = 1;  // how many rows of logits Prefill can return (perplexity wants T)
  Backend backend = Backend::kOptimized;
  bool cuda_graph = true;     // optimized backend: replay greedy decode steps as a graph
  int attention_splits = 16;  // optimized backend: KV ranges per head in decode attention
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

  // Runs one token at position(), through the decode path of the backend.
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
  // Prefill: tokens in tokens_, positions pos_..pos_+T-1 (host value).
  void RunPrefill(int T, bool all_logits, std::vector<std::vector<float>>* hidden);
  // One decode step. Token in token_, position in d_pos_; logits go to row 0; d_pos_ advances.
  void EnqueueDecodeStep();
  void EnqueueDecodeStepNaive();
  // Argmax of logits row 0 into token_, record it in history_, then the decode step.
  void EnqueueGreedyStep();
  void CheckTokens(const std::vector<int>& tokens) const;
  bf16* KCache(int layer) const;
  bf16* VCache(int layer) const;

  const ModelConfig& cfg_;
  const Weights& w_;
  EngineOptions opt_;
  cudaStream_t stream_ = nullptr;
  cudaGraphExec_t greedy_graph_ = nullptr;
  int pos_ = 0;
  int logit_rows_ = 0;  // rows written by the last Prefill/Decode

  // KV cache: for each layer, k then v, each [num_kv_heads, max_seq, head_dim] in bf16.
  DeviceBuffer<bf16> kv_cache_;

  DeviceBuffer<int> tokens_;     // prefill input [T]
  DeviceBuffer<int> token_;      // decode input [1]
  DeviceBuffer<int> d_pos_;      // decode position [1]; equals pos_ between calls
  DeviceBuffer<int> history_;    // [max_seq] token run at each position by DecodeGreedy
  DeviceBuffer<float> x_;        // residual stream [T, H]
  DeviceBuffer<float> h_;        // normed input to a block [T, H]
  DeviceBuffer<float> qkv_;      // [T, qkv_dim]
  DeviceBuffer<float> attn_;     // [T, q_dim]
  DeviceBuffer<float> gate_up_;  // [T, 2I]
  DeviceBuffer<float> act_;      // [T, I]
  DeviceBuffer<float> logits_;   // [max_logit_rows, V]
  DeviceBuffer<float> attn_ws_;  // split-KV partial results
  DeviceBuffer<unsigned long long> argmax_best_;  // packed (value, index) for ArgmaxPartial
};

}  // namespace infer
