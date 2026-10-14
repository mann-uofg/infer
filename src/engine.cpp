#include "infer/engine.h"

#include <stdexcept>

namespace infer {

namespace k = kernels;

Engine::Engine(const ModelConfig& cfg, const Weights& weights, int max_tokens, Backend backend)
    : cfg_(cfg),
      w_(weights),
      max_tokens_(max_tokens),
      backend_(backend),
      tokens_(static_cast<size_t>(max_tokens)),
      x_(static_cast<size_t>(max_tokens) * cfg.hidden_size),
      h_(static_cast<size_t>(max_tokens) * cfg.hidden_size),
      qkv_(static_cast<size_t>(max_tokens) * cfg.qkv_dim()),
      attn_(static_cast<size_t>(max_tokens) * cfg.q_dim()),
      gate_up_(static_cast<size_t>(max_tokens) * 2 * cfg.intermediate_size),
      act_(static_cast<size_t>(max_tokens) * cfg.intermediate_size),
      logits_(static_cast<size_t>(max_tokens) * cfg.vocab_size),
      next_token_(1) {
  CUDA_CHECK(cudaStreamCreate(&stream_));
}

Engine::~Engine() {
  if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

void Engine::ForwardDevice(int T, bool all_logits, std::vector<std::vector<float>>* hidden) {
  const int H = cfg_.hidden_size;
  const int I = cfg_.intermediate_size;
  const float eps = cfg_.rms_norm_eps;

  auto dump = [&] {
    if (hidden == nullptr) return;
    std::vector<float> host(static_cast<size_t>(T) * H);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), x_.get(), host.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    hidden->push_back(std::move(host));
  };

  k::Embedding(w_.embed, tokens_.get(), T, H, x_.get(), stream_);
  dump();

  for (const LayerWeights& l : w_.layers) {
    // Attention block: x += o_proj(attention(rope(qkv_proj(norm(x)))))
    k::RmsNormNaive(x_.get(), l.attn_norm, T, H, eps, h_.get(), stream_);
    k::MatmulNaive(h_.get(), l.wqkv, T, cfg_.qkv_dim(), H, qkv_.get(), false, stream_);
    k::RopeNaive(qkv_.get(), T, cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim, w_.rope_inv_freq,
                 0, stream_);
    k::AttentionNaive(qkv_.get(), T, cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim, attn_.get(),
                      stream_);
    k::MatmulNaive(attn_.get(), l.wo, T, H, cfg_.q_dim(), x_.get(), true, stream_);

    // MLP block: x += down_proj(silu(gate_proj(norm(x))) * up_proj(norm(x)))
    k::RmsNormNaive(x_.get(), l.mlp_norm, T, H, eps, h_.get(), stream_);
    k::MatmulNaive(h_.get(), l.w_gate_up, T, 2 * I, H, gate_up_.get(), false, stream_);
    k::SwiGlu(gate_up_.get(), T, I, act_.get(), stream_);
    k::MatmulNaive(act_.get(), l.w_down, T, H, I, x_.get(), true, stream_);
    dump();
  }

  // Only the rows whose logits are wanted go through the final norm and the LM head.
  const int rows = all_logits ? T : 1;
  const float* last = x_.get() + static_cast<size_t>(T - rows) * H;
  k::RmsNormNaive(last, w_.final_norm, rows, H, eps, h_.get(), stream_);
  k::MatmulNaive(h_.get(), w_.lm_head, rows, cfg_.vocab_size, H, logits_.get(), false, stream_);
}

std::vector<float> Engine::Forward(const std::vector<int>& tokens, bool all_logits,
                                   std::vector<std::vector<float>>* hidden) {
  const int T = static_cast<int>(tokens.size());
  if (T == 0 || T > max_tokens_) throw std::invalid_argument("sequence length out of range");
  for (int t : tokens) {
    if (t < 0 || t >= cfg_.vocab_size) throw std::invalid_argument("token id out of range");
  }
  CUDA_CHECK(cudaMemcpyAsync(tokens_.get(), tokens.data(), T * sizeof(int), cudaMemcpyHostToDevice,
                             stream_));
  ForwardDevice(T, all_logits, hidden);
  std::vector<float> out(static_cast<size_t>(all_logits ? T : 1) * cfg_.vocab_size);
  CUDA_CHECK(cudaMemcpyAsync(out.data(), logits_.get(), out.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  return out;
}

std::vector<int> Engine::GenerateGreedy(const std::vector<int>& prompt, int n) {
  std::vector<int> seq = prompt;
  std::vector<int> out;
  for (int i = 0; i < n; ++i) {
    const int T = static_cast<int>(seq.size());
    if (T > max_tokens_) throw std::invalid_argument("sequence longer than max_tokens");
    // No KV cache yet: every step reruns the whole sequence.
    CUDA_CHECK(cudaMemcpyAsync(tokens_.get(), seq.data(), T * sizeof(int), cudaMemcpyHostToDevice,
                               stream_));
    ForwardDevice(T, false, nullptr);
    k::Argmax(logits_.get(), cfg_.vocab_size, next_token_.get(), stream_);
    int next = 0;
    CUDA_CHECK(
        cudaMemcpyAsync(&next, next_token_.get(), sizeof(int), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    seq.push_back(next);
    out.push_back(next);
  }
  return out;
}

}  // namespace infer
