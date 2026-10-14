#include "infer/engine.h"

#include <nvtx3/nvToolsExt.h>

#include <stdexcept>

namespace infer {

namespace k = kernels;

namespace {

// Marks a region on the Nsight Systems timeline, and lets Nsight Compute select the kernels
// of one decode step (--nvtx --nvtx-include "decode/").
class NvtxRange {
 public:
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

size_t Rows(const EngineOptions& o, size_t width) { return static_cast<size_t>(o.max_seq) * width; }

}  // namespace

Engine::Engine(const ModelConfig& cfg, const Weights& weights, const EngineOptions& options)
    : cfg_(cfg),
      w_(weights),
      opt_(options),
      kv_cache_(static_cast<size_t>(cfg.num_layers) * 2 * options.max_seq * cfg.kv_dim()),
      tokens_(static_cast<size_t>(options.max_seq)),
      x_(Rows(options, static_cast<size_t>(cfg.hidden_size))),
      h_(Rows(options, static_cast<size_t>(cfg.hidden_size))),
      qkv_(Rows(options, static_cast<size_t>(cfg.qkv_dim()))),
      attn_(Rows(options, static_cast<size_t>(cfg.q_dim()))),
      gate_up_(Rows(options, 2 * static_cast<size_t>(cfg.intermediate_size))),
      act_(Rows(options, static_cast<size_t>(cfg.intermediate_size))),
      logits_(static_cast<size_t>(options.max_logit_rows) * cfg.vocab_size),
      generated_(static_cast<size_t>(options.max_seq)) {
  if (opt_.max_seq < 1 || opt_.max_logit_rows < 1) throw std::invalid_argument("bad options");
  CUDA_CHECK(cudaStreamCreate(&stream_));
}

Engine::~Engine() {
  if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

void Engine::Run(const int* d_tokens, int T, bool all_logits,
                 std::vector<std::vector<float>>* hidden) {
  const int H = cfg_.hidden_size;
  const int I = cfg_.intermediate_size;
  const float eps = cfg_.rms_norm_eps;
  const size_t layer_cache = static_cast<size_t>(opt_.max_seq) * cfg_.kv_dim();

  auto dump = [&] {
    if (hidden == nullptr) return;
    std::vector<float> host(static_cast<size_t>(T) * H);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), x_.get(), host.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    hidden->push_back(std::move(host));
  };

  k::Embedding(w_.embed, d_tokens, T, H, x_.get(), stream_);
  dump();

  for (int i = 0; i < cfg_.num_layers; ++i) {
    const LayerWeights& l = w_.layers[static_cast<size_t>(i)];
    bf16* k_cache = kv_cache_.get() + 2 * static_cast<size_t>(i) * layer_cache;
    bf16* v_cache = k_cache + layer_cache;

    // Attention block: x += o_proj(attention(rope(qkv_proj(norm(x)))))
    k::RmsNormNaive(x_.get(), l.attn_norm, T, H, eps, h_.get(), stream_);
    k::MatmulNaive(h_.get(), l.wqkv, T, cfg_.qkv_dim(), H, qkv_.get(), false, stream_);
    k::RopeNaive(qkv_.get(), T, cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim, w_.rope_inv_freq,
                 pos_, stream_);
    k::KvCacheWrite(qkv_.get(), T, cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim, pos_,
                    opt_.max_seq, k_cache, v_cache, stream_);
    k::AttentionNaive(qkv_.get(), T, pos_, cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim,
                      k_cache, v_cache, opt_.max_seq, attn_.get(), stream_);
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
  logit_rows_ = rows;
  pos_ += T;
}

void Engine::CheckTokens(const std::vector<int>& tokens) const {
  const int T = static_cast<int>(tokens.size());
  if (T == 0) throw std::invalid_argument("no tokens");
  if (pos_ + T > opt_.max_seq) throw std::invalid_argument("sequence longer than max_seq");
  for (int t : tokens) {
    if (t < 0 || t >= cfg_.vocab_size) throw std::invalid_argument("token id out of range");
  }
}

void Engine::Prefill(const std::vector<int>& tokens, bool all_logits,
                     std::vector<std::vector<float>>* hidden) {
  CheckTokens(tokens);
  const int T = static_cast<int>(tokens.size());
  if (all_logits && T > opt_.max_logit_rows) {
    throw std::invalid_argument("all_logits needs max_logit_rows >= number of tokens");
  }
  NvtxRange range("prefill");
  CUDA_CHECK(cudaMemcpyAsync(tokens_.get(), tokens.data(), tokens.size() * sizeof(int),
                             cudaMemcpyHostToDevice, stream_));
  Run(tokens_.get(), T, all_logits, hidden);
}

void Engine::Decode(int token) {
  CheckTokens({token});
  NvtxRange range("decode");
  CUDA_CHECK(cudaMemcpyAsync(tokens_.get(), &token, sizeof(int), cudaMemcpyHostToDevice, stream_));
  Run(tokens_.get(), 1, false, nullptr);
}

const float* Engine::LastLogitsRow() const {
  return logits_.get() + static_cast<size_t>(logit_rows_ - 1) * cfg_.vocab_size;
}

std::vector<float> Engine::Logits(int rows) {
  if (rows < 1 || rows > logit_rows_) throw std::invalid_argument("logit rows out of range");
  std::vector<float> out(static_cast<size_t>(rows) * cfg_.vocab_size);
  const float* src = logits_.get() + static_cast<size_t>(logit_rows_ - rows) * cfg_.vocab_size;
  CUDA_CHECK(cudaMemcpyAsync(out.data(), src, out.size() * sizeof(float), cudaMemcpyDeviceToHost,
                             stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  return out;
}

std::vector<int> Engine::DecodeGreedy(int n) {
  if (n <= 0) return {};
  if (logit_rows_ == 0) throw std::logic_error("DecodeGreedy needs a Prefill first");
  if (pos_ + n > opt_.max_seq) throw std::invalid_argument("sequence longer than max_seq");
  for (int i = 0; i < n; ++i) {
    NvtxRange range("decode");
    int* token = generated_.get() + i;
    k::Argmax(LastLogitsRow(), cfg_.vocab_size, token, stream_);
    Run(token, 1, false, nullptr);
  }
  std::vector<int> out(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpyAsync(out.data(), generated_.get(), out.size() * sizeof(int),
                             cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  return out;
}

std::vector<int> Engine::GenerateGreedy(const std::vector<int>& prompt, int n) {
  Reset();
  Prefill(prompt);
  return DecodeGreedy(n);
}

}  // namespace infer
