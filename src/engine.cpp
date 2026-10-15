#include "infer/engine.h"

#include <nvtx3/nvToolsExt.h>

#include <random>
#include <stdexcept>

namespace infer {

namespace k = kernels;

namespace {

// Marks a region on the Nsight Systems timeline.
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
      token_(1),
      d_pos_(1),
      history_(static_cast<size_t>(options.max_seq)),
      x_(Rows(options, static_cast<size_t>(cfg.hidden_size))),
      h_(Rows(options, static_cast<size_t>(cfg.hidden_size))),
      qkv_(Rows(options, static_cast<size_t>(cfg.qkv_dim()))),
      attn_(Rows(options, static_cast<size_t>(cfg.q_dim()))),
      gate_up_(Rows(options, 2 * static_cast<size_t>(cfg.intermediate_size))),
      act_(Rows(options, static_cast<size_t>(cfg.intermediate_size))),
      logits_(static_cast<size_t>(options.max_logit_rows) * cfg.vocab_size),
      attn_ws_(k::AttentionDecodeWorkspace(cfg.num_heads, cfg.head_dim, options.attention_splits)),
      argmax_best_(1),
      uniforms_(static_cast<size_t>(options.max_seq)) {
  if (opt_.max_seq < 1 || opt_.max_logit_rows < 1 || opt_.attention_splits < 1) {
    throw std::invalid_argument("bad engine options");
  }
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMemset(argmax_best_.get(), 0, argmax_best_.bytes()));
}

Engine::~Engine() {
  if (step_graph_ != nullptr) cudaGraphExecDestroy(step_graph_);
  if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

bf16* Engine::KCache(int layer) const {
  return kv_cache_.get() + 2 * static_cast<size_t>(layer) * opt_.max_seq * cfg_.kv_dim();
}

bf16* Engine::VCache(int layer) const {
  return KCache(layer) + static_cast<size_t>(opt_.max_seq) * cfg_.kv_dim();
}

void Engine::RunPrefill(int T, bool all_logits, std::vector<std::vector<float>>* hidden) {
  const int H = cfg_.hidden_size;
  const int I = cfg_.intermediate_size;
  const float eps = cfg_.rms_norm_eps;
  const bool naive = opt_.backend == Backend::kNaive;

  auto dump = [&] {
    if (hidden == nullptr) return;
    std::vector<float> host(static_cast<size_t>(T) * H);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), x_.get(), host.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    hidden->push_back(std::move(host));
  };
  auto norm = [&](const float* in, const bf16* w, int rows, float* out) {
    if (naive) {
      k::RmsNormNaive(in, w, rows, H, eps, out, stream_);
    } else {
      k::RmsNorm(in, w, rows, H, eps, out, stream_);
    }
  };
  // out[T, W.rows] (+)= in[T, W.cols] x W^T
  auto project = [&](const float* in, const Linear& W, float* out, bool accumulate) {
    if (naive) {
      k::MatmulNaive(in, Bf16Data(W), T, W.rows, W.cols, out, accumulate, stream_);
    } else {
      k::Gemm(in, W, T, out, accumulate, stream_);
    }
  };

  k::Embedding(w_.embed, cfg_.vocab_size, tokens_.get(), T, H, x_.get(), stream_);
  dump();

  for (int i = 0; i < cfg_.num_layers; ++i) {
    const LayerWeights& l = w_.layers[static_cast<size_t>(i)];

    // Attention block: x += o_proj(attention(rope(qkv_proj(norm(x)))))
    norm(x_.get(), l.attn_norm, T, h_.get());
    project(h_.get(), l.wqkv, qkv_.get(), false);
    k::RopeNaive(qkv_.get(), T, cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim, w_.rope_inv_freq,
                 pos_, stream_);
    k::KvCacheWrite(qkv_.get(), T, cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim, pos_,
                    opt_.max_seq, KCache(i), VCache(i), stream_);
    if (naive) {
      k::AttentionNaive(qkv_.get(), T, pos_, cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim,
                        KCache(i), VCache(i), opt_.max_seq, attn_.get(), stream_);
    } else {
      k::AttentionPrefill(qkv_.get(), T, pos_, cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim,
                          KCache(i), VCache(i), opt_.max_seq, attn_.get(), stream_);
    }
    project(attn_.get(), l.wo, x_.get(), true);

    // MLP block: x += down_proj(silu(gate_proj(norm(x))) * up_proj(norm(x)))
    norm(x_.get(), l.mlp_norm, T, h_.get());
    project(h_.get(), l.w_gate_up, gate_up_.get(), false);
    k::SwiGlu(gate_up_.get(), T, I, act_.get(), stream_);
    project(act_.get(), l.w_down, x_.get(), true);
    dump();
  }

  // Only the rows whose logits are wanted go through the final norm and the LM head.
  const int rows = all_logits ? T : 1;
  const float* last = x_.get() + static_cast<size_t>(T - rows) * H;
  if (naive) {
    k::RmsNormNaive(last, w_.final_norm, rows, H, eps, h_.get(), stream_);
    k::MatmulNaive(h_.get(), Bf16Data(w_.lm_head), rows, cfg_.vocab_size, H, logits_.get(), false,
                   stream_);
  } else if (rows == 1) {
    k::Gemv(last, w_.lm_head, logits_.get(), k::GemvOut::kStore, w_.final_norm, eps, stream_);
  } else {
    k::RmsNorm(last, w_.final_norm, rows, H, eps, h_.get(), stream_);
    k::Gemm(h_.get(), w_.lm_head, rows, logits_.get(), false, stream_);
  }
}

// The naive decode step: the prefill kernels with T = 1. The position is a host value baked
// into the launches, so this cannot be captured as a reusable graph.
void Engine::EnqueueDecodeStepNaive() {
  const int H = cfg_.hidden_size;
  const int I = cfg_.intermediate_size;
  const float eps = cfg_.rms_norm_eps;
  k::Embedding(w_.embed, cfg_.vocab_size, token_.get(), 1, H, x_.get(), stream_);
  for (int i = 0; i < cfg_.num_layers; ++i) {
    const LayerWeights& l = w_.layers[static_cast<size_t>(i)];
    k::RmsNormNaive(x_.get(), l.attn_norm, 1, H, eps, h_.get(), stream_);
    k::MatmulNaive(h_.get(), Bf16Data(l.wqkv), 1, cfg_.qkv_dim(), H, qkv_.get(), false, stream_);
    k::RopeNaive(qkv_.get(), 1, cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim, w_.rope_inv_freq,
                 pos_, stream_);
    k::KvCacheWrite(qkv_.get(), 1, cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim, pos_,
                    opt_.max_seq, KCache(i), VCache(i), stream_);
    k::AttentionNaive(qkv_.get(), 1, pos_, cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim,
                      KCache(i), VCache(i), opt_.max_seq, attn_.get(), stream_);
    k::MatmulNaive(attn_.get(), Bf16Data(l.wo), 1, H, cfg_.q_dim(), x_.get(), true, stream_);
    k::RmsNormNaive(x_.get(), l.mlp_norm, 1, H, eps, h_.get(), stream_);
    k::MatmulNaive(h_.get(), Bf16Data(l.w_gate_up), 1, 2 * I, H, gate_up_.get(), false, stream_);
    k::SwiGlu(gate_up_.get(), 1, I, act_.get(), stream_);
    k::MatmulNaive(act_.get(), Bf16Data(l.w_down), 1, H, I, x_.get(), true, stream_);
  }
  k::RmsNormNaive(x_.get(), w_.final_norm, 1, H, eps, h_.get(), stream_);
  k::MatmulNaive(h_.get(), Bf16Data(w_.lm_head), 1, cfg_.vocab_size, H, logits_.get(), false,
                 stream_);
  k::SetPosition(d_pos_.get(), pos_ + 1, stream_);
}

// The optimized decode step: 7 launches per layer, every RMSNorm and residual add fused into
// a GEMV, the position read from device memory.
void Engine::EnqueueDecodeStep() {
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  k::Embedding(w_.embed, cfg_.vocab_size, token_.get(), 1, H, x_.get(), stream_);
  for (int i = 0; i < cfg_.num_layers; ++i) {
    const LayerWeights& l = w_.layers[static_cast<size_t>(i)];
    // qkv = Wqkv * norm(x)
    k::Gemv(x_.get(), l.wqkv, qkv_.get(), k::GemvOut::kStore, l.attn_norm, eps, stream_);
    k::RopeKvDecode(qkv_.get(), cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim, w_.rope_inv_freq,
                    d_pos_.get(), opt_.max_seq, KCache(i), VCache(i), stream_);
    k::AttentionDecode(qkv_.get(), cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim, KCache(i),
                       VCache(i), opt_.max_seq, d_pos_.get(), opt_.attention_splits, attn_ws_.get(),
                       attn_.get(), stream_);
    // x += Wo * attn
    k::Gemv(attn_.get(), l.wo, x_.get(), k::GemvOut::kAccumulate, nullptr, eps, stream_);
    // act = silu(Wgate * norm(x)) * (Wup * norm(x))
    k::Gemv(x_.get(), l.w_gate_up, act_.get(), k::GemvOut::kSwiGlu, l.mlp_norm, eps, stream_);
    // x += Wdown * act
    k::Gemv(act_.get(), l.w_down, x_.get(), k::GemvOut::kAccumulate, nullptr, eps, stream_);
  }
  k::Gemv(x_.get(), w_.lm_head, logits_.get(), k::GemvOut::kStore, w_.final_norm, eps, stream_);
  k::AdvancePosition(d_pos_.get(), stream_);
}

void Engine::EnqueueGenerateStep(const Sampling& sampling) {
  if (!sampling.greedy()) {
    k::Sample(logits_.get(), cfg_.vocab_size, sampling.temperature, sampling.top_k, sampling.top_p,
              uniforms_.get(), d_pos_.get(), token_.get(), history_.get(), stream_);
  } else if (opt_.backend == Backend::kOptimized) {
    k::ArgmaxPartial(logits_.get(), cfg_.vocab_size, argmax_best_.get(), stream_);
    k::ArgmaxFinish(argmax_best_.get(), token_.get(), history_.get(), d_pos_.get(), stream_);
  } else {
    k::Argmax(logits_.get(), cfg_.vocab_size, token_.get(), stream_);
    k::StoreToken(token_.get(), history_.get(), d_pos_.get(), stream_);
  }
  if (opt_.backend == Backend::kOptimized) {
    EnqueueDecodeStep();
  } else {
    EnqueueDecodeStepNaive();
  }
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
  RunPrefill(T, all_logits, hidden);
  pos_ += T;
  logit_rows_ = all_logits ? T : 1;
  k::SetPosition(d_pos_.get(), pos_, stream_);
}

void Engine::Decode(int token) {
  CheckTokens({token});
  NvtxRange range("decode");
  CUDA_CHECK(cudaMemcpyAsync(token_.get(), &token, sizeof(int), cudaMemcpyHostToDevice, stream_));
  if (opt_.backend == Backend::kOptimized) {
    EnqueueDecodeStep();
  } else {
    EnqueueDecodeStepNaive();
  }
  pos_ += 1;
  logit_rows_ = 1;
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

std::vector<int> Engine::DecodeGreedy(int n) { return DecodeLoop(n, Sampling{}); }

std::vector<int> Engine::DecodeSampled(int n, const Sampling& sampling) {
  return DecodeLoop(n, sampling);
}

std::vector<int> Engine::DecodeLoop(int n, const Sampling& sampling) {
  if (n <= 0) return {};
  if (logit_rows_ == 0) throw std::logic_error("decoding needs a Prefill first");
  if (pos_ + n > opt_.max_seq) throw std::invalid_argument("sequence longer than max_seq");
  if (!sampling.greedy() && !(sampling.top_p > 0.0f && sampling.top_p <= 1.0f)) {
    throw std::invalid_argument("top_p must be in (0, 1]");
  }

  // The step reads logits row 0; after an all-rows prefill the last row is elsewhere.
  if (logit_rows_ > 1) {
    CUDA_CHECK(cudaMemcpyAsync(
        logits_.get(), logits_.get() + static_cast<size_t>(logit_rows_ - 1) * cfg_.vocab_size,
        static_cast<size_t>(cfg_.vocab_size) * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    logit_rows_ = 1;
  }

  if (!sampling.greedy()) {
    // One uniform number per position, from the seed: the same seed and prompt give the same
    // text, and the step stays a fixed graph because it indexes this buffer by position.
    std::mt19937_64 rng(sampling.seed);
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    std::vector<float> u(static_cast<size_t>(opt_.max_seq));
    for (float& x : u) x = unif(rng);
    CUDA_CHECK(cudaMemcpyAsync(uniforms_.get(), u.data(), u.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));  // u is a local
  }

  const bool graph = opt_.backend == Backend::kOptimized && opt_.cuda_graph;
  if (graph && (step_graph_ == nullptr || !(step_graph_sampling_ == sampling))) {
    // Capture one step. Every pointer it touches is fixed for the engine's lifetime and the
    // position lives in device memory, so the same graph is valid at every step. The sampling
    // parameters are baked in, so a different setting needs a new capture.
    if (step_graph_ != nullptr) CUDA_CHECK(cudaGraphExecDestroy(step_graph_));
    cudaGraph_t g;
    CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    EnqueueGenerateStep(sampling);
    CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
    CUDA_CHECK(cudaGraphInstantiate(&step_graph_, g, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    step_graph_sampling_ = sampling;
  }

  const int start = pos_;
  for (int i = 0; i < n; ++i) {
    NvtxRange range("decode");
    if (graph) {
      CUDA_CHECK(cudaGraphLaunch(step_graph_, stream_));
    } else {
      EnqueueGenerateStep(sampling);
    }
    pos_ += 1;
  }
  std::vector<int> out(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpyAsync(out.data(), history_.get() + start, out.size() * sizeof(int),
                             cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  return out;
}

std::vector<int> Engine::Generate(const std::vector<int>& prompt, int n, const Sampling& sampling) {
  Reset();
  Prefill(prompt);
  return DecodeLoop(n, sampling);
}

std::vector<int> Engine::GenerateGreedy(const std::vector<int>& prompt, int n) {
  Reset();
  Prefill(prompt);
  return DecodeGreedy(n);
}

}  // namespace infer
