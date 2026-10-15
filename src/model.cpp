#include "infer/model.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "infer/safetensors.h"

namespace infer {

namespace {

constexpr size_t kAlign = 256;

size_t AlignUp(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

// Checks dtype and shape, so a wrong checkpoint fails at load time with a clear message
// instead of producing garbage.
const TensorView* Expect(const SafetensorsModel& st, const std::string& name,
                         const std::vector<int64_t>& shape) {
  const TensorView& t = st.Get(name);
  if (t.dtype != DType::kBF16) {
    throw std::runtime_error(name + ": expected BF16, got " + DTypeName(t.dtype));
  }
  if (t.shape != shape) {
    std::string want;
    std::string got;
    for (int64_t d : shape) want += std::to_string(d) + " ";
    for (int64_t d : t.shape) got += std::to_string(d) + " ";
    throw std::runtime_error(name + ": expected shape [ " + want + "], got [ " + got + "]");
  }
  return &t;
}

// A projection as it comes out of the checkpoint: one or more tensors stacked by rows.
struct Source {
  std::vector<const TensorView*> parts;
  int rows = 0;
  int cols = 0;
};

Source Stack(std::initializer_list<const TensorView*> parts) {
  Source s;
  for (const TensorView* t : parts) {
    s.parts.push_back(t);
    s.rows += static_cast<int>(t->shape[0]);
    s.cols = static_cast<int>(t->shape[1]);
  }
  return s;
}

// Bytes of one projection's data and of its scales in `format`.
size_t DataBytes(WeightFormat f, int rows, int cols) {
  const size_t n = static_cast<size_t>(rows) * cols;
  switch (f) {
    case WeightFormat::kBf16:
      return n * sizeof(bf16);
    case WeightFormat::kInt8:
      return n;
    case WeightFormat::kInt4:
      return n / 2;
  }
  return 0;
}

size_t ScaleBytes(WeightFormat f, int rows, int cols, int group) {
  switch (f) {
    case WeightFormat::kBf16:
      return 0;
    case WeightFormat::kInt8:
      return static_cast<size_t>(rows) * sizeof(float);
    case WeightFormat::kInt4:
      return static_cast<size_t>(rows) * (cols / group) * sizeof(__half);
  }
  return 0;
}

size_t ArenaBytes(WeightFormat f, const Source& s, int group) {
  return AlignUp(DataBytes(f, s.rows, s.cols)) + AlignUp(ScaleBytes(f, s.rows, s.cols, group));
}

// Hands out aligned slices of the arena and fills them.
class Uploader {
 public:
  Uploader(std::byte* base, bf16* scratch, int group)
      : base_(base), scratch_(scratch), group_(group) {}

  std::byte* Take(size_t bytes) {
    std::byte* p = base_ + offset_;
    offset_ += AlignUp(bytes);
    return p;
  }

  // Copies tensors back to back into one slice.
  const bf16* PutBf16(const std::vector<const TensorView*>& parts) {
    size_t total = 0;
    for (const TensorView* t : parts) total += t->nbytes;
    std::byte* dst = Take(total);
    size_t n = 0;
    for (const TensorView* t : parts) {
      CUDA_CHECK(cudaMemcpy(dst + n, t->data, t->nbytes, cudaMemcpyHostToDevice));
      n += t->nbytes;
    }
    return reinterpret_cast<const bf16*>(dst);
  }

  // Uploads a projection in `format`. Quantized ones go through the bf16 scratch buffer and
  // are quantized on the GPU.
  Linear PutLinear(const Source& s, WeightFormat format) {
    Linear l;
    l.format = format;
    l.rows = s.rows;
    l.cols = s.cols;
    l.group = group_;
    if (format == WeightFormat::kBf16) {
      l.data = PutBf16(s.parts);
      return l;
    }
    size_t n = 0;
    for (const TensorView* t : s.parts) {
      CUDA_CHECK(cudaMemcpy(reinterpret_cast<std::byte*>(scratch_) + n, t->data, t->nbytes,
                            cudaMemcpyHostToDevice));
      n += t->nbytes;
    }
    std::byte* data = Take(DataBytes(format, s.rows, s.cols));
    std::byte* scales = Take(ScaleBytes(format, s.rows, s.cols, group_));
    if (format == WeightFormat::kInt8) {
      kernels::QuantizeInt8(scratch_, s.rows, s.cols, reinterpret_cast<int8_t*>(data),
                            reinterpret_cast<float*>(scales), nullptr);
    } else {
      kernels::QuantizeInt4(scratch_, s.rows, s.cols, group_, reinterpret_cast<uint8_t*>(data),
                            reinterpret_cast<__half*>(scales), nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());  // the scratch buffer is reused by the next call
    l.data = data;
    l.scales = scales;
    return l;
  }

 private:
  std::byte* base_;
  bf16* scratch_;
  int group_;
  size_t offset_ = 0;
};

size_t LinearBytes(const Linear& l) {
  return DataBytes(l.format, l.rows, l.cols) + ScaleBytes(l.format, l.rows, l.cols, l.group);
}

}  // namespace

Weights Weights::Load(const std::string& dir, const ModelConfig& cfg, const LoadOptions& options) {
  const SafetensorsModel st(dir);
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  const int64_t V = cfg.vocab_size;
  const int64_t Q = cfg.q_dim();
  const int64_t KV = cfg.kv_dim();
  const WeightFormat fmt = options.format;

  const TensorView* embed = Expect(st, "model.embed_tokens.weight", {V, H});
  const TensorView* final_norm = Expect(st, "model.norm.weight", {H});
  const bool tied = cfg.tie_word_embeddings || !st.Has("lm_head.weight");
  const TensorView* lm_head_src = tied ? embed : Expect(st, "lm_head.weight", {V, H});
  const WeightFormat head_fmt = options.head();
  // A tied bf16 head is the embedding table itself; anything else needs its own copy.
  const bool head_is_embed = tied && head_fmt == WeightFormat::kBf16;
  const Source head = Stack({lm_head_src});

  struct LayerSources {
    const TensorView* attn_norm;
    const TensorView* mlp_norm;
    Source qkv, o, gate_up, down;
  };
  std::vector<LayerSources> ls;
  size_t total = AlignUp(embed->nbytes) + AlignUp(final_norm->nbytes);
  const int group = options.int4_group;
  if (!head_is_embed) total += ArenaBytes(head_fmt, head, group);
  size_t scratch = head_fmt == WeightFormat::kBf16 ? 0 : lm_head_src->nbytes;
  for (int i = 0; i < cfg.num_layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    LayerSources l;
    l.attn_norm = Expect(st, p + "input_layernorm.weight", {H});
    l.mlp_norm = Expect(st, p + "post_attention_layernorm.weight", {H});
    l.qkv = Stack({Expect(st, p + "self_attn.q_proj.weight", {Q, H}),
                   Expect(st, p + "self_attn.k_proj.weight", {KV, H}),
                   Expect(st, p + "self_attn.v_proj.weight", {KV, H})});
    l.o = Stack({Expect(st, p + "self_attn.o_proj.weight", {H, Q})});
    l.gate_up = Stack({Expect(st, p + "mlp.gate_proj.weight", {I, H}),
                       Expect(st, p + "mlp.up_proj.weight", {I, H})});
    l.down = Stack({Expect(st, p + "mlp.down_proj.weight", {H, I})});
    total += AlignUp(l.attn_norm->nbytes) + AlignUp(l.mlp_norm->nbytes);
    for (const Source* s : {&l.qkv, &l.o, &l.gate_up, &l.down}) {
      total += ArenaBytes(fmt, *s, group);
      if (fmt != WeightFormat::kBf16) {
        scratch = std::max(scratch, static_cast<size_t>(s->rows) * s->cols * sizeof(bf16));
      }
    }
    ls.push_back(l);
  }

  Weights w;
  w.options = options;
  w.arena_ = DeviceBuffer<std::byte>(total);
  DeviceBuffer<bf16> scratch_buf(scratch / sizeof(bf16));
  Uploader up(w.arena_.get(), scratch_buf.get(), group);
  w.embed = up.PutBf16({embed});
  w.final_norm = up.PutBf16({final_norm});
  if (head_is_embed) {
    w.lm_head =
        Linear{WeightFormat::kBf16, w.embed, nullptr, static_cast<int>(V), static_cast<int>(H)};
  } else {
    w.lm_head = up.PutLinear(head, head_fmt);
  }

  size_t decode_bytes =
      static_cast<size_t>(H) * sizeof(bf16) + final_norm->nbytes + LinearBytes(w.lm_head);
  for (const LayerSources& l : ls) {
    LayerWeights lw{};
    lw.attn_norm = up.PutBf16({l.attn_norm});
    lw.wqkv = up.PutLinear(l.qkv, fmt);
    lw.wo = up.PutLinear(l.o, fmt);
    lw.mlp_norm = up.PutBf16({l.mlp_norm});
    lw.w_gate_up = up.PutLinear(l.gate_up, fmt);
    lw.w_down = up.PutLinear(l.down, fmt);
    w.layers.push_back(lw);
    decode_bytes += l.attn_norm->nbytes + l.mlp_norm->nbytes;
    for (const Linear* x : {&lw.wqkv, &lw.wo, &lw.w_gate_up, &lw.w_down}) {
      decode_bytes += LinearBytes(*x);
    }
  }
  w.decode_bytes_ = decode_bytes;

  w.inv_freq_ = DeviceBuffer<float>(cfg.rope_inv_freq.size());
  CUDA_CHECK(cudaMemcpy(w.inv_freq_.get(), cfg.rope_inv_freq.data(),
                        cfg.rope_inv_freq.size() * sizeof(float), cudaMemcpyHostToDevice));
  w.rope_inv_freq = w.inv_freq_.get();
  return w;
}

}  // namespace infer
