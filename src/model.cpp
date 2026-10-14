#include "infer/model.h"

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

// Hands out aligned slices of the arena and copies tensors into them.
class Uploader {
 public:
  explicit Uploader(std::byte* base) : base_(base) {}

  // Copies one or more tensors back to back into a fresh slice; returns its start.
  const bf16* Put(std::initializer_list<const TensorView*> parts) {
    std::byte* start = base_ + offset_;
    size_t n = 0;
    for (const TensorView* t : parts) {
      CUDA_CHECK(cudaMemcpy(start + n, t->data, t->nbytes, cudaMemcpyHostToDevice));
      n += t->nbytes;
    }
    offset_ += AlignUp(n);
    return reinterpret_cast<const bf16*>(start);
  }

 private:
  std::byte* base_;
  size_t offset_ = 0;
};

}  // namespace

Weights Weights::Load(const std::string& dir, const ModelConfig& cfg) {
  const SafetensorsModel st(dir);
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  const int64_t V = cfg.vocab_size;
  const int64_t Q = cfg.q_dim();
  const int64_t KV = cfg.kv_dim();

  const TensorView* embed = Expect(st, "model.embed_tokens.weight", {V, H});
  const TensorView* final_norm = Expect(st, "model.norm.weight", {H});
  const bool tied = cfg.tie_word_embeddings || !st.Has("lm_head.weight");
  const TensorView* lm_head = tied ? nullptr : Expect(st, "lm_head.weight", {V, H});

  struct LayerViews {
    const TensorView *attn_norm, *q, *k, *v, *o, *mlp_norm, *gate, *up, *down;
  };
  std::vector<LayerViews> lv;
  size_t total = AlignUp(embed->nbytes) + AlignUp(final_norm->nbytes);
  if (lm_head != nullptr) total += AlignUp(lm_head->nbytes);
  for (int i = 0; i < cfg.num_layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    LayerViews l{};
    l.attn_norm = Expect(st, p + "input_layernorm.weight", {H});
    l.q = Expect(st, p + "self_attn.q_proj.weight", {Q, H});
    l.k = Expect(st, p + "self_attn.k_proj.weight", {KV, H});
    l.v = Expect(st, p + "self_attn.v_proj.weight", {KV, H});
    l.o = Expect(st, p + "self_attn.o_proj.weight", {H, Q});
    l.mlp_norm = Expect(st, p + "post_attention_layernorm.weight", {H});
    l.gate = Expect(st, p + "mlp.gate_proj.weight", {I, H});
    l.up = Expect(st, p + "mlp.up_proj.weight", {I, H});
    l.down = Expect(st, p + "mlp.down_proj.weight", {H, I});
    total += AlignUp(l.attn_norm->nbytes) + AlignUp(l.q->nbytes + l.k->nbytes + l.v->nbytes) +
             AlignUp(l.o->nbytes) + AlignUp(l.mlp_norm->nbytes) +
             AlignUp(l.gate->nbytes + l.up->nbytes) + AlignUp(l.down->nbytes);
    lv.push_back(l);
  }

  Weights w;
  w.arena_ = DeviceBuffer<std::byte>(total);
  Uploader up(w.arena_.get());
  w.embed = up.Put({embed});
  w.final_norm = up.Put({final_norm});
  w.lm_head = lm_head != nullptr ? up.Put({lm_head}) : w.embed;

  // A decode step reads one embedding row, the final norm and the whole LM head (which is the
  // embedding table itself when tied), plus every layer's weights below.
  size_t decode_bytes = static_cast<size_t>(H) * sizeof(bf16) + final_norm->nbytes +
                        (lm_head != nullptr ? lm_head->nbytes : embed->nbytes);
  for (const LayerViews& l : lv) {
    LayerWeights lw{};
    lw.attn_norm = up.Put({l.attn_norm});
    lw.wqkv = up.Put({l.q, l.k, l.v});
    lw.wo = up.Put({l.o});
    lw.mlp_norm = up.Put({l.mlp_norm});
    lw.w_gate_up = up.Put({l.gate, l.up});
    lw.w_down = up.Put({l.down});
    w.layers.push_back(lw);
    for (const TensorView* t :
         {l.attn_norm, l.q, l.k, l.v, l.o, l.mlp_norm, l.gate, l.up, l.down}) {
      decode_bytes += t->nbytes;
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
