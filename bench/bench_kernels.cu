// Times each kernel on its own at the shapes of a Llama-3.2-1B decode step.
//
// Every kernel is reported with the minimum number of bytes it has to move (its inputs and
// outputs, read or written once) divided by its time: the effective bandwidth. For the
// memory-bound decode kernels that number against the measured peak (bench/membw) says how
// close each one is to the limit.
//
// The L2 cache is 48 MB, bigger than several of the weight matrices. Run back to back, a
// matrix would stay in L2 and look faster than it can ever be in a real decode step, where the
// rest of the model (2.5 GB) streams through L2 between two uses of it. So each benchmark
// cycles through enough copies of its inputs to cover 256 MB.
//
//   bench_kernels [--peak-gbps 851.5] [--json FILE] [--filter SUBSTRING]

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "infer/cuda_util.h"
#include "infer/kernels.h"

namespace {

using infer::DeviceBuffer;
using infer::kernels::bf16;
namespace k = infer::kernels;

constexpr size_t kRotationBytes = 256ull << 20;
constexpr int kIters = 50;

// Llama-3.2-1B
constexpr int H = 2048, I = 8192, V = 128256, NH = 32, NKV = 8, HD = 64;
constexpr int QKV = (NH + 2 * NKV) * HD;

struct Options {
  double peak_gbps = 0.0;
  std::string json;
  std::string filter;
};

// A set of `copies` equally sized device buffers, filled with small random values.
template <typename T>
struct Rotation {
  std::vector<DeviceBuffer<T>> bufs;
  Rotation(size_t count, int copies, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    std::vector<T> host(count);
    for (auto& x : host) x = static_cast<T>(dist(rng));
    for (int c = 0; c < copies; ++c) {
      bufs.emplace_back(count);
      CUDA_CHECK(
          cudaMemcpy(bufs.back().get(), host.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
  }
  T* get(int i) const { return bufs[static_cast<size_t>(i) % bufs.size()].get(); }
};

int CopiesFor(size_t bytes) {
  return static_cast<int>(std::max<size_t>(2, (kRotationBytes + bytes - 1) / bytes));
}

// Time of one launch, in microseconds.
//
// Launches are timed in batches: kIters launches (cycling through the input copies) are
// captured into a CUDA graph and the graph is timed with events. Timing a single launch
// between two events would mostly measure how long the CPU takes to submit it, which under
// WSL is tens of microseconds, far longer than the small kernels themselves. The median of
// several graph runs is reported.
double TimeUs(const std::function<void(int, cudaStream_t)>& launch) {
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < kIters; ++i) launch(i, s);
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));

  cudaEvent_t a;
  cudaEvent_t b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaGraphLaunch(exec, s));  // warmup
  std::vector<double> us;
  for (int rep = 0; rep < 7; ++rep) {
    CUDA_CHECK(cudaEventRecord(a, s));
    CUDA_CHECK(cudaGraphLaunch(exec, s));
    CUDA_CHECK(cudaEventRecord(b, s));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    us.push_back(static_cast<double>(ms) * 1e3 / kIters);
  }
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(s));
  std::sort(us.begin(), us.end());
  return us[us.size() / 2];
}

class Runner {
 public:
  explicit Runner(Options o) : opt_(std::move(o)) {
    if (!opt_.json.empty()) json_.open(opt_.json, std::ios::app);
    std::printf("%-28s %-10s %10s %10s %8s\n", "kernel", "variant", "time us", "GB/s", "% peak");
  }

  bool Wanted(const std::string& name) const {
    return opt_.filter.empty() || name.find(opt_.filter) != std::string::npos;
  }

  void Report(const std::string& name, const std::string& variant, double bytes, double us) {
    const double gbps = bytes / (us * 1e-6) / 1e9;
    const double pct = opt_.peak_gbps > 0 ? 100.0 * gbps / opt_.peak_gbps : 0.0;
    std::printf("%-28s %-10s %10.1f %10.1f %8.1f\n", name.c_str(), variant.c_str(), us, gbps, pct);
    if (json_.is_open()) {
      json_ << "{\"kernel\":\"" << name << "\",\"variant\":\"" << variant << "\",\"us\":" << us
            << ",\"bytes\":" << bytes << ",\"gbps\":" << gbps << ",\"pct_peak\":" << pct << "}\n";
    }
  }

 private:
  Options opt_;
  std::ofstream json_;
};

void BenchMatmul(Runner& r, const std::string& name, int T, int N, int K) {
  if (!r.Wanted(name)) return;
  const size_t wbytes = static_cast<size_t>(N) * K * sizeof(bf16);
  const int copies = CopiesFor(wbytes);
  Rotation<bf16> W(static_cast<size_t>(N) * K, copies, 1);
  Rotation<float> x(static_cast<size_t>(T) * K, copies, 2);
  DeviceBuffer<float> y(static_cast<size_t>(T) * N);
  const double bytes = static_cast<double>(wbytes) + 4.0 * T * (K + N);
  r.Report(name, "naive", bytes, TimeUs([&](int i, cudaStream_t st) {
             k::MatmulNaive(x.get(i), W.get(i), T, N, K, y.get(), false, st);
           }));
}

void BenchRmsNorm(Runner& r) {
  const std::string name = "rmsnorm H=2048";
  if (!r.Wanted(name)) return;
  Rotation<float> x(H, 64, 3);
  Rotation<bf16> w(H, 64, 4);
  DeviceBuffer<float> out(H);
  const double bytes = 4.0 * H * 2 + 2.0 * H;
  r.Report(name, "naive", bytes, TimeUs([&](int i, cudaStream_t st) {
             k::RmsNormNaive(x.get(i), w.get(i), 1, H, 1e-5f, out.get(), st);
           }));
}

void BenchAttention(Runner& r, int ctx) {
  const std::string name = "attention decode ctx=" + std::to_string(ctx);
  if (!r.Wanted(name)) return;
  const int max_seq = ctx;
  const size_t cache = static_cast<size_t>(NKV) * max_seq * HD;
  const int copies = CopiesFor(2 * cache * sizeof(bf16));
  Rotation<bf16> kc(cache, copies, 5);
  Rotation<bf16> vc(cache, copies, 6);
  Rotation<float> qkv(QKV, copies, 7);
  DeviceBuffer<float> out(static_cast<size_t>(NH) * HD);
  // K and V for every cached position, plus q in and the output.
  const double bytes = 2.0 * cache * sizeof(bf16) + 4.0 * (QKV + NH * HD);
  r.Report(name, "naive", bytes, TimeUs([&](int i, cudaStream_t st) {
             k::AttentionNaive(qkv.get(i), 1, ctx - 1, NH, NKV, HD, kc.get(i), vc.get(i), max_seq,
                               out.get(), st);
           }));
}

void BenchSmallOps(Runner& r) {
  if (r.Wanted("rope")) {
    Rotation<float> qkv(QKV, 64, 8);
    Rotation<float> freq(HD / 2, 64, 9);
    r.Report("rope", "naive", 4.0 * 2 * (NH + NKV) * HD, TimeUs([&](int i, cudaStream_t st) {
               k::RopeNaive(qkv.get(i), 1, NH, NKV, HD, freq.get(i), 100, st);
             }));
  }
  if (r.Wanted("swiglu")) {
    Rotation<float> gu(2 * I, 64, 10);
    DeviceBuffer<float> out(I);
    r.Report("swiglu I=8192", "naive", 4.0 * 3 * I,
             TimeUs([&](int i, cudaStream_t st) { k::SwiGlu(gu.get(i), 1, I, out.get(), st); }));
  }
  if (r.Wanted("argmax")) {
    Rotation<float> logits(V, CopiesFor(4ull * V), 11);
    DeviceBuffer<int> out(1);
    r.Report("argmax V=128256", "naive", 4.0 * V,
             TimeUs([&](int i, cudaStream_t st) { k::Argmax(logits.get(i), V, out.get(), st); }));
  }
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string f = argv[i];
    if (i + 1 >= argc) {
      std::fprintf(stderr, "usage: bench_kernels [--peak-gbps X] [--json FILE] [--filter S]\n");
      return 2;
    }
    if (f == "--peak-gbps") {
      opt.peak_gbps = std::stod(argv[++i]);
    } else if (f == "--json") {
      opt.json = argv[++i];
    } else if (f == "--filter") {
      opt.filter = argv[++i];
    } else {
      std::fprintf(stderr, "unknown flag %s\n", f.c_str());
      return 2;
    }
  }
  try {
    Runner r(opt);
    BenchMatmul(r, "gemv qkv 3072x2048", 1, QKV, H);
    BenchMatmul(r, "gemv o 2048x2048", 1, H, NH * HD);
    BenchMatmul(r, "gemv gate_up 16384x2048", 1, 2 * I, H);
    BenchMatmul(r, "gemv down 2048x8192", 1, H, I);
    BenchMatmul(r, "gemv lm_head 128256x2048", 1, V, H);
    BenchRmsNorm(r);
    BenchAttention(r, 512);
    BenchAttention(r, 2048);
    BenchSmallOps(r);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
