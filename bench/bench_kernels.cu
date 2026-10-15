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
// Prefill GEMMs are compute-bound, so they are reported in TFLOPS instead, next to cuBLAS as a
// baseline: cuBLAS SGEMM (fp32, same precision as mine, W converted to fp32 beforehand) and
// cuBLAS on bf16 inputs with tensor cores (what a production engine would use).
//
//   bench_kernels [--peak-gbps 851.5] [--json FILE] [--filter SUBSTRING]

#include <cublas_v2.h>
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

  void ReportFlops(const std::string& name, const std::string& variant, double flops, double us) {
    const double tflops = flops / (us * 1e-6) / 1e12;
    std::printf("%-28s %-10s %10.1f %10.1f TFLOPS\n", name.c_str(), variant.c_str(), us, tflops);
    if (json_.is_open()) {
      json_ << "{\"kernel\":\"" << name << "\",\"variant\":\"" << variant << "\",\"us\":" << us
            << ",\"flops\":" << flops << ",\"tflops\":" << tflops << "}\n";
    }
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

// Decode projections. `swiglu` means W holds 2N rows (gate and up) and produces N outputs.
// The optimized variant is run the way the engine runs it: with the fused RMSNorm where the
// engine fuses one (qkv, gate_up, lm_head) and the residual add where it adds one (o, down).
void BenchGemv(Runner& r, const std::string& name, int N, int K, k::GemvOut out, bool norm) {
  if (!r.Wanted(name)) return;
  const int rows = out == k::GemvOut::kSwiGlu ? 2 * N : N;
  const size_t wbytes = static_cast<size_t>(rows) * K * sizeof(bf16);
  const int copies = CopiesFor(wbytes);
  Rotation<bf16> W(static_cast<size_t>(rows) * K, copies, 1);
  Rotation<float> x(static_cast<size_t>(K), copies, 2);
  Rotation<bf16> nw(static_cast<size_t>(K), copies, 3);
  DeviceBuffer<float> y(static_cast<size_t>(rows));
  DeviceBuffer<float> act(static_cast<size_t>(N));
  const double bytes = static_cast<double>(wbytes) + 4.0 * (K + rows);
  r.Report(name, "naive", bytes, TimeUs([&](int i, cudaStream_t st) {
             k::MatmulNaive(x.get(i), W.get(i), 1, rows, K, y.get(), false, st);
           }));
  r.Report(name, "optimized", bytes, TimeUs([&](int i, cudaStream_t st) {
             k::Gemv(x.get(i), k::Linear{k::WeightFormat::kBf16, W.get(i), nullptr, rows, K},
                     out == k::GemvOut::kSwiGlu ? act.get() : y.get(), out,
                     norm ? nw.get(i) : nullptr, 1e-5f, st);
           }));

  // Quantized: the same kernel structure reading fewer bytes. GB/s is the bandwidth on the
  // quantized bytes (data + scales), so the % of peak says how well the kernel streams them.
  for (k::WeightFormat f : {k::WeightFormat::kInt8, k::WeightFormat::kInt4}) {
    const size_t n = static_cast<size_t>(rows) * K;
    const size_t data = f == k::WeightFormat::kInt8 ? n : n / 2;
    const size_t scales = f == k::WeightFormat::kInt8 ? rows * sizeof(float)
                                                      : n / k::kDefaultInt4Group * sizeof(__half);
    const int qcopies = CopiesFor(data + scales);
    std::vector<DeviceBuffer<std::byte>> qd;
    std::vector<DeviceBuffer<std::byte>> qs;
    for (int c = 0; c < qcopies; ++c) {
      qd.emplace_back(data);
      qs.emplace_back(scales);
      if (f == k::WeightFormat::kInt8) {
        k::QuantizeInt8(W.get(c), rows, K, reinterpret_cast<int8_t*>(qd.back().get()),
                        reinterpret_cast<float*>(qs.back().get()), nullptr);
      } else {
        k::QuantizeInt4(W.get(c), rows, K, k::kDefaultInt4Group,
                        reinterpret_cast<uint8_t*>(qd.back().get()),
                        reinterpret_cast<__half*>(qs.back().get()), nullptr);
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double qbytes = static_cast<double>(data + scales) + 4.0 * (K + rows);
    r.Report(name, f == k::WeightFormat::kInt8 ? "int8" : "int4", qbytes,
             TimeUs([&](int i, cudaStream_t st) {
               const size_t c = static_cast<size_t>(i % qcopies);
               k::Gemv(x.get(i), k::Linear{f, qd[c].get(), qs[c].get(), rows, K},
                       out == k::GemvOut::kSwiGlu ? act.get() : y.get(), out,
                       norm ? nw.get(i) : nullptr, 1e-5f, st);
             }));
  }
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
  DeviceBuffer<int> pos(1);
  const int last = ctx - 1;
  CUDA_CHECK(cudaMemcpy(pos.get(), &last, sizeof(int), cudaMemcpyHostToDevice));
  for (int splits : {8, 16, 32, 64}) {
    DeviceBuffer<float> ws(k::AttentionDecodeWorkspace(NH, HD, splits));
    r.Report(name, "split" + std::to_string(splits), bytes, TimeUs([&](int i, cudaStream_t st) {
               k::AttentionDecode(qkv.get(i), NH, NKV, HD, kc.get(i), vc.get(i), max_seq, pos.get(),
                                  splits, ws.get(), out.get(), st);
             }));
  }
}

void CublasCheck(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    throw infer::CudaError(std::string(what) + " failed with cuBLAS status " +
                           std::to_string(static_cast<int>(s)));
  }
}

__global__ void ToFloatKernel(const bf16* in, float* out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(in[i]);
}

__global__ void ToBf16Kernel(const float* in, bf16* out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(in[i]);
}

// Prefill projection C[T, N] = A[T, K] x W[N, K]^T.
void BenchGemm(Runner& r, cublasHandle_t cublas, const std::string& name, int T, int N, int K) {
  if (!r.Wanted(name)) return;
  const size_t wn = static_cast<size_t>(N) * K;
  const size_t an = static_cast<size_t>(T) * K;
  const int copies = CopiesFor(wn * sizeof(bf16) + an * sizeof(float));
  Rotation<bf16> W(wn, copies, 60);
  Rotation<float> A(an, copies, 61);
  DeviceBuffer<float> C(static_cast<size_t>(T) * N);
  const double flops = 2.0 * T * N * K;

  if (T <= 256) {  // the naive kernel takes seconds at T = 2048
    r.ReportFlops(name, "naive", flops, TimeUs([&](int i, cudaStream_t st) {
                    k::MatmulNaive(A.get(i), W.get(i), T, N, K, C.get(), false, st);
                  }));
  }
  r.ReportFlops(name, "tiled", flops, TimeUs([&](int i, cudaStream_t st) {
                  k::Gemm(A.get(i), k::Linear{k::WeightFormat::kBf16, W.get(i), nullptr, N, K}, T,
                          C.get(), false, st);
                }));

  // Column-major view: C^T [N, T] = W [N, K] x A^T [K, T], with W read transposed.
  std::vector<DeviceBuffer<float>> wf;
  std::vector<DeviceBuffer<bf16>> ab;
  for (int c = 0; c < copies; ++c) {
    wf.emplace_back(wn);
    ab.emplace_back(an);
    ToFloatKernel<<<static_cast<unsigned>((wn + 255) / 256), 256>>>(W.get(c), wf.back().get(), wn);
    ToBf16Kernel<<<static_cast<unsigned>((an + 255) / 256), 256>>>(A.get(c), ab.back().get(), an);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const float alpha = 1.0f;
  const float beta = 0.0f;
  r.ReportFlops(name, "cublas-f32", flops, TimeUs([&](int i, cudaStream_t st) {
                  CublasCheck(cublasSetStream(cublas, st), "cublasSetStream");
                  CublasCheck(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &alpha,
                                           wf[static_cast<size_t>(i % copies)].get(), CUDA_R_32F, K,
                                           A.get(i), CUDA_R_32F, K, &beta, C.get(), CUDA_R_32F, N,
                                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                              "cublasGemmEx f32");
                }));
  r.ReportFlops(
      name, "cublas-bf16", flops, TimeUs([&](int i, cudaStream_t st) {
        CublasCheck(cublasSetStream(cublas, st), "cublasSetStream");
        CublasCheck(
            cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &alpha, W.get(i), CUDA_R_16BF,
                         K, ab[static_cast<size_t>(i % copies)].get(), CUDA_R_16BF, K, &beta,
                         C.get(), CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
            "cublasGemmEx bf16");
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
  if (r.Wanted("sample")) {
    Rotation<float> logits(V, CopiesFor(4ull * V), 12);
    Rotation<float> u(2, 2, 13);
    DeviceBuffer<int> pos(1);
    CUDA_CHECK(cudaMemset(pos.get(), 0, sizeof(int)));
    DeviceBuffer<int> tok(1);
    DeviceBuffer<int> hist(1);
    struct S {
      const char* name;
      float t;
      int k;
      float p;
    };
    for (const S& s : {S{"sample T=0.6 p=0.9", 0.6f, 0, 0.9f}, S{"sample T=1 k=40", 1.0f, 40, 1.0f},
                       S{"sample T=1 k=40 p=0.95", 1.0f, 40, 0.95f}}) {
      r.Report(s.name, "radix", 4.0 * V, TimeUs([&](int i, cudaStream_t st) {
                 k::Sample(logits.get(i), V, s.t, s.k, s.p, u.get(0), pos.get(), tok.get(),
                           hist.get(), st);
               }));
    }
  }
  if (r.Wanted("argmax")) {
    DeviceBuffer<unsigned long long> best(1);
    CUDA_CHECK(cudaMemset(best.get(), 0, sizeof(unsigned long long)));
    Rotation<float> lg(V, CopiesFor(4ull * V), 14);
    r.Report("argmax V=128256", "multiblock", 4.0 * V, TimeUs([&](int i, cudaStream_t st) {
               k::ArgmaxPartial(lg.get(i), V, best.get(), st);
             }));
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
    BenchGemv(r, "gemv qkv 3072x2048", QKV, H, k::GemvOut::kStore, true);
    BenchGemv(r, "gemv o 2048x2048", H, NH * HD, k::GemvOut::kAccumulate, false);
    BenchGemv(r, "gemv gate_up 16384x2048", I, H, k::GemvOut::kSwiGlu, true);
    BenchGemv(r, "gemv down 2048x8192", H, I, k::GemvOut::kAccumulate, false);
    BenchGemv(r, "gemv lm_head 128256x2048", V, H, k::GemvOut::kStore, true);
    BenchRmsNorm(r);
    BenchAttention(r, 512);
    BenchAttention(r, 2048);
    BenchSmallOps(r);

    // cuBLAS needs a fixed workspace to be capturable in a CUDA graph.
    cublasHandle_t cublas;
    CublasCheck(cublasCreate(&cublas), "cublasCreate");
    DeviceBuffer<std::byte> workspace(64ull << 20);
    CublasCheck(cublasSetWorkspace(cublas, workspace.get(), workspace.bytes()), "workspace");
    BenchGemm(r, cublas, "gemm qkv T=128", 128, QKV, H);
    BenchGemm(r, cublas, "gemm qkv T=2048", 2048, QKV, H);
    BenchGemm(r, cublas, "gemm o T=2048", 2048, H, NH * HD);
    BenchGemm(r, cublas, "gemm gate_up T=2048", 2048, 2 * I, H);
    BenchGemm(r, cublas, "gemm down T=2048", 2048, H, I);
    cublasDestroy(cublas);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
