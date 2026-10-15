// End-to-end benchmark: prefill and decode speed at fixed context lengths.
//
// For each context length C:
//   prefill  C random tokens from an empty cache        -> time to first token, prefill tok/s
//   decode   G greedy steps starting at position C      -> decode tok/s
// Each measurement is repeated (after warmups) and the median and range are reported.
//
// Decode is memory-bound, so it is also reported as achieved bandwidth: bytes a decode step
// must read (all weights except the embedding table, plus the KV cache up to the current
// position) divided by the time per step, and as a percentage of the measured peak passed in
// with --peak-gbps (bench/membw).
//
//   bench_engine --model models/random-1b --contexts 128,512,2048 --generate 128 --runs 5
//       --peak-gbps 851.5 --json results.jsonl
//
// With --profile-step it instead prefills the first context length, runs a few decode steps,
// and wraps exactly one more decode step in cudaProfilerStart/Stop, for
// `ncu --profile-from-start off` and `nsys --capture-range=cudaProfilerApi`.

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "infer/config.h"
#include "infer/engine.h"
#include "infer/model.h"

namespace {

struct Args {
  std::string model;
  std::vector<int> contexts = {128, 512, 2048};
  int generate = 128;
  int runs = 5;
  int warmup = 1;
  double peak_gbps = 0.0;
  std::string backend = "optimized";
  bool graph = true;
  infer::LoadOptions load;
  std::string json;
  std::string label;
  bool profile_step = false;
  infer::Sampling sampling;
};

[[noreturn]] void Usage() {
  std::fprintf(stderr,
               "usage: bench_engine --model DIR [--contexts 128,512,2048] [--generate 128]\n"
               "                    [--runs 5] [--warmup 1] [--peak-gbps X] [--backend NAME]\n"
               "                    [--json FILE] [--label TEXT]\n");
  std::exit(2);
}

std::vector<int> ParseList(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::stoi(item));
  return out;
}

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--profile-step") {
      a.profile_step = true;
      continue;
    }
    if (i + 1 >= argc) Usage();
    const std::string v = argv[++i];
    if (flag == "--model") {
      a.model = v;
    } else if (flag == "--contexts") {
      a.contexts = ParseList(v);
    } else if (flag == "--generate") {
      a.generate = std::stoi(v);
    } else if (flag == "--runs") {
      a.runs = std::stoi(v);
    } else if (flag == "--warmup") {
      a.warmup = std::stoi(v);
    } else if (flag == "--peak-gbps") {
      a.peak_gbps = std::stod(v);
    } else if (flag == "--backend") {
      a.backend = v;
    } else if (flag == "--weights") {
      a.load.format = infer::ParseWeightFormat(v);
    } else if (flag == "--lm-head") {
      a.load.lm_head_format = infer::ParseWeightFormat(v);
    } else if (flag == "--int4-group") {
      a.load.int4_group = std::stoi(v);
    } else if (flag == "--graph") {
      a.graph = v != "off";
    } else if (flag == "--temperature") {
      a.sampling.temperature = std::stof(v);
    } else if (flag == "--top-k") {
      a.sampling.top_k = std::stoi(v);
    } else if (flag == "--top-p") {
      a.sampling.top_p = std::stof(v);
    } else if (flag == "--json") {
      a.json = v;
    } else if (flag == "--label") {
      a.label = v;
    } else {
      Usage();
    }
  }
  if (a.model.empty() || a.contexts.empty() || a.generate < 1 || a.runs < 1) Usage();
  return a;
}

struct Stats {
  double median, min, max;
};

Stats Summarize(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  const double median = n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  return {median, v.front(), v.back()};
}

// Times work enqueued on `stream` by fn, using events on the GPU timeline.
template <typename Fn>
double TimeMs(cudaStream_t stream, Fn fn) {
  cudaEvent_t a;
  cudaEvent_t b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  cudaEventRecord(a, stream);
  fn();
  cudaEventRecord(b, stream);
  cudaEventSynchronize(b);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return static_cast<double>(ms);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    // Inside the try: std::stoi throws on input like "--generate abc".
    const Args args = ParseArgs(argc, argv);
    const infer::ModelConfig cfg = infer::ModelConfig::FromJsonFile(
        (std::filesystem::path(args.model) / "config.json").string());
    const infer::Weights weights = infer::Weights::Load(args.model, cfg, args.load);

    const int max_ctx = *std::max_element(args.contexts.begin(), args.contexts.end());
    infer::EngineOptions opt;
    opt.max_seq = max_ctx + args.generate;
    opt.backend = infer::ParseBackend(args.backend);
    opt.cuda_graph = args.graph;
    infer::Engine engine(cfg, weights, opt);

    if (args.profile_step) {
      std::mt19937 rng(1234);
      std::uniform_int_distribution<int> tok(0, cfg.vocab_size - 1);
      std::vector<int> prompt(static_cast<size_t>(args.contexts[0]));
      for (int& t : prompt) t = tok(rng);
      engine.Prefill(prompt);
      engine.DecodeGreedy(8);
      cudaProfilerStart();
      engine.DecodeGreedy(1);
      cudaDeviceSynchronize();
      cudaProfilerStop();
      return 0;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf(
        "model     %s, %s weights%s (%.3f GB read per decode step, excluding KV cache)\n",
        args.model.c_str(), infer::WeightFormatName(args.load.format),
        args.load.head() != args.load.format
            ? (std::string(", ") + infer::WeightFormatName(args.load.head()) + " LM head").c_str()
            : "",
        static_cast<double>(weights.decode_bytes()) / 1e9);
    std::printf("device    %s\n", prop.name);
    std::printf("backend   %s, %d warmup + %d timed runs, %d decode steps per run\n\n",
                args.backend.c_str(), args.warmup, args.runs, args.generate);
    std::printf("%7s | %10s %10s | %20s | %9s %7s\n", "context", "TTFT ms", "prefill/s",
                "decode tok/s (range)", "GB/s", "% peak");

    std::ofstream json;
    if (!args.json.empty()) json.open(args.json, std::ios::app);

    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> tok(0, cfg.vocab_size - 1);

    for (int ctx : args.contexts) {
      std::vector<int> prompt(static_cast<size_t>(ctx));
      for (int& t : prompt) t = tok(rng);

      std::vector<double> prefill_ms;
      std::vector<double> decode_ms;
      for (int run = 0; run < args.warmup + args.runs; ++run) {
        engine.Reset();
        const double p = TimeMs(engine.stream(), [&] { engine.Prefill(prompt); });
        const double d =
            TimeMs(engine.stream(), [&] { engine.DecodeSampled(args.generate, args.sampling); });
        if (run >= args.warmup) {
          prefill_ms.push_back(p);
          decode_ms.push_back(d);
        }
      }

      const Stats p = Summarize(prefill_ms);
      std::vector<double> tps;
      for (double ms : decode_ms) tps.push_back(args.generate / (ms / 1e3));
      const Stats d = Summarize(tps);

      // Average KV cache read per step: keys and values for every layer, for positions
      // 0..pos, where pos runs from ctx to ctx + generate - 1.
      const double mean_pos = ctx + (args.generate - 1) / 2.0 + 1;
      const double kv_bytes = 2.0 * cfg.num_layers * mean_pos * cfg.kv_dim() * 2.0;
      const double step_bytes = static_cast<double>(weights.decode_bytes()) + kv_bytes;
      const double gbps = step_bytes * d.median / 1e9;
      const double pct = args.peak_gbps > 0 ? 100.0 * gbps / args.peak_gbps : 0.0;

      std::printf("%7d | %10.2f %10.0f | %7.1f (%5.1f-%5.1f) | %9.1f %7.1f\n", ctx, p.median,
                  ctx / (p.median / 1e3), d.median, d.min, d.max, gbps, pct);

      if (json.is_open()) {
        json << "{\"label\":\"" << args.label << "\",\"backend\":\"" << args.backend
             << "\",\"weights\":\"" << infer::WeightFormatName(args.load.format)
             << "\",\"lm_head\":\"" << infer::WeightFormatName(args.load.head())
             << "\",\"sampling\":\"" << (args.sampling.greedy() ? "greedy" : "sampled")
             << "\",\"graph\":\"" << (args.graph ? "on" : "off") << "\",\"context\":" << ctx
             << ",\"generate\":" << args.generate << ",\"runs\":" << args.runs
             << ",\"ttft_ms_median\":" << p.median << ",\"ttft_ms_min\":" << p.min
             << ",\"ttft_ms_max\":" << p.max << ",\"prefill_tok_s\":" << ctx / (p.median / 1e3)
             << ",\"decode_tok_s_median\":" << d.median << ",\"decode_tok_s_min\":" << d.min
             << ",\"decode_tok_s_max\":" << d.max << ",\"step_bytes\":" << step_bytes
             << ",\"decode_gbps\":" << gbps << ",\"pct_peak\":" << pct << "}\n";
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
