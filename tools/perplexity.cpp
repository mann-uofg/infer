// Perplexity over a token file, the same way llama.cpp's llama-perplexity computes it, so the
// numbers can be compared directly:
//
//   - cut the tokens into chunks of --ctx tokens (a partial last chunk is dropped)
//   - optionally replace each chunk's first token with BOS (llama.cpp does this for models that
//     add a BOS token, Llama 3 among them)
//   - run each chunk from an empty cache and score only its second half: the logits at
//     positions ctx/2 .. ctx-2 predicting tokens ctx/2+1 .. ctx-1, so every scored token has
//     at least ctx/2 tokens of context
//   - perplexity = exp(mean negative log-likelihood over all scored tokens)
//
// The token file is raw little-endian int32 (scripts/prepare_wikitext.py writes one).
//
//   perplexity --model models/llama-3.2-1b --tokens models/llama-3.2-1b/wikitext2-test.bin
//              --ctx 512 --bos 128000 --weights int4

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "infer/config.h"
#include "infer/engine.h"
#include "infer/kernels.h"
#include "infer/model.h"

namespace {

struct Args {
  std::string model;
  std::string tokens;
  int ctx = 512;
  int chunks = 0;  // 0: all
  int bos = -1;
  std::string backend = "optimized";
  infer::LoadOptions load;
};

[[noreturn]] void Usage() {
  std::fprintf(stderr,
               "usage: perplexity --model DIR --tokens FILE [--ctx 512] [--chunks N] [--bos ID]\n"
               "                  [--weights bf16|int8|int4] [--quant-lm-head on|off]\n"
               "                  [--backend naive|optimized]\n");
  std::exit(2);
}

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string f = argv[i];
    if (i + 1 >= argc) Usage();
    const std::string v = argv[++i];
    if (f == "--model") {
      a.model = v;
    } else if (f == "--tokens") {
      a.tokens = v;
    } else if (f == "--ctx") {
      a.ctx = std::stoi(v);
    } else if (f == "--chunks") {
      a.chunks = std::stoi(v);
    } else if (f == "--bos") {
      a.bos = std::stoi(v);
    } else if (f == "--weights") {
      a.load.format = infer::ParseWeightFormat(v);
    } else if (f == "--quant-lm-head") {
      a.load.quantize_lm_head = v != "off";
    } else if (f == "--backend") {
      a.backend = v;
    } else {
      Usage();
    }
  }
  if (a.model.empty() || a.tokens.empty() || a.ctx < 4) Usage();
  return a;
}

std::vector<int> ReadTokens(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  const auto size = static_cast<size_t>(in.tellg());
  if (size % sizeof(int32_t) != 0) throw std::runtime_error(path + ": size is not a multiple of 4");
  std::vector<int> tokens(size / sizeof(int32_t));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(size));
  return tokens;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = ParseArgs(argc, argv);
  try {
    const infer::ModelConfig cfg = infer::ModelConfig::FromJsonFile(
        (std::filesystem::path(args.model) / "config.json").string());
    const infer::Weights weights = infer::Weights::Load(args.model, cfg, args.load);
    const std::vector<int> tokens = ReadTokens(args.tokens);

    int chunks = static_cast<int>(tokens.size()) / args.ctx;
    if (args.chunks > 0 && args.chunks < chunks) chunks = args.chunks;
    if (chunks == 0) throw std::runtime_error("fewer tokens than one chunk");

    infer::EngineOptions opt;
    opt.max_seq = args.ctx;
    opt.max_logit_rows = args.ctx;
    opt.backend = infer::ParseBackend(args.backend);
    infer::Engine engine(cfg, weights, opt);

    const int first = args.ctx / 2;           // first scored logit row
    const int scored = args.ctx - 1 - first;  // rows first .. ctx-2
    infer::DeviceBuffer<int> targets(static_cast<size_t>(scored));
    infer::DeviceBuffer<float> nll(static_cast<size_t>(scored));
    std::vector<float> host_nll(static_cast<size_t>(scored));

    std::printf("model %s, %s weights%s, %d chunks of %d tokens, scoring %d tokens per chunk\n",
                args.model.c_str(), infer::WeightFormatName(args.load.format),
                args.load.format != infer::WeightFormat::kBf16 && !args.load.quantize_lm_head
                    ? " (bf16 LM head)"
                    : "",
                chunks, args.ctx, scored);

    double sum = 0.0;
    double sum_sq = 0.0;
    long long count = 0;
    for (int c = 0; c < chunks; ++c) {
      std::vector<int> chunk(tokens.begin() + static_cast<long>(c) * args.ctx,
                             tokens.begin() + static_cast<long>(c + 1) * args.ctx);
      if (args.bos >= 0) chunk[0] = args.bos;
      engine.Reset();
      engine.Prefill(chunk, true);
      CUDA_CHECK(cudaMemcpyAsync(targets.get(), chunk.data() + first + 1,
                                 static_cast<size_t>(scored) * sizeof(int), cudaMemcpyHostToDevice,
                                 engine.stream()));
      infer::kernels::CrossEntropy(
          engine.LogitsDevice() + static_cast<size_t>(first) * cfg.vocab_size, scored,
          cfg.vocab_size, targets.get(), nll.get(), engine.stream());
      CUDA_CHECK(cudaMemcpyAsync(host_nll.data(), nll.get(), host_nll.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost, engine.stream()));
      CUDA_CHECK(cudaStreamSynchronize(engine.stream()));
      for (float v : host_nll) {
        sum += v;
        sum_sq += static_cast<double>(v) * v;
      }
      count += scored;
      std::fprintf(stderr, "[%d] %.4f\n", c + 1, std::exp(sum / static_cast<double>(count)));
    }

    // Standard error of the mean NLL, carried through exp (as llama.cpp reports it).
    const double mean = sum / static_cast<double>(count);
    const double var = sum_sq / static_cast<double>(count) - mean * mean;
    const double se = std::sqrt(var / static_cast<double>(count - 1));
    const double ppl = std::exp(mean);
    std::printf("tokens scored %lld\nmean nll %.6f\nperplexity %.4f +/- %.4f\n", count, mean, ppl,
                ppl * se);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
