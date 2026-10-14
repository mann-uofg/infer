// Command-line front end. Takes and prints token ids; tokenization lives in Python
// (scripts/generate.py).
//
//   infer --model DIR --tokens 128000,791,6864 --generate 32
//   infer --model DIR --tokens ... --dump OUTDIR     # hidden states + logits for comparison

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "infer/config.h"
#include "infer/engine.h"
#include "infer/model.h"

namespace {

struct Args {
  std::string model;
  std::vector<int> tokens;
  int generate = 0;
  std::string dump;
  int max_tokens = 0;
};

[[noreturn]] void Usage() {
  std::fprintf(stderr,
               "usage: infer --model DIR --tokens ID,ID,... [--generate N] [--dump DIR]\n"
               "             [--max-tokens N]\n");
  std::exit(2);
}

std::vector<int> ParseTokens(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(std::stoi(item));
  }
  return out;
}

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (i + 1 >= argc) Usage();
    const std::string value = argv[++i];
    if (flag == "--model") {
      a.model = value;
    } else if (flag == "--tokens") {
      a.tokens = ParseTokens(value);
    } else if (flag == "--generate") {
      a.generate = std::stoi(value);
    } else if (flag == "--dump") {
      a.dump = value;
    } else if (flag == "--max-tokens") {
      a.max_tokens = std::stoi(value);
    } else {
      Usage();
    }
  }
  if (a.model.empty() || a.tokens.empty() || a.generate < 0) Usage();
  return a;
}

void WriteFloats(const std::filesystem::path& path, const std::vector<float>& v) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
  if (!out) throw std::runtime_error("failed to write " + path.string());
}

double Seconds(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = ParseArgs(argc, argv);
  try {
    const auto t0 = std::chrono::steady_clock::now();
    const infer::ModelConfig cfg = infer::ModelConfig::FromJsonFile(
        (std::filesystem::path(args.model) / "config.json").string());
    const infer::Weights weights = infer::Weights::Load(args.model, cfg);
    const auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "loaded %.2f GB of weights in %.2f s\n",
                 static_cast<double>(weights.device_bytes()) / 1e9, Seconds(t0, t1));

    const int needed = static_cast<int>(args.tokens.size()) + args.generate;
    const int max_tokens = args.max_tokens > 0 ? args.max_tokens : needed;
    infer::Engine engine(cfg, weights, max_tokens, infer::Backend::kNaive);

    if (!args.dump.empty()) {
      std::filesystem::create_directories(args.dump);
      std::vector<std::vector<float>> hidden;
      const std::vector<float> logits = engine.Forward(args.tokens, true, &hidden);
      for (size_t i = 0; i < hidden.size(); ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "hidden_%02zu.f32", i);
        WriteFloats(std::filesystem::path(args.dump) / name, hidden[i]);
      }
      WriteFloats(std::filesystem::path(args.dump) / "logits.f32", logits);
    }

    if (args.generate > 0) {
      const auto g0 = std::chrono::steady_clock::now();
      const std::vector<int> out = engine.GenerateGreedy(args.tokens, args.generate);
      const auto g1 = std::chrono::steady_clock::now();
      for (size_t i = 0; i < out.size(); ++i) std::printf(i == 0 ? "%d" : " %d", out[i]);
      std::printf("\n");
      std::fprintf(stderr, "generated %d tokens in %.3f s\n", args.generate, Seconds(g0, g1));
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
