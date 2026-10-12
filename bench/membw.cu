// Measures the peak memory bandwidth of the GPU, in the style of the STREAM benchmark.
//
// Decode is bandwidth-bound: every generated token reads all of the weights once. The numbers
// printed here are the ceiling that the engine's achieved GB/s is reported against, so they
// have to be measured on the actual card instead of taken from a spec sheet.
//
// Three kernels are timed over buffers much larger than the L2 cache:
//   read   sum the buffer            bytes moved = N
//   write  fill the buffer           bytes moved = N
//   copy   dst[i] = src[i]           bytes moved = 2N
// cudaMemset and a device-to-device cudaMemcpy are timed as a cross-check from the runtime.
//
// GB means 1e9 bytes, the convention used for memory bandwidth figures.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kWarmupRuns = 5;

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

// All three kernels use a grid-stride loop over 16-byte elements. On each iteration the 32
// threads of a warp touch 32 consecutive elements, so a warp's accesses coalesce into one
// contiguous 512-byte span.

__global__ void ReadKernel(const uint4* __restrict__ src, size_t n, uint32_t* __restrict__ out) {
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  uint32_t acc = 0;
  for (size_t i = tid; i < n; i += stride) {
    const uint4 v = src[i];
    acc += v.x ^ v.y ^ v.z ^ v.w;
  }
  // One small write per thread keeps the loads from being optimized away.
  out[tid] = acc;
}

__global__ void WriteKernel(uint4* __restrict__ dst, size_t n, uint32_t value) {
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const uint4 v = make_uint4(value, value, value, value);
  for (size_t i = tid; i < n; i += stride) {
    dst[i] = v;
  }
}

__global__ void CopyKernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n) {
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = tid; i < n; i += stride) {
    dst[i] = src[i];
  }
}

struct Stats {
  double median;
  double min;
  double max;
};

// Runs `op` (which must only enqueue GPU work) and returns bandwidth statistics in GB/s.
// Timing uses CUDA events so that it is taken on the GPU's own timeline.
template <typename Op>
Stats Measure(Op op, double bytes_moved, int runs) {
  cudaEvent_t start;
  cudaEvent_t stop;
  Check(cudaEventCreate(&start), "cudaEventCreate");
  Check(cudaEventCreate(&stop), "cudaEventCreate");

  for (int i = 0; i < kWarmupRuns; ++i) op();
  Check(cudaDeviceSynchronize(), "warmup");

  std::vector<double> gbps;
  gbps.reserve(static_cast<size_t>(runs));
  for (int i = 0; i < runs; ++i) {
    Check(cudaEventRecord(start), "cudaEventRecord");
    op();
    Check(cudaEventRecord(stop), "cudaEventRecord");
    Check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float ms = 0.0f;
    Check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    gbps.push_back(bytes_moved / (static_cast<double>(ms) * 1e-3) / 1e9);
  }
  Check(cudaGetLastError(), "kernel launch");

  Check(cudaEventDestroy(start), "cudaEventDestroy");
  Check(cudaEventDestroy(stop), "cudaEventDestroy");

  std::sort(gbps.begin(), gbps.end());
  const size_t mid = gbps.size() / 2;
  const double median = gbps.size() % 2 == 1 ? gbps[mid] : 0.5 * (gbps[mid - 1] + gbps[mid]);
  return {median, gbps.front(), gbps.back()};
}

void PrintRow(const char* name, int threads, const Stats& s) {
  if (threads > 0) {
    std::printf("%-8s %10d %10.1f %10.1f %10.1f\n", name, threads, s.median, s.min, s.max);
  } else {
    std::printf("%-8s %10s %10.1f %10.1f %10.1f\n", name, "-", s.median, s.min, s.max);
  }
}

}  // namespace

int main(int argc, char** argv) {
  size_t buffer_mib = 2048;
  int runs = 20;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mib") == 0 && i + 1 < argc) {
      buffer_mib = std::strtoull(argv[++i], nullptr, 10);
    } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
      runs = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr, "usage: %s [--mib N] [--runs N]\n", argv[0]);
      return 1;
    }
  }
  if (buffer_mib == 0 || runs < 1) {
    std::fprintf(stderr, "--mib and --runs must be positive\n");
    return 1;
  }

  cudaDeviceProp prop{};
  Check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

  const size_t bytes = buffer_mib * 1024 * 1024;
  const size_t n = bytes / sizeof(uint4);

  // The largest grid swept below decides how big the read kernel's output buffer must be.
  const std::vector<int> blocks_per_sm = {1, 2, 4, 8, 16, 32, 64};
  const size_t max_threads =
      static_cast<size_t>(blocks_per_sm.back()) * prop.multiProcessorCount * kBlockSize;

  uint4* src = nullptr;
  uint4* dst = nullptr;
  uint32_t* out = nullptr;
  Check(cudaMalloc(&src, bytes), "cudaMalloc src");
  Check(cudaMalloc(&dst, bytes), "cudaMalloc dst");
  Check(cudaMalloc(&out, max_threads * sizeof(uint32_t)), "cudaMalloc out");
  Check(cudaMemset(src, 1, bytes), "cudaMemset");
  Check(cudaMemset(dst, 0, bytes), "cudaMemset");

  std::printf("device        %s (%d SMs, L2 %d MiB, bus %d bits)\n", prop.name,
              prop.multiProcessorCount, prop.l2CacheSize >> 20, prop.memoryBusWidth);
  std::printf("buffer        %zu MiB, %d timed runs after %d warmups\n\n", buffer_mib, runs,
              kWarmupRuns);
  std::printf("%-8s %10s %10s %10s %10s\n", "test", "threads", "median", "min", "max");
  std::printf("%-8s %10s %10s %10s %10s\n", "", "", "GB/s", "GB/s", "GB/s");

  const double b = static_cast<double>(bytes);
  double best_read = 0.0;
  double best_write = 0.0;
  double best_copy = 0.0;

  // Bandwidth depends on how many threads are issuing loads at once, so sweep the grid size
  // and take the best median for each test.
  for (int per_sm : blocks_per_sm) {
    const int blocks = per_sm * prop.multiProcessorCount;
    const int threads = blocks * kBlockSize;

    const Stats r = Measure([&] { ReadKernel<<<blocks, kBlockSize>>>(src, n, out); }, b, runs);
    const Stats w = Measure([&] { WriteKernel<<<blocks, kBlockSize>>>(dst, n, 7u); }, b, runs);
    const Stats c = Measure([&] { CopyKernel<<<blocks, kBlockSize>>>(src, dst, n); }, 2 * b, runs);
    PrintRow("read", threads, r);
    PrintRow("write", threads, w);
    PrintRow("copy", threads, c);
    best_read = std::max(best_read, r.median);
    best_write = std::max(best_write, w.median);
    best_copy = std::max(best_copy, c.median);
  }

  const Stats ms = Measure([&] { Check(cudaMemsetAsync(dst, 0, bytes), "memset"); }, b, runs);
  const Stats mc =
      Measure([&] { Check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice), "memcpy"); },
              2 * b, runs);
  PrintRow("memset", 0, ms);
  PrintRow("memcpy", 0, mc);

  std::printf("\nbest median   read %.1f GB/s, write %.1f GB/s, copy %.1f GB/s\n", best_read,
              best_write, best_copy);

  Check(cudaFree(src), "cudaFree");
  Check(cudaFree(dst), "cudaFree");
  Check(cudaFree(out), "cudaFree");
  return 0;
}
