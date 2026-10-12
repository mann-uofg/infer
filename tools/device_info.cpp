// Prints the properties of the CUDA device the engine runs on. This replaces the deviceQuery
// sample, which no longer ships with the toolkit.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

double MiB(size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

}  // namespace

int main() {
  int device = 0;
  Check(cudaGetDevice(&device), "cudaGetDevice");

  cudaDeviceProp prop{};
  Check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

  int runtime_version = 0;
  int driver_version = 0;
  Check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  Check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  Check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

  std::printf("device                     %s\n", prop.name);
  std::printf("compute capability         %d.%d\n", prop.major, prop.minor);
  std::printf("CUDA runtime / driver      %d.%d / %d.%d\n", runtime_version / 1000,
              runtime_version % 1000 / 10, driver_version / 1000, driver_version % 1000 / 10);
  std::printf("memory total / free        %.0f / %.0f MiB\n", MiB(total_bytes), MiB(free_bytes));
  std::printf("memory bus width           %d bits\n", prop.memoryBusWidth);
  std::printf("L2 cache                   %.0f MiB\n", MiB(static_cast<size_t>(prop.l2CacheSize)));
  std::printf("multiprocessors (SMs)      %d\n", prop.multiProcessorCount);
  std::printf("warp size                  %d\n", prop.warpSize);
  std::printf("max threads per block      %d\n", prop.maxThreadsPerBlock);
  std::printf("max threads per SM         %d\n", prop.maxThreadsPerMultiProcessor);
  std::printf("registers per block / SM   %d / %d\n", prop.regsPerBlock,
              prop.regsPerMultiprocessor);
  std::printf("shared memory per block    %zu KiB\n", prop.sharedMemPerBlock / 1024);
  std::printf("shared memory per SM       %zu KiB\n", prop.sharedMemPerMultiprocessor / 1024);
  return 0;
}
