// Error checking and owning device memory.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace infer {

class CudaError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline void CudaCheck(cudaError_t err, const char* what, const char* file, int line) {
  if (err != cudaSuccess) {
    throw CudaError(std::string(file) + ":" + std::to_string(line) + ": " + what + ": " +
                    cudaGetErrorString(err));
  }
}

#define CUDA_CHECK(expr) ::infer::CudaCheck((expr), #expr, __FILE__, __LINE__)

// Checks that the last kernel launch was accepted. Launch errors (bad grid size, too much
// shared memory) are reported here; errors inside the kernel show up at the next sync.
#define CUDA_CHECK_LAUNCH() \
  ::infer::CudaCheck(cudaGetLastError(), "kernel launch", __FILE__, __LINE__)

// An owning, typed device allocation.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count > 0) CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
  }
  ~DeviceBuffer() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept
      : ptr_(std::exchange(o.ptr_, nullptr)), count_(std::exchange(o.count_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      if (ptr_ != nullptr) cudaFree(ptr_);
      ptr_ = std::exchange(o.ptr_, nullptr);
      count_ = std::exchange(o.count_, 0);
    }
    return *this;
  }

  T* get() const { return ptr_; }
  size_t size() const { return count_; }
  size_t bytes() const { return count_ * sizeof(T); }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

}  // namespace infer
