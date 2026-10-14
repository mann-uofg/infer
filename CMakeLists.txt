cmake_minimum_required(VERSION 3.25)
project(infer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(INFER_ENABLE_CUDA "Build the CUDA kernels and GPU tests" ON)
option(INFER_BUILD_TESTS "Build tests" ON)

# CI runners have no CUDA toolkit, so the GPU side is skipped there and only the CPU-side
# targets are built.
if(INFER_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Build for the GPU in this machine unless an architecture is given explicitly.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  else()
    message(STATUS "No CUDA compiler found, building CPU-side targets only")
    set(INFER_ENABLE_CUDA OFF)
  endif()
endif()

# Warning flags for our own targets only (not third-party code).
add_library(infer_options INTERFACE)
target_compile_options(infer_options INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings -Xcompiler=-Wall,-Wextra,-Werror>)

# Host-only code: config, JSON and safetensors parsing. No CUDA needed, so CI tests it.
add_library(infer_host
  src/config.cpp
  src/json.cpp
  src/safetensors.cpp
)
target_include_directories(infer_host PUBLIC include)
target_link_libraries(infer_host PRIVATE infer_options)

if(INFER_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_library(infer_cuda
    src/engine.cpp
    src/kernels/decode.cu
    src/kernels/gemv.cu
    src/kernels/naive.cu
    src/kernels/prefill.cu
    src/kernels/quant.cu
    src/kernels/sampling.cu
    src/model.cpp
  )
  target_link_libraries(infer_cuda PUBLIC infer_host CUDA::cudart PRIVATE CUDA::nvtx3 infer_options)

  add_executable(infer src/main.cpp)
  target_link_libraries(infer PRIVATE infer_cuda infer_options)

  add_executable(device_info tools/device_info.cpp)
  target_link_libraries(device_info PRIVATE infer_options CUDA::cudart)

  add_executable(membw bench/membw.cu)

  add_executable(bench_engine bench/bench_engine.cpp)
  target_link_libraries(bench_engine PRIVATE infer_cuda infer_options)

  add_executable(bench_kernels bench/bench_kernels.cu)
  target_link_libraries(bench_kernels PRIVATE infer_cuda CUDA::cublas)
endif()

if(INFER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
