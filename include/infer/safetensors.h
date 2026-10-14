// Reads safetensors files by memory-mapping them.
//
// Layout of a .safetensors file:
//   8 bytes      little-endian u64 N, the size of the header
//   N bytes      JSON header: {"name": {"dtype": "BF16", "shape": [..],
//                              "data_offsets": [begin, end]}, ..., "__metadata__": {...}}
//   rest         raw tensor bytes; offsets in the header are relative to the start of this
//
// Mapping the file means no copy is made on the host: tensor data is read straight from the
// page cache when it is uploaded to the GPU.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace infer {

class SafetensorsError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

enum class DType { kBF16, kF16, kF32, kI8, kU8, kI32 };

size_t DTypeSize(DType t);
const char* DTypeName(DType t);

struct TensorView {
  DType dtype;
  std::vector<int64_t> shape;
  const std::byte* data;  // points into the mapping; valid while the file is open
  size_t nbytes;

  int64_t numel() const;
};

class SafetensorsFile {
 public:
  explicit SafetensorsFile(const std::string& path);
  ~SafetensorsFile();
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;
  SafetensorsFile(SafetensorsFile&& other) noexcept;
  SafetensorsFile& operator=(SafetensorsFile&&) = delete;

  const std::map<std::string, TensorView>& tensors() const { return tensors_; }
  bool Has(const std::string& name) const { return tensors_.count(name) != 0; }
  // Throws if missing.
  const TensorView& Get(const std::string& name) const;

 private:
  std::string path_;
  void* map_ = nullptr;
  size_t map_size_ = 0;
  std::map<std::string, TensorView> tensors_;
};

// A model directory holds either model.safetensors or shards listed in
// model.safetensors.index.json. This opens all of them and looks tensors up across shards.
class SafetensorsModel {
 public:
  explicit SafetensorsModel(const std::string& dir);

  bool Has(const std::string& name) const;
  const TensorView& Get(const std::string& name) const;
  std::vector<std::string> Names() const;

 private:
  std::vector<SafetensorsFile> files_;
};

}  // namespace infer
