#include "infer/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

#include "infer/json.h"

namespace infer {

namespace {

// A header larger than this is almost certainly a corrupt length field. The reference
// implementation uses the same limit.
constexpr uint64_t kMaxHeaderSize = 100 * 1024 * 1024;

DType ParseDType(const std::string& s) {
  if (s == "BF16") return DType::kBF16;
  if (s == "F16") return DType::kF16;
  if (s == "F32") return DType::kF32;
  if (s == "I8") return DType::kI8;
  if (s == "U8") return DType::kU8;
  if (s == "I32") return DType::kI32;
  throw SafetensorsError("unsupported dtype " + s);
}

}  // namespace

size_t DTypeSize(DType t) {
  switch (t) {
    case DType::kBF16:
    case DType::kF16:
      return 2;
    case DType::kF32:
    case DType::kI32:
      return 4;
    case DType::kI8:
    case DType::kU8:
      return 1;
  }
  return 0;
}

const char* DTypeName(DType t) {
  switch (t) {
    case DType::kBF16:
      return "BF16";
    case DType::kF16:
      return "F16";
    case DType::kF32:
      return "F32";
    case DType::kI8:
      return "I8";
    case DType::kU8:
      return "U8";
    case DType::kI32:
      return "I32";
  }
  return "?";
}

int64_t TensorView::numel() const {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

SafetensorsFile::SafetensorsFile(const std::string& path) : path_(path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw SafetensorsError(path + ": " + std::strerror(errno));
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    const int err = errno;
    ::close(fd);
    throw SafetensorsError(path + ": " + std::strerror(err));
  }
  map_size_ = static_cast<size_t>(st.st_size);
  if (map_size_ < 8) {
    ::close(fd);
    throw SafetensorsError(path + ": file too small");
  }
  map_ = ::mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);  // the mapping keeps its own reference to the file
  if (map_ == MAP_FAILED) {
    map_ = nullptr;
    throw SafetensorsError(path + ": mmap failed: " + std::strerror(errno));
  }

  try {
    const auto* base = static_cast<const std::byte*>(map_);
    uint64_t header_size = 0;
    for (int i = 7; i >= 0; --i) {
      header_size = (header_size << 8) | static_cast<uint8_t>(base[i]);
    }
    if (header_size > kMaxHeaderSize || header_size > map_size_ - 8) {
      throw SafetensorsError("header size out of range");
    }
    const std::string_view header_text(reinterpret_cast<const char*>(base + 8),
                                       static_cast<size_t>(header_size));
    const Json header = Json::Parse(header_text);
    const std::byte* data = base + 8 + header_size;
    const uint64_t data_size = map_size_ - 8 - header_size;

    // Tensors must not overlap. Collect [begin, end) ranges and check them after.
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    for (const auto& [name, info] : header.AsObject()) {
      if (name == "__metadata__") continue;
      TensorView t{};
      t.dtype = ParseDType(info["dtype"].AsString());
      int64_t numel = 1;
      for (const Json& d : info["shape"].AsArray()) {
        const int64_t dim = d.AsInt();
        if (dim < 0) throw SafetensorsError(name + ": negative dimension");
        if (dim != 0 && numel > std::numeric_limits<int64_t>::max() / dim) {
          throw SafetensorsError(name + ": shape overflows");
        }
        numel *= dim;
        t.shape.push_back(dim);
      }
      const auto& offsets = info["data_offsets"].AsArray();
      if (offsets.size() != 2) throw SafetensorsError(name + ": bad data_offsets");
      const int64_t begin = offsets[0].AsInt();
      const int64_t end = offsets[1].AsInt();
      if (begin < 0 || end < begin || static_cast<uint64_t>(end) > data_size) {
        throw SafetensorsError(name + ": data_offsets out of range");
      }
      t.nbytes = static_cast<size_t>(end - begin);
      const size_t elem = DTypeSize(t.dtype);
      if (t.nbytes % elem != 0 || t.nbytes / elem != static_cast<size_t>(numel)) {
        throw SafetensorsError(name + ": size does not match shape and dtype");
      }
      t.data = data + begin;
      ranges.emplace_back(static_cast<uint64_t>(begin), static_cast<uint64_t>(end));
      tensors_.emplace(name, std::move(t));
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) {
      if (ranges[i].first < ranges[i - 1].second) throw SafetensorsError("tensors overlap");
    }
  } catch (const std::exception& e) {
    ::munmap(map_, map_size_);
    map_ = nullptr;
    throw SafetensorsError(path + ": " + e.what());
  }
}

SafetensorsFile::~SafetensorsFile() {
  if (map_ != nullptr) ::munmap(map_, map_size_);
}

SafetensorsFile::SafetensorsFile(SafetensorsFile&& other) noexcept
    : path_(std::move(other.path_)),
      map_(other.map_),
      map_size_(other.map_size_),
      tensors_(std::move(other.tensors_)) {
  other.map_ = nullptr;
  other.map_size_ = 0;
}

const TensorView& SafetensorsFile::Get(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) throw SafetensorsError(path_ + ": no tensor named " + name);
  return it->second;
}

SafetensorsModel::SafetensorsModel(const std::string& dir) {
  namespace fs = std::filesystem;
  const fs::path single = fs::path(dir) / "model.safetensors";
  const fs::path index = fs::path(dir) / "model.safetensors.index.json";
  if (fs::exists(single)) {
    files_.emplace_back(single.string());
    return;
  }
  if (!fs::exists(index)) {
    throw SafetensorsError(dir + ": no model.safetensors or model.safetensors.index.json");
  }
  std::ifstream in(index);
  std::stringstream ss;
  ss << in.rdbuf();
  std::set<std::string> shards;
  const Json index_json = Json::Parse(ss.str());
  for (const auto& [name, shard] : index_json["weight_map"].AsObject()) {
    shards.insert(shard.AsString());
  }
  for (const auto& shard : shards) {
    if (shard.find('/') != std::string::npos || shard.find("..") != std::string::npos) {
      throw SafetensorsError("shard name escapes the model directory: " + shard);
    }
    files_.emplace_back((fs::path(dir) / shard).string());
  }
}

bool SafetensorsModel::Has(const std::string& name) const {
  for (const auto& f : files_) {
    if (f.Has(name)) return true;
  }
  return false;
}

const TensorView& SafetensorsModel::Get(const std::string& name) const {
  for (const auto& f : files_) {
    if (f.Has(name)) return f.Get(name);
  }
  throw SafetensorsError("no tensor named " + name);
}

std::vector<std::string> SafetensorsModel::Names() const {
  std::vector<std::string> names;
  for (const auto& f : files_) {
    for (const auto& [name, t] : f.tensors()) names.push_back(name);
  }
  return names;
}

}  // namespace infer
