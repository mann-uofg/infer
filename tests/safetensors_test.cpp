#include "infer/safetensors.h"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace infer {
namespace {

namespace fs = std::filesystem;

class SafetensorsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("infer_st_" +
            std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::remove_all(dir_);
    fs::create_directories(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }

  // Writes [u64 header length][header][data].
  std::string Write(const std::string& name, const std::string& header,
                    const std::vector<uint8_t>& data, uint64_t header_len_override = 0) {
    const fs::path p = dir_ / name;
    std::ofstream out(p, std::ios::binary);
    uint64_t n = header_len_override != 0 ? header_len_override : header.size();
    for (int i = 0; i < 8; ++i) out.put(static_cast<char>((n >> (8 * i)) & 0xFF));
    out << header;
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
    return p.string();
  }

  fs::path dir_;
};

std::vector<uint8_t> Bytes(size_t n) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(i);
  return v;
}

TEST_F(SafetensorsTest, ReadsTensors) {
  const std::string header = R"({"__metadata__":{"format":"pt"},)"
                             R"("a":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},)"
                             R"("b":{"dtype":"F32","shape":[4],"data_offsets":[12,28]}})";
  const auto data = Bytes(28);
  SafetensorsFile f(Write("x.safetensors", header, data));

  ASSERT_EQ(f.tensors().size(), 2u);
  const TensorView& a = f.Get("a");
  EXPECT_EQ(a.dtype, DType::kBF16);
  EXPECT_EQ(a.shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(a.nbytes, 12u);
  EXPECT_EQ(a.numel(), 6);
  EXPECT_EQ(std::memcmp(a.data, data.data(), 12), 0);
  const TensorView& b = f.Get("b");
  EXPECT_EQ(b.dtype, DType::kF32);
  EXPECT_EQ(std::memcmp(b.data, data.data() + 12, 16), 0);
  EXPECT_FALSE(f.Has("__metadata__"));
  EXPECT_THROW(f.Get("c"), SafetensorsError);
}

TEST_F(SafetensorsTest, ScalarAndEmptyTensors) {
  const std::string header = R"({"s":{"dtype":"F32","shape":[],"data_offsets":[0,4]},)"
                             R"("e":{"dtype":"F32","shape":[0,5],"data_offsets":[4,4]}})";
  SafetensorsFile f(Write("x.safetensors", header, Bytes(4)));
  EXPECT_EQ(f.Get("s").numel(), 1);
  EXPECT_EQ(f.Get("e").numel(), 0);
}

TEST_F(SafetensorsTest, RejectsCorruptFiles) {
  struct Case {
    const char* what;
    std::string header;
    size_t data_bytes;
    uint64_t header_len;
  };
  const std::vector<Case> cases = {
      {"offsets past end", R"({"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})", 8, 0},
      {"size mismatch", R"({"a":{"dtype":"F32","shape":[3],"data_offsets":[0,16]}})", 16, 0},
      {"end before begin", R"({"a":{"dtype":"F32","shape":[0],"data_offsets":[8,4]}})", 16, 0},
      {"overlap",
       R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
       R"("b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})",
       16, 0},
      {"bad dtype", R"({"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}})", 8, 0},
      {"negative dim", R"({"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}})", 4, 0},
      {"not json", "{nope", 0, 0},
      {"header length too big", R"({})", 0, 1000},
  };
  for (const Case& c : cases) {
    const std::string p = Write("bad.safetensors", c.header, Bytes(c.data_bytes), c.header_len);
    EXPECT_THROW(SafetensorsFile{p}, SafetensorsError) << c.what;
  }
}

TEST_F(SafetensorsTest, RejectsTinyAndMissingFiles) {
  std::ofstream(dir_ / "tiny.safetensors") << "abc";
  EXPECT_THROW(SafetensorsFile{(dir_ / "tiny.safetensors").string()}, SafetensorsError);
  EXPECT_THROW(SafetensorsFile{(dir_ / "missing.safetensors").string()}, SafetensorsError);
}

TEST_F(SafetensorsTest, ModelReadsSingleFile) {
  Write("model.safetensors", R"({"w":{"dtype":"I8","shape":[3],"data_offsets":[0,3]}})", Bytes(3));
  SafetensorsModel m(dir_.string());
  EXPECT_TRUE(m.Has("w"));
  EXPECT_EQ(m.Get("w").dtype, DType::kI8);
}

TEST_F(SafetensorsTest, ModelReadsShards) {
  Write("model-00001-of-00002.safetensors",
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", Bytes(4));
  Write("model-00002-of-00002.safetensors",
        R"({"b":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})", Bytes(8));
  std::ofstream(dir_ / "model.safetensors.index.json")
      << R"({"metadata":{},"weight_map":{"a":"model-00001-of-00002.safetensors",)"
         R"("b":"model-00002-of-00002.safetensors"}})";
  SafetensorsModel m(dir_.string());
  EXPECT_EQ(m.Get("a").numel(), 1);
  EXPECT_EQ(m.Get("b").numel(), 2);
  EXPECT_EQ(m.Names().size(), 2u);
  EXPECT_THROW(m.Get("c"), SafetensorsError);
}

TEST_F(SafetensorsTest, ModelRejectsShardOutsideDirectory) {
  std::ofstream(dir_ / "model.safetensors.index.json")
      << R"({"weight_map":{"a":"../evil.safetensors"}})";
  EXPECT_THROW(SafetensorsModel{dir_.string()}, SafetensorsError);
}

}  // namespace
}  // namespace infer
