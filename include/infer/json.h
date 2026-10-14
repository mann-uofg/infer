// A small JSON parser, enough for safetensors headers and Hugging Face config files.
//
// Numbers are kept as double. That is exact for every integer up to 2^53, which covers tensor
// offsets in any file this engine will ever load.

#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace infer {

class JsonError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class Json {
 public:
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  Json() = default;
  static Json Parse(std::string_view text);

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::kNull; }
  bool is_number() const { return type_ == Type::kNumber; }
  bool is_string() const { return type_ == Type::kString; }
  bool is_array() const { return type_ == Type::kArray; }
  bool is_object() const { return type_ == Type::kObject; }

  bool AsBool() const;
  double AsNumber() const;
  // Throws unless the number is an integer that fits in int64.
  int64_t AsInt() const;
  const std::string& AsString() const;
  const std::vector<Json>& AsArray() const;
  const std::map<std::string, Json>& AsObject() const;

  bool Has(const std::string& key) const;
  // Object member access. Throws if this is not an object or the key is missing.
  const Json& operator[](const std::string& key) const;

 private:
  friend class JsonParser;

  Type type_ = Type::kNull;
  bool bool_ = false;
  double number_ = 0.0;
  std::string string_;
  std::vector<Json> array_;
  std::map<std::string, Json> object_;
};

}  // namespace infer
