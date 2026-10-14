#include "infer/json.h"

#include <cmath>
#include <cstdlib>
#include <string>

namespace infer {

class JsonParser {
 public:
  explicit JsonParser(std::string_view text) : s_(text) {}

  Json ParseDocument() {
    Json v = ParseValue(0);
    SkipSpace();
    if (pos_ != s_.size()) Fail("trailing characters");
    return v;
  }

 private:
  // Deep nesting would otherwise overflow the stack on hostile input.
  static constexpr int kMaxDepth = 64;

  [[noreturn]] void Fail(const std::string& what) const {
    throw JsonError("json: " + what + " at offset " + std::to_string(pos_));
  }

  void SkipSpace() {
    while (pos_ < s_.size() &&
           (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) {
      ++pos_;
    }
  }

  char Peek() {
    SkipSpace();
    if (pos_ >= s_.size()) Fail("unexpected end of input");
    return s_[pos_];
  }

  void Expect(std::string_view word) {
    if (s_.substr(pos_, word.size()) != word) Fail("expected " + std::string(word));
    pos_ += word.size();
  }

  Json ParseValue(int depth) {
    if (depth > kMaxDepth) Fail("nesting too deep");
    Json v;
    switch (Peek()) {
      case '{':
        v.type_ = Json::Type::kObject;
        ParseObject(v, depth);
        break;
      case '[':
        v.type_ = Json::Type::kArray;
        ParseArray(v, depth);
        break;
      case '"':
        v.type_ = Json::Type::kString;
        v.string_ = ParseString();
        break;
      case 't':
        Expect("true");
        v.type_ = Json::Type::kBool;
        v.bool_ = true;
        break;
      case 'f':
        Expect("false");
        v.type_ = Json::Type::kBool;
        break;
      case 'n':
        Expect("null");
        break;
      default:
        v.type_ = Json::Type::kNumber;
        v.number_ = ParseNumber();
        break;
    }
    return v;
  }

  void ParseObject(Json& v, int depth) {
    ++pos_;  // '{'
    if (Peek() == '}') {
      ++pos_;
      return;
    }
    while (true) {
      if (Peek() != '"') Fail("expected object key");
      std::string key = ParseString();
      if (Peek() != ':') Fail("expected ':'");
      ++pos_;
      if (!v.object_.emplace(std::move(key), ParseValue(depth + 1)).second) {
        Fail("duplicate key");
      }
      const char c = Peek();
      ++pos_;
      if (c == '}') return;
      if (c != ',') Fail("expected ',' or '}'");
    }
  }

  void ParseArray(Json& v, int depth) {
    ++pos_;  // '['
    if (Peek() == ']') {
      ++pos_;
      return;
    }
    while (true) {
      v.array_.push_back(ParseValue(depth + 1));
      const char c = Peek();
      ++pos_;
      if (c == ']') return;
      if (c != ',') Fail("expected ',' or ']'");
    }
  }

  static void AppendUtf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  uint32_t ParseHex4() {
    if (pos_ + 4 > s_.size()) Fail("truncated \\u escape");
    uint32_t cp = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = s_[pos_++];
      cp <<= 4;
      if (c >= '0' && c <= '9') {
        cp |= static_cast<uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        cp |= static_cast<uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        cp |= static_cast<uint32_t>(c - 'A' + 10);
      } else {
        Fail("bad hex digit");
      }
    }
    return cp;
  }

  std::string ParseString() {
    ++pos_;  // opening quote
    std::string out;
    while (true) {
      if (pos_ >= s_.size()) Fail("unterminated string");
      const char c = s_[pos_++];
      if (c == '"') return out;
      if (static_cast<unsigned char>(c) < 0x20) Fail("control character in string");
      if (c != '\\') {
        out += c;
        continue;
      }
      if (pos_ >= s_.size()) Fail("unterminated escape");
      const char e = s_[pos_++];
      switch (e) {
        case '"':
        case '\\':
        case '/':
          out += e;
          break;
        case 'b':
          out += '\b';
          break;
        case 'f':
          out += '\f';
          break;
        case 'n':
          out += '\n';
          break;
        case 'r':
          out += '\r';
          break;
        case 't':
          out += '\t';
          break;
        case 'u': {
          uint32_t cp = ParseHex4();
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            // A high surrogate must be followed by an escaped low surrogate.
            Expect("\\u");
            const uint32_t lo = ParseHex4();
            if (lo < 0xDC00 || lo > 0xDFFF) Fail("bad surrogate pair");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            Fail("lone low surrogate");
          }
          AppendUtf8(out, cp);
          break;
        }
        default:
          Fail("bad escape");
      }
    }
  }

  double ParseNumber() {
    // Validate against the JSON grammar first; strtod alone accepts things like "inf" or hex.
    const size_t start = pos_;
    if (pos_ < s_.size() && s_[pos_] == '-') ++pos_;
    auto digits = [&] {
      const size_t d = pos_;
      while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
      return pos_ - d;
    };
    if (pos_ < s_.size() && s_[pos_] == '0') {
      ++pos_;
    } else if (digits() == 0) {
      Fail("bad number");
    }
    if (pos_ < s_.size() && s_[pos_] == '.') {
      ++pos_;
      if (digits() == 0) Fail("bad fraction");
    }
    if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) ++pos_;
      if (digits() == 0) Fail("bad exponent");
    }
    const std::string text(s_.substr(start, pos_ - start));
    return std::strtod(text.c_str(), nullptr);
  }

  std::string_view s_;
  size_t pos_ = 0;
};

Json Json::Parse(std::string_view text) { return JsonParser(text).ParseDocument(); }

bool Json::AsBool() const {
  if (type_ != Type::kBool) throw JsonError("json: not a bool");
  return bool_;
}

double Json::AsNumber() const {
  if (type_ != Type::kNumber) throw JsonError("json: not a number");
  return number_;
}

int64_t Json::AsInt() const {
  const double d = AsNumber();
  // 2^63 is exactly representable; anything at or above it does not fit.
  if (d != std::floor(d) || d < -9223372036854775808.0 || d >= 9223372036854775808.0) {
    throw JsonError("json: not an int64");
  }
  return static_cast<int64_t>(d);
}

const std::string& Json::AsString() const {
  if (type_ != Type::kString) throw JsonError("json: not a string");
  return string_;
}

const std::vector<Json>& Json::AsArray() const {
  if (type_ != Type::kArray) throw JsonError("json: not an array");
  return array_;
}

const std::map<std::string, Json>& Json::AsObject() const {
  if (type_ != Type::kObject) throw JsonError("json: not an object");
  return object_;
}

bool Json::Has(const std::string& key) const {
  return type_ == Type::kObject && object_.count(key) != 0;
}

const Json& Json::operator[](const std::string& key) const {
  const auto& obj = AsObject();
  const auto it = obj.find(key);
  if (it == obj.end()) throw JsonError("json: missing key '" + key + "'");
  return it->second;
}

}  // namespace infer
