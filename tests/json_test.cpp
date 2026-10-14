#include "infer/json.h"

#include <gtest/gtest.h>

namespace infer {
namespace {

TEST(JsonTest, ParsesScalars) {
  EXPECT_TRUE(Json::Parse("null").is_null());
  EXPECT_TRUE(Json::Parse("true").AsBool());
  EXPECT_FALSE(Json::Parse(" false ").AsBool());
  EXPECT_EQ(Json::Parse("42").AsInt(), 42);
  EXPECT_EQ(Json::Parse("-7").AsInt(), -7);
  EXPECT_DOUBLE_EQ(Json::Parse("1e-05").AsNumber(), 1e-5);
  EXPECT_DOUBLE_EQ(Json::Parse("500000.0").AsNumber(), 500000.0);
  EXPECT_EQ(Json::Parse("\"hi\"").AsString(), "hi");
}

TEST(JsonTest, ParsesNestedStructures) {
  const Json j = Json::Parse(R"({"a": [1, 2, {"b": null}], "c": {"d": "e"}, "f": []})");
  ASSERT_TRUE(j.is_object());
  EXPECT_EQ(j["a"].AsArray().size(), 3u);
  EXPECT_EQ(j["a"].AsArray()[1].AsInt(), 2);
  EXPECT_TRUE(j["a"].AsArray()[2]["b"].is_null());
  EXPECT_EQ(j["c"]["d"].AsString(), "e");
  EXPECT_TRUE(j["f"].AsArray().empty());
  EXPECT_TRUE(j.Has("c"));
  EXPECT_FALSE(j.Has("zzz"));
}

TEST(JsonTest, DecodesEscapes) {
  EXPECT_EQ(Json::Parse(R"("a\"b\\c\/d\n\t")").AsString(), "a\"b\\c/d\n\t");
  EXPECT_EQ(Json::Parse(R"("\u00e9")").AsString(), "\xC3\xA9");
  // U+1F600 as a surrogate pair.
  EXPECT_EQ(Json::Parse(R"("\ud83d\ude00")").AsString(), "\xF0\x9F\x98\x80");
}

TEST(JsonTest, LargeIntegersAreExact) {
  // Tensor offsets in a multi-gigabyte file.
  EXPECT_EQ(Json::Parse("2471645608").AsInt(), 2471645608LL);
  EXPECT_EQ(Json::Parse("9007199254740992").AsInt(), 9007199254740992LL);  // 2^53
}

TEST(JsonTest, RejectsMalformedInput) {
  for (const char* bad :
       {"", "{", "[1,]", "{\"a\" 1}", "{\"a\":1,}", "01", "1.", ".5", "+1", "nul", "\"abc",
        "\"\\x\"", "[1] 2", "{\"a\":1,\"a\":2}", "\"\\ud800\"", "inf", "NaN", "\"\x01\""}) {
    EXPECT_THROW(Json::Parse(bad), JsonError) << bad;
  }
}

TEST(JsonTest, RejectsDeepNesting) {
  std::string deep(1000, '[');
  deep += std::string(1000, ']');
  EXPECT_THROW(Json::Parse(deep), JsonError);
}

TEST(JsonTest, TypeMismatchesThrow) {
  const Json j = Json::Parse(R"({"n": 1.5, "s": "x"})");
  EXPECT_THROW(j["n"].AsInt(), JsonError);
  EXPECT_THROW(j["s"].AsNumber(), JsonError);
  EXPECT_THROW(j["missing"], JsonError);
  EXPECT_THROW(Json::Parse("1e30").AsInt(), JsonError);
}

}  // namespace
}  // namespace infer
