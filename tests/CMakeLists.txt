include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG b514bdc898e2951020cbdca1304b75f5950d1f59  # v1.15.2
  SYSTEM
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(host_tests
  config_test.cpp
  json_test.cpp
  safetensors_test.cpp
)
target_link_libraries(host_tests PRIVATE infer_host infer_options GTest::gtest_main)
gtest_discover_tests(host_tests)
