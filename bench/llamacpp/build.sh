#!/usr/bin/env bash
# Builds llama.cpp with the CUDA backend for the comparison benchmarks.
#
#   bench/llamacpp/build.sh [DIR]      (default ~/src/llama.cpp)
#
# Pinned to one release so the comparison can be reproduced; the version goes into every
# results file.
set -euo pipefail

TAG="${LLAMACPP_TAG:-v0.6.0}"
# The commit that tag pointed to when the results were taken. Tags can be moved; the build
# refuses to run if the tag now names a different commit (override both to build another).
COMMIT="${LLAMACPP_COMMIT:-d81235049384534c167caea52b85a694f6103d14}"
DIR="${1:-$HOME/src/llama.cpp}"

if [ ! -d "$DIR/.git" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp.git "$DIR"
fi
git -C "$DIR" fetch --depth 1 origin tag "$TAG"
git -C "$DIR" checkout -q "$TAG"
actual=$(git -C "$DIR" rev-parse HEAD)
if [ "$actual" != "$COMMIT" ]; then
  echo "llama.cpp $TAG is $actual, expected $COMMIT; refusing to build." >&2
  exit 1
fi

# Only this GPU's architecture, to keep the build short.
cmake -S "$DIR" -B "$DIR/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build "$DIR/build" --target llama-bench llama-perplexity llama-quantize

echo "llama.cpp $TAG ($(git -C "$DIR" rev-parse --short HEAD)) built in $DIR/build/bin"
