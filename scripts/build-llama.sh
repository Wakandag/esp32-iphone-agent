#!/bin/bash
# Builds llama.cpp for iOS (arm64, CPU only) as static libs and stages the headers
# under ios/llama/{libs,headers}, where the Xcode project looks for them.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

LLAMA_TAG=v1.0.0

if [ ! -d llama.cpp ]; then
  git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp llama.cpp
fi

rm -rf build/llama
cmake -S llama.cpp -B build/llama -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=16.4 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_METAL=OFF \
  -DGGML_ACCELERATE=OFF \
  -DGGML_BLAS=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=OFF

cmake --build build/llama --config Release --target llama -j "$(sysctl -n hw.ncpu)"

mkdir -p ios/llama/libs ios/llama/headers
find build/llama -name "*.a" -exec cp {} ios/llama/libs/ \;
cp llama.cpp/include/*.h ios/llama/headers/
cp llama.cpp/ggml/include/*.h ios/llama/headers/
cat > ios/llama/headers/module.modulemap <<'EOF'
module llama {
    header "llama.h"
    export *
}
EOF
echo "--- staged:"
ls -la ios/llama/libs/
