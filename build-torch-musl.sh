#!/bin/sh

set -eu

TORCH_VERSION="${TORCH_VERSION:-v2.9.1}"
PYTHON="${PYTHON:-python3}"
MAX_JOBS="${MAX_JOBS:-2}"
MARCH="${MARCH:-native}"
SRC_DIR="${SRC_DIR:-$(pwd)/.torch-src}"
OUT_DIR="$(pwd)/wheels"

mkdir -p "$OUT_DIR"

if [ ! -d "$SRC_DIR/.git" ]; then
    git clone --depth 1 --branch "$TORCH_VERSION" --recursive \
        https://github.com/pytorch/pytorch "$SRC_DIR"
fi
cd "$SRC_DIR"

mh=torch/headeronly/macros/Macros.h
sed 's/(defined(__EMSCRIPTEN__))/(defined(__EMSCRIPTEN__) || !defined(__GLIBC__))/' \
    "$mh" > "$mh.tmp" && mv "$mh.tmp" "$mh"

BUILD_VENV="${BUILD_VENV:-$SRC_DIR/.buildvenv}"
if [ ! -x "$BUILD_VENV/bin/python" ]; then
    "$PYTHON" -m venv "$BUILD_VENV"
fi
VPY="$BUILD_VENV/bin/python"

"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r requirements.txt

export CC=clang CXX=clang++
export USE_CUDA=0 USE_ROCM=0
export USE_FBGEMM=0
export USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0
export USE_MKLDNN=0
export BLAS=OpenBLAS
export BUILD_TEST=0
export USE_DISTRIBUTED=0
export USE_KINETO=0
export MAX_JOBS
export CMAKE_C_FLAGS="-march=$MARCH -mtune=generic"
export CMAKE_CXX_FLAGS="-march=$MARCH -mtune=generic"
#export USE_CPP_STACKTRACES=0

if command -v ccache >/dev/null 2>&1; then
    ccache -M 25Gi >/dev/null 2>&1 || true
fi

"$VPY" setup.py bdist_wheel

cp dist/torch-*.whl "$OUT_DIR"/
echo
echo "==> Built: $(ls "$OUT_DIR"/torch-*.whl)"
