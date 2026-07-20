#!/bin/sh

set -eu

TORCH_VERSION="${TORCH_VERSION:-v2.9.1}"
PYTHON="${PYTHON:-python3}"
MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 2)}"
MARCH="${MARCH:-native}"
SRC_DIR="${SRC_DIR:-$(pwd)/.torch-src}"
OUT_DIR="$(pwd)/wheels"

echo "==> torch $TORCH_VERSION | python=$PYTHON | jobs=$MAX_JOBS | march=$MARCH"

mkdir -p "$OUT_DIR"

if [ ! -d "$SRC_DIR/.git" ]; then
    git clone --depth 1 --branch "$TORCH_VERSION" --recursive \
        https://github.com/pytorch/pytorch "$SRC_DIR"
fi
cd "$SRC_DIR"

current="$(git describe --tags --exact-match 2>/dev/null || echo '')"
if [ "$current" = "$TORCH_VERSION" ]; then
    echo "==> Incremental build: $SRC_DIR was found."
else
    git fetch --depth 1 origin "refs/tags/$TORCH_VERSION:refs/tags/$TORCH_VERSION"
    git reset --hard "$TORCH_VERSION"
    git submodule sync --recursive
    git submodule update --init --recursive --depth 1
fi

mh="torch/headeronly/macros/Macros.h"
if [ ! -f "$mh" ]; then
    echo "ERROR: $mh not found. Check $TORCH_VERSION." >&2
    exit 1
fi
if grep -q '!defined(__GLIBC__)' "$mh"; then
    echo "==> $mh already patched."
elif grep -qF '(defined(__EMSCRIPTEN__))' "$mh"; then
    sed 's/(defined(__EMSCRIPTEN__))/(defined(__EMSCRIPTEN__) || !defined(__GLIBC__))/' \
        "$mh" > "$mh.tmp" && mv "$mh.tmp" "$mh"
    grep -q '!defined(__GLIBC__)' "$mh" || {
        echo "ERROR: our patch failed to apply to $mh." >&2
        exit 1
    }
    echo "==> Patched $mh successfully."
else
    echo "ERROR: anchor '(defined(__EMSCRIPTEN__))' not found in $mh." >&2
    echo "       Update the patch in $0." >&2
    exit 1
fi

BUILD_VENV="${BUILD_VENV:-$SRC_DIR/.buildvenv}"
if [ -x "$BUILD_VENV/bin/python" ] && [ -f "$BUILD_VENV/bin/pip" ]; then
    interp=$(sed -n '1s/^#!\([^ ]*\).*/\1/p' "$BUILD_VENV/bin/pip")
    if [ -n "$interp" ] && [ ! -x "$interp" ]; then
        echo "==> Recreating the venv due to stale shebangs."
        rm -rf "$BUILD_VENV"
    fi
fi
if [ ! -x "$BUILD_VENV/bin/python" ]; then
    "$PYTHON" -m venv "$BUILD_VENV"
fi
VPY="$BUILD_VENV/bin/python"

"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r requirements.txt

export PATH="$BUILD_VENV/bin:$PATH"
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
    ccache -M "${CCACHE_MAXSIZE:-25Gi}" >/dev/null 2>&1 || true
    export CCACHE_BASEDIR="$PWD"
    export CCACHE_NOHASHDIR=1
    : "${CCACHE_COMPILERCHECK:=content}"
    export CCACHE_COMPILERCHECK
fi

"$VPY" setup.py bdist_wheel

cp dist/torch-*.whl "$OUT_DIR"/
echo
echo "==> Built $(ls "$OUT_DIR"/torch-*.whl)."
