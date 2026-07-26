#!/bin/sh

set -eu

TORCH_VERSION="${TORCH_VERSION:-v2.12.1}"
PYTHON="${PYTHON:-python3}"
MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 2)}"
MARCH="${MARCH:-native}"
SRC_DIR="${SRC_DIR:-$(pwd)/.torch-src}"
OUT_DIR="$(pwd)/wheels"
BUILD_VENV="${BUILD_VENV:-$SRC_DIR/.buildvenv}"
STAMP="$SRC_DIR/.animus-built-version"

echo "==> torch $TORCH_VERSION | python=$PYTHON | jobs=$MAX_JOBS | march=$MARCH"

mkdir -p "$OUT_DIR"

fresh_clone=0
if [ ! -d "$SRC_DIR/.git" ]; then
    git clone --depth 1 --branch "$TORCH_VERSION" --recursive \
        https://github.com/pytorch/pytorch "$SRC_DIR"
    fresh_clone=1
fi
cd "$SRC_DIR"

built=""
if [ -f "$STAMP" ]; then
    built="$(cat "$STAMP")"
fi

current="$(git describe --tags --exact-match 2>/dev/null || echo '')"
if [ "$current" = "$TORCH_VERSION" ]; then
    echo "==> Incremental build: $SRC_DIR was found."
else
    git fetch --depth 1 origin "refs/tags/$TORCH_VERSION:refs/tags/$TORCH_VERSION"
    git reset --hard "$TORCH_VERSION"
    git submodule sync --recursive
    git submodule update --init --recursive --depth 1
fi

if [ "$fresh_clone" = 0 ] && [ "$built" != "$TORCH_VERSION" ]; then
    echo "==> Discarding build artifacts from ${built:-an unknown version}."
    case "$BUILD_VENV" in
        "$SRC_DIR"/*)
            git clean -xfdq -e "${BUILD_VENV#"$SRC_DIR"/}"
            ;;
        *)
            git clean -xfdq
            ;;
    esac
    git submodule foreach --quiet --recursive 'git clean -xfdq'
fi

printf '%s\n' "$TORCH_VERSION" > "$STAMP"

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

ensure_include() {
    file="$1"
    header="$2"
    anchor="$3"
    line="#include <$header>"

    if [ ! -f "$file" ]; then
        echo "ERROR: $file not found. Check $TORCH_VERSION." >&2
        exit 1
    fi
    if grep -qxF "$line" "$file"; then
        echo "==> $file already includes <$header>."
        return 0
    fi
    if ! grep -qxF "$anchor" "$file"; then
        echo "ERROR: anchor '$anchor' not found in $file." >&2
        echo "       Update the patch in $0." >&2
        exit 1
    fi
    awk -v line="$line" -v anchor="$anchor" '
        $0 == anchor && !inserted { print line; inserted = 1 }
        { print }
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    grep -qxF "$line" "$file" || {
        echo "ERROR: our patch failed to apply to $file." >&2
        exit 1
    }
    echo "==> Patched $file with <$header> successfully."
}

ensure_include caffe2/utils/string_utils.h cstdint '#include <memory>'

evf="cmake/EnvVarForwarding.cmake"
evf_marker="# Animus: re-read our build options straight from the environment."
if [ ! -f "$evf" ]; then
    echo "==> No $evf. This PyTorch forwards the environment from setup.py."
elif grep -qxF "$evf_marker" "$evf"; then
    echo "==> $evf already patched."
else
    cat >> "$evf" <<'PATCH'

# Animus: re-read our build options straight from the environment.
foreach(_animus_var
    BUILD_TEST
    CMAKE_C_FLAGS CMAKE_CXX_FLAGS
    USE_CUDA USE_DISTRIBUTED USE_FBGEMM USE_KINETO USE_MKLDNN
    USE_NNPACK USE_QNNPACK USE_ROCM USE_XNNPACK)
  if(DEFINED ENV{${_animus_var}} AND NOT DEFINED ${_animus_var})
    set(${_animus_var} "$ENV{${_animus_var}}"
        CACHE STRING "From env ${_animus_var}" FORCE)
  endif()
endforeach()
PATCH
    grep -qxF "$evf_marker" "$evf" || {
        echo "ERROR: our patch failed to apply to $evf." >&2
        exit 1
    }
    echo "==> Patched $evf successfully."
fi

if [ -x "$BUILD_VENV/bin/python" ] && [ -f "$BUILD_VENV/bin/pip" ]; then
    interp=$(sed -n '1s/^#!\([^ ]*\).*/\1/p' "$BUILD_VENV/bin/pip")
    if [ -n "$interp" ] && [ ! -x "$interp" ]; then
        echo "==> Recreating the venv due to stale shebangs."
        rm -fr "$BUILD_VENV"
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

wheel="$(ls -t dist/torch-*.whl | head -n 1)"
rm -f "$OUT_DIR"/torch-*.whl
cp "$wheel" "$OUT_DIR"/
echo
echo "==> Built $OUT_DIR/$(basename "$wheel")."
