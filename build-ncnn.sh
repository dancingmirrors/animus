#!/bin/sh

set -eu

NCNN_VERSION="${NCNN_VERSION:-20260526}"
PYTHON="${PYTHON:-python3}"
MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 2)}"
SRC_DIR="${SRC_DIR:-$(pwd)/.ncnn-src}"
OUT_DIR="$(pwd)/wheels"
STAMP="$OUT_DIR/.animus-ncnn-build-id"

NCNN_EXTRA_OPTIONS="${NCNN_EXTRA_OPTIONS:-}"

set -- \
    -DCMAKE_BUILD_TYPE=Release \
    -DNCNN_VULKAN=ON \
    -DNCNN_PYTHON=ON \
    -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_BUILD_TOOLS=OFF \
    -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_TESTS=OFF \
    -DNCNN_SHARED_LIB=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

if [ -n "$NCNN_EXTRA_OPTIONS" ]; then
    # shellcheck disable=SC2086
    set -- "$@" $NCNN_EXTRA_OPTIONS
fi

abi="$("$PYTHON" -c 'import sys; print("cp%d%d" % sys.version_info[:2])')"

build_id="ncnn $NCNN_VERSION $abi
$(printf '%s\n' "$@" | sort)"

echo "==> ncnn $NCNN_VERSION | python=$PYTHON ($abi) | jobs=$MAX_JOBS"
printf '    %s\n' "$@"

mkdir -p "$OUT_DIR"

if ls "$OUT_DIR"/ncnn*.so >/dev/null 2>&1 \
   && [ "${NCNN_FORCE_REBUILD:-0}" != "1" ] \
   && [ -f "$STAMP" ] \
   && [ "$(cat "$STAMP")" = "$build_id" ]; then
    echo "==> $OUT_DIR already holds a module built with these options."
    echo "    Set NCNN_FORCE_REBUILD=1 to build it again."
    exit 0
fi

if [ ! -d "$SRC_DIR/.git" ]; then
    git clone --depth 1 --branch "$NCNN_VERSION" --recursive \
        https://github.com/Tencent/ncnn "$SRC_DIR"
else
    echo "==> Reusing $SRC_DIR."
fi

cd "$SRC_DIR"

if ! command -v glslangValidator >/dev/null 2>&1 \
   && [ ! -d glslang/glslang ]; then
    echo "==> Fetching the bundled glslang for the compute shaders."
    git submodule update --init --recursive --depth 1 glslang
fi

mkdir -p build
cd build
cmake "$@" ..
make -j"$MAX_JOBS"

module="$(ls python/ncnn/ncnn*.so 2>/dev/null | head -n 1)"
if [ -z "$module" ]; then
    echo "ERROR: the Python extension was not produced. Check NCNN_PYTHON." >&2
    exit 1
fi

rm -f "$OUT_DIR"/ncnn*.so
cp "$module" "$OUT_DIR"/
printf '%s\n' "$build_id" > "$STAMP"
echo
echo "==> Built $OUT_DIR/$(basename "$module")."
