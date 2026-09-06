#!/bin/sh

set -eu

TORCH_VERSION="${TORCH_VERSION:-v2.13.0}"
PYTHON="${PYTHON:-python3}"
MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 2)}"
MARCH="${MARCH:-native}"
SRC_DIR="${SRC_DIR:-$(pwd)/.torch-src}"
OUT_DIR="$(pwd)/wheels"
BUILD_VENV="${BUILD_VENV:-$SRC_DIR/.buildvenv}"
STAMP="$SRC_DIR/.animus-build-id"
TORCH_PROFILE="${TORCH_PROFILE:-auto}"
TORCH_EXTRA_OPTIONS="${TORCH_EXTRA_OPTIONS:-}"

detect_profile() {
    if [ "$MARCH" != "native" ]; then
        case "$MARCH" in
            x86-64-v3 | x86-64-v4 | haswell | broadwell | skylake* | \
            cascadelake | icelake* | tigerlake | alderlake | raptorlake | \
            sapphirerapids | znver* | core-avx2)
                echo modern
                ;;
            *)
                echo legacy
                ;;
        esac
        return
    fi

    if grep '^flags' /proc/cpuinfo 2>/dev/null | head -n 1 | tr ' ' '\n' \
        | grep -qx avx2; then
        echo modern
    else
        echo legacy
    fi
}

case "$TORCH_PROFILE" in
    auto)
        TORCH_PROFILE="$(detect_profile)"
        echo "==> Profile auto-detected as '$TORCH_PROFILE'."
        ;;
    legacy | modern) ;;
    *)
        echo "ERROR: TORCH_PROFILE must be auto, legacy or modern." >&2
        exit 1
        ;;
esac

set -- \
    CC=clang \
    CXX=clang++ \
    BLAS=OpenBLAS \
    BUILD_TEST=0 \
    USE_CUDA=0 \
    USE_DISTRIBUTED=0 \
    USE_ITT=0 \
    USE_KINETO=0 \
    USE_MAGMA=0 \
    USE_ROCM=0 \
    USE_XPU=0 \
    USE_NNPACK=0 \
    USE_PYTORCH_QNNPACK=0
#   USE_CPP_STACKTRACES=0

if [ "$TORCH_PROFILE" = "modern" ]; then
    set -- "$@" \
        USE_MKLDNN=1 \
        USE_XNNPACK=1 \
        USE_FBGEMM=0
else
    set -- "$@" \
        USE_MKLDNN=0 \
        USE_XNNPACK=0 \
        USE_FBGEMM=0
fi

set -- "$@" \
    "CMAKE_C_FLAGS=-march=$MARCH -mtune=generic" \
    "CMAKE_CXX_FLAGS=-march=$MARCH -mtune=generic"

if [ -n "$TORCH_EXTRA_OPTIONS" ]; then
    # shellcheck disable=SC2086
    set -- "$@" $TORCH_EXTRA_OPTIONS
fi

build_id="$TORCH_VERSION
$(printf '%s\n' "$@" | sort)"

echo "==> torch $TORCH_VERSION | python=$PYTHON | jobs=$MAX_JOBS | march=$MARCH"
echo "==> profile=$TORCH_PROFILE"
printf '    %s\n' "$@"

mkdir -p "$OUT_DIR"

WHEEL_STAMP="$OUT_DIR/.animus-build-id"
if ls "$OUT_DIR"/torch-*.whl >/dev/null 2>&1 \
   && [ "${TORCH_FORCE_REBUILD:-0}" != "1" ]; then
    if [ ! -f "$WHEEL_STAMP" ]; then
        if [ "${TORCH_REUSE_WHEEL:-0}" = "1" ]; then
            echo "==> Reusing $OUT_DIR/$(basename "$(ls "$OUT_DIR"/torch-*.whl)")"
            echo "    even though nothing records how it was built."
            exit 0
        fi
        echo "==> $OUT_DIR holds a wheel with no record of its build options,"
        echo "    so it might not match profile '$TORCH_PROFILE'. Rebuilding."
        echo "    Set TORCH_REUSE_WHEEL=1 to install it as-is instead."
    elif [ "$(cat "$WHEEL_STAMP")" = "$build_id" ]; then
        echo "==> $OUT_DIR already holds a wheel built with exactly these"
        echo "    options. Set TORCH_FORCE_REBUILD=1 to build it again."
        exit 0
    else
        echo "==> The wheel in $OUT_DIR was built with different options."
        echo "    Rebuilding for profile '$TORCH_PROFILE'."
    fi
fi

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

if [ "$fresh_clone" = 0 ] && [ "$built" != "$build_id" ]; then
    was="$(printf '%s\n' "$built" | sed -n '1p')"
    echo "==> Discarding artifacts built from ${was:-an unknown version} under"
    echo "    options that no longer match. Expect a full rebuild."
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

printf '%s\n' "$build_id" > "$STAMP"

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

glibc_only_include() {
    file="$1"
    header="$2"
    line="#include <$header>"
    guard="#ifdef __GLIBC__"

    if [ ! -f "$file" ]; then
        echo "==> No $file. Nothing to guard."
        return 0
    fi
    if awk -v line="$line" -v guard="$guard" '
        prev == guard && $0 == line { found = 1 }
        { prev = $0 }
        END { exit !found }
    ' "$file"; then
        echo "==> $file already guards <$header>."
        return 0
    fi
    if ! grep -qxF "$line" "$file"; then
        echo "ERROR: '$line' not found in $file." >&2
        echo "       Update the patch in $0." >&2
        exit 1
    fi
    awk -v line="$line" -v guard="$guard" '
        $0 == line { print guard; print line; print "#endif"; next }
        { print }
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    grep -qxF "$guard" "$file" || {
        echo "ERROR: our patch failed to apply to $file." >&2
        exit 1
    }
    echo "==> Patched $file to guard <$header> successfully."
}

glibc_only_include \
    third_party/tensorpipe/tensorpipe/channel/cma/context_impl.cc linux/prctl.h

evf="cmake/EnvVarForwarding.cmake"
evf_marker="# Animus: re-read our build options straight from the environment."
if [ ! -f "$evf" ]; then
    echo "==> No $evf. This PyTorch forwards the environment from setup.py."
elif grep -q '_envfwd_apply' "$evf"; then
    echo "==> $evf already overrides the cache from the environment."
    echo "    Nothing to patch."
    if grep -qxF "$evf_marker" "$evf"; then
        awk -v marker="$evf_marker" '
            $0 == marker { exit }
            /^[[:space:]]*$/ { blanks = blanks "\n"; next }
            { printf "%s%s\n", blanks, $0; blanks = "" }
        ' "$evf" > "$evf.tmp" && mv "$evf.tmp" "$evf"
    fi
else
    if grep -qxF "$evf_marker" "$evf"; then
        awk -v marker="$evf_marker" '
            $0 == marker { exit }
            /^[[:space:]]*$/ { blanks = blanks "\n"; next }
            { printf "%s%s\n", blanks, $0; blanks = "" }
        ' "$evf" > "$evf.tmp" && mv "$evf.tmp" "$evf"
    fi
    {
        printf '\n%s\n' "$evf_marker"
        echo 'foreach(_animus_var'
        printf '%s\n' "$@" | sed -n \
            's/^\(BUILD_[A-Za-z0-9_]*\|CMAKE_[A-Za-z0-9_]*\|USE_[A-Za-z0-9_]*\)=.*/    \1/p' \
            | sort -u
        cat <<'PATCH'
    )
  if(DEFINED ENV{${_animus_var}})
    set(${_animus_var} "$ENV{${_animus_var}}"
        CACHE STRING "From env ${_animus_var}" FORCE)
  endif()
endforeach()
PATCH
    } >> "$evf"
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

if [ -f requirements-build.txt ]; then
    "$VPY" -m pip install -r requirements-build.txt
fi

export PATH="$BUILD_VENV/bin:$PATH"
for opt in "$@"; do
    export "$opt"
done
export MAX_JOBS

if command -v ccache >/dev/null 2>&1; then
    ccache -M "${CCACHE_MAXSIZE:-25Gi}" >/dev/null 2>&1 || true
    export CCACHE_BASEDIR="$PWD"
    export CCACHE_NOHASHDIR=1
    : "${CCACHE_COMPILERCHECK:=content}"
    export CCACHE_COMPILERCHECK
fi

"$VPY" -m build --wheel --no-isolation

wheel="$(ls -t dist/torch-*.whl | head -n 1)"
rm -f "$OUT_DIR"/torch-*.whl
cp "$wheel" "$OUT_DIR"/
printf '%s\n' "$build_id" > "$WHEEL_STAMP"
echo
echo "==> Built $OUT_DIR/$(basename "$wheel")."
