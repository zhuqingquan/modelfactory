#!/usr/bin/env bash
# Setup MNNConvert: use bundled tools/mnn if OK, otherwise clone and build MNN.
# Usage: ./setup_mnn_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mnn_runtime.sh
source "${SCRIPT_DIR}/mnn_runtime.sh"

MNN_SRC_DIR="${MNN_SRC_DIR:-${HOME}/.cache/mf_mnn/MNN}"
MNN_BUILD_DIR="${MNN_BUILD_DIR:-${HOME}/.cache/mf_mnn/build}"
MNN_TOOLS_DIR="${MNN_TOOLS_DIR:-${SCRIPT_DIR}/tools/mnn}"
MNN_REPO="${MNN_REPO:-https://github.com/alibaba/MNN.git}"
MNN_BRANCH="${MNN_BRANCH:-master}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

install_bundled_to_tools() {
  local src_build="$1"
  mkdir -p "${MNN_TOOLS_DIR}/express" "${MNN_TOOLS_DIR}/tools/converter"
  cp -f "${src_build}/MNNConvert" "${MNN_TOOLS_DIR}/"
  cp -f "${src_build}/libMNN.so" "${MNN_TOOLS_DIR}/"
  cp -f "${src_build}/express/libMNN_Express.so" "${MNN_TOOLS_DIR}/express/"
  cp -f "${src_build}/tools/converter/libMNNConvertDeps.so" "${MNN_TOOLS_DIR}/tools/converter/"
  chmod +x "${MNN_TOOLS_DIR}/MNNConvert"
}

if mnn_bundled_ok; then
  echo "Bundled MNNConvert is ready: $(mnn_tools_dir)/MNNConvert"
  echo "Skip build. To force rebuild: rm -rf tools/mnn && ./setup_mnn_env.sh"
  exit 0
fi

echo "Bundled MNNConvert not found or not runnable; building from source ..."

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' not found." >&2
    echo "Install build tools, e.g.:" >&2
    echo "  sudo apt install -y git cmake g++ libprotobuf-dev protobuf-compiler" >&2
    exit 1
  fi
}

need_cmd git
need_cmd cmake
need_cmd g++

if [[ ! -d "${MNN_SRC_DIR}/.git" ]]; then
  echo "Cloning MNN -> ${MNN_SRC_DIR}"
  mkdir -p "$(dirname "${MNN_SRC_DIR}")"
  git clone --depth 1 --branch "${MNN_BRANCH}" "${MNN_REPO}" "${MNN_SRC_DIR}"
else
  echo "MNN source exists: ${MNN_SRC_DIR}"
fi

mkdir -p "${MNN_BUILD_DIR}"
echo "Configuring CMake in ${MNN_BUILD_DIR} ..."
cmake -S "${MNN_SRC_DIR}" -B "${MNN_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_CONVERTER=ON \
  -DMNN_BUILD_TRAIN=OFF \
  -DMNN_BUILD_DEMO=OFF \
  -DMNN_BUILD_TOOLS=ON \
  -DMNN_BUILD_BENCHMARK=OFF

echo "Building MNNConvert (-j${JOBS}) ..."
cmake --build "${MNN_BUILD_DIR}" --target MNNConvert -j"${JOBS}"

MNN_CONVERT_BIN="${MNN_BUILD_DIR}/MNNConvert"
if [[ ! -x "${MNN_CONVERT_BIN}" ]]; then
  echo "Error: MNNConvert not found at ${MNN_CONVERT_BIN}" >&2
  exit 1
fi

echo "Installing built binaries into ${MNN_TOOLS_DIR} ..."
install_bundled_to_tools "${MNN_BUILD_DIR}"

if mnn_bundled_ok; then
  echo ""
  echo "MNN setup complete (built and installed to tools/mnn)."
  echo "  MNNConvert: $(mnn_tools_dir)/MNNConvert"
else
  echo "Error: installed MNNConvert still not runnable." >&2
  exit 1
fi

echo ""
echo "Convert a model:"
echo "  ${SCRIPT_DIR}/convert_tflite_to_mnn.sh <input.tflite> [output.mnn]"
