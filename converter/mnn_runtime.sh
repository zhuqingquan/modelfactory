#!/usr/bin/env bash
# Shared helpers for bundled MNNConvert under tools/mnn/.
# Source this file: source "$(dirname ...)/mnn_runtime.sh"

_mnn_script_dir() {
  cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd
}

mnn_tools_dir() {
  local base
  base="$(_mnn_script_dir)"
  echo "${MNN_TOOLS_DIR:-${base}/tools/mnn}"
}

mnn_set_library_path() {
  local tools
  tools="$(mnn_tools_dir)"
  export LD_LIBRARY_PATH="${tools}:${tools}/express:${tools}/tools/converter${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

# Return 0 if bundled MNNConvert and its .so are loadable.
mnn_bundled_ok() {
  local tools mnn
  tools="$(mnn_tools_dir)"
  mnn="${tools}/MNNConvert"
  [[ -x "${mnn}" && -f "${tools}/libMNN.so" ]] || return 1
  [[ -f "${tools}/express/libMNN_Express.so" ]] || return 1
  [[ -f "${tools}/tools/converter/libMNNConvertDeps.so" ]] || return 1
  mnn_set_library_path
  "${mnn}" -h >/dev/null 2>&1
}

# Print path to MNNConvert (bundled preferred). Sets LD_LIBRARY_PATH when bundled.
mnn_resolve_convert() {
  if [[ -n "${MNN_CONVERT:-}" && -x "${MNN_CONVERT}" ]]; then
    mnn_set_library_path
    echo "${MNN_CONVERT}"
    return 0
  fi
  if mnn_bundled_ok; then
    echo "$(mnn_tools_dir)/MNNConvert"
    return 0
  fi
  local cache="${HOME}/.cache/mf_mnn/build/MNNConvert"
  if [[ -x "${cache}" ]]; then
    export LD_LIBRARY_PATH="${HOME}/.cache/mf_mnn/build:${HOME}/.cache/mf_mnn/build/express:${HOME}/.cache/mf_mnn/build/tools/converter${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "${cache}"
    return 0
  fi
  if command -v MNNConvert >/dev/null 2>&1; then
    command -v MNNConvert
    return 0
  fi
  return 1
}
