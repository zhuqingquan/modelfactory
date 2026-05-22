#!/usr/bin/env bash
# Convert a .tflite model to MNN using MNNConvert.
# Usage:
#   ./convert_tflite_to_mnn.sh input.tflite [output.mnn] [bizCode]
# Optional: CONVERT_VIA_ONNX=1 to fall back to TFLite -> ONNX -> MNN.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mnn_runtime.sh
source "${SCRIPT_DIR}/mnn_runtime.sh"

INPUT="${1:?Usage: $0 input.tflite [output.mnn] [bizCode]}"
OUTPUT="${2:-${INPUT%.tflite}.mnn}"
BIZ_CODE="${3:-MNN}"

MNNC="$(mnn_resolve_convert || true)"
if [[ -z "${MNNC}" ]]; then
  echo "Error: MNNConvert not found. Run: ${SCRIPT_DIR}/setup_mnn_env.sh" >&2
  exit 1
fi

run_convert() {
  local framework="$1"
  local model="$2"
  "${MNNC}" -f "${framework}" \
    --modelFile "${model}" \
    --MNNModel "${OUTPUT}" \
    --bizCode "${BIZ_CODE}"
}

mkdir -p "$(dirname "${OUTPUT}")"

echo "Using MNNConvert: ${MNNC}"
if run_convert TFLITE "${INPUT}"; then
  echo "Saved: ${OUTPUT}"
  exit 0
fi

if [[ "${CONVERT_VIA_ONNX:-0}" != "1" ]]; then
  echo "TFLite -> MNN failed. Retry with: CONVERT_VIA_ONNX=1 $0 ..." >&2
  exit 1
fi

ONNX_TMP="$(mktemp "${TMPDIR:-/tmp}/mf_tflite_XXXXXX.onnx")"
trap 'rm -f "${ONNX_TMP}"' EXIT

echo "Falling back: TFLite -> ONNX -> MNN"
"${SCRIPT_DIR}/convert_tflite_to_onnx.sh" "${INPUT}" "${ONNX_TMP}"
run_convert ONNX "${ONNX_TMP}"
echo "Saved: ${OUTPUT}"
