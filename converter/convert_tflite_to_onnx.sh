#!/usr/bin/env bash
# Convert a .tflite model to ONNX using tf2onnx.
# Usage:
#   ./convert_tflite_to_onnx.sh input.tflite [output.onnx] [opset]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="${1:?Usage: $0 input.tflite [output.onnx] [opset]}"
OUTPUT="${2:-${INPUT%.tflite}.onnx}"
OPSET="${3:-16}"

VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
if [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON="${VENV_DIR}/bin/python"
elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${SCRIPT_DIR}/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

mkdir -p "$(dirname "$OUTPUT")"
"${PYTHON}" -m tf2onnx.convert --opset "${OPSET}" --tflite "${INPUT}" --output "${OUTPUT}"
echo "Saved: ${OUTPUT}"
