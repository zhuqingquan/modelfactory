#!/usr/bin/env bash
# First-time setup: create venv and install converter dependencies.
# Usage: ./setup_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found. Please install Python 3.10+." >&2
  exit 1
fi

if [[ ! -f "${REQUIREMENTS}" ]]; then
  echo "Error: requirements.txt not found at ${REQUIREMENTS}" >&2
  exit 1
fi

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  echo "Virtual environment already exists: ${VENV_DIR}"
else
  echo "Creating virtual environment: ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

echo "Upgrading pip..."
"${VENV_DIR}/bin/pip" install --upgrade pip

echo "Installing dependencies from requirements.txt..."
"${VENV_DIR}/bin/pip" install -r "${REQUIREMENTS}"

echo ""
echo "Setup complete."
echo "  Python: ${VENV_DIR}/bin/python"
echo ""
echo "Convert a model:"
echo "  ${SCRIPT_DIR}/convert_tflite_to_onnx.sh <input.tflite> [output.onnx]"
