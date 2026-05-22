#!/usr/bin/env bash
# Install dependencies for both ONNX and MNN conversion pipelines.
# Usage: ./setup_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========== ONNX (Python) =========="
"${SCRIPT_DIR}/setup_env.sh"

echo ""
echo "========== MNN (MNNConvert) =========="
"${SCRIPT_DIR}/setup_mnn_env.sh"

echo ""
echo "All converter environments are ready."
