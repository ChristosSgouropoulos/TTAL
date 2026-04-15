#!/bin/bash
# =============================================================================
# Setup script to recreate 'tango' and 'eval' conda environments
# Both environments use Python 3.11.0
# =============================================================================

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  Conda Environment Setup Script"
echo "============================================"

# --- Create 'tango' environment ---
echo ""
echo "[1/4] Creating 'tango' conda environment (Python 3.11)..."
conda create -n tango python=3.11.0 -y

echo "[2/4] Installing pip packages into 'tango'..."
conda run -n tango pip install -r "${SCRIPT_DIR}/tango_requirements.txt"

echo ""
echo "✅ 'tango' environment created successfully!"

# --- Create 'eval' environment ---
echo ""
echo "[3/4] Creating 'eval' conda environment (Python 3.11)..."
conda create -n eval python=3.11.0 -y

echo "[4/4] Installing pip packages into 'eval'..."
conda run -n eval pip install -r "${SCRIPT_DIR}/eval_requirements.txt"

echo ""
echo "✅ 'eval' environment created successfully!"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "To activate an environment, run:"
echo "  conda activate tango"
echo "  conda activate eval"
echo ""
