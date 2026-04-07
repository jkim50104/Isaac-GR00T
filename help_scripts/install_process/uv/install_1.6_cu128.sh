#!/bin/bash
set -e

# GR00T N1.6 — uv install (CUDA 12.8)
# GPU: A100(sm8.0), A6000(sm8.6), RTX PRO 6000 Blackwell(sm12.0)
# Requires: CUDA 12.8 toolkit installed on the system, uv v0.8.4+
#
# Usage: bash help_scripts/install_process/uv/install_1.6_cu128.sh [--clean]

# Clean start: remove existing .venv
if [[ "${1:-}" == "--clean" ]]; then
    echo "Removing existing .venv..."
    rm -rf .venv
    echo "Clean start."
fi

# Check uv is installed
if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed."
    echo "Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Run from the Isaac-GR00T repo root
# git submodule update --init --recursive

uv sync --python 3.10
uv pip install -e .

# Default PyPI torch ships with cu126. Reinstall torch/torchvision/torchaudio with cu128 wheels.
# Also needed for RTX PRO 6000 Blackwell(sm12.0) — default pytorch only supports up to sm_90.
# Must uninstall first — uv skips reinstall if base version matches.
uv pip uninstall torch torchvision torchaudio
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# ffmpeg required for torchcodec:
# sudo apt install ffmpeg

# Detect Ubuntu 20.04
if grep -q "20.04" /etc/os-release 2>/dev/null; then
    echo ""
    echo "=========================================="
    echo "WARNING: Ubuntu 20.04 detected!"
    echo "GLIBC_2.32 is not available on 20.04."
    echo "Downgrading torch and rebuilding flash-attn for compatibility..."
    echo "=========================================="
    echo ""

    # Downgrade torch for GLIBC compatibility
    uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    uv pip uninstall flash-attn
    uv pip install --no-build-isolation flash-attn==2.7.4.post1 # --no-cache-dir if build error

    # torchcodec version for downgraded torch 2.5.1
    uv pip install torchcodec==0.1.1
fi

echo ""
echo "GR00T N1.6 (CUDA 12.8) installation complete."
echo ""
echo "IMPORTANT: Use 'source .venv/bin/activate' before running python scripts"
echo "           to prevent uv from re-syncing and reverting torch back to cu126."

# Verify installation
source .venv/bin/activate
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "
import torch
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA:         {torch.version.cuda}')
print(f'GPU:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NOT AVAILABLE\"}')
print(f'GPU count:    {torch.cuda.device_count()}')

import flash_attn
print(f'flash-attn:   {flash_attn.__version__}')

from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
print(f'GR00T N1.6:   OK')

print()
print('All checks passed!')
"
