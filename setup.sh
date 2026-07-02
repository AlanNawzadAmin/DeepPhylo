#!/bin/bash
# One-shot install into a fresh conda env `phylo4cuda`. Assumes `conda` is already
# available (run `conda init` once if not). torch / dgl / flash-attn come from
# CUDA-specific wheels; everything else is declared in pyproject.toml and pulled in
# by `pip install -e .`.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

# Pick the torch/flash-attn CUDA wheel tag from the driver.
DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
DRIVER_CUDA_MAJOR=$(echo "$DRIVER_CUDA" | cut -d. -f1)
DRIVER_CUDA_MINOR=$(echo "$DRIVER_CUDA" | cut -d. -f2)
if [ "$DRIVER_CUDA_MAJOR" -ge 13 ]; then
    CUDA_TAG="cu130"
elif [ "$DRIVER_CUDA_MAJOR" -eq 12 ] && [ "$DRIVER_CUDA_MINOR" -ge 8 ]; then
    CUDA_TAG="cu128"
else
    CUDA_TAG="cu126"
fi

conda create -n phylo4cuda python=3.10 -y
conda activate phylo4cuda

pip install dgl==2.4.0+cu121 -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
pip install torch==2.10.0 --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2B${CUDA_TAG}torch2.10-cp310-cp310-linux_x86_64.whl"
conda install -y -c bioconda iqtree==3.0.1 kalign3==3.4.0
conda install -y -c nvidia cuda-cudart=12.1 cuda-toolkit=12.1 libcublas=12.1
pip install -e "$HERE"
pip install "numpy<2"
