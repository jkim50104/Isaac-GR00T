# GPU: A100(sm8.0), A6000(sm8.6)

conda create -n gr00t python=3.10 -y
conda activate gr00t
conda install -c conda-forge git-lfs
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T

# If you've already cloned without submodules, initialize them separately:
# git submodule update --init --recursive

conda install nvidia/label/cuda-12.4.1::cuda -c nvidia/label/cuda-12.4.1

pip install uv
uv sync --python 3.10
uv pip install -e .

# For ubuntu 20.04 /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found. Have to be >= 22.04
# So downgrade torch and recompile flash attention with the downgraded torch
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip uninstall flash-attn
pip install --no-build-isolation flash-attn==2.7.4.post1