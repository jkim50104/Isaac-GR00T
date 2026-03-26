# GPU: A100(sm8.0), A6000(sm8.6), RTX PRO 6000 blackwell(sm12.0)

conda create -n gr00t python=3.10 -y
conda activate gr00t
conda install -c conda-forge git-lfs
git clone --recurse-submodules https://github.com/jkim50104/Isaac-GR00T
cd Isaac-GR00T

# If you've already cloned without submodules, initialize them separately:
git submodule update --init --recursive

conda install nvidia/label/cuda-12.8.1::cuda -c nvidia/label/cuda-12.8.1 -y

pip install uv
uv sync --python 3.10
uv pip install -e .

#################################################################################################################################################################
# For RTX PRO 6000 blackwell(sm12.0) we have to rebuild pytorch (The github uv setting compiled pytorch 2.7.1 supports sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90)
uv pip uninstall torch
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128


#######################################################################################################
# For ubuntu 20.04 /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found. Have to be >= 22.04
# So downgrade torch and recompile flash attention with the downgraded torch
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

uv pip uninstall flash-attn
pip install --no-build-isolation flash-attn==2.7.4.post1 # --no-cache-dir  if build error

# for demo data inference. Pytorch related pacackges compatible version list: https://pytorch.kr/get-started/compatibility/
pip install torchcodec==0.5.0 # for torch 2.5.1=>0.1.1, torch 2.7.1=>0.5.0
conda install -c conda-forge "ffmpeg=6.1.2"