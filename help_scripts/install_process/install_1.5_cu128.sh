# GPU: RTX PRO 6000 blackwell(sm12.0)

conda create -n gr00t python=3.10
conda activate gr00t
conda install -c conda-forge git-lfs
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
# conda install nvidia/label/cuda-12.8.1::cuda -c nvidia/label/cuda-12.8.1

pip install --upgrade setuptools
pip install -e .[base]
# pip install --no-build-isolation flash-attn==2.7.1.post4

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install --no-build-isolation --upgrade flash-attn==2.8.3 #flash-attn>=2.8.0.post2