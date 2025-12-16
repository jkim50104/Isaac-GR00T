# GPU: A100(sm8.0), A6000(sm8.6)

conda create -n gr00t python=3.10 -y
conda activate gr00t
conda install -c conda-forge git-lfs
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T

conda install nvidia/label/cuda-12.4.1::cuda -c nvidia/label/cuda-12.4.1

pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4

# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124