#!/bin/bash

git clone https://github.com/zhenyiqi/LongLive.git
cd LongLive
# install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

source ~/.bashrc

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# create env
conda create -n longlive python=3.10 -y
conda activate longlive

# install dependencies
#
conda install nvidia/label/cuda-12.4.1::cuda
conda install -c nvidia/label/cuda-12.4.1 cudatoolkit
pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -r appdirs==1.4.4
pip install nvidia-pyindex --no-build-isolation
pip install flash-attn --no-build-isolation

# download huggingface-cli
pip install huggingface_hub
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
hf download Efficient-Large-Model/LongLive --local-dir longlive_models

sudo apt update && sudo apt install tmux
sudo apt install neovim
