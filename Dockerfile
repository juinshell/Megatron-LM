# @ Info: Image of megatron-sharded model experiments.
# Author: Chunyu Xue

# Driver Version: 515.48.07
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# # Driver Version: > 525
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
# FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04
# FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
# FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install common tool & conda
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.10 \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && \
    apt install wget -y && \
    apt install git -y && \
    apt install curl -y && \
    apt install vim -y && \
    apt install bc && \
    apt-get install net-tools -y && \
    apt install ssh -y && \
    wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    mkdir -p /opt/conda/envs/finetune && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Workspace
WORKDIR /app

# Install conda finetune env
COPY requirements.txt requirements.txt
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create --name finetune python=3.10 -y && \
    conda activate finetune && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    # && python3.10 -m pip install numpy --pre torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118
    # && python3.10 -m pip install numpy --pre torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu124
    # && python3.10 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
    # && pip3 install torch torchvision torchaudio
    && python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \ 
    && python3.10 -m pip install -r requirements.txt

# Cuda path
ENV CUDA_PATH=/usr/local/cuda
# ENV LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/compat:/usr/lib/x86_64-linux-gnu:$CUDA_PATH/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
# If the host driver is new enough, don't add `$CUDA_PATH/compat` otherwise will occur: 
#   `CUDA Error: system has unsupported display driver / cuda driver combination`.
ENV LD_LIBRARY_PATH=$CUDA_PATH/lib64:/usr/lib/x86_64-linux-gnu:$CUDA_PATH/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
ENV CUDNN_PATH=/usr/include
# Transformer engine path
ENV NVTE_FRAMEWORK=pytorch

# Install flash-attn (this is resource-intensive and time-consuming)
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate finetune && \
    MAX_JOBS=16 pip install flash-attn==2.6.3 --no-build-isolation

# Install megatron-lm 
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate finetune && \
    # apt update && \
    # apt install gcc-11 g++-11 -y && \
    pip install --upgrade setuptools && \
    MAX_JOBS=16 pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable && \
    # git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git && \
    # cd TransformerEngine && \
    # export NVTE_FRAMEWORK=pytorch && \
    # pip install . && \
    # Option 1: Useful in direct installation on the host.
    # pip install --no-build-isolation git+https://github.com/DicardoX/Megatron-LM.git && \
    # Option 2: Works in docker image build.
    git clone --recursive https://github.com/juinshell/Megatron-LM.git && \
    cd Megatron-LM && \ 
    MAX_JOBS=16 pip install -e .

# Install nvidia apex
# NOTE: This requires the version of torch cuda and the CUDA version are the same.
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate finetune && \
    pip install --upgrade setuptools && \
    # pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    MAX_JOBS=16 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
    # conda install packaging && \
    # pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install nvidia cutlass
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate finetune && \
    conda install anaconda::ninja && \ 
    git clone --recursive https://github.com/NVIDIA/cutlass.git
# ENV CUDACXX=/usr/local/cuda/bin/nvcc
# RUN conda install cmake -y && \
#     git clone --recursive https://github.com/NVIDIA/cutlass.git && \
#     cd cutlass && mkdir build && cd build && \
#     cmake .. -DCUTLASS_NVCC_ARCHS=80

# Copy workspace
# COPY . .
RUN apt-get update && apt-get install -y sudo

RUN mkdir -p /tmp/models/gpt2
# ARG UID
# ARG GID
# RUN groupadd -g $GID jxdeng && \
#     useradd -u $UID -g $GID -m -s /bin/bash -G sudo jxdeng

# RUN echo "jxdeng ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# Enterpoint for bash shell
ENTRYPOINT ["/bin/bash"]
# USER jxdeng

# docker build -t megatron-lm:0.4.0 .
# docker run --name megatron-lm -tid --gpus=all --ipc=host megatron-lm:0.4.0
# docker cp /state/partition/jxdeng/gpt2/ megatron-lm:/tmp/models/gpt2/