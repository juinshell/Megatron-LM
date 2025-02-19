FROM nvcr.io/nvidia/pytorch:23.04-py3
# Install megatron-lm 
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate finetune && \
    pip install --upgrade setuptools && \
    MAX_JOBS=16 pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable && \
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
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    MAX_JOBS=16 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./