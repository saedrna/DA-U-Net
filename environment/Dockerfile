# hash:sha256:066663b940ca1fc286ff43fa8dd8dc5b29a2eaf814a79d73bce829d9eddb8847
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-dev=2.56.4-0ubuntu0.18.04.4 \
        python-qt4=4.12.1+dfsg-2 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y --channel=https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main --channel=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
        numpy==1.18.1 \
        opencv==3.4.2 \
        python==3.6.0 \
        tensorflow-gpu==1.9.0 \
    && conda clean -ya
