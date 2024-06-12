FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    g++ \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python
RUN python3 -m pip install --upgrade pip

RUN python -m pip install \
    torch==1.13.1 \
    torchvision==0.14.1

RUN python -m pip install \
    transformers==4.27.4 \
    scikit-learn==1.1.1 \
    tqdm==4.64.0 \
    protobuf==3.20.0 \
    torch-scatter==2.1.1 \
    ftfy==6.1.1 \
    h5py==3.8.0 \
    matplotlib tensorboard\
    mamba-ssm
