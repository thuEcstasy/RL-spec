FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"
WORKDIR /workspace

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    wget bzip2 ca-certificates git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

COPY environment.yml /tmp/environment.yml


RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n jacobi_forcing python=3.12 pip && \
    conda clean -afy

# 先装 torch
RUN conda run -n jacobi_forcing pip install \
    torch==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 \
    --no-cache-dir

RUN conda env update -n jacobi_forcing -f /tmp/environment.yml
RUN conda run -n jacobi_forcing pip install flash-attn==2.8.3 --no-deps --no-cache-dir --no-build-isolation


COPY . /workspace

# 让 bash 启动时自动激活 conda 环境
RUN echo ". /opt/conda/etc/profile.d/conda.sh && conda activate jacobi_forcing" >> /root/.bashrc

ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate jacobi_forcing && exec \"$@\"", "--"]
CMD ["/bin/bash"]