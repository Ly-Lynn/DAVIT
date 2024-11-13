ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
WORKDIR /davit
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub \
    apt-key del 7fa2af8 \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY . /davit/
# Install MMCV
RUN pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html &&\
pip install --no-cache-dir opencv-python terminaltables && \
pip install --no-cache-dir "mmengine<1.0.0" &&\
pip install --no-cache-dir yapf==0.40.1

# Install MMDetection
RUN conda clean --all
# RUN git clone -b v2.28.2 https://github.com/open-mmlab/mmdetection.git /davit/mmdetection
# WORKDIR /davit/mmdetection/
ENV FORCE_CUDA="1"
# WORKDIR /davit/mmdet/
# RUN pip install -r requirements/build.txt
# RUN pip install --no-cache-dir -e .
# WORKDIR /davit