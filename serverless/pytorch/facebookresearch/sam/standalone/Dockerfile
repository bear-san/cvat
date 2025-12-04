FROM ubuntu:22.04
# set NVIDIA container runtime settings
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# disable interactive frontend
ENV DEBIAN_FRONTEND=noninteractive
# set workdir
WORKDIR /opt/nuclio/sam
# install basic deps
RUN apt-get update && apt-get -y install curl git python3 python3-pip ffmpeg libsm6 libxext6
# install sam deps
RUN pip3 install torch torchvision torchaudio pycocotools matplotlib onnxruntime onnx
# install sam code
RUN pip3 install git+https://github.com/facebookresearch/segment-anything.git
# download sam weights
RUN curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# map pip3 and python3 to pip and python
RUN ln -s /usr/bin/pip3 /usr/local/bin/pip && ln -s /usr/bin/python3 /usr/bin/python

ADD main.py /opt/nuclio/sam/main.py
ADD model_handler.py /opt/nuclio/sam/model_handler.py

CMD ["python3", "main.py"]
