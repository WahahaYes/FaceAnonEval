FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime AS gpu

RUN mkdir /root/app
WORKDIR /root/app
COPY ./assets ./assets

# Necessary linux packages
RUN apt update
RUN apt install build-essential -y
RUN DEBIAN_FRONTEND=noninteractive apt install tzdata -y
RUN apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 unzip wget -y

# Copy insightface's pretrained models to the docker image
RUN wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -P /root/.insightface/models
RUN unzip /root/.insightface/models/buffalo_l.zip -d /root/.insightface/models/buffalo_l
RUN rm /root/.insightface/models/buffalo_l.zip

# Copy HSEmotion model to docker image
RUN wget https://github.com/av-savchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/enet_b0_8_best_afew.pt?raw=true -P /root/.hsemotion -O enet_b0_8_best_afew.pt

# Install dependencies from pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# For cpu, replace gpu-only package(s)
FROM gpu AS cpu
RUN pip uninstall onnxruntime-gpu -y && pip install --no-cache-dir onnxruntime

## to attach to image:
# docker-compose run --rm faceanoneval

## if on windows (gives a terminal interface):
# winpty docker-compose run --rm faceanoneval --target [gpu, cpu]
