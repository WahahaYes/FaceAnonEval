FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

RUN mkdir /root/app
WORKDIR /root/app
COPY ./assets ./assets

# necessary linux packages
RUN apt update
RUN apt install build-essential -y
RUN DEBIAN_FRONTEND=noninteractive apt install tzdata -y
RUN apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 unzip wget -y

# commands to copy insightface's pretrained models to the docker image
RUN wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -P /root/app/assets 
RUN unzip /root/app/assets/buffalo_l.zip -d /root/app/assets/buffalo_l
RUN rm /root/app/assets/buffalo_l.zip

RUN pip install --upgrade pip setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# if CPU-only, uncomment these lines
# RUN pip uninstall onnxruntime-gpu -y && pip install onnxruntime

# to attach to image:
# docker-compose run --rm faceanoneval

# if on windows (gives a terminal interface):
# winpty docker-compose run --rm faceanoneval
