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
RUN wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -P /root/.insightface/models 
RUN unzip /root/.insightface/models/buffalo_l.zip -d /root/.insightface/models/buffalo_l
RUN rm /root/.insightface/models/buffalo_l.zip

RUN pip install --upgrade pip setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# to attach to image:
# docker-compose run --rm faceanoneval

# if on windows (gives a terminal interface):
# winpty docker-compose run --rm faceanoneval
