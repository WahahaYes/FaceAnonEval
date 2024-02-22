FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y \
	git g++ gcc ffmpeg libsm6 libxext6 unzip wget

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"


COPY --chown=appuser . /home/appuser
RUN pip install -r requirements.txt

RUN wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -P /home/appuser/.insightface/models && unzip /home/appuser/.insightface/models/buffalo_l.zip -d /home/appuser/.insightface/models/buffalo_l


