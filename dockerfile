FROM nvcr.io/nvidia/pytorch:23.05-py3

ARG REPO_DIR=./
ARG USERID=1000
ARG GROUPID=1000
ARG USERNAME=jordydalcorso
ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Rome
ENV REPO_DIR=${REPO_DIR}

RUN apt-get update -y && apt-get install -y git-flow sudo
RUN groupadd -g $GROUPID rs
RUN useradd -ms /bin/bash -u $USERID -g $GROUPID $USERNAME

RUN echo "$USERNAME ALL=(ALL:ALL) NOPASSWD: ALL" | tee /etc/sudoers.d/$USERNAME

RUN pip install --upgrade pip

COPY entrypoint.sh /opt/app/entrypoint.sh

ENTRYPOINT [ "bash", "/opt/app/entrypoint.sh" ]
CMD [ "bash" ]