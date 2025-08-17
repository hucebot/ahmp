FROM --platform=x86_64 ubuntu:22.04

ARG USERNAME="robot"
ARG HOME_DIR="/home/${USERNAME}"

ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="Europe/Athens"
WORKDIR /deps

# Installation of Ubuntu packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    python3-dev \
    python-is-python3 \
    python3-pip \
    git \
    cmake \
    build-essential \
    unzip \
    wget libx11-dev \
    python3-tk \
    tk-dev \
    g++ \
    gcc \
    x11-apps \
    nano \
    libeigen3-dev \
    pybind11-dev \
    libblas-dev \
    liblapack-dev \
    libmetis-dev \
    gfortran \
    libboost-all-dev \
    libtbb-dev \
    software-properties-common \
    gdb \
    liburdfdom-dev \
    liburdfdom-headers-dev && \
    apt-get clean

# Add the ThirdParty-HSL directory
RUN git clone https://github.com/coin-or-tools/ThirdParty-HSL.git
ADD coinhsl /deps/ThirdParty-HSL/coinhsl
RUN cd ThirdParty-HSL && \
    ./configure && \
    make VERBOSE=1 && \
    make VERBOSE=1 install

# Add the ThirdParty-ASL directory
RUN git clone https://github.com/coin-or-tools/ThirdParty-ASL.git && cd ThirdParty-ASL && \
    ./get.ASL && \
    ./configure && \
    make VERBOSE=1 && \
    make VERBOSE=1 install

# Add the ThirdParty-Mumps directory
RUN git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git && cd ThirdParty-Mumps && \
    ./get.Mumps && \
    ./configure && \
    make VERBOSE=1 && \
    make VERBOSE=1 install

# Add Ipopt directory
RUN git clone https://github.com/coin-or/Ipopt.git && cd Ipopt && \
    ./configure && \
    make VERBOSE=1 && \
    make VERBOSE=1 install
RUN apt install -y coinor-libipopt-dev

RUN apt-get install -y gcc g++ gfortran git cmake liblapack-dev pkg-config --install-recommends
RUN python -m pip install pin meshcat setuptools cyipopt matplotlib imageio

ENV PYTHONPATH="/home/${USERNAME}/code/src"

ARG HOME_DIR="/home/${USERNAME}"

# Working directory
WORKDIR ${HOME_DIR}
