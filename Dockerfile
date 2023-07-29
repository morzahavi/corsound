# Image
FROM tensorflow/tensorflow:latest-gpu
# Update package lists and install Git
RUN apt-get update && apt-get install -y git

#RUN sudo apt update
#RUN sudo apt install git
RUN git clone https://github.com/morzahavi/corsound.git
# Set the working directory inside the container
WORKDIR /corsound
RUN mkdir /corsound/asvspoof
# Install tensorflow repo
RUN cd && git clone https://github.com/tensorflow/tensorflow.git
# Install bazel for GPU integration
RUN apt update && apt install bazel
RUN apt update && apt full-upgrade
RUN apt install bazel-1.0.0
RUN apt install default-jdk
RUN apt install g++ unzip zip
RUN apt-get install default-jdk

# Install project dependencies
RUN pip install -r /corsound/req.txt
RUN pip install -U audio_classification_models
# Install custom functions for project
RUN sh /corsound/install_functions.sh
# Install CUDA backend
RUN sh corsound/install_cuda.sh
FROM nvidia/cuda:10.2-base
CMD nvidia-smi

