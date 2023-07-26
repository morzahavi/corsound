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
# Install project dependencies
RUN pip install -r /corsound/req.txt
# Install custom functions for project
RUN sh /corsound/install_functions.sh

