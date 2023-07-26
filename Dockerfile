# Image
FROM tensorflow/tensorflow:latest-gpu

# Create a directory named 'corsound' inside the container
RUN mkdir /corsound
RUN mkdir /corsound/asvspoof
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3 ./get-pip.py
RUN sudo apt update
RUN sudo apt install git
RUN sudo git clone https://github.com/morzahavi/corsound.git

# Set the working directory inside the container
WORKDIR /corsound

# Copy the entire project into the container


# Install project dependencies
RUN pip install -r req.txt

