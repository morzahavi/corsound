# Image
FROM tensorflow/tensorflow:latest-gpu

# Create a directory named 'corsound' inside the container
RUN mkdir /corsound
RUN mkdir /corsound/asvspoof
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3 ./get-pip.py

# Set the working directory inside the container
WORKDIR /corsound

# Copy the entire project into the container
COPY . .

# Install project dependencies
RUN pip install -r req.txt

