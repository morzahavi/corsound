# Image
FROM tensorflow/tensorflow:latest-gpu

# Create a directory named 'corsound' inside the container
RUN mkdir /corsound
RUN mkdir app
RUN cd corsound
RUN mkdir /asvspoof


# Set the working directory inside the container
WORKDIR /corsound

# Copy the entire project into the container
COPY . .

# Path to data directory
#COPY / . /app


# Install project dependencies
RUN pip install --no-cache-dir -r req.txt

