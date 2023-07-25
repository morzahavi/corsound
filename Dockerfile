# Image
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project into the container
COPY . .
COPY /asvspoof . /app


# Install project dependencies
RUN pip install --no-cache-dir -r req.txt

