# Use the official TensorFlow GPU base image with Ubuntu 18.04
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project into the container
COPY . .
COPY ~/icloud/projects/corsound . /app
COPY ~/icloud/Downloads/asvspoof . /app


# Install project dependencies
RUN pip install --no-cache-dir -r req.txt

# Set any environment variables, if needed
# ENV MY_ENV_VARIABLE=value

# Start your Python script (replace 'your_script.py' with your actual entry point)

