# Use a base image with PyTorch pre-installed
FROM python:3.9
COPY ~/icloud/job_market/corsound/req.txt /tmp
RUN pip install -r /tmp/req.txt

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project into the container
# COPY . /app

# Install any additional dependencies, if required
# For example, if you need to install other Python packages:
# RUN python3 -m pip install -r req.txt

# Command to run your Python script (replace "your_script.py" with the actual filename)

