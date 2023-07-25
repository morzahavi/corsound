# Python runtime base image
FROM python:3.9

# Working directory in the container
WORKDIR /cs

# Copy the req.txt file into the container
COPY req.txt .

# Install project dependencies
RUN pip install --no-cache-dir -r req.txt

# Copy the rest of the project code into the container
COPY . .

# Define the command to run your Python application
CMD ["python", "your_script.py"]
