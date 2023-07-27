# Corsound

# Introduction
The application is based on a Docker image called `tensorflow/tensorflow`
which is an x86_64 20.04 ubuntu based Linux distro, with Cuda toolkit.


## Prerequisites 
Docker Desktop installed



## Installation

Follow these steps after downloading the repo:

Make sure you Docker desktop client has permissions to bind or mount the data libraries you want to connect to the project


### Edit the Docker file to contain the path to the database 
````bash
COPY /asvspoof . /app
````
### Create a Docker Container from the Dockerfile. 
Replace the PATH_TO_DATA with the path to the asvspoof directory on you machine
````bash
$ docker build -t corsound .
$ docker run -it /PATH_TO_DATA:/corsound/asvspoof corsound /bin/bash    
````

Every time you run the container after the initial build, you should run:

```bash
git pull
```
### Install the custom functions 
Run this script inside the container
````bash
$ cd
$ cd corsound
$ sh install_functions.sh
````

## Usage
### Run the main.py file
The file should recreate the tfrecord files, and produce the images to the `images` directory
````bash
$ cd corsound
$ python main.py
````

### GPU test
To test whether the container utilises the machine GPU run the following script inside the container:
````bash
$ cd corsound
$ python test_gpu.py
````

