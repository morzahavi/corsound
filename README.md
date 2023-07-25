# Corsound


## Installation

Follow these steps after downloading the repo:

### Edit the Docker file to contain the path to the database 
````bash
COPY /asvspoof . /app
````
### Create a Docker Container from the Dockerfile. 
````bash
$ docker build -t corsound .
$ docker run -it corsound    
````
### Install the custom functions 
Run this script inside the container
````bash
$ cd
$ cd app
$ sh install_functions.sh
````

## Usage
### Run the main.py file
The file should recreate the tfrecord files, and produce the images to the `images` directory
````bash
$ cd
$ cd app
$ python main.py
````

