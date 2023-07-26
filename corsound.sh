#!/bin/bash

################################################################################
# Title:       Corsound Assignment
# Description: 
# Author:      Mor Zahavi
# Date:        July 23, 2023
# Version:     1.0
################################################################################

# Usage: 



# Description:
# This script will clone the git repo "corsound" into the home dir
# Create a Docker container
# Create a virtual environment
# And execute the Python and LaTeX files

# Notes:

# Exit codes:
#   0   Success
#   1   General error
#   2   Invalid arguments

# Dependencies:

#--- Script starts here ---
# Clone repo if not exists
cd

REPO_URL="https://github.com/morzahavi/corsound.git"
REPO_DIR="corsound"

# Check if the repository directory exists
if [ -d "$REPO_DIR" ]; then
    # If the directory exists, perform git pull to update the repository
    cd "$REPO_DIR"
    git pull
else
    # If the directory doesn't exist, perform git clone to clone the repository
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd corsound
# Pull Docker image
docker pull tensorflow/tensorflow
docker tag tensorflow/tensorflow corsound
docker build --no-cache . -t corsound
docker run -it -v /Users/morzahavi/icloud/Downloads/asvspoof:/corsound corsound /bin/bash
docker run -it -v  ubuntu /bin/bash
#
cd
cd corsound
#--- Script ends here ---

