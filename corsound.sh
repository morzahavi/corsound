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
LOCAL_DIR="repo"

if [ ! -d "$LOCAL_DIR" ]; then
    git clone "$REPO_URL" "$LOCAL_DIR"
else
    echo "Repository already exists. Skipping git clone."
fi
cd corsound
# Pull Docker image
docker pull tensorflow/tensorflow
docker tag tensorflow/tensorflow corsound
docker build . -t corsound
docker run -it corsound /bin/bash
#
cd
cd corsound
#--- Script ends here ---

