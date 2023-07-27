#!/bin/bash

# Function to check if Metal is supported on macOS
function is_mac_with_metal() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        major_version="$(sw_vers -productVersion | cut -d. -f1)"
        minor_version="$(sw_vers -productVersion | cut -d. -f2)"
        if [[ "$major_version" -ge 11 && "$minor_version" -ge 0 ]]; then
            return 0
        fi
    fi
    return 1
}

# Check if Metal is supported on the system
if is_mac_with_metal; then
    echo "Metal is supported on this macOS system."

    # Install TensorFlow-metal
    echo "Installing TensorFlow-metal..."
    pip install tensorflow-metal

    # Optionally, install TensorFlow-macos if needed
    # echo "Installing TensorFlow-macos..."
    # pip install tensorflow-macos

    echo "TensorFlow-metal installed successfully."

else
    echo "Metal is not supported on this macOS system. Skipping installation."
fi
