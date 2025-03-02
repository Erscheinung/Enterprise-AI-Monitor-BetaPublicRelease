#!/bin/bash

# Script to run the Python server

# Set paths
PROJ_DIR="$(dirname "$0")/.."
SERVER_DIR="$PROJ_DIR/python_server"

cd "$SERVER_DIR"

# Ensure we have the required Python packages
python -m pip install flask flask_cors torch onnx numpy

echo "Starting Enterprise AI Monitor server..."
echo "Press Ctrl+C to stop the server"
echo "-----------------------------------"

# Run the server
python app.py