#!/bin/bash

# Script to build the API bridge component

# Set paths
PROJ_DIR="$(dirname "$0")/.."
REACT_DIR="$PROJ_DIR/react_dashboard"
BUILD_DIR="$REACT_DIR/build"

echo "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$REACT_DIR"

echo "Running CMake to configure the project..."
cmake -B "$BUILD_DIR" -S .

if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

echo "Building the project..."
cmake --build "$BUILD_DIR"

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build completed successfully!"
echo "The API bridge executable is located at: $BUILD_DIR/api_bridge"