#!/bin/bash

set -e

TORCH_PYTHON_PATH="/usr/local/lib/python3.10/dist-packages/torch"
LIBTORCH_PATH="$TORCH_PYTHON_PATH"
CUDA_PATH="/usr/local/cuda"

echo "=========================================="
echo "Building SparseDriveV2 C++ Inference"
echo "=========================================="
echo ""
echo "LibTorch path: $LIBTORCH_PATH"
echo "CUDA path: $CUDA_PATH"
echo ""

echo "Checking TorchConfig.cmake..."
TORCH_CONFIG=$(find "$TORCH_PYTHON_PATH" -name "TorchConfig.cmake" 2>/dev/null | head -1)
if [ -n "$TORCH_CONFIG" ]; then
    echo "Found: $TORCH_CONFIG"
    TORCH_DIR=$(dirname "$TORCH_CONFIG")
    echo "Torch_DIR: $TORCH_DIR"
else
    echo "NOT FOUND - checking lib directory..."
    TORCH_DIR="$TORCH_PYTHON_PATH/lib"
    echo "Using Torch_DIR: $TORCH_DIR"
fi

mkdir -p build
cd build

export CUDA_HOME="$CUDA_PATH"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$TORCH_PYTHON_PATH/lib:$LD_LIBRARY_PATH"

cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" \
         -DTorch_DIR="$TORCH_DIR"

echo ""
echo "=========================================="
echo "Starting compilation..."
echo "=========================================="

make -j$(nproc)

echo ""
echo "=========================================="
echo "Build completed!"
echo "Executable: $(pwd)/sparsedrive_infer"
echo "=========================================="