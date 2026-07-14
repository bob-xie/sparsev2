#!/bin/bash

set -e

LIBTORCH_PATH="/usr/local/lib/python3.10/dist-packages/torch"
CUDA_PATH="/usr/local/cuda"
TENSORRT_PATH="/usr/local/tensorrt"

echo "=========================================="
echo "Building SparseDriveV2 C++ Inference"
echo "=========================================="
echo ""
echo "LibTorch path: $LIBTORCH_PATH"
echo "CUDA path: $CUDA_PATH"
echo "TensorRT path: $TENSORRT_PATH"
echo ""

echo "Checking libtorch libraries..."
ls -la $LIBTORCH_PATH/lib/libtorch*.so 2>/dev/null || echo "WARNING: Some libtorch libraries may be missing"

echo "Checking TensorRT libraries..."
ls -la $TENSORRT_PATH/lib/libnvinfer*.so 2>/dev/null || echo "WARNING: Some TensorRT libraries may be missing"

mkdir -p build
cd build

export CUDA_HOME="$CUDA_PATH"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LIBTORCH_PATH/lib:$TENSORRT_PATH/lib:$LD_LIBRARY_PATH"

cmake .. -DCMAKE_BUILD_TYPE=Release

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