#!/bin/bash

set -e

LIBTORCH_PATH="/usr/local/lib/python3.10/dist-packages/torch"
CUDA_PATH="/usr/local/cuda"
TENSORRT_PATH="/usr/local/tensorrt"

export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LIBTORCH_PATH/lib:$TENSORRT_PATH/lib:$LD_LIBRARY_PATH"
export CPUINFO_PRINT_ERRORS=0
export CPUINFO_IGNORE_ERRORS=1

MODEL_PATH="../model_scripted_cpu.pt"

if [ -z "$1" ]; then
    echo "Using default model path: $MODEL_PATH"
else
    MODEL_PATH="$1"
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

echo "=========================================="
echo "Running SparseDriveV2 C++ Inference"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "LibTorch: $LIBTORCH_PATH"
echo "TensorRT: $TENSORRT_PATH"
echo "=========================================="
echo ""

./build/sparsedrive_infer "$MODEL_PATH"