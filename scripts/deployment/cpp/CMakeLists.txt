cmake_minimum_required(VERSION 3.18)
project(sparsedrive_infer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIBTORCH_PATH "/usr/local/lib/python3.10/dist-packages/torch")
set(CUDA_PATH "/usr/local/cuda")
set(TENSORRT_PATH "/usr/local/tensorrt")

include_directories(include)
include_directories(${LIBTORCH_PATH}/include)
include_directories(${LIBTORCH_PATH}/include/torch/csrc/api/include)
include_directories(${CUDA_PATH}/include)
include_directories(${TENSORRT_PATH}/include)

link_directories(${LIBTORCH_PATH}/lib)
link_directories(${CUDA_PATH}/lib64)
link_directories(${TENSORRT_PATH}/lib)

add_executable(sparsedrive_infer 
    src/main.cpp
    src/sparsedrive_infer.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")

target_link_libraries(sparsedrive_infer 
    torch
    torch_cpu
    torch_cuda
    c10
    c10_cuda
    nvinfer
    nvonnxparser
    nvinfer_plugin
    cudart
    pthread
    dl
    m
)

set_property(TARGET sparsedrive_infer PROPERTY CXX_STANDARD 17)