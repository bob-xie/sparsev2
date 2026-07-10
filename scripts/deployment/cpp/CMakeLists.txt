cmake_minimum_required(VERSION 3.18)
project(sparsedrive_infer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

include_directories(include)

add_executable(sparsedrive_infer 
    src/main.cpp
    src/sparsedrive_infer.cpp
)

target_link_libraries(sparsedrive_infer 
    "${TORCH_LIBRARIES}"
    "${CUDA_LIBRARIES}"
)

set_property(TARGET sparsedrive_infer PROPERTY CXX_STANDARD 17)

if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET sparsedrive_infer
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:sparsedrive_infer>)
endif(MSVC)