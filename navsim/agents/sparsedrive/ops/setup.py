import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


def make_cuda_ext(
    name,
    module,
    sources,
    sources_cuda=[],
    extra_args=[],
    extra_include_path=[],
):

    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        sources += sources_cuda
    else:
        print("Compiling {} without CUDA".format(name))
        extension = CppExtension

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    import sys
    thrust_path = os.path.join(sys.prefix, "targets", "x86_64-linux", "include", "cccl")
    if not os.path.exists(thrust_path):
        thrust_path = "/usr/local/cuda-12.2/include"
    extra_include = [thrust_path]
    print(f"Using thrust path: {thrust_path}")
    
    setup(
        name="deformable_aggregation_ext",
        ext_modules=[
            make_cuda_ext(
                "deformable_aggregation_with_depth_ext",
                module=".",
                sources=[
                    f"src/deformable_aggregation_with_depth.cpp",
                    f"src/deformable_aggregation_with_depth_cuda.cu",
                ],
                extra_include_path=extra_include,
            ),
            make_cuda_ext(
                "deformable_aggregation_ext",
                module=".",
                sources=[
                    f"src/deformable_aggregation.cpp",
                    f"src/deformable_aggregation_cuda.cu",
                ],
                extra_include_path=extra_include,
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
    )