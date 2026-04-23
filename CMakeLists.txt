cmake_minimum_required(VERSION 3.20)
project(sigmo_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Target SYCL per GPU NVIDIA
set(SYCL_TARGETS "nvptx64-nvidia-cuda")
set(CUDA_PATH "/usr/local/cuda")

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c
            "import pathlib, dpctl; root = pathlib.Path(dpctl.__file__).resolve().parent; matches = list(root.rglob('dpctl4pybind11.hpp')); print(matches[0].parent if matches else '')"
    OUTPUT_VARIABLE DPCTL_PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "DPCTL pybind11 include dir: ${DPCTL_PYBIND11_INCLUDE_DIR}")

if(NOT DPCTL_PYBIND11_INCLUDE_DIR)
    message(FATAL_ERROR "Could not locate dpctl4pybind11.hpp inside the installed dpctl package")
endif()

pybind11_add_module(_core
    cpp/src/module.cpp
    cpp/src/api.cpp
)

target_include_directories(_core
    PRIVATE
        cpp/include
        ${DPCTL_PYBIND11_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/external/sigmo/library/include
)

target_compile_features(_core PRIVATE cxx_std_17)

target_compile_options(_core PRIVATE
    -fsycl
    -fsycl-targets=${SYCL_TARGETS}
    --cuda-path=${CUDA_PATH}
)

target_link_options(_core PRIVATE
    -fsycl
    -fsycl-targets=${SYCL_TARGETS}
    --cuda-path=${CUDA_PATH}
)

set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/sigmo
)