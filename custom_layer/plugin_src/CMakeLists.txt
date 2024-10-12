cmake_minimum_required(VERSION 3.2)
project(preproc_layer)

# Use C++ 17 standard
set(CMAKE_CXX_STANDARD 17)
set(CPP_STANDARD 17 CACHE STRING "CPP Standard Version")

# Set TensorRT and CUDA library paths
find_package(CUDA REQUIRED)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
link_directories(/usr/local/cuda/lib64)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include /usr/local/include /usr/include/x86_64-linux-gnu)
find_library(TENSORRT_LIBRARY nvinfer PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
find_library(NVONNXPARSER_LIBRARY nvonnxparser PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

find_library(NVINFER_PLUGIN_LIB nvinfer_plugin PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

# Set OpenCV library paths
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

cuda_add_library(preproc_kernel SHARED preprocKernel.cu)
target_link_libraries(preproc_kernel nvinfer cudart)

# generate dynamic library
add_library(
    ${PROJECT_NAME} 
    SHARED 
    preprocPlugin.cpp
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${TENSORRT_LIBRARY}
    ${NVONNXPARSER_LIBRARY}
    ${NVINFER_PLUGIN_LIB}
    ${CUDA_LIBRARIES}
    cudart
    ${OpenCV_LIBS}
    preproc_kernel # plugin
)

# C++ compiler flags (optional)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-Wall")
