cmake_minimum_required(VERSION 3.21)
# Cheshire GPU JPEG checks (docs/19-gpu-jpeg.md).
#   jpeg_cpu_check: the device pipeline run on the host backend against libjpeg-turbo. No GPU.
#   jpeg_gpu_check: the codec on the device against libjpeg-turbo, with timings.
# CHESHIRE_JPEG_GPU selects the device language: HIP (default), CUDA, or OFF for the host check only.
project(cheshire_gpu_jpeg_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CHESHIRE_JPEG_GPU HIP CACHE STRING "HIP, CUDA or OFF")
find_package(JPEG REQUIRED)  # libjpeg-turbo is the reference

set(JPEG_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../port/gpu_jpeg)

add_executable(jpeg_cpu_check jpeg_cpu_check.cpp ${JPEG_SRC}/jpegHost.cpp)
target_include_directories(jpeg_cpu_check PRIVATE ${JPEG_SRC})
target_link_libraries(jpeg_cpu_check PRIVATE JPEG::JPEG)

if(NOT CHESHIRE_JPEG_GPU STREQUAL "OFF")
  add_subdirectory(${JPEG_SRC} gpu_jpeg)
  add_executable(jpeg_gpu_check jpeg_gpu_check.cpp)
  target_link_libraries(jpeg_gpu_check PRIVATE cheshire_gpu_jpeg JPEG::JPEG)
endif()
