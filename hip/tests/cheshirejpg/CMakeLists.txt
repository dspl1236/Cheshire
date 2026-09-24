cmake_minimum_required(VERSION 3.21)
# CheshireJPG checks (docs/19-cheshirejpg.md).
#   cheshirejpg_cpu_check: the device pipeline run on the host backend against libjpeg-turbo. No GPU.
#   cheshirejpg_gpu_check: the codec on the device against libjpeg-turbo, with timings.
# CHESHIREJPG_GPU selects the device language: HIP (default), CUDA, or OFF for the host check only.
project(cheshirejpg_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CHESHIREJPG_GPU HIP CACHE STRING "HIP, CUDA or OFF")
find_package(JPEG REQUIRED)  # libjpeg-turbo is the reference

set(JPEG_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../port/cheshirejpg)

add_executable(cheshirejpg_cpu_check cheshirejpg_cpu_check.cpp ${JPEG_SRC}/jpegHost.cpp)
target_include_directories(cheshirejpg_cpu_check PRIVATE ${JPEG_SRC})
target_link_libraries(cheshirejpg_cpu_check PRIVATE JPEG::JPEG)

if(NOT CHESHIREJPG_GPU STREQUAL "OFF")
  add_subdirectory(${JPEG_SRC} cheshirejpg)
  find_package(Threads REQUIRED)
  add_executable(cheshirejpg_gpu_check cheshirejpg_gpu_check.cpp)
  target_link_libraries(cheshirejpg_gpu_check PRIVATE CheshireJPG JPEG::JPEG Threads::Threads)
endif()
