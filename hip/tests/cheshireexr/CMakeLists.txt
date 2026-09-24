cmake_minimum_required(VERSION 3.21)
# CheshireEXR checks (docs/20-cheshireexr.md).
#   cheshireexr_cpu_check: the device pipeline on the host and deferred-async backends against OpenEXR. No GPU.
#   cheshireexr_gpu_check: the decoder on the device against OpenEXR, with timings.
# CHESHIREEXR_GPU selects the device language: HIP (default), CUDA, or OFF for the host check only.
project(cheshireexr_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CHESHIREEXR_GPU HIP CACHE STRING "HIP, CUDA or OFF")
find_package(OpenEXR REQUIRED)  # the reference (and the writer of the test files)

set(EXR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../port/cheshireexr)

add_executable(cheshireexr_cpu_check cheshireexr_cpu_check.cpp ${EXR_SRC}/exrHost.cpp)
target_include_directories(cheshireexr_cpu_check PRIVATE ${EXR_SRC} ${EXR_SRC}/../cheshiregpu ${CMAKE_CURRENT_SOURCE_DIR}/../cheshiregpu)
target_link_libraries(cheshireexr_cpu_check PRIVATE OpenEXR::OpenEXR)

if(NOT CHESHIREEXR_GPU STREQUAL "OFF")
  add_subdirectory(${EXR_SRC} cheshireexr)
  find_package(Threads REQUIRED)
  add_executable(cheshireexr_gpu_check cheshireexr_gpu_check.cpp)
  target_link_libraries(cheshireexr_gpu_check PRIVATE CheshireEXR OpenEXR::OpenEXR Threads::Threads)
endif()
