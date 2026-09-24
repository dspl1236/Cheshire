cmake_minimum_required(VERSION 3.21)
# Cheshire GPU JPEG as a library and a command-line tool (docs/19-gpu-jpeg.md). Usable on its own
# (cmake -S hip/port/gpu_jpeg) or through add_subdirectory.
#   cheshire_gpu_jpeg: static library, public header jpegCodec.hpp
#   cheshire_jpeg:     decode / encode / bench from the command line
# CHESHIRE_JPEG_GPU selects the device language, HIP (default) or CUDA.
project(cheshire_gpu_jpeg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CHESHIRE_JPEG_GPU HIP CACHE STRING "HIP or CUDA")
set(CHESHIRE_JPEG_COMPAT ${CMAKE_CURRENT_SOURCE_DIR}/../../compat/include)

if(CHESHIRE_JPEG_GPU STREQUAL "CUDA")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set_source_files_properties(jpegGPU.cu PROPERTIES LANGUAGE CUDA)
  add_library(cheshire_gpu_jpeg STATIC jpegGPU.cu jpegHost.cpp)
else()
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 17)
  if(NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx1201 gfx1030 gfx1010)
  endif()
  find_package(hip REQUIRED)
  set_source_files_properties(jpegGPU.cu PROPERTIES LANGUAGE HIP)
  add_library(cheshire_gpu_jpeg STATIC jpegGPU.cu jpegHost.cpp)
  target_include_directories(cheshire_gpu_jpeg PRIVATE ${CHESHIRE_JPEG_COMPAT})
  # nvcc pre-includes cuda_runtime.h; clang does not, and the HIP build maps the CUDA names
  # through Cheshire's compat header (allocations go through the memory bridge)
  if(CMAKE_HIP_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(cheshire_gpu_jpeg PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:/FI${CHESHIRE_JPEG_COMPAT}/cheshire/cuda_to_hip.h>")
  else()
    target_compile_options(cheshire_gpu_jpeg PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:-include${CHESHIRE_JPEG_COMPAT}/cheshire/cuda_to_hip.h>")
  endif()
  target_link_libraries(cheshire_gpu_jpeg PUBLIC hip::host PRIVATE hip::device)
endif()
target_include_directories(cheshire_gpu_jpeg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(cheshire_jpeg jpegTool.cpp)
target_link_libraries(cheshire_jpeg PRIVATE cheshire_gpu_jpeg)
