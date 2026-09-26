cmake_minimum_required(VERSION 3.21)
# CheshireJPG as a library and a command-line tool (docs/19-cheshirejpg.md). Usable on its own
# (cmake -S hip/port/cheshirejpg) or through add_subdirectory.
#   CheshireJPG:     static library, public header jpegCodec.hpp
#   cheshirejpg_cli: the command-line tool, built as `cheshirejpg` (decode / encode / bench). The
#                    target is not called cheshirejpg because on Windows' case-insensitive file
#                    system its build directory would collide with CheshireJPG's.
# CHESHIREJPG_GPU selects the device language, HIP (default) or CUDA.
project(CheshireJPG LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CHESHIREJPG_GPU HIP CACHE STRING "HIP or CUDA")
set(CHESHIREJPG_COMPAT ${CMAKE_CURRENT_SOURCE_DIR}/../../compat/include)

if(CHESHIREJPG_GPU STREQUAL "CUDA")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set_source_files_properties(jpegGPU.cu PROPERTIES LANGUAGE CUDA)
  add_library(CheshireJPG STATIC jpegGPU.cu jpegHost.cpp)
else()
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 17)
  if(NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx1201 gfx1030 gfx1010)
  endif()
  find_package(hip REQUIRED)
  set_source_files_properties(jpegGPU.cu PROPERTIES LANGUAGE HIP)
  add_library(CheshireJPG STATIC jpegGPU.cu jpegHost.cpp)
  # nvcc pre-includes cuda_runtime.h; clang does not, and the HIP build maps the CUDA names
  # through Cheshire's compat header (allocations go through the memory bridge). Inside
  # AliceVision's build (generator step 6r) that build already force-includes its own copy of the
  # header for every HIP source, so adding this checkout's copy as well defines everything twice
  # (two files, so #pragma once does not help): only a standalone build adds it.
  if(NOT DEFINED ALICEVISION_HIP_DIR)
    target_include_directories(CheshireJPG PRIVATE ${CHESHIREJPG_COMPAT})
    if(CMAKE_HIP_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
      target_compile_options(CheshireJPG PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:/FI${CHESHIREJPG_COMPAT}/cheshire/cuda_to_hip.h>")
    else()
      target_compile_options(CheshireJPG PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:-include${CHESHIREJPG_COMPAT}/cheshire/cuda_to_hip.h>")
    endif()
  endif()
  # hip::host only. hip::device would add HIP compile options to every source of the target, so
  # jpegHost.cpp would be compiled as HIP for the default gfx906 without the compat header
  # ('atomicOr' / '__clz' undeclared, RX 9070 box, Windows clang-cl, ROCm 7.2). jpegGPU.cu is
  # LANGUAGE HIP and gets its device compilation from that.
  target_link_libraries(CheshireJPG PUBLIC hip::host)
endif()
target_include_directories(CheshireJPG PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../cheshiregpu)
# Position-independent: the library links into PIE executables (Ubuntu's gcc default) and later into
# AliceVision's shared libraries. Without it the device-compiled object failed to link on Linux
# (R_X86_64_32S against .text, "recompile with -fPIE").
set_target_properties(CheshireJPG PROPERTIES POSITION_INDEPENDENT_CODE ON)

# AliceVision's build (generator step 6r) adds the library only.
option(CHESHIREJPG_BUILD_TOOL "the cheshirejpg command-line tool" ON)
if(CHESHIREJPG_BUILD_TOOL)
  add_executable(cheshirejpg_cli cheshirejpgTool.cpp)
  set_target_properties(cheshirejpg_cli PROPERTIES OUTPUT_NAME cheshirejpg)
  target_link_libraries(cheshirejpg_cli PRIVATE CheshireJPG)
endif()
