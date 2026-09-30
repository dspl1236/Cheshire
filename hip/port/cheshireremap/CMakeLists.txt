cmake_minimum_required(VERSION 3.21)
# CheshireRemap as a library (public header cheshireRemap.hpp): PrepareDenseScene's undistortion on the
# device, AliceVision's arithmetic to the bit. Usable on its own (cmake -S hip/port/cheshireremap) or
# through add_subdirectory (AliceVision's image library, generator step 8g).
# CHESHIREREMAP_GPU selects the device language, HIP (default) or CUDA.
project(CheshireRemap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CHESHIREREMAP_GPU HIP CACHE STRING "HIP or CUDA")
set(CHESHIREREMAP_COMPAT ${CMAKE_CURRENT_SOURCE_DIR}/../../compat/include)

if(CHESHIREREMAP_GPU STREQUAL "CUDA")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  # no multiply-add contraction: the functor must round every product and sum as the host does
  set_source_files_properties(remapGPU.cu PROPERTIES LANGUAGE CUDA COMPILE_OPTIONS "--fmad=false")
  add_library(CheshireRemap STATIC remapGPU.cu)
  # cheshire/env.h on its own: the compat directory also holds the HIP stand-ins for cuda_runtime.h,
  # which would shadow the toolkit's headers under nvcc
  configure_file(${CHESHIREREMAP_COMPAT}/cheshire/env.h ${CMAKE_CURRENT_BINARY_DIR}/env-include/cheshire/env.h COPYONLY)
  target_include_directories(CheshireRemap PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/env-include)
else()
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 17)
  if(NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx1201 gfx1030 gfx1010)
  endif()
  find_package(hip REQUIRED)
  # the functor's own pragma (remapStages.hpp) keeps clang from contracting it
  set_source_files_properties(remapGPU.cu PROPERTIES LANGUAGE HIP)
  add_library(CheshireRemap STATIC remapGPU.cu)
  # As CheshireJPG: inside AliceVision's build every HIP source already force-includes its copy of the
  # compat header; only a standalone build adds this checkout's.
  if(NOT DEFINED ALICEVISION_HIP_DIR)
    target_include_directories(CheshireRemap PRIVATE ${CHESHIREREMAP_COMPAT})
    if(CMAKE_HIP_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
      target_compile_options(CheshireRemap PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:/FI${CHESHIREREMAP_COMPAT}/cheshire/cuda_to_hip.h>")
    else()
      target_compile_options(CheshireRemap PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:-include${CHESHIREREMAP_COMPAT}/cheshire/cuda_to_hip.h>")
    endif()
  endif()
  target_link_libraries(CheshireRemap PUBLIC hip::host)
endif()
target_include_directories(CheshireRemap PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../cheshiregpu)
set_target_properties(CheshireRemap PROPERTIES POSITION_INDEPENDENT_CODE ON)
