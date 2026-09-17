cmake_minimum_required(VERSION 3.28)
project(cheshire_hip_tests LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
if(NOT CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1201 gfx1100)
endif()
find_package(hip REQUIRED)

add_executable(hello_hip hello_hip.hip)
target_link_libraries(hello_hip PRIVATE hip::device)

add_executable(mipmap_tex mipmap_tex.hip)
target_link_libraries(mipmap_tex PRIVATE hip::device)

add_executable(membridge_probe membridge_probe.hip)
target_link_libraries(membridge_probe PRIVATE hip::device)

add_executable(bridge_test bridge_test.hip)
target_include_directories(bridge_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../compat/include)
target_link_libraries(bridge_test PRIVATE hip::device)

add_executable(half_tex half_tex.hip)
target_link_libraries(half_tex PRIVATE hip::device)

add_executable(launch_bench launch_bench.hip)
target_link_libraries(launch_bench PRIVATE hip::device)

add_executable(texfmt_probe texfmt_probe.hip)
target_link_libraries(texfmt_probe PRIVATE hip::device)

add_executable(surf_probe surf_probe.hip)
target_link_libraries(surf_probe PRIVATE hip::device)

add_executable(alloc_bench alloc_bench.hip)
target_link_libraries(alloc_bench PRIVATE hip::device)

add_executable(alloc3d_bench alloc3d_bench.hip)
target_link_libraries(alloc3d_bench PRIVATE hip::device)

add_executable(linear_tex linear_tex.hip)
target_link_libraries(linear_tex PRIVATE hip::device)

add_executable(host_atomics host_atomics.hip)
target_link_libraries(host_atomics PRIVATE hip::device)
add_executable(host_backing host_backing.hip)
target_link_libraries(host_backing PRIVATE hip::device)

# the GPU min-cut (CUDA dialect through the compat header) against a MaxFlow_CSR dump
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../port/gpu_maxflow/maxflowGPU.cu PROPERTIES LANGUAGE HIP)
add_executable(maxflow_test maxflow_test.hip ${CMAKE_CURRENT_SOURCE_DIR}/../port/gpu_maxflow/maxflowGPU.cu)
target_include_directories(maxflow_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../compat/include)
target_compile_options(maxflow_test PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:-include${CMAKE_CURRENT_SOURCE_DIR}/../compat/include/cheshire/cuda_to_hip.h>")
target_link_libraries(maxflow_test PRIVATE hip::device)
