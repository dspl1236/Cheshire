# Cheshire: build PopSIFT (alicevision/popsift v0.10.0) as a HIP library for AMD.
#
# Upstream's own CMake is written around the CUDA language. This builds the same 29 sources as HIP
# instead, with hip/port/popsift/popsift_hip.h force-included in front of each one, and installs a
# PopSiftConfig.cmake exporting PopSift::popsift so AliceVision's find_package(PopSift CONFIG) picks
# it up in place of the CUDA build that vcpkg ships.
#
# Configure with the project's ROCm toolchain, e.g.
#   cmake -G Ninja -S hip/port/popsift -B build/popsift-hip \
#         -DCMAKE_BUILD_TYPE=Release -DCHESHIRE_POPSIFT_ARCH=gfx1201 \
#         -DCMAKE_INSTALL_PREFIX=build/popsift-install
cmake_minimum_required(VERSION 3.21)
project(PopSift LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CHESHIRE_POPSIFT_ARCH "gfx1201" CACHE STRING "HIP architectures for PopSIFT")
set(CMAKE_HIP_ARCHITECTURES ${CHESHIRE_POPSIFT_ARCH})

set(CHESHIRE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(POPSIFT_SRC "${CHESHIRE_ROOT}/third_party/popsift/src")
set(POPSIFT_GEN "${CHESHIRE_ROOT}/build/popsift-gen")

if (NOT EXISTS "${POPSIFT_SRC}/popsift/popsift.cu")
    message(FATAL_ERROR "popsift sources not found at ${POPSIFT_SRC}; clone alicevision/popsift v0.10.0 and run scripts/apply_popsift_patch.py")
endif()
if (NOT EXISTS "${POPSIFT_GEN}/popsift/sift_config.h")
    message(FATAL_ERROR "generated sift_config.h missing; run scripts/apply_popsift_patch.py")
endif()

# One translation unit: HIP has no relocatable device code for COFF objects on Windows, and PopSIFT
# shares __constant__ and __device__ globals across its sources. scripts/apply_popsift_patch.py
# writes the unit; it simply includes all 29 sources.
set(POPSIFT_UNITY "${CHESHIRE_ROOT}/build/popsift-gen/popsift_unity.cu")
if (NOT EXISTS "${POPSIFT_UNITY}")
    message(FATAL_ERROR "${POPSIFT_UNITY} missing; run scripts/apply_popsift_patch.py")
endif()
set_source_files_properties("${POPSIFT_UNITY}" PROPERTIES LANGUAGE HIP)

add_library(popsift SHARED "${POPSIFT_UNITY}")
set_target_properties(popsift PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON)         # upstream relies on CMake for this too; no dllexport in the sources

target_include_directories(popsift PUBLIC
    "$<BUILD_INTERFACE:${POPSIFT_SRC}>"
    "$<BUILD_INTERFACE:${POPSIFT_GEN}>"
    "$<INSTALL_INTERFACE:include>")
# Deliberately NOT ${POPSIFT_SRC}/popsift. PopSIFT has a header called features.h, and with that
# directory on the include path libstdc++'s bits/os_defines.h picks it up for its own
# #include <features.h>, so glibc's never loads, __GLIBC_USE is left undefined and every later
# system header fails. Windows does not see this because the MSVC STL has no <features.h>. Nothing
# needs it: the sources use quoted includes, which resolve relative to the including file.
# ${POPSIFT_GEN}/popsift is needed and is safe - popsift.h includes "sift_config.h" by bare name,
# and that directory holds only the generated sift_config.h.
target_include_directories(popsift PRIVATE
    "${POPSIFT_GEN}/popsift"
    "${CHESHIRE_ROOT}/hip/compat/include"
    "${CMAKE_CURRENT_SOURCE_DIR}")

# The shim is included by the unity source itself (scripts/apply_popsift_patch.py writes it), not
# forced in here: clang-cl and clang++ spell that flag differently, and on Linux forcing a header
# ahead of clang's HIP runtime wrapper breaks glibc's feature detection.

# CHESHIRE_POPSIFT_ERRCHK=ON checks every kernel launch (upstream's ERRCHK_AFTER_KERNEL), which turns
# a silent device fault into a message naming the kernel.
option(CHESHIRE_POPSIFT_ERRCHK "Check after every kernel launch" OFF)
if (CHESHIRE_POPSIFT_ERRCHK)
    target_compile_definitions(popsift PRIVATE ERRCHK_AFTER_KERNEL)
endif()

include(GNUInstallDirs)
install(TARGETS popsift EXPORT PopSiftTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY "${POPSIFT_SRC}/popsift/" DESTINATION include/popsift
        FILES_MATCHING PATTERN "*.h")
install(FILES "${POPSIFT_GEN}/popsift/sift_config.h" DESTINATION include/popsift)
install(EXPORT PopSiftTargets NAMESPACE PopSift:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PopSift)

include(CMakePackageConfigHelpers)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/PopSiftConfig.cmake.in"
"@PACKAGE_INIT@\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/PopSiftTargets.cmake\")\ncheck_required_components(PopSift)\n")
configure_package_config_file("${CMAKE_CURRENT_BINARY_DIR}/PopSiftConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/PopSiftConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PopSift)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/PopSiftConfigVersion.cmake"
    VERSION 0.10.0 COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/PopSiftConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/PopSiftConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PopSift)
