@echo off
rem Build hip/tests/jacobi/lanestest.cpp (step 13a's header against Nullspace2) with the AliceVision build's toolchain and flags
rem (clang-cl /O2 /arch:AVX2, Eigen from the vcpkg deps, AV_EIGEN_MEMORY_ALIGNMENT on; env.h from the patched source tree),
rem then: lanestest.exe [count] [seed]. Extra arguments go to the compiler after the defaults.
setlocal
call "%~dp0..\..\..\scripts\env.cmd" >nul 2>&1
cd /d %~dp0
set EIGEN=%~dp0..\..\..\tools\vcpkg-deps\x64-windows-release\installed\x64-windows-release\include\eigen3
set AVSRC=%~dp0..\..\..\third_party\aliceVision\src
"%ROCM_PATH%/lib/llvm/bin/clang-cl.exe" /nologo /arch:AVX2 /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG -std:c++20 -MD /GR /Zc:__cplusplus -DALICEVISION_EIGEN_REQUIRE_ALIGNMENT=1 -DNOMINMAX -D_USE_MATH_DEFINES /D_CRT_SECURE_NO_WARNINGS "-I%EIGEN%" "-I%AVSRC%" %* lanestest.cpp /Felanestest.exe || exit /b 1
echo built %~dp0lanestest.exe
