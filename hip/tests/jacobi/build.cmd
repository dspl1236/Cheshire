@echo off
rem Build hip/tests/jacobi/jacobibench.cpp with the AliceVision build's toolchain and flags (clang-cl /O2 /arch:AVX2, Eigen
rem from the vcpkg deps, AV_EIGEN_MEMORY_ALIGNMENT on), then: jacobibench.exe [seed]. Extra arguments go to the compiler
rem after the defaults, so "build.cmd /arch:AVX" builds Eigen's path without FMA.
setlocal
call "%~dp0..\..\..\scripts\env.cmd" >nul 2>&1
cd /d %~dp0
set EIGEN=%~dp0..\..\..\tools\vcpkg-deps\x64-windows-release\installed\x64-windows-release\include\eigen3
"%ROCM_PATH%/lib/llvm/bin/clang-cl.exe" /nologo /arch:AVX2 /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG -std:c++20 -MD /GR /Zc:__cplusplus -DALICEVISION_EIGEN_REQUIRE_ALIGNMENT=1 -DNOMINMAX -D_USE_MATH_DEFINES /D_CRT_SECURE_NO_WARNINGS "-I%EIGEN%" %* jacobibench.cpp /Fejacobibench.exe || exit /b 1
echo built %~dp0jacobibench.exe
