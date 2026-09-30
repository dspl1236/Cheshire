@echo off
rem Build hip/tests/acrbound/boundbench.cpp with the build's toolchain (clang-cl, /O2 /arch:AVX2 as the AliceVision build),
rem then: boundbench.exe <capture file> [reps]   (capture.py makes the capture file)
setlocal
call "%~dp0..\..\..\scripts\env.cmd" >nul 2>&1
cd /d %~dp0
"%ROCM_PATH%/lib/llvm/bin/clang-cl.exe" /nologo /O2 /EHsc /D_CRT_SECURE_NO_WARNINGS /std:c++20 /arch:AVX2 /MD boundbench.cpp /Feboundbench.exe || exit /b 1
echo built %~dp0boundbench.exe
