@echo off
rem Compile-check wmma_matcher.hip for the gfx12-generic code object the packages carry.
setlocal
call "%~dp0..\..\..\scripts\env.cmd" >nul 2>&1
cd /d %~dp0
"%CHESHIRE_HIP_CLANG%" -x hip --offload-arch=gfx12-generic --rocm-path=%ROCM_PATH% --rocm-device-lib-path=%HIP_DEVICE_LIB_PATH% -O3 -std=c++17 -D_CRT_SECURE_NO_WARNINGS --cuda-device-only -c wmma_matcher.hip -o "%TEMP%\wmma_generic.o" || exit /b 1
echo compiled for gfx12-generic
