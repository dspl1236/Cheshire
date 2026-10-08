@echo off
rem Build a standalone HIP test of hip/tests/knnfilter for gfx1201 with the build's ROCm 7.2 toolchain.
rem   build.cmd <name>        (name.hip -> name.exe)
setlocal
call "%~dp0..\..\..\scripts\env.cmd" >nul 2>&1
cd /d %~dp0
"%CHESHIRE_HIP_CLANG%" -x hip --offload-arch=gfx1201 --rocm-path=%ROCM_PATH% --rocm-device-lib-path=%HIP_DEVICE_LIB_PATH% -O3 -std=c++17 -D_CRT_SECURE_NO_WARNINGS %1.hip -o %1.exe -L"%ROCM_PATH%/lib" -lamdhip64 || exit /b 1
echo built %~dp0%1.exe
