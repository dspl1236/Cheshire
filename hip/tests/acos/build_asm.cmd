@echo off
setlocal
call "%~dp0..\..\..\scripts\env.cmd" >nul 2>&1
cd /d %~dp0
"%CHESHIRE_HIP_CLANG%" -x hip --offload-arch=gfx1201 --rocm-path=%ROCM_PATH% --rocm-device-lib-path=%HIP_DEVICE_LIB_PATH% -O3 -std=c++17 -D_CRT_SECURE_NO_WARNINGS --cuda-device-only -S %1.hip -o %1.s 2>&1
