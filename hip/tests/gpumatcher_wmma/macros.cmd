@echo off
rem Print the device-side predefined macros of one offload target: macros.cmd <target>, e.g. gfx12-generic
setlocal
call "%~dp0..\..\..\scripts\env.cmd" >nul 2>&1
cd /d %~dp0
type nul > "%TEMP%\cheshire_macro_probe.hip"
"%CHESHIRE_HIP_CLANG%" -x hip --offload-arch=%1 --rocm-path=%ROCM_PATH% --rocm-device-lib-path=%HIP_DEVICE_LIB_PATH% --cuda-device-only -dM -E "%TEMP%\cheshire_macro_probe.hip"
del "%TEMP%\cheshire_macro_probe.hip"
