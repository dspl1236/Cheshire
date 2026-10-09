@echo off
rem Build hip/tests/popsift_generic/divide_probe.cpp against a PopSIFT install (default build\popsift-gfx12-generic-install;
rem first argument names another, e.g. gfx1201). PopSIFT's headers include <cuda_runtime.h>: the project's shim in
rem hip/compat/include, which needs ROCm's headers. Run: set PATH=<install>\bin;%%ROCM_PATH%%\bin;%%PATH%% then divide_probe.exe.
setlocal
call "%~dp0..\..\..\scripts\env.cmd" >nul 2>&1
cd /d %~dp0
set NAME=%~1
if "%NAME%"=="" set NAME=gfx12-generic
set INST=%CHESHIRE_ROOT%\build\popsift-%NAME%-install
"%ROCM_PATH%/lib/llvm/bin/clang-cl.exe" /nologo /DWIN32 /D_WINDOWS /EHsc /O2 /Zi /DNDEBUG -std:c++17 -MD -D__HIP_PLATFORM_AMD__ "-I%CHESHIRE_ROOT%\hip\compat\include" "-I%ROCM_PATH%/include" "-I%INST%\include" divide_probe.cpp /Fedivide_probe-%NAME%.exe /link "%INST%\lib\popsift.lib" "%ROCM_PATH%/lib/amdhip64.lib" dbghelp.lib || exit /b 1
echo built %~dp0divide_probe-%NAME%.exe
