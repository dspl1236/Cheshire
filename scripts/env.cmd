@echo off
rem Cheshire build environment (Windows, native HIP via ROCm 7.2.1 pip wheels).
rem Usage: call scripts\env.cmd   (from the repo root or anywhere)
set CHESHIRE_ROOT=%~dp0..
for %%I in ("%CHESHIRE_ROOT%") do set CHESHIRE_ROOT=%%~fI
set CHESHIRE_TOOLS=%CHESHIRE_ROOT%\tools

rem ROCm root = the rocm_sdk_devel package inside the pip venv (forward slashes: CMake chokes on backslashes in env vars)
set ROCM_PATH=%CHESHIRE_TOOLS%/venv-rocm/Lib/site-packages/_rocm_sdk_devel
rem CHESHIRE_ROCM_PATH: another toolchain root, e.g. a copied HIP SDK 6.2 tree (tools/rocm-6.2) for the
rem HIP 6 runtime that AMD ships with the Windows driver and that still enumerates RDNA2 cards.
if defined CHESHIRE_ROCM_PATH set ROCM_PATH=%CHESHIRE_ROCM_PATH%
set ROCM_PATH=%ROCM_PATH:\=/%
set HIP_PATH=%ROCM_PATH%
set HIP_PLATFORM=amd
set HIP_DEVICE_LIB_PATH=%ROCM_PATH%/lib/llvm/amdgcn/bitcode
if defined CHESHIRE_DEVICE_LIB_PATH set HIP_DEVICE_LIB_PATH=%CHESHIRE_DEVICE_LIB_PATH%
set CHESHIRE_HIP_CLANG=%ROCM_PATH%/lib/llvm/bin/clang++.exe

rem MSVC 2026 Build Tools host compiler (vcvarsall prints a harmless vswhere warning)
if not defined VCToolsInstallDir call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
for /f "delims=" %%I in ('dir /b /s "%VCToolsInstallDir%bin\Hostx64\x64\cl.exe"') do set CHESHIRE_CL=%%I
set CHESHIRE_CL=%CHESHIRE_CL:\=/%

set PATH=%CHESHIRE_TOOLS%\cmake\bin;%CHESHIRE_TOOLS%\ninja;%CHESHIRE_TOOLS%\venv-rocm\Lib\site-packages\_rocm_sdk_devel\bin;%PATH%
echo [cheshire] ROCM_PATH=%ROCM_PATH%
echo [cheshire] CL=%CHESHIRE_CL%
