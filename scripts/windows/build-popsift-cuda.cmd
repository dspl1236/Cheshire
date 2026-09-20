@echo off
rem Build PopSIFT with the CUDA backend on Windows, for GPU SIFT in the CUDA AliceVision build.
rem Usage: scripts\windows\build-popsift-cuda.cmd [configure|build|install]   (default install)
rem
rem Nothing is ported. third_party/popsift is a clone of alicevision/popsift v0.10.0 - the original
rem CUDA project - and scripts/apply_popsift_patch.py is what adapts it *to* HIP. So this resets the
rem tree to pristine upstream and builds that, while scripts\build-popsift.cmd builds the HIP port
rem from hip/port/popsift instead. The two do not share a build or install directory.
rem
rem Same vcvars ordering as build-alicevision-cuda.cmd, for the same reason: env.cmd initialises
rem VS 2026 when VCToolsInstallDir is unset, vcvars will not re-initialise an already-configured
rem shell, and nvcc rejects _MSC_VER >= 1950.
setlocal
set STEP=%~1
if "%STEP%"=="" set STEP=install

if not defined CHESHIRE_VS2022 set CHESHIRE_VS2022=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
if not defined CHESHIRE_MSVC_VER set CHESHIRE_MSVC_VER=14.44
call "%CHESHIRE_VS2022%\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=%CHESHIRE_MSVC_VER% >nul
if errorlevel 1 ( echo vcvars64 for %CHESHIRE_MSVC_VER% failed & exit /b 1 )
call "%~dp0..\env.cmd"

set CLVER=
for /f "tokens=7" %%V in ('cl 2^>^&1 ^| findstr /i /c:"Version"') do set CLVER=%%V
echo %CLVER% | findstr /b /c:"19.4" >nul
if errorlevel 1 ( echo   cl is %CLVER%; nvcc needs _MSC_VER 1910-1949 & exit /b 1 )
echo [msvc] cl %CLVER%

if not defined CHESHIRE_CUDA_PATH set CHESHIRE_CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
if not exist "%CHESHIRE_CUDA_PATH%\bin\nvcc.exe" ( echo no nvcc at "%CHESHIRE_CUDA_PATH%" & exit /b 1 )
rem sm_61 covers both NVIDIA test cards (docs/18)
if not defined CHESHIRE_CUDA_ARCHS set CHESHIRE_CUDA_ARCHS=61

set R=%CHESHIRE_ROOT:\=/%
set SRC=%R%/third_party/popsift
set V=%R%/tools/vcpkg-deps/x64-windows-release
set BLD=%R%/build/popsift-cuda
set INST=%R%/build/popsift-cuda-install

rem A HIP build leaves the tree patched; CUDA wants it as upstream wrote it.
git -C "%CHESHIRE_ROOT%\third_party\popsift" checkout -- . 2>nul
for /f "delims=" %%I in ('git -C "%CHESHIRE_ROOT%\third_party\popsift" log --oneline -1') do echo [popsift] source %%I
echo [popsift] archs=%CHESHIRE_CUDA_ARCHS%  install=%INST%

if /i "%STEP%"=="configure" goto :configure
if not exist "%BLD%/CMakeCache.txt" goto :configure
goto :build

:configure
"%CHESHIRE_TOOLS%\cmake\bin\cmake.exe" -G Ninja -S "%SRC%" -B "%BLD%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_MAKE_PROGRAM=%CHESHIRE_TOOLS:\=/%/ninja/ninja.exe" ^
  "-DCMAKE_INSTALL_PREFIX=%INST%" ^
  "-DCMAKE_CUDA_COMPILER=%CHESHIRE_CUDA_PATH:\=/%/bin/nvcc.exe" ^
  "-DCMAKE_CUDA_HOST_COMPILER=%CHESHIRE_CL%" ^
  "-DCUDAToolkit_ROOT=%CHESHIRE_CUDA_PATH:\=/%" ^
  "-DCMAKE_CUDA_ARCHITECTURES=%CHESHIRE_CUDA_ARCHS%" ^
  "-DCMAKE_TOOLCHAIN_FILE=%V%/scripts/buildsystems/vcpkg.cmake" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows-release -DVCPKG_MANIFEST_MODE=OFF ^
  -DPopSift_BUILD_EXAMPLES=OFF -DPopSift_BUILD_DOCS=OFF ^
  -DBUILD_SHARED_LIBS=ON || exit /b 1
if /i "%STEP%"=="configure" exit /b 0

:build
"%CHESHIRE_TOOLS%\cmake\bin\cmake.exe" --build "%BLD%" || exit /b 1
if /i "%STEP%"=="build" exit /b 0
"%CHESHIRE_TOOLS%\cmake\bin\cmake.exe" --install "%BLD%" || exit /b 1

echo === installed
dir /b "%INST%\lib\cmake\PopSift" 2>nul
echo Now rebuild AliceVision with CHESHIRE_POPSIFT=ON:
echo   set CHESHIRE_POPSIFT=ON ^&^& scripts\windows\build-alicevision-cuda.cmd install
