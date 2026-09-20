@echo off
rem Full AliceVision build with the CUDA depth-map backend (Windows, MSVC, prebuilt vcpkg deps).
rem Usage: scripts\windows\build-alicevision-cuda.cmd [configure|build|install]   (default: build)
rem
rem Why MSVC and not clang-cl: nvcc on Windows drives cl.exe as its host compiler, and the HIP
rem build only uses clang-cl because ROCm requires it. With CUDA there is no reason for two
rem compilers, so this builds everything with cl.
rem
rem Why 14.44 and not the machine's own 14.50: CUDA 12.9's crt/host_config.h reads
rem   #if _MSC_VER < 1910 || _MSC_VER >= 1950
rem so it accepts MSVC 14.1x-14.4x. 14.50 (VS 2026) is exactly 1950 and is rejected with
rem "Only the versions between 2017 and 2022 (inclusive) are supported". Measured both ways,
rem docs/18. That is why VS 2022 Build Tools has to be installed alongside.
setlocal
set STEP=%~1
if "%STEP%"=="" set STEP=build

if not defined CHESHIRE_VS2022 set CHESHIRE_VS2022=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
if not defined CHESHIRE_MSVC_VER set CHESHIRE_MSVC_VER=14.44

rem ORDER MATTERS. env.cmd runs vcvarsall for VS 2026 whenever VCToolsInstallDir is unset, and
rem vcvars refuses to re-initialise a shell it has already configured - it returns 0 and changes
rem nothing. Calling env.cmd first therefore leaves 14.50 on PATH, nvcc picks it up and dies with
rem "Only the versions between 2017 and 2022 (inclusive) are supported". Initialise 2022 first;
rem env.cmd then sees VCToolsInstallDir already set and leaves it alone.
call "%CHESHIRE_VS2022%\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=%CHESHIRE_MSVC_VER% >nul
if errorlevel 1 ( echo vcvars64 for %CHESHIRE_MSVC_VER% failed & exit /b 1 )
call "%~dp0..\env.cmd"

rem Refuse to build with the wrong toolset rather than discover it at the first .cu.
set CLVER=
for /f "tokens=7" %%V in ('cl 2^>^&1 ^| findstr /i /c:"Version"') do set CLVER=%%V
echo [msvc] cl %CLVER%
echo %CLVER% | findstr /b /c:"19.4" >nul
if errorlevel 1 (
  echo.
  echo   cl is %CLVER%, which nvcc will reject ^(it needs _MSC_VER 1910-1949^).
  echo   Expected the %CHESHIRE_MSVC_VER% toolset from "%CHESHIRE_VS2022%".
  echo   Something re-initialised the environment after vcvars64 - check scripts\env.cmd.
  exit /b 1
)

set R=%CHESHIRE_ROOT:\=/%
set V=%R%/tools/vcpkg-deps/x64-windows-release
set BLD=%R%/build/av-cuda%CHESHIRE_BUILD_SUFFIX%
set INST=%R%/build/av-cuda%CHESHIRE_BUILD_SUFFIX%-install

if not defined CHESHIRE_CUDA_PATH set CHESHIRE_CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
rem sm_61 covers both NVIDIA test cards (GTX 1080 Ti, GTX 1050 Ti); beats upstream's FORCEd
rem "all-major" through patch step 5b, which would compile every .cu five times.
if not defined CHESHIRE_CUDA_ARCHS set CHESHIRE_CUDA_ARCHS=61

rem GPU SIFT: needs a CUDA PopSIFT (scripts\windows\build-popsift-cuda.cmd). Default OFF until
rem that exists, because a bundle without it silently loses GPU feature extraction.
if not defined CHESHIRE_POPSIFT set CHESHIRE_POPSIFT=OFF
if not defined CHESHIRE_POPSIFT_DIR set CHESHIRE_POPSIFT_DIR=%R%/build/popsift-cuda-install/lib/cmake/PopSift

if not exist "%CHESHIRE_CUDA_PATH%\bin\nvcc.exe" (
  echo no nvcc at "%CHESHIRE_CUDA_PATH%\bin\nvcc.exe" - set CHESHIRE_CUDA_PATH
  exit /b 1
)

"%CHESHIRE_CUDA_PATH%\bin\nvcc.exe" --version | findstr /i release

python "%CHESHIRE_ROOT%\scripts\apply_hip_patch.py" || exit /b 1

rem STL helper shim: the vcpkg archive was built with a newer MSVC STL that exports
rem __std_min/max_element_*i from msvcp140; 14.44 does not (nor does 14.50 - see
rem scripts\build-alicevision.cmd, which hits the same gap through clang-cl). Built with cl here.
set STLC=%R%/build/stlcompat-cuda
if not exist "%STLC%" mkdir "%CHESHIRE_ROOT%\build\stlcompat-cuda"
cl /nologo /O2 /MD /EHsc /std:c++17 /c "%CHESHIRE_ROOT%\hip\compat\stlcompat\std_minmax_element.cpp" /Fo"%CHESHIRE_ROOT%\build\stlcompat-cuda\std_minmax_element.obj" || exit /b 1
lib /nologo /out:"%CHESHIRE_ROOT%\build\stlcompat-cuda\stlcompat.lib" "%CHESHIRE_ROOT%\build\stlcompat-cuda\std_minmax_element.obj" || exit /b 1

cmake -S "%R%/third_party/aliceVision" -B "%BLD%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_EXE_LINKER_FLAGS=%STLC%/stlcompat.lib" ^
  "-DCMAKE_SHARED_LINKER_FLAGS=%STLC%/stlcompat.lib" ^
  "-DCMAKE_CUDA_COMPILER=%CHESHIRE_CUDA_PATH:\=/%/bin/nvcc.exe" ^
  "-DCUDAToolkit_ROOT=%CHESHIRE_CUDA_PATH:\=/%" ^
  "-DCMAKE_TOOLCHAIN_FILE=%V%/scripts/buildsystems/vcpkg.cmake" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows-release -DVCPKG_MANIFEST_MODE=OFF ^
  "-DCMAKE_INSTALL_PREFIX=%INST%" ^
  -DBUILD_SHARED_LIBS=ON -DTARGET_ARCHITECTURE=none ^
  -DALICEVISION_USE_CUDA=ON -DALICEVISION_USE_HIP=OFF -DALICEVISION_USE_SYCL=OFF ^
  -DALICEVISION_USE_POPSIFT=%CHESHIRE_POPSIFT% "-DPopSift_DIR=%CHESHIRE_POPSIFT_DIR%" ^
  -DALICEVISION_USE_ONNX_GPU=OFF -DALICEVISION_USE_CCTAG=OFF ^
  -DALICEVISION_USE_OPENCV=OFF -DALICEVISION_USE_APRILTAG=OFF -DALICEVISION_BUILD_TESTS=OFF ^
  -DALICEVISION_BUILD_DOC=OFF ^
  -DLEMON_LIBRARY=LEMON::lemon ^
  %CHESHIRE_CMAKE_EXTRA% ^
  || exit /b 1

echo === what the configure decided
findstr /b /c:"ALICEVISION_HAVE_CUDA" /c:"ALICEVISION_HAVE_HIP" /c:"CMAKE_CUDA_ARCHITECTURES" "%CHESHIRE_ROOT%\build\av-cuda%CHESHIRE_BUILD_SUFFIX%\CMakeCache.txt"

if "%STEP%"=="configure" exit /b 0
cmake --build "%BLD%" %CHESHIRE_BUILD_VERBOSE% -- -k 0 || exit /b 1
if "%STEP%"=="build" exit /b 0
cmake --install "%BLD%" || exit /b 1
