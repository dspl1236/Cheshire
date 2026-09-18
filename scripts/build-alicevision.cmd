@echo off
rem Full AliceVision build with the HIP depth-map backend (Windows, clang-cl, prebuilt vcpkg deps).
rem Usage: scripts\build-alicevision.cmd [gfxArch] [configure|build|install]   (default: gfx1201 build)
setlocal
call "%~dp0env.cmd"
set ARCH=%~1
if "%ARCH%"=="" set ARCH=gfx1201
set STEP=%~2
if "%STEP%"=="" set STEP=build
rem CHESHIRE_HIP_ARCHS: semicolon list of code objects to build (cmd splits ";" in arguments, so it
rem cannot be passed as %1); ARCH then only names the build directory.
set ARCHS=%ARCH%
if defined CHESHIRE_HIP_ARCHS set ARCHS=%CHESHIRE_HIP_ARCHS%
set R=%CHESHIRE_ROOT:\=/%
rem GPU SIFT: CHESHIRE_POPSIFT=ON with a HIP popsift build (see hip/port/popsift)
if not defined CHESHIRE_POPSIFT set CHESHIRE_POPSIFT=OFF
if not defined CHESHIRE_POPSIFT_DIR set CHESHIRE_POPSIFT_DIR=%R%/build/popsift-install/lib/cmake/PopSift
set LLVMBIN=%ROCM_PATH%/lib/llvm/bin
if defined CHESHIRE_LLVM_BIN set LLVMBIN=%CHESHIRE_LLVM_BIN%
set V=%R%/tools/vcpkg-deps/x64-windows-release
rem CHESHIRE_BUILD_SUFFIX: keep variants side by side (e.g. -emu for the mipmap-emulation build)
set BLD=%R%/build/av-%ARCH%%CHESHIRE_BUILD_SUFFIX%
set INST=%R%/build/av-%ARCH%%CHESHIRE_BUILD_SUFFIX%-install
rem CHESHIRE_MIPMAP_NATIVE=0: emulated mip levels (array or linear storage, CHESHIRE_MIPMAP_STORAGE at run time)
set MIPFLAG=-DCHESHIRE_NATIVE_MIPMAP
if "%CHESHIRE_MIPMAP_NATIVE%"=="0" set MIPFLAG=

python "%R%/scripts/apply_hip_patch.py" || exit /b 1

rem OpenMP 3+ omp.h shim (see hip/compat/include/omp_shim). Via the environment so CMake's
rem own MSVC defaults (/EHsc /DWIN32 ...) stay intact; -DCMAKE_CXX_FLAGS would replace them.
rem /arch:AVX2: AliceVision's OptimizeForArchitecture (TARGET_ARCHITECTURE=core) emits /arch:SSE2 (ignored by
rem clang-cl on x64) while defining __SSE3__ etc., so Eigen picks SSE3 intrinsics the compiler will not inline.
rem CHESHIRE_ARCH_FLAG: /arch:AVX2 (default; Haswell 2013 and newer) or /arch:AVX for older CPUs
set ARCHFLAG=%CHESHIRE_ARCH_FLAG%
if "%ARCHFLAG%"=="" set ARCHFLAG=/arch:AVX2
rem CHESHIRE_EXTRA_CXXFLAGS: appended to the host compiler flags (e.g. a /FI shim for an older clang)
set CFLAGS=-I%R%/hip/compat/include/omp_shim %ARCHFLAG% %CHESHIRE_EXTRA_CXXFLAGS%
set CXXFLAGS=-I%R%/hip/compat/include/omp_shim %ARCHFLAG% %CHESHIRE_EXTRA_CXXFLAGS%

rem STL helper shim: the vcpkg archive was built with a newer MSVC STL that exports
rem __std_min/max_element_*i from msvcp140; MSVC 14.50.35717 does not. See hip/compat/stlcompat.
set STLC=%R%/build/stlcompat
if not exist "%STLC%" mkdir "%STLC%"
"%LLVMBIN%/clang-cl.exe" /nologo /O2 /MD /c "%R%/hip/compat/stlcompat/std_minmax_element.cpp" /Fo"%STLC%/std_minmax_element.obj" || exit /b 1
rem HIP SDK 6.2 ships no llvm-lib; any archiver works for one object, use the 7.2.1 wheel one
set LLVMLIB=%LLVMBIN%/llvm-lib.exe
if not exist "%LLVMLIB%" set LLVMLIB=%R%/tools/venv-rocm/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/llvm-lib.exe
"%LLVMLIB%" /nologo /out:"%STLC%/stlcompat.lib" "%STLC%/std_minmax_element.obj" || exit /b 1

cmake -S "%R%/third_party/aliceVision" -B "%BLD%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_EXE_LINKER_FLAGS=%STLC%/stlcompat.lib" ^
  "-DCMAKE_SHARED_LINKER_FLAGS=%STLC%/stlcompat.lib" ^
  "-DCMAKE_C_COMPILER=%LLVMBIN%/clang-cl.exe" ^
  "-DCMAKE_CXX_COMPILER=%LLVMBIN%/clang-cl.exe" ^
  "-DCMAKE_HIP_COMPILER=%LLVMBIN%/clang-cl.exe" ^
  "-DCMAKE_HIP_ARCHITECTURES=%ARCHS%" ^
  "-DCMAKE_HIP_FLAGS=--rocm-path=%ROCM_PATH% --rocm-device-lib-path=%HIP_DEVICE_LIB_PATH% %MIPFLAG% %CHESHIRE_HIP_EXTRA_FLAGS%" ^
  "-DCMAKE_PREFIX_PATH=%ROCM_PATH%" ^
  "-DCMAKE_TOOLCHAIN_FILE=%V%/scripts/buildsystems/vcpkg.cmake" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows-release -DVCPKG_MANIFEST_MODE=OFF ^
  "-DCMAKE_INSTALL_PREFIX=%INST%" ^
  -DBUILD_SHARED_LIBS=ON -DTARGET_ARCHITECTURE=none ^
  -DALICEVISION_USE_CUDA=OFF -DALICEVISION_USE_HIP=ON -DALICEVISION_USE_SYCL=OFF ^
  -DALICEVISION_USE_POPSIFT=%CHESHIRE_POPSIFT% "-DPopSift_DIR=%CHESHIRE_POPSIFT_DIR%" -DALICEVISION_USE_ONNX_GPU=OFF -DALICEVISION_USE_CCTAG=OFF ^
  -DALICEVISION_USE_OPENCV=OFF -DALICEVISION_USE_APRILTAG=OFF -DALICEVISION_BUILD_TESTS=OFF ^
  -DALICEVISION_BUILD_DOC=OFF ^
  -DLEMON_LIBRARY=LEMON::lemon ^
  %CHESHIRE_CMAKE_EXTRA% ^
  || exit /b 1
if "%STEP%"=="configure" exit /b 0
cmake --build "%BLD%" %CHESHIRE_BUILD_VERBOSE% -- -k 0 || exit /b 1
if "%STEP%"=="build" exit /b 0
cmake --install "%BLD%" || exit /b 1
