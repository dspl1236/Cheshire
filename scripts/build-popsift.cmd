@echo off
rem Build PopSIFT (alicevision/popsift v0.10.0) as a HIP library for AMD, for GPU SIFT.
rem Usage: scripts\build-popsift.cmd [name] [configure|build|install]   (default: gfx1201 install)
rem
rem [name] only names the build and install directories. The code objects come from
rem CHESHIRE_HIP_ARCHS (a semicolon list, which cmd splits inside arguments so it cannot be passed
rem positionally); without it the name is used as the single architecture. Installs to
rem build\popsift-<name>-install, which is what CHESHIRE_POPSIFT_DIR should point into when
rem build-alicevision.cmd is then run with CHESHIRE_POPSIFT=ON.
rem
rem RDNA2 (gfx1030/1031/1032) needs the HIP SDK 6.2 toolchain instead, through CHESHIRE_ROCM_PATH /
rem CHESHIRE_LLVM_BIN / CHESHIRE_DEVICE_LIB_PATH, and one architecture per build: with several
rem --offload-arch values that toolchain writes an offload bundle whose entries are all the same
rem code object. See docs/01-toolchain-windows.md.
setlocal
call "%~dp0env.cmd"
set NAME=%~1
if "%NAME%"=="" set NAME=gfx1201
set STEP=%~2
if "%STEP%"=="" set STEP=install
set ARCHS=%NAME%
if defined CHESHIRE_HIP_ARCHS set ARCHS=%CHESHIRE_HIP_ARCHS%
set R=%CHESHIRE_ROOT:\=/%
set LLVMBIN=%ROCM_PATH%/lib/llvm/bin
if defined CHESHIRE_LLVM_BIN set LLVMBIN=%CHESHIRE_LLVM_BIN%
set BLD=%R%/build/popsift-%NAME%
set INST=%R%/build/popsift-%NAME%-install

rem ERRCHK checks after every kernel launch, which turns a silent device fault into a message naming
rem the kernel. Off for release builds; on when bringing up a new architecture.
set ERRCHK=OFF
if defined CHESHIRE_POPSIFT_ERRCHK set ERRCHK=%CHESHIRE_POPSIFT_ERRCHK%

rem CHESHIRE_EXTRA_CXXFLAGS reaches the host compiler through the environment, the way
rem build-alicevision.cmd does it, because CMake reads CFLAGS/CXXFLAGS only on the first configure.
rem The HIP SDK 6.2 toolchain needs -D__builtin_verbose_trap(x,y)=__builtin_trap() here and in
rem CHESHIRE_HIP_EXTRA_FLAGS: clang 19 against the MSVC 14.50 STL, and a force-included shim does
rem not reach HIP translation units, whose runtime wrapper includes the STL first.
set CFLAGS=%CHESHIRE_EXTRA_CXXFLAGS%
set CXXFLAGS=%CHESHIRE_EXTRA_CXXFLAGS%

rem the generated sift_config.h, the source fixes and the unity translation unit
python "%R%/scripts/apply_popsift_patch.py" || exit /b 1

for /f "delims=" %%I in ('dir /b /s "%WindowsSdkVerBinPath%x64\rc.exe" 2^>nul') do set RCEXE=%%I
if not defined RCEXE set RCEXE=C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe
for /f "delims=" %%I in ('dir /b /s "%WindowsSdkVerBinPath%x64\mt.exe" 2^>nul') do set MTEXE=%%I
if not defined MTEXE set MTEXE=C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/mt.exe

echo [popsift] name=%NAME% archs=%ARCHS% errchk=%ERRCHK%
echo [popsift] install=%INST%

if /i "%STEP%"=="configure" goto :configure
if not exist "%BLD%/CMakeCache.txt" goto :configure
rem Reconfigure when the requested architecture is not the one the cache holds. Only :configure
rem passes -DCHESHIRE_POPSIFT_ARCH, so without this a build that asks for a different target keeps
rem the cached one and still reports the requested value in the echo above - it says gfx12-generic
rem and ships gfx1201. Regenerating the unity source touches its timestamp, so the HIP object does
rem rebuild, which makes the wrong build look like a real one. Wrong architecture is precisely the
rem failure that returns wrong data with no error (docs/16).
set CACHED=
for /f "usebackq tokens=2 delims==" %%V in (`findstr /b /c:"CHESHIRE_POPSIFT_ARCH:STRING=" "%BLD%\CMakeCache.txt"`) do set CACHED=%%V
if /i not "%CACHED%"=="%ARCHS%" (
  echo [popsift] cache has %CACHED%, want %ARCHS% - reconfiguring
  goto :configure
)
goto :build

:configure
"%CHESHIRE_TOOLS%\cmake\bin\cmake.exe" -G Ninja -S "%R%/hip/port/popsift" -B "%BLD%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_MAKE_PROGRAM=%CHESHIRE_TOOLS:\=/%/ninja/ninja.exe" ^
  "-DCMAKE_INSTALL_PREFIX=%INST%" ^
  "-DCMAKE_C_COMPILER=%LLVMBIN%/clang-cl.exe" ^
  "-DCMAKE_CXX_COMPILER=%LLVMBIN%/clang-cl.exe" ^
  "-DCMAKE_HIP_COMPILER=%LLVMBIN%/clang-cl.exe" ^
  "-DCMAKE_HIP_FLAGS=--rocm-path=%ROCM_PATH% --rocm-device-lib-path=%HIP_DEVICE_LIB_PATH% %CHESHIRE_HIP_EXTRA_FLAGS%" ^
  "-DCHESHIRE_POPSIFT_ARCH=%ARCHS%" ^
  -DCHESHIRE_POPSIFT_ERRCHK=%ERRCHK% ^
  "-DCMAKE_RC_COMPILER=%RCEXE:\=/%" ^
  "-DCMAKE_MT=%MTEXE:\=/%" || exit /b 1
if /i "%STEP%"=="configure" exit /b 0

:build
set TARGET=
if /i "%STEP%"=="install" set TARGET=--target install
"%CHESHIRE_TOOLS%\cmake\bin\cmake.exe" --build "%BLD%" %TARGET% || exit /b 1
echo [popsift] done: %INST%
exit /b 0
