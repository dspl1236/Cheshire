@echo off
rem Usage: scripts\windowsuild-ceres-nosuitesparse.cmd   (source: tools\dl\ceres-solver-2.2.0.tar.gz extracted to build\ceres-solver-2.2.0)
rem then:  python scripts\windows\splice_ceres.py   (puts it into toolscpkg-deps, backup in build\ceres-vcpkg-backup)
rem Ceres 2.2.0 without SuiteSparse (GPL: CHOLMOD, SPQR), for the Windows packages. Same MSVC, same
rem Eigen 3.4.1 / glog 0.7.1 / gflags 2.3.0 / OpenBLAS LAPACK as the prebuilt vcpkg tree, Eigen's sparse
rem backend with METIS ordering (Apache 2) instead. Installs to build\ceres-nosuitesparse-install; the
rem splice into tools\vcpkg-deps is a separate step.
setlocal
call "%~dp0..\scripts\env.cmd"
set R=%CHESHIRE_ROOT:\=/%
set V=%R%/tools/vcpkg-deps/x64-windows-release
set SRC=%R%/build/ceres-solver-2.2.0
set BLD=%R%/build/ceres-nosuitesparse
set INST=%R%/build/ceres-nosuitesparse-install
if not exist "%SRC%\CMakeLists.txt" ( echo no source at %SRC% & exit /b 1 )
echo === configure %date% %time%
cmake -S "%SRC%" -B "%BLD%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_C_COMPILER=%CHESHIRE_CL%" "-DCMAKE_CXX_COMPILER=%CHESHIRE_CL%" ^
  "-DCMAKE_TOOLCHAIN_FILE=%V%/scripts/buildsystems/vcpkg.cmake" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows-release -DVCPKG_MANIFEST_MODE=OFF ^
  "-DCMAKE_INSTALL_PREFIX=%INST%" ^
  -DBUILD_SHARED_LIBS=ON -DSUITESPARSE=OFF -DACCELERATESPARSE=OFF -DEIGENSPARSE=ON -DEIGENMETIS=ON ^
  -DLAPACK=ON -DGFLAGS=ON -DMINIGLOG=OFF -DSCHUR_SPECIALIZATIONS=ON -DUSE_CUDA=OFF ^
  -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_DOCUMENTATION=OFF ^
  -DPROVIDE_UNINSTALL_TARGET=OFF || exit /b 1
echo === build %date% %time%
cmake --build "%BLD%" -- -k 0 || exit /b 1
echo === install %date% %time%
cmake --install "%BLD%" || exit /b 1
echo === done %date% %time%
