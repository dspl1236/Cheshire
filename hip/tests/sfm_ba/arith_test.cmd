@echo off
rem Step 4b check (docs/notes/ba-own-solver.md): baArith.hpp against Ceres' and Eigen's formulas, built with the
rem AliceVision build's compiler and flags. arith_off must report 0 everywhere; arith_on shows what the Windows
rem build's FMA contraction changed. Output in build\ba-arith-test.
setlocal
call "%~dp0..\..\..\scripts\env.cmd" >nul 2>nul
set ROOT=%~dp0..\..\..
set CLX=%ROOT%\tools\venv-rocm\Lib\site-packages\_rocm_sdk_devel\lib\llvm\bin\clang-cl.exe
set VC=%ROOT%\tools\vcpkg-deps\x64-windows-release\installed\x64-windows-release
set INC=-imsvc%VC%\include\eigen3 -imsvc%VC%\include
set DEF=/DNDEBUG /DNOMINMAX /D_USE_MATH_DEFINES /DGLOG_NO_ABBREVIATED_SEVERITIES /DGLOG_USE_GLOG_EXPORT /DGLOG_USE_GFLAGS /DGFLAGS_IS_A_DLL=1
set OUT=%ROOT%\build\ba-arith-test
if not exist "%OUT%" mkdir "%OUT%"
"%CLX%" /nologo /O2 /Ob2 /arch:AVX2 /EHsc /std:c++17 /MD %DEF% %INC% /DARITH_NO_CONTRACT "%~dp0arith_test.cpp" /Fo"%OUT%\\" /Fe"%OUT%\arith_off.exe" /link "%VC%\lib\glog.lib" || exit /b 1
"%CLX%" /nologo /O2 /Ob2 /arch:AVX2 /EHsc /std:c++17 /MD %DEF% %INC% "%~dp0arith_test.cpp" /Fo"%OUT%\\" /Fe"%OUT%\arith_on.exe" /link "%VC%\lib\glog.lib" || exit /b 1
set PATH=%VC%\bin;%PATH%
"%OUT%\arith_off.exe" || exit /b 1
"%OUT%\arith_on.exe"
exit /b 0
