@echo off
rem Cheshire: build the Meshroom pairing launcher (scripts\windows\meshroom-pair-launcher.cpp) with the
rem toolchain's clang-cl. Output: build\meshroom-pair-launcher.exe (no runtime DLLs: static CRT).
setlocal
call "%~dp0..\env.cmd" >nul 2>&1
set R=%CHESHIRE_ROOT%
set LLVMBIN=%ROCM_PATH%/lib/llvm/bin
if defined CHESHIRE_LLVM_BIN set LLVMBIN=%CHESHIRE_LLVM_BIN%
if not exist "%R%\build" mkdir "%R%\build"
"%LLVMBIN%/clang-cl.exe" /nologo /O2 /EHsc /W3 /MT "%R%\scripts\windows\meshroom-pair-launcher.cpp" /Fo"%R%\build\meshroom-pair-launcher.obj" /Fe"%R%\build\meshroom-pair-launcher.exe" /link shell32.lib kernel32.lib || exit /b 1
echo built %R%\build\meshroom-pair-launcher.exe

rem The bundle's card probe: picks the runtime family and GPU target for the layered package (docs/16).
"%LLVMBIN%/clang-cl.exe" /nologo /O2 /EHsc /W3 /MT "%R%\scripts\windows\cheshire-detect.cpp" /Fo"%R%\build\cheshire-detect.obj" /Fe"%R%\build\cheshire-detect.exe" /link kernel32.lib || exit /b 1
echo built %R%\build\cheshire-detect.exe
