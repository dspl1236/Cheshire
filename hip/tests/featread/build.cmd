@echo off
rem Build hip/tests/featread/featbench.cpp with the build's toolchain (clang-cl, MSVC STL), then e.g.
rem   featbench.exe build\e2e-falsedoor-fix\base\cache\FeatureExtraction\<hash> 12
rem Linux: g++ -O2 -std=c++17 -pthread featbench.cpp -o featbench
setlocal
call "%~dp0..\..\..\scripts\env.cmd" >nul 2>&1
cd /d %~dp0
"%ROCM_PATH%/lib/llvm/bin/clang-cl.exe" /nologo /O2 /EHsc /std:c++17 /MD featbench.cpp /Fefeatbench.exe || exit /b 1
echo built %~dp0featbench.exe
