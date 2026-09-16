@echo off
rem Cheshire: pair a Meshroom 2023.3 Windows install with a Cheshire HIP AliceVision package.
rem
rem Meshroom runs its nodes as aliceVision_*.exe from <Meshroom>\aliceVision\bin. Only DepthMap
rem needs the GPU, so the pairing is one file: that binary becomes a launcher (meshroom-pair-launcher.exe,
rem shipped beside this script) that starts the HIP build from the Cheshire package, or the original CUDA
rem binary (kept as aliceVision_depthMapEstimation.cuda.exe) when nvidia-smi finds an NVIDIA card. Decided
rem per run, so a box that swaps cards needs no re-pairing. Everything else (SfM, meshing, texturing) keeps
rem running from the Meshroom install unchanged. --unpair restores the original.
rem
rem   meshroom-pair.cmd <Meshroom dir> <Cheshire package dir>
rem   meshroom-pair.cmd <Meshroom dir> --unpair
rem
rem <Cheshire package dir> is an unzipped release zip (the folder holding bin\, lib\, share\).
rem On AMD, Meshroom's FeatureExtraction node must run with forceCpuExtraction=True (PopSift is CUDA-only).
setlocal
set MR=%~1
set PKG=%~2
if "%PKG%"=="" ( echo usage: %~nx0 ^<Meshroom dir^> ^<Cheshire package dir^> ^| --unpair & exit /b 1 )
set BIN=%MR%\aliceVision\bin
set T=%BIN%\aliceVision_depthMapEstimation
if not exist "%BIN%\" ( echo %BIN% not found: is %MR% a Meshroom 2023.x Windows install? & exit /b 1 )
if /i "%PKG%"=="--unpair" (
  if exist "%T%.cuda.exe" (
    del /q "%T%.exe" "%T%.cheshire.txt" 2>nul
    ren "%T%.cuda.exe" aliceVision_depthMapEstimation.exe
    echo restored CUDA aliceVision_depthMapEstimation.exe
  ) else ( echo not paired )
  exit /b 0
)
if not exist "%PKG%\bin\aliceVision_depthMapEstimation.exe" ( echo no HIP aliceVision_depthMapEstimation.exe in %PKG%\bin & exit /b 1 )
rem the launcher sits beside this script in a release zip, or in build\ in a checkout
set L=%~dp0meshroom-pair-launcher.exe
if not exist "%L%" set L=%~dp0..\..\build\meshroom-pair-launcher.exe
if not exist "%L%" ( echo meshroom-pair-launcher.exe missing: next to this script, or build\ ^(scripts\windows\build-launcher.cmd^) & exit /b 1 )
if not exist "%T%.cuda.exe" ren "%T%.exe" aliceVision_depthMapEstimation.cuda.exe
copy /y "%L%" "%T%.exe" >nul
for %%P in ("%PKG%") do echo %%~fP> "%T%.cheshire.txt"
echo paired: %T%.exe -^> %PKG%\bin\aliceVision_depthMapEstimation.exe (CUDA binary kept as %T%.cuda.exe)
