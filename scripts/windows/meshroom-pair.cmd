@echo off
rem Cheshire: pair a Meshroom 2023.3 Windows install with a Cheshire HIP AliceVision package.
rem
rem Meshroom runs its nodes as aliceVision_*.exe from <Meshroom>\aliceVision\bin. Two of them get the
rem GPU treatment: DepthMap (the HIP port) and FeatureMatching (the GPU descriptor matcher). Each
rem becomes a copy of the launcher (meshroom-pair-launcher.exe, shipped beside this script) that starts
rem the Cheshire build from the package, or Meshroom's own binary (kept as <name>.cuda.exe) when
rem nvidia-smi finds an NVIDIA card. Decided per run, so a box that swaps cards needs no re-pairing.
rem Everything else (SfM, meshing, texturing) keeps running from the Meshroom install unchanged.
rem --unpair restores the originals.
rem
rem   meshroom-pair.cmd <Meshroom dir> <Cheshire package dir>
rem   meshroom-pair.cmd <Meshroom dir> --unpair
rem
rem <Cheshire package dir> is an unzipped release zip (the folder holding bin\, lib\, share\). A package
rem without aliceVision_featureMatching.exe (v0.2.4 and older) pairs DepthMap only.
rem On AMD, Meshroom's FeatureExtraction node must run with forceCpuExtraction=True (PopSift is CUDA-only).
setlocal
set MR=%~1
set PKG=%~2
if "%PKG%"=="" ( echo usage: %~nx0 ^<Meshroom dir^> ^<Cheshire package dir^> ^| --unpair & exit /b 1 )
set BIN=%MR%\aliceVision\bin
if not exist "%BIN%\" ( echo %BIN% not found: is %MR% a Meshroom 2023.x Windows install? & exit /b 1 )
if /i "%PKG%"=="--unpair" (
  for %%N in (aliceVision_depthMapEstimation aliceVision_featureMatching) do call :unpair %%N
  exit /b 0
)
if not exist "%PKG%\bin\aliceVision_depthMapEstimation.exe" ( echo no HIP aliceVision_depthMapEstimation.exe in %PKG%\bin & exit /b 1 )
rem the launcher sits beside this script in a release zip, or in build\ in a checkout
set L=%~dp0meshroom-pair-launcher.exe
if not exist "%L%" set L=%~dp0..\..\build\meshroom-pair-launcher.exe
if not exist "%L%" ( echo meshroom-pair-launcher.exe missing: next to this script, or build\ ^(scripts\windows\build-launcher.cmd^) & exit /b 1 )
call :pair aliceVision_depthMapEstimation
rem only a package with the GPU matcher understands Meshroom 2023.3's --rangeStart/--rangeSize; older
rem packages carry the plain CPU featureMatching, which must not be put in Meshroom's way
set FMOK=
if exist "%PKG%\bin\aliceVision_featureMatching.exe" (
  for /f %%X in ('"%PKG%\bin\aliceVision_featureMatching.exe" --help 2^>^&1 ^| findstr /c:"--rangeStart"') do set FMOK=1
)
if defined FMOK ( call :pair aliceVision_featureMatching ) else ( echo package's aliceVision_featureMatching has no GPU matcher ^(pre-v0.2.5^): DepthMap paired only )
exit /b 0

:pair
set T=%BIN%\%1
if not exist "%T%.cuda.exe" ren "%T%.exe" %1.cuda.exe
copy /y "%L%" "%T%.exe" >nul
for %%P in ("%PKG%") do echo %%~fP> "%T%.cheshire.txt"
echo paired: %T%.exe -^> %PKG%\bin\%1.exe (Meshroom's binary kept as %T%.cuda.exe)
exit /b 0

:unpair
set T=%BIN%\%1
if exist "%T%.cuda.exe" (
  del /q "%T%.exe" "%T%.cheshire.txt" 2>nul
  ren "%T%.cuda.exe" %1.exe
  echo restored %1.exe
) else ( echo %1: not paired )
exit /b 0
