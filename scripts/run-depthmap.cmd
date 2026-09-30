@echo off
rem Run aliceVision_depthMapEstimation on a reference cache (from the CUDA node) with the exact
rem parameters Meshroom 2023.3 used, then compare against the CUDA depth maps.
rem Usage: scripts\run-depthmap.cmd [dataset]   (default monstree-mini6; expects data\ref\<dataset>\{StructureFromMotion,PrepareDenseScene,DepthMap})
rem The comparison needs Python 3 with numpy and OpenEXR: the build's venv if there is one, else py -3.
setlocal
set DS=%~1
if "%DS%"=="" set DS=monstree-mini6
rem CHESHIRE_DOWNSCALE: DepthMap downscale (Meshroom standard preset = 2; 1 = full-resolution depth maps, 4x the memory per tile)
set DSC=%CHESHIRE_DOWNSCALE%
if "%DSC%"=="" set DSC=2
set R=%~dp0..
for %%I in ("%R%") do set R=%%~fI
rem CHESHIRE_INSTALL: an unpacked release zip or a build install. A zip from v0.2.17 on is a bundle
rem (cheshire-run.cmd at its top): each node runs through cheshire-run.cmd, which asks the card for its
rem payload and composes PATH itself. A flat install (a build tree's, or an older zip) runs from bin\,
rem inside the build environment.
set INST=%CHESHIRE_INSTALL%
if "%INST%"=="" set INST=%R%\build\av-gfx1201-install
if exist "%INST%\cheshire-run.cmd" goto :bundle
call "%~dp0env.cmd"
set ALICEVISION_ROOT=%INST%
set PATH=%INST%\bin;%R%\tools\vcpkg-deps\x64-windows-release\installed\x64-windows-release\bin;%CHESHIRE_OMP_DLL_DIR%;%PATH%
set "HWR="%INST%\bin\aliceVision_hardwareResources.exe""
set "DME="%INST%\bin\aliceVision_depthMapEstimation.exe""
goto :cache
:bundle
set "HWR=call "%INST%\cheshire-run.cmd" aliceVision_hardwareResources"
set "DME=call "%INST%\cheshire-run.cmd" aliceVision_depthMapEstimation"
:cache
set REF=%R%\data\ref\%DS%
set OUT=%CHESHIRE_OUT%
if "%OUT%"=="" set OUT=%R%\data\out\%DS%-hip
for /d %%D in ("%REF%\StructureFromMotion\*") do set SFM=%%D\sfm.abc
for /d %%D in ("%REF%\PrepareDenseScene\*") do set IMGS=%%D
for /d %%D in ("%REF%\DepthMap\*") do set REFDM=%%D
if not exist "%OUT%" mkdir "%OUT%"

echo [cheshire] sfm=%SFM%
echo [cheshire] images=%IMGS%
echo [cheshire] out=%OUT%

%HWR% 2>&1 | findstr /i "name: memory HIP CUDA"

%DME% --input "%SFM%" --imagesFolder "%IMGS%" ^
  --downscale %DSC% --minViewAngle 2.0 --maxViewAngle 70.0 --tileBufferWidth 1024 --tileBufferHeight 1024 --tilePadding 64 ^
  --autoAdjustSmallImage True --chooseTCamsPerTile True --maxTCams 10 ^
  --sgmScale 2 --sgmStepXY 2 --sgmStepZ -1 --sgmMaxTCamsPerTile 4 --sgmWSH 4 --sgmUseSfmSeeds True --sgmSeedsRangeInflate 0.2 ^
  --sgmDepthThicknessInflate 0.0 --sgmMaxSimilarity 1.0 --sgmGammaC 5.5 --sgmGammaP 8.0 --sgmP1 10.0 --sgmP2Weighting 100.0 ^
  --sgmMaxDepths 1500 --sgmDepthListPerTile True --sgmUseConsistentScale False ^
  --refineEnabled True --refineScale 1 --refineStepXY 1 --refineMaxTCamsPerTile 4 --refineSubsampling 10 --refineHalfNbDepths 15 ^
  --refineWSH 3 --refineSigma 15.0 --refineGammaC 15.5 --refineGammaP 8.0 --refineInterpolateMiddleDepth False --refineUseConsistentScale False ^
  --colorOptimizationEnabled True --colorOptimizationNbIterations 100 --sgmUseCustomPatchPattern False --refineUseCustomPatchPattern False ^
  --nbGPUs 0 --verboseLevel info --output "%OUT%" %CHESHIRE_DEPTHMAP_EXTRA% || exit /b 1

set "PY="%R%\tools\venv-rocm\Scripts\python.exe""
if not exist "%R%\tools\venv-rocm\Scripts\python.exe" set PY=py -3
%PY% "%R%\scripts\compare_depthmaps.py" "%REFDM%" "%OUT%" --png "%OUT%\compare"
