@echo off
rem Run the HIP-built aliceVision_depthMapEstimation on a reference cache (from the CUDA node)
rem with the exact parameters Meshroom 2023.3 used, then compare against the CUDA depth maps.
rem Usage: scripts\run-depthmap.cmd [dataset]   (default monstree-mini6; expects data\ref\<dataset>\{StructureFromMotion,PrepareDenseScene,DepthMap})
setlocal
call "%~dp0env.cmd"
set DS=%~1
if "%DS%"=="" set DS=monstree-mini6
rem CHESHIRE_DOWNSCALE: DepthMap downscale (Meshroom standard preset = 2; 1 = full-resolution depth maps, 4x the memory per tile)
set DSC=%CHESHIRE_DOWNSCALE%
if "%DSC%"=="" set DSC=2
set R=%CHESHIRE_ROOT%
rem CHESHIRE_INSTALL: an unpacked release zip (self-contained) or another build install
set INST=%CHESHIRE_INSTALL%
if "%INST%"=="" set INST=%R%\build\av-gfx1201-install
set REF=%R%\data\ref\%DS%
set OUT=%CHESHIRE_OUT%
if "%OUT%"=="" set OUT=%R%\data\out\%DS%-hip
for /d %%D in ("%REF%\StructureFromMotion\*") do set SFM=%%D\sfm.abc
for /d %%D in ("%REF%\PrepareDenseScene\*") do set IMGS=%%D
for /d %%D in ("%REF%\DepthMap\*") do set REFDM=%%D
if not exist "%OUT%" mkdir "%OUT%"

set ALICEVISION_ROOT=%INST%
set PATH=%INST%\bin;%R%\tools\vcpkg-deps\x64-windows-release\installed\x64-windows-release\bin;%CHESHIRE_OMP_DLL_DIR%;%PATH%
echo [cheshire] sfm=%SFM%
echo [cheshire] images=%IMGS%
echo [cheshire] out=%OUT%

"%INST%\bin\aliceVision_hardwareResources.exe" 2>&1 | findstr /i "name: memory HIP CUDA"

"%INST%\bin\aliceVision_depthMapEstimation.exe" --input "%SFM%" --imagesFolder "%IMGS%" ^
  --downscale %DSC% --minViewAngle 2.0 --maxViewAngle 70.0 --tileBufferWidth 1024 --tileBufferHeight 1024 --tilePadding 64 ^
  --autoAdjustSmallImage True --chooseTCamsPerTile True --maxTCams 10 ^
  --sgmScale 2 --sgmStepXY 2 --sgmStepZ -1 --sgmMaxTCamsPerTile 4 --sgmWSH 4 --sgmUseSfmSeeds True --sgmSeedsRangeInflate 0.2 ^
  --sgmDepthThicknessInflate 0.0 --sgmMaxSimilarity 1.0 --sgmGammaC 5.5 --sgmGammaP 8.0 --sgmP1 10.0 --sgmP2Weighting 100.0 ^
  --sgmMaxDepths 1500 --sgmDepthListPerTile True --sgmUseConsistentScale False ^
  --refineEnabled True --refineScale 1 --refineStepXY 1 --refineMaxTCamsPerTile 4 --refineSubsampling 10 --refineHalfNbDepths 15 ^
  --refineWSH 3 --refineSigma 15.0 --refineGammaC 15.5 --refineGammaP 8.0 --refineInterpolateMiddleDepth False --refineUseConsistentScale False ^
  --colorOptimizationEnabled True --colorOptimizationNbIterations 100 --sgmUseCustomPatchPattern False --refineUseCustomPatchPattern False ^
  --nbGPUs 0 --verboseLevel info --output "%OUT%" %CHESHIRE_DEPTHMAP_EXTRA% || exit /b 1

"%R%\tools\venv-rocm\Scripts\python.exe" "%R%\scripts\compare_depthmaps.py" "%REFDM%" "%OUT%" --png "%OUT%\compare"
