@echo off
rem Cheshire: run the HIP DepthMap from an unpacked release zip on any Windows box, no repo needed.
rem Usage: run-depthmap-standalone.cmd <unzipped package dir> <Meshroom cache dir> <out dir>
rem   cache dir must contain StructureFromMotion\<id>\sfm.abc and PrepareDenseScene\<id>\
rem   Same DepthMap parameters as Meshroom 2023.3's "standard" preset (downscale 2).
rem   CHESHIRE_DEPTHMAP_EXTRA adds arguments (e.g. --rangeStart 0 --rangeSize 1).
setlocal
set INST=%~1
set CACHE=%~2
set OUT=%~3
rem CHESHIRE_DOWNSCALE: DepthMap downscale (Meshroom standard preset = 2; 1 = full-resolution depth maps, 4x the memory per tile)
set DS=%CHESHIRE_DOWNSCALE%
if "%DS%"=="" set DS=2
if "%OUT%"=="" ( echo usage: %~nx0 ^<package dir^> ^<cache dir^> ^<out dir^> & exit /b 1 )
for /d %%D in ("%CACHE%\StructureFromMotion\*") do set SFM=%%D\sfm.abc
for /d %%D in ("%CACHE%\PrepareDenseScene\*") do set IMGS=%%D
if not exist "%OUT%" mkdir "%OUT%"
set ALICEVISION_ROOT=%INST%
set PATH=%INST%\bin;%PATH%
"%INST%\bin\aliceVision_hardwareResources.exe" -v info 2>&1 | findstr /i "name: memory CUDA"
"%INST%\bin\aliceVision_depthMapEstimation.exe" --input "%SFM%" --imagesFolder "%IMGS%" ^
  --downscale %DS% --minViewAngle 2.0 --maxViewAngle 70.0 --tileBufferWidth 1024 --tileBufferHeight 1024 --tilePadding 64 ^
  --autoAdjustSmallImage True --chooseTCamsPerTile True --maxTCams 10 ^
  --sgmScale 2 --sgmStepXY 2 --sgmStepZ -1 --sgmMaxTCamsPerTile 4 --sgmWSH 4 --sgmUseSfmSeeds True --sgmSeedsRangeInflate 0.2 ^
  --sgmDepthThicknessInflate 0.0 --sgmMaxSimilarity 1.0 --sgmGammaC 5.5 --sgmGammaP 8.0 --sgmP1 10.0 --sgmP2Weighting 100.0 ^
  --sgmMaxDepths 1500 --sgmDepthListPerTile True --sgmUseConsistentScale False ^
  --refineEnabled True --refineScale 1 --refineStepXY 1 --refineMaxTCamsPerTile 4 --refineSubsampling 10 --refineHalfNbDepths 15 ^
  --refineWSH 3 --refineSigma 15.0 --refineGammaC 15.5 --refineGammaP 8.0 --refineInterpolateMiddleDepth False --refineUseConsistentScale False ^
  --colorOptimizationEnabled True --colorOptimizationNbIterations 100 --sgmUseCustomPatchPattern False --refineUseCustomPatchPattern False ^
  --nbGPUs 0 --verboseLevel info --output "%OUT%" %CHESHIRE_DEPTHMAP_EXTRA%
