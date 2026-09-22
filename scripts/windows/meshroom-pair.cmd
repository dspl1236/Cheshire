@echo off
rem Cheshire: pair a Meshroom 2023.3 Windows install with a Cheshire AliceVision package.
rem
rem Meshroom runs its nodes as aliceVision_*.exe from <Meshroom>\aliceVision\bin. Seven of them get
rem the Cheshire treatment when the package carries them: PrepareDenseScene, FeatureExtraction,
rem FeatureMatching, DepthMap, DepthMapFilter, Meshing and Texturing. Each becomes a copy of the
rem launcher (meshroom-pair-launcher.exe, shipped beside this script) that starts the Cheshire build
rem from the package, or Meshroom's own binary (kept as <name>.cuda.exe) when nvidia-smi finds an
rem NVIDIA card. Decided per run, so a box that swaps cards needs no re-pairing; CHESHIRE_DEPTHMAP
rem forces one or the other. Every node the package does not carry keeps running from Meshroom.
rem --unpair restores the originals.
rem
rem   meshroom-pair.cmd <Meshroom dir> <Cheshire package dir>
rem   meshroom-pair.cmd <Meshroom dir> --unpair
rem
rem <Cheshire package dir> is an unzipped release zip, in either of the two shapes the launcher
rem knows (docs/16): a flat package with bin\, lib\, share\ at its root, or a bundle with
rem common\bin\, fam\<family>\bin\ and gpu\<family>\<target>\, which picks its payload per run.
rem Until v0.2.19 only the flat shape was recognised here, so the bundle - which since v0.2.17 is
rem the only Windows download there is - could not be paired at all, while the README said it could.
rem
rem A package without aliceVision_featureMatching.exe (v0.2.4 and older) pairs DepthMap only.
rem FeatureExtraction is paired when the package carries GPU SIFT (popsift.dll). With such a
rem package, leave Meshroom's forceCpuExtraction unticked; an older package has no GPU SIFT and the
rem node must keep forceCpuExtraction=True.
setlocal
set MR=%~1
set PKG=%~2
if "%PKG%"=="" ( echo usage: %~nx0 ^<Meshroom dir^> ^<Cheshire package dir^> ^| --unpair & exit /b 1 )
set BIN=%MR%\aliceVision\bin
if not exist "%BIN%\" ( echo %BIN% not found: is %MR% a Meshroom 2023.x Windows install? & exit /b 1 )
if /i "%PKG%"=="--unpair" (
  for %%N in (aliceVision_depthMapEstimation aliceVision_featureMatching aliceVision_featureExtraction aliceVision_depthMapFiltering aliceVision_meshing aliceVision_texturing aliceVision_prepareDenseScene aliceVision_incrementalSfM) do call :unpair %%N
  exit /b 0
)

rem Which shape is this? A bundle is the one with gpu\ and cheshire-run.cmd; its node binaries live
rem in fam\<family>\bin (or common\bin when it holds a single family), and its payload DLLs - popsift
rem among them - in gpu\<family>\<target>. The launcher composes that per run, so pairing only has to
rem find the node and write the package path; it must not assume bin\.
set BUNDLE=
if exist "%PKG%\gpu\" if exist "%PKG%\cheshire-run.cmd" set BUNDLE=1
call :have aliceVision_depthMapEstimation
if not defined HAVE ( echo no aliceVision_depthMapEstimation.exe in %PKG% ^(looked in bin\, common\bin\ and fam\*\bin\^) & exit /b 1 )

rem the launcher sits beside this script in a release zip, or in build\ in a checkout
set L=%~dp0meshroom-pair-launcher.exe
if not exist "%L%" set L=%~dp0..\..\build\meshroom-pair-launcher.exe
if not exist "%L%" ( echo meshroom-pair-launcher.exe missing: next to this script, or build\ ^(scripts\windows\build-launcher.cmd^) & exit /b 1 )

rem The version gates below probe a node's --help text, which means starting it, which in a flat
rem package needs its own DLLs in front of Meshroom's older AliceVision.
if not defined BUNDLE (
  set "PATH=%PKG%\bin;%PATH%"
  set "ALICEVISION_ROOT=%PKG%"
)

call :pair aliceVision_depthMapEstimation
rem only a package with the GPU matcher understands Meshroom 2023.3's --rangeStart/--rangeSize; older
rem packages carry the plain CPU featureMatching, which must not be put in Meshroom's way
call :gate aliceVision_featureMatching "--rangeStart" FMOK
if defined FMOK ( call :pair aliceVision_featureMatching ) else ( echo package's aliceVision_featureMatching has no GPU matcher ^(pre-v0.2.5^): DepthMap paired only )
rem DepthMapFilter (v0.2.6+): the package's depthMapFiltering carries the GPU vote pass; older packages
rem carry the plain CPU one, which is harmless but pointless, so gate on the newer help text
call :gate aliceVision_depthMapFiltering "CHESHIRE_GPU_FILTER" DFOK
if defined DFOK ( call :pair aliceVision_depthMapFiltering ) else ( echo package's aliceVision_depthMapFiltering has no GPU pass ^(pre-v0.2.6^): not paired )
rem Meshing (v0.2.7+): the package's meshing carries the GPU graph-weight votes; gate on its help text
call :gate aliceVision_meshing "CHESHIRE_GPU_VOTE" MSOK
if defined MSOK ( call :pair aliceVision_meshing ) else ( echo package's aliceVision_meshing has no GPU votes ^(pre-v0.2.7^): not paired )
rem Texturing (v0.2.8+): the package's texturing carries the GPU pyramid + rasterisation; gate on its help text
call :gate aliceVision_texturing "CHESHIRE_GPU_TEX" TXOK
if defined TXOK ( call :pair aliceVision_texturing ) else ( echo package's aliceVision_texturing has no GPU pass ^(pre-v0.2.8^): not paired )
rem PrepareDenseScene (v0.2.9+): the package's prepareDenseScene runs its image loop on every core; gate on its help text
call :gate aliceVision_prepareDenseScene "CHESHIRE_PDS_THREADS" PDOK
if defined PDOK ( call :pair aliceVision_prepareDenseScene ) else ( echo package's aliceVision_prepareDenseScene is upstream's ^(pre-v0.2.9^): not paired )
rem StructureFromMotion (v0.3.3+): the package's incrementalSfM finishes a resection pass with the bundle
rem adjustment upstream skips, the crash of Meshroom #2344 on large sets (docs\04); gate on its help text
call :gate aliceVision_incrementalSfM "CHESHIRE_SFM_PENDING_BA" SFOK
if defined SFOK ( call :pair aliceVision_incrementalSfM ) else ( echo package's aliceVision_incrementalSfM is upstream's ^(pre-v0.3.3^): not paired )
rem GPU SIFT (v0.2.13+, docs\14-gpu-sift.md): gate on popsift.dll being in the package, since
rem without it this node would move CPU SIFT from one build to another for nothing. Note the
rem describer falls back to the CPU silently when no GPU is visible, so a paired node that
rem still logs [cpu] means the runtime cannot see the card, not that pairing failed.
set FEOK=
call :have aliceVision_featureExtraction
if defined HAVE (
  if exist "%PKG%\bin\popsift.dll" set FEOK=1
  rem in a bundle popsift is per GPU target, under gpu\<family>\<target>\
  for /d %%F in ("%PKG%\gpu\*") do for /d %%T in ("%%~fF\*") do if exist "%%~fT\popsift.dll" set FEOK=1
)
if defined FEOK ( call :pair aliceVision_featureExtraction ) else ( echo package has no GPU SIFT ^(no popsift.dll^): featureExtraction not paired )
exit /b 0

:have
rem :have <node> -> HAVE=1 if the package carries that node, in either shape
set HAVE=
if exist "%PKG%\bin\%~1.exe" set HAVE=1
if not defined HAVE if exist "%PKG%\common\bin\%~1.exe" set HAVE=1
if not defined HAVE for /d %%F in ("%PKG%\fam\*") do if exist "%%~fF\bin\%~1.exe" set HAVE=1
exit /b 0

:gate
rem :gate <node> <string that must appear in --help> <result var>
rem A bundle postdates every one of these gates - the shape did not exist before v0.2.17 - so the
rem presence of the node is the answer, and we do not start it. That matters: a bundle's binaries
rem only load once cheshire-run.cmd has composed PATH from the selected GPU payload, so running one
rem directly for its --help would fail for reasons that have nothing to do with its version.
set "%~3="
call :have %~1
if not defined HAVE exit /b 0
if defined BUNDLE (
  set "%~3=1"
  exit /b 0
)
"%PKG%\bin\%~1.exe" --help > "%TEMP%\cheshire-gate.txt" 2>&1
findstr /c:"%~2" "%TEMP%\cheshire-gate.txt" >nul 2>&1 && set "%~3=1"
del /q "%TEMP%\cheshire-gate.txt" 2>nul
exit /b 0

:pair
set T=%BIN%\%1
if not exist "%T%.cuda.exe" ren "%T%.exe" %1.cuda.exe
copy /y "%L%" "%T%.exe" >nul
for %%P in ("%PKG%") do echo %%~fP> "%T%.cheshire.txt"
echo paired: %T%.exe -^> %1 from %PKG% (Meshroom's binary kept as %T%.cuda.exe)
exit /b 0

:unpair
set T=%BIN%\%1
if exist "%T%.cuda.exe" (
  del /q "%T%.exe" "%T%.cheshire.txt" 2>nul
  ren "%T%.cuda.exe" %1.exe
  echo restored %1.exe
) else ( echo %1: not paired )
exit /b 0
