@echo off
rem Cheshire: pair a Meshroom 2023.3 or 2025.1 Windows install with a Cheshire AliceVision package.
rem
rem Meshroom runs its nodes as aliceVision_*.exe from <Meshroom>\aliceVision\bin. Nine of them get
rem the Cheshire treatment when the package carries them: PrepareDenseScene, FeatureExtraction,
rem ImageMatching, FeatureMatching, StructureFromMotion, DepthMap, DepthMapFilter, Meshing and
rem Texturing. Each becomes a copy of the launcher (meshroom-pair-launcher.exe, shipped beside this
rem script) that starts the Cheshire build from the package when the package matches the card (a
rem CUDA package with an NVIDIA card that answers nvidia-smi, an AMD package without one), or
rem Meshroom's own binary (kept as <name>.cuda.exe) otherwise. Decided per run, so a box that swaps
rem cards needs no re-pairing; CHESHIRE_BACKEND=auto|cheshire|meshroom forces one or the other.
rem Every node the package does not carry keeps running from Meshroom.
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
rem
rem Before each node is paired, meshroom-pair-check.ps1 (beside this script) compares the options
rem Meshroom's own binary takes with the package's: a Meshroom newer than the package passes options
rem the package does not know, and the node is then left to Meshroom instead of failing mid-job.
setlocal
set HERE=%~dp0
set MR=%~1
set PKG=%~2
if "%PKG%"=="" ( echo usage: %~nx0 ^<Meshroom dir^> ^<Cheshire package dir^> ^| --unpair & exit /b 1 )
set BIN=%MR%\aliceVision\bin
if not exist "%BIN%\" ( echo %BIN% not found: is %MR% a Meshroom 2023.3 or 2025.1 Windows install? & exit /b 1 )
if /i "%PKG%"=="--unpair" (
  if exist "%MR%\lib\meshroom\nodes\aliceVision\DepthMap.py" ( findstr /c:"Cheshire" "%MR%\lib\meshroom\nodes\aliceVision\DepthMap.py" >nul && del /q "%MR%\lib\meshroom\nodes\aliceVision\DepthMap.py" && echo removed the DepthMap node override )
  if exist "%MR%\aliceVision\share\meshroom\aliceVision\DepthMap.py.meshroom" ( findstr /c:"Cheshire" "%MR%\aliceVision\share\meshroom\aliceVision\DepthMap.py" >nul && move /y "%MR%\aliceVision\share\meshroom\aliceVision\DepthMap.py.meshroom" "%MR%\aliceVision\share\meshroom\aliceVision\DepthMap.py" >nul && echo restored Meshroom's DepthMap node )
  for %%N in (aliceVision_depthMapEstimation aliceVision_featureMatching aliceVision_featureExtraction aliceVision_depthMapFiltering aliceVision_meshing aliceVision_texturing aliceVision_prepareDenseScene aliceVision_incrementalSfM aliceVision_imageMatching) do call :unpair %%N
  if exist "%HERE%meshroom-templates.ps1" powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%meshroom-templates.ps1" "%MR%" remove
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

call :pairif aliceVision_depthMapEstimation sgmFilteringAxes
rem only a package with the GPU matcher understands Meshroom 2023.3's --rangeStart/--rangeSize; older
rem packages carry the plain CPU featureMatching, which must not be put in Meshroom's way
call :gate aliceVision_featureMatching "--rangeStart" FMOK
if defined FMOK ( call :pairif aliceVision_featureMatching ) else ( echo package's aliceVision_featureMatching has no GPU matcher ^(pre-v0.2.5^): DepthMap paired only )
rem DepthMapFilter (v0.2.6+): the package's depthMapFiltering carries the GPU vote pass; older packages
rem carry the plain CPU one, which is harmless but pointless, so gate on the newer help text
call :gate aliceVision_depthMapFiltering "CHESHIRE_GPU_FILTER" DFOK
if defined DFOK ( call :pairif aliceVision_depthMapFiltering ) else ( echo package's aliceVision_depthMapFiltering has no GPU pass ^(pre-v0.2.6^): not paired )
rem Meshing (v0.2.7+): the package's meshing carries the GPU graph-weight votes; gate on its help text
call :gate aliceVision_meshing "CHESHIRE_GPU_VOTE" MSOK
if defined MSOK ( call :pairif aliceVision_meshing ) else ( echo package's aliceVision_meshing has no GPU votes ^(pre-v0.2.7^): not paired )
rem Texturing (v0.2.8+): the package's texturing carries the GPU pyramid + rasterisation; gate on its help text
call :gate aliceVision_texturing "CHESHIRE_GPU_TEX" TXOK
if defined TXOK ( call :pairif aliceVision_texturing ) else ( echo package's aliceVision_texturing has no GPU pass ^(pre-v0.2.8^): not paired )
rem PrepareDenseScene (v0.2.9+): the package's prepareDenseScene runs its image loop on every core; gate on its help text
call :gate aliceVision_prepareDenseScene "CHESHIRE_PDS_THREADS" PDOK
if defined PDOK ( call :pairif aliceVision_prepareDenseScene ) else ( echo package's aliceVision_prepareDenseScene is upstream's ^(pre-v0.2.9^): not paired )
rem StructureFromMotion (v0.3.3+): the package's incrementalSfM finishes a resection pass with the bundle
rem adjustment upstream skips, the crash of Meshroom #2344 on large sets (docs\04); gate on its help text
call :gate aliceVision_incrementalSfM "CHESHIRE_SFM_PENDING_BA" SFOK
if defined SFOK ( call :pairif aliceVision_incrementalSfM ) else ( echo package's aliceVision_incrementalSfM is upstream's ^(pre-v0.3.3^): not paired )
rem ImageMatching (v0.3.5+): the package's imageMatching can pair views by GPS distance
rem (CHESHIRE_GPS_PAIRING_RADIUS, off unless set); gate on its help text
call :gate aliceVision_imageMatching "CHESHIRE_GPS_PAIRING" IMOK
if defined IMOK ( call :pairif aliceVision_imageMatching ) else ( echo package's aliceVision_imageMatching is upstream's ^(pre-v0.3.5^): not paired )
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
if defined FEOK ( call :pairif aliceVision_featureExtraction ) else ( echo package has no GPU SIFT ^(no popsift.dll^): featureExtraction not paired )
rem Meshroom 2025.1 and later (0.3.9): the "Photogrammetry Fast Ransac" and "Photogrammetry Draft Fast Ransac"
rem pipeline templates, the installed Photogrammetry and Draft templates with Meshroom 2023.3's two RANSAC
rem counts set in the graph (FeatureMatching maxIteration 2048, and localizerEstimatorMaxIterations 4096 on
rem StructureFromMotion, or on SfMExpanding in the new pipeline; 2025.1's defaults are 50000). Meshroom's own
rem templates are not touched, so the paired nodes keep its defaults. Removed by --unpair. See
rem meshroom-templates.ps1.
if exist "%HERE%meshroom-templates.ps1" ( powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%meshroom-templates.ps1" "%MR%" install ) else ( echo meshroom-templates.ps1 not beside this script: no Fast Ransac templates )
rem The DepthMap node in blocks of 48 views instead of 12 (docs/04, 0.3.4 "the depth-map node was loading
rem images"): each chunk is a process that loads the SfM data, probes the device and starts cold. Removed
rem by --unpair. Meshroom 2023.3 compiled its nodes into lib\meshroom\nodes: the package's
rem meshroom-overrides\DepthMap.py loads the compiled node and re-declares it with the larger block, and
rem Python prefers the .py beside the .pyc. Meshroom 2025.1 ships them as source in
rem aliceVision\share\meshroom\aliceVision: its DepthMap.py is kept as DepthMap.py.meshroom, a name the
rem node loader skips, and meshroom-overrides\DepthMap.2025.py, which loads that, takes its place.
set OVR=
if exist "%PKG%\share\cheshire\meshroom-overrides\" set OVR=%PKG%\share\cheshire\meshroom-overrides
if exist "%PKG%\common\share\cheshire\meshroom-overrides\" set OVR=%PKG%\common\share\cheshire\meshroom-overrides
set NODES=%MR%\lib\meshroom\nodes\aliceVision
set NODES25=%MR%\aliceVision\share\meshroom\aliceVision
if not defined OVR ( echo package carries no meshroom-overrides ^(pre-0.3.4^): DepthMap keeps Meshroom's block of 12 & exit /b 0 )
if exist "%NODES%\DepthMap.pyc" (
  copy /y "%OVR%\DepthMap.py" "%NODES%\DepthMap.py" >nul
  echo installed the DepthMap node override: blocks of 48 views ^(CHESHIRE_DEPTHMAP_BLOCK=0 for Meshroom's 12^)
  exit /b 0
)
if not exist "%NODES25%\DepthMap.py" ( echo no DepthMap node in %NODES% or %NODES25%: node override not installed & exit /b 0 )
if not exist "%OVR%\DepthMap.2025.py" ( echo package carries no Meshroom 2025 node override ^(pre-0.3.6^): DepthMap keeps Meshroom's block of 12 & exit /b 0 )
rem once only: a re-pair finds its own override in DepthMap.py and must not keep that as Meshroom's
findstr /c:"Cheshire" "%NODES25%\DepthMap.py" >nul || move /y "%NODES25%\DepthMap.py" "%NODES25%\DepthMap.py.meshroom" >nul
copy /y "%OVR%\DepthMap.2025.py" "%NODES25%\DepthMap.py" >nul
echo installed the Meshroom 2025 DepthMap node override: blocks of 48 views ^(CHESHIRE_DEPTHMAP_BLOCK=0 for Meshroom's 12^)
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

:pairif
rem :pairif <node> [option the launcher drops, without dashes]: pair unless the package's binary lacks an option Meshroom's takes
set COMPAT=1
if exist "%HERE%meshroom-pair-check.ps1" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%meshroom-pair-check.ps1" "%MR%" "%PKG%" %1 %2
  if errorlevel 1 set COMPAT=
)
if defined COMPAT ( call :pair %1 ) else ( echo %1: not paired, Meshroom keeps its own binary )
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
  rem a running node holds its binary: the delete fails, so would the rename, and "restored" was a lie
  if exist "%T%.exe" ( echo %1: in use, NOT restored ^(a Meshroom job still running?^) - run --unpair again once it ends & exit /b 0 )
  ren "%T%.cuda.exe" %1.exe
  echo restored %1.exe
) else ( echo %1: not paired )
exit /b 0
