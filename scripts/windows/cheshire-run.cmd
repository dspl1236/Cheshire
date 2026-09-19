@echo off
rem Cheshire: run an AliceVision node from the bundled Windows package (docs/16).
rem
rem   cheshire-run.cmd <node> [args...]      e.g. cheshire-run.cmd aliceVision_meshing --input ...
rem   cheshire-run.cmd --which               print the payload that would be used, and stop
rem
rem The bundle carries one payload per runtime family and GPU target. cheshire-detect.exe asks the
rem card which it needs - whichever HIP runtime enumerates it is the family, and its gcnArchName
rem picks the target - and this composes PATH so Windows resolves each DLL from the right layer.
rem Nothing is copied: the GPU directory comes first so its five DLLs win over anything behind them.
setlocal EnableExtensions
set ROOT=%~dp0
if %ROOT:~-1%==\ set ROOT=%ROOT:~0,-1%

if not exist "%ROOT%\cheshire-detect.exe" (
  echo [cheshire] cheshire-detect.exe missing from %ROOT%
  exit /b 2
)
rem Capture through a temp file: `for /f usebackq` with a quoted executable AND a quoted argument
rem mis-parses ("The filename, directory name, or volume label syntax is incorrect"), and the
rem failure looks like a missing GPU rather than a quoting problem.
set FAM=
set TGT=
set DETOUT=%TEMP%\cheshire-detect-%RANDOM%.txt
"%ROOT%\cheshire-detect.exe" "%ROOT%" > "%DETOUT%" 2>nul
for /f "usebackq tokens=1,2" %%A in ("%DETOUT%") do (
  set FAM=%%A
  set TGT=%%B
)
del "%DETOUT%" 2>nul
if not defined FAM (
  echo [cheshire] no AMD GPU with a matching payload; run cheshire-detect.exe -v for detail
  exit /b 2
)

rem Targets we could not run on real hardware here are listed in gpu\<family>\UNTESTED. Say so on
rem every run that selects one: a README nobody opens is not a warning.
if exist "%ROOT%\gpu\%FAM%\UNTESTED" (
  findstr /x /c:"%TGT%" "%ROOT%\gpu\%FAM%\UNTESTED" >nul 2>&1
  if not errorlevel 1 echo [cheshire] NOTE: %TGT% has not been validated on hardware here - please report how it goes
)

if "%~1"=="--which" (
  echo %FAM% %TGT%
  exit /b 0
)
if "%~1"=="" (
  echo usage: cheshire-run.cmd ^<node^> [args...]   ^(or --which^)
  exit /b 2
)

rem Split off the node name and keep the rest verbatim: a node command line runs to dozens of
rem arguments, so %%1..%%9 with shift would silently drop everything past the ninth.
set REST=
for /f "tokens=1,*" %%A in ("%*") do set REST=%%B

set ALICEVISION_ROOT=%ROOT%\common
set PATH=%ROOT%\gpu\%FAM%\%TGT%;%ROOT%\gpu\%FAM%;%ROOT%\fam\%FAM%\bin;%ROOT%\common\bin;%PATH%
rem The node lives under fam\<family>\bin when the families differ, and under common\bin when the
rem bundle holds only one family - then nothing differs, so everything deduplicates into common.
set EXE=%ROOT%\fam\%FAM%\bin\%~1.exe
if not exist "%EXE%" set EXE=%ROOT%\common\bin\%~1.exe
if not exist "%EXE%" (
  echo [cheshire] no such node in the %FAM% payload: %~1
  exit /b 2
)
"%EXE%" %REST%
exit /b %ERRORLEVEL%
