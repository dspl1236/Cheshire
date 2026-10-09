@echo off
rem Step 13a on the RX 9070: a flat test package of the dev build (bundled HIP runtime moved aside, as for the False
rem Door runs), mini6 on Meshroom 2025.1: base, verify (the four-lane SVD's check against Eigen), blast (the QR
rem nullspace's own announcement) and cpufallback (the lanes on the CPU matcher's path), then 41 views verify.
cd /d D:\MMI\cheshire
set PYTHONUNBUFFERED=1
set S=C:\cheshire-fd-test\gate13a-win.status
echo === gate-13a start %date% %time% > %S%
py -3 scripts\package_windows.py build\av-gfx1201-popsift-install tools\vcpkg-deps\x64-windows-release\installed\x64-windows-release\bin tools\venv-rocm\Lib\site-packages\_rocm_sdk_core\bin tools\rocm-6.2\bin\llvm-objdump.exe C:\cheshire-fd-test\pkg-v13a.zip > C:\cheshire-fd-test\package-v13a.log 2>&1
if errorlevel 1 (echo === package failed >> %S% & exit /b 1)
if not exist C:\cheshire-fd-test\runtime-aside-13a mkdir C:\cheshire-fd-test\runtime-aside-13a
move /Y C:\cheshire-fd-test\pkg-v13a\bin\amdhip64_7.dll C:\cheshire-fd-test\runtime-aside-13a\ >> %S%
move /Y C:\cheshire-fd-test\pkg-v13a\bin\amd_comgr0702.dll C:\cheshire-fd-test\runtime-aside-13a\ >> %S%
echo === packaged %date% %time% >> %S%
py -3 scripts\verify_end_to_end.py build\meshroom\Meshroom-2025.1.0 C:\cheshire-fd-test\pkg-v13a data\monstree\mini6 C:\cheshire-fd-test\gate13a-win base verify blast cpufallback > C:\cheshire-fd-test\gate13a-win.log 2>&1
echo === mini6 exit %errorlevel% %date% %time% >> %S%
py -3 scripts\verify_end_to_end.py build\meshroom\Meshroom-2025.1.0 C:\cheshire-fd-test\pkg-v13a data\monstree\full C:\cheshire-fd-test\gate13a-win-41 verify > C:\cheshire-fd-test\gate13a-win-41.log 2>&1
echo === 41 exit %errorlevel% %date% %time% >> %S%
