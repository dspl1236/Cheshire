@echo off
rem The whole power-pc benchmark, smallest set first (bench_win.py): mini6, the 41 views, the engine bay and the zoo with the
rem stock, cpu and cheshire arms, then the False Door with stock and cheshire. Started hidden; progress in C:\cheshire-bench\runs-win\bench.done.
cd /d %~dp0
set PYTHONUNBUFFERED=1
py -3 bench_win.py mini6 D:\MMI\cheshire\data\monstree\mini6 all
py -3 bench_win.py monstree41 D:\MMI\cheshire\data\monstree\full all
py -3 bench_win.py enginebay D:\MMI\cheshire\data\scans\enginebay\photos all
py -3 bench_win.py zoo D:\MMI\cheshire\data\scans\zoo\photos all
py -3 bench_win.py falsedoor D:\MMI\cheshire\data\scans\falsedoor-photos stock,cheshire
echo all done %date% %time% >> C:\cheshire-bench\runs-win\bench.done
