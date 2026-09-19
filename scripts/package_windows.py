#!/usr/bin/env python3
"""Make a self-contained Windows zip of the HIP AliceVision install: copy the install tree and the
transitive DLL closure (vcpkg runtime + ROCm runtime) next to the executables, so it runs from
any folder with only the AMD display driver installed.
usage: package_windows.py <install dir> <vcpkg bin> <rocm bin> <llvm-objdump.exe> <out zip>"""
import os, shutil, subprocess, sys, zipfile
from pathlib import Path
inst, vbin, rbin, objdump, outzip = map(Path, sys.argv[1:6])
stage = outzip.with_suffix(''); shutil.rmtree(stage, ignore_errors=True); shutil.copytree(inst, stage)
search = {p.name.lower(): p for d in (vbin, rbin) for p in d.glob('*.dll')}
have = {p.name.lower() for p in (stage / 'bin').glob('*.dll')}
def imports(pe):
    out = subprocess.run([str(objdump), '-p', str(pe)], capture_output=True, text=True, errors='replace').stdout
    return [l.split(':', 1)[1].strip().lower() for l in out.splitlines() if l.strip().startswith('DLL Name:')]
todo = [p for p in (stage / 'bin').iterdir() if p.suffix.lower() in ('.exe', '.dll')]
seen = set(); copied = []
while todo:
    pe = todo.pop()
    if pe.name.lower() in seen: continue
    seen.add(pe.name.lower())
    for dep in imports(pe):
        if dep in have or dep not in search: continue
        dst = stage / 'bin' / search[dep].name; shutil.copy2(search[dep], dst); have.add(dep); copied.append(dep); todo.append(dst)
# HIP loads the code-object manager with LoadLibrary at runtime (same as libamd_comgr on Linux):
# without it hipGetDeviceCount reports no GPU. Not in any import table, so copy it explicitly - but
# copy only the ones actually named, not every amd_comgr*.dll in the toolchain. The HIP SDK 6.2 bin
# holds two, amd_comgr_2.dll and amd_comgr0602.dll, 107 MB each; amdhip64_6.dll names the first and
# nothing in the package mentions the second, so globbing shipped a quarter of the zip as dead
# weight. A LoadLibrary name is a plain string in the binary, so grep for it (docs/16).
cand = list(rbin.glob('amd_comgr*.dll'))
wanted = set()
for pe in list((stage / 'bin').glob('*.dll')) + list((stage / 'bin').glob('*.exe')):
    blob = pe.read_bytes().lower()
    wanted |= {c.name for c in cand if c.name.encode().lower() in blob}
for c in cand:
    if c.name not in wanted:
        print(f"skipping unreferenced {c.name} ({c.stat().st_size >> 20} MB)"); continue
    if c.name.lower() not in have: shutil.copy2(c, stage / 'bin' / c.name); copied.append(c.name.lower())
# MSVC runtime + LLVM OpenMP runtime: present on a developer's PC (Visual Studio drops them into
# System32) and absent on a clean one, where the exe dies with 0xC0000135 and no message. Both
# are Microsoft-redistributable; take them from the VS Redist tree, falling back to System32.
import glob
redist = sorted(glob.glob(r'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Redist\MSVC.*d\Microsoft.VC*.CRT'))        + sorted(glob.glob(r'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Redist\MSVC.*d\Microsoft.VC*.OpenMP.LLVM'))
runtime = {}
for d in redist + [r'C:\Windows\System32']:
    for f in Path(d).glob('*.dll'):
        n = f.name.lower()
        if n not in runtime and (n.startswith(('msvcp140', 'vcruntime140', 'concrt140')) or n.startswith('libomp140')) and not n.endswith('d.dll') and 'debug' not in n:
            runtime[n] = f
missing_rt = set()
for pe in list((stage / 'bin').glob('*.dll')) + list((stage / 'bin').glob('*.exe')):
    for dep in imports(pe):
        if dep in have: continue
        if dep in runtime: missing_rt.add(dep)
for n in sorted(missing_rt):
    shutil.copy2(runtime[n], stage / 'bin' / runtime[n].name); have.add(n); copied.append(n)
print(f"runtime DLLs bundled: {' '.join(sorted(missing_rt))}")
print(f"copied {len(copied)} DLLs: {' '.join(sorted(copied))[:600]}")
# the no-repo runner, so the zip is usable on its own: run-depthmap-standalone.cmd <package> <cache> <out>
shutil.copy2(Path(__file__).with_name('run-depthmap-standalone.cmd'), stage / 'run-depthmap-standalone.cmd')
# Meshroom pairing: the pair script plus the launcher it installs (built by scripts/windows/build-launcher.cmd)
shutil.copy2(Path(__file__).parent / 'windows' / 'meshroom-pair.cmd', stage / 'meshroom-pair.cmd')
launcher = Path(__file__).parent.parent / 'build' / 'meshroom-pair-launcher.exe'
if launcher.exists(): shutil.copy2(launcher, stage / 'meshroom-pair-launcher.exe')
else: print('WARNING: build/meshroom-pair-launcher.exe missing (run scripts/windows/build-launcher.cmd); zip has no pairing launcher')
with zipfile.ZipFile(outzip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in stage.rglob('*'):
        if p.is_file(): z.write(p, p.relative_to(stage.parent))
print('zip', outzip, round(outzip.stat().st_size / 2**20), 'MB')
