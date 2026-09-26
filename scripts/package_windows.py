#!/usr/bin/env python3
"""Make a self-contained Windows zip of the HIP AliceVision install: copy the install tree and the
transitive DLL closure (vcpkg runtime + ROCm runtime) next to the executables, so it runs from
any folder with only the AMD display driver installed.
usage: package_windows.py <install dir> <vcpkg bin> <rocm bin> <llvm-objdump.exe> <out zip>"""
import sys
import os, shutil, subprocess, sys, zipfile
from pathlib import Path


def crlf_batch_files(root: Path) -> int:
    """Rewrite every .cmd/.bat under root with CRLF line endings, and return how many there were.

    cmd.exe's label search is unreliable in LF-only batch files: on 2026-09-22 a package whose
    meshroom-pair.cmd had been rewritten with LF (WSL git resetting the shared checkout) failed its
    first `call :pair` with "The system cannot find the batch label specified", and DepthMap went
    unpaired on every card. The package must not depend on how the checkout was written."""
    n = 0
    for f in list(root.rglob('*.cmd')) + list(root.rglob('*.bat')):
        b = f.read_bytes()
        fixed = b.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        if fixed != b:
            f.write_bytes(fixed)
        if fixed.count(b'\n') != fixed.count(b'\r\n'):
            sys.exit(f"{f}: line endings still not CRLF")
        n += 1
    return n
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
# GPL contamination guard (2026-09-22). The Windows packages up to 0.3.2 carried libspqr.dll and
# libcholmod.dll: vcpkg's Ceres links SuiteSparse, and the import walk above copies whatever a DLL
# names. The AliceVision maintainers ruled the same day that pre-built binaries must not package
# SPQR (discussion #2116), and Cheshire is MPL-2.0. 0.3.3's Ceres is built without SuiteSparse
# (build/ceres-nosuitesparse.cmd); this refuses a stage where any of these came back.
gpl = sorted(p.name for p in (stage / 'bin').glob('*.dll')
             if p.name.lower() in ('libspqr.dll', 'libcholmod.dll', 'libumfpack.dll', 'libklu.dll'))
if gpl:
    sys.exit(f"GPL SuiteSparse libraries in the package: {', '.join(gpl)} - a DLL still imports them "
             f"(ceres.dll built with SUITESPARSE=ON?); rebuild Ceres without SuiteSparse first")
print('=== no SuiteSparse GPL libraries in the package')

cand = list(rbin.glob('amd_comgr*.dll'))
wanted = set()
for pe in list((stage / 'bin').glob('*.dll')) + list((stage / 'bin').glob('*.exe')):
    blob = pe.read_bytes().lower()
    wanted |= {c.name for c in cand if c.name.encode().lower() in blob}
# Prune only on positive evidence. amdhip64_6.dll names amd_comgr_2.dll, so the HIP SDK 6.2 tree's
# other 107 MB copy (amd_comgr0602.dll) is provably dead. amdhip64_7.dll names no comgr at all -
# it resolves one some other way - so there the evidence set is empty and everything is copied,
# because pruning the only comgr would ship a package whose runtime cannot start.
for c in cand:
    if wanted and c.name not in wanted:
        print(f"skipping unreferenced {c.name} ({c.stat().st_size >> 20} MB)"); continue
    if c.name.lower() not in have: shutil.copy2(c, stage / 'bin' / c.name); copied.append(c.name.lower())
# MSVC runtime + LLVM OpenMP runtime: present on a developer's PC (Visual Studio drops them into
# System32) and absent on a clean one, where the exe dies with 0xC0000135 and no message. Both
# are Microsoft-redistributable; take them from the VS Redist tree, falling back to System32.
import glob
# NB: the previous pattern here contained literal form-feed bytes (\f interpreted rather than
# kept), so it matched nothing and every runtime DLL silently came from the System32
# fallback instead - fine on a machine with Visual Studio, nothing at all on one without.
redist = sorted(glob.glob(r'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT'))
runtime = {}
for d in redist + [r'C:\Windows\System32']:
    for f in Path(d).glob('*.dll'):
        n = f.name.lower()
        if n not in runtime and n.startswith(('msvcp140', 'vcruntime140', 'concrt140')) and not n.endswith('d.dll') and 'debug' not in n:
            runtime[n] = f
missing_rt = set()
for pe in list((stage / 'bin').glob('*.dll')) + list((stage / 'bin').glob('*.exe')):
    for dep in imports(pe):
        if dep in have: continue
        if dep in runtime: missing_rt.add(dep)
for n in sorted(missing_rt):
    shutil.copy2(runtime[n], stage / 'bin' / runtime[n].name); have.add(n); copied.append(n)
print(f"runtime DLLs bundled: {' '.join(sorted(missing_rt))}")

# OpenMP runtime: LLVM's build, not Microsoft's. Microsoft ships the same runtime (the file says
# Company: LLVM) but only under debug_nonredist / System32, neither of which grants redistribution.
# LLVM's is Apache-2.0 WITH LLVM-exception, which does - provided the licence travels with it.
# Copied under the name -openmp:llvm makes the binaries import. Checked by comparing imports rather
# than exports: Microsoft's build exports 25 OpenMP 6.0 memspace symbols LLVM 20.1.8 lacks, and
# nothing in the package imports any of them. See third_party/llvm-openmp/README.md.
omp_src = Path(__file__).parent.parent / 'third_party' / 'llvm-openmp'
if (omp_src / 'libomp.dll').exists():
    shutil.copy2(omp_src / 'libomp.dll', stage / 'bin' / 'libomp140.x86_64.dll')
    if (omp_src / 'LICENSE.TXT').exists():
        shutil.copy2(omp_src / 'LICENSE.TXT', stage / 'LICENSE.llvm-openmp.txt')
    copied.append('libomp140.x86_64.dll')
    print('OpenMP runtime: LLVM build from third_party/llvm-openmp, with its licence')
else:
    print('WARNING: no third_party/llvm-openmp/libomp.dll - the package will have no OpenMP runtime'
          ' and every binary will die with 0xC0000135 on a machine without Visual Studio')
print(f"copied {len(copied)} DLLs: {' '.join(sorted(copied))[:600]}")
# the no-repo runner, so the zip is usable on its own: run-depthmap-standalone.cmd <package> <cache> <out>
shutil.copy2(Path(__file__).with_name('run-depthmap-standalone.cmd'), stage / 'run-depthmap-standalone.cmd')
# Meshroom pairing: the pair script plus the launcher it installs (built by scripts/windows/build-launcher.cmd)
shutil.copy2(Path(__file__).parent / 'windows' / 'meshroom-pair.cmd', stage / 'meshroom-pair.cmd')
# and the compatibility check it runs before each node (Meshroom's options against the package's)
shutil.copy2(Path(__file__).parent / 'windows' / 'meshroom-pair-check.ps1', stage / 'meshroom-pair-check.ps1')
launcher = Path(__file__).parent.parent / 'build' / 'meshroom-pair-launcher.exe'
if launcher.exists(): shutil.copy2(launcher, stage / 'meshroom-pair-launcher.exe')
else: print('WARNING: build/meshroom-pair-launcher.exe missing (run scripts/windows/build-launcher.cmd); zip has no pairing launcher')
print(f'=== {crlf_batch_files(stage)} batch files written with CRLF line endings')
with zipfile.ZipFile(outzip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in stage.rglob('*'):
        if p.is_file(): z.write(p, p.relative_to(stage.parent))
print('zip', outzip, round(outzip.stat().st_size / 2**20), 'MB')
