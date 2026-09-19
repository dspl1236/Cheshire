#!/usr/bin/env python3
"""Compose one Windows zip from several packaged install trees (docs/16).

Six downloads today, one per toolchain and chip, and picking the wrong one does not fail loudly -
the package enumerates the device, runs every kernel and returns wrong data. This builds a single
package that chooses for itself.

It works because the per-chip delta is tiny: of a 427 MB package, exactly five DLLs carry GPU code
objects (4.7 MB), the vcpkg runtime and share/ tree are bit-identical across toolchains, and only
AliceVision's own CPU DLLs differ between the clang 19 /arch:AVX and clang 22 /arch:AVX2 builds.

    usage: bundle_windows.py <out.zip> <family>:<target>:<package dir> [...]
    e.g.   bundle_windows.py cheshire-windows-x64.zip \\
             rocm7.2:gfx12-generic:build/release/pkg-gfx12 \\
             hip6.2:gfx1031:build/release/pkg-gfx1031

Layout produced:

    cheshire-detect.exe        the card probe (scripts/windows/cheshire-detect.cpp)
    common/bin/, common/share/ everything bit-identical across every input
    fam/<family>/bin/, lib/    AliceVision's own binaries for that toolchain
    gpu/<family>/              amdhip64_* + amd_comgr_* (what the probe loads)
    gpu/<family>/<target>/     the five GPU-bearing DLLs

At run time the launcher sets
    PATH = gpu/<fam>/<target>;gpu/<fam>;fam/<fam>/bin;common/bin
    ALICEVISION_ROOT = common
so nothing is copied or materialised on the user's disk - Windows resolves each DLL from PATH, and
the GPU directory comes first so its five win over anything with the same name.
"""
import hashlib, os, shutil, sys, zipfile
from collections import defaultdict
from pathlib import Path

# A DLL carries GPU code if the target name appears in it; that is how the five were found in the
# first place (docs/16). Detecting rather than hard-coding means a new GPU-bearing DLL is picked up
# without anyone remembering to add it here.
def gpu_bearing(path: Path, target: str) -> bool:
    if path.suffix.lower() not in ('.dll', '.exe'):
        return False
    needle = target.encode().lower()
    try:
        return needle in path.read_bytes().lower()
    except OSError:
        return False

def is_runtime(name: str) -> bool:
    n = name.lower()
    return n.startswith('amdhip64') or n.startswith('amd_comgr')

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main(argv):
    if len(argv) < 3:
        print(__doc__); return 2
    outzip = Path(argv[1])
    inputs = []
    for spec in argv[2:]:
        family, target, d = spec.split(':', 2)
        p = Path(d)
        if not p.is_dir():
            print(f"error: not a directory: {p}"); return 1
        # accept either the package dir or its single child (the zip's top-level folder)
        if not (p / 'bin').is_dir():
            kids = [c for c in p.iterdir() if c.is_dir() and (c / 'bin').is_dir()]
            if len(kids) == 1: p = kids[0]
        if not (p / 'bin').is_dir():
            print(f"error: no bin/ under {p}"); return 1
        inputs.append((family, target, p))

    stage = outzip.with_suffix('')
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    # Pass 1: classify every file of every input, and hash the ones that might be shared.
    shared_hashes = defaultdict(dict)     # relpath -> {family: hash}
    shared_src = {}                       # (family, relpath) -> Path
    counts = defaultdict(int)
    for family, target, root in inputs:
        for f in sorted(root.rglob('*')):
            if not f.is_file(): continue
            rel = f.relative_to(root).as_posix()
            name = f.name
            if is_runtime(name):
                dst = stage / 'gpu' / family / name
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst)
                    counts[f'gpu/{family} runtime'] += 1
                continue
            if gpu_bearing(f, target):
                dst = stage / 'gpu' / family / target / name
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst)
                counts[f'gpu/{family}/{target}'] += 1
                continue
            shared_hashes[rel][family] = sha(f)
            shared_src[(family, rel)] = f

    families = sorted({fam for fam, _, _ in inputs})
    # Pass 2: a file identical in every family goes in once; anything else goes per family.
    for rel, byfam in sorted(shared_hashes.items()):
        vals = set(byfam.values())
        if len(vals) == 1 and set(byfam) == set(families):
            src = shared_src[(sorted(byfam)[0], rel)]
            dst = stage / 'common' / rel
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
            counts['common'] += 1
        else:
            for fam in byfam:
                src = shared_src[(fam, rel)]
                dst = stage / 'fam' / fam / rel
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
                counts[f'fam/{fam}'] += 1

    # The probe, and the runner that uses it.
    here = Path(__file__).parent
    probe = here.parent / 'build' / 'cheshire-detect.exe'
    if probe.exists(): shutil.copy2(probe, stage / 'cheshire-detect.exe')
    else: print('WARNING: build/cheshire-detect.exe missing (run scripts\\windows\\build-launcher.cmd)')
    launcher = here.parent / 'build' / 'meshroom-pair-launcher.exe'
    if launcher.exists(): shutil.copy2(launcher, stage / 'meshroom-pair-launcher.exe')
    for extra in ('windows/meshroom-pair.cmd', 'windows/cheshire-run.cmd'):
        src = here / extra
        if src.exists(): shutil.copy2(src, stage / src.name)

    for k in sorted(counts): print(f"  {k:<34} {counts[k]:5} files")
    total = sum(f.stat().st_size for f in stage.rglob('*') if f.is_file())
    naive = sum(sum(f.stat().st_size for f in r.rglob('*') if f.is_file()) for _, _, r in inputs)
    print(f"bundle {total >> 20} MB, against {naive >> 20} MB as separate packages "
          f"({len(inputs)} payloads, {len(families)} runtime famil{'y' if len(families)==1 else 'ies'})")

    with zipfile.ZipFile(outzip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(stage.parent))
    print('zip', outzip, round(outzip.stat().st_size / 2**20), 'MB')
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
