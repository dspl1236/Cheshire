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
import hashlib, os, re, shutil, sys, zipfile
from collections import defaultdict
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

# Which GPU targets a binary carries, read from its offload bundle entry IDs
# ("...amdhsa--gfx1031", "...amdhsa--gfx12-generic"). Detecting rather than hard-coding a list of
# five means a newly GPU-bearing DLL is picked up without anyone remembering to add it.
#
# Read the bundle rather than just searching for the expected target name: a stale DLL built for
# another architecture would simply not contain the name, so a substring test files it as a plain
# CPU file and ships it - the wrong code object, silently, in a package that looks right. That is
# not hypothetical; it is what a reused build tree did here (docs/16).
OFFLOAD = re.compile(rb'amdhsa--([0-9a-z:+\-]+)')

def offload_targets(path: Path) -> set:
    if path.suffix.lower() not in ('.dll', '.exe'):
        return set()
    try:
        blob = path.read_bytes()
    except OSError:
        return set()
    return {m.decode() for m in OFFLOAD.findall(blob)}

def carries_target(name: str, target: str, tgts: set) -> bool:
    """Does this binary carry the code object the payload claims?

    Normally that means the target itself. popsift.dll is the exception: a generic code object
    makes PopSIFT exit 0xC0000094 on the first photograph, so generic payloads deliberately carry a
    PopSIFT built for the CHIPS that generic covers (docs/16). Accept that, but only when every
    target it carries really is a chip of the same family - so a gfx11 PopSIFT in a gfx12 payload is
    still caught.
    """
    if target in tgts:
        return True
    if name.lower() != 'popsift.dll' or not target.endswith('-generic') or not tgts:
        return False
    fam = target[:-len('-generic')].replace('-', '')      # gfx10-3-generic -> gfx103
    return all(t.startswith(fam) and t[len(fam):].isdigit() for t in tgts)


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
        # A directory with no bin/ is a bare payload from build_targets.py: the five GPU DLLs and
        # nothing else. Only the first input of a family needs to be a full package, since that is
        # where the shared and per-family files come from.
        if not (p / 'bin').is_dir():
            if any(f.suffix.lower() == '.dll' for f in p.iterdir() if f.is_file()):
                inputs.append((family, target, p)); continue
            print(f"error: {p} is neither a package (no bin/) nor a payload (no DLLs)"); return 1
        inputs.append((family, target, p))

    stage = outzip.with_suffix('')
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    # Pass 1: classify every file of every input, and hash the ones that might be shared.
    #
    # Only the first input of a family contributes non-GPU files. Every target in a family is built
    # from one source tree with one toolchain, so their CPU binaries are the same build - measured:
    # between the gfx1030 and gfx1031 packages aliceVision_sfm.dll differs by ONE byte, a PE
    # timestamp at offset 129 (docs/16). Taking them from each target would add a copy per chip and
    # let a meaningless timestamp decide whether a file looks "shared".
    seen_family = set()
    shared_hashes = defaultdict(dict)     # relpath -> {family: hash}
    shared_src = {}                       # (family, relpath) -> Path
    counts = defaultdict(int)
    mismatched = []                       # (family, target, relpath, targets actually found)
    for family, target, root in inputs:
        base_for_family = family not in seen_family
        seen_family.add(family)
        # A bare payload dir carries the five DLLs and the build logs that produced them; only the
        # DLLs belong in the bundle. It also cannot be a family's base, because the shared and
        # per-family files have to come from a real package.
        is_payload = not (root / 'bin').is_dir()
        if is_payload and base_for_family:
            print(f"error: {root} is a payload, but it is the first input for family '{family}' - "
                  f"the first must be a full package so the bundle gets its CPU and runtime files")
            return 1
        for f in sorted(root.rglob('*')):
            if not f.is_file(): continue
            if is_payload and f.suffix.lower() != '.dll': continue
            rel = f.relative_to(root).as_posix()
            name = f.name
            if is_runtime(name):
                dst = stage / 'gpu' / family / name
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst)
                    counts[f'gpu/{family} runtime'] += 1
                continue
            tgts = offload_targets(f)
            if tgts:
                # Every GPU-bearing binary in this payload must carry the target it is filed under.
                # Anything else is a stale artifact from another architecture, and shipping it would
                # put the wrong code object in a package that looks correct.
                if not carries_target(f.name, target, tgts):
                    mismatched.append((family, target, rel, sorted(tgts)))
                    continue
                dst = stage / 'gpu' / family / target / name
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst)
                counts[f'gpu/{family}/{target}'] += 1
                continue
            if not base_for_family: continue
            shared_hashes[rel][family] = sha(f)
            shared_src[(family, rel)] = f

    if mismatched:
        print("error: GPU binaries carrying the wrong architecture - refusing to bundle")
        for fam, tgt, rel, found in mismatched:
            print(f"  {fam}/{tgt}: {rel} carries {', '.join(found)}")
        print("a stale build tree is the usual cause; rebuild that target and check the offload "
              "bundle before retrying")
        return 1

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
    # What this replaces is one full package per target, so compare against that - not against the
    # inputs, most of which are bare five-DLL payloads and would understate it wildly.
    pkgs = [r for _, _, r in inputs if (r / 'bin').is_dir()]
    per_pkg = max((sum(f.stat().st_size for f in r.rglob('*') if f.is_file()) for r in pkgs),
                  default=0)
    print(f"bundle {total >> 20} MB, against {(per_pkg * len(inputs)) >> 20} MB as one package per "
          f"target ({len(inputs)} targets, {len(families)} runtime "
          f"famil{'y' if len(families)==1 else 'ies'})")

    print(f'=== {crlf_batch_files(stage)} batch files written with CRLF line endings')
    with zipfile.ZipFile(outzip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(stage.parent))
    print('zip', outzip, round(outzip.stat().st_size / 2**20), 'MB')
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
