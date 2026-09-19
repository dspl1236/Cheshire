#!/usr/bin/env python3
"""Build the GPU payloads for a bundled Windows package (docs/16).

    build_targets.py <family> <target> [target ...]
    e.g. build_targets.py hip6.2 gfx1010 gfx1012 gfx1030 gfx1031 gfx1032 gfx1034
         build_targets.py rocm7.2 gfx11-generic gfx12-generic

Only five DLLs differ per GPU target, so each target after the first reuses the family's build tree
and changes CHESHIRE_HIP_ARCHS: the CPU objects stay cached and a target costs minutes rather than a
full AliceVision build. The five are harvested into build/payload/<family>/<target>/.

Every harvested DLL is checked against the offload bundle it actually carries. A build tree that
quietly kept an older architecture is the failure this guards - it produces a package that looks
right and returns wrong data (docs/16), and it has happened here.
"""
import os, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GPU_DLLS = ['aliceVision_matching.dll', 'popsift.dll', 'aliceVision_fuseCut.dll',
            'aliceVision_mesh.dll', 'aliceVision_depthMap_cuda.dll']
OFFLOAD = re.compile(rb'amdhsa--([0-9a-z:+\-]+)')

FAMILIES = {
    # family:   (tree name, aliceVision build suffix, extra env for the toolchain)
    'rocm7.2': ('gfx1201', '-popsift', {}),
    'hip6.2':  ('gfx1012', '-hip62-popsift', {
        'CHESHIRE_ROCM_PATH': str(ROOT / 'tools' / 'rocm-6.2'),
        'CHESHIRE_LLVM_BIN': str(ROOT / 'tools' / 'rocm-6.2' / 'bin'),
        'CHESHIRE_DEVICE_LIB_PATH': str(ROOT / 'tools' / 'rocm-6.2' / 'amdgcn' / 'bitcode'),
        'CHESHIRE_ARCH_FLAG': '/arch:AVX',
        'CHESHIRE_EXTRA_CXXFLAGS': '-D__builtin_verbose_trap(x,y)=__builtin_trap()',
        'CHESHIRE_HIP_EXTRA_FLAGS': '-D__builtin_verbose_trap(x,y)=__builtin_trap()',
    }),
}

def targets_of(path: Path) -> set:
    return {m.decode() for m in OFFLOAD.findall(path.read_bytes())}

def run(cmd, env, log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8', errors='replace') as f:
        p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, shell=False)
    return p.returncode

def main(argv):
    if len(argv) < 3:
        print(__doc__); return 2
    family, targets = argv[1], argv[2:]
    if family not in FAMILIES:
        print(f"error: unknown family {family}; expected one of {', '.join(FAMILIES)}"); return 1
    tree, suffix, extra = FAMILIES[family]

    failures = []
    for target in targets:
        print(f"=== {family} / {target}")
        env = dict(os.environ)
        env.update(extra)
        env['CHESHIRE_HIP_ARCHS'] = target
        env['CHESHIRE_POPSIFT'] = 'ON'
        env['CHESHIRE_POPSIFT_DIR'] = str(ROOT / 'build' / f'popsift-{tree}-install' / 'lib' / 'cmake' / 'PopSift').replace('\\', '/')
        env['CHESHIRE_BUILD_SUFFIX'] = suffix
        logdir = ROOT / 'build' / 'payload' / family / target

        rc = run(['cmd', '/c', str(ROOT / 'scripts' / 'build-popsift.cmd'), tree, 'install'],
                 env, logdir / 'popsift.log')
        if rc != 0:
            print(f"    popsift build failed, see {logdir / 'popsift.log'}")
            failures.append((target, 'popsift build')); continue

        rc = run(['cmd', '/c', str(ROOT / 'scripts' / 'build-alicevision.cmd'), tree, 'install'],
                 env, logdir / 'alicevision.log')
        if rc != 0:
            print(f"    aliceVision build failed, see {logdir / 'alicevision.log'}")
            failures.append((target, 'aliceVision build')); continue

        # Harvest. popsift.dll comes from its own install: aliceVision's install step treats an
        # already-present popsift.dll as up to date and leaves the previous target's copy in place.
        avbin = ROOT / 'build' / f'av-{tree}{suffix}-install' / 'bin'
        psbin = ROOT / 'build' / f'popsift-{tree}-install' / 'bin'
        bad = []
        for dll in GPU_DLLS:
            src = (psbin if dll == 'popsift.dll' else avbin) / dll
            if not src.exists():
                bad.append(f"{dll}: missing"); continue
            got = targets_of(src)
            if target not in got:
                bad.append(f"{dll}: carries {', '.join(sorted(got)) or 'no code object'}"); continue
            shutil.copy2(src, logdir / dll)
        if bad:
            print(f"    WRONG ARCHITECTURE, not harvested:")
            for b in bad: print(f"      {b}")
            failures.append((target, 'architecture mismatch')); continue
        print(f"    ok -> {logdir}")

    if failures:
        print(f"\n{len(failures)} of {len(targets)} targets failed:")
        for t, why in failures: print(f"  {t}: {why}")
        return 1
    print(f"\nall {len(targets)} targets built for {family}")
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
