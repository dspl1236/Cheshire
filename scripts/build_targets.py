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

# PopSIFT must NOT be built for a generic target. A generic code object makes it exit 0xC0000094
# (integer divide by zero) on the first photograph, while the identical source built for the chip
# works - verified gfx1201 against gfx12-generic on an RX 9070. Every other GPU DLL is fine under a
# generic target: DepthMap, Meshing and Texturing were all checked. So the generic payloads carry a
# PopSIFT built for the chips that generic covers, which ROCm 7.2 handles as one fat binary (the
# v0.2.16 rdna3-rdna4 package shipped ten targets in one popsift.dll).
#
# Consequence to remember: a future chip in these families runs the generic AliceVision objects but
# finds no PopSIFT code object, so GPU SIFT will not cover it until this list is extended.
POPSIFT_FOR_GENERIC = {
    'gfx11-generic': 'gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153',
    'gfx12-generic': 'gfx1200;gfx1201',
}

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

        psenv = dict(env)
        psarch = POPSIFT_FOR_GENERIC.get(target)
        if psarch:
            psenv['CHESHIRE_HIP_ARCHS'] = psarch
            print(f"    popsift for {psarch} (generic code objects break it)")
        rc = run(['cmd', '/c', str(ROOT / 'scripts' / 'build-popsift.cmd'), tree, 'install'],
                 psenv, logdir / 'popsift.log')
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
            want = set(psarch.split(';')) if (psarch and dll == 'popsift.dll') else {target}
            if not want <= got:
                bad.append(f"{dll}: carries {', '.join(sorted(got)) or 'no code object'}, "
                           f"wanted {', '.join(sorted(want))}"); continue
            shutil.copy2(src, logdir / dll)
            # ...and put the verified copy back into aliceVision's install tree. That tree is what
            # package_windows.py turns into a family's BASE package, and for the same reason the
            # comment above gives - install treats an already-present popsift.dll as up to date -
            # it otherwise keeps the previous target's copy. The bundler's architecture guard
            # catches the result, but only after a full build: a base package declaring
            # gfx12-generic while carrying a gfx11 PopSIFT, which is a package that looks right
            # and hands GPU SIFT the wrong code objects.
            if dll == 'popsift.dll' and src != avbin / dll:
                shutil.copy2(src, avbin / dll)
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
