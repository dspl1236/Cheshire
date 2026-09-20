#!/usr/bin/env python3
"""Assemble the bundled Windows package from built payloads (docs/16).

    assemble_bundle.py <out.zip> <rocm7.2 package> <hip6.2 package>

The two packages are the family bases, giving the bundle its CPU binaries, vcpkg runtime, share/
tree and HIP runtime. Every GPU target comes from build/payload/<family>/<target>/, so the bases
only need to exist for one target each.

Also writes gpu/<family>/UNTESTED. A target listed there has not been run on that hardware here, and
cheshire-run.cmd and the Meshroom launcher say so on every run that selects it - a README nobody
opens is not a warning. The list is what it is: of eleven targets, three have been run on a real
card, and pretending otherwise would be the one thing this project cannot afford.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAYLOADS = ROOT / 'build' / 'payload'

# Run on a real card here, with a point-cloud checksum matching a reference the same machine made.
VALIDATED = {
    'hip6.2':  ['gfx1012', 'gfx1031'],
    'rocm7.2': ['gfx12-generic'],
}

def main(argv):
    if len(argv) != 4:
        print(__doc__); return 2
    outzip, pkg_rocm, pkg_hip = Path(argv[1]), Path(argv[2]), Path(argv[3])

    # Detect the base package's own GPU target instead of assuming one. The build trees are reused
    # across targets, so whichever was built last is what the install holds - declaring the wrong
    # one makes the bundler reject the package for carrying the wrong architecture, which is at
    # least loud, but there is no reason to guess when the answer is in the binary.
    import re as _re
    def base_target_of(pkg: Path) -> str:
        root = pkg if (pkg / 'bin').is_dir() else next(
            (c for c in pkg.iterdir() if c.is_dir() and (c / 'bin').is_dir()), pkg)
        for dll in ('aliceVision_depthMap_cuda.dll', 'aliceVision_fuseCut.dll'):
            f = root / 'bin' / dll
            if not f.exists():
                continue
            found = {m.decode() for m in _re.findall(rb'amdhsa--([0-9a-z:+\-]+)', f.read_bytes())}
            if len(found) == 1:
                return found.pop()
        sys.exit(f"cannot determine the GPU target of {pkg}")

    specs, untested = [], {}
    for family, base, base_target in (('rocm7.2', pkg_rocm, base_target_of(pkg_rocm)),
                                      ('hip6.2', pkg_hip, base_target_of(pkg_hip))):
        famdir = PAYLOADS / family
        if not famdir.is_dir():
            print(f"error: no payloads under {famdir}"); return 1
        targets = sorted(d.name for d in famdir.iterdir()
                         if d.is_dir() and any(d.glob('*.dll')))
        if base_target not in targets:
            print(f"error: {family} base is declared as {base_target} but there is no payload for "
                  f"it; the base package's own GPU DLLs must match the target it is filed under")
            return 1
        # The base package first: it carries the family's CPU and runtime files.
        print(f"  {family}: base package carries {base_target}")
        specs.append(f"{family}:{base_target}:{base}")
        specs += [f"{family}:{t}:{famdir / t}" for t in targets if t != base_target]
        untested[family] = [t for t in targets if t not in VALIDATED.get(family, [])]

    cmd = [sys.executable, str(ROOT / 'scripts' / 'bundle_windows.py'), str(outzip)] + specs
    print('  ' + ' \\\n  '.join(cmd[2:]))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        return rc

    stage = outzip.with_suffix('')
    for family, targets in untested.items():
        if not targets: continue
        p = stage / 'gpu' / family / 'UNTESTED'
        p.write_text('\n'.join(targets) + '\n', encoding='ascii')
        print(f"  {family}: {len(targets)} target(s) marked untested: {', '.join(targets)}")
    print("\nre-zipping with the UNTESTED lists")
    import zipfile
    with zipfile.ZipFile(outzip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(stage.parent))
    print('zip', outzip, round(outzip.stat().st_size / 2**20), 'MB')
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
