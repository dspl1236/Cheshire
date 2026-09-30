#!/usr/bin/env python3
"""Assemble the bundled Windows package from built payloads (docs/16).

    assemble_bundle.py <out.zip> <rocm7.2 package> <hip6.2 package>

The two packages are the family bases, giving the bundle its CPU binaries, vcpkg runtime, share/
tree and HIP runtime. Every GPU target comes from build/payload/<family>/<target>/, so the bases
only need to exist for one target each.

The UNTESTED lists (gpu/<family>/UNTESTED, the targets never run on hardware here) are written by
bundle_windows.py itself since 0.3.8, from its VALIDATED table: this script used to write them after
the fact, the release chains called bundle_windows.py directly, and v0.3.2 to v0.3.7 shipped without
them. Of eleven targets, three have been run on a real card, and pretending otherwise would be the
one thing this project cannot afford.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAYLOADS = ROOT / 'build' / 'payload'

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

    specs = []
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

    cmd = [sys.executable, str(ROOT / 'scripts' / 'bundle_windows.py'), str(outzip)] + specs
    print('  ' + ' \\\n  '.join(cmd[2:]))
    return subprocess.run(cmd).returncode

if __name__ == '__main__':
    sys.exit(main(sys.argv))
