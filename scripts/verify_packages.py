#!/usr/bin/env python3
"""Check that release packages actually contain the changes they are supposed to.

Build timestamps are not evidence. A package rebuilt before a late fix looks identical from the
outside, installs cleanly, and passes any test that does not exercise the fixed path - which is how
a v0.2.15 candidate ended up with five of seven packages missing the AC-RANSAC NaN guard while the
one test that ran against them (matching, where no NaN occurs) passed.

So look inside. Each marker is a string literal that exists in the binary only if the corresponding
source is compiled in. Cheap, exact, and it fails loudly.

usage: verify_packages.py <dir of .zip/.tar.gz>  [--markers name=string ...]
"""
import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

# marker -> (needle, at least one of these files must contain it)
MARKERS = {
    "acransac-nan-guard": (b"residual arrays deferred to std::sort",
                           ("aliceVision_sfm", "aliceVision_featureMatching")),
    "acransac-pairsort-escape": (b"CHESHIRE_ACR_PAIRSORT",
                                 ("aliceVision_sfm", "aliceVision_featureMatching")),
    "gpu-matcher": (b"CHESHIRE_GPU_MATCHER",
                    ("aliceVision_matching", "aliceVision_featureMatching")),
    "gpu-sift": (b"popsift", ("aliceVision_feature", "aliceVision_featureExtraction")),
    # v0.2.16: the similarity-map gaussian on the device
    "gpu-sim-blur": (b"CHESHIRE_GPU_BLUR",
                     ("aliceVision_fuseCut", "aliceVision_meshing")),
}


def _stem(name: str) -> str:
    """Linux ships libaliceVision_sfm.so where Windows ships aliceVision_sfm.dll."""
    return name[3:] if name.startswith("lib") else name


def members(pkg: Path):
    """(name, bytes) for the binaries worth scanning, without extracting to disk."""
    if pkg.suffix == ".zip":
        with zipfile.ZipFile(pkg) as z:
            for i in z.infolist():
                n = Path(i.filename).name
                if n.endswith((".dll", ".exe")) and _stem(n).startswith("aliceVision_"):
                    yield n, z.read(i)
    else:
        with tarfile.open(pkg) as t:
            for m in t:
                n = Path(m.name).name
                if m.isfile() and _stem(n).startswith("aliceVision_"):
                    f = t.extractfile(m)
                    if f:
                        yield n, f.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--markers", nargs="*", default=list(MARKERS),
                    help=f"subset of {', '.join(MARKERS)}")
    args = ap.parse_args()

    pkgs = sorted(p for p in args.directory.iterdir()
                  if p.suffix == ".zip" or p.name.endswith(".tar.gz"))
    if not pkgs:
        print(f"no packages in {args.directory}")
        return 1

    bad = 0
    for pkg in pkgs:
        found = {m: False for m in args.markers}
        for name, blob in members(pkg):
            for m in args.markers:
                needle, stems = MARKERS[m]
                if not found[m] and any(_stem(name).startswith(s) for s in stems) and needle in blob:
                    found[m] = True
        missing = [m for m, ok in found.items() if not ok]
        status = "ok" if not missing else "MISSING " + ", ".join(missing)
        print(f"{'ok ' if not missing else 'FAIL'}  {pkg.name:<62} {status if missing else ''}")
        bad += bool(missing)

    print(f"\n{len(pkgs) - bad} of {len(pkgs)} packages carry every marker")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
