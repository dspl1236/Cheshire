#!/usr/bin/env python3
"""What CPU does a package actually require? Read it out of the instructions, not the build flags.

Two reasons this is not the same question as "what did we pass to the compiler":

  - A dependency built elsewhere can raise the floor silently. The flags say /arch:AVX and one
    vcpkg library brings AVX2 along, and nothing notices until it faults on a user's machine.
  - It lets a package's floor be *proved* without owning the hardware. A build containing no AVX
    instruction anywhere runs on a Core 2, whether or not anyone has a Core 2 to try it on.

usage: check_isa_floor.py <objdump> <package.zip|.tar.gz> [--expect v1|v2|avx|v3|v4]

Tier names follow the x86-64 psABI levels, with one addition. AMD's Bulldozer and Intel's Sandy
Bridge have AVX but not AVX2, which is not a psABI level and is exactly the tier the hip6.2
packages target, so it is called "avx" and sits between v2 and v3.
"""
import argparse
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

# Highest first. Each pattern matches only instructions introduced at that level.
TIERS = [
    ("v4",  re.compile(r"\bzmm\d|\{%?k[1-7]\}")),
    ("v3",  re.compile(r"\b(vfm(add|sub|nmadd|nmsub)\w*"
                       r"|vperm2i128|vpermd|vpermq|vpermps|vpermpd"
                       r"|vpbroadcast\w+|vpgather\w+|vps(ll|rl|ra)vd"
                       r"|vinserti128|vextracti128"
                       r"|shlx|shrx|sarx|bzhi|pdep|pext|mulx|rorx)\b")),
    ("avx", re.compile(r"\bv[a-z0-9]+\s+[^\n]*%?[xy]mm")),
    ("v2",  re.compile(r"\b(pblendvb|blendvps|blendvpd|roundps|roundpd|pmuldq|ptest"
                       r"|pcmpgtq|pcmpestri|pcmpistri|dpps|dppd|popcnt|crc32|pminsd|pmaxsd)\b")),
    ("v1",  re.compile(r"\b(movaps|movdqa|paddd|mulpd|addpd|xorps)\b")),
]
ORDER = [t for t, _ in reversed(TIERS)]  # v1 .. v4

FROM = {
    "v1":  "any x86-64: Athlon 64 (2003), Pentium 4 EM64T",
    "v2":  "Intel Nehalem (2008), AMD Bulldozer (2011)",
    "avx": "Intel Sandy Bridge (2011), AMD Bulldozer (2011)",
    "v3":  "Intel Haswell (2013), AMD Excavator (2015) / Zen (2017)",
    "v4":  "Intel Skylake-X (2017), AMD Zen 4 (2022)",
}


def binaries(pkg: Path, out: Path):
    """Extract the aliceVision binaries worth disassembling; returns their paths."""
    got = []
    if pkg.suffix == ".zip":
        with zipfile.ZipFile(pkg) as z:
            for i in z.infolist():
                n = Path(i.filename).name
                if n.endswith((".dll", ".exe")) and n.startswith("aliceVision_"):
                    (out / n).write_bytes(z.read(i))
                    got.append(out / n)
    else:
        with tarfile.open(pkg) as t:
            for m in t:
                n = Path(m.name).name
                stem = n[3:] if n.startswith("lib") else n
                if m.isfile() and stem.startswith("aliceVision_"):
                    f = t.extractfile(m)
                    if f:
                        (out / n).write_bytes(f.read())
                        got.append(out / n)
    return got


def floor_of(objdump: str, path: Path) -> str:
    text = subprocess.run([objdump, "-d", "--no-show-raw-insn", str(path)],
                          capture_output=True, text=True, errors="replace").stdout
    for name, rx in TIERS:
        if rx.search(text):
            return name
    return "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("objdump")
    ap.add_argument("package", type=Path)
    ap.add_argument("--expect", choices=ORDER)
    ap.add_argument("--quiet", action="store_true", help="only report the highest tier")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        files = binaries(args.package, Path(td))
        if not files:
            print(f"no aliceVision binaries in {args.package.name}")
            return 1
        worst, where = "none", None
        for f in sorted(files):
            t = floor_of(args.objdump, f)
            if t != "none" and (worst == "none" or ORDER.index(t) > ORDER.index(worst)):
                worst, where = t, f.name
            if not args.quiet:
                print(f"  {f.name:<52} {t}")

    print(f"\n{args.package.name}")
    print(f"  requires: {worst}  ({FROM.get(worst, '?')})")
    if where:
        print(f"  set by:   {where}")
    if args.expect:
        ok = worst == args.expect
        print(f"  expected: {args.expect} -> {'ok' if ok else 'MISMATCH'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
