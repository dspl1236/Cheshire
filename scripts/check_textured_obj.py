#!/usr/bin/env python3
"""Compare two textured OBJ files by content: Cheshire's direct writer against Assimp's export of
the same Texturing (CHESHIRE_OBJ_CHECK=1 writes <basename>.assimp.obj beside <basename>.obj).

The two files are not meant to be byte-identical - Assimp deduplicates by value and renumbers,
the direct writer keeps the mesh's own numbering - so each face is resolved to its three
(position, uv) corners and the two multisets of faces, per material, are compared. Values are
compared as float32: that is what Assimp's aiVector3D holds and what both writers print (9
significant digits round-trip a float exactly), so a difference here is a real one.

usage: check_textured_obj.py <cheshire.obj> <assimp.obj>
"""
import struct
import sys
from collections import Counter


def f32(tok: str) -> float:
    """The float32 nearest the printed number, as a Python float, so equal floats compare equal."""
    return struct.unpack("f", struct.pack("f", float(tok)))[0]


def load(path):
    v, vt, faces, mtl = [], [], Counter(), None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                v.append(tuple(f32(t) for t in line.split()[1:4]))
            elif line.startswith("vt "):
                vt.append(tuple(f32(t) for t in line.split()[1:3]))
            elif line.startswith("usemtl "):
                mtl = line.split()[1]
            elif line.startswith("f "):
                corners = []
                for tok in line.split()[1:4]:
                    vi, ti = tok.split("/")[:2]
                    corners.append((v[int(vi) - 1], vt[int(ti) - 1]))
                faces[(mtl, tuple(corners))] += 1
    return len(v), len(vt), faces


def main(a, b):
    nva, nta, fa = load(a)
    nvb, ntb, fb = load(b)
    print(f"{a}: {nva} v, {nta} vt, {sum(fa.values())} faces")
    print(f"{b}: {nvb} v, {ntb} vt, {sum(fb.values())} faces")
    only_a = sum((fa - fb).values())
    only_b = sum((fb - fa).values())
    mats_a = {m for m, _ in fa}
    mats_b = {m for m, _ in fb}
    print(f"materials: {sorted(mats_a)} vs {sorted(mats_b)}")
    print(f"faces only in the first: {only_a}; only in the second: {only_b}")
    if only_a:
        ex = next(k for k in (fa - fb))
        print("  example, first file:", ex)
        print("  nearest in second   :", next((k for k in fb if k[0] == ex[0] and k[1][0][0] == ex[1][0][0]), None))
    ok = only_a == 0 and only_b == 0 and mats_a == mats_b and sum(fa.values()) == sum(fb.values())
    print("IDENTICAL CONTENT" if ok else "CONTENT DIFFERS")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
