"""Compare two Texturing outputs texture by texture.

For every texture_<udim>.exr present in both folders: pixels that differ at all, the largest and mean
absolute difference over the differing pixels, pixels beyond 1e-3 absolute, and how many pixels
carry colour on each side. Textures are written as half floats, so a difference of one float ulp
in the accumulation usually rounds away and "identical" is the common outcome for texels touched by
one triangle per camera; chart-edge texels (two triangles of one camera) can differ by add order.

    python scripts/compare_textures.py <dirA> <dirB>
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import OpenEXR


def read_exr(path: Path) -> np.ndarray:
    with OpenEXR.File(str(path)) as f:
        ch = f.channels()
        if "RGB" in ch:
            return np.asarray(ch["RGB"].pixels, dtype=np.float32)
        return np.stack([np.asarray(ch[c].pixels, dtype=np.float32) for c in ("R", "G", "B")], axis=-1)


def main() -> int:
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    names = sorted(p.name for p in a.glob("texture_*.exr") if (b / p.name).exists())
    if not names:
        print("no common texture files"); return 1
    worst = 0.0
    for n in names:
        x, y = read_exr(a / n), read_exr(b / n)
        if x.shape != y.shape:
            print(f"{n}: shape {x.shape} vs {y.shape}"); worst = float("inf"); continue
        d = np.abs(x.astype(np.float64) - y.astype(np.float64)).max(axis=-1)
        diff = d > 0
        nd = int(diff.sum())
        valid_a, valid_b = int((x != 0).any(axis=-1).sum()), int((y != 0).any(axis=-1).sum())
        mx = float(d.max()) if nd else 0.0
        worst = max(worst, mx)
        mean = float(d[diff].mean()) if nd else 0.0
        big = int((d > 1e-3).sum())
        print(f"{n}: {x.shape[1]}x{x.shape[0]}, coloured {valid_a} / {valid_b}, differing {nd} ({100.0 * nd / d.size:.4f} %), max {mx:.3g}, mean {mean:.3g}, beyond 1e-3: {big}")
    print(f"worst absolute difference over {len(names)} textures: {worst:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
