#!/usr/bin/env python3
"""FeatureMatching benchmark on a set sfmbench.py prepared, with the development install - for the geometric filtering
(AC-RANSAC), whose output has to stay byte for byte the same.

  run <set> --tag <name> [--maxIteration N] [--ref <tag>] [KEY=VALUE ...]
        aliceVision_featureMatching from build/sfmbench/<set>/cache with sfmbench's options (Meshroom 2023.3's; 2025.1
        raised maxIteration from 2048 to 50000), into build/fmbench/<set>/<tag>; prints the wall time, the log's
        geometric filtering time and one digest over every match file, and with --ref whether they are byte for byte
        those of an earlier tag.

Every binary is started through build/dev-run.cmd (the development install), as sfmbench.py does.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "build"
DEVRUN = B / "dev-run.cmd"


def digest(d: Path) -> tuple[str, int, dict]:
    files = sorted(d.glob("*.matches.txt"))
    per = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    h = hashlib.sha256("".join(f"{k}={v}\n" for k, v in per.items()).encode()).hexdigest()[:16]
    return h, len(files), per


def main(argv: list[str]) -> None:
    if len(argv) < 2 or argv[0] != "run":
        sys.exit(__doc__)
    s = argv[1]
    tag = argv[argv.index("--tag") + 1] if "--tag" in argv else sys.exit("--tag <name> required")
    iters = argv[argv.index("--maxIteration") + 1] if "--maxIteration" in argv else "2048"
    ref = argv[argv.index("--ref") + 1] if "--ref" in argv else None
    envs = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in argv[2:] if "=" in a and not a.startswith("--")}
    cache = B / "sfmbench" / s / "cache"
    ci = cache / "CameraInit" / "cameraInit.sfm"
    fe, pairs = cache / "FeatureExtraction", cache / "ImageMatching" / "imageMatches.txt"
    describer = (B / "sfmbench" / s / "describer.txt").read_text(encoding="utf-8").strip()
    out = B / "fmbench" / s / tag
    if out.exists():
        for f in out.iterdir():
            f.unlink()
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["aliceVision_featureMatching", "--input", str(ci), "--featuresFolders", str(fe), "--imagePairsList", str(pairs),
           "--describerTypes", describer, "--photometricMatchingMethod", "ANN_L2", "--geometricEstimator", "acransac",
           "--geometricFilterType", "fundamental_matrix", "--distanceRatio", "0.8", "--maxIteration", iters,
           "--geometricError", "0.0", "--knownPosesGeometricErrorMax", "5.0", "--minRequired2DMotion", "-1.0",
           "--maxMatches", "0", "--savePutativeMatches", "False", "--crossMatching", "False", "--guidedMatching", "False",
           "--matchFromKnownCameraPoses", "False", "--exportDebugFiles", "False", "--verboseLevel", "info",
           "--output", str(out)]
    env = dict(os.environ, **envs)
    log = out / "fm.log"
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        r = subprocess.run(["cmd", "/c", str(DEVRUN)] + cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
    wall = time.perf_counter() - t0
    if r.returncode:
        sys.exit(f"featureMatching failed ({r.returncode}); see {log}")
    text = log.read_text(encoding="utf-8", errors="replace")

    def stamp(pattern: str) -> float | None:  # the first log line matching, as seconds since midnight
        m = re.search(r"^\[(\d+):(\d+):(\d+\.\d+)\][^\n]*" + pattern, text, re.M)
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None

    g0, g1 = stamp("Geometric filtering: using"), stamp("geometric image pair matches")
    geo = f"{(g1 - g0) % 86400:7.1f} s" if g0 is not None and g1 is not None else "      ?"
    h, n, per = digest(out)
    line = f"{s} {tag:<16} maxIteration {iters:>6} | wall {wall:7.1f} s | geometric filtering {geo} | match files {n}, digest {h}"
    if ref:
        rh, rn, rper = digest(B / "fmbench" / s / ref)
        same = rper == per
        line += f" | {'IDENTICAL to' if same else 'DIFFERENT from'} {ref}"
        if not same:
            line += f" ({sum(1 for k in per if rper.get(k) != per[k])} files differ)"
    print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
