#!/usr/bin/env python3
"""Image pairing on a GPS survey: vocabulary tree, exhaustive and GPS radius, compared.

  prepare                       CameraInit and GPU SIFT FeatureExtraction once, into build/gpsbench/zoo/cache
  leg <name> [--method M] [KEY=VALUE ...]
                                ImageMatching (method M, default SequentialAndVocabularyTree, Meshroom's; the
                                environment, e.g. CHESHIRE_GPS_PAIRING_RADIUS=80), FeatureMatching and incrementalSfM
                                into build/gpsbench/zoo/legs/<name>; one JSON line per leg to build/gpsbench/gpsbench.jsonl
  report [--baseline exhaustive]
                                per leg: pairs proposed, pairs verified (geometrically filtered matches), the baseline's
                                verified pairs caught, views placed, times; and the GPS distances of the baseline's
                                verified pairs, which is what a radius has to cover

The survey is OpenDroneMap's "zoo" (CC0 1.0; hub.dronedb.app/r/odm/zoo): 524 photos from a fixed-wing
mapping flight, GPS in every EXIF, in data/scans/zoo/photos. Every binary runs through build/dev-run.cmd, the
development install, as in scripts/sfmbench.py, whose FeatureMatching options and SfM command this reuses.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfmbench  # noqa: E402

ROOT = sfmbench.ROOT
PHOTOS = ROOT / "data" / "scans" / "zoo" / "photos"
BASE = sfmbench.B / "gpsbench" / "zoo"
CACHE = BASE / "cache"
LEGS = BASE / "legs"
JSONL = sfmbench.B / "gpsbench" / "gpsbench.jsonl"
DESCRIBER = "sift"


def prepare() -> None:
    ci = CACHE / "CameraInit" / "cameraInit.sfm"
    if not ci.is_file():
        print(f"CameraInit on {PHOTOS}")
        sfmbench.sh(["aliceVision_cameraInit", "--imageFolder", str(PHOTOS), "--sensorDatabase", str(sfmbench.SHARE / "cameraSensors.db"),
                     "--defaultFieldOfView", "45.0", "--groupCameraFallback", "folder",
                     "--rawColorInterpretation", "LibRawWhiteBalancing", "--viewIdMethod", "metadata", "--verboseLevel", "info",
                     "--allowSingleView", "1", "--output", str(ci)], ci.parent / "log")
    fe = CACHE / "FeatureExtraction"
    if not any(fe.glob("*.feat")):
        print("FeatureExtraction (sift, GPU)")
        dt = sfmbench.sh(["aliceVision_featureExtraction", "--input", str(ci), "--describerTypes", DESCRIBER, "--describerPreset", "normal",
                          "--describerQuality", "normal", "--contrastFiltering", "GridSort", "--gridFiltering", "True",
                          "--workingColorSpace", "sRGB", "--forceCpuExtraction", "False", "--maxThreads", "0", "--verboseLevel", "info",
                          "--output", str(fe)], fe / "log")
        print(f"  {dt:.0f} s")
    print(f"ready: {CACHE}")


def leg(name: str, method: str, envs: dict[str, str]) -> None:
    ci = CACHE / "CameraInit" / "cameraInit.sfm"
    fe = CACHE / "FeatureExtraction"
    out = LEGS / name
    if out.exists():
        sys.exit(f"{out} exists; pick another name")
    im, fm, sfm = out / "ImageMatching", out / "FeatureMatching", out / "sfm"
    for d in (im, fm, sfm):
        d.mkdir(parents=True)
    pairs = im / "imageMatches.txt"
    print(f"[{name}] ImageMatching ({method}, {envs or 'no environment'})")
    t_im = sfmbench.sh(["aliceVision_imageMatching", "--input", str(ci), "--featuresFolders", str(fe), "--method", method,
                        "--tree", str(sfmbench.SHARE / "vlfeat_K80L3.SIFT.tree"), "--weights", "", "--minNbImages", "200",
                        "--maxDescriptors", "500", "--nbMatches", "40", "--nbNeighbors", "5", "--verboseLevel", "info",
                        "--output", str(pairs)], im / "log", envs)
    print(f"[{name}] FeatureMatching ({count_pairs(pairs)} pairs)")
    t_fm = sfmbench.sh(["aliceVision_featureMatching", "--input", str(ci), "--featuresFolders", str(fe), "--imagePairsList", str(pairs),
                        "--describerTypes", DESCRIBER, "--photometricMatchingMethod", "ANN_L2", "--geometricEstimator", "acransac",
                        "--geometricFilterType", "fundamental_matrix", "--distanceRatio", "0.8", "--maxIteration", "2048",
                        "--geometricError", "0.0", "--knownPosesGeometricErrorMax", "5.0", "--minRequired2DMotion", "-1.0",
                        "--maxMatches", "0", "--savePutativeMatches", "False", "--crossMatching", "False", "--guidedMatching", "False",
                        "--matchFromKnownCameraPoses", "False", "--exportDebugFiles", "False", "--verboseLevel", "info",
                        "--output", str(fm)], fm / "log")
    print(f"[{name}] incrementalSfM")
    t_sfm = sfmbench.sh(sfmbench.sfm_cmd(ci, fe, fm, DESCRIBER, sfm), sfm / "sfm.log")
    rec = {"leg": name, "method": method, "env": envs, "pairs": count_pairs(pairs), "verified": len(verified_pairs(fm)),
           "imagematching_s": round(t_im, 1), "featurematching_s": round(t_fm, 1), "sfm_s": round(t_sfm, 1),
           "views": count_views(ci), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    rec.update({k: v for k, v in sfmbench.parse_log(sfm / "sfm.log").items() if k in ("poses", "landmarks", "rmse")})
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  {rec}")


def count_pairs(pairs: Path) -> int:
    """imageMatches.txt: each line is a view followed by the views it is paired with."""
    n = 0
    for line in pairs.read_text(encoding="utf-8").splitlines():
        ids = line.split()
        n += max(len(ids) - 1, 0)
    return n


def verified_pairs(fm: Path) -> set[tuple[str, str]]:
    """The pairs with geometrically verified matches, from FeatureMatching's *.matches.txt: blocks of
    'I J', a describer-type count, then per type 'type count' and that many index lines."""
    got: set[tuple[str, str]] = set()
    for f in fm.glob("*.matches.txt"):
        tok = f.read_text(encoding="utf-8").split()
        i = 0
        while i < len(tok):
            a, b = tok[i], tok[i + 1]
            ntypes = int(tok[i + 2])
            i += 3
            total = 0
            for _ in range(ntypes):
                n = int(tok[i + 1])
                total += n
                i += 2 + 2 * n
            if total > 0:
                got.add((a, b) if a < b else (b, a))
    return got


def count_views(ci: Path) -> int:
    return len(json.loads(ci.read_text(encoding="utf-8")).get("views", []))


def view_positions(ci: Path) -> dict[str, tuple[float, float]]:
    """viewId -> local east/north metres, from each view's EXIF GPS as CameraInit recorded it."""
    ll = {}
    for v in json.loads(ci.read_text(encoding="utf-8")).get("views", []):
        m = v.get("metadata", {})
        try:
            lat = dms(m["GPS:Latitude"], m.get("GPS:LatitudeRef", "N"))
            lon = dms(m["GPS:Longitude"], m.get("GPS:LongitudeRef", "E"))
        except (KeyError, ValueError):
            continue
        ll[str(v["viewId"])] = (lat, lon)
    if not ll:
        return {}
    lat0 = math.radians(sum(p[0] for p in ll.values()) / len(ll))
    return {k: (lon * 111320 * math.cos(lat0), lat * 110540) for k, (lat, lon) in ll.items()}


def dms(s: str, ref: str) -> float:
    """AliceVision keeps EXIF GPS as 'd, m, s' strings."""
    parts = [float(x) for x in re.split(r"[,\s]+", s.strip()) if x]
    v = parts[0] + (parts[1] / 60 if len(parts) > 1 else 0) + (parts[2] / 3600 if len(parts) > 2 else 0)
    return -v if ref.strip().upper() in ("S", "W") else v


def report(baseline: str) -> None:
    ci = CACHE / "CameraInit" / "cameraInit.sfm"
    rows = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines()] if JSONL.is_file() else []
    base = verified_pairs(LEGS / baseline / "FeatureMatching") if (LEGS / baseline).is_dir() else set()
    pos = view_positions(ci)
    print(f"| leg | pairs proposed | verified | {baseline}'s verified caught | views placed | FeatureMatching | SfM |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        mine = verified_pairs(LEGS / r["leg"] / "FeatureMatching")
        caught = f"{len(mine & base)} of {len(base)}" if base else "-"
        print(f"| {r['leg']} | {r['pairs']:,} | {r['verified']:,} | {caught} | {r.get('poses', '?')} of {r['views']} "
              f"| {r['featurematching_s']:.0f} s | {r['sfm_s']:.0f} s |")
    if base and pos:
        d = sorted(math.dist(pos[a], pos[b]) for a, b in base if a in pos and b in pos)
        if d:
            q = lambda p: d[min(len(d) - 1, int(p / 100 * len(d)))]  # noqa: E731
            print(f"\n{baseline}'s {len(d)} verified pairs by GPS distance: median {q(50):.0f} m, p90 {q(90):.0f} m, "
                  f"p99 {q(99):.0f} m, max {d[-1]:.0f} m")
            for r in (40, 60, 80, 100, 120, 160):
                print(f"  within {r} m: {sum(x <= r for x in d)} ({100 * sum(x <= r for x in d) / len(d):.1f} %)")


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("prepare", "leg", "report"):
        sys.exit(__doc__)
    if argv[0] == "prepare":
        prepare()
    elif argv[0] == "leg":
        name = argv[1] if len(argv) > 1 else sys.exit("leg <name>")
        method = argv[argv.index("--method") + 1] if "--method" in argv else "SequentialAndVocabularyTree"
        envs = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in argv[2:] if "=" in a and not a.startswith("--")}
        leg(name, method, envs)
    else:
        report(argv[argv.index("--baseline") + 1] if "--baseline" in argv else "exhaustive")


if __name__ == "__main__":
    main(sys.argv[1:])
