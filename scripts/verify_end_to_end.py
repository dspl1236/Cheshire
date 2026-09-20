#!/usr/bin/env python3
"""Run complete Meshroom pipelines on a Cheshire package and check every GPU port took part.

    verify_end_to_end.py <meshroom dir> <package dir> <photos> <out root> [config ...]

Until 2026-09-20 nothing had ever run a whole Meshroom graph on a Cheshire package. The stage gates
(verify_bundle_stages.py, verify-cuda-stages.ps1) drive one node at a time from a reference cache,
and the reference caches under build/meshroom were themselves made with DepthMap paired and nothing
else - every other node in them ran stock Meshroom, which is why they carry no [cheshire] lines at
all. So the GPU filter, meshing and texturing ports had never once run inside a pipeline; they had
only ever been driven by hand.

This pairs the Meshroom install with the package, runs meshroom_batch end to end under several
parameter sets, and for each run checks BOTH that a textured mesh came out AND that every paired
node logged its own port's line. That second half is the point: a run that silently fell back to
Meshroom's own binaries produces a perfectly good mesh, and file counts cannot tell the difference.

The last config inverts the test. It sets every CHESHIRE_GPU_* switch to 0 and requires each port
to log its DISABLED line instead, because a package whose CPU fallback is broken passes every other
check here and fails on the first machine without a supported card.

Markers follow the rule the stage gates arrived at the hard way: take the line from the success
branch of the port's available(), and include enough of it that the disabled branch cannot match.
Bare 'GPU brute-force' matches "GPU brute-force disabled by CHESHIRE_GPU_MATCHER=0".
"""
import os, re, shutil, subprocess, sys, time
from pathlib import Path

# The line each port prints when its GPU path is live. Absent = that node ran something else.
GPU_MARKERS = {
    "FeatureExtraction": r"Choosing device \d+:",
    "FeatureMatching":   r"GPU brute-force L2 2-NN on",
    "DepthMap":          r"Number of GPU devices",
    "DepthMapFilter":    r"depth map filter: group votes on",
    "Meshing":           r"meshing votes: ray marching on",
    "Texturing":         r"texturing: pyramid \+ rasterisation on",
}
# ... and the line it prints when switched off. FeatureExtraction and DepthMap have no such switch,
# so they stay on the GPU in the fallback run and keep their markers.
CPU_MARKERS = {
    "FeatureExtraction": GPU_MARKERS["FeatureExtraction"],
    "DepthMap":          GPU_MARKERS["DepthMap"],
    "FeatureMatching":   r"GPU brute-force disabled by CHESHIRE_GPU_MATCHER=0",
    "DepthMapFilter":    r"depth map filter: disabled by CHESHIRE_GPU_FILTER=0",
    "Meshing":           r"meshing votes: disabled by CHESHIRE_GPU_VOTE=0",
    "Texturing":         r"texturing: disabled by CHESHIRE_GPU_TEX=0",
}

# Meshroom's FeatureExtraction defaults to dspsift, which goes through vlfeat on the CPU whatever
# the flags - so every config asks for sift, or GPU SIFT is never exercised and the run says
# nothing about PopSIFT. That is the shape of the bug v0.2.17 shipped with.
#
# forceCpuExtraction defaults to TRUE in Meshroom 2023.3, and with it set the node logs [cpu] and
# never touches PopSIFT. meshroom-pair.cmd's own header warns about exactly this - and the first
# run of this script fell into it anyway, reporting a missing GPU SIFT marker as if the package
# were at fault.
SIFT = ["FeatureExtraction:describerTypes=sift", "FeatureExtraction:forceCpuExtraction=False"]

CONFIGS = {
    # Stock settings, to establish that the paired pipeline works at all.
    "base": dict(overrides=SIFT, env={}),
    # Many small tiles: the tiled path is where the memory bridge and the per-tile depth lists live,
    # and a 42-tile run behaves differently from a 1-tile one.
    "tiles": dict(overrides=SIFT + [
        "DepthMap:tileWidth=512", "DepthMap:tileHeight=512", "DepthMap:tileOverlapPercentage=10",
        "DepthMap:autoAdjustSmallImage=False", "DepthMap:downscale=2"], env={}),
    # The other end: one coarse tile, fewer cameras, a smaller point budget.
    "coarse": dict(overrides=SIFT + [
        "DepthMap:downscale=4", "DepthMap:sgmDepthListPerTile=False", "DepthMap:nbNearestCams=6",
        "Meshing:maxPoints=300000"], env={}),
    # A bigger atlas and more frequency bands, which moves the texturing port's allocations.
    "texbig": dict(overrides=SIFT + [
        "Texturing:textureSide=8192", "Texturing:nbBand=3"], env={}),
    # Every GPU port switched off. Must still produce a mesh, and must say it went to the CPU.
    "cpufallback": dict(overrides=SIFT, markers="cpu", env={
        "CHESHIRE_GPU_MATCHER": "0", "CHESHIRE_GPU_FILTER": "0",
        "CHESHIRE_GPU_VOTE": "0", "CHESHIRE_GPU_TEX": "0"}),
}


def node_logs(cache: Path, node: str) -> str:
    """Every log Meshroom wrote for a node, concatenated. Chunked nodes write 0.log, 1.log, ...;
    single-chunk nodes write a file plainly called 'log'."""
    d = cache / node
    if not d.is_dir():
        return ""
    text = []
    for uid in d.iterdir():
        if not uid.is_dir():
            continue
        for f in uid.iterdir():
            if f.is_file() and (f.name == "log" or f.suffix == ".log"):
                text.append(f.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(text)


def run_one(name, cfg, meshroom: Path, photos: Path, outroot: Path) -> bool:
    out = outroot / name
    cache = out / "cache"
    shutil.rmtree(out, ignore_errors=True)
    (out / "out").mkdir(parents=True)

    env = dict(os.environ)
    # Force the paired launcher to the Cheshire package. Its default is 'auto', which hands the node
    # back to Meshroom's own CUDA binary whenever an NVIDIA card is present - correct when Cheshire
    # was AMD-only, and precisely wrong when the package under test is itself the CUDA one.
    env["CHESHIRE_DEPTHMAP"] = "hip"
    env.update(cfg["env"])

    cmd = [str(meshroom / "meshroom_batch.exe"),
           "--input", str(photos), "--output", str(out / "out"), "--cache", str(cache),
           "--pipeline", "photogrammetry"]
    if cfg["overrides"]:
        cmd += ["--paramOverrides"] + cfg["overrides"]

    log = out / "meshroom_batch.log"
    t0 = time.time()
    with log.open("w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    secs = round(time.time() - t0)

    mesh = list((out / "out").glob("*.obj"))
    # Any extension: Meshroom's Texturing writes texture_1001.exr by default, not .png, so globbing
    # for .png reported a perfectly good textured mesh as having no textures.
    tex = list((out / "out").glob("texture_*.*"))
    markers = CPU_MARKERS if cfg.get("markers") == "cpu" else GPU_MARKERS
    missing = [n for n, pat in markers.items() if not re.search(pat, node_logs(cache, n))]

    ok = rc == 0 and mesh and tex and not missing
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:<14} exit={rc:<4} mesh={len(mesh)} tex={len(tex)} "
          f"ports={len(markers) - len(missing)}/{len(markers)}  {secs}s")
    if missing:
        print(f"        no Cheshire line from: {', '.join(missing)}"
              f" - that node ran something else, or its port fell silent")
    if rc != 0:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-4:]:
            print(f"        {line[:110]}")
    return bool(ok)


def main(argv):
    if len(argv) < 5:
        print(__doc__)
        return 2
    meshroom, package, photos, outroot = (Path(a).resolve() for a in argv[1:5])
    names = argv[5:] or list(CONFIGS)
    bad = [n for n in names if n not in CONFIGS]
    if bad:
        sys.exit(f"unknown config(s): {', '.join(bad)} - have {', '.join(CONFIGS)}")

    pair = package / "meshroom-pair.cmd"
    if not pair.exists():
        pair = Path(__file__).resolve().parent / "windows" / "meshroom-pair.cmd"
    if not pair.exists():
        sys.exit("meshroom-pair.cmd not found in the package or in scripts/windows")

    print(f"=== meshroom {meshroom}")
    print(f"=== package  {package}")
    print(f"=== photos   {photos} ({len(list(photos.glob('*.[jJ][pP][gG]')))} jpg)")
    p = subprocess.run([str(pair), str(meshroom), str(package)], capture_output=True, text=True, shell=True)
    print(p.stdout.strip() or p.stderr.strip())
    if p.returncode != 0:
        return 2

    outroot.mkdir(parents=True, exist_ok=True)
    try:
        print()
        results = {n: run_one(n, CONFIGS[n], meshroom, photos, outroot) for n in names}
    finally:
        # Always put Meshroom back: pairing renames its binaries in place, and a half-paired install
        # is a trap for whoever opens Meshroom next.
        u = subprocess.run([str(pair), str(meshroom), "--unpair"], capture_output=True, text=True, shell=True)
        print("\n" + (u.stdout.strip() or u.stderr.strip()))

    passed = sum(1 for v in results.values() if v)
    print(f"\n{passed} of {len(results)} pipelines ran end to end on this package")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
