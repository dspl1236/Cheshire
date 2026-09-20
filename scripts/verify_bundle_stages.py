#!/usr/bin/env python3
"""Run every GPU-bearing stage from a bundle and report which ones work.

    verify_bundle_stages.py <bundle dir> <meshroom cache dir>

v0.2.17 shipped with GPU SIFT broken because the release check ran Meshing and Texturing through
the bundle and nothing else - PopSIFT crashes from a generic code object, and no test asked the
bundle for sift features. The lesson is not about generics: a payload swap changes the code objects
of EVERY stage, so a bundle has to be checked against every stage that has one, whether or not that
stage was touched by the release.

Small ranges throughout. This answers "does each stage run from this package", not "is it fast" or
"is the output right" - those are the fleet checksums, which this does not replace.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def node_dir(cache: Path, name: str) -> str:
    ds = sorted(d for d in (cache / name).iterdir() if d.is_dir()) if (cache / name).is_dir() else []
    if not ds:
        sys.exit(f"no {name} folder under {cache}")
    return ds[0].as_posix()


def main(argv):
    if len(argv) != 3:
        print(__doc__); return 2
    bundle, cache = Path(argv[1]).resolve(), Path(argv[2]).resolve()
    runner = bundle / "cheshire-run.cmd"
    if not runner.exists():
        sys.exit(f"no cheshire-run.cmd in {bundle}")

    out = ROOT / "build" / "bundle-stages"
    out.mkdir(parents=True, exist_ok=True)

    which = subprocess.run([str(runner), "--which"], capture_output=True, text=True, shell=True)
    payload = which.stdout.strip() or "(unknown)"
    print(f"=== bundle  {bundle.name}")
    print(f"=== payload {payload}\n")

    ci = node_dir(cache, "CameraInit") + "/cameraInit.sfm"
    fe = node_dir(cache, "FeatureExtraction")
    im = node_dir(cache, "ImageMatching") + "/imageMatches.txt"
    sfm = node_dir(cache, "StructureFromMotion") + "/sfm.abc"
    pds = node_dir(cache, "PrepareDenseScene")
    dmf = node_dir(cache, "DepthMapFilter")

    # Every stage that carries GPU code. FeatureExtraction must use "sift": dspsift goes through
    # vlfeat on the CPU whatever the flags, so only "sift" actually exercises PopSIFT - which is
    # precisely why the broken build passed every earlier check.
    #
    # Each entry gives its own --output, because Meshing wants a file and the rest want a folder.
    #
    # The last field is a regex that MUST appear in the stage's log. Counting output files is not
    # enough and has already fooled this check once: Texturing writes texturedMesh.obj/.mtl and
    # exits 0 while generating no textures at all, because --colorMappingFileType defaults to NONE
    # (found 2026-09-21 on the CUDA bundle). A bundle whose texturing is entirely dead would have
    # passed. Where a stage carries one of our GPU ports, the regex is that port announcing itself,
    # so this also catches a silent fall back to the CPU path.
    # Markers must be lines the port emits UNCONDITIONALLY. Three of these were wrong when
    # first written and none had been run: 'filter votes GPU' and 'knn check: identical'
    # only appear under CHESHIRE_GPU_FILTER_DEBUG / CHESHIRE_GPU_VIS_CHECK, so both would
    # have reported a healthy GPU run as a failure, and '(?i)popsift|gpu' matched any path
    # containing 'gpu' - a directory named gpusift was enough to pass a broken stage.
    stages = [
        ("FeatureExtraction (sift/PopSIFT)", "aliceVision_featureExtraction", [
            "--input", ci, "--describerTypes", "sift", "--describerPreset", "normal",
            "--describerQuality", "normal", "--contrastFiltering", "GridSort",
            "--gridFiltering", "True", "--forceCpuExtraction", "False",
            "--rangeStart", "0", "--rangeSize", "2"], "*.feat", None, r"Choosing device \d+:"),
        ("FeatureMatching (GPU matcher)", "aliceVision_featureMatching", [
            "--input", ci, "--featuresFolders", fe, "--imagePairsList", im,
            "--describerTypes", "dspsift", "--geometricEstimator", "acransac",
            "--geometricFilterType", "fundamental_matrix",
            "--rangeStart", "0", "--rangeSize", "40"], "*.txt", None, r"GPU brute-force"),
        ("DepthMap", "aliceVision_depthMapEstimation", [
            "--input", sfm, "--imagesFolder", pds, "--downscale", "2",
            "--rangeStart", "0", "--rangeSize", "2"], "*.exr", None, r"Number of GPU devices"),
        ("DepthMapFilter", "aliceVision_depthMapFiltering", [
            "--input", sfm, "--depthMapsFolder", dmf,
            "--rangeStart", "0", "--rangeSize", "2"], "*.exr", None, r"depth map filter: group votes on"),
        # Meshing and Texturing have no range option, so these are full runs and the slow part of
        # this script. They are here anyway: leaving out the stages that happened to be validated
        # elsewhere is exactly how GPU SIFT shipped broken.
        ("Meshing", "aliceVision_meshing", [
            "--input", sfm, "--depthMapsFolder", dmf,
            "--outputMesh", (ROOT / "build/bundle-stages/aliceVision_meshing/mesh.obj").as_posix(),
            "--maxPoints", "1000000", "--maxInputPoints", "10000000",
            "--estimateSpaceFromSfM", "True"], "*.abc", "densePointCloud.abc", r"meshing votes: ray marching on"),
        # Texturing needs Meshing's densePointCloud.abc (not the SfM) for visibility, and
        # --colorMappingFileType or it writes no textures. Both were wrong the first time.
        ("Texturing (GPU)", "aliceVision_texturing", [
            "--input", str(ROOT / "build/bundle-stages/aliceVision_meshing/densePointCloud.abc"),
            "--inputMesh", str(ROOT / "build/bundle-stages/aliceVision_meshing/mesh.obj"),
            "--imagesFolder", pds, "--colorMappingFileType", "png"], "texture_*.png", None,
            r"cheshire texturing profile"),
    ]

    results = []
    for label, exe, args, glob, outfile, expect in stages:
        dest = out / exe
        if dest.exists():
            for f in dest.rglob("*"):
                if f.is_file(): f.unlink()
        dest.mkdir(parents=True, exist_ok=True)
        log = out / f"{exe}.log"
        target = (dest / outfile).as_posix() if outfile else dest.as_posix()
        cmd = [str(runner), exe] + args + ["--output", target, "--verboseLevel", "info"]
        t0 = time.time()
        with log.open("w", encoding="utf-8", errors="replace") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, shell=True).returncode
        made = len(list(dest.glob(glob)))
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        saw = bool(re.search(expect, text)) if expect else True
        ok = rc == 0 and made > 0 and saw
        results.append((label, ok, rc, made, round(time.time() - t0)))
        print(f"  {'ok ' if ok else 'FAIL'}  {label:<34} exit={rc:<12} produced {made:<4} {results[-1][4]}s")
        if not ok:
            if rc == 0 and made > 0 and not saw:
                print(f"        ran and produced output, but nothing matched /{expect}/ -"
                      f" the GPU path is silent or the stage did no real work")
            print(f"        {log}")
            for line in log.read_text(encoding='utf-8', errors='replace').splitlines()[-3:]:
                print(f"        {line[:100]}")

    bad = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(bad)} of {len(results)} stages ran from this bundle")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
