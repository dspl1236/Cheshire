#!/usr/bin/env python3
"""StructureFromMotion benchmark on a local photo set, with the development install.

  prepare <set> [--describer dspsift]   CameraInit, FeatureExtraction, ImageMatching, FeatureMatching once,
                                        into build/sfmbench/<set>/cache (steps whose output exists are skipped)
  run <set> --tag <name> [KEY=VALUE ...] [--repeat N] [--json] [-- <extra incrementalSfM options>]
                                        --json writes sfm.sfm (JSON) instead of sfm.abc, whose Alembic
                                        header carries the date, so two runs can be compared byte for byte
                                        incrementalSfM from that cache with Meshroom 2023.3's default node
                                        options, CHESHIRE_BA_PROFILE=1 and the given environment, into
                                        build/sfmbench/<set>/runs/<tag>[-N]; one JSON line per run is
                                        appended to build/sfmbench/bench.jsonl and a summary printed
  show                                  the bench.jsonl lines, one per row

  set     41 (data/monstree/full, 41 photos) | eb (data/scans/enginebay/photos, 107) | mini6
          fd (the kept 884-view False Door cache, build/e2e-falsedoor-fix; no prepare step, dspsift)

The SfM command line is the one Meshroom 2023.3 wrote for the mini6 job (build/meshroom/job-mini6)
with the paths swapped, so a run here is what a paired Meshroom would run. Every binary is started
through build/dev-run.cmd, i.e. the development install and its DLL layers.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "build"
BENCH = B / "sfmbench"
DEVRUN = B / "dev-run.cmd"
SHARE = B / "meshroom" / "Meshroom-2023.3.0" / "aliceVision" / "share" / "aliceVision"
SETS = {
    "41": ROOT / "data" / "monstree" / "full",
    "eb": ROOT / "data" / "scans" / "enginebay" / "photos",
    "mini6": ROOT / "data" / "monstree" / "mini6",
}

# Meshroom 2023.3's CameraInit passes --allowedCameraModels; the 3.4-dev binary does not know it
# (CameraInit stays Meshroom's own node in a paired install), so the default set is used here.


def sh(cmd: list[str], log: Path, env: dict | None = None) -> float:
    """Run one aliceVision binary through dev-run.cmd, stdout+stderr to log; returns wall seconds."""
    log.parent.mkdir(parents=True, exist_ok=True)
    full = ["cmd", "/c", str(DEVRUN)] + cmd
    e = dict(os.environ)
    if env:
        e.update(env)
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        r = subprocess.run(full, stdout=f, stderr=subprocess.STDOUT, env=e, cwd=str(ROOT))
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        sys.exit(f"{cmd[0]} failed with {r.returncode}; see {log}")
    return dt


# Sets whose features and matches already exist in a kept Meshroom cache (no prepare step): the
# 884-view False Door replay of docs/04, on the cache the 0.3.3 end-to-end run left behind.
KEPT = {
    "fd": (B / "e2e-falsedoor-fix" / "base" / "cache", "sift"),
}


def cache_dirs(s: str) -> dict[str, Path]:
    if s in KEPT:
        c = KEPT[s][0]
        one = lambda node: next(p for p in (c / node).iterdir() if p.is_dir())  # noqa: E731
        return {"ci": one("CameraInit"), "fe": one("FeatureExtraction"), "im": one("ImageMatching"), "fm": one("FeatureMatching")}
    c = BENCH / s / "cache"
    return {"ci": c / "CameraInit", "fe": c / "FeatureExtraction", "im": c / "ImageMatching", "fm": c / "FeatureMatching"}


def prepare(s: str, describer: str) -> None:
    photos = SETS[s]
    if not photos.is_dir():
        sys.exit(f"no photos at {photos}")
    d = cache_dirs(s)
    ci = d["ci"] / "cameraInit.sfm"
    if not ci.is_file():
        print(f"CameraInit on {photos} ({len(list(photos.iterdir()))} files)")
        sh(["aliceVision_cameraInit", "--imageFolder", str(photos), "--sensorDatabase", str(SHARE / "cameraSensors.db"),
            "--defaultFieldOfView", "45.0", "--groupCameraFallback", "folder",
            "--rawColorInterpretation", "LibRawWhiteBalancing", "--viewIdMethod", "metadata", "--verboseLevel", "info",
            "--allowSingleView", "1", "--output", str(ci)], d["ci"] / "log")
    if not any(d["fe"].glob("*.feat")):
        print(f"FeatureExtraction ({describer}, CPU)")
        dt = sh(["aliceVision_featureExtraction", "--input", str(ci), "--describerTypes", describer, "--describerPreset", "normal",
                 "--describerQuality", "normal", "--contrastFiltering", "GridSort", "--gridFiltering", "True",
                 "--workingColorSpace", "sRGB", "--forceCpuExtraction", "True", "--maxThreads", "0", "--verboseLevel", "info",
                 "--output", str(d["fe"])], d["fe"] / "log")
        print(f"  {dt:.0f} s")
    pairs = d["im"] / "imageMatches.txt"
    if not pairs.is_file():
        print("ImageMatching")
        sh(["aliceVision_imageMatching", "--input", str(ci), "--featuresFolders", str(d["fe"]), "--method", "SequentialAndVocabularyTree",
            "--tree", str(SHARE / "vlfeat_K80L3.SIFT.tree"), "--weights", "", "--minNbImages", "200", "--maxDescriptors", "500",
            "--nbMatches", "40", "--nbNeighbors", "5", "--verboseLevel", "info", "--output", str(pairs)], d["im"] / "log")
    if not any(d["fm"].glob("*.matches.txt")):
        print("FeatureMatching")
        dt = sh(["aliceVision_featureMatching", "--input", str(ci), "--featuresFolders", str(d["fe"]), "--imagePairsList", str(pairs),
                 "--describerTypes", describer, "--photometricMatchingMethod", "ANN_L2", "--geometricEstimator", "acransac",
                 "--geometricFilterType", "fundamental_matrix", "--distanceRatio", "0.8", "--maxIteration", "2048",
                 "--geometricError", "0.0", "--knownPosesGeometricErrorMax", "5.0", "--minRequired2DMotion", "-1.0",
                 "--maxMatches", "0", "--savePutativeMatches", "False", "--crossMatching", "False", "--guidedMatching", "False",
                 "--matchFromKnownCameraPoses", "False", "--exportDebugFiles", "False", "--verboseLevel", "info",
                 "--output", str(d["fm"])], d["fm"] / "log")
        print(f"  {dt:.0f} s")
    (BENCH / s / "describer.txt").write_text(describer + "\n", encoding="utf-8")
    print(f"ready: {BENCH / s / 'cache'}")


PROFILE = re.compile(r"BA profile: total ([0-9.e+-]+) s = residuals ([0-9.e+-]+) \+ jacobians ([0-9.e+-]+) \+ linear solver ([0-9.e+-]+)")
ADJUST = re.compile(r"BA adjust: build ([0-9.e+-]+) s, log evaluations ([0-9.e+-]+) s, solve ([0-9.e+-]+) s \(Ceres preprocessor ([0-9.e+-]+), minimizer ([0-9.e+-]+), postprocessor ([0-9.e+-]+)\), update ([0-9.e+-]+) s")
DESTROY = re.compile(r"BA adjust: destroy ([0-9.e+-]+) s")


def parse_log(log: Path) -> dict:
    t = log.read_text(encoding="utf-8", errors="replace")
    out: dict = {}
    for key, rx in (("poses", r"- # poses: (\d+)"), ("landmarks", r"- # landmarks: (\d+)"), ("rmse", r"residual RMSE: ([0-9.]+)")):
        m = re.findall(rx, t)
        if m:
            out[key] = float(m[-1]) if key == "rmse" else int(m[-1])
    out["ba_starts"] = t.count("Bundle adjustment start")
    prof = PROFILE.findall(t)
    if prof:
        out["ba_solves"] = len(prof)
        out["ba_total_s"] = round(sum(float(p[0]) for p in prof), 2)
        out["ba_residuals_s"] = round(sum(float(p[1]) for p in prof), 2)
        out["ba_jacobians_s"] = round(sum(float(p[2]) for p in prof), 2)
        out["ba_linear_s"] = round(sum(float(p[3]) for p in prof), 2)
    adj = ADJUST.findall(t)
    if adj:
        out["adj_build_s"] = round(sum(float(a[0]) for a in adj), 2)
        out["adj_logeval_s"] = round(sum(float(a[1]) for a in adj), 2)
        out["adj_solve_s"] = round(sum(float(a[2]) for a in adj), 2)
        out["adj_preprocess_s"] = round(sum(float(a[3]) for a in adj), 2)
        out["adj_minimizer_s"] = round(sum(float(a[4]) for a in adj), 2)
        out["adj_postprocess_s"] = round(sum(float(a[5]) for a in adj), 2)
        out["adj_update_s"] = round(sum(float(a[6]) for a in adj), 2)
        out["adj_destroy_s"] = round(sum(float(d) for d in DESTROY.findall(t)), 2)
    m = re.findall(r"cheshire: BA jacobians: ([^\r\n]+)", t)
    if m:
        out["ba_jacobians_mode"] = m[0].strip()
    checks = re.findall(r"cheshire: BA check[^\r\n]*", t)
    if checks:
        out["ba_check_last"] = checks[-1].strip()
        out["ba_check_lines"] = len(checks)
    m = re.findall(r"views resected since the last bundle adjustment were still pending", t)
    out["pending_ba_fired"] = len(m)
    return out


def run(s: str, tag: str, envs: dict[str, str], repeat: int, extra: list[str], as_json: bool = False) -> None:
    d = cache_dirs(s)
    ci = d["ci"] / "cameraInit.sfm"
    if not ci.is_file() or not any(d["fm"].glob("*.matches.txt")):
        sys.exit(f"cache for {s} not prepared: python scripts/sfmbench.py prepare {s}")
    if s in KEPT:
        describer = KEPT[s][1]
    else:
        describer = (BENCH / s / "describer.txt").read_text(encoding="utf-8").strip() if (BENCH / s / "describer.txt").is_file() else "dspsift"
    for i in range(repeat):
        name = tag if repeat == 1 else f"{tag}-{i + 1}"
        out = BENCH / s / "runs" / name
        if out.exists():
            sys.exit(f"{out} exists; pick another tag")
        out.mkdir(parents=True)
        env = {"CHESHIRE_BA_PROFILE": "1"}
        env.update(envs)
        cmd = ["aliceVision_incrementalSfM", "--input", str(ci), "--featuresFolders", str(d["fe"]), "--matchesFolders", str(d["fm"]),
               "--describerTypes", describer, "--localizerEstimator", "acransac", "--observationConstraint", "Scale",
               "--localizerEstimatorMaxIterations", "4096", "--localizerEstimatorError", "0.0", "--lockScenePreviouslyReconstructed", "False",
               "--useLocalBA", "True", "--localBAGraphDistance", "1", "--nbFirstUnstableCameras", "30", "--maxImagesPerGroup", "30",
               "--bundleAdjustmentMaxOutliers", "50", "--maxNumberOfMatches", "0", "--minNumberOfMatches", "0", "--minInputTrackLength", "2",
               "--minNumberOfObservationsForTriangulation", "2", "--minAngleForTriangulation", "3.0", "--minAngleForLandmark", "2.0",
               "--maxReprojectionError", "4.0", "--minAngleInitialPair", "5.0", "--maxAngleInitialPair", "40.0",
               "--useOnlyMatchesFromInputFolder", "False", "--useRigConstraint", "True", "--rigMinNbCamerasForCalibration", "20",
               "--lockAllIntrinsics", "False", "--minNbCamerasToRefinePrincipalPoint", "3", "--filterTrackForks", "False",
               "--computeStructureColor", "True", "--useAutoTransform", "True", "--initialPairA", "", "--initialPairB", "",
               "--interFileExtension", ".abc", "--logIntermediateSteps", "False", "--verboseLevel", "info",
               "--output", str(out / ("sfm.sfm" if as_json else "sfm.abc")), "--outputViewsAndPoses", str(out / "cameras.sfm"),
               "--extraInfoFolder", str(out)] + extra
        print(f"[{name}] incrementalSfM on {s} with {env}" + (f" and {' '.join(extra)}" if extra else ""))
        dt = sh(cmd, out / "sfm.log", env)
        rec = {"set": s, "tag": name, "env": envs, "extra": extra, "wall_s": round(dt, 1), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        rec.update(parse_log(out / "sfm.log"))
        rec.update(digests(out))
        with open(BENCH / "bench.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print("  " + row(rec))


def digests(out: Path) -> dict:
    """SHA-256 of the outputs, so two runs can be compared for byte identity."""
    import hashlib
    d = {}
    for name in ("sfm.abc", "sfm.sfm", "cameras.sfm"):
        f = out / name
        if f.is_file():
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            d[name.replace(".", "_") + "_sha256"] = h.hexdigest()[:16]
    return d


def row(r: dict) -> str:
    return (f"{r['set']:>5} {r['tag']:<22} wall {r['wall_s']:>7.1f} s | BA {r.get('ba_total_s', 0):>7.1f} s = jac {r.get('ba_jacobians_s', 0):>6.1f}"
            f" + lin {r.get('ba_linear_s', 0):>6.1f} + res {r.get('ba_residuals_s', 0):>5.1f} ({r.get('ba_solves', 0)} solves)"
            f" | {r.get('poses', '?')} poses, {r.get('landmarks', '?')} landmarks, RMSE {r.get('rmse', '?')}"
            + (f" | {r['ba_check_last'].replace('cheshire: ', '')}" if r.get("ba_check_last") else "")
            + (f" | adjust: build {r['adj_build_s']} + logeval {r['adj_logeval_s']} + solve {r['adj_solve_s']} (pre {r['adj_preprocess_s']}, min {r['adj_minimizer_s']}, post {r['adj_postprocess_s']}) + update {r['adj_update_s']} + destroy {r['adj_destroy_s']} s" if r.get("adj_build_s") is not None else "")
            + (f" | abc {r['sfm_abc_sha256']} cams {r.get('cameras_sfm_sha256', '?')}" if r.get("sfm_abc_sha256") else "")
            + (f" | sfm {r['sfm_sfm_sha256']} cams {r.get('cameras_sfm_sha256', '?')}" if r.get("sfm_sfm_sha256") else ""))


def main(argv: list[str]) -> None:
    if len(argv) < 1 or argv[0] not in ("prepare", "run", "show"):
        sys.exit(__doc__)
    if argv[0] == "show":
        for line in (BENCH / "bench.jsonl").read_text(encoding="utf-8").splitlines():
            print(row(json.loads(line)))
        return
    s = argv[1] if len(argv) > 1 else sys.exit("which set? 41 | eb | mini6")
    if s not in SETS and s not in KEPT:
        sys.exit(f"unknown set {s}")
    if argv[0] == "prepare":
        describer = argv[argv.index("--describer") + 1] if "--describer" in argv else "dspsift"
        prepare(s, describer)
        return
    tag = argv[argv.index("--tag") + 1] if "--tag" in argv else sys.exit("--tag <name> required")
    repeat = int(argv[argv.index("--repeat") + 1]) if "--repeat" in argv else 1
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[2:argv.index("--")] if "--" in argv else argv[2:]
    envs = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in own if "=" in a and not a.startswith("--")}
    run(s, tag, envs, repeat, extra, as_json="--json" in own)


if __name__ == "__main__":
    main(sys.argv[1:])
