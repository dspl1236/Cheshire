#!/usr/bin/env python3
"""Run complete Meshroom pipelines on a Cheshire package and check every GPU port took part.

    verify_end_to_end.py <meshroom dir> <package dir> <photos> <out root> [config ...]

Runs on Windows and on Linux: both platforms pair the same way and take the same two arguments, so
only the pairing script's name, the directory it lives in here, and whether it needs a shell differ.

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
import json
import os, re, shutil, subprocess, sys, time
from pathlib import Path

# The lines each node's ports print when their GPU paths are live - every one must appear. Absent
# = that node ran something else, or a port fell back without saying so. Meshing carries four
# ports (votes, max-flow, visibility knn, sim blur) and only the votes are provable on a default
# run: max-flow prints only under its verbose flag, sim blur prints only when it is NOT used, and
# the visibility knn's "GPU knn index: N points" is behind CHESHIRE_GPU_VIS_LOG / _CHECK - a first
# version of this table required it and would have failed every default run while the port was
# in fact running. Three ports silent on success is a source defect of the kind the marker rule
# forbids, found 2026-09-20 while inventorying the knobs; it is a rebuild to fix (every payload
# plus the Linux bundle), so it is scheduled rather than slipped in. Under `verify`, where the
# check flags are on, the knn port proves itself through its self-check verdict instead.
GPU_MARKERS = {
    "FeatureExtraction": [r"Choosing device \d+:"],
    "FeatureMatching":   [r"GPU brute-force L2 2-NN on"],
    "DepthMap":          [r"Number of GPU devices"],
    "DepthMapFilter":    [r"depth map filter: group votes on"],
    "Meshing":           [r"meshing votes: ray marching on"],
    "Texturing":         [r"texturing: pyramid \+ rasterisation on"],
}
# ... and the lines printed when switched off. FeatureExtraction and DepthMap have no such switch,
# so they stay on the GPU in the fallback run and keep their markers. Sim blur announces its
# disabled state; max-flow and visibility do not, so with CHESHIRE_GPU_MAXFLOW=0 and
# CHESHIRE_GPU_VIS=0 set the fallback run exercises their CPU paths without being able to prove it.
CPU_MARKERS = {
    "FeatureExtraction": GPU_MARKERS["FeatureExtraction"],
    "DepthMap":          GPU_MARKERS["DepthMap"],
    "FeatureMatching":   [r"GPU brute-force disabled by CHESHIRE_GPU_MATCHER=0"],
    "DepthMapFilter":    [r"depth map filter: disabled by CHESHIRE_GPU_FILTER=0"],
    "Meshing":           [r"meshing votes: disabled by CHESHIRE_GPU_VOTE=0",
                          r"sim blur: disabled by CHESHIRE_GPU_BLUR=0"],
    "Texturing":         [r"texturing: disabled by CHESHIRE_GPU_TEX=0"],
}
SILENT_PORTS = "CHESHIRE_GPU_MAXFLOW and CHESHIRE_GPU_VIS are set but neither port prints on either path; not proven here"

# In-process self-checks: the port runs the CPU reference alongside itself and compares. These are
# the strongest correctness tests the project has, and until 2026-09-20 no gate switched them on.
# All fire inside Meshing. The tedge check's SUMS are not asserted - they differ by an ulp of
# summation order (docs/09) - only its cell counts, which must be equal.
SELF_CHECK_ENV = {
    "CHESHIRE_FILTER_CHECK": "1", "CHESHIRE_MAXFLOW_CHECK": "1", "CHESHIRE_GPU_VIS_CHECK": "1",
    "CHESHIRE_SEGMENT_CHECK": "1", "CHESHIRE_GPU_TEDGE_CHECK": "1", "CHESHIRE_GPU_VOTE_LOG": "1",
}
SELF_CHECK_VERDICTS = {
    "Meshing": [
        r"filterByPixSize check: identical to single-threaded upstream on all",
        # The float flow totals of the two algorithms are never equal and are documented as junk
        # (docs/12); the labelling is the verdict. The first version asserted "(identical)" on the
        # flows and failed a run whose labelling was 0 of 1,676,527 cells different.
        r"max-flow check: .*cells labelled differently: 0 of \d+",
        r"GPU knn check: identical to nanoflann on all",
        r"segmentFullOrFree check: identical to upstream on all",
        r"tedge check: cells with on != 0: cpu (\d+), gpu \1;",
        r"facet weight check: .*differing from the sequential computation: 0\b",
    ],
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

# The binary each Meshroom node runs, for the provenance check below.
BINARY = {
    "FeatureExtraction": "aliceVision_featureExtraction",
    "FeatureMatching":   "aliceVision_featureMatching",
    "DepthMap":          "aliceVision_depthMapEstimation",
    "DepthMapFilter":    "aliceVision_depthMapFiltering",
    "Meshing":           "aliceVision_meshing",
    "Texturing":         "aliceVision_texturing",
}

# Override paths are Meshroom ATTRIBUTE paths, not AliceVision command-line flags, and the two are
# not the same: some of a node's parameters sit in a group, so --sgmDepthListPerTile on the command
# line is sgm.sgmDepthListPerTile here and --tileBufferWidth is tiling.tileBufferWidth, while others
# are top level whatever they look like - Meshing's maxPoints is declared with advanced=True, which
# is a kwarg and not a group, so Meshing:advanced.maxPoints is a KeyError and Meshing:maxPoints is
# right. A wrong path is a KeyError before any node runs and shows up as a 0s FAIL with every port
# missing, so a config that fails instantly with ports=0/6 is a naming problem, not a GPU one; the
# failure prints the Overrides lines for that reason. The authority is the node definition in
# <Meshroom>/lib/meshroom/nodes/aliceVision/<Node>.pyc, not the AliceVision --help text.
CONFIGS = {
    # Stock settings, to establish that the paired pipeline works at all.
    "base": dict(overrides=SIFT, env={}),
    # Many small tiles: the tiled path is where the memory bridge and the per-tile depth lists live,
    # and a 42-tile run behaves differently from a 1-tile one.
    "tiles": dict(overrides=SIFT + [
        "DepthMap:tiling.tileBufferWidth=512", "DepthMap:tiling.tileBufferHeight=512",
        "DepthMap:tiling.autoAdjustSmallImage=False", "DepthMap:downscale=2"], env={}),
    # The other end: coarse, one depth list for the whole image, a smaller point budget.
    "coarse": dict(overrides=SIFT + [
        "DepthMap:downscale=4", "DepthMap:sgm.sgmDepthListPerTile=False",
        "Meshing:maxPoints=300000"], env={}),
    # A bigger atlas at full resolution, which moves the texturing port's allocations.
    "texbig": dict(overrides=SIFT + [
        "Texturing:textureSide=8192", "Texturing:downscale=1"], env={}),
    # Every GPU port switched off. Must still produce a mesh, and must say it went to the CPU.
    # Until 2026-09-20 this set four of the seven switches and was described as "every port off".
    "cpufallback": dict(overrides=SIFT, markers="cpu", note=SILENT_PORTS, env={
        "CHESHIRE_GPU_MATCHER": "0", "CHESHIRE_GPU_FILTER": "0", "CHESHIRE_GPU_VOTE": "0",
        "CHESHIRE_GPU_TEX": "0", "CHESHIRE_GPU_BLUR": "0", "CHESHIRE_GPU_MAXFLOW": "0",
        "CHESHIRE_GPU_VIS": "0"}),
    # Every in-process self-check on: each port must agree with its CPU reference, in its own words.
    "verify": dict(overrides=SIFT, env=dict(SELF_CHECK_ENV), checks=SELF_CHECK_VERDICTS),
    # The memory bridge under caps. Planner v2 absorbs 1.5 GB on the six-view set by planning
    # smaller tiles - budget 1200 MB, 0 full cameras + 2 tiles, no spill (docs/02) - so that run
    # asserts the re-plan, not a spill; the first version asserted "must spill", which was true of
    # planner v1 and encoded a fact three versions stale. 500 MB is below one tile and must spill.
    # Byte identity across caps is NOT asserted here: every config regenerates SfM from photos and
    # GPU SIFT is not bit-reproducible run to run, so no two runs share an input. That assertion
    # lives in the stage gate, on a fixed SfM, where it belongs.
    "bridgecap": dict(overrides=SIFT, env={"CHESHIRE_BRIDGE_VRAM_MB": "1500", "CHESHIRE_BRIDGE_LOG": "1"},
                      checks={"DepthMap": [r"bridge: vram cap 1500 MB", r"bridge planner 1: VRAM budget 1200"]}),
    "bridgespill": dict(overrides=SIFT, env={"CHESHIRE_BRIDGE_VRAM_MB": "500", "CHESHIRE_BRIDGE_LOG": "1"},
                        checks={"DepthMap": [r"bridge: vram cap 500 MB", r"bridge summary: [1-9]\d* spills"]}),
    # And the bridge switched off entirely: plain device allocation.
    "bridgeoff": dict(overrides=SIFT, env={"CHESHIRE_BRIDGE": "0"}),
    # Full-resolution depth maps, uncapped: the card's own VRAM is the constraint. Per-view working
    # set is 4x the default, so on a large high-resolution set this is what makes the bridge spill
    # naturally rather than under an artificial cap - and on CUDA, where camera mipmaps never pass
    # through the bridge, it is where a small card may run out of what the bridge cannot see.
    "ds1": dict(overrides=SIFT + ["DepthMap:downscale=1"], env={"CHESHIRE_BRIDGE_LOG": "1"}),
    # ds1 plus every opt-in speed path and the profile logs, so the run reports where time went.
    # Most acceleration is already default-on; what this adds is the 7-point QR nullspace (held
    # opt-in, docs/15) and a larger filter cache.
    "blast": dict(overrides=SIFT + ["DepthMap:downscale=1"], env={
        "CHESHIRE_BRIDGE_LOG": "1", "CHESHIRE_QR_NULLSPACE": "1", "CHESHIRE_FILTER_CACHE_MB": "8192",
        "CHESHIRE_GPU_VOTE_LOG": "1", "CHESHIRE_GPU_VIS_LOG": "1", "CHESHIRE_GPU_TEX_LOG": "1",
        "CHESHIRE_GPU_MATCHER_LOG": "1", "CHESHIRE_FILTER_LOG": "1"}),
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


def wrong_binary(cache: Path):
    """The first paired node whose log exists and names Meshroom's own binary, as a message; None
    while nothing has been decided yet or every decision so far was the Cheshire build. Read on a
    timer while meshroom_batch runs, so a mis-paired graph is stopped at its first node."""
    for node in BINARY:
        text = node_logs(cache, node)
        m = re.search(r"\[cheshire\] (\S+): Meshroom's own binary.*", text)
        if m:
            return f"{node} ran {m.group(0)[:120]}"
    return None


def run_one(name, cfg, meshroom: Path, photos: Path, outroot: Path) -> bool:
    out = outroot / name
    cache = out / "cache"
    shutil.rmtree(out, ignore_errors=True)
    (out / "out").mkdir(parents=True)

    env = dict(os.environ)
    # Force the paired launcher to the Cheshire package. Its default is 'auto', which hands the node
    # back to Meshroom's own CUDA binary whenever an NVIDIA card is present - correct when Cheshire
    # was AMD-only, and precisely wrong when the package under test is itself the CUDA one.
    env["CHESHIRE_BACKEND"] = "cheshire"
    env.update(cfg["env"])

    batch = meshroom / ("meshroom_batch.exe" if os.name == "nt" else "meshroom_batch")
    cmd = [str(batch),
           "--input", str(photos), "--output", str(out / "out"), "--cache", str(cache),
           "--pipeline", "photogrammetry"]
    if cfg["overrides"]:
        cmd += ["--paramOverrides"] + cfg["overrides"]

    log = out / "meshroom_batch.log"
    t0 = time.time()
    with log.open("w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        aborted = None
        while proc.poll() is None:
            time.sleep(5)
            aborted = wrong_binary(cache)
            if aborted:
                proc.kill()
        rc = proc.wait()
    secs = round(time.time() - t0)
    if aborted:
        # Do not let a wrong pairing run for hours to be reported at the end. The 1050 Ti's
        # full-resolution run of 2026-09-20 was two hours and 24 depth maps in before anyone read
        # the first log line: a launcher predating CHESHIRE_BACKEND had handed every node to
        # Meshroom, and the harness would have said so only once the whole graph had finished.
        print(f"  FAIL  {name:<14} ABORTED after {secs}s: {aborted}")
        print("        the launcher paired here does not honour CHESHIRE_BACKEND=cheshire, or the"
              " package was never paired - nothing this run does tests the package")
        return False, None

    mesh = list((out / "out").glob("*.obj"))
    # Any extension: Meshroom's Texturing writes texture_1001.exr by default, not .png, so globbing
    # for .png reported a perfectly good textured mesh as having no textures.
    tex = list((out / "out").glob("texture_*.*"))
    markers = CPU_MARKERS if cfg.get("markers") == "cpu" else GPU_MARKERS

    # Provenance first, then the port. A port marker alone is not proof the Cheshire binary ran:
    # Meshroom's own featureExtraction is a CUDA PopSIFT and prints the very same
    # "Choosing device 0" line, so on an NVIDIA box an UNPAIRED node scored a pass here
    # (found 2026-09-20 on Linux, where the CUDA bundle's versioned libpopsift.so.0.10.0 made the
    # pairing script decline GPU SIFT silently). Both launchers announce the binary they run.
    logs = {n: node_logs(cache, n) for n in markers}
    # A node with no log at all never ran - an earlier node failed and Meshroom stopped there. That
    # is a different fact from "ran Meshroom's own binary", and the first version of this report
    # called it NOT PAIRED: when a comgr-less bundle made DepthMap exit 1, the three nodes behind
    # it were reported as having run the wrong binary, which sent the diagnosis the wrong way.
    never_ran = [n for n in markers if not logs[n].strip()]
    paired = {n: bool(re.search(rf"\[cheshire\] {BINARY[n]}: Cheshire build", logs[n]))
              for n in markers}
    missing = [n for n, pats in markers.items()
               if paired[n] and not all(re.search(p, logs[n]) for p in pats)]
    unpaired = [n for n in markers if not paired[n] and n not in never_ran]
    # Config-specific assertions on top of the port lines: self-check verdicts, bridge announcements.
    unmet = [f"{n}: /{p}/" for n, pats in cfg.get("checks", {}).items()
             for p in pats if not re.search(p, node_logs(cache, n))]
    # Depth and sim maps as bytes, for the cross-config identity check in main().
    dm_dir = next(iter(sorted((cache / "DepthMap").glob("*/"))), None)
    dm_digest = None
    if dm_dir is not None:
        import hashlib
        exrs = sorted(dm_dir.glob("*_depthMap.exr")) + sorted(dm_dir.glob("*_simMap.exr"))
        if exrs:   # otherwise the digest of nothing (e3b0c442...) masquerades as a result
            h = hashlib.sha256()
            for f in exrs:
                h.update(f.name.encode()); h.update(f.read_bytes())
            dm_digest = h.hexdigest()[:16]

    ok = rc == 0 and mesh and tex and not missing and not unpaired and not never_ran and not unmet
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:<14} exit={rc:<4} mesh={len(mesh)} tex={len(tex)} "
          f"ports={len(markers) - len(missing) - len(unpaired) - len(never_ran)}/{len(markers)}"
          f"  dm={dm_digest or '-'}  {secs}s")
    if cfg.get("note"):
        print(f"        note: {cfg['note']}")
    if unmet:
        print(f"        assertions not met: {'; '.join(unmet)}")
    if never_ran:
        print(f"        did not run: {', '.join(never_ran)}"
              f" - an earlier node failed and the pipeline stopped before them")
    if unpaired:
        print(f"        NOT PAIRED: {', '.join(unpaired)}"
              f" - the node ran Meshroom's own binary, so nothing here tested the package")
    if missing:
        print(f"        paired but silent: {', '.join(missing)}"
              f" - the Cheshire binary ran and its GPU port did not announce itself")
    if rc != 0:
        # Name the node that failed and whose binary it was. The skull-turntable run of 2026-09-20
        # died in StructureFromMotion, a node the launcher never pairs: Meshroom's own
        # aliceVision_incrementalSfM hit a Ceres CHECK (zero-rotation pose after resection), and the
        # report should say so rather than leave the reader to work out from the log that no
        # Cheshire code ran there.
        for st in sorted(cache.glob("*/*/status")):
            try:
                j = json.loads(st.read_text(encoding="utf-8", errors="replace"))
            except ValueError:
                continue
            if j.get("status") == "ERROR":
                node = st.parent.parent.name
                ours = bool(re.search(r"\[cheshire\] \S+: Cheshire build", node_logs(cache, node)))
                print(f"        failed node: {node} - "
                      + ("a Cheshire-paired binary" if ours else
                         "Meshroom's own binary (not a paired node, no Cheshire code ran there)"))
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-4:]:
            print(f"        {line[:110]}")
    return bool(ok), dm_digest


def depthmap_identity(results: dict) -> bool:
    """Informational. The first version asserted byte-identical depth maps across configs with the
    same DepthMap parameters and failed every pair: each run regenerates features and SfM from the
    photographs, and GPU SIFT is not bit-reproducible run to run, so no two runs share an input.
    The bridge's byte-identity criterion is asserted by the stage gate on a fixed SfM instead. This
    reports the digests so a run-to-run coincidence is visible, and never fails the matrix."""
    rows = [(n, d) for n, (_, d) in results.items() if d]
    if len(rows) > 1:
        print("  depth-map digests (inputs differ per run - GPU SIFT is not bit-reproducible - so"
              " identity is asserted by the stage gate on a fixed SfM, not here):")
        for n, d in rows:
            print(f"    {n:<14} {d}")
    return True


def main(argv):
    if len(argv) < 5:
        print(__doc__)
        return 2
    meshroom, package, photos, outroot = (Path(a).resolve() for a in argv[1:5])
    names = argv[5:] or list(CONFIGS)
    bad = [n for n in names if n not in CONFIGS]
    if bad:
        sys.exit(f"unknown config(s): {', '.join(bad)} - have {', '.join(CONFIGS)}")

    # Both platforms pair the same way and take the same two arguments; only the script's name,
    # its directory here, and whether it needs a shell differ.
    win = os.name == "nt"
    pair_name = "meshroom-pair.cmd" if win else "meshroom-pair.sh"
    pair = package / pair_name
    if not pair.exists():
        pair = Path(__file__).resolve().parent / ("windows" if win else "linux") / pair_name
    if not pair.exists():
        sys.exit(f"{pair_name} not found in the package or in scripts/{'windows' if win else 'linux'}")
    pair_cmd = [str(pair)] if win else ["bash", str(pair)]

    print(f"=== meshroom {meshroom}")
    print(f"=== package  {package}")
    print(f"=== photos   {photos} ({len(list(photos.glob('*.[jJ][pP][gG]')))} jpg)")
    p = subprocess.run(pair_cmd + [str(meshroom), str(package)],
                       capture_output=True, text=True, shell=win)
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
        u = subprocess.run(pair_cmd + [str(meshroom), "--unpair"],
                           capture_output=True, text=True, shell=win)
        print("\n" + (u.stdout.strip() or u.stderr.strip()))

    passed = sum(1 for ok, _ in results.values() if ok)
    print(f"\n{passed} of {len(results)} pipelines ran end to end on this package")
    identical = depthmap_identity(results)
    return 0 if passed == len(results) and identical else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
