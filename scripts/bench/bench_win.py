r"""Cheshire vs stock Meshroom on power-pc (Ryzen 5600X, RX 9070), the Windows twin of bench.sh (docs/21). Arms: stock
(Meshroom's own binaries), cpu (the Cheshire package with every GPU port but DepthMap switched off), cheshire, and dm (stock
with only Cheshire's DepthMap, the first baseline). Stock runs until a node fails; StructureFromMotion (Meshroom #2344, with
Cheshire's other SfM changes back to upstream) or DepthMap (no CUDA device here) is then filled in by Cheshire's binary alone
and the run resumes. Meshroom 2025.1's default "photogrammetry" pipeline, Meshroom's default settings; runs in
C:\cheshire-bench\runs-win\<set>-<arm> with run.json as on house-pc, so summarize.py reads both. Each finished run is pruned
(prune_win.py) so the next has the room.

  py -3 bench_win.py <set> <photos dir> [stock|dm|cpu|cheshire|all] [paramOverrides...]
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
import prune_win

ROOT = Path(r"D:\MMI\cheshire")
MR = ROOT / "build" / "meshroom" / "Meshroom-2025.1.0"
BIN = MR / "aliceVision" / "bin"
PKG = ROOT / "build" / "bench" / "pkg045" / "cheshire-alicevision-windows-x64"
PAIR = ROOT / "scripts" / "windows" / "meshroom-pair.cmd"
RUNS = Path(r"C:\cheshire-bench\runs-win")  # C: has the room (the False Door and zoo caches are kept)
CPU_OFF = ["CHESHIRE_GPU_MATCHER", "CHESHIRE_GPU_FILTER", "CHESHIRE_GPU_VOTE", "CHESHIRE_GPU_TEX", "CHESHIRE_GPU_BLUR",
           "CHESHIRE_GPU_MAXFLOW", "CHESHIRE_GPU_VIS", "CHESHIRE_UNDISTORT_DEVICE"]


def pair(log):
    with open(log, "w") as f:
        return subprocess.run(["cmd", "/c", str(PAIR), str(MR), str(PKG)], stdout=f, stderr=subprocess.STDOUT).returncode


def unpair(log):
    with open(log, "a") as f:
        subprocess.run(["cmd", "/c", str(PAIR), str(MR), "--unpair"], stdout=f, stderr=subprocess.STDOUT)


UPSTREAM_SFM = {"CHESHIRE_BA_SOLVER": "ceres", "CHESHIRE_BA_JACOBIANS": "autodiff", "CHESHIRE_BA_FUSED_PROJECTION": "0",
                "CHESHIRE_BA_PERSIST": "0", "CHESHIRE_BA_LOG_COST": "1", "CHESHIRE_SFM_TASK_SEED": "0",
                "CHESHIRE_SFM_FILTER_THREADS": "1", "CHESHIRE_ACR_BOUND": "0", "CHESHIRE_ACR_RESIDUALS": "0",
                "CHESHIRE_ACR_SVD_LANES": "0"}  # Cheshire's SfM changes back to upstream, all but the pending-BA fix (5k)
FILLABLE = {"StructureFromMotion": "aliceVision_incrementalSfM", "DepthMap": "aliceVision_depthMapEstimation"}


def keep_only(keep, log):
    """After pairing: every node but the binaries in `keep` back to Meshroom's own, as meshroom-pair.cmd's :unpair does it."""
    restored = []
    for orig in sorted(BIN.glob("aliceVision_*.cuda.exe")):
        stem = orig.name[: -len(".cuda.exe")]
        if stem in keep:
            continue
        (BIN / f"{stem}.exe").unlink()
        (BIN / f"{stem}.cheshire.txt").unlink(missing_ok=True)
        orig.rename(BIN / f"{stem}.exe")
        restored.append(stem)
    with open(log, "a") as f:
        f.write(f"kept paired: {' '.join(sorted(keep))}; restored Meshroom's own {len(restored)} binaries\n")
    for k in keep:
        if not (BIN / f"{k}.cuda.exe").exists():
            sys.exit(f"{k} is not paired")


def keep_only_depthmap(log):
    keep_only({"aliceVision_depthMapEstimation"}, log)

def main():
    s, photos = sys.argv[1], sys.argv[2]
    arms = sys.argv[3] if len(sys.argv) > 3 else "all"
    overrides = sys.argv[4:]
    arms = ["stock", "cpu", "cheshire"] if arms == "all" else arms.split(",")
    RUNS.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        tag = f"{s}-{arm}" + ("-ovr" if overrides else "")
        out = RUNS / tag
        shutil.rmtree(out, ignore_errors=True)
        (out / "out").mkdir(parents=True)
        plog = RUNS / f"{tag}.pair.log"
        unpair(plog)
        if arm != "stock":
            if pair(plog) != 0:
                open(RUNS / "bench.done", "a").write(f"{tag}: pairing failed\n")
                continue
            if arm == "dm":
                keep_only_depthmap(plog)
        env = dict(os.environ)
        for v in CPU_OFF:
            env.pop(v, None)
        env.pop("CHESHIRE_BACKEND", None)
        if arm != "stock":
            env["CHESHIRE_BACKEND"] = "cheshire"
        if arm == "cpu":
            env.update({v: "0" for v in CPU_OFF})
        d = str(MR)  # Meshroom 2025.1 on Windows re-launches itself until ALICEVISION_LIBPATH names its folders
        env["ALICEVISION_LIBPATH"] = os.pathsep.join([os.path.join(d, "aliceVision", "bin"), os.path.join(d, "aliceVision", "lib"),
                                                      os.path.join(d, "lib"), d])
        env["PYTHONPATH"] = os.pathsep.join([os.path.join(d, "aliceVision", "lib", "python"),
                                             os.path.join(d, "aliceVision", "lib", "python3.11", "site-packages")])
        cmd = [str(MR / "meshroom_batch.exe"), "--input", photos, "--output", str(out / "out"), "--pipeline", "photogrammetry",
               "--save", str(out / "project.mg")] + (["--paramOverrides"] + overrides if overrides else [])
        t0 = time.time()
        with open(out / "batch.log", "w", encoding="utf-8", errors="replace") as f:
            rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        wall = time.time() - t0
        info = {"set": s, "arm": arm, "photos": photos, "overrides": " ".join(overrides), "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0))}
        filled, failures, passes = set(), [], 0
        while arm == "stock" and rc != 0 and passes < 3:
            # Pure stock stops where it cannot go on: StructureFromMotion on a large set (upstream's crash, Meshroom #2344)
            # or DepthMap without a CUDA device. Record where, keep the failed chunks' logs, clear what did not succeed,
            # and resume with only the failed nodes' binaries paired: SfM with Cheshire's other SfM changes switched back
            # to upstream (stock plus the fix), DepthMap as is. Every other node stays Meshroom's own.
            failed = sorted({st.parent.parent.name for st in (out / "MeshroomCache").glob("*/*/*status")
                             if json.loads(st.read_text(errors="replace")).get("status") == "ERROR"})
            need = {FILLABLE[n] for n in failed if n in FILLABLE}
            failures.append({"failed_at": failed, "after_s": round(wall, 1), "rc": rc})
            if not need:
                break
            for st in (out / "MeshroomCache").glob("*/*/*status"):
                state = json.loads(st.read_text(errors="replace")).get("status")
                if state != "SUCCESS":
                    log = st.with_name(st.name.replace("status", "log"))
                    if state == "ERROR" and log.exists():
                        shutil.copy2(log, out / f"stock-failure-{st.parent.parent.name}-{log.name}")
                    log.unlink(missing_ok=True)
                    st.unlink()
            filled |= need
            unpair(plog)
            if pair(plog) != 0:
                break
            keep_only(filled, plog)
            env["CHESHIRE_BACKEND"] = "cheshire"
            if "aliceVision_incrementalSfM" in filled:
                env.update(UPSTREAM_SFM)
            passes += 1
            t1 = time.time()
            with open(out / f"batch-resume{passes}.log", "w", encoding="utf-8", errors="replace") as f:
                rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            wall += time.time() - t1
        if failures:
            info.update({"stock_failures": failures, "filled_by_cheshire": sorted(filled),
                         # kept for the earlier runs' readers
                         "stock_failed_at": failures[0]["failed_at"], "stock_wall_s": failures[0]["after_s"], "stock_rc": failures[0]["rc"]})
        info.update({"rc": rc, "wall_s": round(wall, 1)})
        (out / "run.json").write_text(json.dumps(info))
        extra = (" (" + "; ".join(f"stock failed at {','.join(x['failed_at'])} after {x['after_s']:.0f} s, rc {x['rc']}" for x in info['stock_failures']) + f"; filled in by Cheshire: {' '.join(info['filled_by_cheshire'])})") if "stock_failures" in info else ""
        open(RUNS / "bench.done", "a").write(f"{tag} rc={rc} wall={wall:.0f} s{extra}\n")
        prune_win.prune(out)  # the next arm needs the room (the False Door's cache alone is ~70 GB)
    unpair(RUNS / "unpair-last.log")


if __name__ == "__main__":
    main()
