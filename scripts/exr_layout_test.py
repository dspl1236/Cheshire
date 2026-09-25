#!/usr/bin/env python3
"""The deciding test for CheshireEXR (docs/20, dspl1236/Cheshire#2): ZIP level 1 through the host path
against ZIPS through the GPU path, on the same images, end to end.

    exr_layout_test.py <config.json> [--reps N] [--check] [--skip-pds] [--skip-texturing]

CheshireEXR decodes ZIPS (one chunk per scanline) faster than a CPU thread, and ZIP (16 scanlines per
chunk, what PrepareDenseScene writes since steps 5y and 6f) far slower. Whether PrepareDenseScene
should write ZIPS for it depends on every node that touches those files, so this runs all of them:

  1. PrepareDenseScene twice, CHESHIRE_PDS_EXR_COMPRESSION=zip:1 and zips:1: the write cost of each
     layout. Same inputs, so the two outputs must hold identical pixels, and this checks that they do
     (every channel, bit for bit) before anything is timed against them.
  2. DepthMap on one chunk, four ways:
       zip-host   ZIP,  CHESHIRE_DEPTHMAP_GPU_EXR=0   today's default
       zips-host  ZIPS, CHESHIRE_DEPTHMAP_GPU_EXR=0   what ZIPS costs the CPU reader
       zips-gpu   ZIPS, CHESHIRE_DEPTHMAP_GPU_EXR=1   the candidate
       zip-gpu    ZIP,  CHESHIRE_DEPTHMAP_GPU_EXR=1   the chunk threshold: must decode nothing on the device
     The depth and similarity maps of all four must be identical. --check adds a zips-gpu run with
     CHESHIRE_DEPTHMAP_GPU_EXR_CHECK=1 (every image compared with the host path; not timed against
     the others, since it does both).
  3. Texturing on each layout (it reads through the host path on both): what ZIPS costs its reads.

Every run gets CHESHIRE_LOAD_PROFILE=1 and CHESHIRE_EXR_PROFILE=1, and its log is kept. Times are wall
clock and process CPU (user + system of the node's process), median of --reps runs. The result is
<out>/exr_layout_test.md and .json.

The config names the install and the three command lines, with {images} and {out} where the image
folder and the output folder go; take the arguments from the nodes of the Meshroom cache being tested
(the node's log starts with its command line). Paths may use forward slashes on Windows.

    {
      "out": "D:/cheshire/data/out/layout-test",
      "path_prepend": ["D:/cheshire/build/av-gfx1201-install/bin", "D:/cheshire/tools/vcpkg-deps/.../bin"],
      "env": {"ALICEVISION_ROOT": "D:/cheshire/build/av-gfx1201-install"},
      "pds":       ["aliceVision_prepareDenseScene.exe", "--input", "D:/.../sfm.abc", "--output", "{out}", "..."],
      "depthmap":  ["aliceVision_depthMapEstimation.exe", "--input", "D:/.../sfm.abc", "--imagesFolder", "{images}",
                    "--output", "{out}", "--rangeStart", "0", "--rangeSize", "48", "..."],
      "texturing": ["aliceVision_texturing.exe", "--imagesFolder", "{images}", "--output", "{out}", "..."]
    }

With --skip-pds, "images_zip" and "images_zips" name existing folders instead (the pixel check still
runs); "texturing" is optional.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ------------------------------------------------------------------------------------------ running


def _cpu_seconds(proc: subprocess.Popen) -> float | None:
    """User + system CPU of a finished child process."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]

        c, e, k, u = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        ok = ctypes.windll.kernel32.GetProcessTimes(wintypes.HANDLE(int(proc._handle)), ctypes.byref(c), ctypes.byref(e),
                                                    ctypes.byref(k), ctypes.byref(u))
        if not ok:
            return None
        return (((k.hi << 32) | k.lo) + ((u.hi << 32) | u.lo)) / 1e7
    return None  # POSIX: filled in by run() from os.wait4


def run(cmd: list[str], env: dict[str, str], log: Path) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("  $ " + " ".join(cmd[:3]) + (" ..." if len(cmd) > 3 else "") + f"  > {log.name}", flush=True)
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        cpu = None
        if os.name == "nt":
            proc.wait()
            cpu = _cpu_seconds(proc)
        else:
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
            cpu = usage.ru_utime + usage.ru_stime
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.exit(f"FAILED (exit {proc.returncode}): see {log}")
    return {"wall": wall, "cpu": cpu, "log": str(log)}


def expand(template: list[str], **kw: str) -> list[str]:
    return [a.format(**kw) for a in template]


# ----------------------------------------------------------------------------------------- the logs

NUM = r"([0-9.eE+-]+)"
RE_BATCH = re.compile(r"cheshire: depth map batch \d+/\d+: .*decode " + NUM + r" s, load " + NUM + r" s")
RE_DEVICE = re.compile(r"cheshire: depth map batch \d+/\d+: (\d+) decoded on the device \(CheshireEXR\), (\d+) by the host path"
                       r"(?: \((\d+) with too few chunks\))?")
RE_EXR = re.compile(r"cheshire: EXR profile: .* open " + NUM + r" s, read " + NUM + r" s, thread cpu " + NUM + r" s")
RE_CHECK_BAD = re.compile(r"CHESHIRE_DEPTHMAP_GPU_EXR_CHECK: .*(texels differ|did not read)")
RE_CHECK_OK = re.compile(r"CHESHIRE_DEPTHMAP_GPU_EXR_CHECK: .*identical to the host path")


def parse_log(path: str) -> dict:
    s = {"decode": 0.0, "load": 0.0, "batches": 0, "on_device": 0, "host": 0, "few_chunks": 0,
         "exr_files": 0, "exr_read": 0.0, "exr_cpu": 0.0, "check_ok": 0, "check_bad": 0}
    for line in open(path, encoding="utf-8", errors="replace"):
        if m := RE_BATCH.search(line):
            s["decode"] += float(m.group(1))
            s["load"] += float(m.group(2))
            s["batches"] += 1
        elif m := RE_DEVICE.search(line):
            s["on_device"] += int(m.group(1))
            s["host"] += int(m.group(2))
            s["few_chunks"] += int(m.group(3) or 0)
        elif m := RE_EXR.search(line):
            s["exr_files"] += 1
            s["exr_read"] += float(m.group(1)) + float(m.group(2))
            s["exr_cpu"] += float(m.group(3))
        elif RE_CHECK_BAD.search(line):
            s["check_bad"] += 1
        elif RE_CHECK_OK.search(line):
            s["check_ok"] += 1
    return s


# --------------------------------------------------------------------------------------- comparing


def exr_channels(path: Path) -> dict:
    import numpy as np
    import OpenEXR

    with OpenEXR.File(str(path)) as f:
        return {name: np.ascontiguousarray(ch.pixels) for name, ch in f.channels().items()}


def same_pixels(a: Path, b: Path) -> bool:
    """Every channel bit for bit (NaN payloads included), whatever the compression."""
    ca, cb = exr_channels(a), exr_channels(b)
    if ca.keys() != cb.keys():
        return False
    return all(ca[k].dtype == cb[k].dtype and ca[k].shape == cb[k].shape and ca[k].tobytes() == cb[k].tobytes() for k in ca)


def compression_of(path: Path) -> str:
    import OpenEXR

    with OpenEXR.File(str(path)) as f:
        return str(f.header().get("compression", "?")).replace("Compression.", "").replace("_COMPRESSION", "")


def check_layouts(zip_dir: Path, zips_dir: Path) -> dict:
    names = sorted(p.name for p in zip_dir.glob("*.exr"))
    missing = [n for n in names if not (zips_dir / n).exists()]
    if not names or missing:
        sys.exit(f"the two image folders do not hold the same files ({len(names)} in {zip_dir}, missing in {zips_dir}: {missing[:5]})")
    differ = [n for n in names if not same_pixels(zip_dir / n, zips_dir / n)]
    res = {
        "files": len(names),
        "pixels_differ": differ,
        "bytes_zip": sum((zip_dir / n).stat().st_size for n in names),
        "bytes_zips": sum((zips_dir / n).stat().st_size for n in names),
        "compression_zip": compression_of(zip_dir / names[0]),
        "compression_zips": compression_of(zips_dir / names[0]),
    }
    print(f"  {len(names)} images; pixels identical in {len(names) - len(differ)}; "
          f"{res['compression_zip']} {res['bytes_zip'] / 2**20:.0f} MB, {res['compression_zips']} {res['bytes_zips'] / 2**20:.0f} MB")
    if differ:
        sys.exit(f"the ZIP and ZIPS images differ in pixels ({differ[:5]}); nothing below would be a fair comparison")
    return res


def compare_maps(ref: Path, other: Path) -> dict:
    names = sorted(p.name for p in ref.glob("*.exr"))
    res = {"maps": len(names), "bytes_identical": 0, "pixels_identical": 0, "differ": [], "missing": []}
    for n in names:
        if not (other / n).exists():
            res["missing"].append(n)
        elif hashlib.sha256((ref / n).read_bytes()).digest() == hashlib.sha256((other / n).read_bytes()).digest():
            res["bytes_identical"] += 1
        elif same_pixels(ref / n, other / n):
            res["pixels_identical"] += 1
        else:
            res["differ"].append(n)
    return res


# ------------------------------------------------------------------------------------------- main


def median_run(reps: int, cmd: list[str], env: dict[str, str], logdir: Path, tag: str, out: Path) -> dict:
    results = []
    for r in range(reps):
        if out.exists():
            for p in out.glob("*"):
                if p.is_file():
                    p.unlink()
        res = run(cmd, env, logdir / f"{tag}-{r}.log")
        res.update(parse_log(res["log"]))
        results.append(res)
    best = sorted(results, key=lambda x: x["wall"])[len(results) // 2]
    best["walls"] = [x["wall"] for x in results]
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--reps", type=int, default=1, help="runs per DepthMap and Texturing configuration (median)")
    ap.add_argument("--check", action="store_true", help="also run zips-gpu with CHESHIRE_DEPTHMAP_GPU_EXR_CHECK=1")
    ap.add_argument("--skip-pds", action="store_true", help="use images_zip / images_zips from the config")
    ap.add_argument("--skip-texturing", action="store_true")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    out = Path(cfg["out"])
    logs = out / "logs"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in cfg.get("env", {}).items()})
    if cfg.get("path_prepend"):
        env["PATH"] = os.pathsep.join([str(Path(p)) for p in cfg["path_prepend"]] + [env.get("PATH", "")])
    env["CHESHIRE_LOAD_PROFILE"] = "1"
    env["CHESHIRE_EXR_PROFILE"] = "1"
    report: dict = {"config": a.config, "reps": a.reps}

    # 1. the two layouts, and what each costs to write
    if a.skip_pds:
        zip_dir, zips_dir = Path(cfg["images_zip"]), Path(cfg["images_zips"])
    else:
        print("PrepareDenseScene, ZIP level 1 and ZIPS level 1")
        report["pds"] = {}
        for tag, method in (("zip", "zip:1"), ("zips", "zips:1")):
            d = out / f"pds-{tag}"
            d.mkdir(parents=True, exist_ok=True)
            report["pds"][tag] = run(expand(cfg["pds"], out=str(d), images=""), dict(env, CHESHIRE_PDS_EXR_COMPRESSION=method),
                                     logs / f"pds-{tag}.log")
        zip_dir, zips_dir = out / "pds-zip", out / "pds-zips"
    print("pixels: ZIP against ZIPS")
    report["layouts"] = check_layouts(zip_dir, zips_dir)

    # 2. DepthMap, four ways (and the check)
    runs = [("zip-host", zip_dir, "0"), ("zips-host", zips_dir, "0"), ("zips-gpu", zips_dir, "1"), ("zip-gpu", zip_dir, "1")]
    report["depthmap"] = {}
    for tag, images, gpu in runs:
        print(f"DepthMap {tag}")
        d = out / f"dm-{tag}"
        d.mkdir(parents=True, exist_ok=True)
        report["depthmap"][tag] = median_run(a.reps, expand(cfg["depthmap"], images=str(images), out=str(d)),
                                             dict(env, CHESHIRE_DEPTHMAP_GPU_EXR=gpu), logs, f"dm-{tag}", d)
    if a.check:
        print("DepthMap zips-gpu with CHESHIRE_DEPTHMAP_GPU_EXR_CHECK=1")
        d = out / "dm-zips-gpu-check"
        d.mkdir(parents=True, exist_ok=True)
        report["depthmap_check"] = median_run(1, expand(cfg["depthmap"], images=str(zips_dir), out=str(d)),
                                              dict(env, CHESHIRE_DEPTHMAP_GPU_EXR="1", CHESHIRE_DEPTHMAP_GPU_EXR_CHECK="1"), logs,
                                              "dm-zips-gpu-check", d)
    report["maps"] = {tag: compare_maps(out / "dm-zip-host", out / f"dm-{tag}") for tag, _, _ in runs[1:]}

    # 3. Texturing's reads of each layout
    if cfg.get("texturing") and not a.skip_texturing:
        report["texturing"] = {}
        for tag, images in (("zip", zip_dir), ("zips", zips_dir)):
            print(f"Texturing {tag}")
            d = out / f"tex-{tag}"
            d.mkdir(parents=True, exist_ok=True)
            report["texturing"][tag] = median_run(a.reps, expand(cfg["texturing"], images=str(images), out=str(d)), env, logs,
                                                  f"tex-{tag}", d)

    (out / "exr_layout_test.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    md = summary(report)
    (out / "exr_layout_test.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    bad = any(m["differ"] or m["missing"] for m in report["maps"].values())
    bad |= report["depthmap"]["zip-gpu"]["on_device"] != 0
    bad |= report.get("depthmap_check", {}).get("check_bad", 0) != 0
    return 1 if bad else 0


def fmt(x: float | None, unit: str = "s") -> str:
    return "-" if x is None else f"{x:.1f} {unit}"


def summary(r: dict) -> str:
    L = ["# ZIP through the host against ZIPS through the GPU", ""]
    lay = r["layouts"]
    L += [f"{lay['files']} images, pixels identical in both layouts. ZIP ({lay['compression_zip']}) "
          f"{lay['bytes_zip'] / 2**20:.0f} MB, ZIPS ({lay['compression_zips']}) {lay['bytes_zips'] / 2**20:.0f} MB "
          f"({(lay['bytes_zips'] / lay['bytes_zip'] - 1) * 100:+.1f} %).", ""]
    if "pds" in r:
        L += ["## PrepareDenseScene (the write)", "", "| layout | wall | CPU |", "|---|---|---|"]
        for tag in ("zip", "zips"):
            p = r["pds"][tag]
            L.append(f"| {tag} | {fmt(p['wall'])} | {fmt(p['cpu'])} |")
        L.append("")
    L += ["## DepthMap (one chunk)", "",
          "| run | wall | CPU | loads (sum of batches) | decode | on the device / host (too few chunks) | EXR reads on the host: files, read, CPU |",
          "|---|---|---|---|---|---|---|"]
    for tag, d in r["depthmap"].items():
        L.append(f"| {tag} | {fmt(d['wall'])} | {fmt(d['cpu'])} | {fmt(d['load'])} | {fmt(d['decode'])} | "
                 f"{d['on_device']} / {d['host']} ({d['few_chunks']}) | {d['exr_files']}, {fmt(d['exr_read'])}, {fmt(d['exr_cpu'])} |")
    if "depthmap_check" in r:
        c = r["depthmap_check"]
        L.append(f"\nCHESHIRE_DEPTHMAP_GPU_EXR_CHECK: {c['check_ok']} images identical to the host path, {c['check_bad']} not.")
    L += ["", "Maps against zip-host:", ""]
    for tag, m in r["maps"].items():
        L.append(f"- {tag}: {m['maps']} maps, {m['bytes_identical']} byte-identical, {m['pixels_identical']} pixel-identical only, "
                 f"{len(m['differ'])} differ, {len(m['missing'])} missing" + (f" ({m['differ'][:3]})" if m["differ"] else ""))
    if "texturing" in r:
        L += ["", "## Texturing (reads through the host path)", "", "| layout | wall | CPU | EXR reads: files, read, CPU |", "|---|---|---|---|"]
        for tag, t in r["texturing"].items():
            L.append(f"| {tag} | {fmt(t['wall'])} | {fmt(t['cpu'])} | {t['exr_files']}, {fmt(t['exr_read'])}, {fmt(t['exr_cpu'])} |")
    zg = r["depthmap"]["zip-gpu"]
    L += ["", f"Chunk threshold: zip-gpu decoded {zg['on_device']} images on the device (must be 0)."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
