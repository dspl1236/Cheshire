#!/usr/bin/env python3
"""Run the memory-bridge experiment matrix on Windows and tabulate it.

Each case = an install dir (native-mipmap or emulated build) + environment knobs. For every
case the DepthMap stage runs through scripts/run-depthmap.cmd into its own output folder, the
run log is parsed (stage time, planner decision, simultaneous tiles, per-class VRAM/host peaks,
spill count) and the depth maps are compared with a baseline HIP output for bit-identity.

usage: bridge_matrix.py <dataset> <baseline out dir> <cases file> <results dir> [<Meshroom cache dir>]
  cases file: one case per line   name | install dir | KEY=VAL KEY=VAL ...
  ('#' comments allowed). Results: <results dir>/<dataset>.csv and .md, run logs per case.
  Windows: runs scripts/run-depthmap.cmd (data/ref/<dataset>). Linux: runs
  scripts/linux/run-depthmap.sh on <Meshroom cache dir> (default /data/scans/<dataset>/cache).
"""
import csv, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIN = os.name == "nt"
PY = ROOT / "tools/venv-rocm/Scripts/python.exe" if WIN else Path(sys.executable)


def parse_log(text: str) -> dict:
    d = {}
    m = re.search(r"Task done in \(s\): ([0-9.]+)", text); d["stage_s"] = float(m.group(1)) if m else None
    m = re.search(r"# simultaneous tiles computation: (\d+)", text); d["tiles"] = int(m.group(1)) if m else None
    m = re.search(r"Cheshire bridge planner (\d+): (.*)", text); d["planner"] = (m.group(2).strip()[:90]) if m else ""
    m = re.search(r"bridge summary: (\d+) spills", text); d["spills"] = int(m.group(1)) if m else None
    for cls in ("map", "volume", "image", "other"):
        m = re.search(rf"\]\s+{cls}\s+vram: (\d+) allocs, peak (\d+) MB \| host: (\d+) allocs, peak (\d+) MB", text)
        d[f"{cls}_vram_peak_mb"] = int(m.group(2)) if m else 0
        d[f"{cls}_host_peak_mb"] = int(m.group(4)) if m else 0
    d["pass"] = "RESULT: PASS" in text
    return d


def cuda_agreement(out: Path) -> str:
    """Median per-view share of pixels within 1 % of the CUDA reference, from the summary the
    comparison writes next to the outputs. The strict PASS/FAIL flag needs >= 98 % on every view,
    which sits right at the cross-hardware noise floor of some cards (RX 6750 XT: 97.5 %), so the
    number is what gets reported."""
    try:
        import json
        js = json.loads((out / "compare_summary.json").read_text())
        return f"{100 * js['median_of_view_within_1pct']:.1f}%"
    except Exception:
        return "n/a"


def identity(baseline: Path, out: Path) -> tuple[bool, float]:
    """Bit-identical when compare_depthmaps' exact mode passes (every decoded depth and sim pixel equal).
    Until 2026-09-26 this read the table's rounded columns, which print 0.0000 for anything under 5e-5,
    so a small real difference could be reported as identical. The float returned is the worst of mask
    disagreement, mean and p95 relative depth error, for the report."""
    p = subprocess.run([str(PY), str(ROOT / "scripts/compare_depthmaps.py"), str(baseline), str(out), "--mode", "exact"], capture_output=True, text=True)
    worst = 0.0; views = 0
    for line in p.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 11 and parts[0].isdigit():
            views += 1
            agree, meanrel, p95 = float(parts[3]), float(parts[4]), float(parts[6])
            worst = max(worst, 1.0 - agree, meanrel, p95)
    return (views > 0 and p.returncode == 0), worst


def main():
    dataset, baseline, cases_file, results = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    results.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in cases_file.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line: continue
        name, inst, envs = [x.strip() for x in line.split("|")]
        env = dict(os.environ)
        env.pop("CHESHIRE_BRIDGE_VRAM_MB", None)
        inst_path = Path(os.path.expanduser(inst)) if (inst.startswith("/") or inst.startswith("~")) else (ROOT / inst)
        env["CHESHIRE_INSTALL"] = str(inst_path.resolve())
        out = results / f"{dataset}-{name}"
        env["CHESHIRE_OUT"] = str(out.resolve())
        env["CHESHIRE_BRIDGE_LOG"] = "1"
        for kv in envs.split():
            k, v = kv.split("=", 1); env[k] = v
        t0 = time.time()
        if WIN:
            cmd = ["cmd", "/c", str(ROOT / "scripts/run-depthmap.cmd"), dataset]
        else:
            cache = Path(sys.argv[5]) if len(sys.argv) > 5 else Path("/data/scans") / dataset / "cache"
            ref = next(iter(sorted((cache / "DepthMap").glob("*/"))), "")
            runner = ROOT / "scripts/linux/run-depthmap.sh"
            if not runner.exists():   # node layout: the linux scripts sit flat in <apps>/cheshire/scripts/
                runner = ROOT / "scripts/run-depthmap.sh"
            cmd = ["bash", str(runner), str(inst_path), str(cache), str(out.resolve()), str(ref)]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, errors="replace")
        wall = time.time() - t0
        log = p.stdout + p.stderr
        (results / f"{dataset}-{name}.log").write_text(log, encoding="utf-8")
        d = parse_log(log)
        cuda = cuda_agreement(out)   # before identity(): that comparison rewrites compare_summary.json
        ident, worst = identity(baseline, out) if d["stage_s"] else (False, float("nan"))
        row = {"case": name, "install": inst, "env": envs, "stage_s": d["stage_s"], "wall_s": round(wall, 1), "tiles": d["tiles"],
               "spills": d["spills"], "identical": ident, "worst_dev": worst, "cuda_within_1pct": cuda, "strict_pass_vs_cuda": d["pass"], "planner": d["planner"],
               **{k: v for k, v in d.items() if k.endswith("_mb")}}
        rows.append(row)
        print(f"{name:28s} {str(d['stage_s']):>8s} s  tiles={d['tiles']} spills={d['spills']} "
              f"vol {d['volume_vram_peak_mb']}/{d['volume_host_peak_mb']} img {d['image_vram_peak_mb']}/{d['image_host_peak_mb']} "
              f"map {d['map_vram_peak_mb']}/{d['map_host_peak_mb']} MB  identical={ident} cuda<1%={row['cuda_within_1pct']}", flush=True)
    with open(results / f"{dataset}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    md = ["| case | build | knobs | DepthMap s | tiles | spills | volume VRAM/host MB | image VRAM/host MB | map VRAM/host MB | bit-identical | within 1 % of CUDA (median view) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['case']} | {'native mipmap' if (WIN and 'emu' not in r['install']) else 'emulated'} | `{r['env'] or 'defaults'}` | {r['stage_s']} | {r['tiles']} | {r['spills']} | "
                  f"{r['volume_vram_peak_mb']} / {r['volume_host_peak_mb']} | {r['image_vram_peak_mb']} / {r['image_host_peak_mb']} | {r['map_vram_peak_mb']} / {r['map_host_peak_mb']} | "
                  f"{'yes' if r['identical'] else 'no (%.3g)' % r['worst_dev']} | {r['cuda_within_1pct']} |")
    (results / f"{dataset}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
