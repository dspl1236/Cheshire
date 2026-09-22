"""Run the Meshing node on a fixed data set and pull out what the fusion work is measured by.

usage: python scripts/meshbench.py <set> <runner> <tag> [VAR=value ...]
  set     mini6 | 41 | 833 (the local job-mini6 cache; data/ref/monstree-full with the filtered maps
          in build/ref/monstree-41-dmf; the kept False Door cache build/e2e-falsedoor-fix)
  runner  dev (build/dev-run.cmd, the development install) | rel (the extracted 0.3.3 package)
  tag     output folder name under build/tmp-meshprof/

The command line is the Meshroom node's own (from its status file), with the input paths replaced for
the 41-view set and the outputs sent to the tag's folder; nothing in the source caches is written.
Prints the visibility accounting, the per-pass digests and checks, the tetrahedralisation checksums
and the node time, and appends one JSON line per run to build/tmp-meshprof/bench.jsonl.
"""
import glob, json, os, re, shlex, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "build"
REL = B / "tmp-meshprof/pkg/cheshire-alicevision-windows-x64/cheshire-run.cmd"
DEV = B / "dev-run.cmd"


def node_cmd(pattern):
    st = glob.glob(str(pattern))
    if not st:
        sys.exit(f"no Meshing status under {pattern}")
    return shlex.split(json.load(open(st[0]))["commandLine"], posix=True)[1:]


def args_for(dataset):
    if dataset == "mini6":
        return node_cmd(B / "meshroom/job-mini6/cache/Meshing/*/status")
    if dataset == "833":
        return node_cmd(B / "e2e-falsedoor-fix/base/cache/Meshing/*/status")
    if dataset == "41":
        a = node_cmd(B / "meshroom/job-mini6/cache/Meshing/*/status")
        sfm = glob.glob(str(ROOT / "data/ref/monstree-full/StructureFromMotion/*/sfm.abc"))[0]
        a[a.index("--input") + 1] = sfm
        a[a.index("--depthMapsFolder") + 1] = str(B / "ref/monstree-41-dmf")
        return a
    sys.exit(f"unknown set {dataset}")


def main():
    dataset, runner, tag = sys.argv[1:4]
    extra = dict(kv.split("=", 1) for kv in sys.argv[4:])
    out = B / "tmp-meshprof" / tag
    out.mkdir(parents=True, exist_ok=True)
    a = args_for(dataset)
    for flag, name in (("--output", "densePointCloud.abc"), ("--outputMesh", "mesh.obj")):
        a[a.index(flag) + 1] = str(out / name)
    env = {k: v for k, v in os.environ.items() if not k.startswith("CHESHIRE_")}
    env.update(extra)
    launcher = str(DEV if runner == "dev" else REL)
    cmd = [os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe"), "/c", launcher, "aliceVision_meshing"] + a
    t0 = time.time()
    with open(out / "meshing.log", "w", encoding="utf-8", errors="replace") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    text = (out / "meshing.log").read_text(encoding="utf-8", errors="replace")
    pick = lambda pat: [m.group(0) for m in re.finditer(pat, text)]
    vis = pick(r"cheshire: visibilities on the GPU: [^\n]*")
    res = {
        "set": dataset, "runner": runner, "tag": tag, "env": extra, "rc": rc, "wall_s": round(wall, 1),
        "task_s": (pick(r"Task done in \(s\): [0-9.]+") or [""])[0].split(": ")[-1],
        "tetra_in": pick(r"tetrahedralization input: [^\n]*"),
        "vis": vis,
        "checks": pick(r"cheshire: [^\n]*check[^\n]*"),
        "digests": pick(r"cheshire: visibility pass digest[^\n]*"),
    }
    with open(B / "tmp-meshprof/bench.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    print(f"== {tag}: {dataset} on {runner} {extra or ''} rc {rc}, wall {wall:.1f} s, node {res['task_s']} s")
    for line in res["tetra_in"] + res["digests"]:
        print("   ", line[:200])
    for line in vis:
        m = re.search(r"maps ([0-9.]+).*backproject ([0-9.]+).*device wait ([0-9.]+).*kernel ([0-9.]+).*votes ([0-9.]+).*total ([0-9.]+)", line)
        if m:
            print("    vis: maps %s  backproject %s  wait %s  kernel %s  votes %s  total %s ms" % m.groups())
    for line in res["checks"]:
        print("   ", line[:200])


if __name__ == "__main__":
    main()
