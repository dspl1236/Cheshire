#!/usr/bin/env python3
"""Cheshire vs stock: per set, each node's time in both arms (from Meshroom's own chunk status files: the chunks'
elapsed times summed, so a chunked node counts its compute, not its wall clock) and what came out: registered views,
mesh vertices and faces. usage: summarize.py [runs dir]"""
import json, re, sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/tests/bench/runs")
ORDER = ["CameraInit", "FeatureExtraction", "ImageMatching", "FeatureMatching", "StructureFromMotion", "PrepareDenseScene",
         "DepthMap", "DepthMapFilter", "Meshing", "MeshFiltering", "Texturing", "Publish"]
ARM_ORDER = ["stock", "dm", "cpu", "cheshire"]


def node_times(run):
    t = {}
    for st in (run / "MeshroomCache").glob("*/*/*status"):
        try:
            d = json.loads(st.read_text(errors="replace"))
        except ValueError:
            continue
        node = st.parent.parent.name
        e = d.get("elapsedTime") or 0
        t[node] = t.get(node, 0.0) + float(e)
    return t


def outcome(run):
    o = {}
    sfm = list((run / "MeshroomCache" / "StructureFromMotion").glob("*/cameras.sfm"))
    if sfm:
        try:
            o["views"] = len(json.loads(sfm[0].read_text()).get("poses", []))
        except ValueError:
            pass
    ci = list((run / "MeshroomCache" / "CameraInit").glob("*/cameraInit.sfm"))
    if ci:
        try:
            o["inputs"] = len(json.loads(ci[0].read_text()).get("views", []))
        except ValueError:
            pass
    for name, key in (("Meshing", "mesh"), ("MeshFiltering", "filtered")):
        objs = list((run / "MeshroomCache" / name).glob("*/mesh.obj"))
        if objs:
            v = f = 0
            with open(objs[0], "rb") as fh:
                for line in fh:
                    if line.startswith(b"v "):
                        v += 1
                    elif line.startswith(b"f "):
                        f += 1
            o[key] = (v, f)
    return o


runs = {}
for rj in sorted(R.glob("*/run.json")):
    d = json.loads(rj.read_text())
    runs[(d["set"], d["arm"] + ("-ovr" if d.get("overrides") else ""))] = (rj.parent, d)
for s in sorted({k[0] for k in runs}):
    arms = sorted((a for (ss, a) in runs if ss == s), key=lambda a: (ARM_ORDER.index(a.split("-")[0]) if a.split("-")[0] in ARM_ORDER else 9, a))
    print(f"== {s}")
    tt = {a: node_times(runs[(s, a)][0]) for a in arms}
    oc = {a: outcome(runs[(s, a)][0]) for a in arms}
    nodes = [n for n in ORDER if any(n in tt[a] for a in arms)] + sorted({n for a in arms for n in tt[a]} - set(ORDER))
    print(f"   {'node':22}" + "".join(f"{a:>16}" for a in arms))
    for n in nodes:
        print(f"   {n:22}" + "".join(f"{tt[a].get(n, float('nan')):>15.1f}s" for a in arms))
    print(f"   {'wall (rc)':22}" + "".join(f"{runs[(s, a)][1]['wall_s']:>11.0f}s ({runs[(s, a)][1]['rc']})" for a in arms))
    # what each arm reconstructed: a run that places fewer views does less dense work, so its time is not like for like
    print(f"   {'views placed':22}" + "".join(f"{(str(oc[a].get('views', '?')) + '/' + str(oc[a].get('inputs', '?'))):>16}" for a in arms))
    for key, label in (("mesh", "mesh faces"), ("filtered", "filtered faces")):
        print(f"   {label:22}" + "".join(f"{(format(oc[a][key][1], ',') if key in oc[a] else '-'):>16}" for a in arms))
    for a in arms:
        d = runs[(s, a)][1]
        f = d.get("filled_by_cheshire")  # a list, or one name in the first power-pc runs
        if f or d.get("notes"):
            print(f"   {a}: {d.get('notes') or 'filled in by Cheshire: ' + (f if isinstance(f, str) else ' '.join(f))}")
