#!/usr/bin/env python3
"""Check the two Fast Ransac template generators against each other and against their contract.

    check_fast_ransac_templates.py <Meshroom 2025.1+ dir> [--distro Ubuntu-24.04] [--keep]

Windows only: it runs scripts/windows/meshroom-templates.ps1 under Windows PowerShell and the generator inside
scripts/linux/meshroom-pair.sh (the python between <<'PY' and PY) under WSL's python3, each on its own scratch copy
of the Meshroom's stock templates (photogrammetry, photogrammetryDraft, photogrammetryExperimental where present);
the Meshroom itself is only read.

1. Stock: every template written is the stock file plus exactly two lines - "maxIteration": 2048 in the one
   FeatureMatching node and "localizerEstimatorMaxIterations": 4096 in the one StructureFromMotion or SfMExpanding
   node, as the first input, in the stock line endings - Windows and Linux write the same bytes, a second install
   writes them again, and remove deletes them all and nothing else.
2. Crafted inputs, from the reviews of 2026-10-01: user files under our names (other values, other encodings, a
   directory), malformed or unexpected stock files (truncated, empty, not an object, nodes or inputs of other types,
   keys in another case, another layout, a BOM, deep nesting). Each must end the same way on both platforms, with
   the outcome expected for it, and never stop the run.
Exit 0 when everything holds."""
import hashlib, json, re, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PS1 = HERE / "windows" / "meshroom-templates.ps1"
SH = HERE / "linux" / "meshroom-pair.sh"
STOCK = ["photogrammetry.mg", "photogrammetryDraft.mg", "photogrammetryExperimental.mg"]
OURS = {"photogrammetry.mg": "photogrammetryFastRansac.mg", "photogrammetryDraft.mg": "photogrammetryDraftFastRansac.mg",
        "photogrammetryExperimental.mg": "photogrammetryExperimentalFastRansac.mg"}
LINES = ('"maxIteration": 2048,', '"localizerEstimatorMaxIterations": 4096,')


def wsl_path(p: Path) -> str:
    s = str(p).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:]


def run(side, mr: Path, mode, py: Path, distro):
    d = mr / "aliceVision" / "share" / "meshroom"
    if side == "windows":
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PS1), str(mr), mode],
                           capture_output=True, text=True)
    else:
        r = subprocess.run(["wsl", "-d", distro, "--", "python3", wsl_path(py), wsl_path(d), mode], capture_output=True, text=True)
    out = [l.strip().replace(str(d), "<d>").replace(wsl_path(d), "<d>") for l in (r.stdout + r.stderr).splitlines() if l.strip()]
    return r.returncode, out


def state(d: Path):
    return {f.name: (hashlib.sha256(f.read_bytes()).hexdigest()[:16] if f.is_file() else "<dir>")
            for f in sorted(d.glob("*FastRansac.mg"))}


def two_lines_only(stock: bytes, ours: bytes) -> str:
    """'' when ours is stock plus exactly the two expected lines, else what is wrong"""
    eol = b"\r\n" if b"\r\n" in stock else b"\n"
    rest = ours.split(eol)
    for want in LINES:
        at = [i for i, l in enumerate(rest) if l.strip() == want.encode()]
        if len(at) != 1:
            return f"{want} appears {len(at)} times"
        del rest[at[0]]
    return "" if rest == stock.split(eol) else "the other lines are not the stock file's"


def fresh(root: Path, name: str, src: Path) -> Path:
    mr = root / name
    d = mr / "aliceVision" / "share" / "meshroom"
    d.mkdir(parents=True)
    for n in STOCK:
        if (src / n).exists():
            shutil.copy2(src / n, d / n)
    return mr


# ---- crafted inputs: (setup on the scratch share dir, modes, the outcome expected for photogrammetry.mg) ----
def put(data):
    return lambda d: (d / "photogrammetry.mg").write_bytes(data.encode() if isinstance(data, str) else data)


def edit_json(fn):
    def setup(d):
        g = json.loads((d / "photogrammetry.mg").read_bytes())
        (d / "photogrammetry.mg").write_bytes(json.dumps(fn(g) or g, indent=4).replace("\n", "\r\n").encode())
    return setup


def node(g, t):
    return [n for n in g["graph"].values() if n["nodeType"] == t][0]


def sfm_type(g):
    return "SfMExpanding" if any(n["nodeType"] == "SfMExpanding" for n in g["graph"].values()) else "StructureFromMotion"


def set_fm(key, value):
    return edit_json(lambda g: node(g, "FeatureMatching").__setitem__(key, value))


def fm_inputs_empty_multiline(d):
    g = json.loads((d / "photogrammetry.mg").read_bytes())
    node(g, sfm_type(g))["inputs"] = {}
    s = json.dumps(g, indent=4).replace('"inputs": {}', '"inputs": {\n' + " " * 12 + "}")
    (d / "photogrammetry.mg").write_bytes(s.replace("\n", "\r\n").encode())


def user_file(name, data):
    return lambda d: (d / name).write_bytes(data)


def our_values_text(stock_dir: Path) -> bytes:
    return (stock_dir / "photogrammetry.mg").read_bytes().replace(b'"inputs": {', b'"inputs": {"maxIteration": 20480, ', 1)


CASES = {
    "user file with 20480/40960": (user_file("photogrammetryFastRansac.mg", b'{"maxIteration": 20480, "localizerEstimatorMaxIterations": 40960}'),
                                    ["install", "remove"], "left alone"),
    "user file in Latin-1": (user_file("photogrammetryFastRansac.mg", '{"label": "Photogrammétrie"}'.encode("latin-1")),
                             ["install", "remove"], "left alone"),
    "user file in UTF-16 with our values": (user_file("photogrammetryFastRansac.mg", b"\xff\xfe" + '{"maxIteration": 2048, "localizerEstimatorMaxIterations": 4096}'.encode("utf-16-le")),
                                            ["install", "remove"], "left alone"),
    "a directory at our name": (lambda d: (d / "photogrammetryFastRansac.mg").mkdir(), ["install", "remove"], "left alone"),
    "SfM inputs empty over two lines": (fm_inputs_empty_multiline, ["install"], "did not come out"),
    "nodeType without the space": (put(None), ["install"], "laid out"),  # filled in below from the stock text
    "inputs null": (set_fm("inputs", None), ["install"], "laid out"),
    "maxIteration already set, to null": (edit_json(lambda g: node(g, "FeatureMatching")["inputs"].__setitem__("maxIteration", None)), ["install"], "not the graph"),
    "inputs a number": (set_fm("inputs", 5), ["install"], "not the graph"),
    "inputs true": (set_fm("inputs", True), ["install"], "not the graph"),
    "inputs a list": (set_fm("inputs", ["maxIteration"]), ["install"], "not the graph"),
    "inputs a string": (set_fm("inputs", "maxIterationX"), ["install"], "not the graph"),
    "MaxIteration already set": (edit_json(lambda g: node(g, "FeatureMatching")["inputs"].__setitem__("MaxIteration", 7)), ["install"], "not the graph"),
    "top-level Graph": (edit_json(lambda g: {"Graph": g["graph"], **{k: v for k, v in g.items() if k != "graph"}}), ["install"], "not the graph"),
    "an extra node with NodeType": (edit_json(lambda g: g["graph"].__setitem__("Extra_1", {"NodeType": "FeatureMatching", "inputs": {}})), ["install"], "installed"),
    "an extra node with nodeType a list": (edit_json(lambda g: g["graph"].__setitem__("Extra_1", {"nodeType": ["FeatureMatching"], "inputs": {}})), ["install"], "installed"),
    "a graph entry that is a number": (edit_json(lambda g: g["graph"].__setitem__("Extra_1", 5)), ["install"], "installed"),
    "two FeatureMatching nodes": (edit_json(lambda g: g["graph"].__setitem__("FeatureMatching_9", dict(node(g, "FeatureMatching")))), ["install"], "not the graph"),
    "both SfM node types": (edit_json(lambda g: g["graph"].__setitem__("Extra_1", dict(node(g, sfm_type(g)), nodeType="SfMExpanding" if sfm_type(g) == "StructureFromMotion" else "StructureFromMotion"))), ["install"], "not the graph"),
    "null": (put("null"), ["install"], "not the graph"),
    "{}": (put("{}"), ["install"], "not the graph"),
    "[]": (put("[]"), ["install"], "not the graph"),
    "graph null": (put('{"graph": null}'), ["install"], "not the graph"),
    "graph a list": (put('{"graph": []}'), ["install"], "not the graph"),
    "empty file": (put(b""), ["install"], "does not parse"),
    "blank file": (put(b"  \r\n \t"), ["install"], "does not parse"),
    "truncated": (lambda d: (d / "photogrammetry.mg").write_bytes((d / "photogrammetry.mg").read_bytes()[:-20]), ["install"], "does not parse"),
    "deep nesting": (put(b"[" * 100000 + b"]" * 100000), ["install"], "does not parse"),
    "a BOM": (lambda d: (d / "photogrammetry.mg").write_bytes(b"\xef\xbb\xbf" + (d / "photogrammetry.mg").read_bytes()), ["install"], "installed"),
}
OUTCOME = {"installed": "installed the pipeline template", "left alone": "exists and is not Cheshire's: left alone",
           "did not come out": "the edit did not come out", "laid out": "is not laid out as this expects",
           "not the graph": "is not the graph this expects", "does not parse": "does not parse as JSON"}


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    distro = argv[argv.index("--distro") + 1] if "--distro" in argv else "Ubuntu-24.04"
    if "--distro" in argv:
        args.remove(distro)
    if len(args) != 1:
        print(__doc__)
        return 2
    src = Path(args[0]) / "aliceVision" / "share" / "meshroom"
    if not (src / "photogrammetry.mg").exists():
        sys.exit(f"{src}\\photogrammetry.mg not found: not a Meshroom 2025.1 or later")
    root = Path(tempfile.mkdtemp(prefix="fastransac-check-"))
    m = re.search(r"<<'PY'\n(.*?)\nPY\n", SH.read_bytes().decode("utf-8"), re.S)
    py = root / "fast_ransac.py"
    py.write_bytes((m.group(1) + "\n").encode("utf-8"))
    bad = []

    # 1. stock
    print("== stock templates")
    states = {}
    for side in ("windows", "linux"):
        mr = fresh(root / "stock", side, src)
        d = mr / "aliceVision" / "share" / "meshroom"
        rc, out = run(side, mr, "install", py, distro)
        first = state(d)
        rc2, _ = run(side, mr, "install", py, distro)
        again = state(d)
        for stock_name, ours in OURS.items():
            if not (d / stock_name).exists():
                continue
            if not (d / ours).is_file():
                bad.append(f"{side}: {ours} not written: {out}")
                continue
            why = two_lines_only((d / stock_name).read_bytes(), (d / ours).read_bytes())
            if why:
                bad.append(f"{side}: {ours} is not the stock file plus the two lines: {why}")
        if rc or rc2 or first != again:
            bad.append(f"{side}: install rc {rc}/{rc2}, a second install changed the files: {first != again}")
        rc3, _ = run(side, mr, "remove", py, distro)
        if rc3 or state(d) or sorted(f.name for f in d.iterdir()) != sorted(n for n in STOCK if (src / n).exists()):
            bad.append(f"{side}: remove rc {rc3} left {sorted(f.name for f in d.iterdir())}")
        states[side] = first
        print(f"   {side}: " + ", ".join(f"{k} {v}" for k, v in first.items()))
    if states["windows"] != states["linux"]:
        bad.append(f"stock: Windows and Linux wrote different files: {states}")

    # 2. crafted inputs
    stock_text = (src / "photogrammetry.mg").read_bytes()
    print("== crafted inputs")
    for k, (case, (setup, modes, expect)) in enumerate(CASES.items()):
        if case == "nodeType without the space":
            setup = put(stock_text.replace(b'"nodeType": "FeatureMatching",', b'"nodeType":"FeatureMatching",'))
        res = {}
        for side in ("windows", "linux"):
            mr = fresh(root / "cases" / f"{k:02d}", side, src)
            d = mr / "aliceVision" / "share" / "meshroom"
            setup(d)
            steps = []
            for mode in modes:
                rc, out = run(side, mr, mode, py, distro)
                steps.append((mode, rc, out, state(d)))
            res[side] = steps
        same = res["windows"] == res["linux"]
        _, rc, out, _ = res["windows"][0]
        lines = [l for l in out if "photogrammetryFastRansac.mg" in l or "photogrammetry.mg" in l]
        got = lines[0] if lines else "(no line for photogrammetry.mg)"
        ok = same and rc == 0 and OUTCOME[expect] in got and all(s[1] == 0 for s in res["windows"])
        # the other stock templates are still written in every case that does not touch them
        others = [OURS[n] for n in STOCK[1:] if (src / n).exists()]
        written = res["windows"][0][3]
        ok = ok and all(o in written for o in others)
        print(f"   {'ok  ' if ok else 'FAIL'} {case:<40} {expect}" + ("" if same else "  (Windows and Linux differ)"))
        if not ok:
            bad.append(f"{case}: expected '{expect}', Windows {res['windows']}, Linux {res['linux']}")

    if "--keep" not in argv:
        shutil.rmtree(root, ignore_errors=True)
    else:
        print(f"scratch kept in {root}")
    print("\n" + ("ALL HOLD" if not bad else f"{len(bad)} PROBLEMS:\n  " + "\n  ".join(b[:600] for b in bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
