r"""prune.sh for power-pc: every finished run in C:\cheshire-bench\runs-win (run.json written) loses its large
intermediates - depth and similarity maps, PrepareDenseScene's images, features, matches, Alembic clouds, textures - and
keeps what summarize.py reads (chunk statuses, logs, statistics, cameras.sfm, mesh.obj) plus project.mg and run.json.

  py -3 prune_win.py [run dir ...]     (no argument: every finished run)
"""
import shutil, sys
from pathlib import Path

RUNS = Path(r"C:\cheshire-bench\runs-win")
SUFFIXES = {".exr", ".png", ".feat", ".desc", ".abc", ".jpg", ".bin"}


def prune(d):
    d = Path(d)
    if not (d / "run.json").exists() or (d / ".pruned").exists():
        return 0
    freed = 0
    for f in list(d.rglob("*")):
        if not f.is_file():
            continue
        n = f.name.lower()
        if f.suffix.lower() in SUFFIXES or n == "texturedmesh.obj" or n.endswith(".matches.txt") or n.startswith("tetrahedralization"):
            freed += f.stat().st_size
            f.unlink()
    (d / ".pruned").write_text("")
    return freed


if __name__ == "__main__":
    dirs = sys.argv[1:] or [p for p in RUNS.iterdir() if p.is_dir()]
    for d in dirs:
        gb = prune(d) / 2**30
        if gb:
            print(f"pruned {Path(d).name}: {gb:.1f} GB")
    print(f"C: {shutil.disk_usage('C:/').free / 2**30:.0f} GB free")
