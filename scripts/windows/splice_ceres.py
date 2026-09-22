"""Splice the SuiteSparse-free Ceres into the prebuilt vcpkg tree: back up the old files, copy the new
headers, import library, DLL and CMake config, drop the old vcpkg config so find_package(Ceres)
resolves to the new lib/cmake/Ceres. Idempotent (backup taken once)."""
import pathlib, shutil, sys

root = pathlib.Path("D:/MMI/cheshire")
tree = root / "tools/vcpkg-deps/x64-windows-release/installed/x64-windows-release"
new = root / "build/ceres-nosuitesparse-install"
bak = root / "build/ceres-vcpkg-backup"

assert (new / "bin/ceres.dll").is_file(), "new ceres.dll missing"
assert (new / "lib/cmake/Ceres/CeresConfig.cmake").is_file(), "new CeresConfig.cmake missing"

if not bak.exists():
    bak.mkdir(parents=True)
    for rel in ("bin/ceres.dll", "bin/ceres.pdb", "lib/ceres.lib"):
        src = tree / rel
        if src.exists():
            (bak / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, bak / rel)
    for rel in ("include/ceres", "share/ceres"):
        src = tree / rel
        if src.exists():
            shutil.copytree(src, bak / rel)
    print("backup taken at", bak)

# headers
if (tree / "include/ceres").exists():
    shutil.rmtree(tree / "include/ceres")
shutil.copytree(new / "include/ceres", tree / "include/ceres")
# import lib + dll
shutil.copy2(new / "lib/ceres.lib", tree / "lib/ceres.lib")
shutil.copy2(new / "bin/ceres.dll", tree / "bin/ceres.dll")
pdb = new / "bin/ceres.pdb"
if pdb.exists():
    shutil.copy2(pdb, tree / "bin/ceres.pdb")
# cmake config: the new one in lib/cmake/Ceres; the old vcpkg one in share/ceres goes away
if (tree / "share/ceres").exists():
    shutil.rmtree(tree / "share/ceres")
if (tree / "lib/cmake/Ceres").exists():
    shutil.rmtree(tree / "lib/cmake/Ceres")
shutil.copytree(new / "lib/cmake/Ceres", tree / "lib/cmake/Ceres")
print("spliced:", (tree / "bin/ceres.dll").stat().st_size // 1024, "KB ceres.dll;",
      "config at", tree / "lib/cmake/Ceres")
cfg = (tree / "include/ceres/internal/config.h").read_text()
print("config.h:", [l.strip() for l in cfg.splitlines() if l.startswith("#define CERES_")])
