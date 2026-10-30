"""Refuse GPU code that needs hostcall: no kernel in a bundle may ask for a hostcall buffer.

    python3 scripts/check_hostcall.py [--llvm-bin DIR] <library or directory>...

A kernel that may call printf, malloc or another host service from the device gets a hidden_hostcall_buffer
argument, and the HIP runtime on Linux will not dispatch it on a machine without PCIe atomics: "Pcie atomics
not enabled, hostcall not supported", then "AQL dispatch failed!", and the launch does nothing. Cheshire
0.4.5's Linux bundle met exactly that on lgpu-pc (an FX-8120 board, PCIe 2.0, no atomics). PopSIFT v0.10.0's
orientation kernel carried a debug printf, so ori_par never ran there, the image had extrema but no
orientations, and AliceVision's describer segfaulted on the unfilled result (docs/14; apply_popsift_patch.py 3h
is the second half of the fix). Windows is unaffected: its runtime prints from the device without hostcall.

Every shared library with a HIP fat binary is unbundled, code object by code object, and its kernel metadata read
(llvm-readelf --notes). Any kernel with a hostcall buffer fails the check, except the few in ALLOWED: PopSIFT's
print-only debug kernels, which nothing in AliceVision launches. Exit status 1 on a failure, 0 otherwise.
The LLVM tools come from --llvm-bin, else $ROCM_PATH/lib/llvm/bin, else /opt/rocm/lib/llvm/bin.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

# Print-only debug kernels in PopSIFT (features.cu, sift_pyramid.cu, gauss_filter.cu): they exist to printf,
# and no AliceVision path launches them.
ALLOWED = {
    "_ZN7popsift13show_distanceEP15HIP_vector_typeIiLj3EEPNS_7FeatureEPNS_10DescriptorEPiiS4_S6_S7_i",
    "_ZN7popsift25print_gauss_filter_symbolEi",
    "_ZN7popsift21py_print_corner_floatEPfjjj",
    "_ZN7popsift32py_print_corner_float_transposedEPfjjj",
}


def hostcall_kernels(notes: str) -> tuple[int, list[str]]:
    """(number of kernels, the names of those with a hostcall buffer) from llvm-readelf --notes output. Each kernel
    is an item of the amdhsa.kernels list ("  - "); its args are a nested list and its own keys, .name among
    them, sit at four spaces."""
    entries = re.split(r"\n  - ", notes[notes.find("amdhsa.kernels"):])[1:]
    names = []
    for e in entries:
        m = re.search(r"^    \.name:\s+(\S+)", e, re.M)
        if m and re.search(r"\.value_kind:\s+hidden_hostcall_buffer", e):
            names.append(m.group(1))
    return len(entries), names


def scan(lib: str, tool) -> dict[str, tuple[int, list[str]]] | None:
    """{code object target: (kernels, hostcall kernels)} for a library, or None when it carries no HIP code."""
    with tempfile.TemporaryDirectory() as d:
        fb = os.path.join(d, "fatbin")
        for section in (".hip_fatbin", ".hip_fat"):   # ELF, PE (eight-character section names)
            r = subprocess.run([tool("llvm-objcopy"), "--dump-section", f"{section}={fb}", lib, os.path.join(d, "copy")],
                               capture_output=True)
            if r.returncode == 0 and os.path.exists(fb):
                break
        else:
            return None
        listed = subprocess.run([tool("clang-offload-bundler"), "--list", "--type=o", f"--input={fb}"],
                                capture_output=True, text=True, check=True).stdout.split()
        out = {}
        for target in (t for t in listed if "amdgcn" in t):
            co = os.path.join(d, "co.o")
            subprocess.run([tool("clang-offload-bundler"), "--unbundle", "--type=o", f"--input={fb}",
                            f"--targets={target}", f"--output={co}"], capture_output=True, check=True)
            notes = subprocess.run([tool("llvm-readelf"), "--notes", co], capture_output=True, text=True, check=True).stdout
            out[target.split("--")[-1]] = hostcall_kernels(notes)
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--llvm-bin", default=os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib", "llvm", "bin"))
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    exe = ".exe" if os.name == "nt" else ""
    tool = lambda name: os.path.join(a.llvm_bin, name + exe)
    if not os.path.exists(tool("clang-offload-bundler")):
        print(f"check_hostcall: no clang-offload-bundler in {a.llvm_bin} (--llvm-bin)")
        return 1

    libs = []
    for p in a.paths:
        if os.path.isdir(p):
            libs += [os.path.join(p, f) for f in sorted(os.listdir(p))
                     if (".so" in f or f.endswith(".dll")) and not os.path.islink(os.path.join(p, f))]
        else:
            libs.append(p)
    failed = scanned = 0
    for lib in libs:
        res = scan(lib, tool)
        if res is None:
            continue
        scanned += 1
        kernels = max(n for n, _ in res.values())
        bad = sorted({k for _, ks in res.values() for k in ks if k not in ALLOWED})
        allowed = sorted({k for _, ks in res.values() for k in ks if k in ALLOWED})
        if bad:
            failed += 1
            print(f"  FAIL {os.path.basename(lib)}: {len(bad)} kernel(s) need hostcall and will not launch without PCIe atomics:")
            for k in bad:
                print(f"         {k}   [{', '.join(t for t, (_, ks) in res.items() if k in ks)}]")
        else:
            print(f"  ok   {os.path.basename(lib)}: {len(res)} code objects, {kernels} kernels each, none need hostcall"
                  + (f" (beyond {len(allowed)} print-only debug kernel{'s' if len(allowed) != 1 else ''})" if allowed else ""))
    if scanned == 0:
        print("check_hostcall: no library with HIP code found in " + " ".join(a.paths))
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
