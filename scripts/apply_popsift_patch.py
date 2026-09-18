"""Prepare PopSIFT (alicevision/popsift v0.10.0) for the HIP build.

third_party/popsift is a clone, like third_party/aliceVision, and is patched rather than forked:

    git clone --depth 1 --branch v0.10.0 https://github.com/alicevision/popsift.git third_party/popsift
    python scripts/apply_popsift_patch.py

Two things are needed. The generated config header, which upstream's CMake writes and which the
sources include, and one genuine type error that CUDA does not notice: LinearTexture holds a handle
created by cudaCreateTextureObject and read with tex2D, but declares it cudaSurfaceObject_t. CUDA
makes both of those unsigned long long, so it compiles; HIP has them as distinct pointer types and
rejects it.

Everything else PopSIFT needs is supplied by hip/port/popsift/popsift_hip.h, which is force-included
in front of every translation unit. Nothing in the pyramid had to be restructured: HIP 7.2 has
layered arrays, layered surfaces and 2D textures natively, unlike the mipmapped arrays the depth-map
port had to emulate.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POPSIFT = ROOT / "third_party" / "popsift"
GEN = ROOT / "build" / "popsift-gen" / "popsift"
NL = chr(10)


def main() -> None:
    if not POPSIFT.exists():
        sys.exit(f"{POPSIFT} not found; clone alicevision/popsift v0.10.0 there first (see the docstring)")

    # 1. the config header upstream's CMake generates. HIP has the *_sync shuffles, and the grid
    #    filter stays on (rocThrust provides the algorithms it uses).
    template = POPSIFT / "cmake" / "sift_config.h.in"
    if not template.exists():
        sys.exit(f"{template} not found; is this popsift v0.10.0?")
    text = template.read_text(encoding="utf-8")
    text = text.replace("@PopSift_HAVE_SHFL_DOWN_SYNC@", "1").replace("@DISABLE_GRID_FILTER@", "0")
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "sift_config.h").write_text(text, encoding="utf-8", newline=NL)

    # 2. the texture handle declared with the surface object type
    octave = POPSIFT / "src" / "popsift" / "sift_octave.h"
    t = octave.read_text(encoding="utf-8")
    if "cheshire" not in t:
        old = "struct LinearTexture" + NL + "{" + NL + "    cudaSurfaceObject_t tex;" + NL + "};"
        if t.count(old) != 1:
            sys.exit("LinearTexture declaration not found once in sift_octave.h")
        new = ("struct LinearTexture" + NL + "{" + NL
               + "    // cheshire: created by cudaCreateTextureObject and read with tex2D, so it is a texture" + NL
               + "    // object. CUDA makes both handle types unsigned long long and does not notice; HIP has" + NL
               + "    // them as distinct pointer types." + NL
               + "    cudaTextureObject_t tex;" + NL + "};")
        octave.write_text(t.replace(old, new, 1), encoding="utf-8", newline=NL)

    # 3. one translation unit. HIP cannot produce relocatable device code with COFF objects on
    #    Windows, and PopSIFT shares __constant__ and __device__ globals across its sources, so the
    #    device link that -fgpu-rdc would need is unavailable. Compiling the sources together
    #    removes the need for it: every global is defined in the unit that references it.
    src = POPSIFT / "src"
    sources = sorted((src / "popsift").glob("*.cu")) + sorted((src / "popsift" / "common").glob("*.cu"))
    if not sources:
        sys.exit("no popsift sources found")
    lines = ["// Cheshire: PopSIFT as one translation unit; see scripts/apply_popsift_patch.py.", ""]
    for f in sources:
        lines.append('#include "%s"' % f.relative_to(src).as_posix())
    (GEN.parent / "popsift_unity.cu").write_text(NL.join(lines) + NL, encoding="utf-8", newline=NL)

    print(f"popsift prepared; {len(sources)} sources in one unit -> {GEN.parent / 'popsift_unity.cu'}")


if __name__ == "__main__":
    main()
