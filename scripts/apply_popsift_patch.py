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

    # 3a. CHESHIRE_POPSIFT_SORT overrides the extrema filter's sort order. AliceVision asks for
    #     LargestScaleFirst, which keeps the biggest-scale extrema; those live in the most blurred
    #     pyramid levels, so their descriptors can come out sparse. The setter honours the override
    #     so the choice can be measured without rebuilding AliceVision.
    conf = POPSIFT / "src" / "popsift" / "sift_conf.cu"
    t = conf.read_text(encoding="utf-8")
    if "CHESHIRE_POPSIFT_SORT" not in t:
        old_set = "void Config::setFilterSorting( GridFilterMode m )" + NL + "{" + NL
        if t.count(old_set) != 1:
            sys.exit("setFilterSorting(GridFilterMode) not found once")
        new_set = (old_set
                   + "    if( const char* s = getenv( \"CHESHIRE_POPSIFT_SORT\" ) )  // cheshire" + NL
                   + "    {" + NL
                   + "        if( s[0] == 's' ) m = SmallestScaleFirst;" + NL
                   + "        else if( s[0] == 'r' ) m = RandomScale;" + NL
                   + "        else if( s[0] == 'l' ) m = LargestScaleFirst;" + NL
                   + "    }" + NL)
        t = t.replace(old_set, new_set, 1)
        if "#include <cstdlib>  // cheshire" not in t:
            i = t.index("#include")
            t = t[:i] + "#include <cstdlib>  // cheshire" + NL + t[i:]
        conf.write_text(t, encoding="utf-8", newline=NL)

    # 3. CHESHIRE_POPSIFT_DESCMODE picks the descriptor implementation. PopSIFT ships six of them
    #    for the same algorithm and defaults to Loop; on HIP they do not agree, so this makes the
    #    choice testable without rebuilding AliceVision.
    conf = POPSIFT / "src" / "popsift" / "sift_conf.cu"
    t = conf.read_text(encoding="utf-8")
    if "CHESHIRE_POPSIFT_DESCMODE" not in t:
        old_ctor = "    , _desc_mode( Config::Loop )" + NL
        if t.count(old_ctor) != 1:
            sys.exit("desc mode default not found once")
        t = t.replace(old_ctor, "    , _desc_mode( Config::Loop )  // cheshire: overridden below" + NL, 1)
        # set it at the end of the constructor body, where the member list has been applied
        marker = "Config::Config( )" + NL
        if marker not in t:
            sys.exit("Config constructor not found")
        head, rest = t.split(marker, 1)
        brace = rest.index("{")
        depth = 0
        for k in range(brace, len(rest)):
            if rest[k] == "{":
                depth += 1
            elif rest[k] == "}":
                depth -= 1
                if depth == 0:
                    break
        inject = ("    if( const char* m = getenv( \"CHESHIRE_POPSIFT_DESCMODE\" ) )  // cheshire" + NL
                  + "        setDescMode( m );" + NL)
        rest = rest[:k] + inject + rest[k:]
        t = head + marker + rest
        i = t.index("#include")
        t = t[:i] + "#include <cstdlib>  // cheshire" + NL + t[i:]
        conf.write_text(t, encoding="utf-8", newline=NL)

    # 3. CHESHIRE_POPSIFT_DEBUG=1 prints the per-octave extrema counts as they arrive on the host,
    #    which splits the pipeline: zero everywhere means detection or earlier, non-zero with no
    #    descriptors means orientation or the descriptor pass.
    ori = POPSIFT / "src" / "popsift" / "s_orientation.cu"
    t = ori.read_text(encoding="utf-8")
    if "CHESHIRE_POPSIFT_DEBUG" not in t:
        old_call = "    readDescCountersFromDevice( );" + NL
        if t.count(old_call) != 1:
            sys.exit("readDescCountersFromDevice call not found once")
        new_call = (old_call
                    + "    if( getenv( \"CHESHIRE_POPSIFT_DEBUG\" ) ) {  // cheshire" + NL
                    + "        fprintf( stderr, \"[popsift] extrema per octave:\" );" + NL
                    + "        for( int o=0; o<MAX_OCTAVES; o++ ) fprintf( stderr, \" %d\", hct.ext_ct[o] );" + NL
                    + "        fprintf( stderr, \" | ori per octave:\" );" + NL
                    + "        for( int o=0; o<MAX_OCTAVES; o++ ) fprintf( stderr, \" %d\", hct.ori_ct[o] );" + NL
                    + "        fprintf( stderr, \"%c\", 10 );" + NL
                    + "    }" + NL)
        t = t.replace(old_call, new_call, 1)
        i = t.index("#include")
        t = t[:i] + "#include <cstdlib>  // cheshire" + NL + "#include <cstdio>" + NL + t[i:]
        ori.write_text(t, encoding="utf-8", newline=NL)

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
