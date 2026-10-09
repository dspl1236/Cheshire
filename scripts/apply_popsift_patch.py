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

    python scripts/apply_popsift_patch.py --fixes-only

applies only the fixes that are not about HIP and that both backends take (the extremum counter's
uninitialised read, and the race in the descriptor normalisation, 3c and 3e below). The CUDA builds
reset the tree to upstream and then run this, since 0.3.8.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POPSIFT = ROOT / "third_party" / "popsift"
GEN = ROOT / "build" / "popsift-gen" / "popsift"
NL = chr(10)


def fix_extrema_counter() -> None:
    # 3c. The extremum counter, and the reason GPU SIFT reconstructed badly. extrema_count() does
    #
    #         int write_index;
    #         if( threadIdx.x == 0 ) { write_index = atomicAdd( extrema_counter, ct ); }
    #         write_index = popsift::shuffle( write_index, 0 );
    #
    #     so every lane but 0 reads write_index uninitialised, which is undefined behaviour. nvcc
    #     leaves the guard alone. The AMDGPU backend takes the licence and every lane performs the
    #     atomic, so the counter advances by 32 * ct instead of ct and each octave reports exactly
    #     32x the extrema it wrote. The surplus slots are never written, their i_ext_off stays 0 and
    #     they alias the octave's extremum 0; once AliceVision's grid filter runs, copy_if promotes
    #     them to survivors pointing at a zeroed InitialExtremum, whose sigma is 0, and a keypoint
    #     with no scale gets no descriptor at all. Initialising the variable removes the UB. The
    #     other lanes take lane 0's value from the shuffle either way, so nothing nvcc produces
    #     changes, and the CUDA builds take it too.
    extrema = POPSIFT / "src" / "popsift" / "s_extrema.cu"
    t = extrema.read_text(encoding="utf-8")
    if "cheshire" not in t:
        old_decl = ("    int write_index;" + NL
                    + "    if( threadIdx.x == 0 ) {" + NL)
        if t.count(old_decl) != 1:
            sys.exit("write_index declaration not found once in s_extrema.cu")
        new_decl = ("    // cheshire: lanes other than 0 read this below without it being assigned." + NL
                    + "    // That is undefined behaviour, and on AMDGPU it costs the guard: every lane" + NL
                    + "    // performs the atomic and the octave counts come out 32x too large." + NL
                    + "    int write_index = 0;" + NL
                    + "    if( threadIdx.x == 0 ) {" + NL)
        extrema.write_text(t.replace(old_decl, new_decl, 1), encoding="utf-8", newline=NL)


def fix_normalize_race() -> None:
    # 3e. The descriptor normalisation, and why the CUDA packages did not repeat themselves.
    #     normalize_histogram runs one warp per descriptor, 32 to a block. In the last block the warps
    #     past the end are clamped onto the last descriptor so that their shuffles have data, and
    #     ignoreme was meant to stop them writing it - but it was computed after the clamp, so it was
    #     never true. Those warps normalised the last descriptor again, in place, racing the warp that
    #     owns it; now and then one read it half written and it came out normalised twice. At most one
    #     descriptor per image, and only when the count is not a multiple of 32: in the 0.3.7 gate on
    #     a GTX 1080 Ti, one of about 970,000 on Windows and one in each of four views on Linux, each
    #     the other run's descriptor passed through RootSIFT a second time (docs/04). The AMD cards
    #     never showed it, and the fix leaves what they compute as it was.
    norm = POPSIFT / "src" / "popsift" / "s_desc_normalize.h"
    t = norm.read_text(encoding="utf-8")
    if "cheshire" not in t:
        old = ("    offset = ( offset < num_orientations ) ? offset" + NL
               + "                                           : num_orientations-1;" + NL
               + "    Descriptor* desc = &descs[offset];" + NL
               + NL
               + "    bool ignoreme = ( offset >= num_orientations );" + NL)
        if t.count(old) != 1:
            sys.exit("normalize_histogram's clamp and ignoreme not found once in s_desc_normalize.h")
        new = ("    // cheshire: decide before the clamp, or no warp is ever ignored and the warps past the" + NL
               + "    // end normalise the last descriptor a second time, racing its own warp." + NL
               + "    const bool ignoreme = ( offset >= num_orientations );" + NL
               + "    offset = ignoreme ? num_orientations-1 : offset;" + NL
               + "    Descriptor* desc = &descs[offset];" + NL)
        norm.write_text(t.replace(old, new, 1), encoding="utf-8", newline=NL)


def main() -> None:
    if not POPSIFT.exists():
        sys.exit(f"{POPSIFT} not found; clone alicevision/popsift v0.10.0 there first (see the docstring)")
    if "--fixes-only" in sys.argv[1:]:
        fix_extrema_counter()
        fix_normalize_race()
        print("popsift: the fixes both backends take are in (the extremum counter, the descriptor normalisation)")
        return

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
        # the enum is spelled with its class qualifier in the definition
        old_set = "void Config::setFilterSorting( Config::GridFilterMode m )" + NL + "{" + NL
        if t.count(old_set) != 1:
            sys.exit("setFilterSorting(Config::GridFilterMode) not found once")
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

    # 3c. the extremum counter (the function above; the CUDA builds take it too)
    fix_extrema_counter()

    # 3d. CHESHIRE_POPSIFT_DEBUG=1 also reports what the grid filter did per octave. The filter
    #     compacts the surviving extrema with copy_if and separately sets the count with reduce over
    #     the same stencil, so the two must agree; if the count exceeds what copy_if wrote, the
    #     orientation stage reads offsets nothing filled in and the keypoints come out with sigma 0.
    filt = POPSIFT / "src" / "popsift" / "s_filtergrid.cu"
    t = filt.read_text(encoding="utf-8")
    if "CHESHIRE_POPSIFT_DEBUG" not in t:
        old_copy = ("            thrust::copy_if( thrust::make_counting_iterator(0)," + NL
                    + "                             thrust::make_counting_iterator(ocount)," + NL
                    + "                             grid.begin()," + NL
                    + "                             off_ptr," + NL
                    + "                             fun_id );" + NL
                    + NL
                    + "            hct.ext_ct[o] = thrust::reduce( grid.begin(), grid.end() );" + NL)
        if t.count(old_copy) != 1:
            sys.exit("copy_if/reduce pair not found once in s_filtergrid.cu")
        new_copy = ("            thrust::device_ptr<int> off_end = thrust::copy_if(  // cheshire" + NL
                    + "                             thrust::make_counting_iterator(0)," + NL
                    + "                             thrust::make_counting_iterator(ocount)," + NL
                    + "                             grid.begin()," + NL
                    + "                             off_ptr," + NL
                    + "                             fun_id );" + NL
                    + NL
                    + "            hct.ext_ct[o] = thrust::reduce( grid.begin(), grid.end() );" + NL
                    + "            if( getenv( \"CHESHIRE_POPSIFT_DEBUG\" ) ) {  // cheshire" + NL
                    + "                const int wrote = (int)( off_end - off_ptr );" + NL
                    + "                fprintf( stderr, \"[popsift] filter octave %d: in %d, copy_if wrote %d, count set to %d%s%c\"," + NL
                    + "                         o, ocount, wrote, hct.ext_ct[o]," + NL
                    + "                         wrote == hct.ext_ct[o] ? \"\" : \"   <-- MISMATCH\", 10 );" + NL
                    + "            }" + NL)
        t = t.replace(old_copy, new_copy, 1)
        i = t.index("#include")
        t = t[:i] + "#include <cstdlib>  // cheshire" + NL + "#include <cstdio>" + NL + t[i:]
        filt.write_text(t, encoding="utf-8", newline=NL)

    # 3e. the descriptor normalisation's race (the function above; the CUDA builds take it too)
    fix_normalize_race()

    # 3f. The grid filter on the host, and why PopSIFT would not run from a generic code object (0.4.4,
    #     docs/16). The filter's device steps are rocThrust, and rocPRIM launches a trampoline kernel picked
    #     by the card's name (host_target_arch: gfx1201) whose body runs only
    #     `if constexpr(Arch == device_target_arch())`. Compiled for gfx12-generic, __amdgcn_processor__ is
    #     "gfx12-generic", device_target_arch() is unknown, and every rocThrust kernel compiles to nothing:
    #     no error, the counts stay zero, no cell is over the limit, and the host divides by that count
    #     (`/ ct`) - the 0xC0000094 of every generic PopSIFT so far. The steps are a few hundred thousand
    #     integers at most, so they move to the host: download each octave's initial extrema, the cell and
    #     scale extraction, the stable sort by cell and scale (rocPRIM's merge and radix sorts are stable,
    #     so std::stable_sort is the same order), the per-cell counts, the clamp the original already
    #     computed on the host, the disabled extrema and the compaction, then upload the flags and offsets.
    #     The result is byte for byte rocThrust's on a per-chip build. CHESHIRE_POPSIFT_HOST_FILTER=0 keeps
    #     the rocThrust path.
    t = filt.read_text(encoding="utf-8")
    if "CHESHIRE_POPSIFT_HOST_FILTER" not in t:
        old_head = ("    const int slots = conf.getFilterGridSize();" + NL
                    + NL
                    + "    thrust::device_vector<int>   octave_index( ext_total );" + NL)
        if t.count(old_head) != 1:
            sys.exit("extrema_filter_grid's head not found once in s_filtergrid.cu")
        host = r"""    const int slots = conf.getFilterGridSize();

    // cheshire: the filter on the host; see scripts/apply_popsift_patch.py 3f. rocThrust's kernels are
    // empty in a generic code object. CHESHIRE_POPSIFT_HOST_FILTER=0 runs upstream's rocThrust path below.
    const char* cheshire_host = getenv( "CHESHIRE_POPSIFT_HOST_FILTER" );
    if( !( cheshire_host && cheshire_host[0] == '0' ) )
    {
        const int n = slots * slots;
        cudaDeviceSynchronize( );

        // octave and index of every initial extremum, octave by octave, and its cell and scale
        std::vector<InitialExtremum> dat[MAX_OCTAVES];
        std::vector<int>   oct_v;   oct_v.reserve( ext_total );
        std::vector<int>   idx_v;   idx_v.reserve( ext_total );
        std::vector<int>   cell_v;  cell_v.reserve( ext_total );
        std::vector<float> scale_v; scale_v.reserve( ext_total );
        for( int o=0; o<MAX_OCTAVES; o++ ) {
            const int ocount = hct.ext_ct[o];
            if( ocount > 0 ) {
                dat[o].resize( ocount );
                cudaMemcpy( dat[o].data(), dobuf_shadow.i_ext_dat[o], ocount * sizeof(InitialExtremum), cudaMemcpyDeviceToHost );
                for( int i=0; i<ocount; i++ ) {
                    oct_v.push_back( o );
                    idx_v.push_back( i );
                    cell_v.push_back( dat[o][i].cell );
                    scale_v.push_back( ldexpf( dat[o][i].sigma, o ) );  // sigma * 2^octave, exact as powf's
                }
            }
        }
        const int total = (int)oct_v.size();

        // sorted by cell, then by scale as configured; stable, as rocPRIM's merge and radix sorts are
        std::vector<int> perm( total );
        for( int k=0; k<total; k++ ) perm[k] = k;
        const Config::GridFilterMode mode = conf.getFilterSorting();
        std::stable_sort( perm.begin(), perm.end(), [&]( int l, int r ) {
            if( cell_v[l] != cell_v[r] ) return cell_v[l] < cell_v[r];
            if( mode == Config::LargestScaleFirst )  return scale_v[l] > scale_v[r];
            if( mode == Config::SmallestScaleFirst ) return scale_v[l] < scale_v[r];
            return false;
        } );

        // entries per occupied cell, in cell order and packed to the front (reduce_by_key's output)
        std::vector<int> counts( n, 0 );
        int runs = 0;
        for( int k=0; k<total; ) {
            int e = k + 1;
            while( e < total && cell_v[perm[e]] == cell_v[perm[k]] ) e++;
            if( runs < n ) counts[runs] = e - k;
            runs++;
            k = e;
        }

        // from here the arithmetic is upstream's host code, on std::vector
        std::vector<int> offs( n ), lims( n );
        int acc = 0;
        for( int i=0; i<n; i++ ) { offs[i] = acc; acc += counts[i]; lims[i] = acc; }
        std::vector<int> sorted( counts );
        std::sort( sorted.begin(), sorted.end() );
        int ct = 0, pre = 0;
        for( int i=0; i<n; i++ ) {
            pre += sorted[i];
            if( sorted[i] * ( n-1-i ) + pre > conf.getFilterMaxExtrema() ) ct++;
        }
        if( ct == 0 )
            return ext_total;  // nothing over the limit: the offsets the extrema kernel wrote stand
        int tail = 0;
        for( int i=n-ct; i<n; i++ ) tail += sorted[i];
        float tailaverage = float( tail ) / ct;
        int   newlimit    = ::ceilf( tailaverage - ( ext_total - conf.getFilterMaxExtrema() ) / ct );

        // each cell keeps its first min(count, newlimit) entries in sorted order
        for( int i=0; i<n; i++ ) {
            const int from = offs[i] + std::min( counts[i], newlimit );
            for( int k=from; k<lims[i]; k++ )
                dat[oct_v[perm[k]]][idx_v[perm[k]]].ignore = true;
        }

        int ret_ext_total = 0;
        for( int o=0; o<MAX_OCTAVES; o++ ) {
            const int ocount = hct.ext_ct[o];
            if( ocount > 0 ) {
                std::vector<int> off;
                off.reserve( ocount );
                for( int i=0; i<ocount; i++ )
                    if( ! dat[o][i].ignore ) off.push_back( i );
                cudaMemcpy( dobuf_shadow.i_ext_dat[o], dat[o].data(), ocount * sizeof(InitialExtremum), cudaMemcpyHostToDevice );
                if( ! off.empty() )
                    cudaMemcpy( dobuf_shadow.i_ext_off[o], off.data(), off.size() * sizeof(int), cudaMemcpyHostToDevice );
                hct.ext_ct[o] = (int)off.size();
                if( getenv( "CHESHIRE_POPSIFT_DEBUG" ) )
                    fprintf( stderr, "[popsift] filter octave %d (host): in %d, kept %d%c", o, ocount, hct.ext_ct[o], 10 );
                ret_ext_total += hct.ext_ct[o];
            }
        }
        // the orientation kernels on the octave streams read these offsets
        cudaDeviceSynchronize( );
        writeDescCountersToDevice( );
        return ret_ext_total;
    }

    thrust::device_vector<int>   octave_index( ext_total );
"""
        t = t.replace(old_head, host.replace(chr(13), ""), 1)
        i = t.index("#include")
        t = t[:i] + "#include <algorithm>  // cheshire" + NL + "#include <cmath>" + NL + "#include <vector>" + NL + t[i:]
        filt.write_text(t, encoding="utf-8", newline=NL)

    # 3. one translation unit. HIP cannot produce relocatable device code with COFF objects on
    #    Windows, and PopSIFT shares __constant__ and __device__ globals across its sources, so the
    #    device link that -fgpu-rdc would need is unavailable. Compiling the sources together
    #    removes the need for it: every global is defined in the unit that references it.
    src = POPSIFT / "src"
    sources = sorted((src / "popsift").glob("*.cu")) + sorted((src / "popsift" / "common").glob("*.cu"))
    if not sources:
        sys.exit("no popsift sources found")
    lines = ["// Cheshire: PopSIFT as one translation unit; see scripts/apply_popsift_patch.py.",
             "//",
             "// The shim is included here rather than forced in from the command line. The two",
             "// drivers spell that flag differently (/FI against -include), and on Linux forcing a",
             "// header in ahead of clang's HIP runtime wrapper breaks glibc's feature detection",
             "// (__GLIBC_USE undefined). As the first line of the only translation unit it lands",
             "// after the wrapper and before every popsift source, which is what it needs.",
             '#include "popsift_hip.h"',
             ""]
    for f in sources:
        lines.append('#include "%s"' % f.relative_to(src).as_posix())
    (GEN.parent / "popsift_unity.cu").write_text(NL.join(lines) + NL, encoding="utf-8", newline=NL)

    print(f"popsift prepared; {len(sources)} sources in one unit -> {GEN.parent / 'popsift_unity.cu'}")


if __name__ == "__main__":
    main()
