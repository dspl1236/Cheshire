# GPU texturing

Texturing unwraps the filtered mesh into UV atlases, picks the best cameras for every triangle,
then, camera by camera, loads the image, builds a three-band Laplacian pyramid and rasterises
every triangle that camera contributes to into per-atlas, per-band accumulators (weighted colour
sums and weights). At the end each atlas is normalised, its bands fused, padded, downscaled and
written. On the 107-photo engine bay job the node took 220 s inside Meshroom and 165.8 s run
standalone; profiled, that was 21.7 s of image reads, 19.2 s of pyramids, 37.1 s of rasterisation,
about 20 s of single-threaded camera selection, 17 s of padding and downscale, and the rest UV
unwrap, mesh load and save and EXR writes.

## What it is

`hip/port/gpu_texturing/`: the per-camera pyramid and rasterisation and the final
normalise-and-fuse on the GPU, wired into `Texturing::generateTexturesSubSet` by
`scripts/apply_hip_patch.py` (step 4f). The host still reads each image (EXR, exposure
compensation and the OCIO colour conversion are OpenImageIO work) and uploads it; the GPU builds
the Laplacian pyramid with OpenImageIO 3.0's separable resize transcribed (its Gaussian filter,
its `fast_exp` polynomial, its weight normalisation and accumulation order) and the bilinear
difference upstream uses, then runs one thread block per triangle over the triangle's bounding
box: geogram's `point_triangle_squared_distance` for the inside test and barycentrics, the
projection, the in-image margin, the "pure zero" test on the source image, and float atomics into
the accumulators for the triangle's band and every lower frequency, exactly the CPU's expression
order with FMA contraction off. The accumulators live in VRAM: 3.2 GB per 8192² atlas at three
bands, so atlases are processed in chunks of what the card holds (four on a 16 GB card, two on
8 GB); with enough RAM every image is kept for the whole job so a second chunk re-reads nothing,
and the next four cameras are read ahead on their own threads while the current one is on the GPU.
The single-threaded camera selection is scored in contiguous triangle ranges in parallel and
merged in order, so its lists are exactly the sequential ones (`CHESHIRE_TEX_PARTS=1` runs the
sequential loop, and `CHESHIRE_GPU_TEX_LOG=1` prints an order-sensitive checksum of the lists to
compare, plus the time split). Padding, the Lanczos downscale (OpenImageIO's filter is built on
the C runtime's `sinf`, which a GPU cannot reproduce bit for bit) and the EXR writes stay on the
CPU. `CHESHIRE_GPU_TEX=0` restores the CPU loop.

## Measured (RX 9070, Windows, standalone)

| | CPU | GPU |
|---|---|---|
| engine bay, 107 photos, 6 atlases: image reads | 21.7 s | 5.3 s waiting (read ahead) |
| pyramids | 19.2 s | 0.17 s |
| rasterisation | 37.1 s | 3.4 s |
| camera selection | ~20 s | 3 s |
| Texturing node, end to end | 165.8 s (220 s inside Meshroom) | 79.3 s |
| 6 views, 1 atlas, end to end | 16.6 s | 10.3 s |

RX 5500 XT (8 GB) on Linux (house-pc, i3-4330, 14 GB RAM) with the v0.2.8 bundle, 41 views, 3
atlases in chunks of 2 with all 41 images kept in RAM: Texturing 271.5 s on the CPU against 102.5 s
with the GPU passes (uploads 3.0 s, pyramids 0.7 s, rasterisation 4.1 s, image reads 10.1 s
waiting). The textures have the identical set of coloured pixels and differ on 0.008 % of the
pixels, at most 0.05, 19 to 30 pixels beyond 1e-3 per texture.

What is left on the engine bay: UV unwrap 14 s, mesh load 9 s and save 14 s, per-atlas padding,
downscale and write about 3.5 s each. Those are the node now.

## Validation

The rasteriser is a float accumulation in whatever order the contributions arrive, on the CPU as
well: upstream's per-camera triangle loop is an OpenMP `+=` with no atomics, so texels at chart
edges, where two triangles of one camera share a texel, are racy there. Two CPU runs of the same
binary on the engine bay differ on 0.009 to 0.013 % of the pixels per texture, by up to 0.35 in
one texel and beyond 1e-3 on 100 to 320 pixels per texture. GPU against CPU is inside that band:
0.010 to 0.013 % of the pixels, at most 0.045, beyond 1e-3 on 36 to 170 pixels per texture, and the
set of coloured pixels is identical texture for texture. On the 6-view set the two differ on 878
pixels of 16.8 M by one half-float ulp at most, none beyond 1e-3 (two CPU runs: 56 pixels, same
ulp). Texels touched by one triangle per camera, the overwhelming majority, follow the same float
operations in the same camera order and come out equal; the textures are written as half floats,
which rounds most of the remaining accumulation-order noise away.

A note on comparing against Meshroom's own output: its stock `aliceVision_texturing` was built
with another compiler and OpenImageIO version, and 25 to 40 % of the pixels differ from either of
our CPU runs at the ulp level. Run-to-run of one binary is the right baseline.

## Pairing

`meshroom-pair.cmd` / `meshroom-pair.sh` pair `aliceVision_texturing` next to the other four,
gated on the binary's `--help` naming the GPU pass. Meshroom 2023.3's Texturing node options are
accepted unchanged.

## The chart packer (v0.2.16)

Basic UV unwrapping was 14.4 s of Texturing's 79 s, and the obvious suspect was wrong. The walk that
assigns UVs keeps a `std::map<int,int>` per chart - the pattern that cost 47 s elsewhere in this
codebase ([docs/11](11-meshing-cpu.md)) - and it does 7,029,609 lookups over 69,565 charts and
2,343,203 triangles in **0.47 s**. It is not the cost. Timing the four phases of the `UVAtlas`
constructor instead:

| phase | before | after |
|---|---|---|
| createCharts | 4.34 s | 4.81 s |
| packCharts | 3.31 s | 3.63 s |
| finalizeCharts | 0.06 s | 0.07 s |
| **createTextureAtlases** | **6.28 s** | **0.70 s** |
| **packCharts** | **3.31 s** | **1.90 s** |
| UVAtlas total | 14.01 s | 8.07 s |

Unwrap end to end: 14.41 s to 8.56 s.

(The two unchanged phases move by about 10 % between runs; that is the machine, not the change.)

`createTextureAtlases` packs charts into texture atlases with a binary tree of free rectangles, and
`ChartRect::insert` descends the *whole* tree for every chart. Occupied leaves return `nullptr` but
are still visited, so as the tree grows each of 69,565 inserts re-walks thousands of full nodes -
O(n^2), about 90 us per insert for what is nominally a tree descent.

Each node now carries `maxFreeW` and `maxFreeH`, the largest free extent anywhere beneath it, and
the descent skips branches that cannot hold the chart. That is an upper bound rather than an exact
one: a subtree might hold a wide-short rectangle and a narrow-tall one, so the pair describes no
single rectangle. It does not need to. Every free rectangle below the node has width <= `maxFreeW`
and height <= `maxFreeH`, so a chart exceeding either cannot fit in any of them, and no branch that
could have fitted is ever skipped. Everything else is untouched, so the same leaves are visited in
the same depth-first order and the first fit is the same one.

The structure is deliberately not replaced. A quadtree or a grid would pack faster still and would
pack *differently* - the leaf that wins is the packing - and the output would no longer be
upstream's. As it is, the 231 MB `texturedMesh.obj` is byte for byte the one the unmodified build
produces, which is the whole test.

### packCharts

Every triangle contributed three `Edge` objects, each holding a `std::vector<int>` with exactly one
element: 7,029,609 heap allocations and as many frees, an array that reallocated its way to 7 M
elements because it was never reserved, and a sort moving 40-byte objects through a non-trivial move
constructor instead of memcpying 12-byte PODs. It is one contiguous array of
`struct RawEdge { int p0, p1, tri; }` now, reserved up front. 3.31 s to 1.90 s.

That the result is unchanged rests on something narrower than it looks. The merge compares only
*adjacent* entries, so on a non-manifold edge - three or more triangles sharing one - which entries
end up adjacent decides which pairs are emitted, and that is decided by how the sort ordered ties.
`std::sort`'s control flow depends only on comparison outcomes, so the same keys in the same initial
positions give the same permutation, ties included; the keys and their order are preserved exactly.

The merge itself simplifies once written out: upstream appended b's single id to a's list, sorted
the two, and pushed a. Each entry is the left of a pair exactly once and the right exactly once, and
is read before it is written, so every emitted edge held exactly two ids in ascending order - the
`size() != 2` test downstream never fired. The pair is stored directly.

### An upstream bug, left alone

`createCharts` collects `std::vector<std::pair<float, int>>` of (projected area, camera) and sorts
it with `std::greater<std::pair<int, int>>`. The types do not match, so every comparison constructs
two temporary `pair<int, int>` and **truncates the area to an integer**: projected areas of 10.9 and
10.1 compare equal, and cameras are ordered by whatever the sort does with a tie.

That is a defect, not a performance question, and correcting it would change which cameras each
chart keeps and therefore the packing. It is reported rather than patched.

What is left in `createCharts` is 2.44 s in `computeTrisCamsFromPtsCams` and about 2.96 s in a
12-thread projection loop. Making the per-triangle camera list a reference rather than a copy is in
(it was copying a `StaticVector<int>` 2.34 M times to read values it never modifies) but it is
within run-to-run noise: those lists are short and the loop is parallel.