# Meshing on the CPU: the exact quick wins

After the GPU vote passes ([docs/09](09-gpu-meshing-votes.md)) Meshing on the 107-photo engine bay
job still took 496 s, almost all of it CPU work around the Delaunay volume. Four of those blocks
were slow for reasons that have nothing to do with the algorithm, and each has an exact
replacement: same numbers, same order, verified in one process against upstream's construction.

| block | upstream | now | how |
|---|---|---|---|
| cells-around-each-vertex table | 47 s | ~1 s | counting instead of a `std::map` of `std::set` |
| nanoflann index on 61 M points | 71 s | 22 s | built on every core (nanoflann 1.9 builds the same tree from any thread count) |
| facet weights for the s-t graph | 30 s | 4.5 s | computed in parallel into a table, edges added in upstream's order |
| s-t graph build and teardown | 66 s + 25 s | 5.4 s | a compressed sparse row graph instead of `boost::adjacency_list` |
| Meshing node, end to end | 495.9 s | 296.5 s | |
| Meshing on the RX 5500 XT node (i3-4330, 41 views), end to end | 413.5 s (v0.2.7) | 302.9 s | same checks, zero differences on 1.80 M lists, 45.7 M facets, 11.4 M cells |

## The table

`Tetrahedralization::updateVertexToCellsCache` collects, for every vertex, the cells around it,
ascending and duplicate-free, through a `std::map<VertexIndex, std::set<CellIndex>>`: 87 M
insertions into red-black trees. Cells are visited in ascending order and each cell lists a vertex
once, so counting the entries per vertex, reserving, and appending in the same sweep produces
exactly the same lists. `CHESHIRE_MESH_OLD_NEIGHBOURS=1` keeps upstream's construction;
`CHESHIRE_GPU_VOTE_LOG=1` builds both and compares list for list (0 of 3.34 M lists differ on the
engine bay, 0 of 254 k on the 6-view set).

## The kd-tree

`PointCloud::createDensePointCloud` builds a nanoflann index over the 61 M candidate points to
remove duplicates, single-threaded. nanoflann 1.9 takes a thread count for the build and
partitions each subtree with the same split rule, so the tree, and every query against it, is the
same; the change is one constructor argument at the three places the index is built.

## The graph

`GraphFiller::binarize` computes two facet weights per tetrahedron face (two circumsphere centres,
three normalisations, an angle) 87 M times in one thread, then feeds them to a
`boost::adjacency_list` with a `std::vector` of out-edges per cell, whose 195 M `add_edge` calls
take 66 s and whose destructor takes 25 s. Now the weights go into a table in parallel (the same
expressions per facet, so the same floats; log mode recomputes them sequentially and compares:
0 of 87 M differ) and the edges are added from the table in upstream's order into
`hip/port/meshing_csr/MaxFlow_CSR.hpp`, which records the calls and lays the graph out as a
`compressed_sparse_row_graph` at compute time with every node's out-edges in exactly the order the
adjacency list would have held them: the s/t edge first, then facet edges and reverse edges in
insertion order. Boykov-Kolmogorov therefore visits the same edges in the same order and its
augmentations, flow value and labelling are the adjacency list's. `CHESHIRE_MAXFLOW_ADJLIST=1`
keeps upstream's class; `CHESHIRE_MAXFLOW_CHECK=1` runs both on the same graph and compares:
identical flow value and 0 of 21,748,827 cells labelled differently on the engine bay, 0 of
1,600,050 on the 6-view set.

## What is left in Meshing (engine bay, 296 s)

Boykov-Kolmogorov itself, 109 s, single-threaded by nature; the two visibility passes, 30 s each
(per-camera nearest-neighbour lookups of every depth-map pixel against the point cloud); loading
the depth maps, 21 s; the kd-tree, 22 s; the tetrahedralisation and its tables, 14 s; the GPU
votes, 18 s; graph-cut and mesh post-processing, 26 s. A GPU max-flow is the next large item and
its own project; the visibility passes are the next quick one.

## Two more of the same kind, outside Meshing

* **DepthMapFilter** re-read every neighbour depth map from EXR once per reference camera that
  listed it; the decoded maps are now shared across the process ([docs/08](08-gpu-depth-map-filter.md)):
  engine bay vote phase 12.5 s to 7.3 s, outputs byte-identical.
* **PrepareDenseScene** pins its per-image loop (read, exposure, undistort, EXR write) to three
  threads; it now uses every core (`CHESHIRE_PDS_THREADS` overrides): 107 images in 36 s to 30 s
  on the 12-thread desktop, the loop being mostly codec and disk work, outputs byte-identical
  (on the 4-core house-pc: 70 s to 68 s, 41 / 41 identical; the filter cache there: 27.5 s to
  20.4 s, 123 / 123 identical).

## The OBJ writer

Meshing's last step writes the mesh through Assimp: a scene is built, copied, and exported by
single-threaded stream formatting, 6.6 s for the 1.2 M-vertex engine bay mesh. The direct writer
(`Mesh::save`, apply step 4n) emits the same numbers the same way, float with 9 significant digits,
y and z negated as upstream does, 1-based faces, straight into a 4 MB buffer: about 0.2 s. The
file differs only in order: Assimp lists vertices in face-traversal order and renumbers faces,
the direct writer keeps the mesh's own order. Checked with `CHESHIRE_OBJ_CHECK=1`, which writes
both files: identical vertex set and identical triangles by coordinates on the 6-view job.
`CHESHIRE_OBJ_ASSIMP=1` keeps upstream. Engine bay Meshing 158.8 s to 145.1 s.

## The mesh cleaner's validation passes

`MeshClean::cleanMesh(maxIters)` runs three consistency tests before its loop and after every
iteration, 14 full passes over the mesh on the engine bay. They only emit debug-level log lines
and never change the mesh. Apply step 4o turns them off unless `CHESHIRE_MESHCLEAN_TESTS=1` and
logs each iteration's time. Engine bay: the cleaning step 17.1 s to 12.9 s (setup 6.6 s, then
4.0, 2.0, 2.1, 2.0 s per iteration, all single-threaded), Meshing 145.1 s to 139.0 s. What is
left there is the serial per-point loop and the adjacency setup ([docs/13](13-gpu-visibilities.md)
lists the order of the remaining Meshing work).

## The dense point cloud

Meshing starts by turning the filtered depth maps into a raw point array and thinning it: 42 s of
the engine bay job, 20 s reading maps and 22 s in `filterByPixSize`. Apply step 4p
([hip/port/fusion_filter/filterFusion.inc](../hip/port/fusion_filter/filterFusion.inc)) changes two
things.

The load loop runs one camera per thread on every core instead of three cameras at a time. Every
point already lands in a slot whose index is a pure function of camera and block, the per-camera
ranges are disjoint and nothing appends, so the thread count cannot change a value; running the
6-view job on 1, 3 and 12 threads gives identical candidate counts, identical survivor counts and an
identical round-by-round decision trace.

The filter itself was the last racy stage in Meshing: it writes `pixSize = -1.0` into the very array
the other threads' predicates are reading, so which of two mutual losers survives depended on
timing. It now decides in index order, which is upstream's loop at one thread, computed in rounds
that read only the previous round's decisions. `CHESHIRE_FILTER_CHECK=1` runs both in one process
and compares every slot.

The tree got smaller for a reason worth writing down: three quarters of the array is not points at
all. Every block the load loop discards keeps the value the arrays were constructed with, the
coordinate (0,0,0) with pixel size and similarity score zero, and upstream leaves all 50 million of
them in the tree, where they behave as one very cheap point at the origin that underbids everything
in reach. They are replaced by a single sentinel carrying the smallest score any of them presents.
Two conditions make that exact, and both are measured on every call: the excluded slots must share
one coordinate, and no candidate dropped mid-pass may underbid a live one. Upstream's function runs
unchanged if either fails.

| engine bay, 107 photos | before | after |
|---|---|---|
| load depth maps and add points | 20.1 s | 16.5 s |
| first filter (61,154,352 slots, 10,728,129 of them points) | 21.6 s | 5.2 s |
| Meshing end to end | 139.0 s | 112.4 s |

The 6-view job: 17.3 s to 10.8 s. Every filter call on both jobs is identical to single-threaded
upstream: 0 of 18,289,152 and 0 of 664,776 slots differ on the 6-view job, 0 of 61,154,352 and 0 of
5,967,295 on the engine bay.

The dense point cloud is now reproducible, but the mesh is not yet: the grid helper points seed from
`random_device` unless `delaunaycut.seed` is non-zero, and the GPU vote and min-cut kernels both
accumulate with float atomics, so two runs still differ by a handful of vertices.

## The grid helper points, and what is still not reproducible

Meshing surrounds the cloud with a grid of helper points, each nudged by uniform noise. Upstream
seeds that noise from `std::random_device` whenever `--seed` is 0, which is the default, then draws
from one shared `std::mt19937` inside an `omp parallel for`, and records the results in a
`std::vector<bool>` whose neighbouring elements share a word. So the helper set differed from run to
run three times over: a different seed, a scheduling-dependent order of draws, and torn flag writes.
Apply step 4q draws the noise in index order before the loop, which is the sequence one thread
produces, uses a fixed seed when none is configured, and gives each flag its own byte. `--seed`
still chooses a seed and `CHESHIRE_GRID_RANDOM=1` still asks for a random one;
`CHESHIRE_GRID_OLD=1` restores the shared generator.

The 6-view job, three runs (two on the default, one pinned to a single thread): 1205 helper points
with checksum `6b7dedaaa9611426` every time. The same build with `CHESHIRE_GRID_OLD=1`, same seed,
twice: 1204 points and 1200 points, different checksums, which is the parallel draw order alone.

What that leaves. After this step the following are identical run to run on the 6-view job: the
loaded points, both filter passes, the visibility passes, the final dense point cloud (265,271
points) and the tetrahedralisation (1,672,312 cells, 16,722,900 edges). The mesh still is not: the
graph-weight votes accumulate float contributions per cell in parallel, so the weights differ in
their last bits, and with them the cut. Running with `CHESHIRE_GPU_VOTE=0 CHESHIRE_GPU_MAXFLOW=0`
changes nothing about that, because upstream's CPU vote loop accumulates the same way. A
deterministic accumulation for the votes is the next step, and it would make Meshing reproducible
end to end.

## Why the mesh is still not byte-reproducible

With the point cloud and the helper points fixed, the obvious next suspect was the graph-weight
votes, which accumulate float contributions per cell in parallel. Dumping the s-t graph from two
runs of the 6-view job (`CHESHIRE_MAXFLOW_DUMP`) confirmed the weights move: 90 % of the 16,722,900
capacities differ and the total differs by 1.5 %, far beyond rounding, because the sink weight is
multiplied by an accumulated term and clamped at a million, so a small difference in a sum can swing
one capacity a long way.

It also showed something else. The two graphs are not the same graph with different numbers on the
edges: the cell-to-cell adjacency differs, while the internal edge count (13,378,280) and the degree
sequence as a multiset match exactly. That is a relabelling, so the tetrahedra themselves are
numbered differently from run to run. Apply step 4r logs a checksum either side of the
tetrahedralisation to place it. Three runs, and then three more: the 265,271 points handed to
geogram are byte-identical every time, and the 1,672,310 cells that come back are different every
time. Resetting geogram's own random generator before the build does not change that, and neither
does disabling geogram's multithreading (`CHESHIRE_TETRA_SINGLE_THREAD=1`), so it is not the biased
randomised insertion order and not its thread count.

So the votes are one of two sources, and the smaller one. Until the tetrahedralisation numbers its
cells reproducibly, deterministic vote accumulation cannot by itself make two runs agree, and the
cell numbering is upstream's, inside geogram. Both facts are worth knowing before anyone spends
effort on fixed-point accumulation: it is still a sound change, but it buys stability of the weights
given a numbering, not a reproducible mesh.

## The graph-cut post-processing block

The block between the cut and the mesh was one opaque 9.5 s on the engine bay. Apply step 4s times
its six passes separately, which changed what was worth doing: the 4-neighbour majority inversion,
ten rounds of a pure boolean stencil and the obvious candidate for the GPU, is 116 ms of 8358 ms.
The cost is elsewhere.

| pass (engine bay, 21.7 M cells) | before | after |
|---|---|---|
| solid-angle filtering, 2 rounds | 3301 ms | 2714 ms |
| removeBubbles | 2305 ms | 1617 ms |
| invertFullStatusForSmallLabels | 1773 ms | 1943 ms |
| removeDust | 862 ms | 667 ms |
| 4-neighbour inversion, 10 rounds | 116 ms | 98 ms |
| free the cells holding a camera | 0.25 ms | 0.26 ms |

Apply step 4t makes two exact reductions. `segmentFullOrFree`, the flood fill behind three of these
passes, colours a cell when it is pushed rather than when it is popped, so a cell enters the stack
once instead of up to four times; the colouring is unchanged because the seed order is unchanged and
a cell's colour is still the one its component's seed carries. `CHESHIRE_SEGMENT_CHECK=1` recomputes
it upstream's way in the same process and compares: identical on all 1,672,310 cells of the 6-view
job and all 22,078,541 of the engine bay, for both the empty and the full pass. And the solid-angle
filter stops allocating inside its inner loop, where it built a three-element `std::vector` for every
neighbouring cell of every surface vertex, tens of millions of allocations per round, plus a fresh
facet vector per vertex. Same values in the same order, from a fixed array and one buffer reused
across the vertex's cells.

The block is 7040 ms, and the 6-view job's 753 ms is 655 ms. What is left is dominated by the three
segmentation passes and the solid-angle geometry, both of which are parallel connected-component or
per-vertex work rather than anything the GPU would obviously win.

## The inversion count that was never read

`Mesher::graphCutPostProcessing` reads the number of 4-neighbour inversion rounds with
`get<bool>`, while `main_meshing` stores an `int` under that key. Boost's bool translator rejects
"10", the default is returned and collapses to `true`, so the loop ran once however many rounds
`--invertTetrahedronBasedOnNeighborsNbIterations` asked for. Apply step 4u reads it as the int it
is; `CHESHIRE_INVERT_OLD=1` restores upstream's single round.

The pass flips a tetrahedron whose status disagrees with three or four of its neighbours, so extra
rounds remove isolated spikes and pits that the first round exposes. What it costs and what it
changes, three runs per mode on the 6-view job and one each on the engine bay:

| | one round (upstream) | ten rounds |
|---|---|---|
| 6-view faces | 500,572 | 500,057 |
| 6-view pass time | 6.9 ms | 35 ms |
| engine bay faces | 2,364,995 | 2,361,583 |
| engine bay pass time | 92 ms | 918 ms |

The rounds converge on the 6-view job: 10,227 flips in the first, then 446, 8, 1, and nothing more.
The engine bay does not quite converge, settling into a 2-cycle of six cells each way from round 6
onward, which the fixed round count simply stops. The cost is 0.8 s of a 130 s Meshing run and the
effect is 0.14 % of the faces, so this is a small smoothing change rather than a dramatic one, but
it is the behaviour the parameter has always described.

## The similarity-map gaussian (v0.2.16)

Fusion's "Load depth maps and add points" was 17.7 s of Meshing, and the loading is not what it
spends it on. Timing the block's parts (`CHESHIRE_FUSION_PROFILE=1`) gives, in thread-seconds across
twelve threads:

| part | thread-seconds |
|---|---|
| **similarity-map gaussian** | **108.7** |
| depth map read | 26.6 |
| similarity map read | 25.5 |
| nmod map read | 13.3 |
| point loop | 6.7 |

60 % of a block named after the other 36 %. `imageAlgo::convolveImage` hands OIIO a
`make_kernel("gaussian", 10, 10)` - an 11x11 kernel, 121 taps - and calls the general
`ImageBufAlgo::convolve`, which does not exploit separability. Isolated, one call is 0.144 s
single-threaded; in the pipeline it costs about 1.0 thread-second per camera, because twelve threads
each sweep that stencil over a 9 MB image at once and the memory system is the limit rather than the
arithmetic. Moving it to the device removes the contention as much as the work: **17.69 s to 3.06 s**.

### Reproducing OIIO rather than approximating it

The blurred values feed a `score > bestScore` comparison that selects points, so a filter that is
merely close is a different filter. OIIO ships here as headers and a binary, so its arithmetic could
not be read and had to be matched empirically - `scratchpad/gaussbench.cpp` takes OIIO's own output
on a real similarity map as the reference and counts the pixels each candidate gets wrong.

Three things had to be right, and the count says which:

| candidate | pixels differing of 2,286,144 |
|---|---|
| **float accumulator, kernel-major, true divide** | **2** |
| reciprocal multiply instead of divide | 4,269 |
| fused multiply-add | 731,102 |
| double accumulator | 1,845,868 |

and separately, the borders: OIIO divides by the kernel weight that actually landed inside the
image, so a window of constant 1.0 returns exactly 1.0 rather than 0.343151. Without that, 25,136
border pixels are wrong and no interior pixel is - which is how the edge policy was identified, by
splitting the differences into border and interior rather than looking at a maximum.

The third was found by the device disagreeing with the harness. `__fmul_rn` and `__fadd_rn` look
like they prevent fusion; in HIP they are defined as the plain operators, so the compiler contracted
them anyway and 74.3 M of 244.6 M pixels came out a few ULP off. That is 30 %, and the harness had
already measured 32 % for a fused CPU variant - the number identified the cause. `#pragma clang fp
contract(off)`, as `depthMapFilterGPU.cu` already uses, fixes it.

### What it produces

244,617,279 of 244,617,408 pixels match bit for bit across the 107-photo engine bay. The other 129
differ by at most 4.77e-07 and none of them changes a decision: the first filter gives 6,896,447
points as OIIO does, the second 3,388,702, and the tetrahedralisation input checksums
`af3a3cce376e3f12` either way. The point cloud is the same one.

`CHESHIRE_GPU_BLUR_CHECK=1` runs OIIO as well and reports the divergence, so that is checkable on
other data instead of asserted; it roughly doubles the block, since it does the work twice.
`CHESHIRE_GPU_BLUR=0` keeps OIIO.

### Validated on four configurations

The device gaussian is the only new GPU code in the release, so each machine runs Meshing both ways
and compares the point cloud it produced against its own CPU reference - `CHESHIRE_GPU_BLUR=0` keeps
OIIO. Nothing has to be shipped between machines, and a failure points at one host rather than
needing triage.

| | card | host | build | tetrahedralisation input | blur check |
|---|---|---|---|---|---|
| RDNA4 | RX 9070 | Ryzen 5600X | Windows, AVX2 | `af3a3cce376e3f12` | 129 of 244,617,408 |
| RDNA2 | RX 6750 XT | FX-8120 | Windows, AVX | `6e0eb6cb7e58208` | 53 of 124,975,872 |
| RDNA1 | RX 5500 XT | FX-8120 | Windows, AVX only | `6e0eb6cb7e58208` | - |
| RDNA1 | RX 5500 XT | i3-4330 | Linux, GCC 13.3 | `2f0e9773fb427046` | 129 of 244,617,408 |

RDNA1 and RDNA2 on the same host, from the same inputs, give the same checksum - that is agreement
across architectures rather than each card merely agreeing with itself. The worst divergence anywhere
is 4.77e-07 and none of it changes a decision.

Compare the tetrahedralisation **input** checksum, which is the point cloud. The **output** checksum
differs between any two runs whatever you do, because geogram renumbers its cells from byte-identical
input; a comparison that picks the last `checksum` line in the log will report a difference that is
not there.

Timings, same job on each host:

| host | Meshing, OIIO gaussian | Meshing, device gaussian |
|---|---|---|
| RX 9070, Ryzen 5600X | 117 s | 101 s |
| RX 5500 XT, FX-8120 | 219 s | 202 s |
| RX 5500 XT, i3-4330 | 141 s | 111 s |
