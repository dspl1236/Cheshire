# GPU meshing votes

Meshing turns the filtered depth maps into a Delaunay tetrahedralisation and cuts it into inside
and outside with a max-flow. The cut's weights come from voting: for every real vertex and every
camera that sees it, a ray is marched through the tetrahedra twice, towards the camera (every
cell crossed votes "empty", the cells next to the camera are pinned to the source) and a few
pixel sizes behind the vertex (votes "full"). A second pass over the same rays, upstream's
`forceTedgesByGradientIJCV`, reads the finished emptiness scores along each ray and, where the
score jumps behind the point, strengthens the sink edge of the cell behind it (the
"weakly supported surfaces" filter). Both passes are per-ray independent; upstream runs them with
OpenMP in a randomised vertex order.

## What it is

`hip/port/gpu_vote/`: both passes as one thread per ray, wired into `GraphFiller::fillGraph` and
`forceTedgesByGradientIJCV` by `scripts/apply_hip_patch.py` (step 4e). The host packs the
tetrahedralisation once (vertices, the 4 vertex and 4 adjacent-cell indices per cell, the
cells-around-each-vertex lists as CSR) and the ray list (vertex, camera vertex, weight, the two
distance bounds, camera centre), uploads the mesh once and the rays in 4 M chunks, runs the vote
kernel over every chunk, then the tedge kernel over every chunk against the same resident mesh,
and downloads the eight per-cell attributes. `TetrahedronsRayMarching` (`Intersections.cpp`) is
transcribed in double precision with FMA contraction off and the same expression order, and
upstream's quirks are kept so each ray visits the same cells:

* the "is this hit farther than the best ambiguous one" and "did we move at all" tests compare
  `Eigen::Vector3d::size()`, which is the element count 3, not a length, so the first ambiguous
  hit wins and the "too close" test never fires;
* `Point3d::size()` returns 0 for a zero vector where Eigen's `norm()` would not;
* when the first step behind a vertex lands on a vertex or an edge instead of a facet, upstream
  reads `_cellsAttr[geometry.facet.cellIndex]` through the union, i.e. the cell whose index equals
  that vertex index (or the edge's first vertex); the kernel does the same.

The final `cellTWeight = max(cellTWeight, min(1e6, max(1, cellTWeight) * on))` sweep stays on the
host. `CHESHIRE_GPU_VOTE=0` restores both CPU passes; `CHESHIRE_GPU_VOTE_LOG=1` prints the
upload / vote kernel / tedge kernel / download split; `CHESHIRE_GPU_TEDGE_CHECK=1` runs the CPU
tedge loop as well and reports how its `on` vector compares with the GPU's.

## Measured (RX 9070, Windows)

Engine bay, 107 photos: 19.1 M rays, 21.7 M cells, 3.34 M vertices.

| phase | CPU (12 threads) | GPU |
|---|---|---|
| host packing (mesh, CSR, rays) | | 2.3 s |
| uploads | | 0.45 s |
| vote pass | ~22 s | 7.0 s |
| weakly-supported-surfaces pass | ~15 s | 8.0 s |
| download | | 0.05 s |
| Meshing node, end to end | 510.2 s | 495.9 s |

6 views (912 k rays, 1.6 M cells): the whole GPU phase is under a second (uploads 0.31 s, votes
0.26 s, tedges 0.24 s); Meshing 43.7 s either way.

RX 5500 XT on Linux (house-pc, i3-4330, 4 threads) with the v0.2.7 bundle, 41 views, 18.5 M rays
through 11.4 M cells: vote kernels 13.1 s, tedge kernels 12.6 s, uploads 0.36 s. Meshing end to
end 491.2 s on the CPU against 413.5 s with the GPU passes, and the GPU run still ran the CPU
tedge loop for the cross-check. On a node with a slow CPU the two passes are a bigger share of
the node and the saving is a sixth of it.

So the two ray-marching passes go from about 37 s to about 18 s on this desktop, and the node
barely notices, because they were never the majority of Meshing. Where the 510 s go (engine bay,
CPU): dense point cloud 74 s, tetrahedralisation 41 s, neighbour tables 47 s, the two vote passes
37 s, max-flow 207 s (63 s building the adjacency-list graph, 144 s of Boykov-Kolmogorov), mesh
post-processing 45 s. The max-flow is the node, and it is a different kind of problem (a
sequential augmenting-path algorithm on a 21.7 M-node graph); it is its own project.

## Validation

The weakly-supported-surfaces pass reproduces the CPU pass cell for cell: with
`CHESHIRE_GPU_TEDGE_CHECK=1` the CPU loop runs on the same emptiness scores right after the GPU
one and the two `on` vectors are compared. On the RX 9070 they match exactly on both sets (6
views: 1,468 non-zero cells, engine bay: 17,882 non-zero cells, max difference 0). On the RX 5500
XT's 41-view run the same cells are non-zero (25,482) with the same total, and the largest
per-cell difference is 8 on values around 1e8, one float ulp: where several rays add to the same
cell, the additions land in a different order. No cell differs by more than 1e-3 relative.

The vote pass cannot be bit-identical to the CPU, and the CPU is not bit-identical to itself:
votes are float atomics added in whatever order the threads arrive, and upstream randomises the
vertex order on purpose. Two identical CPU runs on the 6-view set gave 239,547 and 239,045 mesh
vertices; the GPU runs land in the same band. What the port guarantees is that
each ray votes on the same cells with the same weights.

| set | CPU runs, mesh vertices | GPU runs, mesh vertices |
|---|---|---|
| 6 views | 239,547 / 239,045 | 238,518 / 238,831 / 239,178 |
| engine bay | 1,180,941 | 1,180,433 / 1,180,474 / 1,180,656 |

## Pairing

`meshroom-pair.cmd` / `meshroom-pair.sh` pair `aliceVision_meshing` next to the other three,
gated on the binary's `--help` naming the GPU votes, so an older package's plain CPU `meshing`
is left alone. Meshroom 2023.3's Meshing node options are accepted unchanged (the runs above use
the exact command line Meshroom 2023.3 wrote into its cache).
