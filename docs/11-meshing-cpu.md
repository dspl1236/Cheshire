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
