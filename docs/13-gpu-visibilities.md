# GPU nearest-neighbour search for Meshing's visibility passes

Meshing runs `createVerticesWithVisibilities` twice: once to score every point of the dense cloud
by the angles it is seen under, once more after the cloud has been filtered. Each pass
backprojects every valid depth-map pixel of every camera, finds the nearest vertex of the cloud in
a nanoflann kd-tree, and if that vertex is close enough (a few pixel sizes) records the camera on
it and pulls the vertex towards the pixel's 3D point. On the 107-photo engine bay job that is
152 M queries per pass against a 6.9 M point tree, then 3.4 M: 29 s and 27 s in 12 threads. This
page is the GPU replacement and how it was checked.

## What it is

`hip/port/gpu_knn/`: nanoflann's own tree, walked on the device. The host builds the tree exactly
as upstream does (nanoflann 1.9, leaves of 10, parallel build), flattens its nodes, and uploads
nodes, point permutation and coordinates. One device thread per query then replays
`KDTreeSingleIndexAdaptor::searchLevel` with an explicit stack: the same child order (the query's
side first), the same pruning test on the running lower bound, the same double arithmetic for the
metric and the split distances, strict `<` when a closer point is found. Because it is the same
tree walked the same way, the answer is nanoflann's, ties included; the only degree of freedom is
whether the host compiler fused the metric's `result += diff * diff` into an FMA. clang-cl under
/arch:AVX2 does, gcc on generic x86-64 (the Linux bundle) cannot, and the kernel has both forms:
the default follows the host build's `__FMA__` macro, `CHESHIRE_GPU_KNN_FMA=0|1` overrides.

Around the search, `gpu/visibilitiesGPU.inc` (textually included by `fuseCut/PointCloud.cpp`)
changes how the pass is organised, with the same arithmetic per vote:

- One snapshot of the coordinates for the whole pass. Upstream's tree reads the live vertex
  positions while other threads move them under per-vertex locks, so which vertex a pixel finds
  can depend on timing. Here every query of a pass is answered from the snapshot, and the votes
  are applied camera by camera in pixel order, each vertex by one thread: the pass is
  deterministic.
- Pinned, double-buffered host memory and one device stream: the device answers camera c while
  the host backprojects camera c + 1 and applies the votes of camera c - 1. Depth maps are read
  three cameras ahead on worker threads.
- The similarity map is not read. Upstream reads it, blurs it with a 10x10 gaussian and computes
  a score per pixel that nothing uses; that read and blur were most of what the CPU was doing
  once the search was gone.
- The per-vertex part of the vote threshold is evaluated once per vertex per pass instead of once
  per pixel (the same float expression).

`CHESHIRE_GPU_VIS=0` keeps upstream's code. `CHESHIRE_GPU_VIS_CHECK=1` answers every query on the
host as well, from the same snapshot tree, and counts differences. `CHESHIRE_GPU_VIS_LOG=1`
prints the timing. A device error mid-pass answers the remaining cameras on the host from the
same tree; a walk deeper than 96 levels (never seen) answers that pixel on the host.

## Measured (RX 9070, Windows, 12 host threads)

| engine bay, 107 photos | upstream | GPU |
|---|---|---|
| pass 1: 152 M queries, 6.9 M points | 29.0 s | 7.9 s (tree 1.9, backproject 1.1, votes 3.7; device 3.2 s, overlapped) |
| pass 2: 152 M queries, 3.4 M points | 27.0 s | 5.5 s (tree 1.0, backproject 1.3, votes 2.5; device 2.7 s, overlapped) |
| Meshing, end to end | 189 s (v0.2.10) | 159 s |

The 6-view job: 1.3 s and 0.8 s per pass, Meshing 18.3 s.

## Validation

`CHESHIRE_GPU_VIS_CHECK=1` on both jobs: the device's vertex and squared distance are identical
to nanoflann's on all 152,201,929 queries of each pass (engine bay) and all 8,808,856 (6-view),
with the FMA form of the metric. The vote counts of a pass differ from run to run by a few
thousand in 111 M because the point cloud that enters Meshing is itself produced by a racy
upstream stage (the depth-map fusion), not by this code; given the same cloud, the pass is
deterministic.

RX 5500 XT on Linux (house-pc, i3-4330, the v0.2.11 bundle, 41 views, 88.3 M queries per pass):
the device names the same vertex as nanoflann on every query of both passes, with either form of
the metric; about 20 % of the squared distances differ in the last ulp with both forms, so the
generic-x86-64 gcc build arranges the three products a third way. The distance only enters the
vote thresholds, where a last-ulp change would need an exact tie to matter. Passes 57 s and 52 s
to 18.3 s and 15.0 s (the kernel itself is 15 s and 13 s on RDNA1's double-precision rate, mostly
overlapped), Meshing 273.7 s (v0.2.10) to 200.9 s.

RX 6750 XT in the same machine, same bundle, same 41 views: passes 8.6 s and 6.8 s (kernel 2.3 s
and 2.0 s), Meshing 157.0 s, the cut 3.2 s; the same vertex on every query, the same last-ulp
distance note.

The engine bay itself on that machine (the same SfM and filtered depth maps as the RX 9070 runs,
copied over; i3-4330, RX 6750 XT, v0.2.11 bundle): Meshing 245.3 s, the same vertex as nanoflann
on all 152,201,929 queries of each pass, the cut 7.1 s. The RX 9070 with 12 host threads does the
same job in 159 s; the difference is almost all host work on four threads.

## What is left in Meshing

With the visibility passes at 13 s together, Meshing on the engine bay is 159 s: the depth-map
fusion that builds the cloud (~43 s, racy upstream), the kd-tree builds (~5 s), the
tetrahedralisation and its neighbour tables (~17 s), the GPU votes, the cut, and the post-processing.
