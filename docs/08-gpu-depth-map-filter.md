# GPU depth map filter

DepthMapFilter removes depth values that other cameras do not confirm. Its "precomputing groups"
pass is 93 % of the node: for every reference camera rc and each of its 10 nearest neighbours tc,
every tc pixel with a depth is back-projected to 3D, projected into rc, and counts as a vote for
that rc pixel if the two depths agree within a pixel-size tolerance derived from both cameras'
geometry. Upstream runs it on the CPU and, per tc pixel, redoes an RQ decomposition of both
projection matrices inside the tolerance computation; that was 115 s of the 107-photo engine bay
job.

## What it is

`hip/port/gpu_filter/`: the vote pass as one thread per tc pixel, wired into
`Fuser::filterGroupsRC` by `scripts/apply_hip_patch.py` (step 4d). The per-camera decompositions
run once on the host with upstream's own function; the per-pixel chain (`updateInSurr`,
`getPixelFor3DPoint`, `getCamPixelSize`, `getCamPixelSizeRcTc`, `getTarEpipolarDirectedLine`,
`get2dLineImageIntersection`, `triangulateMatch`, `lineLineIntersect`, `pointLineDistance3D`) is
transcribed line for line in double precision with FMA contraction off and upstream's own
float/double conversions kept, so the counts come out bit-identical. Votes land through integer
atomics in VRAM; the second pass (`filterDepthMapsRC`, a per-pixel threshold on the counts) stays on
the CPU. `CHESHIRE_GPU_FILTER=0` restores the CPU pass; `CHESHIRE_GPU_FILTER_DEBUG=1` prints one
tc pixel's intermediate values from both implementations side by side and the per-camera vote
counts, which is how the two findings below were made.

## Measured (RX 9070, Windows, all views in one process)

| set | CPU pass | GPU pass | outputs |
|---|---|---|---|
| 6 views | 6.4 s | 1.5 s | 12 / 12 files byte-identical, and pixel-identical to Meshroom 2023.3's own output |
| 41 views | 58.8 s (votes 55.7 s) | 10.0 s (votes 7.1 s) | 123 / 123 files identical |
| engine bay, 107 views | 123.5 s (votes 114.3 s) | 26.5 s (votes 19.0 s) | 321 / 321 files identical |

RX 5500 XT on Linux (house-pc, i3-4330), 41 views with the v0.2.6 bundle: CPU pass 317.0 s, GPU pass
26.9 s, 123 / 123 output files byte-identical. On a node with a slow CPU the win is 12x.

What was left in the GPU pass was reading: each rc read its 10 neighbours' depth maps from EXR
again, 1,177 reads for 107 views. Since v0.2.9 the decoded maps are shared across the reference
cameras of the process (a mutex-guarded table with one `std::call_once` per map, capped by
`CHESHIRE_FILTER_CACHE_MB`, default 4096, 0 disables): the engine bay's vote phase goes from
12.5 s to 7.3 s and the node from 17.8 s to 12.7 s, with all 321 output files byte-identical to the
uncached run.

## An upstream finding: the vote buffer is never cleared

The first GPU build cleared its vote buffer between neighbour cameras and produced far fewer
consistent pixels than the CPU: for one reference view the first neighbour's vote count matched
exactly (1,071,556 pixels) and every later neighbour's CPU count only grew (1,157,210, 1,171,124 ...).
Upstream "resets" the per-camera buffer with `StaticVector::resize_with(w * h, 0)`, which is
`std::vector::resize` and leaves the existing elements alone once the vector has its size. So
`numOfModalsMap` does not count the neighbours that agree with a pixel; it counts the neighbours
from the first one that agreed onward. Meshroom's `minNumOfConsistentCams` thresholds were tuned
against that behaviour, and the whole Meshroom output downstream depends on it, so the GPU pass
replicates it by default (one clear per reference camera) and is bit-identical to the CPU pass.
`CHESHIRE_GPU_FILTER_STRICT=1` clears per neighbour, which is what the code says it does; that mode
removes more depth values and would need its thresholds re-tuned before it is a better filter
rather than just a stricter one.

## Pairing

`meshroom-pair.cmd` / `meshroom-pair.sh` pair `aliceVision_depthMapFiltering` next to the other
two, gated on the binary's `--help` naming the GPU pass, so an older package's plain CPU
`depthMapFiltering` is left alone. Meshroom 2023.3's node options are accepted unchanged.
