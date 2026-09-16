# Validation: CUDA on the same host as the HIP runs (bench-pc, GTX 1080 Ti, Windows, 2026-09-16)

bench-pc (AMD FX-8120, 16 GB, Windows 11) got the GTX 1080 Ti (driver 581.57) after its RX 6750 XT
runs, and ran the same two caches with the CUDA `aliceVision_depthMapEstimation` from the Meshroom
2023.3.0 Windows release (CUDA 11.6 build) through `scripts/run-depthmap-cuda-standalone.cmd`,
which is the HIP runner with the binary path swapped. First CUDA-vs-HIP timing on identical host
hardware:

| bench-pc, same host | GTX 1080 Ti, CUDA 11.6 | RX 6750 XT, HIP 6.2 (v0.2.2 package) |
|---|---|---|
| 6 views | 40.0 s | 28.1 s |
| 41 views | 287.1 s | 190.3 s |
| simultaneous tiles | 14 | 24 |

The 1080 Ti here is slower than the 1080 Ti in house-pc was (31.9 s / 379 s in four chunks, i3-4330,
CUDA 11.3 build): different host, different CUDA build, 14 tiles on 11 GB. The HIP figure is the
one already on record for this host.

## CUDA against CUDA

The same card model with a different CUDA build and OS is **not bit-identical to itself**:
[vs the Linux CUDA 11.3 reference](vs-linux-cuda-reference.md): 37 / 41 depth maps byte-identical,
and of the four that differ, two differ in a handful of pixels, one has 1.75 % of pixels off by more
than 1 % (view 1227295871, masks 97.1 % agreeing), and one has 7.2 % (view 1430763847, p95 error
4.8 %). Median-view agreement 100 %, worst view 92.9 %.

That is the same size as the HIP-vs-CUDA spread (worst view 92.6 % on this set for the RX 6750 XT)
and is the cleanest evidence so far that the residual differences are the algorithm's sensitivity
to rounding in SGM's argmin, not the port: the CUDA build disagrees with another CUDA build by as
much as HIP disagrees with either.

[HIP RX 6750 XT vs this CUDA run](hip-rx6750xt-vs-this-cuda.md): masks 41 / 41, median-view within
1 % 98.8 %, worst 92.6 %, largest p95 5.5 %.
