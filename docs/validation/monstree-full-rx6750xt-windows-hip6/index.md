# Validation: monstree-full (41 views) on bench-pc, RX 6750 XT, Windows, HIP SDK 6.2 build (2026-09-15)

Same card that produced the [Linux / HIP 7.2 result](../monstree-full-rx6750xt-linux/index.md), now on
Windows 11 through AMD's HIP 6 runtime (`amdhip64_6.dll`, driver 32.0.12052.2) with the
`cheshire-alicevision-hip6.2-windows-x64-gfx1031-avx.zip` package from v0.2.2. Host: AMD FX-8120
(no AVX2), 16 GB. One `aliceVision_depthMapEstimation` process over all 41 views
(`run-depthmap-standalone.cmd`), same DepthMap parameters as the CUDA reference.

| | CUDA (GTX 1080 Ti, four Meshroom chunks) | HIP 7.2, same card, Linux (emulated mipmaps) | HIP 6.2, same card, Windows, emulated mipmaps | HIP 6.2, same card, Windows, native mipmaps |
|---|---|---|---|---|
| DepthMap wall time, 41 views | 379.0 s | 226.1 s | 232.3 s | **190.3 s** |
| within 1 % of CUDA, median view / worst view | | 98.1 % / 92.3 % | 98.7 % / 91.7 % | 98.7 % / 91.7 % |
| largest p95 relative error | | 0.062 | 0.071 | 0.071 |

Planner: 6 tiles per image, 4 depth maps at once, 24 simultaneous tiles, peak 6.0 GB of volumes
in VRAM, no spills, in all three HIP runs.

**Where the Linux/Windows gap comes from.** A second Windows package built with mipmap
emulation forced on (`CHESHIRE_MIPMAP_NATIVE=0`, otherwise identical) took 232.3 s and produced
depth maps bit-identical to the native run on 41 / 41 views ([table](mipemu-vs-native.md)).
That is within 3 % of the Linux bundle, which always emulates because ROCm 7.2 on Linux has no
`hipMallocMipmappedArray`. So the 16 % between Linux and Windows on this card is the mipmap
path, not the host CPU (i3-4330 vs FX-8120), not the runtime (HIP 7.2 vs 6.2) and not the OS;
at equal mipmap path the two stacks are within noise of each other. Native mipmaps are worth
18 % on RDNA2 for this job (they were 15-35 % on the RX 9070 depending on the set).

* [vs the CUDA reference](vs-cuda.md): masks agree on 41 / 41 views, per-view median error 0.0000 on
  every view, within 1 %: median over views 0.987, worst 0.942 (view 1317225462),
  best 1.000 (view 88904561); strict criterion FAIL like every other AMD card
  (the cross-GPU noise floor, see the RX 5500 XT pages).
* [vs the RX 9070 on Windows / HIP 7.2](vs-rx9070-windows-hip7.2.md): masks identical, median 0,
  within 1 % median view 0.995, worst 0.942, largest p95 0.016. Two AMD generations and two
  toolchains agree with each other more closely than either agrees with CUDA.

The Linux outputs of this card are on house-pc, which was offline when this page was written, so
the same-card comparison exists only for the [6-view set](../monstree-mini6-rx6750xt-windows-hip6/index.md).

Run log: `data/out/monstree-full-hip6-rx6750-win/run.log` (not in the repo).
