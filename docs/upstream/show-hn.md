# Show HN

Posted 2026-09-14 as a regular submission (HN gates the Show HN section for new accounts):
https://news.ycombinator.com/item?id=49700640

Submit at https://news.ycombinator.com/submit with title + URL (leave "text" empty), then add
the comment below as the first comment. Tuesday to Thursday, 9 to 11 am Eastern.

**Title (79 chars):** Show HN: Meshroom photogrammetry on AMD GPUs via HIP, bit-identical, with a VRAM bridge

**URL:** https://github.com/dspl1236/cheshire

---

AliceVision's dense reconstruction (the DepthMap stage Meshroom needs a GPU for) has only
ever run on NVIDIA. I compile those same CUDA kernels as HIP through a small compatibility
header, without touching the CUDA sources, and package it for Windows (ROCm arrives as pip
wheels there, no admin installer) and Linux (a relocatable bundle that needs only the amdgpu
driver).

Validation is the part I care about: same photos, same SfM, same Meshroom 2023.3 parameters
as a CUDA node, only the GPU stage differs. Depth maps are bit-identical between RDNA1 and
RDNA2, and match the GTX 1080 Ti output to the algorithm's cross-hardware noise floor
(97-99 % of pixels within 1 %, identical validity masks, median error zero). An RX 9070 runs
the stage faster than the 1080 Ti on 6 views; an RX 6750 XT matches it. Full per-view tables
and side-by-side panels are in the repo.

The second half is a memory bridge: a class-aware allocator that spills from VRAM into
mapped system RAM instead of failing, plus a change to AliceVision's tile planner so it
sizes work from the real budget. Measured per buffer class behind PCIe: a card with 1 GB of
VRAM free now runs the stage at full speed (the old planner spilled 6 GB and ran 4x slower),
and 500 MB runs where upstream refuses the job.

Two things found on the way that may matter beyond photogrammetry:

1. GPU atomics into fine-grained mapped host memory are silently wrong on Linux when the
   platform lacks PCIe atomics. atomicMin fails on every element, atomicAdd works, both are
   right on non-coherent (device-cached) memory. Reported to ROCm with a 100-line
   reproducer: https://github.com/ROCm/clr/issues/285
2. Coarse-grained host memory is also the difference between texture-sampled data behind
   PCIe costing 22x and costing nothing; the device caches it. If you spill anything a kernel
   samples through textures, allocate it non-coherent.

Ask: every package carries RDNA3 code objects and none has run on an RDNA3 card, because I
don't own one. If you have an RX 7000, the README has a ten-minute reproduce.
