# Performance tuning (2026-09-03)

Target: the 41-view `monstree-full` DepthMap run on the RX 9070 (Windows, ROCm 7.2.1).
Every change below keeps the output **bit-identical** to the validated build unless stated.

| build | monstree-mini6 (6 views) | monstree-full (41 views) |
|---|---|---|
| CUDA node (GTX 1080 Ti, Meshroom 2023.3) | 31.9 s | 379.0 s (four 12-view chunks; 105.5 s was chunk 0 only, corrected 2026-09-15) |
| first validated HIP build (float4 textures) | 21.2 s | 154.7 s |
| + fused SGM aggregation | 18.0 s | 134.0 s |
| + half4 textures restored (buffer-copy mip levels) | 17.8 s | 131.2 s |
| + parallel image prefetch, cached symbol addresses | **17.4 s** | **124.4 s** |

Net: -18 % on the 6-view set, -20 % on the 41-view set. The RX 9070 beats the 1080 Ti on
the small set and is still ~18 % behind it on the large one.

## How the time is spent (6-view run, 17.8 s, stage-level syncs, `CHESHIRE_PROFILE_SGM=1`)

| stage | per tile | share |
|---|---|---|
| SGM compute similarity volume | 140 ms | 28 % |
| Refine and fuse | 98 ms | 20 % |
| SGM optimize (path aggregation, fused) | 81 ms | 17 % |
| Colour optimisation | 30 ms | 6 % |
| host: HIP init, planner, image loads, mip builds | | ~29 % |

On the 41-view set the host share is larger: ~19 s startup (HIP init + planner probe
object + first batch of 12 MP EXR loads) and ~19 s of camera-switch gaps (loads of the
next batch), i.e. about 30 % of 128 s. The image loads are CPU work (EXR decode + the x2
CPU downscale AliceVision does at load time); the parallel prefetch runs 12 loads at once
but they complete about one per 0.5 s, so something in the load path still serialises.

The GPU part is ~360 ms per tile against an estimated <=250 ms on the 1080 Ti: the kernels
compile at full occupancy (16 waves/SIMD, <=65 VGPRs, no scratch, remark-verified), launch
overhead is 1.7 us, texture format is the same as CUDA's (half4), so the remaining gap is in
kernel throughput that needs a real GPU profiler to explain. That is a Linux job
(`rocprofv3`), which the production node build needs anyway.

## What was done

1. **Fused SGM path aggregation** (`hip/port/sgm_fused/`): the per-row loop launched three
   kernels per volume row (slice copy, aggregation, best-Z reduction with 3 blocks of 64
   threads walking 243 depths); one fused kernel reads the volume directly and produces the
   next row's column minima with a shared-memory pre-reduction and one global atomicMin per
   block and column. Kept behind `#if !defined(TSIM_USE_FLOAT)`; the legacy loop stays for
   float builds. Bit-identical; -3.3 s on mini6, -21 s on full.
2. **half4 textures back** (`miplevel_kernel.cu.txt`, `miplevel_host.cu.txt`): the real
   HIP-Windows defect is `surf2Dwrite` into 16-bit float arrays (stores are dropped;
   `hip/tests/surf_probe.hip`), not texture reads (`texfmt_probe.hip`). Mip levels are now
   computed into device memory and copied with `hipMemcpy2DToArray`. Same texture format as
   the CUDA node, half the VRAM of the float4 workaround. Accuracy vs CUDA unchanged.
3. **Parallel image prefetch** (`prefetch.cpp.txt`, `imagescache_refresh.cpp.txt`): each
   batch's cameras are loaded with an OpenMP loop; `ImagesCache::refreshData` got a slot
   lock so concurrent loads of different cameras are safe. -3 s on full.
4. **Cached symbol addresses** for `cudaMemcpyToSymbol` in the compat layer (harmless
   here: the runtime call itself is ~100 us; the 70 ms per camera seen in logs was the
   mip build around it).
5. Profiling tooling: `CHESHIRE_PROFILE_SGM=1` adds one stream sync per stage (Sgm/Refine)
   and per aggregation pass; `scripts/profile_log.py` turns the log into a table;
   `-Rpass-analysis=kernel-resource-usage` via `CHESHIRE_HIP_EXTRA_FLAGS` gives per-kernel
   registers/occupancy.

## Tried and rejected (measured on mini6)

| experiment | result |
|---|---|
| more HIP hardware queues (`GPU_MAX_HW_QUEUES` 4/8/16) | 17.2 / 17.2 / 17.3 s: no effect; the GPU is already saturated (two processes at once are slower than one) |
| 2048-px tiles instead of 1024 | 75.8 s: 3.4 GB of buffers per tile, 3 tiles in flight; 4x slower |
| block height for occupancy-derived launches (`CHESHIRE_BLOCK_Y` 4/8/16 vs 32) | within noise, 16 slower |
| `-O3` for device code | no change |
| `-ffast-math` | 15.8 s (-11 %) but accuracy drops to 89 % within 1 % of CUDA: rejected |
| `-mcumode` (RDNA CU mode) | 18.6 s: slower |
| `-fgpu-flush-denormals-to-zero` | no change |
| uchar4 camera textures | ran in 8 s but produced empty maps (surface-store defect again); not pursued |
| per-kernel serialized profiling (`AMD_SERIALIZE_KERNEL`) | misleading: ~100 us per forced sync inflates launch-heavy stages 3x |

## Next steps that would move the number

* Profile the similarity / refine kernels with `rocprofv3` on Linux (also the node target).
* Two-level texture-window caching in `volume_computeSimilarity_kernel` (the R-camera
  patch is re-fetched for every depth plane) - an algorithmic change, applies to CUDA too.
* Run the four aggregation passes on separate streams with a float accumulator (results
  would differ by uchar rounding; needs re-validation).
* Chunk-level parallelism in the node's `reconstruct` wrapper is *not* a win here: the GPU
  is saturated by one process.
