# Validation: HIP depth maps vs the CUDA node

## Method
Same photos, same SfM, same DepthMap parameters; only the GPU stage differs.

1. Photos: [alicevision/dataset_monstree](https://github.com/alicevision/dataset_monstree)
   (`mini6` = 6 images, `full` = 41 images), copied to `house-pc:/data/scans/monstree-*`.
2. CUDA reference: Meshroom 2023.3.0 `meshroom_batch` on house-pc (GTX 1080 Ti, CUDA 11.3)
   with `FeatureExtraction:describerTypes=sift FeatureExtraction:forceCpuExtraction=False
   DepthMap:downscale=2` (the node's `standard` preset). The whole cache through
   `DepthMapFilter` is copied to `data/ref/<dataset>/` (tar over ssh).
3. HIP run: `scripts\run-depthmap.cmd <dataset>` feeds the reference `sfm.abc` and
   `PrepareDenseScene` images to the Windows/HIP `aliceVision_depthMapEstimation` with the
   exact command line Meshroom recorded in `DepthMap/*/0.status`, then runs
   `scripts/compare_depthmaps.py` (per-view valid-mask agreement, relative depth error
   distribution, simMap MAD, side-by-side PNGs).

Pass criterion (initial): >= 98 % of jointly-valid pixels within 1 % relative depth and
>= 95 % valid-mask agreement per view. The algorithms are deterministic, so differences
should come only from fast-math / FMA contraction and texture-filter precision.

## Reference numbers

| dataset | node | DepthMap wall time | notes |
|---|---|---|---|
| monstree-mini6 | GTX 1080 Ti, CUDA 11.3 | 31.9 s (6 views, 6 tiles/view, `--rangeSize 12`) | `data/ref/monstree-mini6/DepthMap/820d.../0.status` |
| monstree-mini6 | **RX 9070, HIP (Cheshire)** | 21.2 s first validated build; **17.4 s** after tuning (docs/05) | `build/run-mini6.log` |
| monstree-full (41 views) | GTX 1080 Ti, CUDA 11.3 | **379.0 s** = 4 Meshroom chunks of 12 views (105.4 + 111.1 + 106.3 + 56.2 s); listed as 105.5 s (chunk 0 only) until 2026-09-15 | `data/ref/monstree-full/DepthMap/*/[0-3].log` |
| monstree-full (41 views) | **RX 9070, HIP (Cheshire)** | 154.7 s first validated build; **124.4 s** after tuning (docs/05) | `build/run-full.log` |

## Results

### monstree-mini6, 2026-09-03 — PASS

```
       view valid ref valid tst  agree  meanRel   medRel   p95Rel  <0.5%    <1%    <5%  simMAD
 1136735892     0.972     0.972  1.000   0.0056   0.0000   0.0008  0.985  0.989  0.994  0.2759
 1175796198     0.972     0.972  1.000   0.0019   0.0000   0.0007  0.984  0.988  0.993  0.1261
 1178536846     0.972     0.972  1.000   0.0107   0.0000   0.0012  0.978  0.984  0.990  0.2315
 1302207262     0.972     0.972  1.000   0.0088   0.0000   0.0010  0.980  0.985  0.990  0.2505
 1383319223     0.968     0.968  1.000   0.0039   0.0000   0.0009  0.983  0.987  0.992  0.1410
  677904057     0.972     0.972  1.000   0.0019   0.0000   0.0007  0.986  0.990  0.994  0.1368
RESULT: PASS
```

Valid masks identical, median relative depth error 0, p95 about 0.1 %, 98-99 % of pixels
within 1 %. The residual 1-2 % of pixels off by more than 1 % and the simMap MAD are
consistent with two different AliceVision versions (Meshroom 2023.3 = AliceVision 3.1 on the
node vs the 2026 development tree here) plus FMA/texture-filter differences; a same-version
CUDA reference would tighten this, and the 41-image set will show whether it holds at scale.

What it took to get from "runs" to "PASS": the first HIP run produced all-invalid maps
because **HIP on Windows samples half4 (16-bit float) texture arrays as zeros**
(`hip/tests/half_tex.hip` isolates it; float4 arrays are fine). AliceVision's camera
mipmaps use half4 by default; the HIP build now selects the float4 path (2x camera-image
VRAM). Revisit when ROCm fixes half textures on Windows, or when profiling shows texture
bandwidth matters.

### monstree-full (41 views), 2026-09-03 - strict criterion FAIL, agreement otherwise excellent

Full per-view table, CSV/JSON and downscaled best/median/worst panels:
[docs/validation/monstree-full/index.md](validation/monstree-full/index.md)
(mini6: [docs/validation/monstree-mini6/index.md](validation/monstree-mini6/index.md)).

* mask agreement >= 95 % on 41/41 views (40 of them exactly 1.000; view 1227295871 at 0.971)
* per-view **median** relative depth error is 0.0000 on all 41 views
* fraction within 1 %: median over views 0.988; 34 views >= 0.97; worst view 0.917 (1430763847),
  then 0.925 (528863451) and 0.928 (1317225462); largest per-view p95 = 7.1 %
* wall time 154.7 s on the RX 9070 vs 379.0 s on the 1080 Ti across its four Meshroom chunks (the 105.5 s quoted here until 2026-09-15 was chunk 0 alone; mini6 was 21 s vs 32 s); the
  41-view run is not yet profiled - float4 textures double image bandwidth and the planner's
  tile parallelism has not been looked at

The strict bar (every view >= 98 % within 1 %) is not met on the large set. Looking at the
worst view (`docs/validation/monstree-full/worst_*.png`), the two maps are the same map;
differences are scattered speckle in low-texture regions with no tile or stripe structure,
which is the signature of different AliceVision versions (3.1 on the node vs 2026 dev tree)
and FMA/texture-filter precision, not of a broken kernel. To settle it properly the plan is a
same-version CUDA reference (build this AliceVision tree with CUDA on a NVIDIA box, or run the
CUDA backend of this exact tree on the node) - until then the numbers above are the honest
statement: identical validity masks, zero median error, ~1-8 % of pixels per view differing
by more than 1 %.

### monstree-mini6 on house-pc, RX 5500 XT (Linux), 2026-09-03

Same-machine CUDA reference (1080 Ti, 31.9 s) vs HIP on an RDNA1 card (60.7 s): masks
identical on all views, median error 0, 96-98 % within 1 %. HIP-vs-HIP across GPUs
(RX 5500 XT Linux vs RX 9070 Windows, identical build) shows the same spread, so ~2 % of
pixels differing by >1 % is the cross-GPU noise floor. Details and panels:
[docs/validation/monstree-mini6-rx5500xt-linux/index.md](validation/monstree-mini6-rx5500xt-linux/index.md).

### monstree-full (41 views) on house-pc, RX 5500 XT (Linux), 2026-09-03

447.7 s vs 379.0 s CUDA (four chunks) on the same host; masks identical on 41/41, median error 0, median
view 98.1 % within 1 % (worst 92.3 %). Same envelope as the RX 9070. Details:
[docs/validation/monstree-full-rx5500xt-linux/index.md](validation/monstree-full-rx5500xt-linux/index.md).

### house-pc, RX 5500 XT (Linux), v0.2.3 bundle, 2026-09-16

Same node, the bundle rebuilt with the packed-slot mipmap sampler: 6 views 52.4 s (was 60.7 s),
41 views 396.9 s (was 447.7 s), outputs byte-identical to the earlier Linux runs, same statistics
against CUDA. Page:
[docs/validation/monstree-full-rx5500xt-linux-v0.2.3/index.md](validation/monstree-full-rx5500xt-linux-v0.2.3/index.md).

### house-pc, RX 6750 XT (RDNA2, Linux), 2026-09-03

6 views 31.0 s (CUDA 1080 Ti 31.9 s, same host), 41 views 226.1 s (CUDA 379.0 s in four chunks; the HIP build inside a real four-chunk Meshroom job on this node: 352 s); masks identical
on every view, median error 0, 97.5 % / 98.1 % median-view within 1 %. The RX 6750 XT and the
RX 5500 XT produce identical statistics on both sets (RDNA1 and RDNA2 agree bit-for-bit, see
the compare in this session). Pages: [6 views](validation/monstree-mini6-rx6750xt-linux/index.md),
[41 views](validation/monstree-full-rx6750xt-linux/index.md).

### bench-pc, RX 6750 XT (RDNA2, Windows, HIP SDK 6.2 build), 2026-09-15

The same card under Windows through AMD's HIP 6 runtime (v0.2.2 `gfx1031` package, `/arch:AVX`
build on an FX-8120): 6 views 28.1 s, 41 views 190.3 s in one process (24 simultaneous tiles, no
spills); masks identical on every view, median error 0, 98.7 % / 98.7 % median-view within 1 %.
Against its own Linux / HIP 7.2 output on the 6-view set: identical masks, median 0, 97.1-98.7 %
within 1 %, not bit-identical (different compiler and runtime on the same silicon). With mipmap
emulation forced on, the same package takes 232.3 s on the 41 views and is bit-identical to the
native-mipmap run (41 / 41), which puts it within 3 % of the Linux bundle: the Linux/Windows gap
on this card is the mipmap path, nothing else. The packed-slot sampler that followed (see the
README's mipmap bullet) takes the same package to 208.1 s, again bit-identical. Pages:
[6 views](validation/monstree-mini6-rx6750xt-windows-hip6/index.md),
[41 views](validation/monstree-full-rx6750xt-windows-hip6/index.md).

## v0.2.0 regression (2026-09-04)

The bridge v2 work changed the allocator, the planner and the mip-level storage. Regression
against the v0.1.0 outputs (which this document validates against CUDA):

* 41 views, RX 9070, final native build, default settings: 123.0 s, depth maps bit-identical
  to the v0.1.0 output.
* 6 views: every configuration in the bridge matrices (`docs/validation/bridge-v2/`, 40+ runs
  across the RX 9070 and the RX 6750 XT, VRAM caps down to 500 MB, every class forced to
  system RAM) is bit-identical to the uncapped run.
* The strict per-view criterion (>= 98 % of jointly valid pixels within 1 %) is at the noise
  floor of the RX 6750 XT (97.5 % median view), so the matrix runner reports the number rather
  than PASS/FAIL against CUDA; bit-identity to the validated output is the regression gate.

### Meshroom jobs on the node, four chunks each, 41 photos, `standard` preset (2026-09-11 / 09-15)

The like-for-like comparison with the CUDA reference (which ran the same way):

| GPU | DepthMap chunks | total | textured mesh vertices |
|---|---|---|---|
| GTX 1080 Ti, CUDA (Meshroom 2023.3 as shipped) | 105.4 + 111.1 + 106.3 + 56.2 | 379 s | 1,156,930 |
| RX 6750 XT, HIP (paired into the same Meshroom 2023.3) | 99.2 + 103.7 + 95.0 + 54.0 | 352 s | 1,143,582 |
| RX 5500 XT, HIP (same) | 165.4 + 163.9 + 171.2 + 93.2 | 594 s | 1,136,840 |

Feature extraction ran on the CPU for the HIP jobs (PopSift is CUDA-only), so the SfM inputs
differ slightly from the CUDA job's; the vertex counts are within 2 % of each other.
