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

**The same-version CUDA reference now exists, and it settles this** (2026-09-20, see
`docs/18-cuda-build.md`): this tree built with CUDA 12.9 is **byte-identical** to the Meshroom
2023.3 / CUDA 11.3 reference on all 41 views, depth maps and sim maps alike. So the version
drift contributes **nothing** - AliceVision 3.1 vs 3.4-dev and CUDA 11.3 vs 12.9 are both
exactly zero here. The whole residual is AMD hardware and compiler float behaviour. The
sentence above was a reasonable guess; half of it was wrong.

What it took to get from "runs" to "PASS": the first HIP run produced all-invalid maps
because **HIP on Windows samples half4 (16-bit float) texture arrays as zeros**
(`hip/tests/half_tex.hip` isolates it; float4 arrays are fine). AliceVision's camera
mipmaps use half4 by default; the HIP build selected the float4 path here (2x camera-image
VRAM).

**Superseded the same day** (`docs/05-performance.md`): the real HIP-Windows defect is
`surf2Dwrite` *stores* into 16-bit float arrays, not texture reads. Mip levels are computed
into device memory and copied with `hipMemcpy2DToArray` instead, and every shipped build -
HIP and CUDA - uses half4, the same format as the CUDA reference. `CHESHIRE_TEXTURE_FLOAT4`
survives only as an escape hatch that no build defines; read the paragraph above as history,
not as what ships.

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
against CUDA. The RX 6750 XT, back in the same node later that morning: 28.0 s / 205.7 s (was
31.0 s / 226.1 s), 41 / 41 byte-identical to its 2026-09-03 run. Page:
[docs/validation/monstree-full-rx5500xt-linux-v0.2.3/index.md](validation/monstree-full-rx5500xt-linux-v0.2.3/index.md).

### bench-pc, GTX 1080 Ti (CUDA 11.6, Windows), same host as the RX 6750 XT Windows runs, 2026-09-16

CUDA and HIP on identical host hardware for the first time: 40.0 s / 287.1 s on the 1080 Ti (14
tiles) against 28.1 s / 190.3 s on the RX 6750 XT. And CUDA against CUDA: this run is bit-identical
to the Linux CUDA 11.3 reference on 37 of 41 views, with one view off by 7.2 % of pixels, so the
reference disagrees with itself across builds by as much as HIP disagrees with it. Page:
[docs/validation/monstree-full-gtx1080ti-windows-cuda/index.md](validation/monstree-full-gtx1080ti-windows-cuda/index.md).

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

## Package validation (a different question from the numbers above)

Everything above asks whether the depth maps are right. It does not ask whether the *package* runs,
and those are not the same question: v0.2.17 shipped with GPU SIFT crashing on the first photograph
while its depth maps were bit-identical across two platforms. Three checks answer the second
question, and all three found something the first time they were run.

### Stage gates, 2026-09-20

`scripts/verify_bundle_stages.py` (HIP) and `scripts\windows\verify-cuda-stages.ps1` (CUDA) run
every GPU-bearing stage from the package and require each one to print its own port's line. Neither
had ever been executed.

| | RX 9070, v0.2.18 bundle (`rocm7.2 gfx12-generic`) | GTX 1050 Ti, Windows CUDA package |
|---|---|---|
| FeatureExtraction (GPU SIFT) | ok, 9 s | ok, 3.8 s |
| FeatureMatching (GPU matcher) | ok, 2 s | ok, 11.4 s |
| DepthMap | ok, 12 s | ok, 89.5 s |
| DepthMapFilter (GPU filter) | ok, 1 s | ok, 3.4 s |
| Meshing (GPU votes) | ok, 18 s | ok, 50.4 s |
| Texturing (GPU) | ok, 18 s | ok, 47.6 s |

Five markers were wrong when first run, in two families, and both families produce a *silent* wrong
answer:

* three matched lines the port only emits under a debug variable (`filter votes GPU`,
  `knn check: identical`, `cheshire texturing profile`), so a healthy GPU run reported as a failure;
* two matched the port's own DISABLED message as well as its success one (`GPU brute-force` also
  matches "GPU brute-force disabled by CHESHIRE_GPU_MATCHER=0"; `(?i)popsift|gpu` matches any path
  containing `gpu`, and bench-pc has a `C:\cheshire\gpusift`), so a CPU run reported as a GPU one.

The rule that catches both: take the marker from the success branch of the port's `available()`,
and include enough of it that the disabled branch cannot match.

Both gates also fed DepthMapFilter the *filtered* maps as its input, so the stage ran, produced
output and reported ok while never seeing the input it gets in a pipeline.

### End to end, 2026-09-20

`scripts/verify_end_to_end.py` pairs the package into Meshroom and runs whole graphs, requiring both
a textured mesh and a port line from every paired node - a pipeline that fell back to Meshroom's own
binaries produces a perfectly good mesh, and counting files cannot tell the difference.

Before this, no complete Meshroom graph had ever run on a Cheshire package on Windows, and none
could have: `meshroom-pair.cmd` required `<pkg>\bin\` and rejected the bundle, which has been the
only Windows download since v0.2.17. The reference caches under `build/meshroom` look like
end-to-end runs and are not - only their DepthMap node carries a `[cheshire]` line; every other node
in them ran stock Meshroom.

Six photographs (monstree mini6), all seven nodes paired, 6/6 ports in every row:

| config | what it varies | RX 9070 | GTX 1050 Ti |
|---|---|---|---|
| `base` | defaults | 59 s | 186 s |
| `tiles` | 512 px tile buffers - 24 tiles against 6 | 59 s | 180 s |
| `coarse` | downscale 4, one depth list, `maxPoints=300000` | 41 s | 106 s |
| `texbig` | 8192 atlas at full resolution | 59 s | 179 s |
| `cpufallback` | every `CHESHIRE_GPU_*` switch at 0 | 69 s | 233 s |

The settings demonstrably bit rather than re-running the same work five times: `coarse` produced a
10 MB mesh against 38 MB, `texbig` an 84 MB texture against 27 MB, `tiles` 24 tiles against 6.

`cpufallback` is the one that inverts the test - each port must log its DISABLED line instead - and
it matters more than it looks: a package whose CPU fallback is broken passes every other check here
and fails on the first machine without a supported card. All four ports fell back and the pipeline
still produced a 38 MB mesh and a 26 MB texture, 10 s slower than the GPU run on the RX 9070 at this
scene size.

The Windows CUDA package had to be re-laid-out with `bin\` before it could be paired at all. Per
node it made no difference: FeatureExtraction 4.7 vs 4.8 s, FeatureMatching 2.5 vs 2.5, DepthMap
89.0 vs 87.9, DepthMapFilter 3.5 vs 3.5, Meshing 27.5 vs 27.4, Texturing 31.0 vs 30.7.

### The gate on Linux, 2026-09-20

`verify_end_to_end.py` now runs on both systems - there was no need for a second script, since both
platforms pair the same way and take the same two arguments. Building it found two things.

**The first Linux pass was false.** It reported 6/6 with FeatureExtraction not paired at all.
`meshroom-pair.sh` gates GPU SIFT on `-f "$BUNDLE/lib/libpopsift.so"`, and the Linux CUDA bundle
ships only the versioned soname `libpopsift.so.0.10.0` - the HIP bundle has the plain name - so the
script declined to pair the node. Meshroom's own binary ran instead, and because Meshroom 2023.3's
featureExtraction is itself a CUDA PopSIFT, on an NVIDIA box it printed the very "Choosing device 0"
line the gate was looking for. So v0.3.0's Linux CUDA package never pairs GPU SIFT, and the check
built to catch that class of failure was fooled by it.

The gate now requires PROVENANCE before it looks at any port: both launchers announce the binary
they run, and a node that is not paired is reported apart from a paired node whose port fell silent,
because those mean different things. A port marker was never proof - it says a GPU path ran, not
whose.

**Linux + AMD, first complete runs.** house-pc (i3-4330 + RX 6750 XT, RDNA2), the shipped v0.3.0
Linux HIP bundle, six photographs, all seven nodes paired, 6/6 ports every row:

| config | RX 6750 XT (Linux) |
|---|---|
| `base` | 86 s |
| `tiles` | 85 s |
| `coarse` | 57 s |
| `texbig` | 80 s |
| `cpufallback` | 135 s |

`cpufallback` logged all four disabled lines and still produced a textured mesh; the GPU rows name
the card ("ray marching on AMD Radeon RX 6750 XT"). Before this, the Linux AMD artifact had only
ever been checked stage by stage.

### The HIP rebuild for the VRAM clamp, 2026-09-20

v0.3.0 shipped without the clamp for a VRAM cap set above the card's total memory, because it
landed after the binaries were built. Rebuilding is cheaper than it looks: only five DLLs differ per
GPU target, so `build_targets.py` reuses each family's tree - both rocm7.2 targets took 11.5 minutes
together, and nine hip6.2 targets 31 minutes.

Verified on the RX 9070 by overlaying the rebuilt payload onto the shipped bundle:

* **no regression** - 4/4 depth and sim maps byte-identical to the shipped bundle's own output from
  the same inputs on the same card;
* **the clamp works** - with `CHESHIRE_BRIDGE_VRAM_MB=32000` on a 16304 MB card, v0.3.0 reports
  `vram cap 32000 MB` and plans a 25600 MB budget it cannot have, while the rebuild reports
  `requested vram cap 32000 MB exceeds this device (16304 MB total); clamping to 14537 MB` and
  re-plans from 12 full cameras to 5 plus 2 tiles. First hardware demonstration of that fix on AMD;
  the commit that made it had only measured it on a GTX 1080 Ti.

Three of the nine hip6.2 targets failed the first time with "Access is denied" while linking a
*different* auxiliary executable each - exportAlembic, exportMeshroomMaya, imageSegmentation. That
is a real-time antivirus scan holding a freshly linked exe while ninja replaces it; which one gets
hit is random, and none are GPU payload DLLs. Re-running is the fix.

### v0.3.1: the AMD packages rebuilt for the clamp, 2026-09-20

Both AMD bundles rebuilt with the VRAM-cap clamp; both CUDA packages carried over byte-identical,
having had it already. Every artifact validated as the file a user downloads.

| | Windows AMD zip, RX 9070 | Linux AMD tarball, RX 6750 XT |
|---|---|---|
| stage gate | 6/6 | (Windows-only harness) |
| `base` | 60 s | 86 s |
| `tiles` | 57 s | 86 s |
| `coarse` | 38 s | 57 s |
| `texbig` | 56 s | 80 s |
| `cpufallback` | 68 s | 136 s |

All 6/6 ports, every node proving which binary ran. On the Windows artifact the clamp was run
directly (`clamping to 14537 MB`, 5 cameras + 2 tiles) and its `gfx12-generic` payload is
byte-identical to the verified payload directory.

The first Linux bundle of the day passed every packaging check and saw no GPU: `libamd_comgr.so.3.0.0`
was a symlink to itself (docs/06). The gate reported `ports=0/6`, DepthMap exit 1. The second, from
the fixed script, is the one in the table.

The clamp on the Linux artifact, RX 6750 XT (12272 MB), `CHESHIRE_BRIDGE_VRAM_MB=32000`, one view:
v0.3.0 reports `vram cap 32000 MB` and plans 12 full cameras against a 25600 MB budget; v0.3.1
reports `requested vram cap 32000 MB exceeds this device (12272 MB total); clamping to 11014 MB`
and plans 4 against 8811 MB.

## Capability review: what the gates exercise against what the binaries can do (2026-09-20)

The binaries read 57 `CHESHIRE_*` settings. Before this review the end-to-end gate set five of
them, and the phrase "every GPU port switched off" described a run in which three of seven were
still on. What the inventory found, in order of consequence:

* **Three ports are silent on success.** Max-flow prints only under its verbose flag; the sim blur
  prints only when it is *not* used; the visibility knn's `GPU knn index: N points` is behind
  `CHESHIRE_GPU_VIS_LOG` / `_CHECK`. None can be shown to have run on a default run, which is the
  situation the marker rule exists to forbid. They *do* run - the knn's unconditional fallback
  warning is absent, and under `verify` its self-check reports identical answers on 8,770,375
  queries - but proving it needs an unconditional announce on both paths in each, which is a source
  change, a rebuild of every payload and the Linux bundle, and a full re-gate. Scheduled for 0.3.2.
* **The in-process self-checks were never on in any gate.** `CHESHIRE_FILTER_CHECK`,
  `MAXFLOW_CHECK`, `GPU_VIS_CHECK`, `SEGMENT_CHECK`, `GPU_TEDGE_CHECK` and the facet-weight
  comparison under `GPU_VOTE_LOG` each run the CPU reference alongside the port and say so - the
  strongest correctness tests the project has. The gate's `verify` config now turns all of them on
  and requires each verdict in the port's own words. Max-flow's verdict is the *labelling*
  ("cells labelled differently: 0 of N"); its float flow totals are never equal and are documented
  as such, and the first version of the assertion required them equal and failed a perfect run.
* **End-to-end runs are not byte-reproducible, and the reason is benign.** GPU SIFT finds the
  identical keypoint set run to run (sorted, the `.feat` files differ in 0 lines; unsorted, in
  24,500) in a different order; SfM consumes them in order, so everything downstream differs. The
  first version of the matrix asserted byte-identical depth maps across configs with the same
  DepthMap parameters and failed every pair for this reason. Byte identity across bridge caps -
  the bridge's standing criterion - is asserted in the stage gate on a fixed SfM instead:
  uncapped, 1500 MB, 500 MB and bridge-off must produce the same bytes. A stable sort after
  extraction would make whole pipelines reproducible; a candidate for 0.3.2.
* **Planner v2 absorbs a 1.5 GB cap on the six-view set with no spill** - budget 1200 MB, 0 full
  cameras + 2 tiles - exactly as docs/02 says. The matrix's first bridge config asserted "must
  spill" at 1.5 GB, a planner-v1 fact three versions stale. 500 MB is below one tile and spills.
* Eleven settings a user might reach for were undocumented, `CHESHIRE_BRIDGE_HOST_MB` among them.
  USING.md now lists them.

The matrix grew from five configs to nine (`verify`, `bridgecap`, `bridgespill`, `bridgeoff`), the
fallback run switches all seven ports off and says which two it cannot prove, and the stage gate
gained the fixed-SfM bridge pass.

Run on the v0.3.1 Windows bundle, RX 9070, 2026-09-20, after the corrections above:

* stage gate 10/10 - six stages plus the fixed-SfM bridge pass, where uncapped, 1500 MB, 500 MB
  and bridge-off produced byte-identical depth and sim maps (digest `7c9f1102f0141e58`);
* end to end 9/9: `base` 59 s, `tiles` 57, `coarse` 38, `texbig` 56, `cpufallback` 76 (all seven
  switches off, the two unprovable ones labelled), `verify` 80 (all six self-checks satisfied:
  max-flow labelling 0 of 1,676,527 cells different, knn identical on 8,770,375 queries),
  `bridgecap` 91 (planner absorbed 1.5 GB), `bridgespill` 122 (spilled at 500 MB), `bridgeoff` 57.

## Skull turntable: a failure that was not ours (2026-09-20)

Seventy-five 4272x2848 photographs of a skull on a turntable, a set no Cheshire build had seen.
Two configs on the v0.3.1 Windows bundle, RX 9070: `verify` passed 6/6 (623 s, 73 poses, 66,294
landmarks, every self-check satisfied); `base` failed at StructureFromMotion, exit 1, with a Ceres
`CHECK` - `Manifold::PlusJacobian computation failed for x: 0 0 0 0 ...` - a pose whose rotation is
all zeros reaching bundle adjustment after the 68th resection.

What settled the attribution, in order:

* **The node is not a paired one.** The launcher pairs seven binaries; incrementalSfM is not among
  them, and the node's log has no `[cheshire]` provenance line. The binary that crashed is
  Meshroom 2023.3.0's own `aliceVision_incrementalSfM.exe` (145,920 bytes, dated 2023-12-07).
* **Replaying its exact command line on the same features and matches reproduces the CHECK** with
  that stock binary (this time after the 55th resection - RANSAC inside SfM is not seeded either),
  so the failure is a property of those inputs, not of a flaky run.
* **What differs between the passing and failing inputs is keypoint order only.** Same 1,083,238
  features, same 74 candidate pairs, match files four lines apart; GPU SIFT orders its keypoints
  differently each run (docs/04, capability review), which changed the automatic initial pair:
  `822551637, 1010062824` crashes, `861327282, 1489813614` reconstructs. A second `base` run drew
  the second pair and went through.

So the finding is an upstream fragility of incremental SfM on turntable data (a static background
rotating against the object gives inconsistent geometry; the usual answer is masking or a
turntable-aware pipeline), exposed roughly one run in two by keypoint order. Meshroom's own
PopSIFT has the same order nondeterminism, so stock users see the same rate. Two consequences for
this project: the end-to-end report now names the failed node and whether it was a paired binary,
so this is read off the summary rather than dug out of a log; and the stable keypoint sort already
listed for 0.3.2 would make which initial pair a dataset draws a fixed fact rather than a coin toss.

## The strain run that tested nothing (2026-09-20)

The GTX 1050 Ti's full-resolution engine-bay run was two hours in, 24 depth maps written at 480 s
a chunk, when the DepthMap log's first line was read: `[cheshire] aliceVision_depthMapEstimation:
Meshroom's own binary`. Every node before it had done the same. The launcher beside the package on
bench-pc (sha `3ce7c1e6`, built 2026-09-19 16:56) predates `CHESHIRE_BACKEND` (f50f8df, 19:04 the
next day): it read only `CHESHIRE_DEPTHMAP`, saw an NVIDIA card, and handed every node back to
Meshroom, ignoring the `cheshire` the harness set. And the shipped v0.3.0 CUDA Windows zip has no
launcher or pairing script at all - the CUDA packager never staged them - so a user following the
README could not have paired it either.

Fixes: `package-cuda.ps1` stages both files and checks the launcher by substance (the wide-string
`CHESHIRE_BACKEND` must be present; the old launcher is refused, verified); the 0.3.1 CUDA zip is
repacked, 484 entries byte-identical plus the two files. The run was restarted with the current
launcher, all seven nodes announcing the Cheshire build. (The first relaunch was found dead a
few minutes later and blamed on the ssh session ending; the event log later showed bench-pc had
bugchecked at 23:38 - see the next section - so that diagnosis was wrong and is withdrawn.)

Two lessons for the gates. The end-to-end harness *would* have caught this at the end of the run,
which is the wrong end of a two-hour job: it should check the first paired node's provenance line
as soon as that node's log exists and abort. And a file's name and date are not its version - the
launcher was checked by the pairing script's "exists" test and nothing else.

## Engine bay at full resolution, RX 6750 XT 12 GB, Linux 0.3.1 bundle (2026-09-21)

107 photographs, `DepthMap:downscale=1`, the whole graph through Meshroom, both configs passing
6/6 ports with every node announcing the Cheshire build:

| config | total | SfM | Meshing | Texturing | depth maps | spills |
|---|---|---|---|---|---|---|
| `ds1` | 2863 s | 140 s | 310 s | 176 s | 107, 9 chunks | 0 in every chunk |
| `blast` (every opt-in path + profile logs) | 2919 s | 152 s | 434 s | 175 s | 107, 9 chunks | 0 |

The bridge on a 12 GB card at full resolution: cap 11014 MB, budget 8811 MB, 1 full R camera +
7 tiles, then 15 tiles per view - it never spills; 12 GB is simply enough. Peak VRAM by stage,
from the allocator summaries: depth map filter 593 MB, meshing 3.4 GB, **texturing 9.8 GB** -
the hungriest stage at full resolution is texturing, not DepthMap, which is what an 8 GB card
will meet first. `blast` is not faster than `ds1`: its extra 124 s in Meshing is the facet-weight
comparison `CHESHIRE_GPU_VOTE_LOG` runs alongside the port, and the two opt-ins it turns on
(`CHESHIRE_QR_NULLSPACE`, `CHESHIRE_FILTER_CACHE_MB`) announce nothing in any log - two more for
the 0.3.2 silent-ports list. Profile lines that did appear: meshing votes 19.8 M rays / 25.6 M
cells, vote kernels 6.2 s + tedge 5.2 s; knn index 21.3 M points in 7.5 s; texturing image loads
25.9 s of a stage dominated by I/O. Depth-map digests differ between the two runs, as they must
(GPU SIFT order → different SfM).

## bench-pc bugchecks under the full-resolution run (2026-09-21)

The paired 1050 Ti run stopped at 00:30:32, mid-chunk, 48 of 107 depth maps done, 0 spills in the
three finished chunks, with no error in any log and no process left: bench-pc had rebooted from
**bugcheck 0x1A (MEMORY_MANAGEMENT, subcode 0x403)** at 00:31. The System log has the same
bugcheck at 17:59 (the run that "died at 17:58" earlier that day, then attributed to an ssh
session) and at 23:38 (the first relaunch, attributed to `Start-Process`), plus 0x1A/0x411 on
09-16 - four kernel memory-management crashes in five days, all under heavy runs, on a box with
mixed DIMMs (2 x 4 GB 800 MHz + 2 x 4 GB 667 MHz). The minidumps name no third-party driver.
User-mode code cannot bugcheck a machine; this is RAM or a driver (NVIDIA 581.57), and a memory
test is the next step. Every earlier "the run died" in this document was this.

The run was resumed rather than restarted: the harness gained `CHESHIRE_E2E_RESUME=1`, which
keeps the cache so Meshroom skips the SUCCESS chunks (three of nine) and redoes the one it was
killed in.

## Engine bay at full resolution on 4 GB: GTX 1050 Ti, Windows CUDA package (2026-09-21)

The resumed run passed - `ds1` 6/6 ports, mesh + 6 texture atlases, every node announcing the
Cheshire build, no further bugcheck (uptime held from 00:31). Chunk times, sum over chunks:
DepthMap 4837 s (9 chunks, ~9 min per 12 views), DepthMapFilter 196 s, Meshing 602 s, Texturing
593 s; SfM 280 s. Wall clock 4695 s from the resume, with three DepthMap chunks carried over.

What 4 GB did to each stage:

* **DepthMap: 0 spills in all 9 chunks.** The bridge capped at 3021 MB and budgeted 2417 MB - 0
  full R cameras + 2 tiles resident, 15 tiles per view - and that fits without ever spilling. So
  the "spill for sure" premise of this run was wrong for the CUDA path at this resolution: the
  planner's tile-only regime is enough, and the bridge's host-spill path stayed untested here.
  Spilling needs a cap below one tile (the stage gate's 500 MB) or a larger image.
* **Meshing** ran the CPU-side fusion (kd-trees of 34.5 M, 18 M, 10 M, 7.8 M points) and the GPU
  ports within 4 GB; no fallback line, no error.
* **Texturing** ran `pyramid + rasterisation on NVIDIA GeForce GTX 1050 Ti` with an atlas pyramid
  of 3072 MB and `memoryPerAtlas: 3488` - one atlas at a time inside the card's 4096 MB, where the
  RX 6750 XT's run peaked at 9.8 GB by holding more at once. The CUDA build prints no allocator
  summary (`vram: N allocs, peak M MB` is HIP-side), so the CUDA peaks are not measured here.

This is the strongest package result so far: the smallest card the project has, at the largest
per-view working set, through Meshroom end to end on the download as shipped (plus the pairing
files 0.3.1 adds).

**Postscript, 08:22 the same morning:** the `blast` config on the same card died the same way -
bugcheck 0x1A (0x403) with 44 of 107 depth maps done, 0 spills in the three finished chunks, no
error in any log - the fifth in six days. bench-pc is retired from hours-long runs until its
memory is tested; the `ds1` result above stands (it completed, and its every stage is in its
logs), and `blast` on 4 GB is recorded as not run.


## 0.3.2 - the queue (2026-09-21)

Everything below is a source change, so it rides one rebuild of all eleven Windows payloads, the
Linux bundle and both CUDA packages, followed by the full set of gates. Collected here because
the items were scattered through the sections above.

1. **Unconditional announce lines for the silent ports** - max-flow, sim blur, visibility knn,
   QR nullspace, filter cache. Each must print one line on the GPU path and one on the fallback,
   long enough that the gate's marker rule can tell them apart. Without it the gate cannot prove
   five of the ports ran (capability review above).
2. **Stable keypoint sort after GPU SIFT.** Same keypoint set every run, different order; a sort
   makes SfM, and so whole pipelines, byte-reproducible, and turns the skull's initial-pair coin
   toss into a fixed draw.
3. **Texturing: padding on the GPU and a direct textured-OBJ writer.** On the engine bay (RX 6750
   XT, 2026-09-21) the ported stage is about 6 s of a 175 s node: uploads 2.1 + pyramids 0.6 +
   rasterisation 2.6 + finish 0.6. The rest is upstream CPU: per-atlas edge padding + downscale +
   EXR write ~10 s x 6 atlases, UV generation ~53 s, mesh load and textured-mesh save through
   Assimp ~20-25 s, image loads 26 s (I/O). Padding is a dilate on an atlas that is already in VRAM
   at rasterisation time - do it before the download, not after. The save is patch step 4n again
   (`Mesh::save`'s direct writer, docs/11) with `vt` / `f v/vt` lines and the MTL; the Assimp load
   stays, since the node accepts non-OBJ input. `Texturing::loadWithAtlas` and `saveAs` have no
   Cheshire code today.
4. **CUDA allocator peak summary.** `vram: N allocs, peak M MB` is HIP-side only; the 1050 Ti run
   reported no peaks. The CUDA build should print the same line so both backends' memory is
   measured the same way.
5. **Camera mipmaps through the bridge on CUDA.** They bypass it today, so on a small NVIDIA card
   there is memory the planner cannot see; the 1050 Ti absorbed the engine bay, a larger set may
   not.
6. **Announce lines in the end-to-end gate for items 1 and 4**, and the 8 GB HIP texturing run
   (RX 5500 XT) recorded before the release, whatever it shows.

Not source, and not waiting for 0.3.2: memtest on bench-pc; the RX 5500 XT gate on the Windows
hip6.2 gfx1012/gfx1031 payloads (closes the one inference in 0.3.1's notes).

## 0.3.3 - the CPU nodes, ranked by the clock (2026-09-21)

Per-node wall clock of the engine bay at full resolution on the RX 6750 XT (`ds1`, 2863 s), from
Meshroom's status timestamps:

| node | wall | share | today |
|---|---|---|---|
| DepthMap | 1612 s | 56 % | ported |
| Meshing | 310 s | 11 % | ported; CPU fusion remains |
| FeatureMatching | 255 s | 9 % | GPU 2-NN ported; the rest is geometric filtering |
| Texturing | 176 s | 6 % | ported; padding, atlas writes and Assimp are 0.3.2 item 3 |
| DepthMapFilter | 168 s | 6 % | ported |
| StructureFromMotion | 140 s | 4.9 % | stock, not a paired node |
| PrepareDenseScene | 133 s | 4.6 % | stock: image conversion to EXR |
| MeshFiltering | 37 s | 1.3 % | stock |
| FeatureExtraction | 31 s | 1 % | GPU SIFT |
| ImageMatching | ~0 s | - | stock |

The order for 0.3.3, by what the numbers say rather than by which node sounds most algorithmic:

1. **Geometric filtering in FeatureMatching.** Already Cheshire code, already deterministic, and
   the byte-identity gate exists; `CHESHIRE_QR_NULLSPACE` showed 1.86x on this phase at a measured
   cost (docs/17). The 7-point solver's remaining time is the same kind of allocation and
   vectorisation work that carried the AC-RANSAC and meshing CPU wins.
2. **PrepareDenseScene.** 133 s of reading photographs and writing EXR, embarrassingly parallel,
   no algorithm in it and nothing to change in the output: a throughput problem.
3. **StructureFromMotion, profiled first.** The split between Ceres and resectioning is not known;
   our vcpkg Ceres already links CHOLMOD, SPQR, METIS and OpenBLAS, so "a better sparse backend" is
   not a lever - the profile decides between Ceres assembly overhead and the PnP/RANSAC loops.
   Two costs specific to this node: it would be the eighth paired binary on both platforms, and
   nothing counts until the launcher carries it. One thing 0.3.2 gives for free: with keypoints in
   stable order SfM becomes deterministic, so a resectioning optimisation can be gated on a
   byte-identical `sfm.abc`. Global SfM (rotation + translation averaging) is a different product -
   it changes the geometry - and stays out.

Not targets: MeshFiltering (37 s, not the 2-5 s a generic estimate gives, but 1.3 %) and
ImageMatching (below the resolution of the timestamps on 107 photographs; only a 1000-photo
scan would move it).

## 0.3.2 progress: items 1 and 2 landed (2026-09-21)

Built for gfx1201 only (the release rebuild of all payloads comes at the end), packaged flat, gated
on the RX 9070 with mini6: `base` 65 s, `cpufallback` 90 s, `blast` 181 s, 3/3, 6/6 ports each.

**Item 1, announce lines.** Every port now prints on both paths, and the logs show it:
`max-flow: GPU push-relabel on AMD Radeon RX 9070` / `disabled by CHESHIRE_GPU_MAXFLOW=0,
Boykov-Kolmogorov`; `sim blur: Gaussian on the GPU (...)` / `disabled by CHESHIRE_GPU_BLUR=0,
OIIO`; `visibility knn on the GPU` / `disabled by CHESHIRE_GPU_VIS=0, nanoflann`; `7-point
nullspace: SVD (default...)` in `base` and `Householder QR (CHESHIRE_QR_NULLSPACE=1)` in `blast`;
`depth map filter cache: cap 4096 MB` / `cap 8192 MB`. The gate requires all of them, and the
fallback run now proves the max-flow and visibility CPU paths instead of noting that it cannot.

**Item 2, keypoint order.** Generator step 5c sorts PopSIFT's keypoints by (x, y, scale,
orientation, descriptor bytes). Two `base` runs on the same package: **12 of 12 feature and
descriptor files byte-identical, `0.matches.txt` byte-identical** - extraction and matching are now
a function of the images. SfM is not: same 6 poses and 9433 landmarks both times, but
`cameras.sfm` differs from the fourth significant digit (focal 3.98899 vs 3.98876), which is
upstream incremental SfM's own run-to-run variation (unseeded RANSAC, threaded bundle adjustment)
and was there before Cheshire. The skull's initial-pair coin toss is settled by this - pair choice
reads the matches - while byte-identical pipelines end at SfM. Seeding SfM is a 0.3.3 question.

**One bug found by its absence.** The sort's announce line did not print, and the DLL did not
contain the literal though it contained the `getenv` string beside it. `ALICEVISION_LOG_INFO(a)`
expands to `stream << a` without parentheses, so `stream << on ? "A" : "B"` parsed as
`(stream << on) ? "A" : "B"`: it logged the bool and both strings were dead. Any ternary handed to
that macro does this; the fix is an if/else, and the check that caught it - search the binary for
the literal - is now the habit for every new line.

## 0.3.2 progress: item 3 landed - texturing padding on the GPU, direct textured-OBJ writer (2026-09-21)

**Padding.** `writeTexture`'s "dilate gutter" is two sequential sweeps over the atlas in which a
texel reads neighbours already updated in the same sweep. That dependency is only on the left and
up neighbour (forward) or right and down (backward), so every anti-diagonal is independent once
the previous one is done: the port runs the same sweeps as one kernel launch per diagonal, in
place, on the device, before the atlas is downloaded - 16,000 launches of up to 8,190 threads for
an 8192^2 atlas, 0.36 s including the download. `CHESHIRE_GPU_PAD_CHECK=1` runs upstream's loops
on the host copy and compares: **0 of 67,108,864 texels differ**, colours and count values both.
The CPU fallback path still pads on the host; `writeTexture` logs which one ran.

**The writer.** `Texturing::saveAs` built an Assimp scene (every (vertex, uv) pair through a
`std::map`, positions copied into `aiMesh` arrays) and exported through Assimp's OBJ writer. The
direct writer emits every mesh vertex, every uv, then per atlas `usemtl` and `f v/vt v/vt v/vt`,
and the MTL; OBJ only, and only without normal/bump/displacement maps. Content is asserted, not
assumed: `CHESHIRE_OBJ_CHECK=1` writes Assimp's file beside it and `scripts/check_textured_obj.py`
resolves every face of both to its (position, uv) corners and compares the multisets per material.
First run: 20,409 of 499,080 faces differed by one float ulp in a uv - the mesh holds doubles,
Assimp stores floats, and printing a double to 9 digits then reading it back as float
double-rounds differently from casting first. The writer casts to float before printing, as
Assimp's numbers are; second run **0 faces differ either way** (249,933 v / 269,826 vt / 499,517
faces, Assimp 269,934 v by its own deduplication). Save 0.4 s direct against 2.8 s Assimp on that
mesh; the engine bay's 14 s should scale the same way.

Texturing on mini6: 11.7 s to 10.8 s. What the timeline shows is left per atlas: the Lanczos
downscale (`imageAlgo::resizeImage`, 4.1 s at 8192 to 4096), the EXR write 0.5 s. The downscale
is the next texturing target and is not part of this item.

Gate: `base` and `texcheck` both 6/6; `texcheck` asserts the padding verdict, the "done on the
GPU" line, the writer's line and the OBJ content comparison.

## 0.3.2 progress: items 4 and 5 landed - the bridge sees everything (2026-09-21)

**Item 4.** The seven GPU ports allocated with plain `cudaMalloc`, which on HIP the force-included
`cuda_to_hip.h` turned into bridge calls and on CUDA reached the driver directly: the CUDA
summaries had no `other` line and a CUDA card's VRAM had a population the planner could not see.
The ports now call `cheshire::devMalloc` / `devFree` (`cheshire/devalloc.h`, the bridge on either
backend); 53 call sites, none left.

**Item 5.** The camera mipmaps were never counted on CUDA - `cudaMallocMipmappedArray` is driver
memory - and, it turned out, not on the RX 9070 either: the Windows HIP build passes
`-DCHESHIRE_NATIVE_MIPMAP`, so its mipmaps are `hipMallocMipmappedArray`, equally invisible. The
9070's summaries had never had an `image` line. The bridge gained `noteExternal` / `forgetExternal`
(count bytes it does not own; free() never sees the keys), the depth-map code notes every array
after creation and forgets it before freeing, guarded on `CHESHIRE_EMULATE_MIPMAP` so the Linux
RDNA1 emulation, which allocates its levels through the bridge, is untouched; the emulation's
array-storage mode counts its levels too. The estimate sums every level at the build's texel size
and is generous next to upstream's own "single mipmap image size" figure (62 MB against 35 MB per
image on mini6); it errs on the side of a smaller budget, which is the side to err on.

Gate on the rebuilt gfx1201, RX 9070, mini6: `base` 65 s, `cpufallback` 80 s, `blast` 160 s,
`texcheck` 65 s, 4/4, 6/6 ports each. Under `blast` the summaries now read: DepthMap `map` 1179 MB
/ `volume` 7485 MB / **`image` 372 MB (6 allocs)**; DepthMapFilter `other` 1186 MB; Meshing `other`
853 MB; Texturing `other` 3847 MB; FeatureMatching `other` 6 MB. The `blast` config asserts the
`image` and `other` lines from here on, so a build that loses either fails the gate. The CUDA side
of both items is proven on bench-pc once the CUDA package is rebuilt.

One bug on the way: upstream's `if (_mipmappedArray != nullptr)` has no braces, so the first
inserted `forgetExternal` became the if-body and the free ran unconditionally on a null handle -
"invalid argument" from every destructor. Braced now; a patch that inserts a statement after an
unbraced `if` is a pattern to look for.

## 0.3.2 Linux HIP bundle on the RX 6750 XT: 11 of 12, and the twelfth explained (2026-09-21)

The bundle packed from the WSL build (123 MB, `ad5e5984`, every pack check green) ran the full
matrix on house-pc: `base` 85 s, `tiles` 85, `coarse` 60, `texbig` 80, `cpufallback` 155,
`bridgecap` 90, `bridgespill` 180, `bridgeoff` 90, `texcheck` 90 (padding 0 of 67,108,864 texels
differ; direct OBJ written), `ds1` 200, `blast` 225 - all 6/6 ports - and under `blast` the
bridge classes read DepthMap `map` 825 / `volume` 4990 / **`image` 744 MB (48 allocs: the
emulation's six images x eight levels)**, filter `other` 790, meshing `other` 846, texturing
`other` 3846 MB. Items 1-5 hold on Linux.

`verify` failed on one assertion: the knn self-check printed `0 of 8,801,169 queries name a
different vertex, 1,639,701 a different distance` where the gate demanded "identical to nanoflann
on all". The 0.3.1 bundle on the same card, run for the comparison, prints the same line (0 of
8,687,477 vertices, 1,621,566 distances) - this predates 0.3.2 and had never been seen because the
Linux matrix of 0.3.1 did not include `verify`. The RX 9070 reports "identical". The vertex is the
verdict that matters (visibilities are per vertex; the distance is an intermediate the GPU rounds
differently from nanoflann on this card), so the gate accepts either wording, and the difference
itself is recorded here as a card-dependent rounding, not a defect.

## 0.3.2 release gates (2026-09-21)

Every artifact gated as the file a user downloads, all five items in, source 8f48532 + gate fixes.

| artifact | hardware | stage gate | end to end |
|---|---|---|---|
| `cheshire-alicevision-windows-x64.zip` (AMD, 11 payloads) | RX 9070, Windows | 10/10, depth maps byte-identical across uncapped / 1500 MB / 500 MB / bridge-off (`42abe20a879c546f`) | 12/12 |
| `cheshire-alicevision-hip-linux-x64-rocm7.2.tar.gz` | RX 6750 XT, Linux | - | 12/12 (knn verdict by vertex) |
| `cheshire-alicevision-cuda-linux-x64-cuda12.9.tar.gz` | GTX 1050 Ti 4 GB, Linux | - | **12/12** |
| `cheshire-alicevision-cuda-windows-x64-cuda12.9.zip` | (no Windows NVIDIA box: bench-pc's RAM) | literals checked in the DLLs | not run on hardware |

**The CUDA matrix is the proof of items 4 and 5 on NVIDIA.** Under `blast` on the 1050 Ti the
bridge summary reads DepthMap `map` 163 / `volume` 874 / **`image` 372 MB (6 allocs)** - the
camera mipmaps counted for the first time on CUDA - and `other` for the filter (790 MB), meshing
(846), texturing (3506) and matcher (6), where 0.3.1's CUDA logs had no such lines at all. The
planner on 4 GB budgets 2860 MB (cap 3574) as two resident tiles, `bridgespill` spilled at 500 MB,
`verify` satisfied all six self-checks, `texcheck` padded 0 of 67,108,864 texels differently and
wrote the direct OBJ. `ds1` 430 s and `blast` 455 s on mini6 are the card, not the code.

**Gate corrections found by the run, all committed:** the stage gate's FeatureMatching asked for
`dspsift` regions since it was written while every mini6 cache and its own extraction stage are
`sift` (it had passed 10/10 for 0.3.1 on a cache since replaced); the reference cache's image
paths are JSON-escaped Linux paths, so a local copy with them rewritten (`build/stage-cache-mini6`)
is what the gate consumes on Windows; the knn self-check verdict is the vertex, not the distance.

**Not done:** the Windows CUDA zip has not run on NVIDIA hardware under Windows. Its binaries come
from the same source as the Linux CUDA bundle that passed 12/12, its package layout and pairing
files are the ones 0.3.1 validated, and the new literals are present in its DLLs; that is
inference, stated here. bench-pc (the only Windows NVIDIA box) bugchecks under load until its RAM
is sorted.

## 0.3.3 item 2: PrepareDenseScene, profiled and the undistortion mapped (2026-09-21)

`CHESHIRE_PDS_PROFILE=1` splits the node into read (JPEG decode + sRGB→linear), exposure + mask,
undistort and EXR write, summed over threads. Engine bay, 107 views (4032x2268), RX 9070 box (6
cores / 12 threads), 34.5 s wall: **read 134, undistort 153, write 80 thread-seconds**, exposure
0. Threads were all busy; the lever is the work.

The undistortion evaluated the camera model per output pixel for every view - `ima2cam`, the
radial polynomial, `cam2ima` - while all 107 views share one intrinsic. Generator step 5h computes
the distorted source coordinate once per (intrinsic parameters, output size, principal-point
correction) with exactly the expression the loop used, keeps it (139 MB for this size, capped by
`CHESHIRE_UNDISTORT_MAP_MB`, default 2048; `CHESHIRE_UNDISTORT_MAP=0` disables), and the per-view
work is the bilinear sample alone. **107 of 107 EXRs byte-identical** to the reference run;
undistort 153 → 79 thread-seconds.

Wall clock on this box: unchanged, 34.1 s against 34.5. Where the saved 75 thread-seconds went:

| threads | map | wall | read | undistort | write | sum |
|---|---|---|---|---|---|---|
| 12 | on | 33.9 s | 171 | 80 | 108 | 359 |
| 12 | on, output on the other SSD | 33.8 s | 177 | 87 | 101 | 365 |
| 6 | on | 35.2 s | 89 | 56 | 55 | 200 |
| 6 | off | 36.6 s | 66 | 106 | 37 | 209 |
| 3 | on | 48.7 s | 43 | 32 | 66 | 141 |

Six threads run as fast as twelve, and which SSD the output goes to makes no difference, so the
node is not disk-bound and the SMT threads add nothing: it is bound by the six cores and the memory
they share. With the map, the undistortion turns from arithmetic into a gather - 16 bytes of map
plus a bilinear read of the source per pixel - and the read and write phases of the other views
slow by about what the undistortion gained, which is the signature of memory bandwidth, not CPU.
On this machine PrepareDenseScene is a memory-bound 34 s and the map moves work, not time. On a
CPU-bound box - house-pc's four slow threads at 133 s - the saved CPU should be wall time, and the
next Linux build carries the change to measure it. What remains here is the read phase (JPEG
decode and the OCIO colour conversion, now 47 % of thread time), which is upstream OIIO/OCIO work.

## 0.3.3 item 3: SfM profiled (2026-09-21)

`CHESHIRE_BA_PROFILE=1` prints Ceres' own timers per solve. Engine bay, 107 views, dspsift, 12
threads, RX 9070 box, SfM wall 92 s: **140 bundle-adjustment solves, 1095 iterations, 50.9 s** =
Jacobians 25.1 s (49 %) + linear solver 13.1 s (26 %) + other 11.0 s (22 %: per-solve problem
construction and trust-region bookkeeping) + residuals 1.7 s. Solvers used: SPARSE_SCHUR
(SuiteSparse), DENSE_SCHUR for the small local problems, DENSE_QR for the tiny ones; 12 threads
throughout. The largest solves are the global ones: 9.0 s for 15 iterations over 325 k residual
blocks, 7.2 s for 8 over 415 k. Final scene 141,077 landmarks.

So the lever inside SfM is the Jacobian evaluation - Ceres autodiff over every observation's
reprojection functor - not the Schur solve. Analytic Jacobians for the pinhole + radial models
(AliceVision's intrinsics already expose `getDerivative*WrtParams`) would cut that phase by
perhaps 2-3x, about 17 s of the 92, at the cost of iterates that differ in rounding from
autodiff's; SfM is already non-reproducible run to run, so that is a change within the existing
band, not a new one. Worth doing, but it is an 18 % win on a 5 % node: below the texturing
Lanczos downscale (25 s of a 175 s node on the 6750 XT, portable exactly, docs/04 item 3 of 0.3.2)
in the queue. Recorded, not started.

## 0.3.3: the texture downscale on the GPU, exact against OpenImageIO (2026-09-21)

After padding moved to the device (0.3.2), the largest piece left in Texturing was
`imageAlgo::resizeImage` on the finished atlas: 4.1 s per 8192^2 atlas on the RX 9070 box. That
is `ImageBufAlgo::resize` with an empty filter name, and OIIO 3.0.9 - the version in the
dependency tree - picks **lanczos3, width 6** for downsizing and runs its separable path (both
confirmed in that release's source). The tap weights depend on the destination column or row
alone, so they are computed on the host with OIIO's own expressions - `FilterLanczos3_1D::lanczos3`
verbatim, the C runtime's `sinf`, contraction off - and uploaded; the device does the inner loop in
OIIO's order (rows outer, taps inner, `w = wy * xfilt[i]`, zero weights skipped, clamped reads,
float sums). `finish()` now returns the downscaled atlas beside the full one, and `writeTexture`
uses it. `CHESHIRE_GPU_RESIZE_CHECK=1` runs OIIO on the host as well: **0 of 50,331,648 texel
channels differ** (4096^2 x 3). `CHESHIRE_GPU_RESIZE=0` keeps the host path.

mini6, RX 9070: Texturing 9.1 s -> 7.1 s; `texcheck` (padding, resize, OBJ) and `base` 2/2, 6/6.
On the 6750 XT's six engine-bay atlases this is the ~25 s the 0.3.2 notes pointed at, to be
measured on the next Linux build. What remains in the node: image loads (I/O), UV generation,
the EXR write, and Assimp's load on the way in.

## The shipped 0.3.2 Windows bundle on the RX 5500 XT (hip6.2/gfx1012), matched DIMMs (2026-09-21)

bench-pc came back with a matched 2 x 8 GB pair in place of the mixed 667/800 MHz set that had
bugchecked five times in six days. The zip published as v0.3.2 ran the full matrix on the RX 5500
XT: **12 of 12**, 6/6 ports each - `base` 166 s, `tiles` 151, `coarse` 106, `texbig` 146,
`cpufallback` 211, `verify` 211 (all six self-checks), `bridgecap` 196, `bridgespill` 422, `bridgeoff`
156, `texcheck` 176 (padding 0 of 67 M, OBJ identical), `ds1` 382, `blast` 422 - with **no bugcheck
across the 55 minutes**, including the memory-heaviest configs the old RAM never survived. The
payload picked was `hip6.2/gfx1012`; the bridge capped at 7232 MB and budgeted 5786 MB as 14
tiles per view; and under `blast` the summary carries `image vram: 6 allocs, peak 372 MB` - the
camera mipmaps counted on Windows HIP as well, the RDNA1 half of 0.3.2's item 5.

This closes the one inference both the v0.3.1 and v0.3.2 notes had to state: the rebuilt Windows
hip6.2 payloads for RDNA1 now have a hardware run behind them. What remains unrun on hardware is
the Windows CUDA zip (no Windows NVIDIA machine; the 1050 Ti went into house-pc for the Linux CUDA
matrix and out again).

## The shipped 0.3.2 Windows CUDA zip on the GTX 1080 Ti: no GPU SIFT (2026-09-21)

The one artifact the v0.3.2 notes could not vouch for has now run on Windows NVIDIA hardware: the
1080 Ti went into bench-pc (driver 581.57, matched DIMMs) and the zip published as v0.3.2 ran the
full matrix. **0 of 12.** Every config finished (`exit=0`, mesh and texture present, 0 bugchecks
over the 45 minutes) and five of the six ports announced themselves - DepthMap capped at 9180 MB and
planned 12 tiles per view with images resident, DepthMapFilter, Meshing and Texturing all on the
device, `image vram: 6 allocs, peak 372 MB` in the blast summary - but FeatureExtraction was
"paired but silent" twelve times: the Cheshire binary ran and every view came out `[cpu]`.

The cause is in the build script, not the code. `build-alicevision-cuda.cmd` defaulted
`CHESHIRE_POPSIFT` to OFF, the 09-21 rebuild ran in a shell that had not set it, and CMake
configured `ALICEVISION_USE_POPSIFT=OFF`; `package-cuda.ps1` still staged `popsift.dll` from its
own install tree, so the package looked complete while `aliceVision_feature.dll` imported nothing
from it. The Linux CUDA script defaults PopSIFT on, which is why the 1050 Ti bundle was fine. Two
fixes (4ad97eb): the Windows script defaults to ON whenever the PopSift install exists and echoes
the choice, and the packager reads the staged feature library and throws if a `popsift.dll` is
shipped that it does not import. A package that says GPU SIFT and runs the CPU extractor can no
longer come out of the packager.

The zip was rebuilt from the v0.3.2 tag (e5a7cea) with PopSIFT on, through the new guard
(`feature.dll imports popsift: True`), sha256 `2832289c…`, 104 MB, and staged at
`build/release/0.3.2-fix/`. Its matrix on the 1080 Ti is recorded below.

**The rebuilt zip on the same card: 12 of 12**, every config 6/6 ports, no bugcheck across the
40 minutes - `base` 126 s (151 on the defective zip), `tiles` 131, `coarse` 80, `texbig` 121,
`cpufallback` 196, `verify` 186 (all six self-checks), `bridgecap` 156, `bridgespill` 271,
`bridgeoff` 126, `texcheck` 136, `ds1` 301, `blast` 346. FeatureExtraction now reads
`Choosing device 0: NVIDIA GeForce GTX 1080 Ti` followed by the stable-order line; the bridge
summaries match the defective run to the allocation (DepthMap volume peak 5244 MB, image 372 MB
with the mipmaps counted, Texturing 3850 MB), which is what one expects when only the extractor
changed. With this every one of the four v0.3.2 packages has run its full matrix on its own
hardware - RX 9070, RX 5500 XT, RX 6750 XT, GTX 1050 Ti and now the GTX 1080 Ti under Windows -
and the published Windows CUDA asset is the only one that is not the build validated here.

## 0.3.3-dev on the engine bay, RX 6750 XT (2026-09-21)

The Linux HIP bundle built from main after the undistortion map and the GPU downscale ran the
engine bay (107 photos, `ds1`) through Meshroom on house-pc: **ok**, 6/6 ports, 2794 s against
2863 s for 0.3.2 on the same box. By node, against the 0.3.2 run:

| node | 0.3.2 | 0.3.3-dev |
|---|---|---|
| PrepareDenseScene | 133 s | 127 s |
| Texturing | 176 s | 124 s |
| StructureFromMotion | 140 s | 143 s |
| Meshing | 310 s | 314 s |

Texturing is the win: edge padding and the 2x downscale both report "done on the GPU", and the
node lost 52 s. PrepareDenseScene shows the map computed once for the one intrinsic (4032x2268,
139 MB, 1 cached) but the profile line says where the node's time goes on this box - thread-seconds
read 50.6, undistort 36.5, write 94.7 - so it is bound on the JPEG read and the EXR write, not on
the arithmetic the map removed; the 6 s it lost matches the 9070 measurement. SfM and Meshing are
untouched by 0.3.3 so far and moved within noise (SfM sees a different landmark set every run,
152,179 here). The node app on house-pc is paired to this build.

**Published (2026-09-21 evening).** The Windows CUDA asset on the v0.3.2 GitHub release is now the
rebuild (sha `2832289c…`), SHA256SUMS carries the new hash, and the release body says what was
replaced and why; both were downloaded back and checked. Anyone whose copy hashes to `14ecee08…`
has the CPU-extractor zip. The Forgejo release carries notes only, nothing to swap there.

## 884 photographs: incremental SfM's loop exit, found and fixed (2026-09-21)

The first large set. The False Door of Ptahshepses (British Museum, Daniel Pett, CC BY-NC-SA:
884 JPEGs at 6000x3376 from a Sony A6000 over three mornings) ran through Meshroom on the RX
9070 with the 0.3.3 tree: FeatureExtraction 8 min for all 884 views on GPU SIFT, FeatureMatching
10 min, then **StructureFromMotion died at view 826 of 884** with `[fatal] invalid map<K, T> key`,
two seconds after "Bundle adjustment start". Meshroom's own binary, not a paired node. A resume
(cached features and matches kept, SfM rerun; the initial pair varies) died at 834 the same way,
and a third run of Meshroom's own incrementalSfM with `--verboseLevel debug` died at 833. Three of
three. Meshroom issue #2344 reports the same signature (`map::at` during bundle adjustment,
18,000 images, 21 hours in, open). Not memory: 5 GB in use with 43 GB free, and the message is a
map lookup, not an allocation.

**The mechanism**, from the debug log and the source. The main loop resects views in groups and
runs a bundle adjustment every ten resected views; a group below that count `continue`s and the
views accumulate as "reconstructed since the last BA". `findNextBestViews` returns false when no
remaining candidate reaches its score threshold, and the `while` exits there - with the
accumulated views still pending. Nothing after the loop bundle-adjusts them, and nothing hands
them to `LocalBundleAdjustmentGraph::updateGraphWithNewViews`. The outer pass then re-reads
`prevReconstructedViews = getValidViews()`, so from then on they are old. In the debug run nine
such views (resected at 650-656 in single-image groups, each "succeed") never appear in the
graph; every later bundle adjustment prints `The pose #... does not exist in the
'_mapDistancePerPoseId'` for them. They keep their resection pose, they observe landmarks, and
`getNewEdges` proposes an edge from every new view that shares landmarks with them - to a view
that has no node. `_graph.addEdge(_nodePerViewId.at(a), _nodePerViewId.at(b))` throws. It takes a
set where the candidate search runs dry mid-way, which is why 107 photographs never showed it and
884 did three times, and why upstream's 18,000-image report is the same crash.

**The fix**, generator step 5k, and the eighth paired node: after the resection loop, the views
resected since the last bundle adjustment are triangulated and bundle-adjusted the way a full
group is (same calls, same order, `registerChanges` included), before the next pass; and the
graph skips an edge whose endpoint it does not hold, with a warning, instead of throwing. Both
announce - the fix at the start of the reconstruction, the guard if it ever fires - the switch is
named in `--help` so the pairing scripts gate on it, and `CHESHIRE_SFM_PENDING_BA=0` restores
upstream's loop for comparison. Everything the fix runs is upstream code; the change is that it
runs.

**Verified on the same features and matches, local BA on (upstream's default), with the fixed
binary from the 0.3.3 stage tree: completed.** 830 poses and 1,354,686 landmarks in 3 passes
(64 resection groups), against 839 poses and 1,351,101 landmarks from the `useLocalBA=False`
workaround run that also got through (2436 s, every bundle adjustment global). The fix fired
three times - 5, 7 and 9 views pending at the end of a pass - which is exactly the count of
resected-but-orphaned views the debug run showed, and the graph guard never had to fire: with
the engine handing every view over, no edge reaches a view without a node. Three of three
upstream runs died at 830-834; the fixed binary went past that point in every pass and finished.

**The pipeline, end to end, on the RX 9070** (the `useLocalBA=False` workaround run, started before
the fix existed; the fixed-SfM run with every self-check on follows): **ok**, 6/6 ports, 21,139 s
wall from the resume. Per node: FeatureExtraction 558 s (884 views, GPU SIFT), FeatureMatching
824 s, StructureFromMotion 2436 s (839 poses), PrepareDenseScene 817 s, DepthMap 11,979 s (839
depth maps at downscale 2 on 20-megapixel frames - 53 % of the 6.25 h of compute), DepthMapFilter
460 s, Meshing 320 s, MeshFiltering 80 s, Texturing 5022 s (51 atlases of 8192^2, thirteen passes
over the 839 cameras). The textured mesh is 4,055,488 vertices and 8,104,312 faces, 679 MB as OBJ.
The two nodes that had never seen this many views, Meshing and Texturing, went through on the
first attempt; Texturing's cost is the atlas count times the camera count, which is the thing to
look at next for large sets.

## 0.3.3 stage tree on the RX 9070: 12 of 12 at 7 of 7 ports (2026-09-22)

With incrementalSfM paired the mini6 matrix reads seven ports per config, the seventh being the
SfM fix announcing itself. All twelve configs passed - `base` 60 s through `blast` - including
`verify` (all six Meshing self-checks) and `texcheck` (padding, resize and the direct OBJ writer
against Assimp). One trap on the way: the first launch reported DepthMap "NOT PAIRED" in every
config with no `paired:` line for it, while the same script paired all eight nodes by hand a
minute later and on the relaunch. **Corrected the same afternoon:** this was not a scanner race.
`meshroom-pair.cmd` had become LF-only (see "The package that could not pair DepthMap" below), and
cmd.exe's label search failed the first `call :pair`; converting the file to CRLF before the relaunch
is what made it pass.

## Rubble, 1678 views, RX 6750 XT: 1054 depth maps in, then the system disk (2026-09-22)

The Mill 19 Rubble set (Mega-NeRF, 1678 drone photographs at 4608x3456) through Meshroom on
house-pc with the 0.3.3-dev bundle: GPU SIFT 10 min for all views, ImageMatching 44,377 pairs,
FeatureMatching 40 min, StructureFromMotion 4265 s to 1591 poses under Meshroom's own binary
(no crash: this set's candidate search never ran dry mid-pass), PrepareDenseScene 1590 EXRs,
then DepthMap at about 3 views a minute, 3 full R cameras + 5 tiles per view on the 12 GB card.
It stopped at 1054 of 1590 after 9.3 hours: `Can't write output image file ... Failed OpenEXR
write` followed by a segfault in the HIP runtime on the error path. Not a GPU fault - the run's
cache sat under the home directory on house-pc's 219 GB system disk, not the data disk, and 124 GB
of depth maps filled it. The cache was moved to the data disk, symlinked back, and the run resumed
from its 1054 finished chunks (`CHESHIRE_E2E_RESUME=1` clears the SUBMITTED and RUNNING statuses).
The rule for this box: large-set outputs go under `/data`.

## The False Door with the fixed SfM paired, local BA on, every self-check on: ok, 7 of 7 (2026-09-22)

The same 884 photographs through the harness with the 0.3.3 stage tree's incrementalSfM paired,
upstream's default local bundle adjustment, and every self-check switched on: **ok**, 7/7 ports,
23,054 s wall, 6.40 h of compute. StructureFromMotion took 1636 s to 833 poses and 1,355,362
landmarks - the fix fired twice, 5 and 6 views pending at the end of a pass, and the run went past
the 830-834 range where upstream's binary died three times without a fatal. Its inputs were the
crash run's: all 44 match files byte-identical, and 1,767 of 1,768 feature files (one descriptor file,
view 216823232, differs in a few bytes and changes no match), so its 833 poses against the replay's
830 are incremental SfM's own run-to-run variation, not different input. That is 800 s faster
than the global-BA workaround (2436 s), which is the point of local BA. Per node: FeatureExtraction
560 s, FeatureMatching 796 s, PrepareDenseScene 721 s, DepthMap 12,010 s, DepthMapFilter 521 s,
Meshing 852 s (with five self-checks running alongside), MeshFiltering 83 s, Texturing 5847 s (52
atlases, with the padding and resize checks). Textured mesh: 4,071,685 vertices, 8,136,271 faces,
679 MB. *Corrected 2026-09-22:* this entry first gave "320 s without" for Meshing and "5022 s
without" for Texturing. Both came from the earlier `useLocalBA=False` workaround run (839 poses,
51 atlases), a different reconstruction, not this one without its checks. This run's Meshing,
rerun from the same cache with no self-checks through the 0.3.3 release package, takes 479 s with
the depth maps read cold from disk and 489 s warm (see the fusion profile below).

**Every self-check, at 884 views, zero:** filterByPixSize identical to single-threaded upstream on
all 52,305,736 slots and on each of the four later passes; GPU knn identical to nanoflann on all
3,124,680,410 queries, twice; max-flow 0 of 29,080,924 cells labelled differently (the two flow
totals differ as documented, docs/12); segmentFullOrFree identical on all 29,080,924 cells, both
passes; tedge 175,896 cells on both sides, 0 beyond 1e-3 relative; GPU padding 0 of 67,108,864
texels differing on each of 52 atlases; GPU resize 0 of 50,331,648 channels differing on each of
52. VRAM peaks: DepthMap image  vram: 99 allocs, peak 1133 MB;map    vram: 848 allocs, peak 1146 MB;volume vram: 96 allocs, peak 7734 MB; DepthMapFilter other  vram: 120 allocs, peak 946 MB;other  vram: 120 allocs, peak 965 MB;other  vram: 120 allocs, peak 985 MB;other  vram: 85 allocs, peak 985 MB; Meshing 3910 MB, Texturing 14,162 MB on the 16 GB
card - the largest allocation of the whole run, as on the engine bay. Planner:      70 planner 1: VRAM budget 11630.2 MB, 3 full R cameras + 7 tiles.

## The 0.3.3 Windows CUDA package on the GTX 1080 Ti: 11 of 12, and what the twelfth found (2026-09-22)

The CUDA package built from the fixed tree (SuiteSparse still in, see the next section) ran the
matrix on bench-pc: 11 of 12, 7/7 ports everywhere, 0 bugchecks. The one failure is `texcheck`:
padding 0 of 67,108,864 texels as on every card, OBJ identical, but **GPU resize 7,772,469 of
50,331,648 channels differing from OpenImageIO**, where every HIP card reports 0. The cause is the
compiler, not the kernel: every port file keeps FMA contraction off with `#pragma clang fp
contract(off)`, so a*b+c rounds twice as the CPU reference does; nvcc does not honour that pragma
and contracts by default (`--fmad=true`). The resize check is the first float-exact self-check that
has ever run on the CUDA build - the Meshing checks compare labels, counts and vertex choices -
which is why nothing said so before. Generator step 5l gives the port sources `--fmad=false` on
the CUDA backend; upstream's own depth-map kernels are left as they were built, since byte-identity
with the CUDA reference depends on it. The rebuilt package's matrix follows.

**A second disk incident on the same run (2026-09-22).** During the 124 GB cache move, one SATA
write to house-pc's data disk timed out (`sd 5:0:0:0: [sdb] FAILED Result: hostbyte=DID_TIME_OUT`,
31 s command age) and ext4 reported "potential data loss" for one inode: PrepareDenseScene's
undistorted image for view 1387742125. SMART is clean on both disks (0 reallocated, 0 pending, 0
CRC errors) and nothing has recurred. The file decodes as damaged - `EXR_ERR_CORRUPT_CHUNK` on every
channel with the OpenEXR reader - yet Texturing read it three times with "contributions to 3 texture
files" and no error, so OpenImageIO tolerated the bad chunk and that camera contributed whatever
came out of it to three atlases. One camera of 1590; the mesh and counts are unaffected, the
texture of the affected patches is suspect, and the run stands as a scale test with that caveat.
The lesson is the harness's: a resumed run has no check that its cached inputs still decode.

## SuiteSparse out, fmad off: the 0.3.3 Windows packages on both cards (2026-09-22)

Ceres 2.2.0 rebuilt without SuiteSparse (`scripts/windows/build-ceres-nosuitesparse.cmd`: Eigen's
sparse backend with METIS ordering, LAPACK from the tree's OpenBLAS, the same MSVC, Eigen 3.4.1,
glog 0.7.1 and gflags 2.3.0 as the prebuilt vcpkg tree), spliced into `tools/vcpkg-deps`
(`splice_ceres.py`, the old files in `build/ceres-vcpkg-backup`). The new `ceres.dll` imports glog,
METIS and LAPACK only. Both packagers now refuse a stage carrying `libspqr.dll` or `libcholmod.dll`,
and the guard earned its keep twice on the first day: the CUDA install tree and then the CUDA
build's output folder each still held the old DLLs from the previous build's applocal deployment,
and the packager staged them until they were removed. The packager also deletes its previous zip
before staging, after a guard failure left the day's earlier zip in place and it travelled to the
bench as if it were new.

- **RX 9070, HIP package (`test033c-gfx1201`)**: 12 of 12, 7/7 ports; the packager reported no
  GPL library. Timings of five configs are inflated (`coarse` 451 s, `cpufallback` 456 s, `verify`
  467 s) because the CUDA rebuild compiled on the same CPU alongside them.
- **GTX 1080 Ti, CUDA package (sha `861ec66c…`)**: 12 of 12, 7/7 ports, 0 bugchecks, no GPL
  library in the extracted package, and `texcheck` now passes: GPU padding 0 of 67,108,864 texels,
  **GPU resize 0 of 50,331,648 channels** - step 5l's `--fmad=false` on the port sources is what
  turned the 7,772,469 of the morning into 0. Depth-map digests differ from the SuiteSparse
  package's run as they must: Eigen's sparse solver rounds the bundle adjustment differently, the
  poses move in the last digits, and the depth maps follow; within the run every bridge
  configuration still matches `base` bit for bit.

Meshroom 2023.3's own Windows release ships `libspqr.dll` and `libcholmod.dll` as well, and so does
AliceVision's 2026.09.01 prebuilt vcpkg zip this tree came from, so 0.3.0-0.3.2 matched upstream's
practice; the maintainers' ruling in discussion #2116 (no SPQR in pre-built binaries) is the one
0.3.3 follows. The Linux bundle's Ceres was built without SuiteSparse from the start.

## Rubble, complete: 1678 views through a 2013 dual-core (2026-09-22)

The resumed run finished: **ok**, 6/6 ports (the b033 bundle predates the SfM pairing), 16.0 h
of compute over the two sessions. Per node: FeatureExtraction 756 s, ImageMatching 37 s,
FeatureMatching 2265 s, StructureFromMotion 4291 s (1591 poses, upstream's binary), PrepareDenseScene
3363 s, DepthMap 34,267 s (1590 maps at downscale 2, 3 full R cameras + 5 tiles per view on the
12 GB card), DepthMapFilter 1544 s, Meshing 1020 s (8 GB RAM peak on 14.6 GB, no swap), MeshFiltering
37 s, Texturing 10,013 s (34 atlases of 8192^2). The textured mesh is 1,679,493 vertices and
3,350,047 faces, 265 MB; the cache is 164 GB, 98 GB of it PrepareDenseScene's EXRs. The i3-4330's
CPU-only nodes - the SfM, the AC-RANSAC part of matching (the split was not measured on this box)
and PrepareDenseScene's decode, undistortion and EXR write (3363 s) - are about 2.5 of the 16 hours;
Meshing's host-side phases and MeshFiltering add a few hundred seconds, and the GPU nodes decode
their images on the two cores.
Two disk incidents along the way (the system disk filling, one write timeout that damaged one
undistorted image) are recorded above; the result stands with the one-camera texture caveat.

**The bundle-adjustment cost of leaving SuiteSparse, measured.** Two replays of the False Door's
StructureFromMotion on the same cached features and matches, upstream's default local BA,
`CHESHIRE_BA_PROFILE=1`, on an idle RX 9070 box (a first attempt overlapped the CUDA rebuild and
was discarded: its Jacobian time, which does not depend on the sparse backend, was 903 s against
421 s):

| Ceres | SfM wall | BA solves | BA total | Jacobians | linear solver | poses / landmarks |
|---|---|---|---|---|---|---|
| SuiteSparse (CHOLMOD/SPQR) | 1944 s | 938 | 1069 s | 597 s | 323 s | 815 / 1,342,489 |
| Eigen sparse + METIS (0.3.3) | 1498 s | 904 | 733 s | 421 s | 304 s | 831 / 1,354,886 |

The linear solver, the only part the backend touches, is 304 s against 323 s: no penalty at this
size. The reduced camera system of a few hundred poses is small enough that Eigen's simplicial
factorisation with METIS ordering keeps up with CHOLMOD's supernodal one; the rest of the gap is
the two runs' different trajectories (incremental SfM's initial pair and resection order vary run
to run, 815 against 831 poses), not the solver. Upstream's `ALICEVISION_REQUIRE_CERES_WITH_SUITESPARSE`
default is OFF and the build accepted the new Ceres without a change.

**Linux, RX 6750 XT, the fixed bundle (b033b, pre-release tree): 12 of 12** through the mini6
matrix on house-pc, `verify` and `texcheck` included - scored at 6/6 ports because the harness
copy on that box predated the eighth node; the pairing script did pair `aliceVision_incrementalSfM`.
The release bundle's run below is scored at 7/7 with the current harness.

The damaged EXR was regenerated after the run: PrepareDenseScene's chunks index every view in the
scene (1678, by view id), not the posed ones, so the view sat in chunk 26 rather than the 25 a
posed-only count gave; rerunning that one chunk with the bundle's binary rewrote the file, which
now decodes on every channel. The Rubble texture caveat stands for the run as it was scored; the
cache is clean for what comes next.

## 0.3.3 release packages: the Linux pair, and a restart mid-build (2026-09-22)

**Linux HIP (`cheshire-alicevision-hip-linux-x64-rocm7.2.tar.gz`, sha `2ab6d229…`) on the RX 6750 XT:
12 of 12, 7/7 ports** with the release harness, `aliceVision_incrementalSfM` paired and announcing
the fix. `base` 80 s, `tiles` 80, `coarse` 55, `texbig` 85, `cpufallback` 155, `verify` 110,
`bridgecap` 85, `bridgespill` 175, `bridgeoff` 80, `texcheck` 90, `ds1` 195, `blast` 220. Self-checks:
max-flow 0 of 1,673,871 cells labelled differently, knn 0 of 8,791,919 queries naming a different
vertex, segments identical on all 1,673,871 cells, padding 0 of 67,108,864 texels, resize 0 of
50,331,648 channels. house-pc's node app is paired with this bundle.

**Linux CUDA (`cheshire-alicevision-cuda-linux-x64-cuda12.9.tar.gz`, sha `2809619a…`)**: packaging
checks all pass (no libcuda, libcudart 12.9.79, PopSIFT a CUDA build linked by the feature library,
no HIP artefacts, every dependency resolved from the bundle); all seven port kernels compiled with
`--fmad=false`; the SfM binary names `CHESHIRE_SFM_PENDING_BA` in its help. Its matrix waits for an
NVIDIA card in house-pc.

**Windows: the build that the app restart interrupted.** The release build's hip6.2 family lost five
of nine targets without a source error: two links refused by the virus scanner holding a freshly
linked executable ("being used by another process", "Access is denied."), and three steps that
could not start (`0xC0000142`) once the Claude session that had launched the build went away. The
payload builder now retries build steps whose own output carries one of those signatures, the
resumed build runs detached from the session, and both HIP build folders turned out to hold the
old `libspqr.dll`/`libcholmod.dll` from earlier applocal deployments (moved out before the resume,
as the CUDA tree's had been).

**Linux CUDA (`cheshire-alicevision-cuda-linux-x64-cuda12.9.tar.gz`, sha `2809619a…`) on a GTX 1080 Ti
in house-pc: 12 of 12, 7/7 ports.** `base` 85 s, `tiles` 80, `coarse` 50, `texbig` 85, `cpufallback`
160, `verify` 115, `bridgecap` 105, `bridgespill` 210, `bridgeoff` 85, `texcheck` 95, `ds1` 210, `blast`
235. Self-checks: max-flow 0 of 1,671,337 cells labelled differently, knn identical to nanoflann on
all 8,790,341 queries, segments identical on all 1,671,337 cells, padding 0 of 67,108,864 texels,
resize 0 of 50,331,648 channels - the resize check now exact on CUDA because the port kernels are
built with `--fmad=false`. Bridge peaks under `blast`: DepthMap maps 553 MB and camera mipmaps
372 MB, DepthMapFilter 790 MB, Meshing 845 MB, Texturing 4038 MB.

## The package that could not pair DepthMap (2026-09-22)

The first run of the bundled Windows package failed every configuration on both cards at 3/7 ports:
DepthMap ran Meshroom's own CUDA binary ("This program needs a CUDA-Enabled GPU") and the three
nodes after it never ran. Pairing by hand showed why: `The system cannot find the batch label
specified - pair` on the first `call :pair`, while the seven later calls to the same label worked.
`meshroom-pair.cmd` in the package was LF-only - the earlier CR count had come from Git Bash's
`grep`, which misreports it; Python reads zero CRs - and cmd.exe's label search is unreliable in
LF-only batch files, depending on where the label falls against its 512-byte reads.

The file became LF-only in this checkout. It is the Windows checkout, written by Windows git with
`core.autocrlf=true`, and `scripts/linux/wsl-bundle.sh` had been run on it from WSL: the script
stashes local changes and hard-resets to `origin/main`, and Linux git, with no autocrlf, writes every
text file with LF as it does. At 11:53 that rewrote `meshroom-pair.cmd`; the morning's first
stage-gate failure had the same cause through the previous evening's build, not a scanner race. The
stashes held nothing but line-ending churn and one regenerated patch export.

Fixed four ways (04b30fe): `.gitattributes` pins `*.cmd`, `*.bat` and `*.ps1` to CRLF and `*.sh` to LF
for whichever git writes the checkout; all three Windows packagers rewrite every staged batch file
with CRLF and verify it; `wsl-bundle.sh` refuses a checkout under `/mnt`; and the bundle was rebuilt
from the unchanged payloads, after which all eight nodes pair by hand, DepthMap first. The Windows
CUDA zip built at 10:45 was unaffected: its copy had CRLF endings.

**The release CUDA package against upstream's own CUDA, on the GTX 1080 Ti in house-pc.** The 41-view
depth maps (`scripts/linux/run-depthmap.sh`, Meshroom's standard DepthMap parameters) from the Linux
CUDA release bundle, compared with the CUDA 11.3 reference computed by upstream's binary on this
card in September: **82 of 82 files byte-identical** - all 41 depth maps and all 41 similarity maps,
whole-file SHA-256 - in 213 s against the reference's 379 s. The bridge planned 2 full R cameras +
2 tiles on the 11 GB card with images resident. Step 5l's `--fmad=false` applies only to the ports,
so this is also the check that it left upstream's depth-map kernels as they were.

## The rebuilt Windows bundle on both AMD cards (2026-09-22)

The bundle with CRLF batch files (`cheshire-alicevision-windows-x64.zip`, 192,889,989 bytes, sha
`eadce479…`), all eight nodes paired, DepthMap first:

- **RX 9070 (rocm7.2 gfx12-generic payload): 12 of 12, 7/7 ports.** `base` 70 s, `tiles` 65,
  `coarse` 40, `texbig` 65, `cpufallback` 85, `verify` 90, `bridgecap` 100, `bridgespill` 141,
  `bridgeoff` 65, `texcheck` 75, `ds1` 146, `blast` 171. Self-checks: max-flow 0 of 1,669,809 cells
  labelled differently, knn identical to nanoflann on all 8,770,744 queries, segments identical on
  all 1,669,809 cells, neighbour table and facet weights 0 differing, filterByPixSize identical on
  all 18,289,152 slots, padding 0 of 67,108,864 texels, resize 0 of 50,331,648 channels.
- **RX 5500 XT on bench-pc (hip6.2 gfx1012 payload): 12 of 12, 7/7 ports, 0 bugchecks.** `base`
  161 s, `tiles` 146, `coarse` 100, `texbig` 151, `cpufallback` 217, `verify` 206, `bridgecap` 191,
  `bridgespill` 412, `bridgeoff` 151, `texcheck` 171, `ds1` 372, `blast` 417. Self-checks: max-flow
  0 of 1,699,013 cells, knn 0 of 8,802,911 queries naming a different vertex (1,635,610 report a
  different distance in the last bits, the RDNA1 behaviour the vertex verdict exists for), segments
  identical on all 1,699,013 cells, neighbour table and facet weights 0 differing, filterByPixSize
  identical on all 18,289,152 slots, padding 0 of 67,108,864 texels, resize 0 of 50,331,648 channels.

The RDNA2 payloads (hip6.2 gfx1030-1036) come from the same source and build as the gfx1012 one
and were not run on hardware this round; the RX 6750 XT is in house-pc under Linux.

## 0.3.3 release gates (2026-09-22)

Every artifact gated as the file a user downloads, with the harness that scores seven ports
(StructureFromMotion paired as the eighth node, checked by its announce line).

| artifact | hardware | end to end | beyond the matrix |
|---|---|---|---|
| `cheshire-alicevision-windows-x64.zip` (AMD, 11 payloads) | RX 9070, Windows | 12/12 at 7/7 | stage tree 12/12 at 7/7 before bundling |
| same | RX 5500 XT, Windows (bench-pc) | 12/12 at 7/7, 0 bugchecks | - |
| `cheshire-alicevision-hip-linux-x64-rocm7.2.tar.gz` | RX 6750 XT, Linux | 12/12 at 7/7 | - |
| `cheshire-alicevision-cuda-linux-x64-cuda12.9.tar.gz` | GTX 1080 Ti, Linux | 12/12 at 7/7 | 41-view depth maps 82/82 byte-identical to upstream's CUDA 11.3 |
| `cheshire-alicevision-cuda-windows-x64-cuda12.9.zip` | GTX 1080 Ti, Windows (bench-pc) | 12/12 at 7/7, 0 bugchecks | no GPL library; resize exact with `--fmad=false` |

The Windows packages carry no SuiteSparse (both packagers refuse one that does), and every batch
file in them is CRLF (all three packagers rewrite and verify it). Outside the matrix, the fixed
StructureFromMotion carried the 884-view False Door through Meshroom with every self-check on and
zero differences, and the 1678-view Rubble ran to a textured mesh on a 2013 dual-core.

## Meshing at 833 views: where fusion's time goes (2026-09-22)

The False Door fix run's Meshing, rerun from its kept cache with no self-checks through the 0.3.3
release package (`cheshire-run.cmd aliceVision_meshing`, the node's own command line, Ryzen 5 5600X +
RX 9070): **479 s** with the depth maps read cold from disk, **489 s** warm with
`CHESHIRE_GPU_VIS_LOG=1`. `fuseFromDepthMaps` is about 290 s of it:

| phase | cold run | warm run, profiled |
|---|---|---|
| load depth maps (12 threads, gaussian on the GPU) | 111 s | 43 s |
| `filterByPixSize` x5 (kd-trees on the CPU, index-order rounds) | ~16 s | ~16 s |
| removeInvalidPoints, margin setup | ~9 s | ~9 s |
| visibility pass 1 (9,937,481 points, 831 cameras) | 84 s | 72 s |
| visibility pass 2 (4,699,910 points) | 66 s | 133 s |

The visibility accounting (pass 1): votes 38.3 s, backproject 23.3 s, maps 4.7 s, index 3.2 s,
device wait 1.7 s; the device's own kernel time, 38.7 s, overlaps the host work. So the two
visibility passes are bound by host work - collecting votes and backprojecting pixels, camera by
camera - not by the knn kernels and not by the disk. The kd-tree filter that was left on the CPU
after step 4p is about 16 s of 480. After fusion: tetrahedralisation 25 s (geogram), votes, graph
and cut about 60 s, post-cut processing and cleaning about 95 s, writes about 20 s. Pass 2's 133 s
in the profiled run against 66 s in the cold one is not explained yet; other work was running on
the box during the second run.

## Fusion's visibility passes: the harness and bucketed votes (2026-09-22)

The first steps of the plan in docs/roadmap.md ("Fusion: the visibility passes on the GPU"), measured
with `scripts/meshbench.py` (the Meshing node's own command line on the mini6 cache, the 41-view
reference with its filtered maps generated by the 0.3.3 DepthMapFilter into `build/ref/monstree-41-dmf`,
and the kept 833-view False Door cache) on an idle 5600X + RX 9070.

**Step 0.** The development tree (`build/av-gfx1201-popsift`, the release's gfx12-generic payload tree)
reproduces the release package exactly: tetrahedralisation input checksums `7d875321...` (6 views),
`cdd23710...` (41) and `64b36ee4...` (833, the same as the False Door run of this morning). Idle, the
two visibility passes at 833 views are 61 + 50 s (votes 28.5 + 20.3 s, backprojection 20.9 + 22.0 s,
device kernel 21.5 + 17.2 s); the 133 s pass 2 of the first profile was load on the box. The Linux
bundle's nanoflann header is byte-identical to Windows's (1.9.0), and hipCUB is present in both
Windows toolchains and in the WSL build distro (house-pc needs none: it is header-only).

**Step 2 (7213bfa), no output change.** A per-pass FNV-1a digest of every vertex (coordinates, nrc,
camera list in order, pixSize) on every path, stable across runs and identical at one OpenMP thread;
CHECK now also redoes the votes on shadow copies from the host's own nanoflann answers and compares
every vertex. Found on the way: 0 NaN depths in all three sets (the row count now uses the fill's
test anyway), tree depth 28-35, and upstream's loop-position quirk at 833 views - the loop visits
camera ids 0..830, among them 86 and 217 that are not in the camera list, and never visits the listed
831 and 832 (reproduced exactly; worth reporting upstream).

**Step 3 (ebc33f0), bucketed host votes.** Every thread used to scan all of a camera's answers and
apply only its own vertex range; a camera's votes fall in a narrow range, so one or two threads did
the work. The decisions are now made in parallel and the voting queries scattered stably into
buckets of 4096 vertices, applied in parallel in pixel order. Digests identical to step 2 on all three
sets, at one thread, and with `CHESHIRE_GPU_VIS_BUCKETS=0`; CHECK identical to the ordered reference on
all 9,937,481 and 4,699,910 vertices of the 833-view passes (3.1 billion queries each).

| 833 views | step 2 | step 3 |
|---|---|---|
| votes, pass 1 / pass 2 | 28.5 / 20.3 s | 16.6 / 12.1 s |
| visibility passes | 110.5 s | 94.5 s |
| Meshing node | 357-439 s | 340 s, twice |

Pass 1 now runs at 62 ms per camera, the device-bound figure the design predicted. Not explained
yet: the device's own kernel time rose once it ran back to back (21.5 -> 36.5 s in pass 1, 0.64 ->
2.26 s at 41 views) - clocks under sustained load or the event timing; it is the floor from here,
so it is measured before any device work.

## A 444-photo drone survey: pairing, not matching (2026-09-22)

The user's DJI Phantom 4 set (FC330, 4000x3000, 444 photos over a 400 m x 400 m grid, shots about
15 m apart, GPS in every EXIF) through house-pc's node app, Standard preset, `CHESHIRE_QR_NULLSPACE=1`.
**First run: SfM placed 49 of 444 views** (94,238 landmarks), and the rest of the pipeline built a
5.5 M-face mesh of one 267 m x 134 m patch in 28 minutes.

Not the matcher and not QR: one matching chunk rerun with QR on and off kept the same 29 of 1051
pairs, 40,236 against 40,344 matches. The pairs themselves were wrong. Above `minNbImages=200`
Meshroom's ImageMatching stops comparing every pair and takes ~50 partners per photo from the
vocabulary tree; on uniform ground the partners it chose were a median 214 m away, only 20 % of each
shot's 8 GPS-nearest neighbours were ever proposed (95 % of those verified), and 419 of 10,922
proposed pairs survived geometric verification (3.8 %). The 150-250-photo captures the app was
built for never see this: below 200 photos the pairing is exhaustive.

**Rerun with `ImageMatching:method=Exhaustive`** (the app's new "Image pairing" option, everything
else unchanged): 98,346 pairs, 2377 verified, **444 of 444 views placed**, 829,908 landmarks,
6,455,085 faces, 26 atlases, 3.66 h on the i3-4330 + RX 6750 XT. Per node: FeatureExtraction 175 s,
FeatureMatching 2542 s (23 chunks), StructureFromMotion 871 s, PrepareDenseScene 717 s, DepthMap
6219 s, DepthMapFilter 291 s, Meshing 348 s, MeshFiltering 81 s, Texturing 1917 s.

**What GPS pairing would have done**, from the rerun's verified pairs and the photos' coordinates:
the real pairs sit a median 29 m apart, 90 % within 39 m, 99 % within 56 m, one at 302 m (almost
certainly a false match).

| radius | pairs proposed | share of exhaustive | verified pairs caught |
|---|---|---|---|
| 40 m | 2,584 | 2.6 % | 2,169 of 2,377 (91.2 %) |
| 50 m | 3,593 | 3.7 % | 2,330 (98.0 %) |
| 60 m | 5,674 | 5.8 % | 2,360 (99.3 %) |
| 80 m | 9,071 | 9.2 % | 2,376 (100 % but the outlier) |
| 100 m | 13,687 | 13.9 % | 2,376 |

An 80 m radius (5.4x the shot spacing) finds every real pair with a tenth of the exhaustive
matching, about 4 minutes instead of 42 on this box; that is the target for the GPS pairing item in
docs/roadmap.md (0.3.5). The first run's folder was deleted after the rerun completed; the rerun's
photos are hard links and keep their data.

## The Linux knn distances are not a contraction (2026-09-22)

Step 1 of the fusion plan added `#pragma clang fp contract(off)` to `knnGPU.cu` (5d6045b) on the
reading that HIP fused the "unfused" metric and the initial bounding-box distances, which would
explain the ~20 % of last-bit distance differences the Linux knn check reports. Tested on house-pc
(RX 6750 XT, Linux) on the 41-view job's own Meshing cache, the 0.3.3 bundle against a rebuild at
72f9c14: the check reports **exactly the same counts** - 17,383,299 and 18,404,248 of 87,354,192
queries with a different distance, 0 naming a different vertex - so the pragma changed nothing and
the cause is elsewhere. The two bundles are otherwise identical on this box (votes 79,843,309 and
73,069,905, tetrahedralisation input `4fab73b4bdbd49` in both), the bucketed votes check
("identical to the ordered host reference") passes on Linux, and the Windows results were unchanged
by the pragma (docs above). Next for the distances: dump a handful of differing (query, vertex)
pairs from the check and recompute the metric on the host in both forms to see which arithmetic
the device is actually doing.

## 0.3.4: bundle adjustment's Jacobians without the autodiff passes (2026-09-23)

The roadmap item was "analytic Jacobians for bundle adjustment", on the profile above (Jacobians
49 % of BA on the engine bay, 421 s of 733 s on the False Door). Reading upstream's cost function
changed the shape of the work. `ProjectionSimpleErrorFunctor` (`sfm/bundle/costfunctions/projection.hpp`)
is a `ceres::DynamicAutoDiffCostFunction` whose functor moves the point into the camera frame with
Jets and then calls `CostIntrinsicsProject` through `DynamicCostFunctionToFunctorTmp`.
`CostIntrinsicsProject` is *already analytic*: it returns the projection and its derivatives with
respect to the intrinsics, the distortion and the camera-frame point. The cost is in how Ceres
drives it: a dynamic autodiff functor is evaluated in passes of `Stride` (4) derivative
components, and every pass calls `CostIntrinsicsProject::Evaluate` with all three Jacobian blocks
again. With the point (3), the pose (6) and the intrinsics (4 + distortion) free, that is three to
five full analytic evaluations per residual block per Jacobian, each with the Eigen temporaries
those derivative methods allocate, plus the Jet arithmetic and one heap allocation per pass
(`dynamic_cost_function_to_functor.h:127-131`).

Step 5m (`hip/port/sfm_ba/projectionCheshire.hpp`) gives `CHESHIRE_BA_JACOBIANS` three settings:

* `autodiff`: upstream, unchanged.
* `stride`: upstream's functor instantiated with `Stride` 32, so one pass. A derivative component
  is computed by the same operations whatever the stride, so the numbers cannot change; only the
  number of times the inner function runs does. **The default.**
* `analytic`: one `CostIntrinsicsProject::Evaluate`, then the chain rule by hand through the pose
  (angle-axis and centre) and, for rigs, the sub-pose. The rotation's derivative comes from Ceres'
  `AngleAxisRotatePoint` on 3-component Jets, i.e. the arithmetic autodiff itself uses for it, so
  the only difference from upstream is the order of the chain-rule sums. Opt-in until the quality
  gate of 0.3.5 exists, as the roadmap item said.

`CHESHIRE_BA_CHECK=1` evaluates a reference cost function next to the selected one on every call
(upstream's autodiff, or the analytic one when autodiff is selected) and prints, after each solve,
how many residual and Jacobian values differed and by how much. `scripts/sfmbench.py` runs
`aliceVision_incrementalSfM` from a prepared feature-and-match cache with Meshroom 2023.3's
default node options and `CHESHIRE_BA_PROFILE=1`, and records one JSON line per run.

**41 views (monstree full, dspsift), RX 9070 box, 12 threads, idle.** Two runs per setting, one
check run each; every run ended with 41 poses and 68 bundle-adjustment solves:

| Jacobians | SfM wall | BA total | Jacobians | linear solver | residuals | landmarks | RMSE |
|---|---|---|---|---|---|---|---|
| autodiff (upstream) | 67.0 s, 66.5 s | 26.9 s, 26.4 s | 13.0 s, 12.8 s | 6.4 s, 6.2 s | 0.8 s, 0.7 s | 80,808; 80,795 | 1.2359; 1.2324 |
| stride 32 | 60.6 s, 60.6 s | 20.6 s, 20.3 s | 6.9 s, 6.8 s | 6.2 s, 6.1 s | 0.7 s, 0.7 s | 80,811; 80,793 | 1.2334; 1.2322 |
| analytic | 56.6 s, 57.0 s | 17.9 s, 18.0 s | 4.3 s, 4.4 s | 6.2 s, 6.2 s | 0.6 s, 0.6 s | 80,803; 80,802 | 1.2329; 1.2338 |

The check runs (`build/sfmbench/41/runs/stride-check`, `analytic-check`), 3.54 million
evaluations of which 2.12 million with Jacobians, over the same 68 solves:

* stride against autodiff: **0 of 7,089,684 residual values and 0 of 67,949,760 Jacobian values
  differ.** Bit-identical, as the argument says it must be.
* analytic against autodiff: residuals 0 of 7,089,912 differ; Jacobians 12,700,913 of 67,952,448
  (18.7 %) differ, by at most 9.09e-13 absolute and 4.38e-10 relative. Rounding of the chain-rule
  order, nothing else.

So the Jacobian phase is 1.9x faster at the same numbers, and 3.0x faster at rounding-level
differences; bundle adjustment as a whole goes from 26.7 s to 20.5 s and 18.0 s, the node from
66.8 s to 60.6 s and 56.8 s. The landmark counts (80,793 to 80,811) and RMSE (1.2322 to 1.2359)
spread the same way within a setting as across them: that is incremental SfM's run-to-run
variation (unseeded RANSAC across threads, docs/17), not the Jacobians. What is left in BA after
the change is the linear solver (6.2 s) and the per-solve problem construction and trust-region
bookkeeping ("other", 6.7 s in every setting); the 4.3 s of analytic Jacobians is one inner
evaluation per block, with the Eigen dynamic-size temporaries `getDerivativeTransformProjectWrt*`
return - the next slice, and the shape the device version of 0.3.5 will need anyway.

**Engine bay (107 photos, dspsift), same box.** One run per setting and one check run; 107 poses
in every run, 140 solves (141 in the analytic runs: the trajectory differs by rounding, so the
solve count can too):

| Jacobians | SfM wall | BA total | Jacobians | linear solver | residuals | landmarks | RMSE |
|---|---|---|---|---|---|---|---|
| autodiff (upstream) | 69.2 s | 34.7 s | 18.7 s | 8.1 s | 1.2 s | 140,636 | 1.6851 |
| stride 32 | 60.0 s | 25.5 s | 9.9 s | 7.8 s | 1.2 s | 140,387 | 1.6890 |
| analytic | 58.6 s | 24.2 s | 6.8 s | 8.7 s | 1.2 s | 140,647 | 1.6839 |

The analytic check on this set (929,238 evaluations, 421,505 with Jacobians): residuals 0 of
1,858,476 values differ; Jacobians 2,847,747 of 7,036,470 differ, by at most 4.55e-13 absolute and
5.45e-11 relative. Jacobians 1.9x and 2.75x, BA 34.7 s to 25.5 s and 24.2 s, the node 69 s to
60 s and 59 s; the landmark spread (140,387 to 140,647) is the run-to-run kind again.

**False Door (884 views, sift), the kept cache of the 0.3.3 end-to-end run, one run per setting,
same box, idle (the residual-evaluation cost per unit of work, the same code in all three, is
45-52 ns in every run above and below, which is the check that nothing else was running).** Here
the wall clock says nothing about the Jacobians, because incremental SfM's trajectory varies run
to run far more than the change does: the three runs did 1.42, 0.57 and 1.54 billion
residual-block-iterations of bundle adjustment (960, 922 and 936 solves; 48, 36 and 39 of them
global, over 300 k blocks), ended at 825, 831 and 828 poses, and took 2368 s, 1302 s and 1667 s.
The two earlier autodiff replays of the same cache (the CHOLMOD table above) spread the same way,
1069 s against 733 s of BA. What is comparable is the cost per residual-block-iteration, which
`CHESHIRE_BA_PROFILE` gives per solve:

| Jacobians | Jacobians, ns per residual-block-iteration (global solves) | same, all solves | residuals | SfM wall | poses / landmarks |
|---|---|---|---|---|---|
| autodiff (upstream) | 709 | 711 | 51 | 2368 s | 825 / 1,345,755 |
| stride 32 | 440 (1.6x) | 443 | 52 | 1302 s | 831 / 1,355,046 |
| analytic | 251 (2.8x) | 251 | 45 | 1667 s | 828 / 1,350,277 |

The same measure on the small sets: 41 views 865-899 / 453-461 / 293-322 ns, engine bay 701 /
380 / 216 ns (global solves), so the per-work gain is the same at every size and the small sets'
wall-clock ratios are the honest ones there because their trajectories barely vary. Two things
follow. A bundle-adjustment change at 884 views cannot be judged by wall clock until SfM is
seeded per task (the 0.3.4 determinism item), and the per-work numbers are what
`scripts/sfmbench.py`'s logs should be read for at that size. And the stride mode's 1.6x here
against 1.9x on the small sets is the intrinsics: at 884 views one intrinsic block is shared by
every camera and locked most of the time, so the active parameter count is 9 and upstream needs
three passes, not five.

## 0.3.4: incremental SfM, reproducible (2026-09-23)

The roadmap item said: seed SfM per task instead of per process, start with a one-thread rerun.
The one-thread rerun was the surprise. Two runs of `aliceVision_incrementalSfM` on the 6-view set
with `--maxCoresAvailable 1` (one OpenMP thread, one Ceres thread, the default seed 5489) ended
with the same 9290 landmarks and the same RMSE to six digits, and poses that differed in the last
bits (`0.075924026963859248` against `...373`), so threads were not the whole story. The rest is
Ceres: `ParameterBlockOrdering` is `std::map<int, std::set<double*>>`, so within an elimination
group the parameter blocks are ordered by *address*; upstream keeps them in `std::map` nodes, and
the Windows heap hands out node addresses in an order that changes from run to run. A different
elimination order changes the rounding of the reduced camera system, and the trajectory drifts
from there. Step 5n (`hip/port/sfm_ba/deterministic.hpp`) does three things, always on:

* **A generator per task.** Resection and LO-RANSAC triangulation drew from the engine's one
  `std::mt19937` inside OpenMP loops (a data race as well as a scheduling dependence). Each view's
  resection and each track's triangulation now derive their own generator from (seed, task kind,
  task id, resection pass), so a task draws the same samples on any thread at any time
  (`CHESHIRE_SFM_TASK_SEED=0` restores the shared generator).
* **Key order, not address order, in Ceres.** Landmark blocks live in one contiguous array in key
  order (`OrderedBlocks`), and every pose, rig sub-pose, intrinsic and distortion block gets its own
  ordering group numbered by key (poses from 1, rig sub-poses from 1e6, intrinsics from 2e6,
  distortions from 3e6); the Schur solvers only need the first group to be the points, so the
  groups above it are pure order.
* **A total order** in the next-best-views ranking, where `std::sort` on the score alone broke
  ties by the order threads finished in.

What that leaves is Ceres' own multi-threading: with `num_threads > 1` the Schur eliminator adds
each chunk's contribution to the reduced matrix in arrival order. `CHESHIRE_SFM_DETERMINISTIC=1`
therefore runs bundle adjustment on one Ceres thread (`CHESHIRE_BA_THREADS=n` sets it
explicitly) while SfM's own OpenMP loops keep every core, because the three points above make them
order-independent.

**The ladder, 6 views, `--json` output (the Alembic file carries the date it was written, the JSON
does not), SHA-256 of `sfm.sfm` and `cameras.sfm`:**

| runs | sfm.sfm | cameras.sfm | landmarks |
|---|---|---|---|
| all single-threaded, twice | `c468440e1cf81fc3`, `c468440e1cf81fc3` | `2dc4f6b3b9ed671d`, same | 9294, 9294 |
| `CHESHIRE_SFM_DETERMINISTIC=1`, 12 OpenMP threads, twice | `c468440e1cf81fc3`, `c468440e1cf81fc3` | `2dc4f6b3b9ed671d`, same | 9294, 9294 |
| default (12 threads everywhere), twice | `760d02d78b7a64e5`, `0ea1f5a1f1af676a` | differ | 9294, 9294 |
| shared generator (`CHESHIRE_SFM_TASK_SEED=0`), Ceres 1 thread, 12 OpenMP threads, twice | `0cf9f2c162b6ab89`, `6404929cef7fb28e` | differ | 9298, 9292 |

Byte-identical across the two single-threaded runs (the address-order fix), byte-identical across
the two deterministic multithreaded runs, and identical *between* the two rows - the parallel loops
now produce exactly the single-thread result. The default differs only by Ceres' threads, and the
shared generator is the row that changes the landmark count, so it was the larger of the two.

**41 views:**

| runs | sfm.sfm | cameras.sfm | landmarks / RMSE | SfM (the node's own "took") | BA total |
|---|---|---|---|---|---|
| `CHESHIRE_SFM_DETERMINISTIC=1`, 12 OpenMP threads, twice | `1321c85863ce62fe`, `1321c85863ce62fe` | `b45f6f46859ae678`, same | 80,808 / 1.23415 | 115.3 s, 111.9 s | 73.6 s, 71.2 s |
| all single-threaded (`--maxCoresAvailable 1`) | `1321c85863ce62fe` | `b45f6f46859ae678` | 80,808 / 1.23415 | 163.0 s | 70.2 s |
| default (12 threads everywhere) | `a55815d5fc24375f` | `d0952dddaef546ad` | 80,808 / 1.23415 | 64.3 s | 23.2 s |

Identical again, and again identical to the single-threaded result. The default run reached the
same landmark count and RMSE to six digits - Ceres' thread order moves the last bits, nothing a
user would see - and it is the reference for the cost: the deterministic mode is bundle adjustment
on one Ceres thread, 72 s against 23 s here, so the node takes 1.8x as long (112-115 s against
64 s); fully single-threaded it is 163 s. The JSON export the comparison needs costs 17.5 s at
this size against 0.2 s for Alembic (`cameras.sfm`, always written and small, plus the landmark
count is the cheap first check). The per-task generators and the key-ordered blocks are on in
every mode, and cost nothing measurable: two default runs took 64.9 s and 66.1 s of SfM against
69.8 s and 70.4 s for two runs with the shared generator restored (`CHESHIRE_SFM_TASK_SEED=0`),
on a box that was getting busier as the four ran (residual evaluation, the same code in all of
them, drifted from 55 to 61 ns per residual-block-iteration; the 57.9 s runs before step 5n were
on the idle box at 49 ns). If anything the shared generator is the slower one - twelve threads
writing the same 2.5 KB of generator state - but the drift is of the same size as the difference.

`scripts/sfmbench.py run <set> --tag X --json CHESHIRE_SFM_DETERMINISTIC=1`, twice, and equal
digests, is the gate for every later SfM change; it replaces the n=10 repeats docs/17 needed.

## 0.3.4: what a bundle adjustment costs around Ceres' Solve (2026-09-23)

The 5i profile is Ceres' own clock and stops at `Solve`. Step 5o times the rest of
`BundleAdjustmentCeres::adjust` (`CHESHIRE_BA_PROFILE=1` now prints a second line per solve,
`cheshire: BA adjust: build ... solve ... update ... destroy`), and the first run said where the
"other" was. 41 views, 68 solves, 4.42 million residual blocks in all, the box busy (residual
evaluation 62 ns per residual-block-iteration against 49 idle):

| phase | total | per residual block | what it is |
|---|---|---|---|
| build | 9.2 s | 2090 ns | `createProblem`: a cost function, a `std::vector` of four pointers, nine `std::map` lookups, four ordering inserts and Ceres' `AddResidualBlock` per observation |
| log-only evaluations | (skipped) | | upstream evaluates every landmark residual before and after the solve, single-threaded, for the two `landmarksBlocks cost` log lines; off unless `CHESHIRE_BA_LOG_COST=1` |
| Ceres preprocessor | 5.5 s | 1253 ns | program reordering and the evaluator's block structure, per `Solve` |
| Ceres minimizer | 20.4 s | | residuals 0.9 s, Jacobians 8.8 s, linear solver 7.9 s, trust-region bookkeeping the rest |
| update | 0.2 s | | the solution back into the SfMData |
| destroy | 2.1 s | 476 ns | the Problem's destructor: every cost function and residual block freed |

So building and tearing down the problem cost more than evaluating its Jacobians. Two steps on it:

* 5p: a block enters the Ceres ordering once (upstream did four `AddElementToGroup` calls per
  observation, each a `std::map` find over every block, nearly all on a block already there).
  **No measurable change** (2160 ns on a busier box): the ordering was never the cost.
* 5q: the per-view part of the lookup - view, pose state, the three block pointers, the
  intrinsic object, the ordering - is done once per view per solve through an `unordered_map`
  filled on first use; the landmark enters the ordering once; the four pointers go through
  Ceres' array overload instead of a heap-allocated vector; and the problem is built with
  `disable_all_safety_checks` (the sort and duplicate scan of the pointers per residual block; the
  problem is well-formed by construction).

| Jacobians | build, before | build, after 5q | box (residual ns) |
|---|---|---|---|
| stride (default) | 2160 ns per block (9.55 s) | 1486 ns (6.57 s) | 68 -> 58 |
| analytic | 1811 ns (8.01 s) | 1177 ns (5.20 s) | 56 -> 49 |

Corrected for the box drift that is 19 % and 27 % off the build. What is left, 1.2-1.5 us per
residual block, is Ceres' `AddResidualBlock` (its own allocations and hash lookups) and the cost
function's allocations - the analytic cost function is one object holding the intrinsic part by
value, the autodiff one is a functor, an inner cost function and a wrapper - and the destroy is
their mirror image. Only a Problem that lives across solves would remove those, and Ceres would
still run its preprocessor per `Solve`, so the ceiling of that redesign is build + destroy, about
1.9 of the 7.5 us every residual block costs per solve here (a quarter of bundle adjustment, a
tenth of the node). It is on the roadmap with that number, not in the code.

The deterministic gate after these steps: the 6-view and 41-view runs with
`CHESHIRE_SFM_DETERMINISTIC=1` reproduce the digests recorded above exactly (41 views:
`1321c85863ce62fe` / `b45f6f46859ae678`), so none of 5o-5q changed a number.

## 0.3.4: a bundle-adjustment Problem that lives across solves (2026-09-23)

Step 5r (`hip/port/sfm_ba/persistent.inc`). The engine keeps one `BundleAdjustmentCeres` for the
whole reconstruction (`ReconstructionEngine_sequentialSfM::_cheshireBA`), and each `adjust()`
applies the delta to a `ceres::Problem` that is never torn down: a merge walk of the scene's
landmarks against the records adds residual blocks for new landmarks and observations, removes
them for gone ones, copies values in, toggles constant/variable; ignored landmarks lose their
residual blocks and ignored poses go constant (upstream leaves both out of a fresh problem; Ceres
drops blocks without residuals and residuals whose blocks are all constant from the reduced
program, so the solver sees the same problem). Rigs, survey points, 2D and point constraints,
rotation priors, temporal smoothness, depth observations and mesh-referenced landmarks fall back
to the rebuild; `CHESHIRE_BA_PERSIST=0` restores it.

Two upstream traps on the way, both invisible while the problem was rebuilt every solve. The
intrinsic block vector was assigned `intrinsicPtr->getParameters()`, a temporary, so it was
move-assigned a fresh buffer every solve under the pointer the Problem held: one extra parameter
block per intrinsic per solve, then residuals evaluating freed memory (an access violation, or
Ceres' "Map key not found" when the stale block's index no longer existed). It is copied in place
now. And a kept observation whose view had lost its pose since the residual was created would
dangle once the pose block went; kept observations are re-validated.

**The invariant, checked.** `CHESHIRE_BA_PERSIST_CHECK=1` recomputes after every sync the set of
(landmark, view) pairs the scene says should have a residual block - the non-ignored landmarks'
observations from posed, non-ignored views - and compares it with the records, and asks Ceres for
the residual ids it holds per landmark block and compares those with the records too; before
each sync it checks the ids again, so a change between solves would show. Consistent on every
solve: 13 of 13 checks on 6 views, 96 of 96 on 41, and **1036 of 1036 on the 884-view False
Door** - the first set above 100 poses, so the first with local bundle adjustment and its
ignored/constant states (961 solves, 815 poses; the landmark slab, reserved at 1 M, overflowed
once at 1.34 M landmarks and the problem was rebuilt, so the minimum is 2 M now). The
deterministic gate: two runs byte-identical on 6 views (`40d4bbb85c8d521f`) and on 41
(`50759f7d5b5e2eb3`); the digests differ from the rebuild's, since the residual and landmark-block
order is creation order rather than key order, which is the order of floating-point sums.

**Cost, 41 views, rebuild against persistent back to back, twice each** (the per-solve line of
`CHESHIRE_BA_PROFILE=1`; the second persistent run overlapped the start of a Linux build):

| | build | destroy | Ceres preprocessor | update | SfM wall |
|---|---|---|---|---|---|
| rebuild (`CHESHIRE_BA_PERSIST=0`) | 6.54 s, 6.43 s | 1.87 s, 1.71 s | 5.97 s, 5.84 s | 0.16 s | 65.3 s, 64.5 s |
| persistent | 2.40 s, 2.31 s | 0.09 s, 0.08 s | 8.25 s, 7.40 s | 0.29 s | 63.8 s, 60.0 s |

Build 6.5 s to 2.4 s (the sync still walks every landmark and adds the new observations) and
destroy 1.8 s to 0.1 s, against Ceres' preprocessor 5.9 s to 7.8 s: it reorders and scans the
residual blocks every `Solve`, and in a persistent problem those objects were allocated at
different times and shuffled by removals, so its scans miss cache more. Net about 4 s of 65 s at
41 views, 6 % of the node - the ceiling of a quarter of BA was optimistic, as the preprocessor
does not shrink with the build. The False Door's build and preprocessor per residual block are
not measured yet without the check (the check is O(observations) per solve and sat inside the
build figure); the rebuild-against-persistent pair there is queued.

### The persistent problem at 884 views, and where it is switched off (2026-09-23, later)

The False Door pair without the check, persistent then rebuild, one run each (box slower during
the persistent run: residual evaluation 58 against 49 ns per residual-block-iteration):

| | main solves | active residual blocks, summed | build | Ceres preprocessor | destroy |
|---|---|---|---|---|---|
| persistent | 419 | 70.8 M | 177 s (2492 ns per active block) | 146 s (2066 ns) | 2 s |
| rebuild | 418 | 78.9 M | 177 s (2249 ns) | 121 s (1536 ns) | 45 s |

Box-corrected the build is a wash and the preprocessor 12 % worse; only the destroy is a clear
win. The reason is the local strategy: the ignored region moves with the frontier every solve, so
the sync added 23.6 million residual blocks and removed 18.4 million over those 419 solves
against 70.8 million active in all - most of the active set is torn down and rebuilt anyway - and
Ceres' preprocessor scans every residual block in the Problem per `Solve`, not the active ones.
Both paths also pay about the same to walk every landmark each solve (1.35 million, mostly
ignored, per solve), which is why the rebuild's build is 2.2 us per active block here against
1.2 us at 41 views; that walk is a shared inefficiency, noted on the roadmap. So the engine sets
`CeresOptions::cheshirePersist` to "local strategy off": the Problem lives across solves while
every landmark is active (up to 100 poses, the 6 % above) and is dropped when the strategy
switches on; the engine bay, 107 poses, crosses that line mid-run and its check run stayed
consistent through the switch (168 of 168). The sync also keeps each landmark's block pointer in
its record now - a `std::map` lookup per landmark per solve was most of its cost at this size.

## 0.3.4: the inner projection fused (2026-09-23)

`CostIntrinsicsProject` computed the projection and its three Jacobian blocks by four separate
walks of the same chain - `project()`, then the derivative with respect to the intrinsics, the
distortion and the point - each recomputing P = X/z and the distortion polynomial through virtual
calls, and three of them returning dynamic Eigen matrices, i.e. heap allocations per residual
block per Jacobian. `CheshireIntrinsicsProject` (in `hip/port/sfm_ba/projectionCheshire.hpp`)
walks the chain once for a pinhole with no distortion or a radial K1 / K3 / Brown one - what
incremental SfM sees - straight into Ceres' row-major blocks, with upstream's formulas in
upstream's order of operations where it has one, and contraction off (`#pragma clang fp
contract(off)`, as in every port): with contraction on, 20 % of the residuals differed from
upstream by up to 3e-13; with it off, under 1 % differ, by up to 6e-13, a different association
somewhere in Eigen's evaluation rather than a formula. Fisheye, 3DE and undistortion models keep
`CostIntrinsicsProject`, as does `CHESHIRE_BA_FUSED_PROJECTION=0`.

41 views, same box, same day: Jacobians 5.5 s to 3.2-3.5 s, bundle adjustment 24 s to 19-21 s,
the node 59 s to 50-52 s. The check against upstream's autodiff over 3.18 million evaluations:
Jacobians differ by at most 1.4e-12 absolute (4.5e-11 relative on the small sets), as before the
fusion; the deterministic pair is byte-identical (`15618f943864c8ba`).

## 0.3.4: the passes after every bundle adjustment, restricted to what the solve touched - and a retraction (2026-09-23)

Upstream follows each bundle adjustment with `removeOutliers()` - the pixel-residual test over
every observation of every landmark, then the angle test over every landmark - and with
`eraseUnstablePosesAndObservations()`, a recount of every observation per pose. At 884 views that
is 64 such passes (one per resection group) over a scene that reaches 1.35 million landmarks, for
solves that under the local strategy moved about a hundred poses.

Step 5s (`hip/port/sfm_ba/postAdjust.inc`) restricts them, exactly. The tests depend only on
(pose, intrinsics, landmark position, observation), so an observation whose inputs did not change
since its last test keeps its verdict. After a solve the changed ones are the observations of the
landmarks seen by a REFINED pose - which covers every refined landmark (a landmark is refined only
when a refined camera sees it), everything triangulation added since the last pass (it hangs off
the new, refined, views), and the landmarks the local strategy ignores because a refined and an
ignored camera both see them (their refined observer moved) - and the landmarks observed through
a REFINED intrinsic, whose parameters apply to every view that uses it, active or not. The
candidates come from those views through the tracks-per-view index, never from walking the
landmarks. A pose can only have become unstable by losing observations, so only the poses of views
that lost one, and the poses resected in the group, are recounted, through the same index; if a
pose does go, upstream's full pass finishes the iteration. Without the local strategy every pose
is refined and upstream's passes run unchanged.

That intrinsic source matters more than it sounds. Upstream's local strategy turns an intrinsic
constant only once its focal length has been stable across a window of 25 posed views, so an
intrinsic with fewer views than that stays refined for the whole reconstruction. The False Door
has five: the 820-view one settles at 113 poses, the other four (1, 13, 23 and 27 views) never do.
The first cut of this step took a full pass whenever an intrinsic was refined - on this set that
was every pass, and the deterministic run said so (0 restricted passes by view 315) - so the rule
became the one above: a refined intrinsic's views are candidate sources like a refined pose's.

**Exactness, measured with the deterministic gate** (same input, one Ceres thread, digests of
`sfm.sfm` and `cameras.sfm`; the before-digests were taken with the kept pre-5s install, the
after-digests with the restricted passes on). The engine bay, 107 poses, crosses into the local
strategy for its last three solves, which ran restricted - 137,867, 29,238 and 28,013 candidates
of about 138,000 landmarks, 37, 6 and 10 poses recounted, 0 refined intrinsics - and ended with
the same digests as the full passes (`a76946037b3860e6` / `38450e663421b3e4`, 140,490
landmarks). The 41-view set never enters the local strategy and is unchanged
(`15618f943864c8ba`). The 884-view set, 833 poses and 1,355,427 landmarks, 64 passes of which 34 ran restricted
(4 refined intrinsics throughout; 25 % to 60 % of the landmarks each, 6 to 30 poses recounted):
the same digests as upstream's passes, `60b2841c14e21182` / `e0984f42c3f2512e`, RMSE 1.36595.

**What it is worth: a retraction.** The first version of this section, and the commit message
that introduced the step, said these passes were 605 s of a 1908 s node at 884 views. That
figure summed the log's silences after every `adjust end`, and most of it was a single silence
of 494 s - the triangulation phase of one resection group, while a build on the same box was
starving the run - not the passes. Measured directly, upstream's full passes cost about 40 s over
the whole 884-view run (0.6 s at 300,000 landmarks, 1.2 s at 590,000, under 3 s at 1.35 million)
and the restricted ones 43 s (34 passes of 25 % to 60 % of the landmarks each: at this size the
refined frontier is still a large share of the scene, and a candidate list costs more per
landmark than a sequential walk). So at 884 views the step is neutral, and it ships **opt-in**,
`CHESHIRE_SFM_LOCAL_PASSES=1`. Its case is scale - a full pass grows with the scene, a restricted
one with the frontier, and the local strategy exists for reconstructions of thousands of views -
which is unmeasured here (the 1678-view set is no longer on disk). What the mining did settle,
on a quiet box, is where the time outside Ceres goes at 884 views: nowhere in particular. The
largest silences are the solves themselves; the per-solve walk over every landmark inside the
bundle adjustment's build - 177 s over the run, the roadmap item this step was mistaken for -
remains the open one.

## 0.3.4: the depth-map node was loading images, not computing (2026-09-23)

The 884-view job on house-pc (i3-4330, RX 6750 XT) spent 6.5 h in DepthMap at 28 s per view, and a
minute of the card's busy counter read 0 for 55 of its 60 seconds. Splitting a 12-view chunk by its
log timestamps and the output files' write times:

| per 12-view chunk | house-pc | RX 9070 box |
|---|---|---|
| chunk setup | 19 s | 27 s |
| image loading, first batch | 48 s | 30 s |
| image loading, each later batch of 3 views | 64 s | 34 s |
| GPU tiles (SGM + refine), per view | 4.3 s | 4.6 s |
| writing the depth and sim maps, per batch | 2 s | 1 s |
| whole chunk | 325 s | 185 s |

The kernels take about 4.5 s per view on both cards. Everything else is the CPU getting the
6000x3376 half-float EXRs that PrepareDenseScene wrote, 73 MB each, into the host image cache -
and the disks are not the limit (both are SATA SSDs at 250-385 MB/s; a batch's 2.2 GB reads in
under 10 s). It took four wrong explanations to find the right one; they are kept here because
each was plausible from the outside, and the method that settled it was not inference but a
per-file profile inside the node.

**1. A chunk's cameras were unrelated.** The camera index order is the SfM view-id order, which
is hash-like, so a 12-view chunk held 12 cameras from all over the scene: the debug run loaded
117 images for 12 views, 114 of them distinct. Nothing was shared between consecutive R cameras
or consecutive batches, and the device cache, sized for one batch, never had a hit. Step 5u orders
the cameras as a nearest-neighbour tour over their centres (greedy, ties by index, so the tour is
the same on every machine) and a chunk is a slice of the tour; the estimator's batch slots and
its write loop now go by position in the tile list, since upstream assumed a batch's cameras are
consecutive indices (the first tour run wrote 514 maps for a 12-view chunk through that
assumption, and two cameras of a batch could share a slot). Outputs are per view id and do not
depend on the grouping. `CHESHIRE_DEPTHMAP_ORDER=0` restores the index order.

**2. Every camera of a batch was loaded, whether or not the device had it.** The prefetch loaded
the R and all T cameras of the batch, including the ones the device cache still held from the
previous batch. Step 5t touches the resident cameras first, so the device LRU keeps them, then
loads only the missing ones in parallel, in groups no larger than the host cache (61 images at
the working resolution: 5000 MB over 81 MB each), uploading group by group. One line per batch
says what happened: `cheshire: depth map batch 2/3: 23 cameras, 10 decoded and uploaded (13
already on the device)`. On the same chunk in tour order: 42 loads instead of 117. (An earlier
draft of this section said the host cache held 16 images and thrashed; that used the full
resolution, and it was wrong - the cache is sized after the downscale below.)

**3. The reader.** OpenImageIO's `ImageBuf::read` of one of these files takes 1.8-2.0 s on the RX
9070 box whatever its thread setting; the same file through `Imf::InputFile` with the OpenEXR
thread pool takes 0.3 s, and the pixels are identical (half to float is exact; a full-image
check: max difference 0). Step 5v reads an EXR straight through OpenEXR when it is what these
nodes read - R, G, B and optionally A, half or float, full data window, requested as float RGB or
RGBA in the colour space it is stored in - and leaves everything else to OpenImageIO. The reader
is its own translation unit of the image library because `main_cameraInit.cpp` includes `io.cpp`
directly and cannot see OpenEXR's headers; inside a parallel loop it decodes each file in its own
thread, in blocks of 64 scanlines, and a lone read keeps the pool. `CHESHIRE_EXR_DIRECT=0`
restores the OpenImageIO path; `CHESHIRE_EXR_PROFILE=1` prints one line per file with the open
time, the read time and the thread's CPU time. That profile is what found the real cost:

**4. The host downscale after every read.** With the node's `downscale 2`, `loadImage` resizes
each image on the host from 6000x3376 to 3000x1688 through OpenImageIO's `resize` right after
reading it, inside the prefetch's parallel loop. The reader took 0.7 s per file and ran twelve
wide; the resize took the rest of the 10-12 s per batch, because each of the twelve threads
handed OpenImageIO its own twelve workers and 144 threads thrashed - 100 s of CPU for 11 s of
wall, no less wall than a single thread. Step 5w gives `resize` one thread when the caller is
already inside a parallel region; the result does not depend on the thread count - and it
changed nothing, which says the resize is expensive by itself, not through its threading: OpenImageIO
evaluates a 25 x 25-tap footprint per output pixel for a 2x lanczos3 downscale (169 non-zero taps),
about 7-9 s of CPU per 20-megapixel image. Twelve images at a time or twelve threads on one image,
the CPU is the same. The next step is an exact port of that resize for the integer-downscale case,
same weights and summation order, with the footprint's weights computed once instead of per pixel;
until then the depth-map node's loading is bounded by it. (Three explanations tried before the
profile - a serialised page-fault path, the shared OpenEXR pool,
and the runtime environment of the node - were each refuted by a standalone reader that decoded
the node's own 16 files in 0.83 s with the same libraries, and by a benchmark that stayed fast
with the HIP runtime and pinned memory in the process.)

**5. The downscale itself, exact and 18x cheaper.** `ImageBufAlgo::resize` with the default
filter is lanczos3 with a width of 6 destination pixels; for a 2x downscale that is a 13 x 13
footprint per output pixel, evaluated tap by tap with the filter function called per tap, which
is where the 7.3 s of CPU per 6000x3376 image went (single-threaded; the same total across
threads). Step 5x (`hip/port/sgm_fused/cheshireResize.hpp.txt`) computes the same thing with
the same arithmetic in the same order - OpenImageIO's coordinate mapping, its lanczos3 from
`libutil/filter.cpp`, the per-column and per-row normalised weight tables, rows outer and
columns inner, a multiply then an add per tap with contraction off, the same clamping at the
borders - with each column's and row's clamped tap indices and non-zero weights computed once
instead of per pixel, four output pixels' add chains interleaved, and the channels added with
SSE. Checked against `ImageBufAlgo::resize` itself in `build/exrbench/exrresize.cpp`: 0 of
20,256,000 values differ on a real image, 0 on odd-sized random images at 2x and 3x. On the RX
9070 box: 7.32 s to 0.41 s on one thread, 0.096 s with the rows in parallel. It applies to float
images of 1, 3 or 4 channels downscaled with the default filter; anything else keeps the
OpenImageIO call, as does `CHESHIRE_RESIZE_EXACT=0`.

**Measured on False Door chunk 0 (12 views) on the RX 9070 box:**

| | loads | image loading | chunk wall |
|---|---|---|---|
| before (index order, old loader, OpenImageIO) | 117 | 25-34 s per batch | 214 s |
| tour order + once-per-batch loader (OpenImageIO reader) | 42 | 16, 8, 10 s per batch | 142 s |
| + direct OpenEXR reader | 42 | 15, 8, 10 s per batch | 142 s |
| + resize single-threaded inside the loop | 42 | 16, 8, 9 s per batch (unchanged: the resize is the cost, not its threading) | 146 s |
| + the downscale computed directly (5x) | 42 | 4.7, 2.4, 2.8 s per batch | 94 s |

**Exactness.** The 12 maps of the index-order chunk are byte-identical to the reference cache
with the new loader; the tour-order chunk's 12 views, compared by view id, are byte-identical to
the reference (24 of 24 files) with the direct reader; the mini6 depth maps with the reader off
and on are byte-identical (12 of 12); the mini6 texturing output with the reader off and on
differs in 522 texels by at most 0.000977, which is the node's own run-to-run variation (two
runs with the reader off differ in 556 texels by the same amount; the OBJ is identical).

**6. What PrepareDenseScene writes.** The writer's default for the undistorted images is ZIPS,
one zlib stream per scanline - 3,376 of them per 6000x3376 image - and every later read inflates
them: the depth-map node loads each image about 3.5 times over a job, texturing once per atlas
sheet. Step 5y has PrepareDenseScene write ZIP, sixteen scanlines per block: the same pixels, a
slightly smaller file, and a cheaper write and read. On the local mini6 job (6 images, 4032x3024):
283,096,208 bytes with ZIPS, 278,680,274 (1.6 % smaller) with ZIP; PrepareDenseScene 1.4 s against 1.3 s (the node is not where the time goes at six images); the decoded
pixels are identical in all 6 files (channels compared value by value); the depth maps computed from each set are byte-identical (12 of 12 depth and sim maps); the direct reader spends 16 % less CPU on the ZIP files (1.22 against 1.45 s for the six).
`CHESHIRE_PDS_EXR_COMPRESSION` names another method (the lossy ones change the values and are for
experiments only). Uncompressed was considered and rejected: 162 MB per image instead of 73, a
143 GB cache for the 884-view job on a SATA SSD, and the depth-map node's reads would be bound by
the disk instead of scaling across the threads that inflate.

**7. Fewer chunks.** Each Meshroom chunk is a separate process: it loads the SfM data with its
1.35 million landmarks, probes the device and starts with a cold image cache - 19-27 s before
its first tile at 884 views, plus a cold first batch - and Meshroom's block of 12 views makes 74
of them. Step 5z makes it 48, so 19. For Meshroom 2025 pairings that is the block size in the
AliceVision-provided node; for Meshroom 2023.3, whose nodes are compiled `.pyc` files, the
package carries `share/cheshire/meshroom-overrides/DepthMap.py`, a module that loads the
compiled node and re-declares it with the larger block, and the pairing scripts copy it beside
the `.pyc` (Python prefers the `.py`); `--unpair` removes it. Same attributes, same command line,
same UID, so an existing cache stays valid. `CHESHIRE_DEPTHMAP_BLOCK` sets another size for the
2023.3 override, 0 for Meshroom's 12.

**The 884-view job on house-pc, before and after (2026-09-24).** The same photographs, the
same standard preset, on the i3-4330 with the RX 6750 XT: the s1 bundle (the 0.3.3 release plus
the knn pragma) against the s7 bundle (everything above, steps 5m through 5z).

| stage | s1 | s7 |
|---|---|---|
| FeatureExtraction | 7.9 min | 9 min |
| FeatureMatching | 21.0 min | 21 min |
| StructureFromMotion | 77.9 min | 32 min |
| PrepareDenseScene | 37.5 min | 40 min |
| DepthMap | 395.0 min | 143 min (19 chunks) |
| DepthMapFilter | 16.2 min | 17 min |
| Meshing | 12.6 min | 12 min |
| Texturing | 220.6 min | 250 min (17 passes of 14.5 min) |
| whole job | 13 h 11 min | 8 h 46 min |

SfM did better than the local numbers predicted, since the four-thread box was even more
Jacobian-bound. PrepareDenseScene did not gain: ZIP is a read-side change, and the sixteen-line
blocks cost the i3 slightly more to write. Texturing's pace did not move either, which says a
pass's time is not in the per-camera reads there; that node's profile is the next measurement.

What is left in the node after this is the first batch of each chunk, which is cold by
construction, and the downscale, now a fraction of the read: the level stored in the EXR at
PrepareDenseScene time would remove it from the node entirely with the values unchanged, and a
2x2 average on the device would remove the read bytes too, at the price of changed values and a
new reference.

## 0.3.4: texturing was loading images too, and the depth-map downscale moves to the device (2026-09-24)

**Texturing is load-bound.** The same per-phase view that found the depth-map cost, on the False
Door's texturing (884 views, RX 9070 box, `CHESHIRE_GPU_TEX_LOG=1`): every pass re-reads the
cameras that contribute to its atlases, and the loads were 443 s of the first 486 s pass and
600-640 s of each later 11-12 minute pass, against 30-90 s for the upload, pyramids and
rasterisation together. Step 6a adds `CHESHIRE_LOAD_PROFILE=1`, one line per image load with its
parts. A load averaged 2.01 s: the read 1.79 s, the conversion to sRGB 0.18 s (OpenImageIO resolves
"linear" and "srgb" through OpenColorIO's built-in config, whose fast approximation matches the
textbook formula in only 1 % of values, so it stays), the exposure multiply 0.04 s and the second
open for the metadata 4 ms. The read was 0.1-0.25 s for the first loads and about 3 s once four
read-ahead threads were decoding at once: the texturing node reads ahead through `std::async`,
outside any OpenMP region, so every read went through OpenEXR's one shared pool, which serialises
concurrent files - the depth-map finding of 5v again.

Step 6c marks the image cache's read-ahead threads (a thread-local `ConcurrentLoadScope`) and the
direct reader decodes on such a thread instead of the pool. First pass 486 s to 243 s, loads 443 s
to 204 s, a read 1.79 s to 0.88 s. Step 6e reads ahead on every hardware thread but one, within the
image cache's slots. It did not speed this box up (257 s with eleven reads in flight): a pass reads
829 images of 73 MB, about 60 GB, and 60 GB in 214 s is 283 MB/s, the read rate of the SATA SSD the
cache sits on, so texturing here is now disk-bound. It matters on house-pc, whose available RAM
missed the 4 GB margin of the read-ahead rule and so read one camera ahead, with the image cache's
20 default slots allocated anyway; it now reads three ahead at no extra memory. Gate: the mini6
texture differs from the pre-6c run in 557-569 texels by at most one half-float step, the same as
two runs of the same build (556); the OBJ is identical.

**The depth-map process downscale on the device (6d).** Even exact and 18x cheaper (5x), the host
lanczos3 downscale was the larger half of a depth-map load's CPU, and loads were about half of each
chunk on house-pc's i3. With 6d the image cache keeps full-resolution floats, `DeviceCache::
addMipmapImage` uploads them, and `imageProcessing/cheshireDownscale.cu` applies the tap tables
computed on the host by the same code, in the host's order, then the host's half(value x 255)
conversion. The first build differed from the host path in every file of the gate, at the rounding
level (99.7-99.9 % of pixels within 0.5 %), and a check mode found two fusions the source did not
ask for. `__fmul_rn` and `__fadd_rn` on HIP come from the device library's bitcode with the
contraction flag set, so the multiply-adds became FMAs: 32 % of the accumulated floats differed.
And the GPU fused `value * 255` and the conversion to half into one mixed-precision instruction
with a single rounding of the exact product, where the host rounds to float first: at a half-float
tie such as 0.417279422 x 255 = 106.40625261 the host gets the tie 106.40625 and rounds to even
(106.375), the device 106.4375 - 524 of 3 million texels. An empty `asm volatile` on each product
is a barrier neither fusion can cross. With it: device floats and texels 0 of 12.2 million and 0 of
3 million different from the host path on mini6, **the 884-view chunk's 24 files byte-identical to
the reference, mini6 identical with the switch off and on.** On this box the batch loads drop by
about 40 % (4.4, 2.0, 1.6 s against 5.7, 3.0, 3.6 s) but the chunk barely moves (116 against
120 s): DepthMap here is now GPU-bound. It is for the i3. HIP builds only; the CUDA backend keeps
the host path. `CHESHIRE_DEPTHMAP_DEVICE_DOWNSCALE=0` restores the host path and
`CHESHIRE_DEPTHMAP_DEVICE_DOWNSCALE_CHECK=1` compares every image with it.

**Smaller items in the same batch.** The direct reader takes one-channel EXRs (depth and similarity
maps; DepthMapFilter and Meshing read them through OpenImageIO before): DepthMapFilter's twelve
mini6 files identical with the reader off and on, and Meshing's tetrahedralization input identical
(265,271 points, the same checksum) - the cells then differ as they do between any two runs
(geogram's numbering, a parked item). On house-pc the depth-map load was the largest single step of
the 884-view Meshing, 120 s of 716. Step 6b computes F from E once per model in
`RelativePoseKernel_K::errors()` instead of once per correspondence (SfM's initial pair): the
deterministic digests are unchanged on the 41-view set and the engine bay. And step 5f counted the
native camera mipmaps at half their size - `CudaRGBA` is already the 8-byte half4 - so mini6's image
peak reads 186 MB instead of 93, with the maps unchanged.

## 0.3.4: Meshing's host phases after the s7 run, and PrepareDenseScene at ZIP level 1 (2026-09-24)

**Where the 884-view Meshing goes on house-pc.** The s7 run's Meshing (i3-4330 with 4 threads, RX
6750 XT, 819 of 884 views registered) took 716 s, split from its log:

| phase | s |
|---|---|
| depth-map load, first filters, first kd-tree | 131 |
| visibility pass 1 | 153 |
| max observation angle per point | 48 |
| angle and similarity filter, pixel-size filter rounds | 23 |
| visibility pass 2 | 143 |
| tetrahedralization | 29 |
| s-t weights (GPU votes) | 32 |
| facet weights and graph | 33 |
| CSR layout and GPU cut | 13 |
| post-cut processing | 24 |
| mesh cleaning | 58 |
| mesh save | 19 |

The two visibility passes are 296 s, 41 % of the node, at 187 ms per camera where the RX 9070 box
runs 62 ms: the i3 backprojects each camera's 3.76 million pixels and applies its votes on four
threads, and only the knn search overlaps them (the next item). The rest of this section takes the
host phases that are pure CPU and can be made exact: each is proven in-process against upstream's
own computation, and at scale by the tetrahedralization input checksum of the kept False Door
cache (4,701,419 points, `64b36ee445e30d38`).

**6f. PrepareDenseScene at ZIP level 1.** Step 5y's ZIP was written at OpenEXR's default zlib level,
4. Level 1 on the mini6 set: 282,699,910 bytes against 278,680,274 (1.4 % larger), the write's
thread-seconds 4.0 against 5.7, the decoded pixels identical in all 6 files. Inflate costs the same
at either level, so every later reader is unchanged. `CHESHIRE_PDS_EXR_COMPRESSION` now takes
`method[:level]`; a tree patched before this step is upgraded in place.

**6g. The max observation angle over pairs of cameras.** For every fused point, upstream takes the
largest angle between any two of its cameras, over ordered pairs: two normalisations and an acos per
pair, twice per pair. The angle is symmetric bit for bit (two separate normalisations and a dot
product of the same products in the same order), it is never NaN (0 is returned instead), and the
angles are non-negative, so the maximum over unordered pairs is the same value. Each camera's
direction from the point is normalised once with the same expression, and the angle uses
angleBetwV1andV2's: half the acos calls and k normalisations instead of 2k(k-1). Locally the loop went
from 5.5 s to 3.4 s with the pairs alone and to 1.9 s with the directions too; the tetrahedralization
input is unchanged on mini6 and at 884 views. Upstream's count of the points this filter removes is a
plain int incremented inside the OpenMP loop, so it loses increments at random (225,329 and 225,314 in
two runs); the count of points actually removed, 225,360, is the one to compare.

**6h. Facet weights once per interior facet.** Before the cut, every cell's four entries computed
both directed weights of their facet with getFaceWeight, and each weight takes the circumsphere
centres of the facet's two cells: 16 centre computations per cell, every interior facet twice. The two
weights of a facet need the same two centres and the mirror's entry is the same pair of values
swapped, so each interior facet is now computed once, from its lower-numbered cell, which computes its
own centre once: about 3 centres per cell. `cheshireFaceWeight` is getFaceWeight's expressions with the
centres passed in. The existing check (`CHESHIRE_GPU_VOTE_LOG=1`) compares every entry with upstream's
per-facet computation: 0 of 6,689,240 facets differ on mini6 and 0 of 116,323,696 at 884 views. The phase
went from 7.7 s to 5.3 s locally (4.6 s of it the weights, the rest the serial edge recording); on
house-pc, where four threads compute it, it was 33 s. `CHESHIRE_FACET_PAIRS=0` restores
upstream's loop.

**6i. MeshClean with a parallel pre-screen.** Mesh cleaning splits the vertices whose triangle fan is
not a single disc, one point after another in index order, and repeats until a pass splits nothing:
four passes of 8.4 s over every point on house-pc, and a first pass of 19 s. A point's pass reads only
its own triangle list, those triangles and the edge entries between it and its one-ring, and a split
rewrites only those of the split point, its one-ring and the new points (the edge index stays sorted
with unique keys, so appending to it does not move a lookup). So each pass now runs upstream's read-only
path::isWrongPt on the candidates in parallel, upstream's deployAll in parallel on the candidates that
would not split (they write only their own entries), and upstream's deployAll in index order on the
points that split and on every point an earlier split of the pass touched. After the first pass the
candidates are the points a split touched in the previous one; the others would write the same values
again. The first pass was slow for another reason: the arrays a split appends to grew by a fixed 1,000
or 3,000 entries, so the 24-million-entry edge array was copied every few hundred splits; they now grow
by an eighth. `CHESHIRE_MESHCLEAN_CHECK=1` runs upstream's passes from the same state and compares every
structure (points, triangles, colours, the triangle and neighbour lists, the boundary flags, the edge
index, the new points' origins): identical on mini6 (3 passes, 250,207 points) and at 884 views (3 passes,
4,071,677 points, 8,136,361 triangles). mini6's passes: 110, 5.4 and 0.4 ms. At 884 views the passes take 2.5, 0.11 and 0.01 s where
upstream's took 10.0, 8.5 and 8.7 s in the same process, and the cleaning step went from 44.3 s to
14.0 s, 10.5 s of it now the setup's sorts. The mesh itself differs between runs as it always has
(geogram numbers the cells differently), so a run may take a fourth pass: the check compares within one.
`CHESHIRE_MESHCLEAN_PRESCREEN=0` restores upstream's passes.

**6j. removeInvalidPoints moves the camera lists.** The variant with the vertex attributes copied each
surviving vertex's camera list into the new array, a heap allocation and a free per vertex, five times
per Meshing; it now moves them. Same contents; the checksum is unchanged.

**The device downscale on house-pc (6d, Linux).** The s8 bundle, built from 919eba8, ran 4 views of
the s7 job's first DepthMap chunk with `CHESHIRE_DEPTHMAP_DEVICE_DOWNSCALE_CHECK=1` on the RX 6750 XT:
every image 0 of 20,256,000 floats and 0 texels different from the host path, and the 8 maps
byte-identical to the s7 job's (73 s for the 4 views with the check).

## 0.3.4: the visibility passes' queries built on the device, and MeshClean's setup by counting (2026-09-24)

**6k. The backprojection on the device.** On house-pc the two visibility passes are 296 of the
884-view Meshing's 716 s: per camera the i3 backprojects 3.76 million pixels and applies the votes
on four threads, and only the knn search overlaps them. Now the host counts each row's valid pixels
and stages the depth map, and the device builds the queries: `backprojectKernel` in `knnGPU.cu`, one
block per row, the valid pixels numbered by a block scan so query k is the host's k, and
MultiViewParams::backproject and getCamPixelSize replayed operation by operation. The host build's
fused forms follow the same rule as the knn metric: clang-cl under /arch:AVX2 contracts within an
expression (fma(m11, x, m12*y) + m13, fma(z, z, fma(x, x, y*y)), fma(a.y, b.z, -(a.z*b.y)), the
4-term rows likewise), a generic x86-64 build (the Linux bundle) fuses nothing. The device answers the
queries in place and returns them with their pixel sizes, and the host votes as before.

The first build did not match: 63 % of the queries differed from the host's in the last bit, by the
same count whether the device computed the fused or the plain forms, so the device was fusing on
its own. HIP's `__dadd_rn`, `__dsub_rn` and `__dmul_rn` are plain operators defined in its math
header, which the compiler includes before a source file's first line; they are compiled with device
code's default contraction, not under the file's `#pragma clang fp contract(off)`, so a product
feeding a sum fuses into an FMA once inlined. The arithmetic now goes through three helpers defined
in the file under the pragma (division, sqrt and explicit `__fma_rn` cannot be contracted and stay).
The knn kernel used the same intrinsics for its plain metric and the bounding-box sums, which is the
likely cause of the Linux knn distance differences of 2026-09-22 that the pragma alone did not change
(docs above); the next Linux bundle will show. No other port uses them.

Gate (`CHESHIRE_GPU_VIS_CHECK=1`: the host backprojects every camera again and compares every query
and pixel size bit for bit, and its reference votes use its own queries): identical to
MultiViewParams on all 8,808,856 queries of each mini6 pass and all 88,267,121 of each 41-view pass;
knn and votes checks identical; the pass digests equal step 3's on all three sets (mini6
`806ef990.../51779edd...`, 41 `4d536979.../db94ece3...`, 884 views `c572bc16.../f0ff15ea...`), as do the
tetrahedralization inputs (`7d875321...`, `cdd23710...`, `64b36ee4...`). Negative control:
`CHESHIRE_GPU_VIS_BP_FMA=0` on this build reports 8,792,017 of 8,808,856 queries different. Under CHECK
the first mismatches of a pass are printed bit for bit with their pixel and the camera, which is how
the fusion was found.

On the RX 9070 box, back to back at 884 views, idle:

| | host backprojection | device backprojection |
|---|---|---|
| pass 1 | 57.4 s (backproject 24.0, votes 19.2) | 49.3 s (count and stage 2.8, device wait 17.5, votes 17.9) |
| pass 2 | 48.1 s | 39.9 s |
| Meshing node | 290.9 s | 278.9 s |

Here the passes were already near device-bound (62 ms per camera after step 3); the step is for
four-thread hosts, where the backprojection was more than half of the host's 187 ms per camera. The
device's own event split (upload, backprojection, kernel, download) is not reliable on HIP - copies
run on the DMA engine and the markers can land out of order, down to negative uploads - so the pass
totals are the measure. `CHESHIRE_GPU_VIS_BACKPROJECT=0` keeps the host's backprojection;
`CHESHIRE_GPU_VIS_BP_FMA=0|1` overrides the form.

**6l. MeshClean's setup by counting.** After 6i the cleaning step was mostly its setup: two qsorts
over three entries per triangle (the per-point triangle lists and the edge index) and a sort of every
list, about 10 s at 884 views. Their result is fixed: the lists ascending and the edge entries in
lexicographic (larger point, smaller point, triangle) order, whatever qsort does with equal keys
(its comparator never returns 0). Counting by point and by larger point, then sorting each small
bucket, gives the same arrays with the same capacities: 0.86-0.91 s at 884 views, 0.24 s at 41,
51 ms on mini6. `CHESHIRE_MESHCLEAN_CHECK=1` now also runs upstream's setup and compares: identical
on mini6, at 41 views (1,157,308 lists, 7,034,610 edge entries) and at 884 (4,053,812 lists,
24,409,083 edge entries), and the cleaning passes after it identical as before.
`CHESHIRE_MESHCLEAN_SETUP=0` restores upstream's.

**On Linux, the RX 6750 XT (house-pc), 2026-09-24.** The s9 bundle (built from 39e8490 in WSL, gcc
host, so the plain arithmetic forms) on the 41-view set: DepthMap and DepthMapFilter with the engine
bay job's own node parameters through Meshroom's paired wrappers, then Meshing with
`CHESHIRE_GPU_VIS_CHECK=1`, `CHESHIRE_GPU_VOTE_LOG=1` and `CHESHIRE_MESHCLEAN_CHECK=1`:

| check | pass 1 | pass 2 |
|---|---|---|
| GPU backprojection | identical on all 88,257,069 queries | identical on all 88,257,069 |
| GPU knn against nanoflann | identical on all | identical on all |
| visibility votes against the ordered host reference | identical on all 4,139,042 vertices | identical on all 1,861,491 |

and the facet weights 0 of 47,100,904 different, MeshClean's setup and its 4 passes identical to
upstream's. The knn line is the news: the 0.3.3 and step-1 bundles on the same card reported
17,383,299 and 18,404,248 of 87,354,192 distances different (docs above, "The Linux knn distances are
not a contraction"). It was the HIP intrinsics fusing despite the pragma, as 6k found; the gate
(`scripts/verify_end_to_end.py`) now requires "identical to nanoflann" in both passes, and the
backprojection, votes and MeshClean verdicts per pass. On the 107-photo engine bay the s9 bundle's
Meshing is 126 s against s8's 159 s on house-pc (visibility passes 24.3 to 18.1 s, max angle 4.5 to
1.1 s, facet weights and graph 23.7 to 12.8 s, cleaning 15.8 to 3.2 s), the job 21.1 to 20.5 minutes.

## 0.3.4: the JPEG read without OpenImageIO's full-image passes (2026-09-24)

The plan was CheshireJPG (docs/19) for PrepareDenseScene's reads, so first the read was split.
`hip/tests/pdsread/pdsreadbench.cpp` times each step `image::readImage` takes for a JPEG read as
linear RGBA float, one thread per step as the node's image threads run it (8 monstree iPhone photos,
4032x3024, RX 9070 box):

| step | ms per image | share |
|---|---|---|
| JPEG decode to 8-bit | 60 | 11 % |
| 8-bit to float (the rest of `ImageBuf::read(FLOAT)`) | 16 | 3 % |
| `ColorConfig` built for the call | 14 | 2.5 % |
| `colorconvert` sRGB to linear | 267 | 48 % |
| `ImageBufAlgo::channels` to RGBA | 152 | 27 % |
| `get_pixels` into the caller's buffer | 44 | 8 % |
| total | 556 | |

The decode is the smallest part; the time is OpenImageIO copying full float images around the
colour transform, whose OCIO processor itself takes 102 ms of the conversion's 267 (creating it is
0.1 ms). A 256-entry table per channel cannot replace the conversion: AliceVision's `sRGB` to
`linear` is the sRGB curve and then two 3x3 matrices that nearly cancel (Rec.709 to ACES2065-1 and
back), so every channel depends on all three; 98 % of the converted values are not a function of
their own 8-bit value.

**6n.** `colorconvert` works per row: a scratch line of RGBA floats (the three channels, alpha 0),
the processor applied to that line, the three channels stored back (OpenImageIO 3.0
`colorconvert_impl`). The direct path builds the same line straight from the 8-bit pixels through
OpenImageIO's own uint8-to-float values (`convert_pixel_values`), applies the same processor to it
with the same call, and stores the line in the caller's buffer with alpha 1, which is what the
channels pass would have added. The processor comes from a `ColorConfig` kept for the process
(upstream builds one per read, 14 ms), resolved as `colorconvert` resolves it. The path takes only
what it reproduces - three-channel 8-bit JPEG or PNG, read as float RGB or RGBA, no DCP profile,
converted through the AliceVision config or not converted - after the same colour-space decisions
as upstream; anything else, including grayscale reads, takes upstream's path. Rows run on
OpenImageIO's pool unless the caller is already parallel (5w's rule). In the bench: 17 ms to fill and
102 to apply, byte-identical to the node's buffer on 8 of 8 photos, so the read is about 180 ms
with the decode instead of 556.

Gate, engine bay (107 Pixel photos, 4032x2268), `aliceVision_prepareDenseScene` from the same
sfm.abc with `CHESHIRE_READ_DIRECT=0` and with the default: **107 of 107 EXRs byte-identical**.
`CHESHIRE_READ_DIRECT_CHECK=1`, which reads every image both ways and keeps upstream's result: 107 of
107 identical on the engine bay, 6 of 6 on mini6 (iPhone). On the RX 9070 box, 12 threads:

| | upstream read | direct read |
|---|---|---|
| wall | 28.1 s | 19.1 s |
| read (thread-seconds) | 139.3 | 67.4 |
| undistort | 67.9 | 60.8 |
| write | 71.6 | 56.4 |

The undistort and write phases got cheaper too: 0.3.3 item 2 found this node memory-bound on this
box, and the direct read moves four fewer full-size float images per view. On house-pc's four
threads PrepareDenseScene was 133 s on the engine bay and 2,425 s at 884 views; the next Linux
bundle measures it. What remains of the read is the decode and the transform itself, which is where
CheshireJPG comes in. `CHESHIRE_READ_DIRECT=0` restores upstream's read.

## Texturing's lone read-ahead back on the pool (step 6o, 2026-09-24)

house-pc's 884-view False Door texturing (51 atlases, 3 atlas slots in the RX 6750 XT's VRAM, so 17
passes over 835 cameras) took 14.7 minutes per pass on the s7 bundle and 32 on s9: 2.2x slower,
with the process using about 1.3 of the i3's four threads and the disk nearly idle. Both runs had
the same cache: at 6000x3376 an image is 231 MB, the host had 4.1-4.4 GB over the read-ahead's 4 GB
margin, not enough for the 5 slots of the deeper read-ahead, so texturing kept upstream's 2 slots
and read one camera ahead. What changed between s7 and s9 is step 6c: every image-cache read-ahead
decodes on its own thread instead of through OpenEXR's pool. With eleven read-aheads in flight (the
12-thread box after 6e) that is the faster arrangement; with one, it put texturing's only read on a
single thread. Step 6e's comment assumed the default cache has about 20 slots; texturing sets 2.

Step 6o counts the read-aheads in flight: one that starts while another is loading still decodes
inline, a lone one keeps the pool. Concurrency only, no value can change. The engine bay never
showed it: at 4032x2268 the images fit the deeper read-ahead.

Fold-in gate, RX 9070 (2026-09-24): main with PR #2 (CheshireEXR, off by default), 6n and 6o,
packaged flat (`package_windows.py`): the mini6 end-to-end matrix 12 of 12 at 7 of 7 ports, and the
`verify` config now also runs `CHESHIRE_READ_DIRECT_CHECK=1` and requires its verdict (6 of 6 images
identical to OpenImageIO's path).

## The bundled HIP runtime and this box since 2026-09-22

The first run of that gate failed every config: FeatureMatching crashed inside `amdhip64_7.dll` and
DepthMap reported "no kernel image is available". `AMD_LOG_LEVEL=3` gave the runtime's own reason:
"KMD failed to setup the trap handler", then "AMD HSA Code Object loading failed". v0.3.3's released
binaries, laid out the same way, fail identically; the same code objects load under the driver's
runtime. The ROCm 7.2.1 `amdhip64_7.dll` the packages bundle (HIP 7.2.53211) stopped working here
when the box rebooted at 13:16 on 2026-09-22 with a pending Windows update (KB5129195); the 0.3.3
flat test package had passed at 02:01 that day. The driver's own runtime in System32 (7.2.60201)
works. Windows loads a DLL from the executable's folder first and System32 second, so the flat test
packages used the bundled copy and the unified release zip, which keeps it in `gpu/rocm7.2` on PATH,
has been using the driver's. For gates on this box a flat package goes without the two runtime DLLs;
before the release, whether to bundle the 7.2.1 runtime at all is an open item.

## The max-flow check judges cut values (step 6p, 2026-09-25)

The fold-in 41-view gate on the RX 9070 passed every self-check but one: `CHESHIRE_MAXFLOW_CHECK`
reported 2 of 11,773,764 cells labelled differently by the GPU push-relabel cut and upstream's
Boykov-Kolmogorov on the same graph. The graph dumped for `hip/tests/maxflow_test`
(`CHESHIRE_MAXFLOW_DUMP`) gave the answer: the two cuts' values, computed in double from each
labelling, were both 218,448,238.9, and Boykov-Kolmogorov's own search left exactly 2 cells gray
(undetermined). A minimum cut need not be unique; each algorithm put a tie on a different side.
Earlier graphs happened to have no ties (0 of 11,417,156 at 41 views on v0.2.10, 0 of 29,080,924
at 884 views on 0.3.3).

The check now also evaluates both labellings on the adjacency-list graph, in double, and says
whether the values are equal (relative difference at most 1e-9, which leaves room only for the
order of a double sum over two different edge sets); the gate's verdict is that, and the cell count
stays in the line for information. On the same 41-view cache: 2 cells, values 218,447,937.91999644
and 218,447,937.91999739 (equal, relative difference 4.4e-15); mini6 `verify`: 0 cells, values
identical. The cut itself is unchanged.

## The 0.3.4 release gate (2026-09-25)

Each package as downloaded, paired into Meshroom 2023.3 by the gate harness
(`scripts/verify_end_to_end.py`): the 12-configuration mini6 matrix, then the 41-view monstree set
with `base` and `verify`. Times are the harness's per-configuration wall clock; the RX 9070 runs
shared the box with the Windows CUDA build, so theirs are not comparable with the others.

| package | card | mini6 | 41 base | 41 verify | max-flow check (41 verify) | direct read check |
|---|---|---|---|---|---|---|
| Windows AMD (unified zip) | RX 9070, rocm7.2 gfx12-generic | 12/12 at 7/7 | 332 s | 534 s | 0 of 11,779,311 cells, cut 218,659,035.18347341 both ways | 41 of 41 |
| Windows AMD (unified zip) | RX 5500 XT, hip6.2 gfx1012 | 12/12 at 7/7 | 988 s | 1432 s | 0 of 11,870,467 cells, cut 220,779,488.50049472 both ways | 41 of 41 |
| Linux HIP | RX 6750 XT | 12/12 at 7/7 | 551 s | 861 s | 0 of 11,832,140 cells, cut 219,354,958.61730957 both ways | 41 of 41 |
| Windows CUDA | GTX 1080 Ti | 12/12 at 7/7 | 793 s | 1252 s | 0 of 11,839,067 cells, cut 221,608,285.68726414 both ways | 41 of 41 |
| Linux CUDA | GTX 1050 Ti (4 GB) | 12/12 at 7/7 | 1216 s | 1506 s | 0 of 11,823,103 cells, cut 219,195,886.31766492 both ways | 41 of 41 |

Every configuration reached 7 of 7 ports with the paired SfM node, and every self-check the
`verify` and `texcheck` configurations assert passed. On the Linux CUDA package, the 41-view depth
and similarity maps from `scripts/linux/run-depthmap.sh` are byte-identical to the Meshroom CUDA
11.3 reference (`ref-cuda113`, from a GTX 1080 Ti): 82 of 82 files, on the GTX 1050 Ti. bench-pc did
not reboot during either of its gates (last boot 2026-09-22 before the AMD one, then the card swap
for the CUDA one). The Windows AMD, Linux HIP and Linux CUDA packages are built at a152325, the
Windows CUDA package at 303bd69 (step 6l's comparator named outside an OpenMP loop for MSVC, C3014;
no change in behaviour).

## 0.3.5: every CHESHIRE_* variable read one way (step 6q, 2026-09-25)

The ports and the generator's inline code had about 127 reads of `CHESHIRE_*` variables, in five styles:
`getenv(X) != nullptr` (so `X=0` turned a check or a profile ON), `e[0] == '0'`, `e[0] == '1'` (so
`X=true` did nothing), atoi / strtoll / strtod (so `X=abc` meant 0), and string compares. Every read
now goes through `hip/compat/include/cheshire/env.h`: `flag` (unset: the default; `0`, `false`,
`off`, `no` or empty: off; anything else: on), `integer` and `real` (unset, empty or not a number:
the default), `text` and `isSet`. Step 6q of the generator copies the header, adds its `#include` to
the 22 files that use it (for an `.inc`, to the file that includes it, since an `.inc` sits inside a
namespace), and refuses a tree that still reads a `CHESHIRE_*` variable any other way. The two codec
libraries, which build on their own outside AliceVision, keep their one probe each.

Behaviour changes, beyond `=0` now meaning off and a bad number meaning the default: a value such as
`true`, `yes` or `2` now turns on the options that took only a leading `1` (`CHESHIRE_PROFILE_SGM`,
`CHESHIRE_EXR_PROFILE`, `CHESHIRE_LOAD_PROFILE`, `CHESHIRE_READ_DIRECT_CHECK`,
`CHESHIRE_SFM_LOCAL_PASSES`, `CHESHIRE_GPU_FILTER_STRICT`, `CHESHIRE_BRIDGE_IMAGE_SPILL`,
`CHESHIRE_CUDA_MANAGED`); a value that starts with `0` but is not a false word (`00`, `0.5`) no
longer turns a default-on switch off; `CHESHIRE_GPU_RESIZE` now does what its documentation said
(`=0` gives the host resize; before, any value, `1` included, did); an empty
`CHESHIRE_BRIDGE_VRAM_MB` is the default cap, where it used to mean no cap (`=0` still does); and
`CHESHIRE_MAXFLOW_CHECK_EVERY=0`, which never advanced the max-flow loop, is taken as 1.

The conversion also found five upstream files the generator patches but step 0 did not reset
(`ImageDescriber_SIFT_popSIFT.cpp`, `cameraUndistortImage.hpp`, `main_incrementalSfM.cpp`,
`LocalBundleAdjustmentGraph.cpp`, `MaxFlow_AdjList.hpp`): their steps skip a file that already
carries the patch, so a changed snippet never reached a tree patched before it. They are in the
reset list now, and the generator's stray-file warning is empty.

Checked: `hip/tests/env/env_test.cpp` (every rule, MSVC and Linux g++); two generator runs give the
same patch; the Windows HIP tree (gfx12-generic, 284 steps) and the Windows CUDA tree (MSVC, 536
steps) build clean; mini6 12/12 at 7/7 on the RX 9070 with the HIP build, including `cpufallback`
(the `=0` switches), `bridgecap` / `bridgespill` (the numbers) and `verify` / `texcheck` (the
`*_CHECK=1` verdicts); and on the SfM binary, `CHESHIRE_BA_PROFILE=0 CHESHIRE_BA_CHECK=0` print 0
profile and 0 check lines where `=1` prints 9 of each (before, `=0` printed 9 of each too).

## 0.3.5: the reconstruction-quality gate for SfM (2026-09-25)

`scripts/quality_gate.py` asks whether a change makes incremental SfM worse than its own run-to-run
spread. It runs `sfmbench.py`'s fixed features-and-matches cache n times per leg, legs interleaved
(A B A B ...) so drift on the box falls on all of them, and compares each candidate with the first
leg: poses, landmarks and RMSE by Welch's t (two-sided p, and the **resolution**, the smallest
difference that comparison could have called significant: t_crit x SE), and pose agreement by
aligning every pair of runs with a similarity (Umeyama on the shared camera centres) - the residual
RMS over the centres' spread, and the median rotation difference - cross-leg pairs against the
pairs within the baseline, which are the noise floor. With no ground truth the geometry says
"different", not "worse", so it flags only past an absolute floor too (0.001 of the spread, 0.1
degrees); on 41 views two ways of rounding differed by 0.024 degrees against a 0.013-degree floor,
which a ratio alone called 1.8x. FAIL = significantly worse by more than the tolerance (poses: any
loss; landmarks and RMSE: 0.5 %), WARN = significantly worse within it. Checked first: the p-values
against t tables (t 2.0 at 10 df: 0.0734; the 5 % critical value 2.228), a run aligned with itself
(6e-16), two default 41-view runs (1.8e-5 of the spread, 0.002 degrees).

**Upstream-equivalent SfM against the defaults** (`CHESHIRE_BA_JACOBIANS=autodiff`,
`CHESHIRE_BA_PERSIST=0`, `CHESHIRE_SFM_TASK_SEED=0` against nothing set), n=10 per leg, RX 9070 box:

| set | landmarks | RMSE | resolution (landmarks / RMSE) | pose agreement, cross / within | verdict |
|---|---|---|---|---|---|
| 41 views | 80,801.4 both, p 1.0 | -0.049 %, p 0.17 | 0.007 % / 0.073 % | centres 1.16x, rotation 1.84x (0.024 degrees) | PASS |
| engine bay, 107 | -0.018 %, p 0.61 | +0.038 %, p 0.41 | 0.073 % / 0.098 % | 0.97x, 0.91x | PASS |

An A/A control (the defaults twice, 41 views) passes with p 0.26 and 0.29 and geometry at 0.36x and
1.00x. The defaults are also more repeatable than upstream: the median rotation difference between
two of their runs is 0.0008 degrees on 41 views against upstream's 0.013, and 0.046 against 0.089 on
the engine bay - the per-task generators of 5n.

**The QR nullspace as a positive control did not reach significance**: -73.8 landmarks (-0.053 %),
p 0.16, against a resolution of 107 (0.076 %) on the engine bay. docs/17 measured -165 at p 0.0008
at n=10 on a v0.2.17 build; either the effect is smaller on today's SfM or it sits at the edge of
what n=10 resolves here. Either way this is what a PASS means: no difference larger than its
resolution. A regression near the 0.5 % tolerance is about seven times the resolution on the engine
bay and would not pass unnoticed. Reports: `build/quality/41-upstream-vs-cheshire.md`,
`41-AA-control.md`, `eb-upstream-vs-cheshire.md`, `eb-cheshire-vs-qr.md`.

## 0.3.5: CheshireJPG in PrepareDenseScene's read (step 6r, 2026-09-25, off by default)

`CHESHIRE_GPU_JPEG=1` swaps the libjpeg-turbo decode inside 6n's direct read for CheshireJPG's
(docs/19): the file's bytes to the device, the 8-bit pixels back, and the rest of the read unchanged.
A file the codec does not take (progressive, arithmetic, CMYK, a damaged stream, no device) reads as
before, and `CHESHIRE_GPU_JPEG_CHECK=1` decodes each image through OpenImageIO as well, compares the
8-bit pixels and keeps OpenImageIO's on a difference. The decoders (`CHESHIRE_GPU_JPEG_CODECS`,
default 4) are shared by the read threads. After 6n the decode is about 60 of the read's 180 ms per
12 MP photo, about 12 % of the node's thread time on the RX 9070 box, which is the ceiling here
(docs/notes/cheshirejpg-reads-plan.md).

Build: the generator adds the codec to the image library from the checkout (`add_subdirectory` of
`hip/port/cheshirejpg`) with the build's own GPU language and architectures, so
`aliceVision_image` now carries device code and is one of `build_targets.py`'s per-target
libraries (six, where there were five). Two fixes the integration needed: inside AliceVision the
codec must not force-include its own copy of `cuda_to_hip.h` beside the build's (two files, so
`#pragma once` does not stop the redefinitions), and `jpegTypes.hpp`'s `__builtin_clz` has an MSVC
branch (`_BitScanReverse`, checked against a loop on 70,159 values), since the Windows CUDA build is
the first to compile the codec's host code with MSVC.

| where | result |
|---|---|
| RX 9070, gfx12-generic code object, mini6 | 6 of 6 decoded on the device, check 6 of 6 identical, EXRs 6 of 6 byte-identical to the libjpeg-turbo run |
| RX 9070, engine bay | 107 of 107 decoded, check 107 of 107 identical, EXRs 107 of 107 byte-identical |
| GTX 1080 Ti (CUDA, the codec's first NVIDIA run), 41 views | 41 of 41 decoded, check 41 of 41 identical, EXRs 41 of 41 byte-identical; the node 30.0 s to 26.2 s on bench-pc's FX-8120 (one run each) |
| Linux HIP build | compiles; `libaliceVision_image.so` carries all 19 bundle targets |
| hip6.2 gfx1012 payload (Windows) | compiles; the harvest finds `aliceVision_image.dll` with gfx1012 only |

The end-to-end harness has a `gpujpeg` configuration (the switch and its check, both verdicts
required).

**Exact on every card, and no faster.** PrepareDenseScene's wall time, off against on (no check):

| box | identical to libjpeg-turbo | off | on |
|---|---|---|---|
| RX 9070, Ryzen 5 5600X, engine bay | 107 of 107 | 16.9, 16.5, 16.6 s | 17.8, 17.6, 17.7 s (+6.6 %) |
| RX 5500 XT (gfx1012, hip6.2), FX-8120, 41 views | 41 of 41 | 25.6, 26.5 s | 26.9 s |
| GTX 1080 Ti (CUDA), FX-8120, 41 views | 41 of 41 | 30.0 s | 26.2 s (one run) |
| RX 6750 XT (Linux, b035a), i3-4330, engine bay | 107 of 107 | 93.9, 95.0, 94.9 s | 94.6, 94.8, 98.1 s |

EXRs byte-identical to the libjpeg-turbo run in every case. The ~12 % of the node's thread time that
the decode takes does not reach the wall clock: on 12 threads the host decodes the photos in well
under a second of wall time and the device path adds a file read, an upload and a 36 MB download per
photo, and on house-pc's 4 threads the node is bound elsewhere (the EXR writes and the colour
transform), so taking the decode off the CPU changes nothing there either. It stays off by default,
and FeatureExtraction's read waits for a measurement that shows the decode on its critical path.

## 0.3.5: the visibility votes on the device (step 6t, off by default) and the knn kernel's layout (6s) (2026-09-26)

**The decision point** (docs/notes/fusion-visibility-plan.md, after step 3): go on to device votes
if house-pc is still host-bound. Measured with `CHESHIRE_GPU_VIS_LOG=1` (b035a bundle on house-pc,
the dev install here):

| where | pass 1 | pass 2 |
|---|---|---|
| house-pc, engine bay (107 views), per camera | host 53.5 ms (votes 47) against device 46.6 ms | host-bound as well |
| house-pc, False Door (884 views) | 87.5 s: host 70.9 (votes 59.2, maps 7.0, staging 4.6), device 66.7 (knn 39.3, download 20.0) | 69.0 s: host 56.8 (votes 43.4), device 56.0 |
| RX 9070, False Door | 46 s, device-bound (votes 12.7 s on 12 threads) | 35 s |

The whole 884-view Meshing on house-pc is 483 s: 157 s in the two passes, 131 s reading the depth
maps. On the 5600X the host votes split as decide 3.7, scan 0.4, scatter 0.9 and apply 7.3 s in
pass 1. Only the camera-list appends have to stay on the host. 128 M of the 946 M votes add a
camera in pass 1, and 7.2 M in pass 2, whose lists already hold pass 1's cameras.

**6t.** With `CHESHIRE_GPU_VIS_VOTES=1` the pass's vertices (coordinates, nrc, sim, scoreV) live on
the device. Per camera, after the backprojection and the knn:

1. Every query is decided with the host's expressions: `simScorePrepare[v] * pixSize * pixSize`
   rounded to float, `std::max` as `(a < b) ? b : a`, and the float products compared in double.
2. A vote sets the vertex's bit in the camera's bitmap. The host appends the camera to each voted
   list from that bitmap, one camera behind, while the device works on the next camera.
3. A contribution counts itself on its vertex. An exact integer scan of the counts gives each
   vertex a range, and the contributions are scattered into it. One thread per vertex sorts its
   range by query (insertion sort, heapsort past 32) and folds it with Point3d's three separate
   operations. Each vertex sees its contributions in pixel order, as the ordered host votes apply
   them.
4. At the end of the pass the coordinates and nrc come back. The host's copies are untouched until
   then.

Before a pass, the device's fold is compared with this build's on 4096 triples. Every index the
scatter and the fold compute is range-checked into a counter that must stay 0. Any failure mid-pass
truncates the camera lists to their pass-start sizes and reruns the pass with host votes. The
failures covered are a device error, an overflowed query, a map larger than announced, a nonzero
counter, and `CHESHIRE_GPU_VIS_VOTES_FAIL_AT=c`. After a device error, the rerun is upstream's CPU
pass and the device is not used again.

**The first version used hipCUB's radix sort and crashed this machine.** rocPRIM, under hipCUB and
rocThrust, resolves its kernel configuration twice. On the host it takes the device it finds
(gfx1201). In the kernel it takes the compile target, and a generic target (gfx12-generic,
Cheshire's default) resolves there to "unknown". So the host launched one configuration and the
kernel ran another. The mini6 run folded the wrong contributions into every vertex. The
fault-injection run hit "unspecified launch failure", and its recovery kept using the faulted
context. The machine then bugchecked (0x119, VIDEO_SCHEDULER_INTERNAL_ERROR). Devices rocPRIM
doesn't name (gfx1031, gfx1012) would have worked by accident. The scan and scatter above replace
the library, and a device error now ends the device's part in that pass. PopSIFT's grid filter uses
Thrust and is being checked for the same trap.

| check | result |
|---|---|
| mini6, `CHESHIRE_GPU_VIS_CHECK=1` | votes identical to the ordered host reference on all 741,973 / 264,054 vertices (coordinates, nrc, camera lists in order, pixSize); backprojection and knn identical on all 8,808,856 queries per pass; digests `806ef990.../51779edd...` as before |
| mini6, failure injected at camera 3 | both passes rerun with host votes; same digests and tetrahedralization checksum |
| 41 views, CHECK | identical on all 4,138,408 / 1,860,629 vertices and all 88,267,121 queries per pass; digests `4d536979.../db94ece3...`, tetrahedralization input `cdd23710...` |
| False Door, 884 views | digests `c572bc16.../f0ff15ea...` and tetrahedralization input `64b36ee4...` unchanged |

RX 9070 box, False Door, back to back:

| | host votes | device votes |
|---|---|---|
| pass 1 | 45.7 s, 44.9 s | 42.6 s (cold, first run after a reboot), 39.0 s |
| pass 2 | 33.7 s, 34.7 s | 29.7 s, 27.6 s |
| camera-list appends on the host | inside votes of 13.4 / 9.2 s | 3.0 / 1.4 s |
| device decide, scan, scatter and fold | | 0.3 s per pass |
| Meshing node (warm pair) | 237.8 s | 230.3 s |

Here the passes stay knn-bound, at about 38 s of kernel per pass. The step is aimed at four-thread
hosts, where the votes were most of the host's time.

**On house-pc (Linux, RX 6750 XT, i3-4330; bundle b035b built from 25898f6 with gcc, so the plain
arithmetic forms).** The engine bay with `CHESHIRE_GPU_VIS_CHECK=1`: votes identical to the ordered
host reference on all 6,653,300 and 3,292,402 vertices, knn and backprojection identical on all
147,299,450 queries per pass, and digests equal to the b035a bundle's (`3bc9f8f4.../27fe49ef...`).
The False Door's digests with device votes also equal the host votes' (`a43c4e50.../8594172d...`,
tetrahedralization input `8070538c...`). Back to back, old knn layout:

| | engine bay, host votes | engine bay, device votes | False Door, host votes | False Door, device votes |
|---|---|---|---|---|
| pass 1 | 9.3 s | 7.4 s | 88.8 s | 65.6 s |
| pass 2 | 7.3 s | 5.3 s | 70.0 s | 51.3 s |
| Meshing node | 124.2 s | 120.9 s | 489.9 s | 450.6 s |

At 884 views the host's own votes were 59.0 and 42.4 s. With device votes it only appends cameras
(9.3 and 5.1 s), and the next limit shows up: waiting for depth maps from the three reader threads
rises from 7.3 and 8.5 s to 29.8 and 25.1 s. That is the plan's step 9 (readers). The knn kernel
itself is 39.5 and 28.0 s.

**Step 9, the readers (`CHESHIRE_GPU_VIS_READERS`, default 3).** The depth maps are read ahead on
1-16 threads and consumed in camera order, so the result does not depend on the count (every run
below has the same digests). LOG now gives the time the readers spent, summed over the maps. The
False Door's filtered depth maps are ZIPS at 14.8 MB each (20.3 MB raw), so a pass reads 12.3 GB.
house-pc's `/data` is the 512 GB mSATA drive on a 3 Gbps link, and `/` is the system SSD on a
6 Gbps link, so both were measured (b035c bundle, device votes):

| house-pc, 884 views | waiting for maps | reading, summed | passes | Meshing node |
|---|---|---|---|---|
| `/data` (3 Gbps), 3 readers | 29.8 + 25.1 s | | 116.9 s | 450.6 s |
| `/data` (3 Gbps), 6 readers | 18.0 + 26.2 s | 274 + 283 s | 119.6 s | 450.4 s |
| `/` (6 Gbps), 3 readers | 21.4 + 24.0 s | 132 + 127 s | 111.3 s | 428.8 s |
| `/` (6 Gbps), 4 readers | 18.3 + 20.1 s | 177 + 167 s | 110.2 s | 429.5 s |
| `/` (6 Gbps), 6 readers | 12.2 + 16.9 s | 243 + 238 s | 105.6 s | 423.7 s |

On the RX 9070 box, a read takes 35 ms and 3 readers already keep up. The wait is about 3.7 s per
pass with 3 or 6 readers. So the default stays at 3.

- On house-pc, more readers wait less, but each read takes longer, because the i3's two cores
  decode the maps.
- On the slower drive, 6 readers are a little slower than 3.
- The faster drive saves about 20 s of the 884-view Meshing, most of it in the depth-map load.

What remains is decode time: about 130 s of reading per pass with 3 readers. The Linux bundle's
OpenEXR is 3.1, which inflates with zlib; the Windows packages have 3.4 with libdeflate.

**6s. The knn kernel's layout.** Four changes:

- The points are stored in leaf order, so a leaf is one contiguous run and the answer is
  `perm[slot]`.
- A stack frame is 24 bytes instead of 40. The other child, the axis and the phase share one int,
  and once the frame is revisited the cut distance is replaced by the distance it displaced.
- The stack is 40, 64 or 96 frames depending on the tree's depth, instead of always 96.
- The three axis distances are held in registers.

Same points, order, metric and comparisons: mini6 and 41-view CHECK identical to nanoflann, and the
884-view digests unchanged. On the RX 9070 it saves 0.65 s of the passes' 82 s (two alternating
pairs), because RDNA4's cache already hid the scattered reads. It is unmeasured on cards without
that cache. On house-pc's RX 6750 XT it is slower: on the engine bay the knn kernel takes 5.3 s and 4.7 s
per pass against the old layout's 3.6 s and 3.1 s. So it is off by default, and
`CHESHIRE_GPU_KNN_LAYOUT=1` selects it.

## 0.3.5: device votes by default (step 10) and the dense point cloud's SfMData on every core (6u) (2026-09-26)

**Step 10.** `CHESHIRE_GPU_VIS_VOTES` now defaults to on, after the exactness runs above on the
RX 9070 (Windows, fused host forms) and the RX 6750 XT (Linux, plain forms). The RX 9070 run also
includes the full 884-view CHECK below. `CHESHIRE_GPU_VIS_VOTES=0` announces "visibility votes:
disabled by CHESHIRE_GPU_VIS_VOTES=0, host votes". Gate changes in `scripts/verify_end_to_end.py`:

- The `verify` configuration now requires "(pass N, GPU votes)" in both votes verdicts, so a pass
  that fell back to host votes fails it.
- A new `hostvotes` configuration (host votes under CHECK) keeps the bucketed host path gated as the
  fallback.
- "visibility votes on the GPU" is a Meshing marker.
- `scripts/verify_packages.py` looks for `CHESHIRE_GPU_VIS_VOTES` in fuseCut. (That script was retired
  in 0.3.6; see "0.3.6: tooling debts" at the end.)

The Windows CUDA tree compiles the device votes (`knnGPU.cu`, CUDA 12.9, MSVC 14.44).

**6u.** At the end of Meshing, `main_meshing.cpp`'s `createDenseSfMData` builds the SfMData saved as
`densePointCloud.abc`. Upstream copies the whole input SfMData, its landmarks included, only to
clear them. It then builds one landmark per mesh vertex on one thread, with one projection (with
distortion) per camera that sees the vertex, and inserts it into the map. It printed nothing, and
at 884 views it was the 13.8 s between "Mesh post-processing done" and "Save dense point cloud".

Now the input's landmarks are set aside while the rest is copied. The landmarks are built in
parallel into a vector, since every lookup (views, poses, intrinsics, MultiViewParams) is const,
and are then moved into the map in index order. `Landmarks` is a `std::map`, so its content does
not depend on insertion order.

`CHESHIRE_DENSE_SFM_CHECK=1` also builds upstream's version and compares every landmark bit for bit:
position, describer type, colour, and each observation's view, coordinates, feature id and scale.
It is in the gate's `verify` configuration.

| | result |
|---|---|
| mini6, CHECK | identical to upstream on all 250,207 landmarks (916,597 observations) |
| False Door, 884 views, RX 9070, full CHECK (with the device votes) | identical to upstream on all 4,071,638 landmarks (58,169,569 observations); in the same run, backprojection and knn identical on all 3,124,680,410 queries of each pass, and votes identical to the ordered host reference on all 9,937,481 and 4,699,910 vertices (plan step 8) |
| False Door, 884 views, RX 9070 box | 13.8 s to 3.3 s before the save (the function: 1.5 s building the landmarks on 12 threads, 0.4 s filling the map); the Meshing node 241.5 s to 228.8 s |

`CHESHIRE_DENSE_SFM=0` runs upstream's function.

**RX 5500 XT (bench-pc, Windows, HIP SDK 6.2, gfx1012, 8 GB; FX-8120), 41 views, package b035 from
e99273e.** With every Meshing self-check on: backprojection and knn identical on all 88,366,419
queries of each pass, votes identical to the ordered host reference on all 4,163,049 and 1,876,606
vertices, and the dense point cloud identical on all 1,181,106 landmarks. Timing, alternating runs:

| | device votes | host votes |
|---|---|---|
| passes | 21.2 + 15.5 s, 21.1 + 15.7 s | 21.2 + 16.4 s, 21.1 + 16.4 s |
| Meshing node | 166.7 s, 165.4 s | 164.2 s, 165.0 s |

Same digests (`f65c6a14...` in pass 1) either way. On this card the passes are device-bound, so the
default costs nothing and gains under a second.

## 0.3.5: the EXR reader inflates with libdeflate where OpenEXR uses zlib (6v, 2026-09-26)

The Linux bundle's OpenEXR is 3.1, which inflates ZIP and ZIPS chunks with zlib. OpenEXR 3.2 and
later use libdeflate, as the Windows packages' 3.4 does. Upgrading the Linux OpenEXR would change
the bytes of every EXR Cheshire writes, and file-level gates compare those. Only the read path
needed to change, so the direct EXR reader (5v) now inflates the chunks itself:

1. It takes each chunk's compressed bytes through `InputFile::rawPixelData`.
2. It inflates them with libdeflate, the system's on Linux (already in the bundle through libtiff)
   or vcpkg's `deflate.dll` on Windows.
3. It undoes OpenEXR's byte predictor and interleave.
4. It writes the lines into the float image, widening half to float exactly as a FLOAT slice does.

Chunks stored uncompressed are copied as they are. The decode runs in batches on OpenEXR's own
thread pool, so concurrent readers share it as their `readPixels` calls did. On any failure the
file goes back to `readPixels`. Decompression has only one correct output, so the pixels are
OpenEXR's. `CHESHIRE_EXR_DEFLATE_CHECK=1` reads every such file both ways and reports at exit:

| where | files identical to `readPixels` |
|---|---|
| Windows, forced on, mini6 Meshing (ZIPS one-channel depth and similarity maps) | 24 of 24 |
| Windows, forced on, mini6 DepthMap on PrepareDenseScene's ZIP (16-line) half RGBA | 6 of 6 |
| Linux (house-pc, b035e), engine bay Meshing | 428 of 428 |

It is on by default only with OpenEXR before 3.2, and `CHESHIRE_EXR_DEFLATE=0|1` overrides that. On
the RX 9070 box (OpenEXR 3.4) it read slightly slower than OpenEXR's own inflate: 30.6 and 34.5 s
of reading per 884-view pass, against 27.1 and 30.8 s. On house-pc (OpenEXR 3.1), False Door on
`/data`, two pairs in opposite orders, all with the same digests:

| house-pc, 884 views | libdeflate | OpenEXR 3.1 (zlib) |
|---|---|---|
| Meshing node | 442.3 s, 438.3 s | 468.1 s, 460.6 s |
| depth-map load | 129.3 s, 128.3 s | 137.4 s, 142.5 s |
| reading, summed over the maps, pass 1 / pass 2 | 135.5 / 156.8 s, 135.7 / 150.6 s | 167.8 / 178.4 s, 157.4 / 158.5 s |
| visibility passes | 62.8 + 63.3 s, 62.6 + 60.4 s | 71.3 + 72.1 s, 68.0 + 64.6 s |

About 24 s (5 %) of the node. The same reader serves DepthMap, DepthMapFilter and Texturing, so the
Linux bundle's other EXR reads inflate this way too.

**On CUDA: GTX 1080 Ti in house-pc (Linux; bundle built from 870683d with CUDA 12.9 for sm_61),
2026-09-26.** Engine bay Meshing with device votes (the default), the visibility CHECK, the dense
point cloud check and the libdeflate check:

| check | result |
|---|---|
| visibility votes, both passes | identical to the ordered host reference on all 6,653,300 and 3,292,402 vertices |
| knn and backprojection, both passes | identical on all 147,299,450 queries |
| pass digests and tetrahedralization input | `3bc9f8f4...` / `27fe49ef...` and `df6828cc...`, the same as the RX 6750 XT's Linux runs |
| dense point cloud (6u) | identical to upstream on all 1,181,458 landmarks (7,401,407 observations) |
| EXR reads through libdeflate (6v) | 428 of 428 files identical to OpenEXR's readPixels |

So device votes, 6u and 6v are exact on the CUDA backend as on HIP. This bundle predates the GPU
matcher check, which gets its CUDA run in the release round.

## 0.3.5: GPS-radius image pairing (6w, opt-in, 2026-09-26)

For the drone survey above: with `CHESHIRE_GPS_PAIRING_RADIUS=<metres>`, `aliceVision_imageMatching`
pairs every two views that carry GPS within that distance. The position is the EXIF one in
Cartesian metres, through `ImageInfo::getGpsPositionFromMetadata`. The configured method's pairs
between two GPS views are dropped, or kept as well with `CHESHIRE_GPS_PAIRING_UNION=1`. Pairs
with a view that has no GPS stay as the method proposed them, so a mixed set still pairs, and a
pair is added once in either orientation. The node logs how many views carried GPS, the pairs
within the radius with their median distance, and what happened to the method's pairs. It is
paired as the ninth node, gated on its help text.

Checked on the 41-view monstree set (GPS in every photo, about 7 m across), with Exhaustive as the
method:

| radius | pairs | log |
|---|---|---|
| none | 820 | |
| 3 m | 303 | 41 of 41 views with GPS; 303 within the radius, median 1.95 m; 820 method pairs dropped |
| 10 m, 30 m | 820 | every pair within the radius |
| 3 m, 10 views stripped of GPS | 527 | 31 of 41 with GPS; 172 radius pairs; 355 method pairs with a view without GPS kept (10 x 40 - 45, all of them) |

What it is for, the 444-photo survey (80 m caught every real pair with 9 % of the exhaustive
matching), is not run yet: that photo set is no longer on house-pc.

## 0.3.5: in-process checks for the last two unchecked GPU ports (2026-09-26)

**The matcher** (`CHESHIRE_GPU_MATCHER_CHECK=1`). Evenly spaced queries of every search (64 per
search) are answered again by upstream's `ArrayMatcher_bruteForce`. The check compares both
distances, and the nearest row unless the two nearest are tied: upstream's partial sort leaves
tied rows in no particular order, and a tie fails Lowe's ratio test anyway. mini6 on the RX 9070:
975 of 975 sampled queries identical, with 0 rows and 0 distances different.

**The depth-map filter's vote pass** (`CHESHIRE_GPU_FILTER_CHECK=1`). Upstream's CPU vote loop runs
again into its own buffers and the modal counts are compared pixel by pixel, per camera. The
buffers are not cleared between cameras, which is upstream's quirk, or are cleared under
`CHESHIRE_GPU_FILTER_STRICT=1`, as the GPU then does. mini6: 6 of 6 cameras identical (0 of
18,289,152 pixels) in both modes. The two modes' filtered depth maps differ by 3,462,453 pixels, so
the check agreed with two different results, each against its own reference.

Both are in the gate's `verify` configuration, with verdicts for FeatureMatching and
DepthMapFilter. Every GPU port now has an in-process check against its CPU reference.

## 0.3.5: the reconstruction-quality gate's mesh part (2026-09-26)

`scripts/mesh_distance.py A.obj B.obj` measures how far apart two meshes are. It samples each
surface (300,000 points, area-weighted) and takes each sample's exact distance to the other mesh:
the closest point on the nearest triangle, found through a voxel grid of triangle bounding boxes
and tested against the 27 surrounding cells. Coarser grids pick up whatever is further away, so
holes and extra surface are measured too. It reports both directions: mean, median, p95, p99 and
max, absolute and as a fraction of the bounding-box diagonal. It needs only numpy.

Checks:

- The triangle distance matches eight closed-form cases (face, three vertices, three edges, a point
  on the face) exactly, and is never above a dense brute-force sampling on 200 random triangles.
- A mesh against itself measures about 1e-16.
- A copy shifted 1 mm along x has a maximum of exactly 1 mm and a mean of 0.5 mm, the shift
  projected on the surface normals.
- mini6 with device votes against mini6 with host votes also measures about 1e-16. Meshing's
  output renumbers run to run, but its geometry here is the same surface.

`scripts/quality_gate.py mesh --leg base=A1.obj,A2.obj --leg cand=B.obj` uses it as a gate. Pairs
within the first leg are its run-to-run spread, each candidate is measured against every baseline
mesh, and the verdict is on the median symmetric p95:

- FAIL when it is more than 50 % over the baseline's largest spread and over 0.1 % of the diagonal
  (`--mesh-tol`, `--mesh-floor`);
- WARN past the spread but within the tolerance;
- PASS otherwise.

On mini6, measured raw, the 1 mm shift (0.018 %) passes under the floor and a 20 mm shift (0.35 %)
fails. Against upstream the gate no longer measures raw (below), and a rigid shift is then aligned
away by design.

### Against upstream Meshroom 2023.3 on the engine bay (2026-09-26)

Upstream: Meshroom 2023.3's own binaries on every node (`CHESHIRE_BACKEND=meshroom` through the
paired launchers), Meshroom's default photogrammetry graph, house-pc's GTX 1080 Ti, two runs of
about 2 hours each (CPU DSP-SIFT and matching on the i3). Cheshire: the same graph through the
end-to-end gate's new `defaults` configuration (Meshroom's parameters, nothing overridden, 7/7
ports), a package of this tree, the RX 9070 box, two runs. 107 photos.

Comparing these meshes raw measured their coordinate frames, not their surfaces, for two reasons:

- **SfM fixes its own frame.** After aligning the camera centres (Umeyama), upstream's frame sat
  0.5 degrees and 5 % of the camera spread from Cheshire's. The cameras then agreed to 0.13 %, but
  the surfaces still needed about 1 more degree. Camera centres carry their own pose noise and
  cannot pin the frame as tightly as the surfaces can. So `mesh_distance.py --refine` follows the
  camera alignment with a trimmed point-to-plane ICP with a similarity. On a synthetic mesh moved
  by 2 degrees, 1 % scale and 3 cm it recovers the move exactly (p95 6e-17) in 5 rounds. A local
  bump survives it untouched.
- **Meshing's volume moves run to run, upstream too.** `divideSpaceFromSfM` takes the axis-aligned
  box of the landmarks with at least 3 observations and 10 degrees of parallax, from their 0.1-99.9 %
  quantiles plus 5 %. The quantile is the 40th most extreme of about 40,000 such landmarks, and
  the box flips between two sizes: upstream's run 1 got 3.66 x 1.75 x 1.77, its run 2 got
  2.73 x 0.96 x 1.73, both Cheshire runs about 2.74 x 0.97 x 1.74, and an earlier Cheshire job on
  house-pc got 3.71 x 1.77 x 1.76. A Python copy of the function reproduces both of today's logged
  boxes from the SfM outputs, and both pipelines' landmark counts and filtered counts are within
  1 %. So the flip is in which tail landmarks pass the filter, not in Meshing's code or in Cheshire.
  `--common` measures only inside the volume both meshes cover, and reports how much of each mesh
  was left out. The relative figures are then fractions of that volume's diagonal.

Two further fixes came out of this run. Distances are capped at 2 % of the diagonal, and samples
farther than that count as "beyond" and are reported as a fraction. The sample-triangle pairs are
evaluated in chunks of at most 2 million. An unbounded first version reached 45 GB on two meshes in
different frames, and a watchdog now also guards these runs.

`quality_gate.py mesh` aligns on the cameras, refines, and uses the common volume by default
(`--raw` for neither). It runs pairs in parallel with `--jobs`. Result
(`build/quality/mesh-eb-upstream-vs-cheshire.md`):

| pair | symmetric p95 | medians | beyond 2 % | in the common volume | ICP rotation |
|---|---|---|---|---|---|
| upstream 1 / upstream 2 (baseline) | 1.2225 % | 0.00179 / 0.00169 | 2.86 / 1.43 % | 88.6 / 100.0 % | 0.056 degrees |
| Cheshire 1 / upstream 1 | 1.4643 % | 0.00238 / 0.00249 | 1.83 / 3.38 % | 100.0 / 88.6 % | 1.027 degrees |
| Cheshire 1 / upstream 2 | 0.7410 % | 0.00220 / 0.00217 | 0.73 / 0.47 % | 99.5 / 99.9 % | 1.070 degrees |
| Cheshire 2 / upstream 1 | 1.3418 % | 0.00243 / 0.00252 | 2.15 / 2.95 % | 100.0 / 88.7 % | 1.017 degrees |
| Cheshire 2 / upstream 2 | 0.8926 % | 0.00225 / 0.00222 | 1.44 / 0.47 % | 99.7 / 100.0 % | 1.058 degrees |

**PASS**: Cheshire's median symmetric p95 to upstream is 1.117 %, against upstream's own run-to-run
spread of 1.2225 %. The p95 follows the volume more than the pipeline: the pairs with upstream's run
1, the large box, are the high ones on both sides. Every ICP converged, with final steps of 2-5e-6.

One difference below that verdict, not yet explained. The medians across the pipelines, 0.0022 to
0.0025, sit about 30 % above upstream's own (0.0017-0.0018). Two Cheshire runs, refined the same way,
measure 0.0015. Cheshire is as repeatable as upstream, and a systematic difference of about 0.05 % of
the diagonal separates the two. There are two candidates:

- **The AliceVision versions.** Meshroom 2023.3 ships 3.2, and Cheshire is built on `develop` 3.4
  (docs/roadmap.md, "Upstream's next pipeline").
- **Cheshire's changes that are not bit-identical.** These are the analytic BA Jacobians, the
  per-task SfM generators, and the persistent Problem.

Running this tree with those switches at upstream's settings would separate the two.

## The 0.3.5 release gate (2026-09-26/27)

Every binary was built at 4720cb7. The two Windows zips were repacked with the pairing scripts of
82a4fcc (below), and the checksums are in `build/release/0.3.5/SHA256SUMS`. Each package was gated as
the file you download, through Meshroom 2023.3, on its own card: the 15-configuration mini6 matrix,
then the 41-view set with `base` and `verify`, harness at 65246c2.

| package | hardware | mini6 | 41 views |
|---|---|---|---|
| Windows AMD | RX 9070 (rocm7.2 gfx12-generic) | 15/15 at 7/7 | base and verify at 7/7 |
| Windows AMD | RX 5500 XT (hip6.2 gfx1012, bench-pc) | 15/15 at 7/7 | base and verify at 7/7 |
| Linux AMD | RX 6750 XT (house-pc) | 15/15 at 7/7 | base and verify at 7/7 |
| Linux CUDA | GTX 1080 Ti (house-pc) | 15/15 at 7/7 (11 + 4 rerun after the harness fixes) | base and verify at 7/7 |
| Windows CUDA | GTX 1080 Ti (bench-pc) | 15/15 at 7/7 | base and verify at 7/7 |

house-pc ran the gates in a scratch Meshroom 2023.3, extracted from its tarball into
`/data/tests/gate-035`, so the node app's own install and pairing stayed as they were.

The gate found four problems, all in tooling, none in the binaries:

- The Windows pairing option check refused every node (`--help`: Meshroom 2023.3's Windows binaries
  list it, and ours and its Linux ones do not). It had never truly run on Windows before, because
  Meshroom's binaries crashed on `--help` in the earlier test and "no options" pairs unchecked.
- The Windows CUDA zip did not ship the check.
- `hostvotes` required the GPU votes' line.
- A harness copy without `check_textured_obj.py` crashed the matrix.

All are fixed (82a4fcc, c765d94, 65246c2).

## 0.3.6: the mesh gate's median gap is not Cheshire's SfM defaults (2026-09-27)

The same engine-bay gate as in 0.3.5, with two new Cheshire runs (0.3.5 release package, `defaults`
configuration, RX 9070). Here Cheshire's non-bit-identical SfM defaults were set to upstream's:
`CHESHIRE_BA_JACOBIANS=autodiff`, `CHESHIRE_BA_PERSIST=0` and `CHESHIRE_SFM_TASK_SEED=0`, the SfM
quality gate's upstream-equivalent set. The SfM log confirms the autodiff Jacobians took effect.
Report: `build/quality/mesh-eb-upstream-vs-upeq.md`.

| candidate against upstream Meshroom 2023.3 | median distances | median symmetric p95 | ICP rotation | verdict |
|---|---|---|---|---|
| Cheshire, 0.3.5 defaults | 0.0022-0.0025 | 1.117 % | 1.02-1.07 degrees | PASS |
| Cheshire, upstream-equivalent SfM | 0.0022-0.0025 | 1.062 % | 1.04-1.09 degrees | PASS |
| upstream's own two runs (baseline) | 0.0017-0.0018 | 1.2225 % | 0.056 degrees | |

Putting the SfM switches back changes neither the medians nor the frame. Both Cheshire
configurations sit about 1 degree from upstream's frame, where upstream's own two runs sit 0.06
degrees apart. So the gap comes from something these switches do not reach. The candidates are
AliceVision 3.2, which Meshroom 2023.3 ships, against the 3.4 `develop` base, or the libraries
around it, Ceres among them. The next test runs Cheshire's dense stages on upstream's own SfM
result: if the gap disappears it is in SfM, and if it stays it is in the dense stages.

**The stage swap places the gap in SfM.** Meshroom caches each node by its inputs, not by the binary
that ran it. So upstream run 1's cache up to StructureFromMotion (Meshroom 2023.3's own
FeatureExtraction, FeatureMatching and SfM) was copied into a fresh cache on house-pc, and
Meshroom's default graph was run on it with the app's install, paired with the 0.3.5 bundle. The
copied nodes count as done and the SfM log was unchanged, while PrepareDenseScene through Texturing
ran the Cheshire build (RX 6750 XT, 645 s). Report: `build/quality/mesh-eb-upstream-vs-swap.md`.

| pair | medians | symmetric p95 | ICP rotation |
|---|---|---|---|
| upstream SfM + Cheshire's dense stages / upstream run 1 (the same SfM) | 0.00146 / 0.00152 | 0.686 % | 0.004 degrees |
| upstream SfM + Cheshire's dense stages / upstream run 2 | 0.00171 / 0.00169 | 0.934 % | 0.056 degrees |
| upstream run 1 / upstream run 2 (baseline) | 0.00179 / 0.00169 | 1.2225 % | 0.056 degrees |

On upstream's own SfM, Cheshire's dense stages land at least as close to upstream's mesh as
upstream's two runs are to each other. The gap and the 1-degree frame offset appear only with
Cheshire's SfM, and the upstream-equivalent run above showed that Cheshire's SfM switches do not
cause them. What remains is the SfM of the AliceVision base: 3.4 `develop` in Cheshire, 3.2 in
Meshroom 2023.3. The frame offset makes the end-of-SfM automatic transform (`useAutoTransform`) the
first suspect; that is unchecked. Without ground truth the gate cannot say which SfM is better, only
that the two differ by about 0.05 % of the diagonal.

## 0.3.6: GPS-radius pairing on a real survey, and the AltitudeRef fix (2026-09-27)

OpenDroneMap's "zoo" survey (CC0 1.0, hub.dronedb.app/r/odm/zoo) stands in for the DJI set: 524
photos from a Sony DSC-WX220 on a fixed-wing mapping flight, 1.85 km x 1.08 km, a median 25 m between
shots, GPS in every EXIF. `scripts/gpsbench.py` runs GPU SIFT once, then each pairing leg through
ImageMatching, FeatureMatching and SfM with Meshroom's default options (development install, RX 9070
box).

**The first GPS legs changed nothing, and that was a bug.** The log said "GPS pairing: 0 of 524 views
carry GPS". Upstream's `ImageInfo::hasGpsMetadata()` requires all six GPS tags, and this camera writes
no `GPSAltitudeRef`, which EXIF makes optional (absent means above sea level). 6w now needs only
latitude and longitude with their references, takes the altitude if present, and defaults a missing
AltitudeRef to 0 (9e1562c). After the fix, 524 of 524 views were found.

| leg | pairs proposed | verified | exhaustive's verified pairs caught | views placed | landmarks | FeatureMatching |
|---|---|---|---|---|---|---|
| vocabulary tree (Meshroom's rule above 200 photos) | 13,660 | 842 | 842 of 6,897 | 90 of 524 | 79,197 | 290 s |
| exhaustive | 137,026 | 6,897 | 6,897 | 521 of 524 | 683,790 | 2766 s |
| GPS radius 120 m | 5,677 | 4,707 | 4,707 (68 %) | 521 of 524 | 673,731 | 120 s |
| GPS radius 160 m | 9,770 | 6,328 | 6,328 (92 %) | 521 of 524 | 681,878 | 231 s |
| GPS radius 200 m | 14,801 | 6,823 | 6,823 (98.9 %) | 521 of 524 | 683,645 | 286 s |

Exhaustive's verified pairs sit a median 96 m apart, 99 % within 201 m. The photo footprint at about
100 m above ground is roughly 140 x 105 m. On this survey the vocabulary tree placed 90 of 524 views,
the DJI set's failure again. A 200 m radius gives exhaustive's result within 0.02 % of the landmarks
for 11 % of the pairs and a tenth of the matching time. The FeatureMatching and SfM times of the GPS
legs were taken while a Linux bundle built beside them, so they are upper bounds.

**Whole jobs through the house-pc app** (Meshroom 2023.3 paired with the b036a bundle, RX 6750 XT,
standard preset, mesh only, 2026-09-27):

| job | FeatureMatching | whole job | views placed | landmarks |
|---|---|---|---|---|
| GPS radius 200 m | 9.8 min | 122 min | 521 of 524 | 681,153 |
| exhaustive | 101.1 min | 208 min | 521 of 524 | 681,061 |

Every other node took the same time within a minute. The 86 minutes are all matching.
PrepareDenseScene took 20 minutes of each job for 524 photos.


## 0.3.6: Meshroom 2025.1 (2026-09-27)

Meshroom 2025.1.0 (2025-08-18, AliceVision 3.3.0) from Zenodo record 16887472, checked against the
record's md5s. The Windows zip is 10.2 GB and the Linux tarball 14.3 GB. Most of that is not
Meshroom's pipeline: `plugins/mrSegmentation` (10.9 GB unpacked, a segmentation plugin nothing here
uses) and, on Linux, an 8.3 GB PySide6. Both were unpacked without the segmentation plugin.

**What changed for pairing.** The Meshroom layout is the one 2023.3 has:
- `aliceVision/bin` holds the node binaries;
- the frozen start-up sets `ALICEVISION_ROOT` and puts `aliceVision/bin` on PATH;
- the photogrammetry template runs the same twelve nodes (StructureFromMotion and PrepareDenseScene
  are still there; the new SfM chain is `develop`'s, not this release's).

The node descriptions moved. 2023.3 compiled them into `lib/meshroom/nodes/aliceVision/*.pyc`;
2025.1 ships them as source in `aliceVision/share/meshroom/aliceVision/*.py` (DepthMap 5.0,
FeatureExtraction 1.3, FeatureMatching 2.0, DepthMapFilter 4.0, Meshing 7.0, Texturing 6.0). The
option check passes on all nine paired nodes: the 0.3.5 package (AliceVision 3.4) takes every option
2025.1's own binaries take. Every attribute path the gate's configurations override still exists.

**Two `meshroom_batch` behaviours made the first run look empty** ("0 of 1 pipelines", 5 s, nothing
in the cache):

- **`--cache` is dropped, on Windows and Linux.** `executeGraph` calls `graph.save()` before
  computing. A graph with no project file is saved to `generateTempProjectFilepath()`, and
  `_setFilepath` then moves the cache next to it, to `<temp>/MeshroomCache`. Every run of the day
  computed there whatever `--cache` said. The same happens with `--save` and `--cache` together:
  that saves without keeping the file path. `--save <dir>/project.mg` alone keeps the cache at
  `<dir>/MeshroomCache`. Meshroom issue #2174 reports the symptom.
- **On Windows the process ends in half a second.** Meshroom's `setupInitScriptWindows.py` sets
  `ALICEVISION_LIBPATH` and `PYTHONPATH` and calls `os.execv(sys.executable, sys.argv)` unless the
  variable already lists its folder. On Windows `os.execv` starts a new process and ends the
  current one, so `meshroom_batch.exe` exited with 0 while its copy computed the graph. The harness
  then unpaired Meshroom under the running FeatureExtraction. The locked binary survived the
  delete, and the unpair still printed "restored". On Linux `execv` keeps the process, so only
  Windows is affected. It also passes arguments unquoted, so a path with a space would split.

`verify_end_to_end.py` now handles a 2025 layout (`aliceVision/share/meshroom` present) as follows:
- It runs with `--save <run>/project.mg` and reads `<run>/MeshroomCache`.
- On Windows it presets the two variables, so the process it waits on is the one that computes.
- Should a re-launch still happen, it finds the copy by parent PID and waits for it.

The Windows unpair now says "in use, NOT restored" when a running node holds its binary.

**The DepthMap override.** The 2023.3 override loads the compiled node beside it, and 2025.1 has
none, so pairing reported "no compiled DepthMap node" and DepthMap kept Meshroom's block of 12.
`meshroom-overrides/DepthMap.2025.py` does the same job for the source layout:
- the pairing scripts keep Meshroom's `DepthMap.py` as `DepthMap.py.meshroom`, a name the node
  loader does not import, and put the override in its place;
- the override loads that file and re-declares the class with blockSize 48 (version, attributes and
  command line are Meshroom's);
- `--unpair` puts Meshroom's file back.

The package's own `share/meshroom` node files (step 5z) are not an option here: they are AliceVision
3.4's (DepthMap 5.1, importing `pyalicevision`), written for a Meshroom newer than 2025.1. Step 5z
now ships every file in `meshroom-overrides/`.

**The gate on 2025.1** (Windows, RX 9070). The package was the 0.3.5 Windows AMD zip, with the 0.3.6
pairing script and `DepthMap.2025.py` added as a 0.3.6 package will carry them.
- **mini6: 15 of 15 configurations**, 7/7 ports each, DepthMap run with `--rangeSize 48`. Times are
  the 0.3.5 gate's on 2023.3 or a little lower (blast 155 s against 196 s, ds1 135 s against 156 s,
  the rest within 10 s).
- **41 views: base and verify both pass**, 7/7 ports, 3 textures each. They took 489 s and 846 s,
  against 393 s and 720 s on 2023.3.

**The Linux gate on 2025.1** (house-pc, RX 6750 XT, i3-4330). The package was the 0.3.5 Linux HIP
tarball as published, plus `DepthMap.2025.py`, with the 0.3.6 harness and `meshroom-pair.sh`, paired
into a scratch Meshroom 2025.1 under `/data/tests`.
- **mini6: 15 of 15**, 7/7 ports each. The override was installed and `--unpair` restored Meshroom's
  node. Times were within 10 s of 0.3.5's on 2023.3.
- **41 views: base 931 s and verify 1697 s, both pass.** 0.3.5 on 2023.3 took 531 s and 1281 s. The
  i3 pays for the host AC-RANSAC below more heavily than the Ryzen does (+75 % on base against +24 %).

**The Windows CUDA gate on 2025.1** (bench-pc, GTX 1080 Ti). The package was the 0.3.5 CUDA zip, with
the 0.3.6 pairing script and `DepthMap.2025.py`.
- **mini6: 15 of 15**, 7/7 ports each.
- **41 views: base 1352 s and verify 2284 s, both pass.** 0.3.5 on 2023.3 took 788 s and 1720 s.
  `--unpair` restored Meshroom's node.

All three packages that can be gated here pass on 2025.1. The Linux CUDA bundle is not gated: the
GTX card is in bench-pc, and house-pc, where the Linux CUDA gate ran, now holds an AMD card.

**Why the 41-view runs are slower: Meshroom 2025.1 raised two RANSAC defaults.** Per node, the
41-view base run on 2025.1 matches a same-day run on 2023.3 with the same package, to the second,
everywhere except two nodes:

| 41 views, base, RX 9070 | 2023.3 | 2025.1 | 2025.1 at 2023.3's iteration counts |
|---|---|---|---|
| FeatureMatching | 18.7 s | 63.2 s | 19.0 s |
| StructureFromMotion | 47.4 s | 135.2 s | 47.6 s |
| whole graph | 348 s | 483 s | 366 s |
| views placed | 41 | 41 | 41 |

The two command lines differ in exactly two options:
- FeatureMatching's `--maxIteration` went from 2048 to 50000. Geometric filtering, the host
  AC-RANSAC after the GPU matcher, went from 1.5 s to 35.3 s for 389 verified pairs instead of 387.
  The GPU brute-force matching itself took 10.4 s in both.
- StructureFromMotion's `--localizerEstimatorMaxIterations` went from 4096 to 50000. Resection went
  from 39 s to 127 s over the same 29 resection groups, with the same final bundle adjustment.

With both set back (`FeatureMatching:maxIteration=2048`,
`StructureFromMotion:localizerEstimatorMaxIterations=4096`), 2025.1 runs at 2023.3's speed. Under
2025.1's defaults, the host AC-RANSAC in geometric filtering and in resection becomes the largest
cost in both nodes. That is a lead for the next port, not a regression in the package.

**The same on the engine bay** (107 photos, DSP-SIFT, sfmbench cache, dev install, RX 9070;
`build/ransac/measure.py`):

| engine bay | at 2023.3's counts | at 2025.1's counts |
|---|---|---|
| FeatureMatching, whole node | 68.8 s | 530.8 s |
| of which geometric filtering | 21.4 s | 487.5 s (22.8x; 50000/2048 = 24.4x) |
| verified pairs / matches | 906 / 502,538 | 1,028 / 512,749 |
| SfM on the 2048-iteration matches (localizer 4096 / 50000) | 40.1 s | 75.4 s |
| SfM on the 50000-iteration matches (localizer 4096) | 107 poses, 139,058 landmarks, 37.9 s | |
| SfM on the 2048-iteration matches | 107 poses, 140,340 landmarks (localizer 4096); 140,319 (50000) | |

AC-RANSAC runs very nearly its whole budget here: geometric filtering grows 22.8x for 24.4x the
iterations. At 50000 it verifies 13 % more pairs, weak ones: 2 % more matches. SfM places the same
107 views either way, with landmarks within the run-to-run spread of an unseeded SfM. On this set,
2025.1's defaults cost 8 minutes of matching and 35 s of SfM for the same reconstruction. The house-pc
app offers the 2023.3 counts as an option ("RANSAC iterations"). A device port of the hypothesis loop
would remove most of the 487 s.

**The False Door** (884 photos, SIFT, the 0.3.3 end-to-end cache's features and 25,354 pairs; dev
install, RX 9070):

| False Door | maxIteration 2048 | maxIteration 50000 |
|---|---|---|
| FeatureMatching, whole node | 489.5 s | 2617.3 s |
| of which geometric filtering | 104.7 s | 2233.9 s (21.3x) |
| verified pairs / matches | 4,954 / 5,158,159 | 8,499 / 5,718,065 |

Here the larger budget verifies 72 % more pairs but only 11 % more matches, so the added pairs are
small ones: pairs whose few inliers 2048 hypotheses rarely sample. SfM on them (localizer 4096,
the same options) placed **830 of 884 views with 1,268,756 landmarks in 1960 s**. Twelve earlier runs on
the 2048-iteration matches placed 815 to 833 views, with 1.338 to 1.355 million landmarks, in 1139 to
2368 s. The extra pairs bought no registration and about 5 % fewer landmarks, for 35 more minutes of
matching. On neither set does 2025.1's matching budget improve the reconstruction.

## 0.3.6: tooling debts (scripts only, 2026-09-27)

Four items from the roadmap's "Packaging and platforms", in scripts only. Nothing here was run on a
GPU. Each was checked with `py_compile` / `bash -n` and a fake-input run (the commit messages give
the cases); the owner's next gate run is the first real test. What changes for that run:

- **`scripts/verify_end_to_end.py`.**
  - `cpufallback` now runs FeatureExtraction on the CPU (`forceCpuExtraction=True`). Until now it
    kept PopSIFT on the card while every `CHESHIRE_GPU_*` switch was off. Its FeatureExtraction
    markers are empty, as in `defaults`: the node is held to being the paired binary only.
  - PrepareDenseScene is gated. It is in `BINARY`, so Meshroom's own binary there aborts the run.
    Its marker is step 6n's "cheshire: 8-bit images read directly into float RGB(A)" in every
    configuration, since no switch in the matrix turns the direct read off. A photo set the direct
    read does not take (not three-channel 8-bit JPEG or PNG) would fail this marker.
  - StructureFromMotion's step 5k warning ("local BA graph: N edges ... were skipped") fails any
    configuration that logs it.
  - A node in `BINARY` that the pairing script declines ("not paired", or the Linux script's
    "DepthMap paired only") stops the matrix before `meshroom_batch` starts, with the script's line.
- **`CHESHIRE_POPSIFT` is auto in the HIP build scripts.** Unset, it is ON when a PopSift install is
  at `CHESHIRE_POPSIFT_DIR` (Windows) or `CHESHIRE_POPSIFT_INSTALL` (Linux), or at their defaults,
  and OFF with a warning otherwise. `package_windows.py` (import table through `llvm-objdump`) and
  `wsl-pack-bundle.sh` (`readelf -d` on `libaliceVision_feature.so*`) now refuse a package whose
  feature library does not link PopSift, as `package-cuda.ps1` already did.
- **`scripts/linux/meshroom-pair.sh`.** The generated launcher reads `CHESHIRE_BACKEND` (and the
  older `CHESHIRE_DEPTHMAP`) from the bundle's `env.sh` before deciding the backend, when the node's
  environment sets neither. It reads them in a subshell rather than sourcing the whole file first,
  because `node-amd-setup.sh` writes `LD_LIBRARY_PATH=<bundle>/lib` into `env.sh` and that must not
  reach Meshroom's own binary. Nodes must be re-paired to get the new launcher.
- **`scripts/verify_packages.py` is retired.** It looked for one string literal per change inside the
  release archives. It stopped growing at v0.2.16 with five markers, and later steps added a couple
  by hand (the visibility votes, 6u). The end-to-end gate proves more than a literal can: that the
  paired binary ran and that its port announced itself, per node, on a real pipeline. The stage gates
  cover the per-node outputs. Nothing called the script.

## The 0.3.6 release gate (2026-09-28)

All four packages built at 863384e (`build/chain-036.cmd`, then `chain-036-resume.cmd` after the
Windows CUDA link lost to a file lock; the second attempt built clean). The packers' checks pass,
including the new ones from the tooling-debts task: every package's feature library links PopSift
(Windows: `aliceVision_feature.dll` imports `popsift.dll`; Linux: `libaliceVision_feature.so.3.4`
needs `libpopsift`). Every package carries `share/cheshire/meshroom-overrides/DepthMap.2025.py`.

Each package was gated as built, on its own card. The full gate runs through Meshroom 2025.1: the
15-configuration mini6 matrix, then 41 views with `base` and `verify`. A shorter mini6 check runs
through Meshroom 2023.3: `base`, `defaults`, `cpufallback`, `verify`. Every run is scored at eight
nodes: the seven GPU ports and the SfM node as before, and PrepareDenseScene, new in this gate.

| package | hardware | 2025.1: mini6 | 2025.1: 41 views | 2023.3: mini6 |
|---|---|---|---|---|
| Windows AMD | RX 9070 (rocm7.2 gfx12-generic payload) | 15/15 at 8/8 | base 478 s, verify 832 s | 4/4 at 8/8 |
| Windows AMD | RX 5500 XT (hip6.2 gfx1012 payload, bench-pc) | 15/15 at 8/8 | base 1469 s, verify 2377 s | 4/4 at 8/8 |
| Linux AMD | RX 6750 XT (house-pc) | 15/15 at 8/8 | base 916 s, verify 1672 s | 4/4 at 8/8 |
| Windows CUDA | GTX 1080 Ti (bench-pc) | 15/15 at 8/8 | base 1343 s, verify 2273 s | 4/4 at 8/8 |
| Linux CUDA | GTX 1080 Ti (house-pc) | 15/15 at 8/8 | base 966 s, verify 1707 s | 4/4 at 8/8 |

Pairing installed the matching DepthMap override on each Meshroom, and `--unpair` restored Meshroom's
own node in every run. The last two rows ran on 2026-09-28 after a card swap (the GTX 1080 Ti into
house-pc, the RX 5500 XT into bench-pc), so every package and both halves of the Windows AMD package
are gated. 105 runs, none failed.

## 0.3.7: OpenMP's threads after a parallel region (step 7g), and where incremental SfM's time goes (7h) (2026-09-29)

**Spinning threads.** After a parallel region, LLVM's OpenMP runtime (libomp140 on Windows) keeps its
threads spinning for `KMP_BLOCKTIME`, 200 ms by default, before they sleep. Beside a single-threaded
pass that follows a region, that spinning costs time. It was found in the bundle adjustment's
factorisation (docs/notes/ba-own-solver.md, 4d), where it took the large solves from 10.7 s to
15.7 s. It costs the rest of SfM too.

Step 7g sets `KMP_BLOCKTIME=0` in incrementalSfM's environment, at the top of `aliceVision_main`,
before the runtime starts:
- **Why the environment.** It reaches every team. `kmp_set_blocktime` reached only the main thread's
  teams, and missed the loading's and the colours' work. It stays as the fallback if the runtime has
  already started, and the log line says which took.
- **Switches.** `CHESHIRE_OMP_BLOCKTIME=<ms>` sets another time. A `KMP_BLOCKTIME` of the user's own
  is left alone.
- **Scope.** Windows only. On Linux it is not measured yet: GCC's libgomp (the CUDA build) spins for
  far shorter by default, while the HIP build's clang may link LLVM's runtime.

Only the timing changes. False Door, the same build one run after another (a browser was using
about 1.5 cores throughout), all with the same output `a5cc8437…`:

| threads after a region | SfM wall |
|---|---|
| spin 200 ms (the runtime's default) | 523.4 s |
| sleep, `kmp_set_blocktime(0)` from main | 506.6 s |
| sleep, `KMP_BLOCKTIME=0` in the environment (7g) | 498.3 s |

The environment's extra 8 s is in the loading (6.4 s) and the colours (2.3 s). The 41-view and
engine-bay digests are unchanged.

**Where the time goes (7h).** `CHESHIRE_SFM_PROFILE=1` prints two lines:
- the engine's phases, each timed at its call sites where they do not nest, with "other" the rest;
- the program's steps.

False Door with 7g, 490.6 s in all:

| program step | s |
|---|---|
| features (read) | 41.5 |
| matches (read) | 3.1 |
| reconstruction | 385.6 |
| colours (every image read) | 55.2 |
| report, save, alignment | 5.1 |

| reconstruction phase | s | calls |
|---|---|---|
| resection (parallel over the group's views) | 88.3 | 128 |
| bundle adjustment's solve (`adjust`) | 64.7 | 64 |
| after each solve: outliers, unstable views (5s) | 61.7 | 64 |
| triangulation | 47.5 | 63 |
| choosing the next views | 36.8 | 131 |
| local-BA graph: update 21.5 s, states 15.5 s | 37.0 | |
| initial pairs | 16.2 | 1 |
| tracks | 14.6 | 1 |
| register changes | 9.1 | 61 |
| resection of new cameras (serial), apply | 4.3 | |
| statistics | 4.9 | 1 |
| other | 0.5 | |

The solver is now 17 % of the reconstruction. The next targets are around it, largest first:
- resection;
- the passes after each solve;
- the colours: 884 JPEG decodes, where CheshireJPG decodes identically;
- triangulation;
- the features' read;
- the next-views scoring;
- the local-BA graph.


## 0.3.7: incremental SfM around the solver, exact (steps 7i-7m) (2026-09-29)

The 7h profile pointed at host code around the solver that grew with the scene. Each step keeps
every value and every order that reaches the output: the output of all three sets is unchanged
(41 `801d1fb7…`, engine bay `814ae0d5…`, False Door `a5cc8437…`). Each False Door pair below ran
one after the other on the same box, and each step is judged by its own phase. Other phases move by
±5 s from run to run here.

- **7i, the colours' view assignment.**
  - **Before:** `colorizeTracks` gives each landmark the first view, in order of how many landmarks
    each view sees, that observes it. It found that view by re-scanning every unassigned landmark
    once per view: about 560 M observation lookups on the False Door, on one thread.
  - **Now:** each view's rank is looked up once, and one pass over the landmarks makes the same
    assignment, with each view's landmarks in the same order.
  - **False Door:** colours 47.2 s → 36.7 s.
- **7j, the landmark-id sets.** Three places copied every landmark's id into a `std::set`, 1.35 M
  nodes, to intersect a view's track list with it:
  - the next-views scoring, once per call;
  - every view's resection, inside the parallel loop;
  - the local-BA graph's update.

  **Now:** a byte per track id is built once per use, and the view's sorted list is filtered by it.
  The result is exactly the ids `std::set_intersection` copied, in the same order.
  - **A first version was worse in two places.** It intersected against the landmarks' map
    directly. That helped the resection, but walked the map's large nodes once per view elsewhere:
    next views 30.5 s → 44.2 s, graph update 18.2 s → 42.6 s.
  - **With the mask:** next views 29.4 s → 7.7 s, resection 79.0 s → 42.1 s, graph update
    16.7 s → 3.5 s.
  - **A side effect:** the solver's own phase no longer swings between 58 s and 74 s from run to
    run. The million-node sets had churned the heap it allocates from.
- **7k, triangulation's critical sections.**
  - **Where the time went:** tracks were handed out one `omp single` at a time, and each result was
    inserted under a critical section. Each observation also copied the camera's `shared_ptr`
    (shared atomic counts) and recomputed the view's projection matrix.
  - **Now:** a slot per track, each view's data once per call, and the results applied in track
    order.
  - **False Door:** triangulation 38.6 s → 34.8 s. Most of the rest is the LO-RANSAC arithmetic.
- **7l, the full passes after each solve** (`postAdjust.inc`).
  - **Where the time went:** upstream's pixel-residual filter and the observation count behind
    `eraseUnstablePoses` ran on one thread over every observation, with map lookups and a pose copy
    each.
  - **Now:** each view's pose and intrinsic are read once, and the verdicts and integer counts are
    made on every core. The erasures are applied in upstream's order.
  - **False Door:** post-adjust 54.5 s → 15.1 s.
- **7m, the local-BA graph's landmark states.**
  - **Where the time went:** each observation's view was looked up twice, on one thread, and every
    landmark was inserted from the root into a map that is cleared each call.
  - **Now:** each view's state once, the states on every core, and the map refilled in id order
    with an end hint.
  - **False Door:** graph states 12.6 s → 6.7 s.

**The whole run.** False Door on this box, all with the same output:
- The build before 7i took 436-448 s.
- The installed build with 7i-7m takes 284.3 s, 36 % less.
- Bundle adjustment is 46.9 s of it.

MSVC (the Windows CUDA tree) compiles the changed libraries.

| program step | s | | reconstruction phase | s |
|---|---|---|---|---|
| features (read) | 45.9 | | the solve (`adjust`) | 55.9 |
| colours | 36.9 | | resection | 38.0 |
| reconstruction | 192.8 | | triangulation | 32.8 |
| report, save, alignment, matches | 7.2 | | post-adjust | 14.9 |
| | | | initial pairs | 11.6 |
| | | | tracks | 11.4 |
| | | | register changes | 7.7 |
| | | | graph states | 6.7 |
| | | | next views | 4.5 |
| | | | statistics | 4.2 |
| | | | graph update | 2.5 |

The largest pieces left outside the solver are:
- the features' read, 46 s;
- resection's arithmetic;
- the colours' image reads;
- triangulation's LO-RANSAC.


## The 0.3.7 release gate (2026-09-30)

All four packages were built at 3104a30 (`build/chain-037.cmd`). The first launch, at bad0dfa, stopped at
22:51 with ninja's "User interrupt" partway through the hip6.2 payloads, most likely from a closed console
window. It was relaunched from WMI with the window hidden, after the solver was named Grin.

The packers' checks pass, and three more were made by hand:
- all 11 GPU targets of the Windows AMD bundle carry their own `aliceVision_sfm_bundle.dll`, each
  with its target's code object and Grin's announce line;
- both Linux bundles carry Grin, the HIP one for 19 targets and the CUDA one for sm_61;
- every package copied to a test box matched its SHA-256.

The gate is 0.3.6's, and the StructureFromMotion node now also has to print
`cheshire: BA solver: Grin, Cheshire's own, for the Schur solves`. Every run is scored at eight nodes.

| package | hardware | 2025.1: mini6 | 2025.1: 41 views | 2023.3: mini6 |
|---|---|---|---|---|
| Windows AMD | RX 9070 (rocm7.2 gfx12-generic payload) | 15/15 at 8/8 | base 433 s, verify 781 s | 4/4 at 8/8 |
| Windows AMD | RX 5500 XT (hip6.2 gfx1012 payload, bench-pc) | 15/15 at 8/8 | base 1398 s, verify 2312 s | 4/4 at 8/8 |
| Linux AMD | RX 6750 XT (house-pc) | 15/15 at 8/8 | base 881 s, verify 1632 s | 4/4 at 8/8 |
| Windows CUDA | GTX 1080 Ti (bench-pc) | 15/15 at 8/8 | base 1277 s, verify 2216 s | 4/4 at 8/8 |
| Linux CUDA | GTX 1080 Ti (house-pc) | 15/15 at 8/8 | base 931 s, verify 1662 s | 4/4 at 8/8 |

105 runs, none failed. Every 41-view time is below 0.3.6's on the same card (478/832, 1469/2377,
916/1672, 1343/2273 and 966/1707 s).

**What repeatable SfM showed.** On all three AMD cards, the 41-view `base` and `verify` runs now give
the same depth maps. In 0.3.6 every pair differed, because SfM did. On both CUDA packages they still
differ, and a stage-by-stage hash of the two Windows runs puts the first difference in
FeatureExtraction:
- the 41 `.feat` files are identical;
- one descriptor of about 970,000 differs (view 1827716366, index 6213, 88 of 128 bytes, the same bins
  at different magnitudes);
- on Linux, one descriptor each in four views.

That points at a race in PopSIFT's CUDA descriptor kernel. The HIP build repeats exactly. It is filed
for a separate investigation.

## 0.3.8: PopSIFT's CUDA descriptor race, found and fixed (2026-09-30)

The 0.3.7 gate's CUDA runs repeated everything but one descriptor in about 970,000 (above). The cause
is upstream PopSIFT's `normalize_histogram` (`s_desc_normalize.h`), on both backends:
- It runs one warp per descriptor, 32 to a block. In the last block the warps past the end are
  clamped onto the last descriptor so that their shuffles have data, and `ignoreme` was meant to keep
  them from writing it. It was computed after the clamp, so it was never true: those warps normalised
  the last descriptor again, in place, racing the warp that owns it.
- So at most one descriptor per image can come out wrong, and only when the image's descriptor count
  is not a multiple of 32. Which descriptor is last in the device buffer changes between runs (the
  extrema are appended with atomics), which is why the affected row moves.
- The Linux evidence fits all of it. Four views, one descriptor each, with counts 23692, 23362, 23628
  and 23861 (12, 2, 12 and 21 past a multiple of 32). In each pair one copy is the other passed through
  RootSIFT a second time: RootSIFT applied to the stored bytes reproduces it to within 1 in most bins,
  the rest being the rounding of the stored intermediate, which the square root magnifies in small
  bins.

The fix decides `ignoreme` before the clamp (`scripts/apply_popsift_patch.py`, step 3e). It is not
about HIP, so the CUDA builds take it too: they reset PopSIFT to upstream and now apply
`--fixes-only`, this step and the extremum counter's (3c).

| build | runs (41 views, 968,726 descriptors) | result |
|---|---|---|
| the shipped 0.3.7 Linux CUDA library, GTX 1080 Ti (house-pc) | 3 | run 1 against 2: 2 descriptors differ, in 2 views; run 1 against 3: 1 |
| the same bundle with the fixed library | 4 | identical, every `.desc` and `.feat` |
| the fixed library against the 0.3.7 gate's two runs | | identical but for the gate's four racy descriptors, where the fixed build gives the copy normalised once, in each case |
| the published 0.3.7 Windows AMD bundle, RX 9070 | 2 | identical |
| the same bundle with the fixed `popsift.dll` (gfx1201) | 1 | identical to the published bundle's: the HIP bytes are as they were |

The AMD cards never showed the race: every AMD base and verify pair in the 0.3.7 gate was identical.
The Windows CUDA library takes the same fix through `build-popsift-cuda.cmd` and gets its hardware
check at the 0.3.8 gate, since no Windows box has an NVIDIA card today. The release chains must
rebuild PopSIFT CUDA on both systems before AliceVision: WSL's `/opt/popsift-cuda` and
`build/popsift-cuda-install` still hold 0.3.7's library. The test build was
`build/popsift038/wsl-build.lf.sh` (a scratch prefix), the runs `/data/tests/popsift-race` on
house-pc and `build/popsift038/hip_check.py` here.

## 0.3.8: SfM around Grin, continued - the features' read and the colours' read (2026-09-30)

**8a, the features' read.** Every node that reads features parses the `.feat` text through an
`istream_iterator` over `operator>>`, four floats a keypoint. Measured on its own over the 884-view False
Door's 20,023,190 features (`hip/tests/featread/featbench.cpp`, files already in the page cache):

| parser | MSVC STL (clang-cl), 1 thread | 12 threads | libstdc++ (GCC 13, WSL), 1 thread | 12 threads |
|---|---|---|---|---|
| `istream_iterator` (upstream) | 30.75 s | 42.54 s | 30.15 s | 5.32 s |
| one read + `std::from_chars` | 5.72 s | 0.79 s | 8.33 s | 2.12 s |

MSVC's stream extraction gets slower with more threads, because it serialises on its locale. Both parsers
round correctly, and the two agree on every one of the 80 M floats under both libraries. The WSL runs read
through `/mnt/d`, which is slow; native Linux reads are faster.

The step reads a PointFeature file in one go and parses it with `from_chars`. It stops where the stream
stops: at the end of the file, or at the first token the stream would not take (`nan` and `inf`
included). It keeps only whole features, as the stream does. `CHESHIRE_FEAT_READ=0` restores upstream's
read.

**8b, the colours' read.** `colorizeTracks` reads each view's photo whole as `Image<RGBColor>`. Upstream's
`readImage` decodes it into a float `ImageBuf` (144 MB for 12 MP), converts no colour (sRGB to sRGB), and
hands the pixels back as 8 bits. When OpenImageIO's uint8 to float to uint8 round trip gives every value
back, those are the decoder's bytes. Step 6n's direct read now copies them straight out in that case; it
checks the round trip once per process, through the same calls, for all 256 values. The check mode
compares 8-bit images byte for byte.

Development install, RX 9070 box, `scripts/sfmbench.py` (Meshroom 2023.3's SfM options):

| set | digest (sfm.abc, cameras) | features | colours | SfM wall |
|---|---|---|---|---|
| 41 views, 0.3.7 | `801d1fb7`, `361ddd5c` | 1.50 s | 1.19 s | 27.2 s |
| 41 views, 8a + 8b | the same | 0.04 s | 0.52 s | 23.0 s |
| engine bay, 0.3.7 | `814ae0d5`, `af98cde3` | 4.02 s | 2.03 s | 24.1 s |
| engine bay, 8a + 8b | the same | 0.09 s | 0.80 s | 17.9 s |

With `CHESHIRE_READ_DIRECT_CHECK=1`, the direct read and upstream's path agreed on 38 of 38 and 98 of 98
images.

The 884-view False Door, back to back in one session on an idle box: the published 0.3.7 bundle
(`build/run-037.cmd`, through its `cheshire-run.cmd`) against the development install with 8a and 8b.

| False Door | SfM wall | features | colours | reconstruction | digests (sfm.abc, cameras) |
|---|---|---|---|---|---|
| 0.3.7 bundle | 307.8 s | 44.1 s | 38.0 s | 216.2 s | `a5cc8437`, `e05e2fa4` |
| 8a + 8b | 240.9 s | 0.84 s | 14.4 s | 216.7 s | the same |

The node is 22 % faster with the same bytes. The 0.3.7 run is 24 s slower than the release notes'
284 s; that is the same box on another day, the drift those notes describe. What is left of the colours
is about the disk: the 884 photos are 4.4 GB on a SATA SSD.

The reconstruction's own profile now reads:

| phase | time |
|---|---|
| adjust (Grin) | 60.7 s |
| resection | 42.1 s |
| triangulation | 40.3 s |
| post-adjust | 16.7 s |
| initial pairs | 12.8 s |
| tracks | 12.5 s |
| register | 8.9 s |
| graph states | 7.4 s |
| next views | 5.1 s |
| statistics | 4.4 s |
| graph update | 2.9 s |

## 0.3.8: AC-RANSAC skips the models that cannot win (step 8d, 2026-09-30)

`ACRANSAC` uses `bestNFA`'s answer only when it beats the best model so far, and 0.036 % of the
models do. Step 8d bounds the NFA a model could reach from a histogram of its unsorted residuals,
and skips the residual sort and the scan when the bound cannot beat `minNFA`. The method is in
[docs/15](15-acransac-cpu.md), "The question the caller asks". `scripts/fmbench.py` runs
FeatureMatching from `sfmbench.py`'s cache with the development install. It reports the log's
geometric-filtering time and one digest over the match files.

41 views at Meshroom 2025.1's 50,000 iterations, RX 9070 box, alternating runs:

| run | geometric filtering | wall | digest |
|---|---|---|---|
| `CHESHIRE_ACR_BOUND=0` | 36.9 s | 43.4 s | `2b4452a79d79b819` |
| bound | 18.8 s | 25.4 s | the same |
| `CHESHIRE_ACR_BOUND=0` | 36.7 s | 43.3 s | the same |
| bound | 18.8 s | 25.4 s | the same |

The digest is `base50k`'s, taken before this round's AC-RANSAC work. With
`CHESHIRE_ACR_BOUND_CHECK=1`, every skipped model is sorted and scanned as well:

| set, iterations | models skipped | below `minNFA` in `bestNFA` | digest |
|---|---|---|---|
| 41 views, 50,000 | 23,526,805 of 23,607,436 (99.66 %) | 0 | `2b4452a79d79b819` |
| 41 views, 2048 | 1,063,685 of 1,074,515 (98.99 %) | 0 | `64397675b2fed423`, the same as before |
| engine bay, 2048 | 22,888,066 of 22,991,443 (99.55 %) | 0 | `f063ea6d15c0aa2e`, the same as with the bound off |

The engine bay at 2048 iterations goes from 20.2 s to 14.6 s of geometric filtering (one run each).

**Tried first, and not kept.** A bounded scan with the same answers: a cheap `log10` of proven error
bounded `nfa(k)` at every rank, and the library's `log10` ran only where the bound reached the
smallest upper bound. It was exact on all 23.6 M models, but geometric filtering took 38.8 s against
about 40 s. On its own it ran at 0.61-0.79x the full scan's speed, because the library's `log10`
costs only 4-7 ns an element, no more than the cheap one. The sort and the scan had to go for
most models, not just get cheaper.

After 8d, `kernel.fit` (the 7-point solver) is 63.5 % of AC-RANSAC's CPU time and `kernel.errors`
19.8 % (docs/15, "What is left").

## 0.3.8: the epipolar residuals in one vectorised loop (step 8e, 2026-09-30)

The loop computes `FundamentalEpipolarDistanceError` for every correspondence of a model. It repeats
Eigen's arithmetic operation for operation, and a self-test on first use falls back to the
per-point loop wherever a bit differs ([docs/15](15-acransac-cpu.md), "The residual loop,
vectorised"). With `CHESHIRE_ACR_RESIDUALS_CHECK=1`, every residual is computed both ways and
`error()`'s value is kept:

| run | residuals identical | digest |
|---|---|---|
| 41 views, 50,000 iterations | 7,861,787,651 of 7,861,787,651 | `2b4452a79d79b819`, `base50k`'s |
| engine bay, 2048 iterations | 2,264,668,180 of 2,264,668,180 | `f063ea6d15c0aa2e`, as before |

41 views at 50,000 iterations, alternating runs with the 8d bound on:

| run | geometric filtering | wall | digest |
|---|---|---|---|
| `CHESHIRE_ACR_RESIDUALS=0` | 18.9 s | 25.5 s | `2b4452a79d79b819` |
| 8e | 16.3 s | 22.8 s | the same |
| `CHESHIRE_ACR_RESIDUALS=0` | 18.9 s | 25.4 s | the same |
| 8e | 16.4 s | 23.0 s | the same |

SfM's initial pair uses the same kernel. `scripts/sfmbench.py` gives the same digests with the loop
in use: 41 views `801d1fb7`, `361ddd5c` and engine bay `814ae0d5`, `af98cde3`.

With 8d and 8e, geometric filtering on 41 views at 50,000 iterations went from 36.8 s to 16.35 s
in this round. `kernel.fit` is now 74.8 % of AC-RANSAC's CPU time (docs/15, "What is left").

## 0.3.8: PrepareDenseScene on a four-thread host, and its EXR chunks deflated with libdeflate (step 8f, 2026-09-30)

**Measured first, on the host the roadmap names.** house-pc (i3-4330: two cores, four threads; 14 GB;
`/data` on a SATA SSD; GTX 1080 Ti) ran the zoo job again with the 0.3.7 CUDA bundle. The job was
`zoo-pds-038`: 524 photos at 4896x3672, GPS pairing at 200 m, the app's 2023.3 RANSAC counts, and
mesh only.

| zoo, mesh only | 0.3.6 (RX 6750 XT, HIP bundle) | 0.3.7 (GTX 1080 Ti, CUDA bundle) |
|---|---|---|
| FeatureExtraction | 295.9 s | 258.7 s |
| FeatureMatching | 585.7 s | 483.9 s |
| StructureFromMotion | 510.6 s (521 of 524 views) | 287.0 s (522 of 524) |
| PrepareDenseScene | 1245.6 s | 930.3 s |
| DepthMap | 3660.8 s | 3403.0 s |
| DepthMapFilter | 620.9 s | 531.2 s |
| Meshing + MeshFiltering | 384.9 s | 469.0 s |
| total | 7311.7 s | 6370.2 s |

PrepareDenseScene is all CPU there. A sampler ran every 5 s through the stage: user 91.8 %, system
6.3 %, iowait 0.3 %, idle 1.6 %. The SSD wrote 34.1 GB at 40.5 MB/s on average and was busy
16.1 % of the time. The stage costs about 7.1 thread-seconds per image. The job's first chunk (40
views) was run again with `CHESHIRE_PDS_PROFILE=1`; thread-seconds, 4 threads unless named:

| run | wall | read | undistort | write | output |
|---|---|---|---|---|---|
| as released (ZIP level 1) | 72.0, 70.7 s | 49.3, 43.1 | 70.1, 64.7 | 148.4, 162.2 | 2696 MB |
| `CHESHIRE_PDS_EXR_COMPRESSION=none` | 37.5 s | 39.6 | 61.1 | 40.7 | 5489 MB |
| `CHESHIRE_PDS_THREADS=2` | 80.7 s | 23.7 | 40.6 | 93.4 | 2696 MB |
| `CHESHIRE_GPU_JPEG=1` (all 40 decoded on the device) | 70.0 s | 41.0 | 65.7 | 160.4 | 2696 MB |

The compression is 47 % of the stage. The Linux bundle's OpenEXR is 3.1, which deflates with the
system's zlib (1.3). On the job's own EXR chunks, that zlib compresses at level 1 at 73 MB/s on the
i3. libdeflate, already in the bundle for the reader (6v), does it at 180 MB/s, with slightly smaller
output (68.5 against 70.4 MB for one image). The level really is 1: re-compressing 64 chunks at each
level reproduces the file's bytes at level 1 only. Measured on Windows first, the two looked equal,
but Windows' Python ships zlib-ng. Two other ideas were rejected:
- decoding the JPEGs on the device saves nothing; the read is not decode-bound;
- the colour conversion is not per channel, so a table of the 256 byte values cannot replace it.
  The config's sRGB to scene-linear path composes two matrices that are not exactly inverse. On all
  16,777,216 RGB triples, 16.66 M, 16.74 M and 16.77 M outputs (R, G, B) differ from per-channel
  tables. Besides, it costs 9 ns a pixel.

**8f.** Inside PrepareDenseScene (an `ExrDeflateWriteScope` around its writes), `writeImage` hands a
ZIP or ZIPS EXR to `cheshireWriteExrDeflate` (`hip/port/sgm_fused/cheshireExr.cpp.txt`):
1. OpenImageIO writes the image uncompressed into memory, header and lines as it writes them.
2. The ZIP chunks are built from those lines as OpenEXR's zip compressor builds them: the lines in
   order, the interleave, the predictor, and a zlib stream from libdeflate at the same level.
3. A chunk that does not come out smaller is stored as it is. Only the compression byte of the
   header changes.

It is on by default with OpenEXR before 3.2 (the Linux bundles); OpenEXR 3.2 and later deflate with
libdeflate themselves. `CHESHIRE_EXR_DEFLATE_WRITE=0|1` overrides that, and
`CHESHIRE_EXR_DEFLATE_WRITE_CHECK=1` reads every file back through OpenEXR. Every other EXR writer
keeps OpenEXR's, so depth maps and the rest are the bytes they were.

| check | result |
|---|---|
| Windows, forced on, 41 views, against OpenEXR 3.4's own writer | 41 of 41 files identical as bytes |
| Windows, the check | 41 of 41 read back identical |
| house-pc (0.3.8 development CUDA bundle), zoo chunk, libdeflate against zlib | 40 of 40: headers identical, every line identical |
| house-pc, the check | 40 of 40 read back identical through OpenEXR 3.1 |
| house-pc, the zoo job's 0.3.7 files against 0.3.8 with zlib | 40 of 40 identical as bytes |
| house-pc, the zoo job's 0.3.7 files against 0.3.8 with libdeflate | 40 of 40: headers identical, every line identical |

On Windows the files are byte for byte what OpenEXR 3.4 writes, so its libdeflate at level 1 is the
same compressor. On Linux they are now what the Windows packages write from the same pixels.

Timing on house-pc, the same chunk, alternating:

| zoo chunk, 40 views | wall | write (thread-seconds) | output |
|---|---|---|---|
| zlib (`CHESHIRE_EXR_DEFLATE_WRITE=0`) | 70.5 s, 71.8 s | 154.4, 155.1 | 2696 MB |
| libdeflate (8f) | 51.1 s, 51.3 s | 99.0, 98.7 | 2625 MB |

That is 28 % off the stage. On the whole zoo job, PrepareDenseScene would go from about 930 s to
about 670 s. On Windows (OpenEXR 3.4) the path stays off. Forced on there it was slower (11.2
against 9.5 s on 41 views): OpenEXR spreads one file's chunks over its pool, while the path
deflates them in the image's own thread. That is enough on four threads, where every thread is
busy with an image.

What is left on the four-thread host is the undistortion (about a quarter of the stage). It
samples in double precision without fused operations, so the device could run it to the same bits
while the GPU sits idle.

## 0.3.8: PrepareDenseScene's undistortion on the device (step 8g, 2026-09-30)

`UndistortImage` samples the photo at each undistorted pixel's distorted position through
`Sampler2d<SamplerLinear>`. That means four neighbours weighted in double precision, summed per
channel, divided by the weight where a border cut it short, and rounded back to float. CheshireRemap
(`hip/port/cheshireremap`, built into the image library like CheshireJPG) runs that arithmetic with
one device thread per output pixel. Every product, sum and quotient is a separate IEEE operation
rounded to nearest, as on the host: clang's `fp contract(off)` pragma covers the HIP build, and
`--fmad=false` the CUDA one. HIP's `__dadd_rn` family is plain operators in HIP's own headers,
which clang's HIP default fuses across, so it is not used.

The coordinate map is converted on the host once per map and source size, to what the sampler
actually receives:
- the two coordinates as floats, since Sampler2d's parameters are floats;
- NaN where `Image::contains` rejects the pixel. It truncates the double coordinates to int, so
  -0.5 counts as inside.

Inside PrepareDenseScene (`DeviceUndistortScope`) an RGBA float image with a cached map goes to a
remapper. A remapper holds the map, a source and a destination image on the device, and pinned
staging. There are up to min(hardware threads, 4) of them, and no more than fit in 60 % of the device
memory free at the first image. An image thread that finds none free runs the CPU loop instead of
waiting.

| check | images identical to the CPU loop | output files |
|---|---|---|
| RX 9070 (HIP, gfx12-generic), 41 views, forced on | 41 of 41; 32 of 32 and 31 of 31 with 4 remappers | 41 of 41 identical as bytes |
| house-pc GTX 1080 Ti (CUDA 12.9, sm_61), the zoo's first 40 views | 40 of 40 | 40 of 40 identical as bytes |

On house-pc (i3-4330, four threads; 0.3.8 development bundle with 8f; the zoo chunk, alternating):

| undistortion | wall | undistort (thread-seconds) |
|---|---|---|
| CPU loop (`CHESHIRE_UNDISTORT_DEVICE=0`) | 51.8 s, 50.7 s | 59.1, 62.4 |
| device (the default there) | 43.5 s, 44.1 s | 26.6, 26.9 |

That takes 15 % off the stage, and with 8f 38 %: the chunk went from 71.2 s (0.3.7) to 43.8 s. On
the whole zoo, PrepareDenseScene would go from about 930 s to about 570 s.

On the RX 9070 box (twelve threads) the device path does not pay. A remap there costs about 40 ms
of upload and 70 ms of kernel and download per 12 MP image with four running, and more with twelve
threads competing for memory bandwidth. That is more than that CPU's loop, about 140 ms of the
undistort phase, whose other half is the output image's allocation and fill, the same on both
paths. The 41 views took 9.4-9.8 s on the device against 8.9-9.2 s on the CPU. So the device takes
the undistortion by default only with four hardware threads or fewer, where the loop is the costly
side. `CHESHIRE_UNDISTORT_DEVICE=1|0` overrides that, `CHESHIRE_UNDISTORT_DEVICE_REMAPPERS` sets the
bound, and `CHESHIRE_UNDISTORT_DEVICE_CHECK=1` runs both and compares. The Windows CUDA build first ran
it at the 0.3.8 gate, on the GTX 1080 Ti: 4 of 4 images identical on mini6 and 40 of 40 on 41 views.

## The 0.3.8 release gate (2026-09-30)

All four packages were built at e8cd509 (`build/chain-038.cmd`), in 56 minutes: the Windows side took 36, nine
hip6.2 targets of it 19 now that the generator keeps unchanged files' times, and the Linux HIP bundle 16. The
Linux CUDA build had nothing left to compile after the development bundle of the same sources and only relinked
against the rebuilt PopSIFT. Both PopSIFT CUDA libraries were rebuilt first, and both build logs show the race
fix in.

The packers' checks pass, and four more were made by hand:
- each of the Windows AMD bundle's 11 targets carries an `aliceVision_image.dll` with its own code object and
  CheshireRemap's kernel, and the zip lists the eight untested targets in `gpu/<family>/UNTESTED` again
  (hip6.2: seven, rocm7.2: `gfx11-generic`);
- the Windows CUDA image library carries CheshireRemap's sm_61 code, and the Linux HIP build compiled it for
  its 19 targets;
- the Linux bundles do not ship `meshroom-pair.sh`, so the Linux gates take the repository's (0.3.8's, which
  records each wrapper's backend) from beside the harness;
- every package copied to a test box matched its SHA-256.

The gate is 0.3.7's with the 0.3.8 steps in it (660f19f):
- FeatureMatching must announce AC-RANSAC's bound (8d) in every configuration;
- `verify` also checks the colours' read (8b), the bound and the epipolar loop (8d, 8e), and PrepareDenseScene's
  libdeflate writer and device undistortion (8f, 8g), both switched on whatever the platform's default;
- a failing count in any chunk's log fails the run, as does any of the steps' fallback warnings;
- `cpufallback` also turns the device undistortion off.

| package | hardware | 2025.1: mini6 | 2025.1: 41 views | 2023.3: mini6 |
|---|---|---|---|---|
| Windows AMD | RX 9070 (rocm7.2 gfx12-generic payload) | 15/15 at 8/8 | base 383 s, verify 806 s | 4/4 at 8/8 |
| Windows AMD | RX 5500 XT (hip6.2 gfx1012 payload, bench-pc) | 15/15 at 8/8 | base 1056 s, verify 2316 s | 4/4 at 8/8 |
| Linux AMD | RX 6750 XT (house-pc) | 15/15 at 8/8 | base 616 s, verify 1707 s | 4/4 at 8/8 |
| Windows CUDA | GTX 1080 Ti (bench-pc) | 15/15 at 8/8 | base 909 s, verify 2211 s | 4/4 at 8/8 |
| Linux CUDA | GTX 1080 Ti (house-pc) | 15/15 at 8/8 | base 671 s, verify 1727 s | 4/4 at 8/8 |

105 runs, none failed. `base` is faster than 0.3.7's on every card (433, 1398, 881, 1277 and 931 s at
0.3.7), most on house-pc's four-thread i3. `verify` takes about as long as before: it now also sorts and
scans every model the bound skipped and computes every residual both ways.

The 41-view `verify` runs' new self-checks:

| card | 8d: models skipped (none below `minNFA`) | 8e: residuals identical | 8g: images identical | 8f: files | 8b: SfM, PrepareDenseScene |
|---|---|---|---|---|---|
| RX 9070 | 31,876,553 of 31,905,517 | 9,709,341,894 of 9,709,341,894 | 33 of 33 (8 took the CPU loop while every remapper was busy) | 41 of 41 | 39 of 39, 41 of 41 |
| RX 5500 XT | 31,809,482 of 31,837,595 | 9,697,870,823 of 9,697,870,823 | 37 of 37 (4) | 41 of 41 | 39 of 39, 41 of 41 |
| RX 6750 XT | 31,996,079 of 32,029,909 | 9,710,963,964 of 9,710,963,964 | 41 of 41 | 41 of 41 | 38 of 38, 41 of 41 |
| GTX 1080 Ti, Windows | 31,712,219 of 31,742,467 | 9,687,690,236 of 9,687,690,236 | 40 of 40 (1) | 41 of 41 | 39 of 39, 41 of 41 |
| GTX 1080 Ti, Linux | 31,996,349 of 32,030,204 | 9,709,966,459 of 9,709,966,459 | 41 of 41 | 41 of 41 | 38 of 38, 41 of 41 |

The epipolar loop held on every arithmetic path the packages have: AVX2 with fused multiply-adds (rocm7.2),
AVX without them (hip6.2), GCC for core2 (both Linux packages) and MSVC (Windows CUDA). The device undistortion
ran on RDNA1, RDNA2, RDNA4 and Pascal, its first runs on all but the RX 9070 and the Linux GTX.

**PopSIFT's race, on hardware.** On every card the 41-view `base` and `verify` runs gave the same depth maps.
On the GTX 1080 Ti they also gave the same features (all 82 `.desc` and `.feat` files) and the same
`sfm.abc` on both systems, where 0.3.7's pairs differed in four descriptors (Linux) and one (Windows). This
gate was the Windows CUDA library's first run on hardware.

## 0.3.9: the Fast Ransac pipeline templates (2026-10-01)

Meshroom 2025.1 raised two RANSAC limits to 50000 (0.3.6, above): FeatureMatching's `maxIteration` (2048
in 2023.3) and StructureFromMotion's `localizerEstimatorMaxIterations` (4096). The paired nodes keep
Meshroom's defaults, so a paired run of Meshroom's own templates gives what a stock run gives (decided
2026-09-30). The faster counts are an opt-in instead. On Meshroom 2025.1 the pairing writes two templates
beside Meshroom's own: **Photogrammetry Fast Ransac** and **Photogrammetry Draft Fast Ransac**
(`photogrammetryFastRansac.mg`, `photogrammetryDraftFastRansac.mg`; Meshroom labels a template by
splitting its file name at capitals, so a name with parentheses or "RANSAC" in capitals reads badly).

How they are made:
- Each is the installed stock template plus two lines: the 2023.3 value as the first input of the
  FeatureMatching node and of the StructureFromMotion node, indented like their neighbours, with the
  stock file's line endings. It is checked to parse as the stock graph plus those two values before it
  is written.
- It is built from the Meshroom being paired, so its node versions are that Meshroom's.
- `--unpair` removes both. A file of the same name that does not carry both values is the user's and is
  left alone. Meshroom 2023.3 keeps its templates in `lib/meshroom/pipelines` and already runs these
  counts, so it gets nothing.
- Windows writes them with `meshroom-templates.ps1`, shipped beside `meshroom-pair.cmd`; Linux with
  python3 inside `meshroom-pair.sh`. On the stock files the two give the same bytes apart from line
  endings.

The gate's `fastransac` configuration runs the template by name, so Meshroom must list it, and requires
`* maxIteration = 2048` and `* localizerEstimatorMaxIterations = 4096` among the two nodes' logged
options. On 2025.1, `base` now requires 50000 in both, so the pairing provably leaves Meshroom's own
template alone.

| check, mini6, Meshroom 2025.1 | RX 9070, Windows (0.3.8 bundle) | RX 6750 XT, Linux (house-pc, 0.3.8 bundle) |
|---|---|---|
| `base` and `fastransac` | both ok at 8/8 | both ok at 8/8 |
| `fastransac` depth maps | `352f0abe56eede6d` | `c4e51fa59e526c8c` |
| the same card's 0.3.8 gate run on Meshroom 2023.3 (`base`) | `352f0abe56eede6d` | `c4e51fa59e526c8c` |
| `base` depth maps, and the 0.3.8 gate's 2025.1 `base` | `7c5369fc24ee7540`, the same | `26fc558e3d75361f`, the same |
| `--unpair` | both templates removed | both templates removed |

On both systems the template reproduces Meshroom 2023.3's run on the same card, byte for byte as far as
the depth maps show.

**What the counts cost with 0.3.8's AC-RANSAC.** RX 9070 box, 0.3.8 binaries; the 0.3.6 numbers above
predate steps 8d and 8e.

| | Meshroom 2025.1's defaults (50000 / 50000) | Fast Ransac (2048 / 4096) |
|---|---|---|
| 41 views, whole graph | 378 s | 307 s |
| 41 views, FeatureMatching / StructureFromMotion | 38.0 s / 65.7 s | 15.2 s / 18.5 s |
| 41 views, views placed | 41 of 41 | 41 of 41 |
| engine bay, FeatureMatching (geometric filtering) | 373.8 s (333.5 s) | 116.8 s (18.6 s) |
| engine bay, SfM on the same matches (localizer only) | 35.3 s: 107 poses, 140,284 landmarks | 15.1 s: 107 poses, 140,394 landmarks |
| False Door, FeatureMatching (geometric filtering) | 1778.9 s (1410.8 s) | 443.0 s (66.8 s) |

Notes on the runs:
- The 41-view numbers are a second pair, run on an idle box with the template first. In the first pair
  the Fast Ransac run lost 53 s in FeatureMatching's GPU matching phase (63 s against 10 s, the same
  work in both runs) to other load on the box. Both pairs gave the same outputs, byte for byte.
- The engine bay and the False Door are `fmbench.py` and `sfmbench.py` runs (Meshroom 2023.3's other
  options). The matches at 2048 are `f063ea6d15c0aa2e` on the engine bay, as recorded for 8d, and
  `ebffcb37a3bd2b0b` on the False Door; at 50000 `5d5f6af548e43556` and `66a6c11497ce0c5d`. The SfM at
  4096 gives the engine bay's reference digest, `814ae0d5`.
- 0.3.8 took the False Door's geometric filtering at 50000 from 2234 s to 1411 s, but at 884 views the
  larger budget still costs 22 minutes of FeatureMatching. 0.3.6 found that its extra pairs placed no
  more views.

`sfmbench.py` now replaces an option it already passes when the same option is given after `--` (the SfM
binary refuses an option given twice), and `fmbench.py` takes sfmbench's sets, the kept False Door cache
(`fd`) included.

## The 0.3.9 release gate (2026-10-01)

0.3.9 is 0.3.8's binaries with the Fast Ransac pairing. The two Windows zips were repacked from 0.3.8's
(`scratchpad repack039.py`): `meshroom-pair.cmd` replaced, `meshroom-templates.ps1` added beside each copy,
and every other entry checked to carry 0.3.8's CRC. The Linux tarballs are the 0.3.8 files, since Linux
pairing comes from the repository's script. The binaries passed 0.3.8's full gate; the changed packages
ran as built, mini6:

| package | hardware | 2025.1: `base`, `fastransac` | 2023.3: `base` |
|---|---|---|---|
| Windows AMD | RX 9070 | both ok at 8/8 | ok at 8/8 |
| Windows CUDA | GTX 1080 Ti (bench-pc) | both ok at 8/8 | ok at 8/8 |
| Linux, the repository's `meshroom-pair.sh` with the HIP tarball | RX 6750 XT (house-pc) | both ok at 8/8 | |

On 2023.3 the pairing printed that there is nothing to write and the harness skipped `fastransac`; every
`--unpair` removed both templates. On each card the Fast Ransac run's depth maps were byte for byte the
same card's Meshroom 2023.3 run: `352f0abe56eede6d` on the RX 9070, `e9de6475cc3c0662` on the GTX 1080 Ti
(in this gate and in 0.3.8's), `c4e51fa59e526c8c` on the RX 6750 XT (against 0.3.8's gate).

## 0.3.9 on RDNA2 under Windows: the RX 6750 XT back in bench-pc (2026-10-01)

gfx1031 had not run on Windows since v0.2.17 (roadmap, "Hardware coverage"). With the RX 6750 XT back in
bench-pc (FX-8120, Windows, the `hip6.2/gfx1031` payload), the 0.3.9 Windows AMD zip ran:

| Meshroom | set | configs | result |
|---|---|---|---|
| 2025.1 | mini6 | all 16, `fastransac` included | 16 of 16 ok at 8/8 ports |
| 2025.1 | 41 views | `base`, `verify` | both ok at 8/8 (779 s, 2080 s) |
| 2023.3 | mini6 | `base`, `defaults`, `cpufallback`, `verify` | 4 of 4 ok at 8/8 |

The RX 5500 XT (RDNA1, gfx1012) ran the same binaries (0.3.9 repacks 0.3.8's) in 0.3.8's gate on the same
host. Every config both ran gives byte-identical depth maps on the two cards except `cpufallback`:
- 2025.1 mini6: `base` and the eight configs that share its digest give `a4bcbe2f74c26e27`, and `defaults`,
  `tiles`, `coarse`, `ds1` and `blast` match the RX 5500 XT's too.
- 41 views: `824f24ba9b01de86`.
- 2023.3 `base` and `verify`: `bfd6150b4ced08b5`.

`cpufallback` runs CPU SIFT and every port's CPU fallback. The gate checks it for the ports' DISABLED lines
and a finished pipeline, not for identity, and its digest differs between the two cards. `fastransac` on
2025.1 gave `bfd6150b4ced08b5`, the same as this card's Meshroom 2023.3 `base`, so the Fast Ransac
templates reproduce 2023.3 on this card too.

## After 0.3.9: NACRANSAC takes 8d's bound and the key sort (step 9a, 2026-10-01)

The new SfM nodes' AC-RANSAC, `NACRANSAC`, gets 8d's bound and the key sort
([docs/15](15-acransac-cpu.md), "The new SfM nodes' AC-RANSAC"). `build/nightly-cmp/replay.py`
replays the nightly comparison's command lines (roadmap, "Upstream's next pipeline", step 2) with
the development install. RelativePoseEstimating and SfMExpanding read the 0.3.8 build's
TracksBuilding and SfMBootStrapping outputs, so every run has the same inputs. One binary,
alternating runs, RX 9070 box, idle; `CHESHIRE_ACR_PAIRSORT=1` is upstream's path:

| run | RelativePoseEstimating, 41 views | SfMExpanding, 41 views | RelativePoseEstimating, engine bay | SfMExpanding, engine bay |
|---|---|---|---|---|
| 9a | 10.4 s | 60.0 s | 31.2 s | 43.3 s |
| upstream's path | 17.8 s | 236.5 s | 37.5 s | 108.0 s |
| 9a | 10.1 s | 57.4 s | | |
| upstream's path | 18.2 s | 242.7 s | | |

The 0.3.8 build took 17.2 s, 235.1 s, 34.6 s and 103.9 s. In SfMExpanding the resection goes from
214.9 s to 38.2 s on 41 views and from 80.8 s to 18.9 s on the engine bay. Triangulation (about
8 s), bundle adjustment (about 5.5 s) and the rest stay. RelativePoseEstimating gains less on the
engine bay: each of its five chunks first spends about 3 s parsing the 87 MB tracks file.

The output is the upstream path's:
- **RelativePoseEstimating**: every run gives the 0.3.8 build's pairs, as a set (1,469 on 41 views,
  2,093 on the engine bay). Its files list them in the order the threads finish, since they are
  written from a critical section.
- **SfMExpanding**: two threaded runs cannot be compared, upstream or not.
  `SfmTriangulation::process` triangulates the tracks on every core, and every track's LO-RANSAC
  draws from one shared `std::mt19937`, so the draws follow thread timing. On one thread
  (`--maxCoresAvailable 1`), 9a's `cameras.sfm`, `sfmExpanded.usda` and `sfmExpanded_landmarks.usdc`
  are byte for byte the upstream path's on both sets: 81.5 s against 299.2 s on 41 views, 89.2 s
  against 225.9 s on the engine bay.
- **`CHESHIRE_ACR_BOUND_CHECK=1`**, which also sorts and scans every skipped model:
  - SfMExpanding skips 762,734 of 780,156 models on 41 views and 2,936,259 of 2,977,540 on the
    engine bay.
  - RelativePoseEstimating skips 97.6-98.1 % per chunk.
  - None of the skipped models is below `minNFA`.
- **`CHESHIRE_ACR_NANLOG=1`**: 44 (41 views) and 773 (engine bay) of the resection's residual arrays
  hold a NaN and sort upstream's pairs. RelativePoseEstimating has none.
- **`ACRANSAC`'s callers**: unchanged. Its bucket table gained the switch with the same values.
  FeatureMatching on 41 views at 50,000 iterations gives `base50k`'s `2b4452a79d79b819`, and
  incrementalSfM gives `801d1fb7b32bdd32` / `361ddd5cf663cfe6`, as in 0.3.8.

## After 0.3.9: NACRANSAC in batches on every core (step 9b, 2026-10-01)

SfMExpanding's single-view rounds run `NACRANSAC`'s iterations in speculative batches on every core
([docs/15](15-acransac-cpu.md), "One resection on every core"). Same harness and inputs as 9a, one
binary, alternating runs, `CHESHIRE_ACR_BATCH=0` being 9a:

| run | SfMExpanding, 41 views | engine bay |
|---|---|---|
| 9b | 29.5 s | 31.9 s |
| `CHESHIRE_ACR_BATCH=0` | 55.6 s | 41.1 s |
| 9b | 28.3 s | 33.0 s |
| `CHESHIRE_ACR_BATCH=0` | 55.7 s | 41.3 s |

The resection goes from 35.7 s to 8.2 s on 41 views and from 18.2 s to 8.9-10.0 s on the engine
bay. `CHESHIRE_ACR_BATCH_CHECK=1` runs the loop as well for every batched call. It found 28 of 28
calls on 41 views and 28 of 28 on the engine bay returning the loop's model, inliers, error and NFA,
and leaving the generator where the loop leaves it. The other parts of SfMExpanding cannot be
compared between threaded runs until its triangulation draws reproducibly (9a, above).

## After 0.3.9: SfMExpanding reproducible (step 9c, 2026-10-01)

`SfmTriangulation::process` triangulates a round's tracks on every core, and every track's LO-RANSAC
drew from the one generator its caller made. So the draws followed thread timing, and no two runs of
SfMExpanding were the same (9a, above). Step 9c gives each track its own generator, derived the way
5n derives the legacy engine's (0.3.4, above): from the caller's seed, the task kind, the track and
the call. The caller's generator is only copied. `CHESHIRE_SFM_TASK_SEED=0` restores the shared
generator, as for the legacy engine. Same harness and inputs as 9a and 9b:

| run | SfMExpanding, 41 views | engine bay |
|---|---|---|
| 12 threads | 28.6 s | 30.2 s |
| 12 threads again | 28.2 s | 30.1 s |
| 4 threads | 38.7 s | 37.7 s |
| 1 thread | 81.0 s | 85.1 s |
| 12 threads, `CHESHIRE_ACR_BATCH=0` | 57.7 s | 41.4 s |
| 12 threads, `CHESHIRE_SFM_TASK_SEED=0` (upstream's shared generator) | 29.9 s | 35.9 s |

Every run but the last gives the same `cameras.sfm`, `sfmExpanded.usda` and
`sfmExpanded_landmarks.usdc`, byte for byte, on both sets. SfMExpanding no longer depends on the
run, the thread count or 9b's batches; this is the first end-to-end comparison of the batches.

The draws differ from upstream's, so the reconstruction is a different sample of the same one:
- camera centres within 0.0060 % (41 views) and 0.12 % (engine bay) of the scene's radius from the
  stock nightly's, where the shared generator's run lands at 0.0094 % and 0.21 %;
- 80,672 and 134,804 landmarks, against stock's 80,680 and 134,482.
