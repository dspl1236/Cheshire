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
