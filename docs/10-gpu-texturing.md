# GPU texturing

Texturing unwraps the filtered mesh into UV atlases, picks the best cameras for every triangle,
then, camera by camera, loads the image, builds a three-band Laplacian pyramid and rasterises
every triangle that camera contributes to into per-atlas, per-band accumulators (weighted colour
sums and weights). At the end each atlas is normalised, its bands fused, padded, downscaled and
written. On the 107-photo engine bay job the node took 220 s inside Meshroom and 165.8 s run
standalone; profiled, that was 21.7 s of image reads, 19.2 s of pyramids, 37.1 s of rasterisation,
about 20 s of single-threaded camera selection, 17 s of padding and downscale, and the rest UV
unwrap, mesh load and save and EXR writes.

## What it is

`hip/port/gpu_texturing/`: the per-camera pyramid and rasterisation and the final
normalise-and-fuse on the GPU, wired into `Texturing::generateTexturesSubSet` by
`scripts/apply_hip_patch.py` (step 4f). The host still reads each image (EXR, exposure
compensation and the OCIO colour conversion are OpenImageIO work) and uploads it; the GPU builds
the Laplacian pyramid with OpenImageIO 3.0's separable resize transcribed (its Gaussian filter,
its `fast_exp` polynomial, its weight normalisation and accumulation order) and the bilinear
difference upstream uses, then runs one thread block per triangle over the triangle's bounding
box: geogram's `point_triangle_squared_distance` for the inside test and barycentrics, the
projection, the in-image margin, the "pure zero" test on the source image, and float atomics into
the accumulators for the triangle's band and every lower frequency, exactly the CPU's expression
order with FMA contraction off. The accumulators live in VRAM: 3.2 GB per 8192² atlas at three
bands, so atlases are processed in chunks of what the card holds (four on a 16 GB card, two on
8 GB); with enough RAM every image is kept for the whole job so a second chunk re-reads nothing,
and the next four cameras are read ahead on their own threads while the current one is on the GPU.
The single-threaded camera selection is scored in contiguous triangle ranges in parallel and
merged in order, so its lists are exactly the sequential ones (`CHESHIRE_TEX_PARTS=1` runs the
sequential loop, and `CHESHIRE_GPU_TEX_LOG=1` prints an order-sensitive checksum of the lists to
compare, plus the time split). Padding, the Lanczos downscale (OpenImageIO's filter is built on
the C runtime's `sinf`, which a GPU cannot reproduce bit for bit) and the EXR writes stay on the
CPU. `CHESHIRE_GPU_TEX=0` restores the CPU loop.

## Measured (RX 9070, Windows, standalone)

| | CPU | GPU |
|---|---|---|
| engine bay, 107 photos, 6 atlases: image reads | 21.7 s | 5.3 s waiting (read ahead) |
| pyramids | 19.2 s | 0.17 s |
| rasterisation | 37.1 s | 3.4 s |
| camera selection | ~20 s | 3 s |
| Texturing node, end to end | 165.8 s (220 s inside Meshroom) | 79.3 s |
| 6 views, 1 atlas, end to end | 16.6 s | 10.3 s |

RX 5500 XT (8 GB) on Linux (house-pc, i3-4330, 14 GB RAM) with the v0.2.8 bundle, 41 views, 3
atlases in chunks of 2 with all 41 images kept in RAM: Texturing 271.5 s on the CPU against 102.5 s
with the GPU passes (uploads 3.0 s, pyramids 0.7 s, rasterisation 4.1 s, image reads 10.1 s
waiting). The textures have the identical set of coloured pixels and differ on 0.008 % of the
pixels, at most 0.05, 19 to 30 pixels beyond 1e-3 per texture.

What is left on the engine bay: UV unwrap 14 s, mesh load 9 s and save 14 s, per-atlas padding,
downscale and write about 3.5 s each. Those are the node now.

## Validation

The rasteriser is a float accumulation in whatever order the contributions arrive, on the CPU as
well: upstream's per-camera triangle loop is an OpenMP `+=` with no atomics, so texels at chart
edges, where two triangles of one camera share a texel, are racy there. Two CPU runs of the same
binary on the engine bay differ on 0.009 to 0.013 % of the pixels per texture, by up to 0.35 in
one texel and beyond 1e-3 on 100 to 320 pixels per texture. GPU against CPU is inside that band:
0.010 to 0.013 % of the pixels, at most 0.045, beyond 1e-3 on 36 to 170 pixels per texture, and the
set of coloured pixels is identical texture for texture. On the 6-view set the two differ on 878
pixels of 16.8 M by one half-float ulp at most, none beyond 1e-3 (two CPU runs: 56 pixels, same
ulp). Texels touched by one triangle per camera, the overwhelming majority, follow the same float
operations in the same camera order and come out equal; the textures are written as half floats,
which rounds most of the remaining accumulation-order noise away.

A note on comparing against Meshroom's own output: its stock `aliceVision_texturing` was built
with another compiler and OpenImageIO version, and 25 to 40 % of the pixels differ from either of
our CPU runs at the ulp level. Run-to-run of one binary is the right baseline.

## Pairing

`meshroom-pair.cmd` / `meshroom-pair.sh` pair `aliceVision_texturing` next to the other four,
gated on the binary's `--help` naming the GPU pass. Meshroom 2023.3's Texturing node options are
accepted unchanged.
