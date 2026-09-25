# CheshireJPG in the reads: plan (0.3.5)

Status 2026-09-25: steps 1-4 in (generator step 6r, off by default; docs/04 has the first results), 5-6 open. CheshireJPG (docs/19) decodes baseline JPEG on the device,
identical to libjpeg-turbo; no node calls it yet.

## What it can win

After step 6n, PrepareDenseScene's read is about 180 ms per 12 MP photo on the RX 9070 box: the
libjpeg-turbo decode (60 ms), filling colorconvert's lines (17 ms) and applying the OCIO processor
(102 ms). The read is 67.4 of the node's 184.6 thread-seconds on the engine bay (docs/04, 6n), so
the decode is about 12 % of the node there. That is the ceiling on a 12-thread box; on house-pc's
i3 the codec measured 2.7-2.9x libjpeg-turbo on the RX 6750 XT (107 of 107 photos
identical, PR #1), and the node is CPU-bound, so the share it frees is worth more. FeatureExtraction opens
every photo as well and is the second caller.

The OCIO `apply` (102 ms) is now the largest piece of the read; it is a separate item (an exact
device version of the sRGB-to-linear transform), not part of this one.

## The shape of the change

1. **Build.** CheshireJPG is a static library with its own CMakeLists (HIP or CUDA). The generator
   adds it to AliceVision's build next to the image library (`add_subdirectory` of a copy under
   `src/aliceVision/image/cheshirejpg`, with the HIP architectures of the build, the CUDA ones on
   CUDA), and links `aliceVision_image` to it PRIVATE. The API crosses the boundary as
   `std::vector<uint8_t>` only, so the Eigen AVX2/HIP allocation trap (docs/04, CheshireEXR check
   mode) does not apply.
2. **Packaging.** `aliceVision_image` then carries device code objects, so it joins the per-target
   GPU libraries: `GPU_DLLS` in `scripts/build_targets.py` (it would otherwise ship the base
   target's code objects to every card), the payload harvest check, and the Linux bundle's
   code-object census. The CUDA packages build one architecture and need only the link.
3. **The read.** In 6n's direct read (`hip/port/image_read/direct8.txt`), for a JPEG: the file's
   bytes to `Codec::decode`, and on `Ok` a 3-channel image of the same size as the header, its
   pixels in place of `inBuf.read(..., UINT8)`; anything else (not baseline, a decode status other
   than Ok, no device) goes the current way. One `Codec` per thread from a small pool (as
   CheshireEXR's loader), so the OpenMP read threads share the device without a lock around decode.
   Switch: `CHESHIRE_PDS_GPU_JPEG` (off until its gate passes; `::cheshire::env::flag`).
4. **Check.** `CHESHIRE_PDS_GPU_JPEG_CHECK=1`: the libjpeg-turbo decode as well, byte comparison of
   the 8-bit pixels, one summary line like the direct-read check's; added to the e2e `verify`
   config and its verdict list.
5. **Gate and measure.** mini6 12/12 and 41 base/verify on RX 9070, RX 5500 XT (hip6.2), RX 6750 XT
   (Linux), GTX 1080 Ti (CUDA); PrepareDenseScene wall and thread-seconds on the engine bay and the
   False Door on the RX 9070 box and house-pc, switch off against on.
6. **FeatureExtraction** after that, through the same helper, if the numbers in 5 justify it.

## Risks

- A device JPEG decode per read thread: VRAM per `Codec` for a 24 MP photo is on the order of
  100-200 MB; the pool size bounds it.
- Files libjpeg-turbo decodes with warnings (a damaged stream) come back `Corrupt` from the codec and
  must take the old path, which is what `Corrupt` is for (docs/19).
- OpenImageIO's reader and CheshireJPG must use the same decoder settings (islow IDCT, fancy
  upsampling); the check in 4 is what proves it per file.
