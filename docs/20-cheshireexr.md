# CheshireEXR: decoding the pipeline's EXRs on the GPU

Texturing and DepthMap are 6.5 of the 8.75 hours of the 884-photograph job on house-pc (the s7 run,
docs/04), and both spend much of their time reading the EXRs PrepareDenseScene writes. Each one is
6000x3376 half RGBA, ZIP-compressed, 73 MB. Texturing re-reads every contributing camera on every
pass: 17 passes of about 60 GB. CheshireEXR is a decoder for those files that runs on the GPU and
returns exactly the floats OpenEXR returns.

**Whether it helps depends on what those reads are waiting for, and that has not been measured on
house-pc.** On the RX 9070 box Texturing's reads are disk-bound (60 GB per pass at 283 MB/s, the
SATA SSD's rate, docs/04), and no decoder changes that. On house-pc's four-thread i3 the inflate may
be the cost instead. The first measurement is one Texturing pass and one DepthMap chunk there, with
`CHESHIRE_LOAD_PROFILE=1` and `CHESHIRE_EXR_PROFILE=1`. If read time is close to bytes divided by
the disk's rate, the lever is fewer bytes or fewer passes. If decompression dominates, it is this.

## Why it can be exact

ZIP, ZIPS and RLE are lossless. A decode is either OpenEXR's bytes turned into OpenEXR's floats or
it is wrong; there is no rounding to reproduce, as there was for JPEG. What has to match is behaviour:

- **Inflate** (RFC 1950/1951), including every stream zlib rejects: bad header or preset
  dictionary, block type 3, a stored block whose length check fails, an over-subscribed code, an
  incomplete code other than a single code of length 1 (zlib's `inflate_table()` rule), symbols
  286-287 and distance codes 30-31, a distance before the start of the output, output longer or
  shorter than the chunk, a bad adler32. A stream zlib refuses is refused here, so no file decodes
  to pixels OpenEXR would never return.
- **OpenEXR's post-processing**: the predictor (`t[i] = t[i-1] + t[i] - 128`) and the byte
  interleave (first half of the buffer to even bytes, second half to odd), and its RLE format.
- **Half to float as Imath converts it**, NaN payloads included.

## How it runs

`hip/port/cheshireexr/`, on the same machinery as CheshireJPG: stages are functors run by a backend,
and the backends live in `hip/port/cheshiregpu/`. Each `Codec` has its own stream and a pinned
staging area, and the host check runs the same code.

| step | where | what |
|---|---|---|
| parse | host | header, sorted channel list, offset table, each chunk's leader; the same file restrictions as Cheshire's direct reader (`image/cheshireExr.cpp`, step 5v) |
| upload | host to device | the whole file, in 16 MB pieces through pinned staging |
| `InflateChunks` | device, one thread per chunk | inflate (ZIP, ZIPS) or RLE-decode into the chunk's scratch slice, check it as zlib and OpenEXR do, undo the predictor; chunks stored raw need nothing |
| error flag | device to host | one word; any failed chunk makes the file `Corrupt` |
| `ConvertPixels` | device, one thread per pixel | gather each requested channel's bytes through the interleave mapping (never materialised), convert half to float |
| download | device to host | the floats, in 16 MB pieces, into the caller's `allocate(width, height)` buffer |

The API has the direct reader's contract, so CheshireEXR can sit in front of it:
`decode(file, size, nchannels, allocate)` with `nchannels` 1 (the file's only channel), 3 (R, G, B)
or 4 (R, G, B, A). `allocate` is only called once the file has decoded. Anything outside scope
returns `Unsupported`, and a damaged chunk returns `Corrupt`; in both cases the caller keeps
OpenEXR. That covers tiled, multi-part, deep, PIZ, PXR24, B44, DWA, UINT, subsampled channels, and a
data window not at the origin. `readHeader()` parses the header alone, so a caller can check
`AliceVision:ColorSpace` first.

`decodeToDevice(file, size, nchannels, out)` is the same decode with the last step left out: the
floats stay in the `Codec`'s device buffer, and `out` gives their pointer, size and row pitch. They are
complete when the call returns (it waits for the stream) and stay valid until the `Codec`'s next
decode, `trim()` or destruction. `stream()` is the `Codec`'s stream, for work that reads them in order
with its own; `download()` copies them to host memory; `trim()` frees the device buffers, which are
otherwise kept for the next decode (about 0.6 GB for a 6000x3376 RGBA file).

## DepthMap: decoding into device memory (step 6m)

The depth-map node uploads every image it reads. Since step 6d a load is: OpenEXR inflates the file
on the CPU, the image cache keeps 324 MB of full-resolution floats, `DeviceCache::addMipmapImage`
uploads them and 6d's kernel downscales them on the device. With `CHESHIRE_DEPTHMAP_GPU_EXR=1` the
batch loader (`DepthMapEstimator.cpp`, the prefetch of step 5t) hands each new camera to
`imageProcessing/cheshireDeviceExr.cu` instead:

1. parse the header from the file's first 64 KB;
2. take the file only if it has enough chunks to decode faster on the device than on the host
   (`CHESHIRE_DEPTHMAP_GPU_EXR_MIN_CHUNKS`, 1024 by default: ZIPS, RLE and NONE have one chunk per
   scanline, ZIP one per 16, so ZIP stays on the host), and only if the host path would have read it
   through the direct reader: a `.exr`, `AliceVision:ColorSpace` absent or `linear`, the size the SfM
   data expects, `CHESHIRE_EXR_DIRECT` not 0, and within CheshireEXR's scope (RGBA requested, so the
   file must have alpha);
3. read the whole file (73 MB for a 6000x3376 ZIP file);
4. `decodeToDevice` on one of the loader's `Codec`s;
5. 6d's downscale kernel, unchanged, reading that buffer, queued on the `Codec`'s stream, into the
   `CudaRGBA` image `addMipmapImage` would have built;
6. `DeviceCache::addMipmapImageFromDevice` fills the mipmap from it (6d's `fillFromDevice`).

The texels are the host path's by construction: the same floats (CheshireEXR is bit-identical to
OpenEXR), the same kernel on them. Anything the loader does not take, or a decode that fails, goes
through the image cache as before, and the host path then reports whatever is wrong with the file.

The loader is made per batch on the thread that owns the device and destroyed when the batch's loads
are done, so none of its memory is held while the tiles run. It sizes itself from the memory
bridge's budget: VRAM minus what is in use, minus the image reserve the planner holds for camera
mipmaps (`Budget::imageReserveLeft`, new), minus 256 MB. Each decoder needs about 32 bytes per pixel
and each image waiting for the device cache its downscaled texels; if not even one decoder and one
image fit, the batch uses the host path, so a decoder buffer never spills to host memory.

| switch | default | |
|---|---|---|
| `CHESHIRE_DEPTHMAP_GPU_EXR` | 0 | 1 decodes on the device (needs 6d's device downscale, i.e. a process downscale above 1) |
| `CHESHIRE_DEPTHMAP_GPU_EXR_MIN_CHUNKS` | 1024 | files with fewer chunks go to the host path, where they are faster; 0 takes every file |
| `CHESHIRE_DEPTHMAP_GPU_EXR_CODECS` | 2 | decoders at most, each with its own stream |
| `CHESHIRE_DEPTHMAP_GPU_EXR_GROUP` | 8 | images decoded before they are handed to the device cache (at most the image cache's slots) |
| `CHESHIRE_DEPTHMAP_GPU_EXR_CHECK` | 0 | 1 also runs the host path for every image and compares the texels, logging any difference |

Each batch logs how many images were decoded on the device and how many went through the host path,
and how many of those had too few chunks.

**One design choice to measure.** An EXR ZIP chunk is 16 scanlines, so a 3376-line image is 211
independent jobs; a ZIPS file (one scanline per chunk) is 3376. PrepareDenseScene switched from ZIPS
to ZIP in step 5y because fewer zlib streams made the CPU write cheaper. For a GPU decoder with one
thread per chunk, ZIPS gives 16 times the parallelism per image. With a dozen images in flight (the
depth-map prefetch, the texturing read-ahead) ZIP may be enough; with one image at a time it will
not be.

## Validation

**Done here, without a GPU.** `cheshireexr_cpu_check` writes its test files with OpenEXR 3.1.5 and
decodes them through the host backend and through the device backend's own code on the deferred
runtime (every copy and kernel held back until the stream is waited on, allocations filled with
garbage). The floats are compared bitwise with OpenEXR's `Imf::InputFile` reading into FLOAT
slices. Both backends:

| check | cases | result |
|---|---|---|
| NONE, RLE, ZIPS, ZIP x 9 sizes (1x1 to 640x97) x half RGBA, half RGB, float RGBA, mixed half/float RGB, float Y, half Y; noise (NaNs, infinities and denormals included), gradients, constant runs; increasing and decreasing line order | 256 decodes | 256 identical |
| every half bit pattern, in all four channels, under each compression | 4 | identical |
| 6000x3376 half RGBA ZIP, PrepareDenseScene's format (`--large`) | 1 | identical |
| PIZ, alpha requested from an RGB file, one channel requested from RGB, not an EXR | 4 | `Unsupported` x 3, `NotExr` |
| 1-3 bit flips inside the chunks of ZIP, ZIPS and RLE files | 600 | 157 identical to OpenEXR, 443 refused, **0 decoded differently** |
| `decodeToDevice` on every file and mutation above, read in place and through `download()` | 861 | same status and floats as `decode` |

Of the mutated files, every ZIP and ZIPS one is refused by both CheshireEXR and OpenEXR (adler32
catches them). Of the RLE ones, 157 decode identically in both, 33 are refused by both, and 10 are
decoded by OpenEXR but refused here, where OpenEXR's classic RLE decoder tolerates an output length
this one checks. A refusal only means the caller's OpenEXR decodes the file, so those 10 come out
as OpenEXR's.

The `decodeToDevice` comparison starts each file from freed buffers, so on the deferred runtime its
output is garbage until the stream runs. With the stream wait at the end of `decodeToDevice` removed
it fails 256 of 260 files; a first version of the check reused the buffer `decode` had just filled
and passed without the wait, and was changed for that reason.

One bug was found by this check and fixed: the fixed-Huffman distance code was built with 30
codes instead of zlib's 32, so the completeness check rejected valid streams with fixed blocks.

**Compiled, not run.** `exrGPU.cu` builds with HIP 5.7 for gfx1010, gfx1030 and gfx1100.
`ConvertPixels` uses 13 VGPRs with no scratch and runs at 16 waves per SIMD. `InflateChunks`
uses 128 VGPRs and 5.5 KB of scratch per lane (its Huffman tables) and runs at 8 waves. That is
the expected shape for one thread per chunk, but its speed on a card is unknown.
`cheshireexr_gpu_check` builds and links. For step 6m, the patched DepthMap unity translation unit
(CheshireEXR, the loader and 6d's kernels) and `DepthMapEstimator.cpp` compile with hipcc 5.7 for
gfx1030 against the pinned AliceVision, and `apply_hip_patch.py` applies and re-applies cleanly.

**RX 9070 (gfx1201), Windows 11, ROCm 7.2, 2026-09-24** (dspl1236/Cheshire#2):

- **Identity:** `cheshireexr_gpu_check` 85 of 85 synthetic cases identical to OpenEXR, and 16 of 16
  PrepareDenseScene files: 8 from the engine bay set (4032x2268, ZIP level 1) and 8 from False Door
  (6000x3376, ZIPS).
- **Depth maps:** with `CHESHIRE_DEPTHMAP_GPU_EXR=1`, byte-identical to the host path: 12 of 12 on
  mini6 (ZIP level 1) and 96 of 96 on a 48-view False Door chunk (ZIPS).
- **Speed follows the chunk count.** One inflating thread per chunk is fast only when there are
  thousands of chunks:

| file | chunks | CheshireEXR to device | OpenEXR, 1 thread | OpenEXR, pool | images/s at 12 threads, CheshireEXR / OpenEXR |
|---|---|---|---|---|---|
| generated 6000x3376 ZIP | 211 | 2214 ms | 309 ms | 100 ms | |
| engine bay, ZIP level 1 | 142 | 850-1010 ms | 153-160 ms | 46-50 ms | 5.2 / 27.4 |
| False Door, ZIPS | 3376 | 163-243 ms | 337-359 ms | 113-123 ms | 14.5 / 12.8 |

  On ZIPS the 48-view chunk's image loads dropped from 30.5 to 9.4 s and the node from 326.9 to
  305.6 s. On ZIP, which PrepareDenseScene writes since steps 5y and 6f, it is slower than the host:
  the mini6 decode went from 0.3 s to 4.4 s and the node from 13.6 to 17.4 s. Each thread
  inflates about 0.5 MB serially. The loader now leaves such files to the host
  (`CHESHIRE_DEPTHMAP_GPU_EXR_MIN_CHUNKS`).
- **Compare per megapixel, not per image.** The engine bay photographs are 9.1 megapixels and False
  Door's 20.3, so images per second across the two sets say nothing about ZIP against ZIPS. In
  megapixels per second, with the RX 5500 XT on bench-pc (FX-8120, 8 threads) alongside:

| MP/s | RX 9070 box, 1 thread | RX 9070 box, 12 threads | bench-pc, 1 thread | bench-pc, 8 threads |
|---|---|---|---|---|
| CPU (OpenEXR), ZIP | 54 | 250 | 26 | 81 |
| CPU (OpenEXR), ZIPS | 55 | 259 | 28 | 73 |
| GPU (CheshireEXR), ZIPS | 118 | 294 | 36 | 79 |
| GPU (CheshireEXR), ZIP | 9 | 48 | 5 | failed (below) |

  On ZIPS the GPU is twice a CPU thread on the RX 9070 box and a little ahead of all twelve.
  Whether PrepareDenseScene should write ZIPS for it is open: that takes the same images through
  both routes end to end (Next, below).
- **A second bug, in the check itself:** on bench-pc at 8 threads every GPU decode failed ("stream
  create failed: out of memory", eight decoders on the RX 5500 XT), and the throughput mode, which
  ignored decode results, printed 25.5 images/s and PASS. It now counts only successful decodes,
  reports megapixels per second as well, and fails the run on any failed decode, in the timing
  loops too; it also no longer re-reads every file inside the timed loop to find its channels.
- **A bug the run found:** `CHESHIRE_DEPTHMAP_GPU_EXR_CHECK=1` corrupted the heap
  (`0xC0000374`). The check read the host image with `image::readImage` (image library, built with
  `/arch:AVX2`, so Eigen uses its own aligned allocator) and freed it in the HIP unity TU (no
  `/arch:AVX2`, so Eigen calls plain `free`). The check now reads through `cheshireReadExr` into a
  `std::vector<float>`. Step 6d's `CHESHIRE_DEPTHMAP_DEVICE_DOWNSCALE_CHECK` had the same crossing
  (the library allocated the resized image, the unity TU freed it). Its images are now allocated in
  the TU at their final sizes, with a guard in case the library ever reallocates. With the image
  leaked instead of freed, the RX 9070 run gave 6 of 6 images identical and 12 of 12 maps
  byte-identical, so the check itself was right.

**The deciding test on the RX 9070 box, 2026-09-25** (`scripts/exr_layout_test.py`, b0a53ff; one
48-view False Door chunk, 3 runs each, dspl1236/Cheshire#2). Texturing was left out: one pass takes
about 97 minutes there.

- **Exact everywhere.** The 833 undistorted images had identical pixels in both layouts. The 96
  maps were byte-identical in all four runs. The check found 104 of 104 images identical to the host
  path, and the chunk threshold kept every ZIP file off the device.
- **Writing ZIPS costs nothing but disk.** PrepareDenseScene took 290 s against 287 s (3025 against
  3067 s of CPU), for 61.3 GB against 57.7 GB (+6.3 %).
- **DepthMap gains little:**

| run | wall | CPU | image loads (sum of batches) | decode |
|---|---|---|---|---|
| ZIP, host path (today) | 349.3 s | 439.0 s | 12.1 s | 7.2 s |
| ZIPS, host path | 333.6 s | 424.6 s | 11.3 s | 7.1 s |
| ZIPS, device path | 328.8 s | 401.0 s | 7.2 s | 6.4 s |
| ZIP, device path switched on (all handed to the host) | 338.5 s | 434.6 s | 13.0 s | 8.1 s |

  Against the host path on the same ZIPS files, the device path saves 4 s of loads and 24 s of CPU,
  about 1.5 % of the node. The ZIP-host figure is inflated by a cold first run (its runs were 379,
  348 and 329 s), so the host paths are level. After steps 5t-5z, loads are about 3 % of this node
  on a 12-thread box, which leaves a decoder little to win.

On this box it is a wash: exact, harmless while off, a small gain for 6.3 % more disk. It stays
unmerged until the same test runs on house-pc's 4-thread i3, where the host decode is a larger share
of the node and the CPU it frees is worth more.

## Build and run

```
# the host check (any machine with OpenEXR)
cmake -S hip/tests/cheshireexr -B build/exr -DCHESHIREEXR_GPU=OFF
cmake --build build/exr && build/exr/cheshireexr_cpu_check --large

# the device check (HIP SDK / ROCm; CHESHIREEXR_GPU=CUDA for NVIDIA)
cmake -S hip/tests/cheshireexr -B build/exr-gpu -DCMAKE_HIP_ARCHITECTURES="gfx1201;gfx1030;gfx1010"
cmake --build build/exr-gpu
build/exr-gpu/cheshireexr_gpu_check --reps 5                          # generated set, incl. 6000x3376
build/exr-gpu/cheshireexr_gpu_check --reps 3 --threads 1,2,4,8,12 <PrepareDenseScene output>/*.exr
```

Per file it prints the chunk count and the median end-to-end time for CheshireEXR (read the file,
decode, floats in host memory), for CheshireEXR to device (the same, floats left on the device, as
DepthMap uses it; checked against the host result first), against OpenEXR on the calling thread (how the read-ahead threads
read) and with its thread pool (a lone read). `--threads` gives images and megapixels per second with one `Codec`
per thread; any failed decode fails the run.

## Next

1. **The deciding test on house-pc** (done on the RX 9070 box, above: a wash). One False Door chunk
   with identical pixels, once as ZIP level 1 through the host path and once as ZIPS through the
   device path, including what ZIPS costs PrepareDenseScene to write and, if the DepthMap numbers
   justify its 97 minutes, one Texturing pass per layout. If ZIPS with the GPU wins overall there,
   PrepareDenseScene gets a ZIPS output option and this merges; if not, it is parked, since with the
   chunk threshold it never runs on ZIP output. `scripts/exr_layout_test.py <config.json> --reps 3 --check` runs it:
   PrepareDenseScene with `CHESHIRE_PDS_EXR_COMPRESSION=zip:1` and `zips:1` (the write cost, and
   the two outputs checked pixel-identical), DepthMap on one chunk as zip-host, zips-host, zips-gpu
   and zip-gpu (maps compared, and zip-gpu must decode nothing on the device), then Texturing on each
   layout. It writes `exr_layout_test.md` with wall and process CPU times and the load, decode and
   EXR-read totals from the logs. The config holds the three command lines from the Meshroom cache
   (see the script's docstring).
2. **`cheshireexr_gpu_check` on house-pc's RX 6750 XT**, and the house-pc load profiles
   (`CHESHIRE_LOAD_PROFILE=1`, `CHESHIRE_EXR_PROFILE=1`).
3. **Then** put it in front of `cheshireReadExr` behind a switch, and hold the result to the
   existing gates: DepthMap's chunks byte-identical, Texturing's output within its run-to-run band.
   The texturing node uploads its images too and could take the step 6m route.
