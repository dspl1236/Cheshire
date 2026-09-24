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

Of the mutated files, every ZIP and ZIPS one is refused by both CheshireEXR and OpenEXR (adler32
catches them). Of the RLE ones, 157 decode identically in both, 33 are refused by both, and 10 are
decoded by OpenEXR but refused here, where OpenEXR's classic RLE decoder tolerates an output length
this one checks. A refusal only means the caller's OpenEXR decodes the file, so those 10 come out
as OpenEXR's.

One bug was found by this check and fixed: the fixed-Huffman distance code was built with 30
codes instead of zlib's 32, so the completeness check rejected valid streams with fixed blocks.

**Compiled, not run.** `exrGPU.cu` builds with HIP 5.7 for gfx1010, gfx1030 and gfx1100.
`ConvertPixels` uses 13 VGPRs with no scratch and runs at 16 waves per SIMD. `InflateChunks`
uses 128 VGPRs and 5.5 KB of scratch per lane (its Huffman tables) and runs at 8 waves. That is
the expected shape for one thread per chunk, but its speed on a card is unknown.
`cheshireexr_gpu_check` builds and links.

**Not done: running on a GPU.** No timings exist yet.

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
decode, floats in host memory) against OpenEXR on the calling thread (how the read-ahead threads
read) and with its thread pool (a lone read). `--threads` gives images per second with one `Codec`
per thread.

## Next

1. **Measure on house-pc first**: the Texturing and DepthMap load profiles above. They decide
   whether this is the lever at all.
2. **`cheshireexr_gpu_check` on the RX 9070 and house-pc's RX 6750 XT**: identity, then single-file
   latency and threaded throughput against OpenEXR, on real PrepareDenseScene output.
3. **Decode into device memory.** DepthMap uploads every image it reads, so for that node a decode
   that leaves the result on the device skips a 324 MB download and the re-upload. The texturing
   node uploads too.
4. **Then** put it in front of `cheshireReadExr` behind a switch, and hold the result to the
   existing gates: DepthMap's chunks byte-identical, Texturing's output within its run-to-run band.
5. If a single image is too slow with one thread per ZIP chunk, compare ZIPS output from
   PrepareDenseScene, or decode one chunk per workgroup.
