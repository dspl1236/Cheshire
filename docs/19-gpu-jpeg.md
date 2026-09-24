# A JPEG codec on the GPU, for any Radeon

Cheshire's pipeline starts from JPEGs and keeps decoding them: PrepareDenseScene's read phase is JPEG
decode plus the OCIO conversion, 47 % of that node's thread time on the RX 9070 box and 50.6
thread-seconds on house-pc ([docs/04](04-validation.md), 0.3.3 item 2); FeatureExtraction opens
every photo; Mill 19 "Rubble" is 9.8 GB of JPEG. This document is the codec that can move that work
to the device: a baseline JPEG **decoder and encoder in ordinary HIP compute kernels**, whose output
is identical to libjpeg-turbo's - the library OpenImageIO decodes and encodes JPEG with.

It is not wired into any Meshroom node yet. That is the next step, and the step after it is measuring
it on the cards; neither has happened, and nothing below claims a speed.

## Why not rocJPEG

AMD has a JPEG library, [rocJPEG](https://github.com/ROCm/rocJPEG). Read its prerequisites and the
design falls out: `libva-amdgpu-dev` and `mesa-amdgpu-va-drivers`. rocJPEG hands the bitstream to the
fixed-function JPEG engine in the VCN video block through VA-API. That has three consequences:

| | rocJPEG | this codec |
|---|---|---|
| how | VCN hardware JPEG engine via VA-API (Mesa) | HIP compute kernels |
| OS | Linux only (Ubuntu, RHEL, SLES) - there is no VA-API on Windows | Windows and Linux, wherever Cheshire builds |
| GPUs | "gfx908 or higher", the ROCm support list | anything HIP compiles for, RDNA1 (RX 5500 XT) included; NVIDIA through the CUDA build |
| direction | decode only | decode and encode |
| output | the engine's | libjpeg-turbo's, byte for byte |
| status | deprecated, moved into ROCm/rocm-systems | |

The engine cannot be "ported"; it is silicon. Wider use therefore means a compute codec, and the
hard part of that is the one thing the engine does in hardware: Huffman decoding a stream that has
no restart markers, which is most camera JPEGs (80 of the 88 camera files below have none).

## Layout

`hip/port/gpu_jpeg/`, CUDA dialect like every other port; the HIP build force-includes
`cheshire/cuda_to_hip.h`, so device buffers go through the memory bridge.

| file | what |
|---|---|
| `jpegCodec.hpp` | the public API: `Codec::decode`, `Codec::encode`, `Status` |
| `jpegTypes.hpp` | per-element arithmetic, host + device: IDCT, FDCT, quantiser, colour, Huffman bit access |
| `jpegStages.hpp` | every pipeline stage as a functor `f(i)` |
| `jpegPipeline.hpp` | the decode and encode sequences, written once over a backend |
| `jpegHost.cpp` | marker parsing, unstuffing, table building, header writing |
| `jpegGPU.cu` | the device backend (one thread per index) and `Codec` |
| `jpegCpuBackend.hpp` | the host backend (a loop), for verification |
| `jpegTool.cpp` | `cheshire_jpeg decode / encode / bench` |
| `CMakeLists.txt` | `cheshire_gpu_jpeg` static library + the tool; `CHESHIRE_JPEG_GPU=HIP` or `CUDA` |

The backend split is what makes the claims below checkable without a GPU. A stage is
`forEach(n, functor)`: the device backend launches `forEachKernel<F>`, the host backend runs the loop.
No stage uses warp or workgroup primitives - scans are a chunk pass, a serial pass over the chunk
totals and a chunk rewrite - so every line a kernel executes is also executed by
`hip/tests/jpeg/jpeg_cpu_check`, in the same order, against libjpeg-turbo.

## Decoding

Host: parse the markers, strip byte stuffing and RSTn markers from the entropy-coded data into
32-bit words (bit 31 first; restart segments start on byte boundaries), cut each restart segment
into **subsequences of 1024 bits**. Device, in order:

1. **Initial pass.** Every subsequence is decoded from a guessed state (its first bit, first block of
   the MCU, DC coefficient next). The state carried across a boundary is (bit position, block within
   the MCU, coefficient index). Each run records its entry and exit state, how many blocks it started
   and its DC differences.
2. **Synchronisation rounds.** A subsequence whose entry no longer equals its left neighbour's exit
   is decoded again from that exit - and the thread keeps going into the following subsequences
   (which it alone writes, since it stops at the next one flagged) until its state meets an entry
   recorded in the previous round, or 32 subsequences. Repeat until a round changes nothing. The first
   subsequence of each restart segment starts from a known state, and the fixed point of
   "entry(i) = exit(i-1)" is the sequential decode, so the result is exact however many rounds it
   takes; Huffman codes resynchronise within a few symbols, so it takes few. This is Weissenberger &
   Schmidt's method (*Massively Parallel Huffman Decoding on GPUs*, ICPP 2018) with the JPEG state
   added and the walk-ahead.
3. **Scans.** A segmented scan over the runs gives each subsequence its first block index and its DC
   predictors (reset at every restart marker).
4. **Verification.** Every restart segment must have produced exactly its blocks, ended on an MCU
   boundary and met no invalid code. Otherwise `Status::Corrupt`, and the caller's libjpeg decodes it.
5. **Write pass**, **IDCT** (one thread per block), **upsampling + colour** (one thread per pixel).

Two findings shaped step 2, both from the host check's round counts:

- **Phase lock with shared tables.** When every block of the MCU uses the same DC and AC tables -
  RGB-colourspace files, and at least one camera file (`xmp/no_exif.jpg`, 4:2:0 with the chroma on
  the luma tables) - a subsequence decoded from the wrong block index parses the bits identically
  and never notices. The error only travels left to right: 297 rounds for that 322x466 file. But in
  that case the block index plays no part in the parse, so the decoder synchronises on (bit
  position, coefficient index) alone, records DC sums per block slot relative to the entry, and
  resolves the block index afterwards from the exact block counts. 297 rounds became 10.
- **A count-based phase round, tried and removed.** Deriving the block index from block counts in
  one extra round also fixes phase lock, but on the camera files it cost more than it saved: with it,
  3.61 decodes of every subsequence and 6.8 rounds on average; without it, 2.23 and 5.1.

What the final version does on the 88 camera JPEGs of the check (from 49x500 up to 4896x3264):

| sync rounds | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| files | 11 | 18 | 13 | 15 | 4 | 10 | 9 | 3 | 2 | 1 | 2 |

Each round is a flag readback, so the round count is latency, not work: the work is the 2.23
count-mode decodes plus one writing decode of each subsequence. On the synthetic files the worst
case is 66 rounds (a 640x480 4:1:1 file at q92, four luma blocks in a row on the same tables in
every MCU, most likely a milder form of the phase lock above); the walk-ahead is what keeps that
from being one round per subsequence.

### Exactness

| stage | reproduces |
|---|---|
| Huffman, DC prediction, run lengths | `jdhuff.c`, including where a corrupt run length pushes the coefficient index past 63 (coefficient 63 is written, as libjpeg's padded `jpeg_natural_order` does) |
| IDCT | `jidctint.c` `jpeg_idct_islow`, and the post-IDCT range-limit table of `jdmaster.c` |
| upsampling | `jdsample.c`: h2v1 and h2v2 fancy (plain replication below 3 samples wide, as libjpeg), h1v2 fancy, replication for other integer ratios; edge rows as `jdmainct.c`'s context pointers |
| colour | `jdcolor.c` YCbCr->RGB tables; grayscale; RGB passthrough for Adobe transform 0 or `R`,`G`,`B` ids |

The IDCT is 32-bit where libjpeg's C path is 64-bit, and libjpeg-turbo's SIMD paths dequantise in
16 bits and pack the column pass to 16 bits with saturation. All three agree exactly when the
dequantised coefficients stay within +/-8191 and the column-pass outputs within +/-16383: the worst
case over every sign pattern is then 5.0e8 in the column pass and 1.0e9 in the row pass
(`scripts/jpeg_idct_bounds.py`). A real 8-bit encoder stays far inside - a black block's DC is
-1024. Outside the limits the three libjpeg-turbo builds already disagree with each other, so the
IDCT refuses the block and the decode returns `Corrupt`; the host check's mutated streams found this
case (bit flips that pushed a DC coefficient far outside that range).

## Encoding

Colour conversion and downsampling per sample (`jccolor.c`, `jcsample.c` h2v1/h2v2 with their
alternating bias, `int_downsample` otherwise, the edge replication of `expand_right_edge` and
`jcprepct.c`), FDCT + quantisation per block (`jfdctint.c`, `jcdctmgr.c`'s reciprocal quantiser),
the dummy blocks of edge MCUs (`jccoefct.c`: zero AC, DC copied from the block before them in the
MCU buffer), then entropy coding in parallel:

1. every block's bit length (`jchuff.c` `encode_one_block`, standard tables), DC differences from
   the previous block of the component within the restart segment;
2. a scan gives every block its bit offset; segments start on byte boundaries;
3. every block writes its bits with 32-bit atomic ORs; each segment's last byte is padded with ones;
4. a count of 0xFF bytes per 256-byte chunk, a scan, and a scatter that writes the stuffed stream
   with an RSTn marker before every segment after the first.

The host writes SOI, JFIF APP0, DQT, SOF0, DHT, DRI and SOS exactly as `jcmarker.c` does for
`jpeg_set_defaults` + `jpeg_set_quality(q, TRUE)`. Options: quality 1-100, 4:4:4 / 4:2:2 / 4:2:0 /
4:4:0 / grayscale, restart interval. Optimised Huffman tables are not implemented (libjpeg's default
is off; the decoder handles files that use them).

## Validation

**Done here, without a GPU** (container with no device; libjpeg-turbo 2.1.5 as the reference):

`jpeg_cpu_check` - the device pipeline on the host backend - run against libjpeg-turbo's SIMD
paths and again with `JSIMD_FORCENONE=1` against its C paths. Both runs:

| check | cases | result |
|---|---|---|
| encode, byte-identical file | 301: 15 sizes from 1x1 to 1023x67 plus 4032x3024, all five layouts, q1-100, restart 0/1/3/7, noise, gradients, saturated checkerboards, rings | 301 identical |
| decode of libjpeg's files, identical pixels and identical coefficients for every block | 532: the same sizes; optimised and standard Huffman tables; restart intervals; 4:1:1, h3, mixed h/v layouts, chroma-at-full-luma-at-half; RGB colourspace; our own encodes | 532 identical |
| camera JPEGs ([ianare/exif-samples](https://github.com/ianare/exif-samples), 89 files, up to 4896x3264, restart intervals from none to one every 4 MCUs) | 89 | 88 identical in pixels and coefficients; 1 (`corrupted.jpg`, 17,268 libjpeg warnings) handed back as `Corrupt` |
| mutated streams (1-3 random bit flips in the entropy-coded data) | 900 | 460 decoded identical to libjpeg, 440 handed back as `Corrupt`, **0 decoded differently** |
| libjpeg-turbo's test images | 4 | `testorig.jpg`, `testimgint.jpg` identical; arithmetic-coded and 12-bit ones `Unsupported` |
| progressive, not-a-JPEG | 2 | `Unsupported`, `NotJpeg` |

**Compiled, not run:** `jpegGPU.cu` with HIP 5.7 (clang 17, Ubuntu's packages) for gfx1010, gfx1030
and gfx1100, through the compat header. No kernel uses scratch; the IDCT and FDCT kernels use 82
VGPRs (10-11 waves/SIMD on RDNA1/2, 16 on RDNA3), every other kernel runs at 16 waves. RDNA4
(gfx1201) needs a newer compiler than that container had. The CUDA configuration of the CMake is
written but not compiled here.

**Not done: running on a GPU.** Everything above says the device code computes the right thing
and compiles; `jpeg_gpu_check` is what says the device runtime executes it as written, and it has
not run yet. Timings do not exist.

## Build and run

```
# the host check (any machine with libjpeg-turbo)
cmake -S hip/tests/jpeg -B build/jpeg -DCHESHIRE_JPEG_GPU=OFF
cmake --build build/jpeg && build/jpeg/jpeg_cpu_check --testimages <libjpeg-turbo>/testimages photos/*.jpg

# the device check and the tool (HIP SDK / ROCm; CHESHIRE_JPEG_GPU=CUDA for NVIDIA)
cmake -S hip/tests/jpeg -B build/jpeg-gpu -DCMAKE_HIP_ARCHITECTURES="gfx1201;gfx1030;gfx1010"
cmake --build build/jpeg-gpu
build/jpeg-gpu/jpeg_gpu_check --reps 10 photos/*.jpg     # identity, then GPU vs libjpeg-turbo times
build/jpeg-gpu/gpu_jpeg/cheshire_jpeg bench photo.jpg 20
```

`jpeg_gpu_check` prints per file the subsequence and round counts and the median end-to-end decode
and q90 re-encode time against libjpeg-turbo's, after checking both are identical.

## API

```cpp
#include "jpegCodec.hpp"
cheshire::jpeg::Codec codec;                  // owns reusable device buffers; one per thread
cheshire::jpeg::Image img;
auto s = codec.decode(bytes, size, img);      // RGB or gray, libjpeg's bytes
if (s != cheshire::jpeg::Status::Ok) { /* libjpeg path */ }
std::vector<uint8_t> jpeg;
codec.encode(img.pixels.data(), img.width, img.height, img.channels, img.width * img.channels,
             {90, cheshire::jpeg::Subsampling::S420, 0}, jpeg);
```

`CHESHIRE_GPU_JPEG=0` disables the device path (`Status::NoDevice`).

## Scope

Handled: baseline and extended-sequential Huffman, 8-bit, one scan holding every component,
grayscale or three components, any sampling factors libjpeg accepts (1-4, integer ratios, up to 10
blocks per MCU), restart intervals, entropy-coded data up to 256 MB. Returned as `Unsupported`:
progressive, arithmetic, lossless, hierarchical, 12-bit, CMYK/YCCK, one scan per component. Returned
as `Corrupt`: anything the synchronised decode cannot account for exactly, and blocks outside the
IDCT limits. In every non-`Ok` case the caller keeps its CPU path, so no file decodes differently
from libjpeg because of this codec.

## Credits

The codec is new code, but its arithmetic is libjpeg-turbo's on purpose, because that is what makes
the output identical. This software is based in part on the work of the Independent JPEG Group.
[`hip/port/gpu_jpeg/ATTRIBUTION.md`](../hip/port/gpu_jpeg/ATTRIBUTION.md) lists every derived part
with its original file, copyright notice and the changes made; the IJG License ships unaltered as
`hip/port/gpu_jpeg/README.ijg`. The parallel Huffman decode follows Weissenberger & Schmidt (ICPP
2018).

## Next

1. **Run `jpeg_gpu_check` on the RX 9070, RX 6750 XT and RX 5500 XT**, Windows and Linux. Identity
   first, then the timings - end to end, since that is what a node sees.
2. **Profile before optimising.** Likely costs, in the order I expect them: the host unstuffing
   (sequential, about memcpy speed), the pageable-memory copies, the round-trip per synchronisation
   round, the per-thread scans. Each has a known fix (a device unstuffing pass, pinned staging, a
   persistent-kernel round loop, workgroup scans); none is worth doing before the numbers.
3. **Wire it into PrepareDenseScene's read**, behind a switch, and hold it to the node's existing
   bar: 107 of 107 EXRs byte-identical. Then FeatureExtraction's loads.
4. Batch decode (several files per launch) if per-file latency dominates at 4032x2268.
