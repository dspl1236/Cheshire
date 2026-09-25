# Attribution: OpenEXR, Imath and puff

CheshireEXR, the GPU EXR decoder in this directory, is new code written for Cheshire. It is not a
fork of OpenEXR or of zlib, and none of their source files are included here. Its output has to be
identical to OpenEXR's, so it reproduces behaviour from the projects below. Each part is listed with
its origin and what was changed, and the licences are included.

| Cheshire code | reproduces | from | licence |
|---|---|---|---|
| `exrTypes.hpp` `unpredict` | the predictor OpenEXR undoes after ZIP and RLE decompression (`reconstruct()`) | OpenEXR, `src/lib/OpenEXRCore/internal_zip.c` (v3.1.5) | BSD-3-Clause, Copyright (c) Contributors to the OpenEXR Project. [LICENSE.OpenEXR.md](LICENSE.OpenEXR.md) |
| `exrTypes.hpp` `interleavedSource`, `exrStages.hpp` `ConvertPixels` | the byte reordering after decompression (`interleave()`), here read in place rather than applied as a pass | same file | same |
| `exrTypes.hpp` `rleDecode` | OpenEXR's run-length format | OpenEXR, `internal_rle.c` | same |
| `exrHost.cpp` | the file layout: header, channel list order, lines per chunk per compression, the chunk offset table and chunk leaders, the "packed size equals unpacked size means stored raw" rule | the OpenEXR file format specification and OpenEXR's `chunk.c` and `decoding.c` | same |
| `exrTypes.hpp` `halfToFloatBits` | half to float, bit for bit, NaN payloads included | Imath, `imath_half_to_float()` in `half.h` (v3.1.9) | BSD-3-Clause, Copyright Contributors to the OpenEXR Project. [LICENSE.Imath.md](LICENSE.Imath.md) |
| `exrTypes.hpp` `huffDecode` (the slow path) | the canonical Huffman decoding loop | puff.c, Mark Adler (zlib `contrib/puff`) | zlib licence, below |
| `exrTypes.hpp` `buildHuff`, `zlibInflate` | inflate per RFC 1950 and RFC 1951, rejecting exactly what zlib's `inflate()` rejects: over-subscribed and incomplete codes (zlib's `inflate_table()` rules), invalid symbols, distances before the start, stored-block length, the adler32 trailer | RFC 1950, RFC 1951, zlib's behaviour | written from the RFCs; no zlib source used |

How they were changed: every routine was rewritten as a C++ function that runs on both the host and
a GPU (`__host__ __device__`), one chunk or one sample at a time. OpenEXR's SIMD variants are
replaced by their scalar definitions, and the interleave is folded into the per-pixel read.
Buffering, threading, error reporting and memory management are removed.

## puff.c notice (zlib licence)

```
Copyright (C) 2002-2013 Mark Adler, all rights reserved
version 2.3, 21 Jan 2013

This software is provided 'as-is', without any express or implied
warranty.  In no event will the author be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

Mark Adler    madler@alumni.caltech.edu
```

The Cheshire code is licensed under the MPL-2.0 (see the repository's LICENSE and NOTICE). The
OpenEXR Project and Mark Adler do not endorse this software, and it is not affiliated with them.
