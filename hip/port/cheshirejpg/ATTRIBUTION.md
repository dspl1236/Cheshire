# Attribution: libjpeg-turbo and the Independent JPEG Group

**This software is based in part on the work of the Independent JPEG Group.**

CheshireJPG, the GPU JPEG codec in this directory, is new code, written for Cheshire and not a fork of any JPEG
library. Its purpose is to produce output identical to libjpeg-turbo's, so the integer arithmetic,
tables and edge rules of several libjpeg-turbo source files are reproduced in it. Those files are
covered by the IJG License (libjpeg-turbo's modifications to IJG code are also under the IJG
License). The IJG README, which contains that license, is included unaltered as
[README.ijg](README.ijg), and libjpeg-turbo's license roll-up as
[LICENSE.libjpeg-turbo.md](LICENSE.libjpeg-turbo.md).

The reference was libjpeg-turbo 2.1.5 (tag `2.1.5`, commit `3b19db4e6e7493a748369974819b4c5fa84c7614`).
None of its source files are included here. The code in this directory is Cheshire's, licensed under
the MPL-2.0 (see the repository's LICENSE and NOTICE), and the parts listed below are derived from
the IJG/libjpeg-turbo files named.

## What is derived from where, and how it was changed

The IJG License asks that changes to the original files be clearly indicated. Across all of the
parts below, the changes are the same kind: the C routines were rewritten as C++ functions that
run on both the host and a GPU (`__host__ __device__`), one block, sample or pixel at a time
instead of whole rows or MCU rows at a time; the lookup tables are computed inline rather than
built into per-image arrays; and libjpeg's buffering, suspension, error handling and memory
manager are removed. Arithmetic that must stay bit-exact was kept as the original computes it.
Where the result can differ it is said so below.

| Cheshire file | reproduces | original file | copyright |
|---|---|---|---|
| `jpegTypes.hpp` `idctIslow` | accurate integer inverse DCT | `jidctint.c` | Copyright (C) 1991-1998, Thomas G. Lane. Modification developed 2002-2018 by Guido Vollbeding. libjpeg-turbo Modifications: Copyright (C) 2015, 2020, D. R. Commander. |
| `jpegTypes.hpp` `idctRangeLimit` | post-IDCT range-limit table, as a function | `jdmaster.c` (`prepare_range_limit_table`) | Copyright (C) 1991-1997, Thomas G. Lane. Modified 2002-2009 by Guido Vollbeding. libjpeg-turbo Modifications: Copyright (C) 2009-2011, 2016, 2019, 2022, D. R. Commander. Copyright (C) 2013, Linaro Limited. Copyright (C) 2015, Google, Inc. |
| `jpegTypes.hpp` `fdctIslow` | accurate integer forward DCT | `jfdctint.c` | Copyright (C) 1991-1996, Thomas G. Lane. libjpeg-turbo Modifications: Copyright (C) 2015, 2020, D. R. Commander. |
| `jpegTypes.hpp` `quantize`, `jpegHost.cpp` `buildQuantEncode` | reciprocal quantiser (`quantize`, `compute_reciprocal`) | `jcdctmgr.c` | Copyright (C) 1994-1996, Thomas G. Lane. libjpeg-turbo Modifications: Copyright (C) 1999-2006, MIYASAKA Masaru. Copyright 2009 Pierre Ossman for Cendio AB. Copyright (C) 2011, 2014-2015, D. R. Commander. |
| `jpegTypes.hpp` `yccToRgb` | YCbCr to RGB tables and conversion | `jdcolor.c`, `jdcolext.c` | Copyright (C) 1991-1997, Thomas G. Lane. Modified 2011 by Guido Vollbeding. libjpeg-turbo Modifications: Copyright 2009 Pierre Ossman for Cendio AB. Copyright (C) 2009, 2011-2012, 2014-2015, 2023, D. R. Commander. Copyright (C) 2013, Linaro Limited. |
| `jpegTypes.hpp` `rgbToY`, `rgbToCb`, `rgbToCr` | RGB to YCbCr tables and conversion | `jccolor.c`, `jccolext.c` | Copyright (C) 1991-1996, Thomas G. Lane. libjpeg-turbo Modifications: Copyright 2009 Pierre Ossman for Cendio AB. Copyright (C) 2009-2012, 2015, 2022, D. R. Commander. Copyright (C) 2014, MIPS Technologies, Inc., California. |
| `jpegTypes.hpp` `huffExtend`, `huffDecode`; `jpegStages.hpp` `decodeRun`; `jpegHost.cpp` `buildHuffDecode` | Huffman decoding: `HUFF_EXTEND`, the canonical-code derived table and lookahead, DC prediction, run lengths including past-the-end coefficient indices | `jdhuff.c`, `jdhuff.h` | Copyright (C) 1991-1997, Thomas G. Lane. libjpeg-turbo Modifications: Copyright (C) 2009-2011, 2015-2016, 2018-2019, 2021, D. R. Commander. Copyright (C) 2018, Matthias Räncker. |
| `jpegStages.hpp` `upsampled` | h2v1, h1v2 and h2v2 fancy upsampling, plain replication | `jdsample.c` | Copyright (C) 1991-1996, Thomas G. Lane. libjpeg-turbo Modifications: Copyright 2009 Pierre Ossman for Cendio AB. Copyright (C) 2010, 2015-2016, D. R. Commander. Copyright (C) 2014, MIPS Technologies, Inc., California. Copyright (C) 2015, Google, Inc. Copyright (C) 2019-2020, Arm Limited. |
| `jpegStages.hpp` `upsampled` (edge rows) | context rows at the top and bottom of the image | `jdmainct.c` | Copyright (C) 1994-1996, Thomas G. Lane. libjpeg-turbo Modifications: Copyright (C) 2010, 2016, D. R. Commander. |
| `jpegStages.hpp` `EncodePlanes` | h2v1, h2v2 and integer downsampling, right-edge expansion | `jcsample.c` | Copyright (C) 1991-1996, Thomas G. Lane. libjpeg-turbo Modifications: Copyright 2009 Pierre Ossman for Cendio AB. Copyright (C) 2014, MIPS Technologies, Inc., California. Copyright (C) 2015, 2019, D. R. Commander. |
| `jpegStages.hpp` `EncodePlanes` (edge rows) | bottom-edge expansion | `jcprepct.c` | Copyright (C) 1994-1996, Thomas G. Lane. libjpeg-turbo Modifications: Copyright (C) 2022, D. R. Commander. |
| `jpegStages.hpp` `EncodeDummies` | dummy blocks at the right and bottom edges | `jccoefct.c` | Copyright (C) 1994-1997, Thomas G. Lane. |
| `jpegStages.hpp` `encodeBlock`, `EncodePad` | `encode_one_block`, the ones-filled final byte | `jchuff.c` | Copyright (C) 1991-1997, Thomas G. Lane. libjpeg-turbo Modifications: Copyright (C) 2009-2011, 2014-2016, 2018-2022, D. R. Commander. Copyright (C) 2015, Matthieu Darbois. Copyright (C) 2018, Matthias Räncker. Copyright (C) 2020, Arm Limited. |
| `jpegHost.cpp` `kNatural` and `Zigzag` | `jpeg_natural_order` with its safety entries | `jutils.c` | Copyright (C) 1991-1996, Thomas G. Lane. libjpeg-turbo Modifications: Copyright (C) 2022, D. R. Commander. |
| `jpegHost.cpp` `qualityTables` and the quantisation tables | `jpeg_quality_scaling`, `jpeg_add_quant_table`, the Annex K tables | `jcparam.c` | Copyright (C) 1991-1998, Thomas G. Lane. Modified 2003-2008 by Guido Vollbeding. libjpeg-turbo Modifications: Copyright (C) 2009-2011, 2018, D. R. Commander. |
| `jpegHost.cpp` standard Huffman tables | the Annex K.3 tables | `jstdhuff.c` | Copyright (C) 1991-1998, Thomas G. Lane. libjpeg-turbo Modifications: Copyright (C) 2013, 2022, D. R. Commander. |
| `jpegHost.cpp` `writeHeaders` | the marker layout written for `jpeg_set_defaults` | `jcmarker.c` | Copyright (C) 1991-1998, Thomas G. Lane. Modified 2003-2010 by Guido Vollbeding. libjpeg-turbo Modifications: Copyright (C) 2010, D. R. Commander. |
| `jpegHost.cpp` `makeDecodeGeom` (colour space) | the colour-space guess from JFIF, Adobe and component ids | `jdapimin.c` (`default_decompress_parms`) | Copyright (C) 1994-1998, Thomas G. Lane. libjpeg-turbo Modifications: Copyright (C) 2016, 2022, D. R. Commander. |
| `jpegHost.cpp` `parseJpeg` (APP0, APP14) | recognising the JFIF and Adobe markers | `jdmarker.c` | Copyright (C) 1991-1998, Thomas G. Lane. libjpeg-turbo Modifications: Copyright (C) 2012, 2015, 2022, D. R. Commander. |

Where the result can differ from the original: the IDCT uses 32-bit integers where libjpeg's C
code uses `JLONG` (64-bit on LP64). The two agree exactly inside the limits `kMaxDequantized` and
`kMaxColumnPass`. `scripts/jpeg_idct_bounds.py` derives those limits, and every block a real 8-bit
encoder produces falls inside them. Outside them the codec does not decode the block; it returns
`Status::Corrupt` so that the caller's libjpeg decodes the file.

Everything else in this directory is Cheshire's own design and is not derived from libjpeg or
libjpeg-turbo. That covers the parallel Huffman synchronisation (after Weissenberger & Schmidt,
*Massively Parallel Huffman Decoding on GPUs*, ICPP 2018), the scans, the parallel bit packing and
byte stuffing, the backends and the API.

The IJG and the libjpeg-turbo Project do not endorse this software, and it is not affiliated with
them.
