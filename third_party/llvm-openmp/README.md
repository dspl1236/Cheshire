# LLVM OpenMP runtime (vendored)

`libomp.dll` here is the OpenMP runtime from the official LLVM release
[`llvmorg-20.1.8`](https://github.com/llvm/llvm-project/releases/tag/llvmorg-20.1.8),
taken from `clang+llvm-20.1.8-x86_64-pc-windows-msvc.tar.xz` (`bin/libomp.dll`).
`LICENSE.TXT` is `openmp/LICENSE.TXT` from the same tag: Apache-2.0 **WITH LLVM-exception**.

## Why it is here

The Windows packages need an OpenMP runtime. CMake's `FindOpenMP` selects `-openmp:llvm` because
AliceVision uses OpenMP 3.0+ constructs, and the redistributable `vcomp140.dll` is OpenMP 2.0
only. Visual Studio supplies `libomp140.x86_64.dll` for that, and it is present on any machine
with VS - but **not** in the VC++ Redistributable, so a user machine does not have it. Packages
therefore have to ship one.

The obvious candidate is Microsoft's copy, and that is what was shipped until 2026-09-21. It sits
at

    VC\Redist\MSVC\<ver>\debug_nonredist\x64\Microsoft.VC145.OpenMP.LLVM\libomp140.x86_64.dll

and `debug_nonredist` is Microsoft's marker for files outside the Distributable Code terms.
Attribution does not change a licence grant, so citing it would not have helped.

The file's own metadata gives the way out - `ProductName: LLVM* OpenMP* Runtime Library`,
`Company: LLVM`. Microsoft ships LLVM's runtime. Upstream LLVM's build of the same code is
Apache-2.0 with the LLVM exception, which permits redistribution **on condition** of preserving
the licence and notice. So the packages ship LLVM's binary and this `LICENSE.TXT` beside it.

## The rename, and why it is safe

Binaries built with `-openmp:llvm` import the name `libomp140.x86_64.dll`; LLVM ships the same
runtime as `libomp.dll`. Packaging copies it under the imported name.

That is only sound if LLVM's runtime exports everything the package imports, and the export tables
are *not* identical: Microsoft's has 25 symbols LLVM 20.1.8 lacks, all `OMP_GET_DEVICES_*` /
memspace / allocator entry points, which are OpenMP 6.0 additions with Fortran-style bindings.
What settles it is imports rather than exports - measured on the CUDA package:

    80 binaries import from libomp140.x86_64.dll, 32 distinct symbols
    all 32 present in LLVM's runtime (__kmpc_barrier, __kmpc_critical, __kmpc_dispatch_*, ...)

So nothing imports the 25. If a future AliceVision starts using OpenMP 6.0 memory management,
that changes, and `scripts/windows/verify-cuda-stages.ps1` is where it would show up - as a load
failure rather than silently.

## Updating it

Download the matching `clang+llvm-<ver>-x86_64-pc-windows-msvc.tar.xz` from the LLVM release,
extract `bin/libomp.dll` and `openmp/LICENSE.TXT` from the tag, replace both files here, and
re-run the import check before trusting it.
