# RX 6750 XT on Windows via HIP SDK 6.2 (2026-09-15)

bench-pc: Windows 11, AMD FX-8120 (no AVX2, so an `/arch:AVX` build), Adrenalin driver as installed by
the HIP SDK 6.2.4 package (32.0.12052.2), package `cheshire-alicevision-hip6.2-windows-x64-gfx1031-avx.zip`
(HIP 6 runtime `amdhip64_6.dll`, single code object: the HIP SDK 6.2 toolchain writes a broken bundle
when given several architectures, see docs/01).

6 views, standard preset: **28.1 s**, 24 simultaneous tiles, no spills.

* [vs the CUDA reference](vs-cuda.md): masks identical, median error 0, 98.7-99.0 % within 1 % (strict PASS).
* [vs the same card under Linux / HIP 7.2](vs-linux-hip7.2-same-card.md): masks identical, median error 0,
  97.1-98.7 % within 1 %, p95 error under 0.21 %. Not bit-identical: same silicon, different compiler
  (clang 19 vs 20), runtime (HIP 6.2 vs 7.2) and OS; the cross-build noise floor is the same size as
  the cross-architecture one.
