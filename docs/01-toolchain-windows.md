# Windows toolchain (verified 2026-09-03 on RX 9070 / gfx1201)

## Install (no admin required)

```bat
uv venv --python 3.12 tools\venv-rocm
uv pip install --python tools\venv-rocm\Scripts\python.exe ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
tools\venv-rocm\Scripts\rocm-sdk test      # 25 tests OK
tools\venv-rocm\Scripts\rocm-sdk targets   # gfx1100;gfx1201;gfx1151;gfx1150;gfx1200;gfx1101;gfx1102
```

Portable CMake 4.4.3 and Ninja 1.13.2 zips live in `tools\cmake` and `tools\ninja`.
MSVC 2026 Build Tools (14.50) + Windows SDK 10.0.26100 provide the CRT/SDK headers.
Driver: Adrenalin 26.8.1 (ROCm 7.2.1 needs 26.2.2+).

## Layout of the pip SDK

| | |
|---|---|
| root | `tools/venv-rocm/Lib/site-packages/_rocm_sdk_devel` (`rocm-sdk path --root`) |
| runtime | `root/bin/amdhip64_7.dll`, `amd_comgr0702.dll` |
| compiler | `root/lib/llvm/bin/clang++.exe`, `clang-cl.exe`, `lld-link.exe` (Clang 22) |
| device libs | `root/lib/llvm/amdgcn/bitcode` (clang does NOT find these on its own) |
| CMake configs | `root/lib/cmake/{hip,hip-lang,rocprim,hipcub,rocthrust,...}` |
| import lib | `root/lib/amdhip64.lib` |

## CMake rules that cost time to learn

1. Every ROCm path handed to CMake or exported as an env var must use **forward slashes**.
   `HIP_PATH=D:\...` makes CMake's generated `CMakeHIPCompiler.cmake` fail with
   "Invalid character escape".
2. Pass `-DCMAKE_HIP_FLAGS="--rocm-path=<root> --rocm-device-lib-path=<root>/lib/llvm/amdgcn/bitcode"`
   or clang errors with "cannot find ROCm device library".
3. CMake refuses to mix `cl.exe` for CXX with clang for HIP
   ("mixes Clang and MSVC ... not supported"). Use the same clang for both:
   * `clang` variant: `clang++` for CXX and HIP (GNU-style flags, links via `clang++ -fuse-ld=lld`).
   * `clangcl` variant: `clang-cl` for CXX and HIP (MSVC-style flags, links via `lld-link`
     through CMake's `vs_link_exe`). **This is the variant for AliceVision**, whose Windows
     CMake assumes MSVC-style switches.
4. `vcvarsall.bat x64` prints a `vswhere.exe` warning on this Build Tools install; harmless.

`scripts\env.cmd` sets all of this up; `scripts\build-tests.cmd [arch] [clang|clangcl]` builds and
runs `hip/tests`.

## Smoke-test results (both variants identical)

```
[0] AMD Radeon RX 9070 arch=gfx1201 CUs=28 warp=32 VRAM=16304 MB shared/block=65536
    maxTex2D=16384x16384 canMapHost=1 managed=0
vadd: OK
hipMallocMipmappedArray float4 256x128 x4 levels: OK
surf2Dwrite level0: OK
mip chain built via tex2D+surf2Dwrite: OK
tex2DLod lod=0.0 / 1.0 / 2.5: OK
pitch2D float texture: OK
hipMallocMipmappedArray half4: OK
hipMallocManaged 64MB: OK          (note managed=0 in props: coarse-grained, no page migration)
hipHostMalloc(mapped)+device ptr: OK
```

Conclusion: nothing in AliceVision's `depthMap/cuda` needs a feature HIP-on-Windows lacks.

## Shipping a self-contained zip
`scripts/package_windows.py` copies the install tree and walks the import tables
(`llvm-objdump -p`) of every exe/dll to pull the transitive closure of vcpkg and ROCm
runtime DLLs into `bin/` (93 DLLs for the gfx1201 build). One DLL is *not* in any import
table: `amd_comgr0702.dll`. `amdhip64_7.dll` loads it with `LoadLibrary` when it first
needs a code object; without it `hipGetDeviceCount` returns 0 and AliceVision reports
"No CUDA-Enabled GPU" with no other diagnostic. The packager copies it explicitly (the
Linux bundle has the same rule for `libamd_comgr.so`). Verified by running the packaged
`aliceVision_depthMapEstimation` on the 6-view set with `PATH` reduced to `System32`.

Two more things a clean machine taught (2026-09-15, bench-pc: RX 6750 XT, AMD FX-8120, no Visual
Studio): the packager's import walk stops at the MSVC runtime (`msvcp140*`, `vcruntime140*`,
`concrt140`) and the LLVM OpenMP runtime `libomp140.x86_64.dll`, because on the build PC those
live in System32. On a machine without them the exe exits with `0xC0000135` (STATUS_DLL_NOT_FOUND)
and prints nothing. `package_windows.py` now bundles them from the VS Redist tree. And the build
uses `/arch:AVX2` (needed so Eigen's SSE3 paths inline under clang-cl); a Bulldozer-era CPU
without AVX2 exits with `0xC000001D` (STATUS_ILLEGAL_INSTRUCTION). `CHESHIRE_ARCH_FLAG=/arch:AVX`
builds a variant for those.

Third lesson from the same machine: its Adrenalin driver was dated 2025-09 and installs
`amdhip64_6.dll`; the ROCm 7.2.1 packages carry `amdhip64_7.dll`, and with that driver
`hipGetDeviceCount` returns `hipErrorNoDevice` (AliceVision prints "No CUDA-Enabled GPU").
The HIP 7 runtime needs Adrenalin 26.2.2 or newer, as the ROCm-for-Windows release notes say.

And the last one: with Adrenalin 26.8 (driver 32.0.21045.5002, 2026-08-16) the HIP 7.2.1
runtime still answers `hipErrorNoDevice` for the RX 6750 XT, and AMD's Windows support table
marks every RX 6000 card unsupported by the current HIP SDK. RDNA2 code objects in a Windows
build are therefore inert; RX 6000 owners get the Linux bundle, where the same card works. A
Windows RDNA2 build would need the older HIP SDK 6.x toolchain and runtime (`amdhip64_6.dll`,
which the driver ships), a separate build environment this repo does not have.

## RDNA2 on Windows through HIP SDK 6.2 (2026-09-15)

AMD's Windows driver ships the HIP 6 runtime (`amdhip64_6.dll`), and that runtime still
enumerates RX 6000 cards where the 7.2 one refuses them. Building against the HIP SDK 6.2.4
toolchain (clang 19; installed on bench-pc and copied to `tools/rocm-6.2`; no llvm-lib; device
libs under `amdgcn/bitcode`) needs three things beyond the 7.2 recipe:

1. `CHESHIRE_ROCM_PATH`, `CHESHIRE_LLVM_BIN`, `CHESHIRE_DEVICE_LIB_PATH` (env.cmd /
   build-alicevision.cmd) point at the copied tree.
2. clang 19 against the MSVC 14.50 STL: `-D__builtin_verbose_trap(x,y)=__builtin_trap()` on both
   the host and the HIP flags (`CHESHIRE_EXTRA_CXXFLAGS`, `CHESHIRE_HIP_EXTRA_FLAGS`). A force-included
   shim does not reach HIP TUs, whose runtime wrapper includes the STL first. CMake reads the host
   flags only on the first configure, so a build directory configured without the macro must be wiped.
3. **One architecture per build.** With several `--offload-arch` values the HIP SDK 6.2 toolchain
   writes an offload bundle whose entries are all the same code object (the last one compiled;
   verified by parsing `__CLANG_OFFLOAD_BUNDLE__`: six entries, one md5, `e_flags` of gfx1102).
   The runtime then reports "program ISA gfx1102 is not compatible with the device ISA gfx1031",
   `hipGetSymbolAddress` returns `hipErrorNoBinaryForGpu`, and the first constant upload crashes.
   Single-architecture builds are correct (entry `e_flags` 0x37 = gfx1031).

Result on bench-pc's RX 6750 XT: the 6-view set in 28.1 s, masks identical to CUDA, 98.7 %
within 1 % (`docs/validation/monstree-mini6-rx6750xt-windows-hip6/`). Against the same card's
Linux / HIP 7.2 output: masks identical, median 0, 97-98.7 % within 1 %, not bit-identical.

## Meshroom pairing on Windows (2026-09-16)

`scripts/windows/meshroom-pair.cmd` + `meshroom-pair-launcher.exe` (source
`scripts/windows/meshroom-pair-launcher.cpp`, built by `scripts/windows/build-launcher.cmd` with the
toolchain's clang-cl, static CRT, Win32 only). Meshroom 2023.3 runs the DepthMap node as
`aliceVision_depthMapEstimation {allParams}` through a shell (`Popen(..., shell=True)` in
`meshroom/core/desc.py`), which cmd.exe resolves on PATH with PATHEXT, so the replacement has to be
an `.exe` with the original name; a `.cmd` would also resolve but only when nothing else does, and
batch quoting of `%*` is fragile. The launcher:

* keeps the original as `aliceVision_depthMapEstimation.cuda.exe` and reads the package path from
  `aliceVision_depthMapEstimation.cheshire.txt` beside itself;
* runs `nvidia-smi -L` (System32, installed with the NVIDIA driver); exit 0 means CUDA, else HIP;
  `CHESHIRE_DEPTHMAP=cuda|hip` overrides;
* on the HIP path sets `ALICEVISION_ROOT` to the package and puts `<package>\bin` first on PATH
  (the package is self-contained: vcpkg, ROCm, MSVC and OpenMP DLLs), and drops
  `--sgmFilteringAxes` like the Linux wrapper (removed upstream);
* `CreateProcessW` with the arguments re-quoted by the MSVC rules, waits, returns the child's exit code.

Verified on bench-pc (GTX 1080 Ti): auto-selects CUDA, one view in 15.4 s through the launcher, and
`CHESHIRE_DEPTHMAP=hip` switches to the HIP package (which then correctly reports no AMD device).
Verified on this PC (RX 9070): a complete `meshroom_batch` photogrammetry run with
`FeatureExtraction:forceCpuExtraction=True` whose DepthMap node logs
`[cheshire] DepthMap backend: HIP`, see the README.
