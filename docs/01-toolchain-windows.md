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
