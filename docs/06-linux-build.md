# Linux build (for the AMD photogrammetry node)

The production node pattern (haus-infrastructure `photogrammetry-node.md`) is Linux Mint 22
(= Ubuntu 24.04) with Meshroom's `meshroom_batch` and the `reconstruct`/`newjob` scripts.
`house-pc` itself has a GTX 1080 Ti and no AMD card, so the Linux HIP build is produced and
packaged here and deployed to whatever AMD box replaces or joins it.

## Build environment: WSL2 Ubuntu 24.04 + ROCm 7.2 (this PC)

```bash
# Windows side (once): wsl --install -d Ubuntu-24.04 --no-launch
# inside the distro (root):
apt-get update && apt-get install -y build-essential cmake ninja-build git wget curl python3 python3-venv
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
apt-get install -y ./amdgpu-install_7.2.70200-1_all.deb
amdgpu-install -y --usecase=wsl,rocm --no-dkms      # ~10 min, installs /opt/rocm-7.2.0
/opt/rocm/bin/rocminfo | grep -E "gfx|Marketing"     # RX 9070 shows up as gfx1201
# Kitware CMake (AliceVision wants >= 3.30; noble ships 3.28) + superbuild prerequisites:
#   see scripts/linux/build-deps.sh header and docker/Dockerfile_ubuntu_deps upstream
```

Same-version base as the node (Ubuntu 24.04 / glibc 2.39), so the bundle transfers.

### What WSL can and cannot do (measured with `hip/tests`)

| | WSL2 + ROCm 7.2 | native Windows ROCm 7.2.1 |
|---|---|---|
| plain kernels, `hipMalloc`, copies | OK (launch 1.9 us) | OK |
| `hipMallocPitch`, `hipMalloc3D` | **invalid argument** | OK |
| arrays, textures, surfaces, mipmaps | **operation not supported** (`maxTexture2D = 0`) | OK (except surf stores to half arrays) |
| `rocprofv3` | present, but the depth-map kernels cannot run | n/a |

So WSL is a **build box**: it compiles the HIP backend for every RDNA target, but
`aliceVision_depthMapEstimation` cannot execute there and neither can it be profiled there.
Correctness of the kernels is established on Windows from the same sources (docs/04); the
Linux binaries need a native-Linux AMD machine for a run-through.

## Steps

1. `scripts/linux/build-deps.sh` - AliceVision's own dependency superbuild
   (`ALICEVISION_BUILD_DEPENDENCIES=ON`), CUDA-free: no PopSift, CCTag, AprilTag, OpenCV,
   ONNX Runtime, USD, PCL, E57, FFmpeg, Python bindings. Everything the node's default
   `meshroom_batch` graph needs (features, matching, SfM, PrepareDenseScene, DepthMap,
   DepthMapFilter, Meshing, MeshFiltering, Texturing) is in. Install: `/opt/AliceVision_deps`.
2. `scripts/linux/build-alicevision.sh [configure|build|install|bundle] [gfx list]` -
   applies `scripts/apply_hip_patch.py`, configures with `ALICEVISION_USE_HIP=ON`,
   `CMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++`, default targets
   `gfx1201;gfx1200;gfx1100;gfx1101;gfx1102`, installs to `/opt/AliceVision_hip` and makes
   AliceVision's relocatable `bundle` (binaries + libraries + share/) like the Meshroom
   tarball the node uses today.
3. `scripts/linux/run-depthmap.sh` - the node-side equivalent of the Windows validation
   runner (same DepthMap command line as Meshroom 2023.3's `standard` preset).
4. `scripts/linux/gpu-precheck.sh` - drop-in for the `nvidia-smi` precondition in the
   node's `reconstruct`: reports the GPU via `amd-smi` / `rocm-smi` / `nvidia-smi`, whichever
   exists.

## Notes

* Background processes started from a `wsl -e bash -c` session die with the session;
  start long builds with `setsid -f ... < /dev/null` or from a Windows-side background task.
* `-fgpu-rdc` (relocatable device code) is expected to work on Linux; the unity device TU is
  still the default (`ALICEVISION_HIP_RDC=ON` switches).
* The Boost.WinAPI define block in the patch is `if (WIN32)`-guarded and does nothing here.

## Result (2026-09-03)

* Dependency superbuild: OK after three generic fixes (CMake 4 policy floor for lz4/OpenMesh,
  `-include cstdint` for assimp's Draco on GCC 13, pybind11 needed by OpenImageIO). About
  40 minutes on 12 threads. 157 libraries in `/opt/AliceVision_deps`.
* AliceVision + HIP backend: **671/671 targets, 0 failures** on the first full pass
  (after `ALICEVISION_BUILD_LIDAR=OFF`; E57 was excluded from the deps). HIP 7.2.26015,
  targets `gfx1201;gfx1200;gfx1100;gfx1101;gfx1102` in one fat binary. The Linux build
  was noticeably less trouble than Windows: no STL shim, no OpenMP shim, no arch flag issue.
* Bundle: `cmake --build . --target bundle` -> `/opt/AliceVision_hip/bundle` (167 MB;
  bin/ with 208 tool entries, lib/ including `libamdhip64.so.7`, `libhsa-runtime64.so.1`,
  `libamd_comgr`, share/). Run it like the Meshroom tarball: `LD_LIBRARY_PATH=<bundle>/lib`
  and `ALICEVISION_ROOT=<bundle>`.
* Tarball: `build/cheshire-alicevision-hip-linux-x64-rocm7.2-<commit>.tar.gz` (60 MB),
  also staged on `house-pc:~/apps/cheshire/`. It loads and runs its CPU tools on Mint 22.
* Under WSL the bundle finds the GPU (`hardwareResources` reports the RX 9070 through the
  HIP runtime) and `depthMapEstimation` fails exactly where expected, at the first array /
  texture creation: `CUDA Error: operation not supported`. A native-Linux AMD box is
  needed for the first real Linux depth-map run; the kernels are the Windows-validated ones.

## Deploying to an AMD node

1. Unpack the tarball to `~/apps/cheshire/bundle`; the ROCm user-space runtime is inside the
   bundle, but the **amdgpu kernel driver** must be present (Ubuntu 24.04 stock kernel 6.8+
   is fine for RDNA3; RDNA4 wants the ROCm 7.2 `amdgpu-dkms` or a 6.12+ kernel) and the
   service user must be in the `render` and `video` groups.
2. Meshroom must match the AliceVision generation: this tree is AliceVision 3.4-dev, and the
   node's Meshroom 2023.3.0 graph passes options these binaries no longer accept
   (`--sgmFilteringAxes`). Use a Meshroom release built for AliceVision >= 3.3 (2025.x) and
   point it at the bundle's `bin`.
3. Replace the `nvidia-smi` precondition in `reconstruct` with `scripts/linux/gpu-precheck.sh`
   (amd-smi / rocm-smi / nvidia-smi, whichever exists) and the UI's GPU tiles accordingly.
4. Validate with `scripts/linux/run-depthmap.sh <bundle> <cache> <out> <reference DepthMap dir>`
   on a copied Meshroom cache (`data/ref/monstree-*` here), same procedure as docs/04.

## Pairing a Meshroom bundle with the HIP DepthMap (2026-09-04)

Meshroom runs every node as an `aliceVision_*` executable from `<Meshroom>/aliceVision/bin`,
and only `DepthMap` needs the GPU. So a production node keeps its Meshroom 2023.3 tarball
and swaps one file: `scripts/linux/meshroom-pair.sh ~/apps/Meshroom-2023.3.0` renames the
bundle's `aliceVision_depthMapEstimation` to `.cuda` and writes a wrapper in its place that
execs `~/apps/cheshire/bundle/bin/aliceVision_depthMapEstimation` with the Cheshire bundle's
`lib/` in front of `LD_LIBRARY_PATH` (so the HIP build resolves its own OpenImageIO / Boost /
ROCm sonames, not the 2023 ones) and sources `~/apps/cheshire/env.sh` for per-node knobs
(`HSA_OVERRIDE_GFX_VERSION`, `CHESHIRE_BRIDGE_*`). `--unpair` restores the CUDA binary.
`node-amd-setup.sh pair` does the same from the node script.

Meshroom 2023.3's DepthMap command line is accepted unchanged by the newer AliceVision the
HIP build is based on (that is exactly the command line docs/04 validates with). Feature
extraction must stay on the CPU (`FeatureExtraction:forceCpuExtraction=True`): PopSift is
CUDA-only. Done on house-pc for the haus-infrastructure photogrammetry service, whose
`reconstruct` wrapper now detects the vendor (`nvidia-smi` vs `/dev/kfd` + `amdgpu`) and
whose dashboard reads load / VRAM / temperature from the `amdgpu` sysfs tree.

## RDNA2 / house-pc with an RX 6700 XT (prepared 2026-09-03)

* Bundle rebuilt with `gfx1030` added (targets now gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201),
  `build/cheshire-alicevision-hip-linux-x64-rocm7.2-39f0251-rdna2.tar.gz`, staged as the only
  tarball in `house-pc:~/apps/cheshire/` together with `scripts/` (node-amd-setup.sh,
  run-depthmap.sh, gpu-precheck.sh, compare_depthmaps.py) and a Python venv for the compare tool.
* ROCm 7.2 user-space still ships gfx1030 code objects and the HSA runtime knows gfx1031; the
  6700 XT runs the gfx1030 code with `HSA_OVERRIDE_GFX_VERSION=10.3.0` (set in `env.sh` by the
  setup script). Mint 22.3's kernel has `amdgpu` for RDNA2; nothing from AMD's apt repo needed.
* Known caveat: community reports of a SIGSEGV regression on gfx1031/1032 with ROCm >= 6.4.3,
  unconfirmed on 7.2. Fallback if it shows up: rebuild the bundle against ROCm 6.4.
* Procedure on the node after the card swap (1080 Ti out): `sudo bash scripts/node-amd-setup.sh setup`,
  re-login, `scripts/node-amd-setup.sh check`, then `scripts/node-amd-setup.sh run monstree-mini6`
  (compares against the CUDA DepthMap produced on the same machine). Meshroom 2023.3 cannot drive
  these binaries; the test runs `aliceVision_depthMapEstimation` directly.

## First real Linux GPU: house-pc with a Radeon RX 5500 XT (RDNA1, gfx1012, 8 GB) — 2026-09-03

What it took to get from "bundle on disk" to "HIP sees the card", each a real deployment lesson:

1. **The WSL build box ships the WSL flavour of the HSA runtime.** `libhsa-runtime64.so.1`
   from `amdgpu-install --usecase=wsl` probes `/dev/dxg` and `/dev/accel*` and never opens
   `/dev/kfd`; on a real Linux box `hsa_init` fails with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`
   and the tools say "No CUDA-Enabled GPU". The bundle step now swaps in `hsa-rocr` from
   AMD's apt repo automatically (`build-alicevision.sh`).
2. **`libamd_comgr` must be in the bundle.** HIP `dlopen`s the code-object manager at runtime,
   so `fixup_bundle` never sees it; without it every device fails with
   "Failed to load COMGR library" / "Code object manager initialization failed". 160 MB, now
   included (bundle tarball 116 MB compressed).
3. **RDNA1 is fine with ROCm 7.2's runtime** once the two above are fixed: HIP enumerates the
   RX 5500 XT natively as gfx1012 (compute capability 10.1); no `HSA_OVERRIDE_GFX_VERSION`
   needed when the fat binary carries gfx1012 (targets are now
   gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201). Presenting the card as
   gfx1030 and running the RDNA2 code object crashes with
   `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`; do not do that.
4. **Mipmapped arrays are not supported on this Linux stack**
   (`hipMallocMipmappedArray`: "Mipmap not supported on one of the devices"). Everything
   else the depth map needs works on the card (plain float4/half4/uchar4 textures, pitch2D
   textures, the memory bridge, 2.8 us launches). The compat layer now emulates mipmaps:
   one plain array + texture per level and a device-side level table behind the 64-bit
   handle; `tex2DLod` becomes a table lookup + `tex2D`. AliceVision only samples at integer
   levels, and on the RX 9070 the emulated output is **bit-identical** to native mipmaps -
   but it costs ~35 % on that card (23.5 s vs 17.4 s on mini6), so the Windows build keeps
   native mipmaps (`-DCHESHIRE_NATIVE_MIPMAP`) and the Linux bundle emulates. On the RX 6750 XT
   the same experiment under Windows / HIP 6.2 (41 views: 232.3 s emulated vs 190.3 s native,
   bit-identical) lands within 3 % of this Linux bundle's 226.1 s, so emulation accounted for the
   entire Linux-vs-Windows difference on that card. Most of it was the level table, not the
   extra textures: the packed-slot sampler (2026-09-15, `mipmap_emu.h`) brings emulation to
   208.1 s on that card (9 % over native) and 17.2 s vs 16.5 s on the RX 9070, bit-identical.
   The remainder is one scalar descriptor load per sample that the compiler cannot hoist past
   the uniformity check; a hand-built hardware mip chain would remove it, at the price of
   per-generation image-descriptor layouts.
5. The half-array `surf2Dwrite` defect reproduces on Linux/RDNA1 too: it is a HIP runtime
   bug, not a Windows one. The buffer-copy mip builder covers both.
6. Small ones: `getent`-based home under `sudo` in `node-amd-setup.sh`; git must carry the
   executable bits (a `reset --hard` from the Windows checkout dropped them: exit 126).

Node-side layout: `~/apps/cheshire/{bundle,scripts,venv,env.sh}`; `env.sh` sets
`ALICEVISION_ROOT`, `LD_LIBRARY_PATH` and `PATH`. The node's static-IP profile is bound to
the old NIC name (`enp3s0`); with the new card the NIC is `enp5s0` and the box is on DHCP
(`house-pc.local`). Left untouched on purpose: the network is the user's infra.

### Running WSL-built test binaries on a node

`hip/tests` can be built in WSL for a node's card (`cmake -DCMAKE_HIP_ARCHITECTURES=gfx1031 ...`)
and copied over; they link against the bundle's `libamdhip64` through `env.sh`. One trap: a
binary that carries only the card's exact code object segfaults at the first kernel launch if
`HSA_OVERRIDE_GFX_VERSION` is set (the override makes the runtime look for gfx1030 code). Run
them with the variable unset, or build them for the override's target as well.

## Not supported: GCN 4 (Polaris), tested with an RX 570 4 GB (2026-09-11)

A PowerColor RX 570 (Ellesmere, `0x67df` rev `ef`, gfx803) went into house-pc. The kernel side
is fine: `amdgpu` brings it up, KFD registers it (`gfx_target_version 80003`, 128 SIMDs). The
ROCm 7.2 HSA runtime does not: `rocminfo` prints `Failed to map remapped mmio page on gpu_mem 0`
and fails with `HSA_STATUS_ERROR`, with or without `ROC_ENABLE_PRE_VEGA=1`,
`HSA_ENABLE_SDMA=0` or `HSA_OVERRIDE_GFX_VERSION=8.0.3`. HIP therefore sees no device, and no
code object in the bundle can change that. GCN needs a ROCm 5.x stack end to end; the bundle
is built against 7.2 and is not going to carry two runtimes. Practical floor for Cheshire:
RDNA1 (RX 5000).

The node handles it gracefully: the dashboard still shows the card (sysfs does not care
about ROCm), and `reconstruct` now asks the HIP build whether it can see a GPU before
starting a job, so an unsupported card is refused at launch instead of failing an hour later
at the DepthMap node.

## APU and Vega code objects (v0.2.1, 2026-09-11)

The bundle's default target list now includes the integrated RDNA parts (gfx1035/1036 Rembrandt,
gfx1103 Phoenix, gfx1150 Strix Point, gfx1151 Strix Halo, gfx1152/1153 Krackan) and Vega
(gfx900/906). The ROCm 7.2 compiler accepts all of them (`build/archcheck.sh` style probe);
whether the 7.2 *runtime* initialises a given APU or a Vega is untested here. APUs share system
memory, so they are the case where the bridge's VRAM budget is a software knob rather than a
board limit: expect `hipMemGetInfo` to report the carve-out, and `CHESHIRE_BRIDGE_VRAM_MB` to be
the thing to tune. The Windows RDNA3+RDNA4 package carries the RDNA3 APUs too.

### Card swaps and `HSA_OVERRIDE_GFX_VERSION` (2026-09-15)

`node-amd-setup.sh` used to write `HSA_OVERRIDE_GFX_VERSION=10.3.0` into `env.sh` for the
RX 6700 XT era, and the Meshroom pairing wrapper sources `env.sh`. With the RX 5500 XT swapped
back in, the first Meshroom DepthMap chunk died with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`
("Failed to move Gaussian filter to symbol"): the override made the runtime pick the gfx1030
code object for an RDNA1 card, the day-one RDNA1 crash. The bundle has native code objects for
every RDNA part, so the override is gone from the setup script; set it by hand only for a chip
whose gfx id the bundle lacks, and unset it when the card changes.

## Headless card and runtime PM (house-pc, 2026-09-16)

Twice the node hung with nothing running: the journal shows the display-less card going through an
`amdgpu` BACO runtime suspend/resume cycle every 8 s for hours (something polls it; the node service
is guarded, the remaining suspect is `thermald` reading hwmon) and then stopping mid-resume. Boots
where the card never suspends never hang. The fix is to keep a compute-only card out of runtime
suspend, which a display-less card gains nothing from:

```
# /etc/udev/rules.d/80-amdgpu-no-runpm.rules
ACTION=="add|change", SUBSYSTEM=="pci", DRIVER=="amdgpu", ATTR{power/control}="on"
```

(`udevadm control --reload-rules && udevadm trigger --subsystem-match=pci --action=change` applies
it without a reboot; `amdgpu.runpm=0` on the kernel command line is the equivalent.) A few watts
idle; `power/runtime_status` stays `active`.
