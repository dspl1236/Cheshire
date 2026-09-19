# One Windows package (2026-09-19)

Six Windows downloads today, one per toolchain and chip, and picking the wrong one does not fail:
the package enumerates the device, runs every kernel and returns wrong data with no error. That is
the worst failure mode in the project and it is a packaging problem, not a code one. gfx1010
(RX 5600/5700), gfx1034 (RX 6500 XT / 6400) and every APU have no Windows package at all.

This is the measurement behind collapsing them into one.

## Only five DLLs are per-chip

Hash-diffing the gfx1030 and gfx1031 packages, which differ by nothing but the GPU target:

| | files | size |
|---|---|---|
| identical | 856 | 384 MB |
| differing | 152 | 43 MB |

The 43 MB overstates it. `aliceVision_sfm.dll` is in the differing set and differs by **one byte**, at
offset 129 - inside the PE header, a build timestamp. Grepping the binaries for the target name finds
the DLLs that actually carry code objects:

    aliceVision_matching.dll   1163 KB
    popsift.dll                1135 KB
    aliceVision_fuseCut.dll    1072 KB
    aliceVision_mesh.dll        715 KB
    aliceVision_depthMap_cuda.dll 669 KB

**4.7 MB per chip**, not 427. A chip is an overlay on a shared base, and adding one - RDNA5, an APU,
whatever appears next - is a 5-DLL rebuild rather than another whole package.

## The two runtime families share more than they differ

hip6.2 (gfx1030) against rocm7.2.1 (rdna3-rdna4):

| | files | size |
|---|---|---|
| bit-identical in both | 820 | 132 MB |
| same path, different build | 185 | 59 MB per side |
| only in hip6.2 (`amdhip64_6` + two comgr) | 3 | 236 MB |
| only in rocm7.2 (`amdhip64_7` + comgr) | 2 | 119 MB |

The vcpkg DLLs and the `share/` tree are bit-identical across toolchains - they are prebuilt vcpkg
binaries, not rebuilt per compiler. Only AliceVision's own DLLs differ, because clang 19 / `/arch:AVX`
and clang 22 / `/arch:AVX2` are different builds.

**`amd_comgr0602.dll` is 107 MB of dead weight.** `amdhip64_6.dll` references `amd_comgr_2.dll`, and
nothing in the package mentions `amd_comgr0602` at all. `package_windows.py` copies it because it
globs `amd_comgr*.dll` - the glob is there because HIP loads comgr through `LoadLibrary` and it is in
no import table, which is correct, but it takes both copies.

Deduplicated, minus the dead comgr, plus the per-chip overlays: **about 540 MB for one package
covering every card from an RX 5500 to an RX 9070**, against 747 MB for just two of the six today.

## Generic targets: yes for RDNA3/4, no for RDNA1/2

Both toolchains carry `gfx10-1-generic`, `gfx10-3-generic`, `gfx11-generic` and `gfx12-generic`
bitcode. One generic code object runs on every chip in its family, including chips released after
the compiler - which is the whole point for a future architecture.

Confirmed on ROCm 7.2: a probe built `--offload-arch=gfx12-generic` runs on gfx1201 and returns
correct results, same as the chip-specific build. So RDNA3/3.5 and RDNA4 collapse to
`gfx11-generic` + `gfx12-generic`, and ROCm 7.2 already carries gfx1250/1251 bitcode for what comes
after.

RDNA1 and RDNA2 cannot follow, for two stacked reasons:

1. They need the HIP 6 runtime. The 7.2 runtime answers `hipErrorNoDevice` for an RX 6750 XT even on
   Adrenalin 26.8, and AMD's Windows support table marks every RX 6000 card unsupported by the
   current HIP SDK (docs/01).
2. HIP SDK 6.2 emits a generic target only under code object v6 - the default v5 fails with
   "gfx10-3-generic is only available on code object version 6 or better" - and with
   `-mcode-object-version=6` clang answers *"code object v6 is still in development and not ready
   for production use yet; use at your own risk"*. The runtime that would load it is the driver's,
   not ours.

Shipping a not-ready object format to hardware whose failure mode is silent wrong output is not a
trade worth making, and it buys nothing: the per-chip overlay is 4.7 MB.

## Linux

Linux is already one bundle, so it has no selection problem - but it carries 19 hand-listed code
objects, and every new chip needs another entry. Generics apply cleanly there because Linux uses
ROCm 7.2 for everything: the Linux runtime enumerates RDNA1 and RDNA2 natively, so none of the
HIP 6.2 constraint above applies. `gfx10-1-generic`, `gfx10-3-generic`, `gfx11-generic` and
`gfx12-generic` replace 17 of the 19 and cover future chips in each family.

Two things to keep straight when doing it:

- **Vega is not a free addition.** PopSift's Linux list omits gfx900/906 and the RDNA2 APUs, and
  the README records the effect (CPU SIFT there) without a reason. Vega has an architectural one:
  it is wave64 where all of RDNA is wave32, and the PopSift kernels have only ever run wave32. So
  `gfx9-generic` must not be folded into the PopSift list on the strength of "generics are
  cheaper". The RDNA2 APUs are a different case - same wave32 as the rest of RDNA2 - so
  `gfx10-3-generic` picking them up is architecturally sound, and it gives those APUs GPU SIFT for
  free.
- Check a generic build on the card before trusting the collapse, the same way the Windows half
  was checked: an identical point-cloud checksum from the same inputs, not merely a successful run.

## Detection

The launcher has to know the card before it can pick a payload, and the obvious route - a PCI device
ID table - needs maintaining for exactly the hardware that does not exist yet.

Instead: one small probe per runtime family, each loading its own `amdhip64` and reporting
`gcnArchName`. **Whichever runtime enumerates the card is the family to use**, and the arch name
picks the chip directory inside it. Detection and family selection are the same question, and a new
chip in an existing family needs no table entry at all.

### Each family has to be probed in its own process

Loading a HIP runtime that does not support the installed driver is not guaranteed to report no
device. It can take the process down. On bench-pc's RX 6750 XT, **under a scheduled task**, loading
`amdhip64_7.dll` exits `0xC0000005` before printing anything; the identical call in an interactive
SSH session returns "no device" politely and carries on.

That split is why this survived several rounds of testing. Every interactive check passed. The
failure needs a non-interactive session - which is how Meshroom runs a node, and how anything in CI
would run. Worse, the symptom is not a crash report: the wrapper sees a missing answer and says
"no AMD GPU with a matching payload", so the bundle would look broken on exactly the cards that
need the HIP 6 fallback, for a reason the message actively misdirects you away from.

So `cheshireSelectPayload` spawns `cheshire-detect.exe --family <n>` once per family and reads the
one line it prints. A child that crashes is a family that did not answer:

    [detect] family 0: no answer (exit 0xC0000005)
    [detect] hip6.2: 1 device(s), arch gfx1031
    hip6.2 gfx1031

The rule is unchanged; what changed is that a vendor runtime can no longer decide whether the
caller survives. A child that hangs is killed after 60 seconds, because a wedged runtime must not
wedge a node run.

One debugging note, since it cost two rounds: under `-v` stderr is unbuffered. The lines naming the
failing step were sitting in a block buffer when the process died, so the first two attempts to
localise the crash produced an exit code and no output at all.

## What is actually validated

Eleven targets, three run on a real card. That ratio is worth stating plainly rather than burying,
because the failure mode for a wrong code object is not a crash - it is wrong data with no error.

| tier | targets | basis |
|---|---|---|
| run on hardware here | gfx1012, gfx1031, gfx12-generic | point-cloud checksum matching a reference the same machine produced |
| same architecture as a validated chip, not run | gfx1010, gfx1030, gfx1032, gfx1034 | compiled, differs from a tested sibling only in chip configuration |
| architecturally different, not run | gfx1033, gfx1035, gfx1036 (RDNA2 APUs), gfx11-generic | no such hardware here |

The APUs are the row to be careful about. They are unified-memory parts, and the memory bridge -
the one genuinely novel piece of this project - was designed and measured against a discrete card
behind PCIe: 22x for texture-sampled images, 4.7x for streamed volumes, 1.0x once the images are
coarse-grained. None of those numbers mean anything when system RAM *is* the video memory. The
bridge should degrade to something harmless there, but "should" is exactly the word this project
has twice been wrong about (HIP dropping `surf2Dwrite` into fp16 arrays; GPU atomics into mapped
host memory silently wrong on Linux unless the memory is non-coherent).

They ship anyway, for two reasons. The probe matches on the exact `gcnArchName`, so an APU payload
can only ever be handed to an APU - there is no path by which an unvalidated target reaches
hardware that is tested. And those users have no Windows package at all today, so the alternative
is not safety, it is nothing. Every target outside the first row is listed in
`gpu/<family>/UNTESTED` and announced on every run that selects it.

Two of these were already shipping unflagged: v0.2.16 released gfx1030 and gfx1032 packages that
have never run on an RX 6800 or an RX 6600. Marking them is more honest than the status quo.

## Fleet results (v0.2.17)

Every row compares against a value the same host produced before, so nothing is shipped between
machines and a failure names one host. The point cloud is the tetrahedralisation **input**
checksum; the output differs between any two runs because geogram renumbers its cells.

| | card | host | payload chosen | point cloud | blur check |
|---|---|---|---|---|---|
| RDNA4 | RX 9070 | Ryzen 5600X, Windows | `rocm7.2/gfx12-generic` | `af3a3cce376e3f12` | 129 of 244,617,408 |
| RDNA2 | RX 6750 XT | FX-8120, Windows | `hip6.2/gfx1031` | `6e0eb6cb7e58208` | 53 of 124,975,872 |
| RDNA1 | RX 5500 XT | FX-8120, Windows | `hip6.2/gfx1012` | `6e0eb6cb7e58208` | 53 of 124,975,872 |
| RDNA2 | RX 6750 XT | i3-4330, Linux | (Linux bundle) | `2f0e9773fb427046` | 78 of 124,975,872 |
| none | Intel UHD 630 | i5-8500, Windows | - | clean exit 1 | - |

Two of those are cross-architecture rather than self-consistency checks. On the FX-8120 the RDNA1
and RDNA2 cards give the same checksum **through the same bundle**, and on the i3-4330 the RDNA2
card reproduces what the RDNA1 card produced for v0.2.16. Worst divergence anywhere is 4.77e-07 and
none of it changes a decision.

The last row matters more than it looks: a machine with no AMD card must fail cleanly rather than
crash or choose something arbitrary, and it is the only way to test that.

## Layout

AliceVision's own binaries cannot be shared between the families - clang 19 `/arch:AVX` and
clang 22 `/arch:AVX2` are different builds - but the vcpkg runtime and `share/` tree can, and they
are the larger half.

    cheshire-detect.exe      the card probe
    common/bin/, common/share/   bit-identical across every input (132 MB)
    fam/hip6.2/bin/, lib/    AliceVision's binaries, clang 19 /arch:AVX
    fam/rocm7.2/bin/, lib/   AliceVision's binaries, clang 22 /arch:AVX2
    gpu/hip6.2/              amdhip64_6 + amd_comgr_2 (no amd_comgr0602)
      gfx1010/ gfx1012/      5 DLLs   RX 5600/5700, RX 5500
      gfx1030/ gfx1031/ gfx1032/ gfx1034/
      gfx1033/ gfx1035/ gfx1036/      RDNA2 APUs
    gpu/rocm7.2/             amdhip64_7 + amd_comgr0702
      gfx11-generic/         5 DLLs   RDNA3, RDNA3.5, future gfx11xx
      gfx12-generic/         5 DLLs   RDNA4, future gfx12xx

Nothing is copied or materialised on the user's disk. The launcher sets

    PATH = gpu/<fam>/<target>;gpu/<fam>;fam/<fam>/bin;common/bin
    ALICEVISION_ROOT = common

and Windows resolves each DLL from `PATH`, with the GPU directory first so its five win over any
same-named file behind them.

## The overlay model, confirmed on hardware

The design rests on one assumption: that a base built for one chip, plus another chip's five GPU
DLLs, is a correct package. Tested on bench-pc's RX 6750 XT with a deliberately harsher case than
the bundle needs - the base taken from the **gfx1012 (RDNA1)** package, the five DLLs from the
**gfx1031 (RDNA2)** package, `amd_comgr0602.dll` deleted - against the pure gfx1031 package on the
same machine and inputs:

| | size | tetrahedralisation input checksum |
|---|---|---|
| pure gfx1031 package | 427 MB | `6e0eb6cb7e58208` |
| RDNA1 base + RDNA2 GPU DLLs, no dead comgr | 320 MB | `6e0eb6cb7e58208` |

Identical, and equal to the v0.2.16 fleet result for this card. The hybrid's
`CHESHIRE_GPU_BLUR_CHECK` reports 53 of 124,975,872 pixels, worst 4.77e-07 - also identical - which
confirms the overlaid DLLs are doing the work rather than quietly falling back.

Compare the tetrahedralisation **input** checksum, never the output: geogram renumbers its cells
from byte-identical input on every run.
