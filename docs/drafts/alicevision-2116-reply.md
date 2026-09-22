# Draft: reply to alicevision discussion #2116 ("Proposal: Rework the build system from scratch")

*Not posted. Review before posting. This is a reply in the build-system thread, so it stays on
build, bundling and portability; the metrics and large-set results go in a separate show-and-tell.*

---

A data point for the portability and bundling parts of this, from the outside.

For the last few months I have been maintaining **Cheshire** (MPL-2.0,
https://github.com/dspl1236/Cheshire): AliceVision's CUDA stages built for AMD through HIP, plus
the CPU stages that were never on a GPU moved there, packaged so that a Meshroom 2023.3 install
runs them by swapping seven binaries. It is a downstream tree, not a fork of the build system, so
most of what follows is what the current CMake made easy or hard.

**HIP instead of SYCL, and what it cost.** DepthMap and PopSift port to HIP through a
compatibility header (`hip/compat/include/cheshire/cuda_to_hip.h`) plus a handful of runtime
fixes for HIP on Windows (half-float texture arrays, mip levels, surface writes). The CUDA build of
the same tree is byte-identical to the CUDA 11.3 reference on a GTX 1080 Ti (41/41 depth maps), and
the HIP build is bit-identical to it on RDNA1 and RDNA2 and within the RDNA4-vs-RDNA2 noise floor
on RDNA4. The stages that upstream runs on the CPU on every vendor (matcher, depth-map filter,
meshing votes, max-flow, visibility, texturing) are written once in CUDA dialect and build for both;
every one carries an in-process self-check against upstream's CPU path that reports differences
(0 on an 884-view set, docs/04). No SYCL and no LLVM build needed on the user's side: the HIP SDK
on Windows and ROCm on Linux are the only prerequisites, and only for building.

**Bundling on Windows.** One package for every AMD card is 181 MB, because only five DLLs carry
GPU code (about 4.7 MB per architecture) and the vcpkg runtime plus `share/` are bit-identical
across toolchains, so the package holds eleven small payloads and a probe picks one at run time
(docs/16). The probe runs each HIP runtime family in its own process: loading a runtime the
installed driver does not support can exit 0xC0000005 rather than report "no device", and only in a
non-interactive session, which is how Meshroom starts a node. Relevant to a bundle target: the
`bundle` target's two search-path defects and the `share/` layout are the only places I had to
patch CMake, and both are in the tree's generator script rather than in a fork.

**SPQR.** Same conclusion as this thread, reached by checking after reading it: the Windows
packages built on vcpkg's Ceres carried `libspqr.dll` and `libcholmod.dll`; the Linux bundle's
Ceres was built without SuiteSparse and did not. Removing them from the Windows build and
measuring the bundle-adjustment cost of Eigen's sparse backend instead is on the list for the
next release, and if the numbers are interesting I will post them.

**One upstream bug found on the way**, in case it helps prioritise: incremental SfM's resection
loop can exit with views pending their bundle adjustment, which later throws in the local-BA
graph (`invalid map<K, T> key`, issue #2344). Reproduced three of three on an 884-view public set,
fixed with existing code in a different order; details on the issue.

Happy to answer questions on any of it, and if a HIP or bundling piece would be useful upstream
rather than downstream, say which.
