# Spilling GPU buffers to system RAM on ROCm: two things the allocation flag decides

Written for anyone who puts device buffers in host memory on AMD GPUs: KV-cache offload in
local LLM inference, out-of-core solvers, anything that maps system RAM into a kernel's
address space. Nothing below is specific to photogrammetry, which is where it was found.
Every number is measured; the probes are in this repository and take a minute to run.

## The short version

`hipHostMalloc(..., hipHostMallocMapped)` gives you **fine-grained** (coherent) host memory by
default. Two consequences that do not announce themselves:

1. **GPU atomics into it can be silently wrong on Linux.** Only the operations PCIe can carry
   as AtomicOps (FetchAdd, Swap, CAS) are performed correctly. `atomicMin` into fine-grained
   host memory returned the wrong value on every element on an RX 6750 XT under ROCm 7.2,
   while `atomicAdd` on the same memory was correct. No error, no warning, wrong data.
2. **Texture-sampled reads from it cost 20x more than they need to.** With fine-grained memory
   every fetch crosses PCIe; the device is not allowed to cache it. Camera images behind PCIe
   made a kernel 22x slower (81x on a 5 GT/s link). The same images in coarse-grained host
   memory: 1.0x, because the device caches host memory like its own.

Both go away with one flag: `hipHostMallocMapped | hipHostMallocNonCoherent` (coarse-grained,
device-cached). The trade is visibility: the CPU sees coarse-grained memory only at
synchronisation points, which is what you already have if the CPU only touches the buffer
through `hipMemcpy` after a sync. Windows (PAL) behaved correctly and fast in all cases; this
is a Linux (KFD / ROCr) story.

## Atomics: the measurement

`hip/tests/host_atomics.hip` (about 100 lines). For each placement, on one stream: memset to
0xFF, 64 threads per slot `atomicMin` their candidate (the slot index is the minimum), read
back; memset to 0, 64 threads per slot `atomicAdd(1)`, read back; then two store/memset
ordering checks.

RX 6750 XT (gfx1031), Linux Mint 22.3, kernel 7.0, ROCm 7.2 runtime (`libamdhip64.so.7.2.70200`):

```
VRAM                         atomicMin bad=0/65536      atomicAdd bad=0/65536
host mapped (default)        atomicMin bad=65536/65536  atomicAdd bad=0/65536
host mapped coherent         atomicMin bad=65536/65536  atomicAdd bad=0/65536
host mapped non-coherent     atomicMin bad=0/65536      atomicAdd bad=0/65536
```

RX 5500 XT (gfx1012, RDNA1) in the same box, same runtime: identical result, `atomicMin`
wrong on 65536/65536 for both fine-grained placements, correct on non-coherent. Two GPU
generations, one platform, one behaviour: it is the runtime's handling of fine-grained
memory, not a card.

RX 9070 (gfx1201), Windows 11, ROCm 7.2.1: all four placements correct.

The platform is not the excuse. The Haswell root port advertises `AtomicOpsCap: 32bit+ 64bit+
128bitCAS+` and the GPU has `AtomicOpsCtl: ReqEn+`, and `atomicAdd` working is the proof that
PCIe atomics are carried. `atomicMin` has no PCIe AtomicOp; on fine-grained memory it is
silently dropped rather than emulated with CAS or refused. Reported with the reproducer:
https://github.com/ROCm/clr/issues/285 (dumps: `docs/upstream/house-pc-pcie.txt`,
`house-pc-rocminfo.txt`).

What it looked like from the application side, so you recognise it: an SGM aggregation step
`atomicMin`s into a per-row accumulator. With that accumulator spilled to fine-grained host
memory every depth map came out wrong (median depth error 17-29 %, extra "valid" pixels),
with nothing in any log pointing at memory. Volumes and images spilled to the same memory
were fine, because nothing does atomics on them. Two days went into copies and memsets
before the atomics probe took ten minutes to write and pointed at the real cause.

## Bandwidth: the measurement

AliceVision's depth-map stage, 6 views, one buffer class at a time forced into host memory,
everything else in VRAM, output bit-identical in every row (`scripts/bridge_matrix.py`;
full tables with per-class peaks in `docs/validation/bridge-v2/`).

| buffer class in host RAM | access pattern | fine-grained | coarse-grained |
|---|---|---|---|
| nothing (all in VRAM) | | 20.6 s | 20.2 s |
| depth/sim maps, 879 MB | kernels read and write, some atomics | 35.9 s (1.7x) | 25.5 s (1.3x) |
| similarity volumes, 6.0 GB | streamed, coalesced, rewritten per pass | 96.4 s (4.7x) | 76.6 s (3.8x) |
| camera images, 186 MB | random texture fetches by every kernel | **460 s (22x)** | **20.4 s (1.0x)** |

RX 9070, PCIe 4.0 x16, Windows. On the Linux node with a 5 GT/s x16 link (RX 6750 XT) the
fine-grained image case is 81x; coarse-grained brings it to 1.3x.

Reading it: per byte, streamed data is the cheapest thing to put behind PCIe (the link runs at
full width, 13 s/GB here) and texture-sampled data is the most expensive (1,500 s/GB) *as long
as the memory is fine-grained*. Coarse-grained memory lets the device cache it, the working
set of a texture lives in L2 / Infinity Cache, and the link is almost idle. If you spill
something kernels sample repeatedly, the allocation flag is worth more than any placement
heuristic.

## The rules that fell out

* Allocate spilled buffers `hipHostMallocMapped | hipHostMallocNonCoherent` unless the CPU
  has to observe kernel writes without a synchronisation point.
* Never rely on atomics other than add / exchange / compare-and-swap into fine-grained host
  memory on Linux. Min, max, and, or, xor: assume they are wrong until you have run the probe
  on your platform.
* Order copies and memsets that touch host-resident memory yourself if you mix them with
  kernels on a stream; the runtime may execute a host-side copy at the call. (This was not
  the cause of the bug above, but it is cheap insurance: `hipStreamSynchronize` before the copy.)
* Measure per buffer class before choosing what to spill. The intuitive order (spill the
  read-mostly stuff) was exactly backwards with the default flags.

## Reproduce

```
hipcc --offload-arch=<your gfx> -O2 hip/tests/host_atomics.hip -o host_atomics && ./host_atomics
hipcc --offload-arch=<your gfx> -O2 hip/tests/linear_tex.hip   -o linear_tex   && ./linear_tex
```

`linear_tex` times a 9x9-patch texture pass over the same data in VRAM, fine-grained host
memory and coarse-grained host memory. On a WSL-built binary run on a node with
`HSA_OVERRIDE_GFX_VERSION` set, unset it first (docs/06).

Context for the numbers: https://github.com/dspl1236/cheshire, `docs/02-memory-bridge.md`.
MPL-2.0.
