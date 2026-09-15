# Draft: r/LocalLLaMA (not posted)

Flair: Discussion (or Resources). Text post; no link in the title. Paste the body as is.
Post the page link in the first line, not only at the end, so people who skim get it.

**Title:** PSA for anyone spilling GPU buffers to system RAM on ROCm: the default hipHostMalloc flag silently breaks some atomics on Linux and makes texture/random reads 20x slower than they need to be

---

Write-up with the numbers and two 100-line probes: https://github.com/dspl1236/cheshire/blob/main/docs/notes/gpu-host-memory-on-rocm.md

I hit this while porting a CUDA photogrammetry stage to HIP and giving it a VRAM-to-system-RAM spill tier, but nothing about it is photogrammetry-specific. If you offload KV cache or weights to host memory on an AMD card under Linux, two things about `hipHostMalloc(..., hipHostMallocMapped)` (which gives you fine-grained, coherent memory by default) may be costing you:

**1. Atomics into that memory can be silently wrong.** On an RX 6750 XT, ROCm 7.2, `atomicMin` into default mapped host memory returned the wrong value on 65536 of 65536 elements. `atomicAdd` on the same memory was correct. Both are correct on `hipHostMallocNonCoherent` memory, and Windows passes everything. The platform does carry PCIe atomics (the root port advertises them and Add works); Min just isn't a PCIe AtomicOp, and instead of emulating it with CAS or refusing, the runtime lets the kernel run and produces garbage with no error anywhere. Filed with a reproducer: https://github.com/ROCm/clr/issues/285

**2. Random / texture-sampled reads from fine-grained host memory are ~20x slower than from coarse-grained.** Same data, same kernel, 186 MB of texture-sampled buffers behind PCIe 4.0: 22x slower fine-grained, 1.0x coarse-grained (81x vs 1.3x on a 5 GT/s link). Fine-grained means the device may not cache it, so every fetch crosses the link; coarse-grained lets it live in L2 / Infinity Cache. Streamed buffers care much less (4.7x vs 3.8x for 6 GB of sequentially swept data).

The fix for both is one flag: `hipHostMallocMapped | hipHostMallocNonCoherent`, with the usual coarse-grained caveat that the CPU only sees kernel writes at synchronisation points. If your host-side code only reads the buffer through `hipMemcpy` after a sync, that is already true for you.

Happy to be told this is documented somewhere I missed; I could not find the atomics behaviour written down, and it cost two days of chasing copies and memsets before a ten-minute probe pointed at the real thing.
