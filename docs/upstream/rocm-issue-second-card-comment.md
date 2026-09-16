**Reproduced on a second card (RDNA1), same platform**

Same host (i3-4330, in-kernel amdgpu, HIP 7.2 from the apt repo), the RX 6750 XT swapped for an RX 5500 XT (gfx1010), same reproducer built with `--offload-arch=gfx1010`:

```
device 0
VRAM                         atomicMin bad=0/65536  atomicAdd bad=0/65536  store->memset bad=0  memset->store bad=0
host mapped (default)        atomicMin bad=65536/65536  atomicAdd bad=0/65536  store->memset bad=0  memset->store bad=0
host mapped coherent         atomicMin bad=65536/65536  atomicAdd bad=0/65536  store->memset bad=0  memset->store bad=0
host mapped non-coherent     atomicMin bad=0/65536  atomicAdd bad=0/65536  store->memset bad=0  memset->store bad=0
RESULT: some placements FAIL
```

Identical pattern: every `atomicMin` into fine-grained mapped host memory wrong, `atomicAdd` right, everything right on non-coherent memory. So this is not specific to Navi 22 or to one card's firmware; it follows the memory type and the platform.

For scale, and because "wrong" needs a baseline: with the accumulator moved to non-coherent memory, the application's depth maps on this card are byte-identical to the RX 6750 XT's and agree with the CUDA reference on 98 % of pixels per view (worst view 92 %). That 2-8 % is the algorithm's build-to-build noise, not a port artefact: the CUDA reference itself, rerun on another GTX 1080 Ti with a CUDA 11.6 build instead of 11.3, is bit-identical on only 37 of 41 views and differs by 7 % of pixels on one of them. Against that baseline, the fine-grained `atomicMin` failure is unmistakable: no view survives, and nothing in the logs says why.
