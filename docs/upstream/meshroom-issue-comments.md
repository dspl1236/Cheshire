# Comments on the two Meshroom threads

#3179 posted 2026-09-15: https://github.com/alicevision/Meshroom/issues/3179#issuecomment-5685964192
#595: not posted yet (post a day later).

Highest-intent venues there are: the people in these threads asked for exactly this. Post
#3179 first (it names HIP porting and is 26 days old with one comment), #595 a day later.
Keep both short; the README carries the detail. No hype words.

---

## Meshroom#3179 "Feature Request: ROCm/Radeon Open Compute Support for AMD GPUs"

https://github.com/alicevision/Meshroom/issues/3179

Not an official answer, but it exists and you can run it today: https://github.com/dspl1236/cheshire

It is AliceVision's DepthMap (the only CUDA-only stage) compiled as HIP through a compatibility header, with the CUDA sources untouched, packaged as a self-contained Windows zip (RDNA3/RDNA4, plus the RDNA3 APUs) and a relocatable Linux bundle (RDNA1 through RDNA4, APUs, Vega untested). Validated per architecture against a Meshroom 2023.3 CUDA run on a GTX 1080 Ti: identical validity masks, zero median depth error, 97-99 % of pixels within 1 %; RDNA1 and RDNA2 are bit-identical to each other. On 41 views an RX 9070 does the stage in 124 s against the 1080 Ti's 379 s; an RX 6750 XT paired into a real Meshroom 2023.3 job on the same node did it in 352 s.

Pairing with an existing Meshroom 2023.3 install is one script (`scripts/linux/meshroom-pair.sh`): it swaps the `aliceVision_depthMapEstimation` binary for a wrapper that runs the HIP build, or the original CUDA binary when `nvidia-smi` answers. Meshroom 2025.x is untested; the DepthMap node's command line changed little, and the wrapper already drops the one option upstream removed.

Re the SYCL backend mentioned above: it is the right long-term answer, and it has not shipped in a release yet, so this is the bridge until it does. If anyone here has an RX 7000 or a Ryzen AI laptop, the README has a ten-minute reproduce; nothing has run on RDNA3 hardware yet because I don't own any.

---

## Meshroom#595 "[FR]: Use OpenCL instead privative alternatives (CUDA, Metal)"

https://github.com/alicevision/Meshroom/issues/595

For the AMD half of this thread: a HIP build of the DepthMap stage now exists and is packaged for Windows and Linux, validated against a CUDA reference per architecture (RDNA1/2/4; identical masks, zero median error, 97-99 % of pixels within 1 %): https://github.com/dspl1236/cheshire. It pairs with a Meshroom 2023.3 install by replacing one binary, and falls back to the CUDA binary when an NVIDIA card is present.

Two things learned on the way that apply to any non-CUDA backend, including the SYCL one: HIP drops `surf2Dwrite` stores into 16-bit float arrays (the mip chain has to be built through a buffer copy), and GPU atomics into fine-grained mapped host memory are silently wrong on Linux unless the allocation is non-coherent (the SGM aggregation `atomicMin`s into a buffer; reproducer filed as ROCm/clr#285). Details in the repo's docs.
