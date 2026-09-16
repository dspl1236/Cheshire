# Validation: monstree-full (41 views) on house-pc, RX 5500 XT, Linux, v0.2.3 bundle (2026-09-16)

The v0.2.3 Linux bundle (packed-slot mipmap sampler, 19 code objects, ROCm 7.2 user space) on the
same RX 5500 XT / i3-4330 node as the [v0.2.1 result](../monstree-full-rx5500xt-linux/index.md).

| | v0.2.1 bundle | v0.2.3 bundle |
|---|---|---|
| 6 views | 60.7 s | **52.4 s** |
| 41 views | 447.7 s | **396.9 s** |

Outputs: the 6 depth maps are byte-identical to the v0.2.1 run on this card; the 41 depth maps are
byte-identical to the first Linux 41-view run of 2026-09-03 (`/data/scans/monstree-full/out/cheshire-hip`
on house-pc, RX 6750 XT at the time), so the sampler change alters nothing but time, and RDNA1 and
RDNA2 still agree bit for bit.

[vs the CUDA reference](vs-cuda.md): masks agree on 41 / 41 views, per-view median error
0.0000, within 1 %: median over views 0.9806, worst 0.9233, largest p95
0.06221: the same figures as every earlier Linux run of this set.

Both runs planned 24 simultaneous tiles with no spills (`CHESHIRE_BRIDGE_LOG=1`). The active bundle on
house-pc is now `bundle -> bundle-v0.2.3` (the previous one is kept as `bundle-v0.2.1`).
