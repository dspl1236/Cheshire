| case | build | knobs | DepthMap s | tiles | spills | volume VRAM/host MB | image VRAM/host MB | map VRAM/host MB | bit-identical | within 1 % of CUDA (median view) |
|---|---|---|---|---|---|---|---|---|---|---|
| default | emulated | `defaults` | 61.148 | 16 | 0 | 3992 / 0 | 186 / 0 | 596 / 0 | yes | n/a |
| img-host | emulated | `CHESHIRE_BRIDGE_HOST_CLASSES=image` | 109.839 | 16 | 48 | 3992 / 0 | 0 / 186 | 596 / 0 | yes | n/a |
| map-host | emulated | `CHESHIRE_BRIDGE_HOST_CLASSES=map` | 90.283 | 16 | 362 | 3992 / 0 | 186 / 0 | 0 / 596 | yes | n/a |
| vol-host | emulated | `CHESHIRE_BRIDGE_HOST_CLASSES=volume CHESHIRE_BRIDGE_HOST_MB=7000` | 347.172 | 16 | 51 | 0 / 3992 | 186 / 0 | 596 / 0 | yes | n/a |
| cap4000 | emulated | `CHESHIRE_BRIDGE_VRAM_MB=4000` | 60.106 | 8 | 0 | 1996 / 0 | 186 / 0 | 312 / 0 | yes | n/a |
| cap1500 | emulated | `CHESHIRE_BRIDGE_VRAM_MB=1500 CHESHIRE_BRIDGE_HOST_MB=7000` | 59.947 | 2 | 0 | 499 / 0 | 186 / 0 | 100 / 0 | yes | n/a |
| cap700 | emulated | `CHESHIRE_BRIDGE_VRAM_MB=700 CHESHIRE_BRIDGE_HOST_MB=7000` | 61.028 | 1 | 191 | 249 / 0 | 186 / 0 | 58 / 29 | yes | n/a |
| cap500 | emulated | `CHESHIRE_BRIDGE_VRAM_MB=500 CHESHIRE_BRIDGE_HOST_MB=7000` | 248.039 | 1 | 197 | 187 / 155 | 186 / 0 | 32 / 45 | yes | n/a |
