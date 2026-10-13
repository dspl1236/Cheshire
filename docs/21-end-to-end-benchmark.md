# End to end: stock Meshroom against Cheshire (2026-10-11/13)

Every other page measures one node, or one change, against the same node before it. This one measures the whole
default pipeline: Meshroom 2025.1's `photogrammetry` template with Meshroom's default settings, on five photo sets
from 6 to 884 views, on two machines, as somebody who downloads Meshroom and then Cheshire would run it.

## The arms

| arm | what runs |
|---|---|
| **stock** | Meshroom 2025.1 as it ships: its own binaries, nothing paired |
| **cpu** | the published 0.4.5 package paired, with every GPU port but DepthMap switched off (`CHESHIRE_GPU_MATCHER`, `_FILTER`, `_VOTE`, `_TEX`, `_BLUR`, `_MAXFLOW`, `_VIS`, `CHESHIRE_UNDISTORT_DEVICE` = 0): Cheshire's CPU-side work alone |
| **cheshire** | the published 0.4.5 package paired, defaults |

Stock runs until a node fails. Two failures are expected and filled in, each by Cheshire's binary for that node
alone, every other node staying Meshroom's own:

- **DepthMap without a CUDA device.** Meshroom's DepthMap is CUDA-only, so on an AMD card stock stops there. This is
  the first thing an AMD user meets, and it is filled in on every power-pc stock run.
- **StructureFromMotion on large sets** (Meshroom #2344, step 5k). Filled in with every other Cheshire SfM change
  switched back to upstream's behaviour, so it is stock plus the fix. It did not trigger in this round: stock's SfM
  finished on all sets, the 884-view False Door included.

`run.json` in each run records where stock stopped and what filled in. DSP-SIFT, the default describer, is a CPU
describer, so FeatureExtraction is CPU work in every arm.

## Machines and sets

| | power-pc | house-pc |
|---|---|---|
| CPU | Ryzen 5 5600X (6C/12T) | Core i3-4330 (2C/4T, Haswell) |
| GPU | Radeon RX 9070 | GeForce GTX 1080 Ti |
| RAM | 48 GB | 15 GB |
| OS | Windows 11 | Linux Mint 22.3 |
| package | 0.4.5 Windows | 0.4.5 Linux CUDA |

| set | views |
|---|---|
| mini6 | 6 |
| monstree41 | 41 |
| enginebay | 107 |
| zoo | 524 |
| falsedoor | 884 |

Times are wall clock for the whole `meshroom_batch` run (stock's resumes included). The per-node tables sum each
node's chunk times from Meshroom's own status files. **views placed** is the cameras StructureFromMotion posed, out
of the views in the set; a run that places fewer views has less dense work to do, so its time is only comparable next
to that number.

## Results

### power-pc (Ryzen 5600X + RX 9070)

| set | stock* | cpu | cheshire | stock / cheshire | views placed (stock / cpu / cheshire) |
|---|---|---|---|---|---|
| mini6 | 113 s | 75 s | 56 s | 2.0x | 6 / 6 / 6 |
| monstree41 | 1,062 s | 592 s | 374 s | 2.8x | 41 / 41 / 41 |
| enginebay | 2,409 s | 1,338 s | 575 s | 4.2x | 107 / 107 / 107 |
| zoo | 11,041 s | 7,474 s | 4,036 s | 2.7x | 464 / 487 / 490 of 524 |
| falsedoor | 36,381 s | - | 9,641 s | 3.8x | 853 / - / 854 of 884 |

\* with Cheshire's DepthMap filling in (no CUDA). The False Door ran stock and cheshire only. mini6 was run again on
an idle box on 2026-10-13 (the first round had a desktop app open): 113 / 75 / 56 s against 106 / 72 / 55 s, the same
within a few seconds.

### house-pc (i3-4330 + GTX 1080 Ti)

| set | stock | cpu | cheshire | stock / cheshire | views placed (stock / cpu / cheshire) |
|---|---|---|---|---|---|
| mini6 | 279 s | 218 s | 148 s | 1.9x | 6 / 6 / 6 |
| monstree41 | 3,185 s | 2,089 s | 1,065 s | 3.0x | 41 / 41 / 41 |
| enginebay | 11,104 s | 5,301 s | 2,230 s | 5.0x | 107 / 107 / 107 |
| zoo | 41,580 s | 71,246 s | 23,179 s | not like for like | **152** / 490 / 486 of 524 |
| falsedoor | running | | | | |

Stock is Meshroom's own pipeline end to end here, its CUDA DepthMap included.

## Where the time goes

The False Door on power-pc, per node:

| node | stock | cheshire |
|---|---|---|
| FeatureExtraction | 3,004 s | 3,169 s |
| FeatureMatching | 7,351 s | 727 s |
| StructureFromMotion | 3,368 s | 701 s |
| PrepareDenseScene | 847 s | 469 s |
| DepthMap | 1,190 s (Cheshire's) | 1,067 s |
| DepthMapFilter | 2,255 s | 888 s |
| Meshing | 2,399 s | 646 s |
| Texturing | 15,843 s | 1,862 s |

- **FeatureMatching** is the largest single saving everywhere: 10x on the False Door, 7x on the zoo (power-pc),
  8x on the engine bay. The cpu arm alone halves it (Cheshire's CPU-side matcher work), the GPU matcher does the rest.
- **FeatureExtraction is unchanged**: DSP-SIFT is CPU work in all three arms. It is the largest node Cheshire leaves
  alone on default settings, and on the i3 it is about 2.6 h of the zoo in every arm.
- **StructureFromMotion**: 3-5x from the cpu arm already (Cheshire's own BA solver and the AC-RANSAC work); the GPU
  adds nothing here, as expected.
- **DepthMapFilter, Meshing, Texturing**: the cpu arm takes 25-45 % off each, the GPU ports most of the rest.

## Three things the numbers need next to them

**Stock's StructureFromMotion varies from run to run, and on house-pc it placed 152 of the zoo's 524 views.** It
stalled with 349 images left to resect; every later node then skipped the views it never placed. That run's dense
nodes did about a third of the work the cpu and cheshire arms did, so its 41,580 s is not a like-for-like baseline,
and the cpu arm's 71,246 s for the whole zoo is not a slowdown. On power-pc stock placed 464. The table keeps the
measured times with the views next to them.

**Texturing is bounded by memory, not by the GPU, on small machines.** Meshroom textures as many atlases at once as
it thinks fit in free memory (`memoryPerAtlas` ~3.9 GB) and re-reads every contributing image for each pass. house-pc
(15 GB) fits one atlas per pass; the zoo's 45 atlases over 490 images took the cpu arm 45,056 s, about 2 s per image
per pass, the same per-image cost as stock's 14 atlases over 152 images. power-pc (48 GB) fits about 11 and did the
same node in 1,990 s. Cheshire's GPU pyramid and rasterisation brought house-pc's zoo to 7,513 s.

**power-pc's False Door stock Texturing (15,843 s) ran with 33 GB free against ~48 GB on the zoo runs**, so it worked
in smaller batches (54 atlases). Part of its 8.5x against Cheshire's Texturing is that.

## Reproducing

The scripts are in `scripts/bench/`, as run:

- house-pc: `run-all.sh` runs `bench.sh <set> <photos> all` per set into `/data/tests/bench/runs`; `prune.sh` deletes
  the large intermediates of every finished run (depth and similarity maps, images, features, matches, Alembic, textures)
  and keeps what `summarize.py` reads.
- power-pc: `run-all-win.cmd` runs `bench_win.py` per set into `C:\cheshire-bench\runs-win`; each finished run is
  pruned by `prune_win.py` (the False Door's cache alone is ~70 GB).
- `summarize.py <runs dir>` prints the per-node tables, wall times, views placed and mesh sizes.

The paths in the scripts are the two machines' own (a scratch Meshroom 2025.1 install, the unpacked 0.4.5 packages).
