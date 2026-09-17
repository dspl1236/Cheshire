# GPU min-cut for Meshing

Meshing decides which tetrahedra are inside the object with an s-t min-cut over a graph of one
node per cell: 21.7 M nodes and 217 M directed edges on the 107-photo engine bay job. Upstream
solves it with boost's Boykov-Kolmogorov, single-threaded by nature: 109 s there, 36 s for the
41-view set on house-pc. This page is the GPU replacement and how it was checked.

## What Meshing needs from the cut

Only one bit per cell: `_cellIsFull[ci] = isTarget(ci)`, BK's white tree, the cells from which
the sink is still reachable in the residual graph once the flow is maximal. That set is the same
for every maximum flow (it is the sink side of the min-cut with the largest source side), so any
exact max-flow algorithm followed by one search from the sink yields the labelling upstream
computes. Nothing downstream uses the flow value.

## What it is

`hip/port/gpu_maxflow/`: a push-relabel preflow in Bo Hong's lock-free form. Every node holding
excess, in parallel, pushes to its lowest residual neighbour when it stands above it and otherwise
raises itself to one above that neighbour; heights only rise, residuals and excesses move through
float atomics. A global relabel, a breadth-first search from the sink over residual edges with
frontier queues, runs every 128 sweeps. Only the first phase of push-relabel is needed: when no
excess can move, the last search from the sink is the labelling; stranded excess never has to flow
back to the source. Sweeps and BFS levels are queued in batches with no host round trip inside a
batch (the BFS reads its frontier size from device memory, the sweeps accumulate a work flag),
because a host synchronisation on Windows costs as much as a whole sweep on a small graph.

Two departures from the textbook, both for float. Upstream pins the infinite cells to the source
with a capacity of 2^31, where float's ulp is 256 and any ordinary push would vanish from such a
node's excess; those cells are contracted into the source instead (their edges saturated once,
their heights held at the top), which is what an infinite supply means. And each sweep runs over a
device-built list of the nodes that currently hold excess.

`MaxFlow_CSR::compute` ([docs/11](11-meshing-cpu.md)) hands the laid-out CSR arrays to the device
directly, no boost graph at all. `CHESHIRE_GPU_MAXFLOW=0` keeps Boykov-Kolmogorov;
`CHESHIRE_MAXFLOW_CHECK=1` runs both in one process and compares every cell's label;
`CHESHIRE_MAXFLOW_DUMP=<file>` writes the graph and BK's result for `hip/tests/maxflow_test`,
the standalone harness the solver was developed against; `CHESHIRE_MAXFLOW_RELABEL_EVERY` and
`CHESHIRE_MAXFLOW_BFS_BATCH` are the two cadences.

## Measured (RX 9070, Windows)

| | Boykov-Kolmogorov | GPU |
|---|---|---|
| 6-view graph, 1.6 M nodes, 16 M edges: the cut | 4.6 s | 2.0 s |
| engine bay graph, 21.7 M nodes, 217 M edges: the cut | 109 s | 5.3 s |
| engine bay Meshing, end to end | 296.5 s (v0.2.9), 495.9 s (v0.2.6), 510 s (upstream) | 189 s (two runs: 190.3, 188.7) |

RX 5500 XT on Linux (house-pc, i3-4330, 8 GB VRAM) with the v0.2.10 bundle, 41 views, 11.4 M nodes and 114 M edges: the cut 38 s (BK) to 9.0 s, labellings identical (0 of 11,417,156 cells in the in-process check), Meshing 302.9 s (v0.2.9) to 270.3 s, 413.5 s at v0.2.7.

## Validation, and a note on the flow value

On both graphs the GPU labelling is identical to Boykov-Kolmogorov's: 0 of 1,600,050 and 0 of
21,748,827 cells labelled differently (the in-process check, and the harness against a dumped
run). The min-cut value computed in double from each labelling is identical too, 309,163,583 on
the engine bay.

The flow totals the two algorithms report are not that number: BK's is 18 % under it and the
GPU's 7 % under, on the same graph, for the same labelling. Both accumulate float flow at
magnitudes around 1e7 where pushes of a few units are below the ulp, in different orders. Upstream
never uses the value, and this repository's checks compare labellings and cut values, never flow
totals.

## What is left in Meshing

With the cut on the GPU, Meshing on the engine bay was 189 s; the two visibility passes
(30 s each, per-camera nearest-neighbour lookups of every depth-map pixel) went to the GPU next
([docs/13](13-gpu-visibilities.md)): 159 s.
