// Cheshire: Meshing's s-t min-cut on the GPU.
//
// Upstream cuts the Delaunay volume with boost's Boykov-Kolmogorov on a graph of one node per
// tetrahedron (21.7 M nodes, 195 M directed edges on the 107-photo engine bay job, 109 s in one
// thread). What Meshing uses from the cut is one bit per cell, "full" = the cell can still reach
// the sink in the residual graph of a maximum flow (BK's white tree), and that set is the same for
// every maximum flow: it is the sink side of the min-cut with the largest source side. So the GPU
// runs a push-relabel preflow (Hong's lock-free formulation: one thread per active node, float
// atomics on residuals and excesses, periodic global relabelling by a breadth-first search from the
// sink) until no excess can move, then one more search from the sink marks the full cells. Only
// the first phase of push-relabel is needed; stranded excess never has to flow back to the source.
//
// The flow value comes out equal to BK's up to float summation order; the labelling can differ only
// on cells whose residual capacities are within rounding of zero. CUDA dialect; the HIP build
// force-includes cheshire/cuda_to_hip.h.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cheshire {
namespace maxflow {

// A directed graph in CSR form where every edge has a partner (its reverse) and the source and
// sink are ordinary node ids. rowstart has nbNodes + 1 entries.
struct Graph
{
    std::uint32_t nbNodes = 0;
    std::uint32_t nbEdges = 0;
    std::uint32_t source = 0;
    std::uint32_t sink = 0;
    const std::uint32_t* rowstart = nullptr;
    const std::uint32_t* target = nullptr;
    const float* capacity = nullptr;
    const std::uint32_t* partner = nullptr;
};

struct Stats
{
    float flow = 0;          // excess that reached the sink
    int pulses = 0;          // push/relabel sweeps
    int globalRelabels = 0;
    double seconds = 0;      // device time, uploads and downloads included
};

bool available();

// sinkSide[v] = 1 when the sink is reachable from v in the final residual graph (BK's "white").
// verbose > 0 prints progress to stderr.
bool minCut(const Graph& g, std::vector<std::uint8_t>& sinkSide, Stats& stats, int verbose = 0);

}  // namespace maxflow
}  // namespace cheshire
