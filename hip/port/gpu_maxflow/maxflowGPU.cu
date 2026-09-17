// Cheshire: Meshing's s-t min-cut on the GPU (see maxflowGPU.hpp).
//
// Push-relabel preflow in Bo Hong's lock-free form ("A lock-free multi-threaded algorithm for the
// maximum flow problem", 2008): every active node, in parallel, pushes to its lowest residual
// neighbour when it stands above it, otherwise relabels itself to one above that neighbour; heights
// only rise, residuals and excesses move by float atomics. A global relabel (breadth-first search
// from the sink over residual edges, frontier queues) every batch of sweeps keeps the heights
// honest. Phase one only: when no node with excess can move, the nodes that still reach the sink
// are the sink side of the cut.
//
// Two departures from the textbook, both for float:
//  * upstream pins the infinite cells to the source with a 2^31 capacity, where float's ulp is 256
//    and every ordinary push would vanish from such a node's excess; those nodes are contracted
//    into the source instead (their edges saturated once, their heights held at the top), which is
//    what an infinite supply means and keeps every number in the graph's own range;
//  * a sweep runs over a list of the nodes that currently hold excess, rebuilt on the device each
//    sweep, so a 21.7 M-node graph with a few hundred thousand active nodes costs what they cost.
// Kernels are launched in batches with no host round trip inside a batch (the BFS reads its
// frontier size from device memory, the sweeps accumulate a work flag), because on Windows a host
// synchronisation costs as much as a whole sweep on a small graph.
#include "maxflowGPU.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace cheshire {
namespace maxflow {

namespace {

constexpr int kBlock = 256;
constexpr std::uint32_t NONE = 0xffffffffu;
constexpr unsigned kGrid = 4096;             // fixed grid for the list kernels, grid-stride inside
constexpr float kPinCapacity = 1073741824.0f; // 2^30: a source edge at or above this is a pin

template<class T> bool up(T** d, const T* h, size_t n) {
    if (cudaMalloc((void**)d, n * sizeof(T)) != cudaSuccess) return false;
    return cudaMemcpy(*d, h, n * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}

// saturate every edge out of the source; a pinned target is contracted into the source: flagged,
// and every one of its own edges saturated as well
__global__ void initSourceKernel(std::uint32_t s, const std::uint32_t* __restrict__ rowstart, const std::uint32_t* __restrict__ target,
                                 const std::uint32_t* __restrict__ partner, float* __restrict__ residual, float* __restrict__ excess,
                                 std::uint8_t* __restrict__ pinned)
{
    const std::uint32_t i = rowstart[s] + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowstart[s + 1]) return;
    const float c = residual[i];
    if (c <= 0.0f) return;
    const std::uint32_t p = target[i];
    if (c >= kPinCapacity) {
        pinned[p] = 1;
        residual[i] = 0.0f;
        for (std::uint32_t f = rowstart[p]; f < rowstart[p + 1]; ++f) {
            const std::uint32_t w = target[f];
            if (w == s) continue;
            const float d = residual[f];
            if (d <= 0.0f) continue;
            residual[f] = 0.0f;
            atomicAdd(&residual[partner[f]], d);
            atomicAdd(&excess[w], d);
        }
        return;
    }
    residual[i] = 0.0f;
    atomicAdd(&residual[partner[i]], c);
    atomicAdd(&excess[p], c);
}

__global__ void fillKernel(std::uint32_t* __restrict__ a, std::uint32_t n, std::uint32_t v)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = v;
}

__global__ void setKernel(std::uint32_t* __restrict__ a, std::uint32_t v) { *a = v; }

// one BFS level from the sink: for every frontier node u, every neighbour w with residual(w->u) > 0
// that is still unlabelled gets level + 1 and joins the next frontier (pinned nodes never do)
__global__ void bfsLevelKernel(const std::uint32_t* __restrict__ frontier, const std::uint32_t* __restrict__ nbFrontier, std::uint32_t level,
                               const std::uint32_t* __restrict__ rowstart, const std::uint32_t* __restrict__ target,
                               const std::uint32_t* __restrict__ partner, const float* __restrict__ residual,
                               const std::uint8_t* __restrict__ pinned, std::uint32_t* __restrict__ height, std::uint32_t unlabelled,
                               std::uint32_t* __restrict__ next, std::uint32_t* __restrict__ nextCount)
{
    const std::uint32_t n = *nbFrontier;
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const std::uint32_t u = frontier[i];
        const std::uint32_t e0 = rowstart[u], e1 = rowstart[u + 1];
        for (std::uint32_t e = e0; e < e1; ++e) {
            if (residual[partner[e]] <= 0.0f) continue;   // w -> u must have capacity left
            const std::uint32_t w = target[e];
            if (pinned[w]) continue;
            if (atomicCAS(&height[w], unlabelled, level + 1) == unlabelled) {
                const std::uint32_t slot = atomicAdd(nextCount, 1u);
                next[slot] = w;
            }
        }
    }
}

// the nodes that can still act: excess, below the top, not the terminals
__global__ void collectActiveKernel(std::uint32_t nbNodes, std::uint32_t s, std::uint32_t t, const float* __restrict__ excess,
                                    const std::uint32_t* __restrict__ height, const std::uint8_t* __restrict__ pinned,
                                    std::uint32_t* __restrict__ list, std::uint32_t* __restrict__ count)
{
    const std::uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= nbNodes || u == s || u == t) return;
    if (excess[u] > 0.0f && height[u] < nbNodes && !pinned[u]) {
        const std::uint32_t slot = atomicAdd(count, 1u);
        list[slot] = u;
    }
}

// Hong's push-or-relabel step for every listed node
__global__ void pushRelabelKernel(const std::uint32_t* __restrict__ list, const std::uint32_t* __restrict__ nbList, std::uint32_t nbNodes,
                                  const std::uint32_t* __restrict__ rowstart, const std::uint32_t* __restrict__ target,
                                  const std::uint32_t* __restrict__ partner, float* __restrict__ residual,
                                  float* __restrict__ excess, std::uint32_t* __restrict__ height, unsigned* __restrict__ work)
{
    const std::uint32_t n = *nbList;
    __shared__ unsigned blockWork;
    if (threadIdx.x == 0) blockWork = 0;
    __syncthreads();
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const std::uint32_t u = list[i];
        const float ex = excess[u];
        const std::uint32_t hu = height[u];
        if (!(ex > 0.0f) || hu >= nbNodes) continue;
        std::uint32_t hmin = UINT_MAX, emin = NONE;
        const std::uint32_t e0 = rowstart[u], e1 = rowstart[u + 1];
        for (std::uint32_t e = e0; e < e1; ++e) {
            if (residual[e] <= 0.0f) continue;
            const std::uint32_t hv = height[target[e]];
            if (hv < hmin) { hmin = hv; emin = e; }
        }
        if (emin == NONE || hmin >= nbNodes) {
            height[u] = nbNodes;   // nothing left to push to: stranded, phase one is done for u
        } else if (hu > hmin) {
            const float r = residual[emin];
            const float d = ex < r ? ex : r;
            if (d > 0.0f) {
                atomicAdd(&residual[emin], -d);
                atomicAdd(&residual[partner[emin]], d);
                atomicAdd(&excess[target[emin]], d);
                atomicAdd(&excess[u], -d);
            }
        } else {
            height[u] = hmin + 1;
        }
        blockWork = 1;
    }
    __syncthreads();
    if (threadIdx.x == 0 && blockWork) atomicAdd(work, 1u);
}

bool g_checked = false, g_available = false;
std::mutex g_mutex;

struct Device {
    std::uint32_t *rowstart = nullptr, *target = nullptr, *partner = nullptr, *height = nullptr, *listA = nullptr, *listB = nullptr, *countA = nullptr, *countB = nullptr;
    float *residual = nullptr, *excess = nullptr;
    std::uint8_t* pinned = nullptr;
    unsigned* work = nullptr;
    ~Device() { for (void* p : {(void*)rowstart, (void*)target, (void*)partner, (void*)height, (void*)listA, (void*)listB, (void*)countA, (void*)countB, (void*)residual, (void*)excess, (void*)pinned, (void*)work}) if (p) cudaFree(p); }
};

// heights = BFS distance to the sink over residual edges, nbNodes where the sink is unreachable;
// levels are queued levelsPerSync at a time before the host looks at the frontier size
bool globalRelabel(const Graph& g, Device& d, int& levels, int levelsPerSync)
{
    const std::uint32_t V = g.nbNodes;
    fillKernel<<<(V + kBlock - 1) / kBlock, kBlock>>>(d.height, V, V);
    const std::uint32_t zero = 0;
    if (cudaMemcpy(d.height + g.sink, &zero, 4, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (cudaMemcpy(d.listA, &g.sink, 4, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    setKernel<<<1, 1>>>(d.countA, 1u);
    std::uint32_t* cur = d.listA; std::uint32_t* nxt = d.listB;
    std::uint32_t* curCount = d.countA; std::uint32_t* nxtCount = d.countB;
    levels = 0;
    for (;;) {
        for (int k = 0; k < levelsPerSync; ++k) {
            setKernel<<<1, 1>>>(nxtCount, 0u);
            bfsLevelKernel<<<kGrid, kBlock>>>(cur, curCount, std::uint32_t(levels), d.rowstart, d.target, d.partner, d.residual, d.pinned, d.height, V, nxt, nxtCount);
            std::uint32_t* tmp = cur; cur = nxt; nxt = tmp;
            tmp = curCount; curCount = nxtCount; nxtCount = tmp;
            ++levels;
        }
        std::uint32_t n = 0;
        if (cudaMemcpy(&n, curCount, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        if (n == 0) break;
    }
    // the source never pushes
    if (cudaMemcpy(d.height + g.source, &V, 4, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    return cudaGetLastError() == cudaSuccess;
}

int envInt(const char* name, int def) { const char* e = std::getenv(name); return e ? std::atoi(e) : def; }

}  // namespace

bool available()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_checked) return g_available;
    g_checked = true;
    if (const char* e = std::getenv("CHESHIRE_GPU_MAXFLOW")) if (e[0] == '0') return g_available = false;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) return g_available = false;
    return g_available = true;
}

bool minCut(const Graph& g, std::vector<std::uint8_t>& sinkSide, Stats& stats, int verbose)
{
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint32_t V = g.nbNodes, E = g.nbEdges;
    Device d;
    if (!up(&d.rowstart, g.rowstart, size_t(V) + 1) || !up(&d.target, g.target, E) || !up(&d.partner, g.partner, E) || !up(&d.residual, g.capacity, E)) return false;
    if (cudaMalloc((void**)&d.excess, size_t(V) * 4) != cudaSuccess || cudaMemset(d.excess, 0, size_t(V) * 4) != cudaSuccess) return false;
    if (cudaMalloc((void**)&d.height, size_t(V) * 4) != cudaSuccess) return false;
    if (cudaMalloc((void**)&d.pinned, size_t(V)) != cudaSuccess || cudaMemset(d.pinned, 0, size_t(V)) != cudaSuccess) return false;
    if (cudaMalloc((void**)&d.listA, size_t(V) * 4) != cudaSuccess || cudaMalloc((void**)&d.listB, size_t(V) * 4) != cudaSuccess) return false;
    if (cudaMalloc((void**)&d.countA, 4) != cudaSuccess || cudaMalloc((void**)&d.countB, 4) != cudaSuccess || cudaMalloc((void**)&d.work, 4) != cudaSuccess) return false;

    const std::uint32_t nbSourceEdges = g.rowstart[g.source + 1] - g.rowstart[g.source];
    if (nbSourceEdges) initSourceKernel<<<(nbSourceEdges + kBlock - 1) / kBlock, kBlock>>>(g.source, d.rowstart, d.target, d.partner, d.residual, d.excess, d.pinned);
    if (cudaDeviceSynchronize() != cudaSuccess) return false;

    const int relabelEvery = envInt("CHESHIRE_MAXFLOW_RELABEL_EVERY", 1024);   // sweeps between global relabels, queued without host syncs (engine bay: 256 -> 47 s, 1024 -> 35 s)
    const int levelsPerSync = envInt("CHESHIRE_MAXFLOW_BFS_BATCH", 32);
    int levels = 0;
    if (!globalRelabel(g, d, levels, levelsPerSync)) return false;
    stats.globalRelabels = 1;
    if (verbose) std::fprintf(stderr, "[maxflow] %u nodes, %u edges, %u source edges; first BFS: %d levels\n", V, E, nbSourceEdges, levels);

    stats.pulses = 0;
    const unsigned nodeGrid = (V + kBlock - 1) / kBlock;
    auto sweep = [&](int count) {
        for (int k = 0; k < count; ++k) {
            setKernel<<<1, 1>>>(d.countA, 0u);
            collectActiveKernel<<<nodeGrid, kBlock>>>(V, g.source, g.sink, d.excess, d.height, d.pinned, d.listA, d.countA);
            pushRelabelKernel<<<kGrid, kBlock>>>(d.listA, d.countA, V, d.rowstart, d.target, d.partner, d.residual, d.excess, d.height, d.work);
        }
        stats.pulses += count;
    };
    for (;;) {
        if (cudaMemset(d.work, 0, 4) != cudaSuccess) return false;
        sweep(relabelEvery);
        unsigned working = 0;
        if (cudaMemcpy(&working, d.work, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        if (!globalRelabel(g, d, levels, levelsPerSync)) return false;
        ++stats.globalRelabels;
        if (working == 0) {
            // nothing moved in a whole batch: one sweep with fresh heights decides
            if (cudaMemset(d.work, 0, 4) != cudaSuccess) return false;
            sweep(1);
            if (cudaMemcpy(&working, d.work, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
            if (working == 0) break;
        }
        if (verbose > 1) {
            float flow = 0; std::uint32_t active = 0;
            if (cudaMemcpy(&flow, d.excess + g.sink, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
            if (cudaMemcpy(&active, d.countA, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
            std::fprintf(stderr, "[maxflow] pulses %d, global relabels %d (%d levels), active %u, flow so far %.6g, %.1f s\n", stats.pulses, stats.globalRelabels, levels, active, flow,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    // the last global relabel is the final labelling: what still reaches the sink
    std::vector<std::uint32_t> h(V);
    if (cudaMemcpy(h.data(), d.height, size_t(V) * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    sinkSide.assign(V, 0);
    for (std::uint32_t i = 0; i < V; ++i) sinkSide[i] = h[i] < V ? 1 : 0;
    if (cudaMemcpy(&stats.flow, d.excess + g.sink, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (verbose) std::fprintf(stderr, "[maxflow] done: flow %.7g, %d pulses, %d global relabels, %.2f s\n", stats.flow, stats.pulses, stats.globalRelabels, stats.seconds);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace maxflow
}  // namespace cheshire
