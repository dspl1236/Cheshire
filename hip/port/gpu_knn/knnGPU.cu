// Cheshire: nanoflann's nearest-neighbour search, replayed on the GPU. See knnGPU.hpp.
//
// Each thread answers one query with an explicit stack in place of searchLevel's recursion. A
// frame is pushed when an inner node is entered (the child on the query's side is walked first);
// when the walk returns to the frame it does what searchLevel does after its first recursive call:
//   dst = dists[idx]; mindist += cut_dist - dst; dists[idx] = cut_dist;
//   if (mindist <= worst) walk the other child; dists[idx] = dst;
// The metric is L2_Simple_Adaptor's: diff = q - p per axis, result += diff * diff, in double, in
// axis order; with fma set, the fused form clang-cl emits for that statement under /arch:AVX2.
// accum_dist and the initial bounding-box distances are unfused products in both builds.
#include "knnGPU.hpp"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace cheshire {
namespace knn {

namespace {

constexpr int kMaxDepth = 96;

struct Frame
{
    int other;
    int idx;
    double cut;
    double mind;
    double dst;
    int phase;
};

template<bool FMA>
__device__ __forceinline__ double metric(const double* __restrict__ pts, std::uint32_t id, double q0, double q1, double q2)
{
    const double d0 = __dsub_rn(q0, pts[3 * (std::size_t)id + 0]);
    const double d1 = __dsub_rn(q1, pts[3 * (std::size_t)id + 1]);
    const double d2 = __dsub_rn(q2, pts[3 * (std::size_t)id + 2]);
    double r = 0.0;
    if (FMA)
    {
        r = __fma_rn(d0, d0, r);
        r = __fma_rn(d1, d1, r);
        r = __fma_rn(d2, d2, r);
    }
    else
    {
        r = __dadd_rn(r, __dmul_rn(d0, d0));
        r = __dadd_rn(r, __dmul_rn(d1, d1));
        r = __dadd_rn(r, __dmul_rn(d2, d2));
    }
    return r;
}

__device__ __forceinline__ double sqdiff(double a, double b)
{
    const double d = __dsub_rn(a, b);
    return __dmul_rn(d, d);
}

template<bool FMA>
__global__ void knnKernel(const Node* __restrict__ nodes,
                          const double* __restrict__ pts,
                          const std::uint32_t* __restrict__ perm,
                          const double* __restrict__ queries,
                          std::uint32_t nbQueries,
                          double lo0, double lo1, double lo2,
                          double hi0, double hi1, double hi2,
                          std::uint32_t* __restrict__ outIndex,
                          double* __restrict__ outDist2,
                          unsigned int* __restrict__ overflow)
{
    const std::uint32_t qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nbQueries) return;

    double vec[3] = {queries[3 * (std::size_t)qi], queries[3 * (std::size_t)qi + 1], queries[3 * (std::size_t)qi + 2]};
    const double lo[3] = {lo0, lo1, lo2};
    const double hi[3] = {hi0, hi1, hi2};

    // computeInitialDistances
    double dists[3] = {0.0, 0.0, 0.0};
    double mind = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        if (vec[i] < lo[i])
        {
            dists[i] = sqdiff(vec[i], lo[i]);
            mind = __dadd_rn(mind, dists[i]);
        }
        if (vec[i] > hi[i])
        {
            dists[i] = sqdiff(vec[i], hi[i]);
            mind = __dadd_rn(mind, dists[i]);
        }
    }

    // KNNResultSet<double, size_t>(1): worstDist() is max until the first point is added
    double best = DBL_MAX;
    std::uint32_t bestId = 0xFFFFFFFFu;

    Frame st[kMaxDepth];
    int sp = 0;
    int node = 0;
    bool failed = false;

    while (true)
    {
        const Node n = nodes[node];
        if (n.child1 < 0)
        {
            // leaf: `if (dist < worstDist()) addPoint(dist, id)`; addPoint replaces only on strict <
            for (std::uint32_t i = n.a; i < n.b; ++i)
            {
                const std::uint32_t id = perm[i];
                const double d = metric<FMA>(pts, id, vec[0], vec[1], vec[2]);
                if (d < best)
                {
                    best = d;
                    bestId = id;
                }
            }
            // return: unwind frames until one wants its other child walked
            bool descend = false;
            while (sp > 0)
            {
                Frame& f = st[sp - 1];
                if (f.phase == 0)
                {
                    f.phase = 1;
                    const double dst = dists[f.idx];
                    f.dst = dst;
                    const double m = __dsub_rn(__dadd_rn(f.mind, f.cut), dst);
                    // nanoflann: mindist = mindist + cut_dist - dst  (left to right)
                    dists[f.idx] = f.cut;
                    if (m <= best)  // epsError == 1
                    {
                        node = f.other;
                        mind = m;
                        descend = true;
                        break;
                    }
                    dists[f.idx] = dst;
                    --sp;
                }
                else
                {
                    dists[f.idx] = f.dst;
                    --sp;
                }
            }
            if (!descend) break;
            continue;
        }

        // inner node: which child first
        const int idx = (int)n.a;
        const double val = vec[idx];
        const double diff1 = __dsub_rn(val, n.lo);
        const double diff2 = __dsub_rn(val, n.hi);
        int bestChild, otherChild;
        double cut;
        if (__dadd_rn(diff1, diff2) < 0.0)
        {
            bestChild = n.child1;
            otherChild = n.child2;
            cut = sqdiff(val, n.hi);
        }
        else
        {
            bestChild = n.child2;
            otherChild = n.child1;
            cut = sqdiff(val, n.lo);
        }
        if (sp == kMaxDepth)
        {
            failed = true;
            break;
        }
        Frame& f = st[sp++];
        f.other = otherChild;
        f.idx = idx;
        f.cut = cut;
        f.mind = mind;
        f.dst = 0.0;
        f.phase = 0;
        node = bestChild;
    }

    if (failed)
    {
        atomicAdd(overflow, 1u);
        outIndex[qi] = 0xFFFFFFFFu;
        outDist2[qi] = DBL_MAX;
    }
    else
    {
        outIndex[qi] = bestId;
        outDist2[qi] = best;
    }
}

bool ok(cudaError_t e, const char* what)
{
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "cheshire knn: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

}  // namespace

bool available()
{
    static int g_available = -1;
    if (g_available < 0)
    {
        int n = 0;
        g_available = (cudaGetDeviceCount(&n) == cudaSuccess && n >= 1) ? 1 : 0;
    }
    return g_available == 1;
}

void* hostAlloc(std::size_t bytes)
{
    void* p = nullptr;
    if (!ok(cudaHostAlloc(&p, bytes, cudaHostAllocDefault), "pinned alloc")) return nullptr;
    return p;
}

void hostFree(void* p)
{
    if (p) cudaFreeHost(p);
}

bool Index::build(const double* points, std::size_t nbPoints, const Node* nodes, std::size_t nbNodes,
                  const std::uint32_t* perm, const double bboxLo[3], const double bboxHi[3], std::size_t maxQueries)
{
    release();
    if (nbPoints == 0 || nbNodes == 0 || nbPoints > 0xFFFFFFF0u || maxQueries == 0 || maxQueries > 0xFFFFFFF0u) return false;
    for (int i = 0; i < 3; ++i)
    {
        _lo[i] = bboxLo[i];
        _hi[i] = bboxHi[i];
    }
    cudaStream_t s = nullptr;
    if (!ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream")) return false;
    _stream = s;
    for (int i = 0; i < 4; ++i)
    {
        cudaEvent_t e = nullptr;
        if (!ok(cudaEventCreate(&e), "event")) return false;
        _events[i] = e;
    }
    if (!ok(cudaMalloc(&_points, nbPoints * 3 * sizeof(double)), "alloc points")) return false;
    if (!ok(cudaMalloc(&_nodes, nbNodes * sizeof(Node)), "alloc nodes")) return false;
    if (!ok(cudaMalloc(&_perm, nbPoints * sizeof(std::uint32_t)), "alloc perm")) return false;
    if (!ok(cudaMalloc(&_overflowCounter, sizeof(unsigned int)), "alloc counter")) return false;
    _overflowHost = hostAlloc(sizeof(unsigned int));
    if (_overflowHost == nullptr) return false;
    if (!reserve(maxQueries)) return false;
    if (!ok(cudaMemcpy(_points, points, nbPoints * 3 * sizeof(double), cudaMemcpyHostToDevice), "upload points")) return false;
    if (!ok(cudaMemcpy(_nodes, nodes, nbNodes * sizeof(Node), cudaMemcpyHostToDevice), "upload nodes")) return false;
    if (!ok(cudaMemcpy(_perm, perm, nbPoints * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "upload perm")) return false;
    return true;
}

bool Index::reserve(std::size_t maxQueries)
{
    if (_stream == nullptr || maxQueries == 0 || maxQueries > 0xFFFFFFF0u) return false;
    if (maxQueries <= _queryCapacity) return true;
    if (!wait()) return false;
    for (void** p : {&_queries, &_outIndex, &_outDist})
    {
        if (*p) cudaFree(*p);
        *p = nullptr;
    }
    _queryCapacity = 0;
    if (!ok(cudaMalloc(&_queries, maxQueries * 3 * sizeof(double)), "alloc queries")) return false;
    if (!ok(cudaMalloc(&_outIndex, maxQueries * sizeof(std::uint32_t)), "alloc out index")) return false;
    if (!ok(cudaMalloc(&_outDist, maxQueries * sizeof(double)), "alloc out dist")) return false;
    _queryCapacity = maxQueries;
    return true;
}

bool Index::queryAsync(const double* queries, std::size_t nbQueries, std::uint32_t* outIndex, double* outDist2, bool fma)
{
    if (_points == nullptr || _inFlight || nbQueries > _queryCapacity) return false;
    cudaStream_t s = (cudaStream_t)_stream;
    cudaEvent_t* ev = (cudaEvent_t*)_events;
    if (nbQueries == 0) return true;
    if (!ok(cudaEventRecord(ev[0], s), "event 0")) return false;
    if (!ok(cudaMemcpyAsync(_queries, queries, nbQueries * 3 * sizeof(double), cudaMemcpyHostToDevice, s), "upload queries")) return false;
    if (!ok(cudaMemsetAsync(_overflowCounter, 0, sizeof(unsigned int), s), "reset counter")) return false;
    if (!ok(cudaEventRecord(ev[1], s), "event 1")) return false;
    const unsigned int block = 128;
    const unsigned int grid = (unsigned int)((nbQueries + block - 1) / block);
    if (fma)
        knnKernel<true><<<grid, block, 0, s>>>((const Node*)_nodes, (const double*)_points, (const std::uint32_t*)_perm, (const double*)_queries,
                                               (std::uint32_t)nbQueries, _lo[0], _lo[1], _lo[2], _hi[0], _hi[1], _hi[2],
                                               (std::uint32_t*)_outIndex, (double*)_outDist, (unsigned int*)_overflowCounter);
    else
        knnKernel<false><<<grid, block, 0, s>>>((const Node*)_nodes, (const double*)_points, (const std::uint32_t*)_perm, (const double*)_queries,
                                                (std::uint32_t)nbQueries, _lo[0], _lo[1], _lo[2], _hi[0], _hi[1], _hi[2],
                                                (std::uint32_t*)_outIndex, (double*)_outDist, (unsigned int*)_overflowCounter);
    if (!ok(cudaGetLastError(), "launch")) return false;
    if (!ok(cudaEventRecord(ev[2], s), "event 2")) return false;
    if (!ok(cudaMemcpyAsync(outIndex, _outIndex, nbQueries * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, s), "download index")) return false;
    if (!ok(cudaMemcpyAsync(outDist2, _outDist, nbQueries * sizeof(double), cudaMemcpyDeviceToHost, s), "download dist")) return false;
    if (!ok(cudaMemcpyAsync(_overflowHost, _overflowCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost, s), "download counter")) return false;
    if (!ok(cudaEventRecord(ev[3], s), "event 3")) return false;
    _inFlight = true;
    return true;
}

bool Index::wait()
{
    _overflowed = 0;
    if (!_inFlight) return true;
    _inFlight = false;
    if (!ok(cudaStreamSynchronize((cudaStream_t)_stream), "sync")) return false;
    cudaEvent_t* ev = (cudaEvent_t*)_events;
    float ms = 0;
    if (cudaEventElapsedTime(&ms, ev[0], ev[1]) == cudaSuccess) _msUpload += ms;
    if (cudaEventElapsedTime(&ms, ev[1], ev[2]) == cudaSuccess) _msKernel += ms;
    if (cudaEventElapsedTime(&ms, ev[2], ev[3]) == cudaSuccess) _msDownload += ms;
    _overflowed = *(const unsigned int*)_overflowHost;
    return true;
}

void Index::release()
{
    if (_stream) cudaStreamSynchronize((cudaStream_t)_stream);
    _inFlight = false;
    for (void** p : {&_points, &_nodes, &_perm, &_queries, &_outIndex, &_outDist, &_overflowCounter})
    {
        if (*p) cudaFree(*p);
        *p = nullptr;
    }
    hostFree(_overflowHost);
    _overflowHost = nullptr;
    for (int i = 0; i < 4; ++i)
    {
        if (_events[i]) cudaEventDestroy((cudaEvent_t)_events[i]);
        _events[i] = nullptr;
    }
    if (_stream) cudaStreamDestroy((cudaStream_t)_stream);
    _stream = nullptr;
    _queryCapacity = 0;
    _overflowed = 0;
}

}  // namespace knn
}  // namespace cheshire
