// cheshire: a float-filtered walk for Meshing's visibility knn (hip/port/gpu_knn/knnGPU.cu), checked against the double walk.
//
// The port's kernel replays nanoflann's search in double: per leaf point a 9-operation FP64 metric, which consumer cards run
// at a fraction of their float rate (about 33 s per False Door pass on the RX 9070). Here each leaf's first point is a local
// origin; every point's offset from it is stored as a float, and per leaf the query's offset is computed once in double. A
// point's squared distance in float, df, comes with a bound B on |df - D| (D the double metric), covering every rounding with
// a 4x margin. When df - B >= best (rounded up to a float), D >= best, and nanoflann's `if (dist < worstDist())` would not
// take the point: it is skipped. Every other point gets the double metric and the double comparison, exactly as before. Only
// points the double walk rejects are skipped, so the answers - index, distance and ties - are the double walk's by
// construction; the inner nodes (the child order and the pruning chain) are unchanged double arithmetic.
//
// Input: a dump of the False Door's visibility pass 1 (CHESHIRE_KNN_DUMP, a temporary hook): the tree, the permutation, the
// bounding box and the host-built queries of every 100th camera.
//
// RESULT (2026-10-08, RX 9070, 32,065,518 queries of 9 cameras): no index or distance differs from the double walk, and only
// 17.9 % of the leaf points need the double metric - but the walk is no faster (366.6 ms against 368.3 ms). The kernel is not
// bound by FP64: it is a latency-bound walk of about 40 nodes and 18 points per query, scattered in memory, which is also why
// 6s's leaf-order layout gained about 1 % here. Kept as a recorded dead end and as the harness for any later knn work.
//   build.cmd knn_test && knn_test.exe C:\cheshire-fd-test\knn-dump
#include <hip/hip_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#pragma clang fp contract(off)

struct Node
{
    std::int32_t child1 = -1;
    std::int32_t child2 = -1;
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    double lo = 0;
    double hi = 0;
};

constexpr int kMaxDepth = 96;

__device__ __forceinline__ double dAdd(double a, double b) { return a + b; }
__device__ __forceinline__ double dSub(double a, double b) { return a - b; }
__device__ __forceinline__ double dMul(double a, double b) { return a * b; }

struct Frame
{
    int other;
    int idx;
    double cut;
    double mind;
    double dst;
    int phase;
};

template <bool FMA>
__device__ __forceinline__ double metric(const double* __restrict__ pts, std::uint32_t id, double q0, double q1, double q2)
{
    const double d0 = dSub(q0, pts[3 * (std::size_t)id + 0]);
    const double d1 = dSub(q1, pts[3 * (std::size_t)id + 1]);
    const double d2 = dSub(q2, pts[3 * (std::size_t)id + 2]);
    double r = 0.0;
    if (FMA)
    {
        r = __fma_rn(d0, d0, r);
        r = __fma_rn(d1, d1, r);
        r = __fma_rn(d2, d2, r);
    }
    else
    {
        r = dAdd(r, dMul(d0, d0));
        r = dAdd(r, dMul(d1, d1));
        r = dAdd(r, dMul(d2, d2));
    }
    return r;
}

__device__ __forceinline__ double sqdiff(double a, double b)
{
    const double d = dSub(a, b);
    return dMul(d, d);
}

// a float no smaller than x (x >= 0); +inf for x beyond float range
__device__ __forceinline__ float floatAtLeast(double x)
{
    if (x >= 3.4e38)
        return INFINITY;
    float f = (float)x;
    if ((double)f < x)
        f = nextafterf(f, INFINITY);
    return f;
}

// FILTER = false: the port's kernel (knnGPU.cu knnKernel) verbatim. FILTER = true: the leaf loop filtered as described above.
template <bool FMA, bool FILTER>
__global__ void knnKernel(const Node* __restrict__ nodes, const double* __restrict__ pts, const std::uint32_t* __restrict__ perm,
                          const float* __restrict__ rel, const double* __restrict__ queries, std::uint32_t nbQueries, double lo0, double lo1,
                          double lo2, double hi0, double hi1, double hi2, std::uint32_t* __restrict__ outIndex, double* __restrict__ outDist2,
                          unsigned int* __restrict__ overflow, unsigned long long* __restrict__ stats)
{
    const std::uint32_t qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nbQueries)
        return;

    double vec[3] = {queries[3 * (std::size_t)qi], queries[3 * (std::size_t)qi + 1], queries[3 * (std::size_t)qi + 2]};
    const double lo[3] = {lo0, lo1, lo2};
    const double hi[3] = {hi0, hi1, hi2};

    double dists[3] = {0.0, 0.0, 0.0};
    double mind = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        if (vec[i] < lo[i])
        {
            dists[i] = sqdiff(vec[i], lo[i]);
            mind = dAdd(mind, dists[i]);
        }
        if (vec[i] > hi[i])
        {
            dists[i] = sqdiff(vec[i], hi[i]);
            mind = dAdd(mind, dists[i]);
        }
    }

    double best = DBL_MAX;
    std::uint32_t bestId = 0xFFFFFFFFu;
    float bestHi = INFINITY;
    unsigned long long nPoints = 0, nExact = 0;

    Frame st[kMaxDepth];
    int sp = 0;
    int node = 0;
    bool failed = false;

    while (true)
    {
        const Node n = nodes[node];
        if (n.child1 < 0)
        {
            if (!FILTER)
            {
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
            }
            else if (n.a < n.b)
            {
                const std::uint32_t oid = perm[n.a];
                const float s0 = (float)dSub(vec[0], pts[3 * (std::size_t)oid + 0]);
                const float s1 = (float)dSub(vec[1], pts[3 * (std::size_t)oid + 1]);
                const float s2 = (float)dSub(vec[2], pts[3 * (std::size_t)oid + 2]);
                const float as0 = fabsf(s0), as1 = fabsf(s1), as2 = fabsf(s2);
                for (std::uint32_t i = n.a; i < n.b; ++i)
                {
                    const float r0 = rel[3 * (std::size_t)i + 0], r1 = rel[3 * (std::size_t)i + 1], r2 = rel[3 * (std::size_t)i + 2];
                    const float x0 = s0 - r0, x1 = s1 - r1, x2 = s2 - r2;
                    const float ax0 = fabsf(x0), ax1 = fabsf(x1), ax2 = fabsf(x2);
                    const float df = fmaf(x2, x2, fmaf(x1, x1, x0 * x0));
                    // |x - (q - p)| per axis <= 2^-24 (|s| + |r| + |x|) (three roundings); 2^-22 is a 4x margin
                    const float e0 = (as0 + fabsf(r0) + ax0) * 0x1p-22f;
                    const float e1 = (as1 + fabsf(r1) + ax1) * 0x1p-22f;
                    const float e2 = (as2 + fabsf(r2) + ax2) * 0x1p-22f;
                    // |df - D| <= sum e (2|x| + e) + the float sum's and the double metric's own roundings (df * 2^-21 covers both)
                    const float B = df * 0x1p-21f + e0 * (2.0f * ax0 + e0) + e1 * (2.0f * ax1 + e1) + e2 * (2.0f * ax2 + e2);
                    ++nPoints;
                    if (df - B >= bestHi)
                        continue;  // D >= best: the double walk would not take it
                    ++nExact;
                    const std::uint32_t id = perm[i];
                    const double d = metric<FMA>(pts, id, vec[0], vec[1], vec[2]);
                    if (d < best)
                    {
                        best = d;
                        bestId = id;
                        bestHi = floatAtLeast(best);
                    }
                }
            }
            bool descend = false;
            while (sp > 0)
            {
                Frame& f = st[sp - 1];
                if (f.phase == 0)
                {
                    f.phase = 1;
                    const double dst = dists[f.idx];
                    f.dst = dst;
                    const double m = dSub(dAdd(f.mind, f.cut), dst);
                    dists[f.idx] = f.cut;
                    if (m <= best)
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
            if (!descend)
                break;
            continue;
        }

        const int idx = (int)n.a;
        const double val = vec[idx];
        const double diff1 = dSub(val, n.lo);
        const double diff2 = dSub(val, n.hi);
        int bestChild, otherChild;
        double cut;
        if (dAdd(diff1, diff2) < 0.0)
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
    if (FILTER && stats)
    {
        atomicAdd(&stats[0], nPoints);
        atomicAdd(&stats[1], nExact);
    }
}

#define CHECK(c) do { hipError_t e_ = (c); if (e_ != hipSuccess) { std::printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

static std::vector<char> slurp(const std::string& path)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        return {};
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> v(n);
    std::fread(v.data(), 1, n, f);
    std::fclose(f);
    return v;
}

template <bool FMA, bool FILTER>
static float run(const Node* dNodes, const double* dPts, const std::uint32_t* dPerm, const float* dRel, const double* dQ, std::uint32_t nq,
                 const double* box, std::uint32_t* dIdx, double* dDist, unsigned int* dOv, unsigned long long* dStats)
{
    hipEvent_t a, b;
    CHECK(hipEventCreate(&a));
    CHECK(hipEventCreate(&b));
    CHECK(hipEventRecord(a));
    knnKernel<FMA, FILTER><<<(nq + 255) / 256, 256>>>(dNodes, dPts, dPerm, dRel, dQ, nq, box[0], box[1], box[2], box[3], box[4], box[5], dIdx,
                                                      dDist, dOv, dStats);
    CHECK(hipEventRecord(b));
    CHECK(hipEventSynchronize(b));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, a, b));
    return ms;
}

int main(int argc, char** argv)
{
    const std::string dir = argc > 1 ? argv[1] : "C:/cheshire-fd-test/knn-dump";
    const bool fma = !(argc > 2 && std::string(argv[2]) == "nofma");
    const auto pointsB = slurp(dir + "/points.bin"), nodesB = slurp(dir + "/nodes.bin"), permB = slurp(dir + "/perm.bin"), boxB = slurp(dir + "/bbox.bin");
    if (pointsB.empty() || nodesB.empty() || permB.empty() || boxB.size() != 48)
    {
        std::printf("dump not found in %s\n", dir.c_str());
        return 1;
    }
    const double* pts = reinterpret_cast<const double*>(pointsB.data());
    const Node* nodes = reinterpret_cast<const Node*>(nodesB.data());
    const std::uint32_t* perm = reinterpret_cast<const std::uint32_t*>(permB.data());
    const double* box = reinterpret_cast<const double*>(boxB.data());
    const std::size_t nPts = pointsB.size() / 24, nNodes = nodesB.size() / sizeof(Node), nPerm = permB.size() / 4;
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s (%s); %zu points, %zu nodes, %zu perm; metric %s\n", prop.name, prop.gcnArchName, nPts, nNodes, nPerm,
                fma ? "fused" : "unfused");

    // the float offsets: slot i of leaf [a, b) holds float(p(perm[i]) - p(perm[a]))
    std::vector<float> rel(3 * nPerm, 0.0f);
    std::size_t leaves = 0;
    for (std::size_t k = 0; k < nNodes; ++k)
    {
        const Node& n = nodes[k];
        if (n.child1 >= 0 || n.a >= n.b)
            continue;
        ++leaves;
        const double* o = pts + 3 * (std::size_t)perm[n.a];
        for (std::uint32_t i = n.a; i < n.b; ++i)
        {
            const double* p = pts + 3 * (std::size_t)perm[i];
            for (int d = 0; d < 3; ++d)
                rel[3 * (std::size_t)i + d] = (float)(p[d] - o[d]);
        }
    }
    std::printf("%zu leaves, %.2f points per leaf\n", leaves, double(nPerm) / leaves);

    Node* dNodes;
    double* dPts;
    std::uint32_t* dPerm;
    float* dRel;
    CHECK(hipMalloc(&dNodes, nodesB.size()));
    CHECK(hipMalloc(&dPts, pointsB.size()));
    CHECK(hipMalloc(&dPerm, permB.size()));
    CHECK(hipMalloc(&dRel, rel.size() * 4));
    CHECK(hipMemcpy(dNodes, nodes, nodesB.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dPts, pts, pointsB.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dPerm, perm, permB.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dRel, rel.data(), rel.size() * 4, hipMemcpyHostToDevice));
    unsigned int* dOv;
    unsigned long long* dStats;
    CHECK(hipMalloc(&dOv, 4));
    CHECK(hipMalloc(&dStats, 16));

    double totRef = 0, totFil = 0;
    unsigned long long totQ = 0, totBadIdx = 0, totBadDist = 0, allPoints = 0, allExact = 0;
    for (int cam = 0; cam < 2000; cam += 100)
    {
        const auto qB = slurp(dir + "/queries_" + std::to_string(cam) + ".bin");
        if (qB.empty())
            continue;
        const std::uint32_t nq = std::uint32_t(qB.size() / 24);
        double* dQ;
        std::uint32_t *dIdxA, *dIdxB;
        double *dDistA, *dDistB;
        CHECK(hipMalloc(&dQ, qB.size()));
        CHECK(hipMalloc(&dIdxA, nq * 4ull));
        CHECK(hipMalloc(&dIdxB, nq * 4ull));
        CHECK(hipMalloc(&dDistA, nq * 8ull));
        CHECK(hipMalloc(&dDistB, nq * 8ull));
        CHECK(hipMemcpy(dQ, qB.data(), qB.size(), hipMemcpyHostToDevice));
        CHECK(hipMemset(dOv, 0, 4));
        CHECK(hipMemset(dStats, 0, 16));
        float msRef, msFil;
        if (fma)
        {
            msRef = run<true, false>(dNodes, dPts, dPerm, dRel, dQ, nq, box, dIdxA, dDistA, dOv, nullptr);
            msFil = run<true, true>(dNodes, dPts, dPerm, dRel, dQ, nq, box, dIdxB, dDistB, dOv, dStats);
            // again for timing without the stats atomics' effect on the first run's caches
            msRef = run<true, false>(dNodes, dPts, dPerm, dRel, dQ, nq, box, dIdxA, dDistA, dOv, nullptr);
            msFil = run<true, true>(dNodes, dPts, dPerm, dRel, dQ, nq, box, dIdxB, dDistB, dOv, nullptr);
        }
        else
        {
            msRef = run<false, false>(dNodes, dPts, dPerm, dRel, dQ, nq, box, dIdxA, dDistA, dOv, nullptr);
            msFil = run<false, true>(dNodes, dPts, dPerm, dRel, dQ, nq, box, dIdxB, dDistB, dOv, dStats);
            msRef = run<false, false>(dNodes, dPts, dPerm, dRel, dQ, nq, box, dIdxA, dDistA, dOv, nullptr);
            msFil = run<false, true>(dNodes, dPts, dPerm, dRel, dQ, nq, box, dIdxB, dDistB, dOv, nullptr);
        }
        std::vector<std::uint32_t> ia(nq), ib(nq);
        std::vector<double> da(nq), db(nq);
        unsigned long long st[2];
        unsigned int ov;
        CHECK(hipMemcpy(ia.data(), dIdxA, nq * 4ull, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(ib.data(), dIdxB, nq * 4ull, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(da.data(), dDistA, nq * 8ull, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(db.data(), dDistB, nq * 8ull, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(st, dStats, 16, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(&ov, dOv, 4, hipMemcpyDeviceToHost));
        unsigned long long badI = 0, badD = 0;
        for (std::uint32_t k = 0; k < nq; ++k)
        {
            badI += ia[k] != ib[k];
            badD += std::memcmp(&da[k], &db[k], 8) != 0;
        }
        std::printf("camera %4d: %8u queries, double walk %8.1f ms, filtered %8.1f ms (%.2fx); %llu index and %llu distance differences; "
                    "%.1f leaf points per query, %.2f %% of them computed in double; %u overflows\n",
                    cam, nq, msRef, msFil, msRef / msFil, badI, badD, double(st[0]) / nq, 100.0 * st[1] / std::max(1ull, st[0]), ov);
        totRef += msRef;
        totFil += msFil;
        totQ += nq;
        totBadIdx += badI;
        totBadDist += badD;
        allPoints += st[0];
        allExact += st[1];
        CHECK(hipFree(dQ));
        CHECK(hipFree(dIdxA));
        CHECK(hipFree(dIdxB));
        CHECK(hipFree(dDistA));
        CHECK(hipFree(dDistB));
    }
    std::printf("all: %llu queries, double walk %.1f ms, filtered %.1f ms (%.2fx); %llu index and %llu distance differences; %.2f %% of leaf points in double\n",
                totQ, totRef, totFil, totRef / totFil, totBadIdx, totBadDist, 100.0 * allExact / std::max(1ull, allPoints));
    return 0;
}
