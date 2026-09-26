// Cheshire: nanoflann's nearest-neighbour search, replayed on the GPU. See knnGPU.hpp.
//
// Each thread answers one query with an explicit stack in place of searchLevel's recursion. A
// frame is pushed when an inner node is entered (the child on the query's side is walked first);
// when the walk returns to the frame it does what searchLevel does after its first recursive call:
//   dst = dists[idx]; mindist += cut_dist - dst; dists[idx] = cut_dist;
//   if (mindist <= worst) walk the other child; dists[idx] = dst;
// The metric is L2_Simple_Adaptor's: diff = q - p per axis, result += diff * diff, in double, in
// axis order; with fma set, the fused form clang-cl emits for that statement under /arch:AVX2.
// accum_dist and the initial bounding-box distances are unfused products in both host builds, and
// nanoflann's unfused metric is what a generic x86-64 host (the Linux bundle) computes. On HIP device
// code contracts by default, and the pragma below reaches only code written after it: HIP's own
// __dadd_rn/__dmul_rn are defined in a header included before this file and fused anyway, so the
// arithmetic goes through dAdd/dSub/dMul below (step 6k). CUDA gets --fmad=false from the build.
#ifdef __clang__
#pragma clang fp contract(off)
#endif
#include "knnGPU.hpp"
#include <cuda_runtime.h>
#include <aliceVision/depthMap/cuda/hip/cheshire/devalloc.h>  // cheshire: bridge on both backends
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cheshire {
namespace knn {

namespace {

constexpr int kMaxDepth = 96;

// Double sums, differences and products written in this file, under the pragma above. HIP's
// __dadd_rn/__dsub_rn/__dmul_rn are plain operators defined in its math header, which the compiler
// includes before this file's first line, so they are compiled with device code's default
// contraction: a product feeding a sum fuses into an FMA after inlining, whatever this file says.
// That is why the pragma alone changed nothing in the Linux knn check (docs/04, "The Linux knn
// distances are not a contraction") and why step 6k's first build did not match the host. These
// helpers are compiled here, so nothing fuses unless __fma_rn says so; on CUDA --fmad=false keeps
// them unfused as well. Division and sqrt cannot fuse and stay the intrinsics.
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

template<bool FMA>
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
            mind = dAdd(mind, dists[i]);
        }
        if (vec[i] > hi[i])
        {
            dists[i] = sqdiff(vec[i], hi[i]);
            mind = dAdd(mind, dists[i]);
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
                    const double m = dSub(dAdd(f.mind, f.cut), dst);
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
}

// cheshire (step 6s): the same walk with a smaller memory footprint. knnKernel's frames are 40 bytes
// and its stack is 96 frames whatever the tree, in private (scratch) memory, and every leaf point is
// read through the permutation, a random 24-byte gather per point. Here:
// - the points are stored in leaf order (slot i holds point perm[i]), so a leaf is one contiguous
//   run; the answer is perm[slot], read once per query;
// - a frame is 24 bytes: the other child, the split axis and the phase share one int, and the cut
//   distance is replaced by the distance it displaced (dst) once the frame is revisited, the only
//   value phase 1 needs;
// - the stack holds DEPTH frames, the smallest of 40/64/96 that covers the tree (Index::build);
// - the three axis distances are registers selected by the axis, not an indexed private array.
// Same points in the same order, same metric, same comparisons: the answers are knnKernel's, ties
// included. Off by default (CHESHIRE_GPU_KNN_LAYOUT=1 selects it): about 1 % faster on the RX 9070,
// whose cache hid the scattered reads anyway, and about 50 % slower on the RX 6750 XT (docs/04 6s).
struct PackedFrame
{
    double x;     // phase 0: the cut distance; phase 1: the axis distance it replaced
    double mind;  // mindist when the node was entered
    int packed;   // other child << 3 | axis << 1 | phase
};

__device__ __forceinline__ double axisGet(double a0, double a1, double a2, int i)
{
    return i == 0 ? a0 : (i == 1 ? a1 : a2);
}

__device__ __forceinline__ void axisSet(double& a0, double& a1, double& a2, int i, double v)
{
    if (i == 0)
        a0 = v;
    else if (i == 1)
        a1 = v;
    else
        a2 = v;
}

// computeInitialDistances for one axis, in nanoflann's order: below the box, then above it
__device__ __forceinline__ void initialAxis(double v, double lo, double hi, double& dist, double& mind)
{
    if (v < lo)
    {
        dist = sqdiff(v, lo);
        mind = dAdd(mind, dist);
    }
    if (v > hi)
    {
        dist = sqdiff(v, hi);
        mind = dAdd(mind, dist);
    }
}

template<bool FMA, int DEPTH>
__global__ void knnLeafKernel(const Node* __restrict__ nodes,
                              const double* __restrict__ leafPts,
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

    const double q0 = queries[3 * (std::size_t)qi];
    const double q1 = queries[3 * (std::size_t)qi + 1];
    const double q2 = queries[3 * (std::size_t)qi + 2];

    double d0 = 0.0, d1 = 0.0, d2 = 0.0;
    double mind = 0.0;
    initialAxis(q0, lo0, hi0, d0, mind);
    initialAxis(q1, lo1, hi1, d1, mind);
    initialAxis(q2, lo2, hi2, d2, mind);

    double best = DBL_MAX;
    std::uint32_t bestSlot = 0xFFFFFFFFu;

    PackedFrame st[DEPTH];
    int sp = 0;
    int node = 0;
    bool failed = false;

    while (true)
    {
        const Node n = nodes[node];
        if (n.child1 < 0)
        {
            for (std::uint32_t i = n.a; i < n.b; ++i)
            {
                const double d = metric<FMA>(leafPts, i, q0, q1, q2);
                if (d < best)
                {
                    best = d;
                    bestSlot = i;
                }
            }
            bool descend = false;
            while (sp > 0)
            {
                PackedFrame& f = st[sp - 1];
                const int idx = (f.packed >> 1) & 3;
                if ((f.packed & 1) == 0)
                {
                    const double dst = axisGet(d0, d1, d2, idx);
                    const double cut = f.x;
                    const double m = dSub(dAdd(f.mind, cut), dst);
                    axisSet(d0, d1, d2, idx, cut);
                    if (m <= best)
                    {
                        f.packed |= 1;
                        f.x = dst;
                        node = f.packed >> 3;
                        mind = m;
                        descend = true;
                        break;
                    }
                    axisSet(d0, d1, d2, idx, dst);
                    --sp;
                }
                else
                {
                    axisSet(d0, d1, d2, idx, f.x);
                    --sp;
                }
            }
            if (!descend) break;
            continue;
        }

        const int idx = (int)n.a;
        const double val = axisGet(q0, q1, q2, idx);
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
        if (sp == DEPTH)
        {
            failed = true;
            break;
        }
        PackedFrame& f = st[sp++];
        f.x = cut;
        f.mind = mind;
        f.packed = (otherChild << 3) | (idx << 1);
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
        outIndex[qi] = bestSlot == 0xFFFFFFFFu ? 0xFFFFFFFFu : perm[bestSlot];
        outDist2[qi] = best;
    }
}

// cheshire (step 6t): the votes on the device. PointCloud.cpp's votes, per query k with nearest
// vertex v (skipped when v is not a vertex: an overflowed or NaN query):
//   const float pixSizeScoreI = simScorePrepare[v] * pixSize * pixSize;  (float promoted, double
//                                                                         products, rounded to float)
//   const float pixSizeScoreV = scoreV[v];
//   if (dist < voteMarginFactor * std::max(pixSizeScoreI, pixSizeScoreV))   (float product, compared in double)
//       cams.push_back_distinct(c);
//       if (dist < contributeMarginFactor * pixSizeScoreV)
//           vc = (vc * (double)nrc + p) / double(nrc + 1); nrc += 1;
// No sum is involved in a decision, so nothing can be contracted there; the fold is three separate
// Point3d operators per component (x * n, + q, / (n + 1)), done here with the helpers above.
//
// A vote sets bit v of the camera's bitmap (the host appends the camera from it). A contribution
// counts itself on its vertex; an exact integer scan of the counts gives each vertex a range of
// slots, the contributions are scattered into their vertex's range (in any order), and one thread
// per vertex puts its range in ascending query order and folds it: each vertex sees its
// contributions in pixel order, as the host's ordered votes apply them. No library sort: rocPRIM
// picks its kernel configuration from the device on the host side but from the compile target on
// the device side, so a generic target (gfx12-generic) on a device it knows (gfx1201) launches one
// configuration and runs another, which faults (2026-09-25). Anything out of range is counted in
// counts[2] and never used; the host then reruns the pass with host votes.
__device__ __forceinline__ float fMul(float a, float b) { return a * b; }

constexpr int kVoteBlock = 256;
constexpr int kScanBlock = 256;
constexpr int kScanItems = 4;
constexpr int kScanTile = kScanBlock * kScanItems;

__global__ void decideKernel(const std::uint32_t* __restrict__ nn, const double* __restrict__ dist2, const double* __restrict__ pixSize,
                             std::uint32_t nq, std::uint32_t nbVertices, const float* __restrict__ sim, const float* __restrict__ scoreV,
                             float voteMargin, float contributeMargin, std::uint32_t* __restrict__ bitmap, std::uint32_t* __restrict__ keys,
                             unsigned int* __restrict__ cnt, unsigned long long* __restrict__ counts)
{
    __shared__ unsigned int blockVotes, blockContrib;
    if (threadIdx.x == 0)
    {
        blockVotes = 0;
        blockContrib = 0;
    }
    __syncthreads();
    const std::uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nq)
    {
        const std::uint32_t v = nn[k];
        std::uint32_t key = nbVertices;
        if (v < nbVertices)
        {
            const double dist = dist2[k];
            const double ps = pixSize[k];
            const float scoreI = __double2float_rn(dMul(dMul((double)sim[v], ps), ps));
            const float scoreV_ = scoreV[v];
            const float m = (scoreI < scoreV_) ? scoreV_ : scoreI;  // std::max(a, b) is (a < b) ? b : a
            if (dist < (double)fMul(voteMargin, m))
            {
                atomicOr(&bitmap[v >> 5], 1u << (v & 31u));
                atomicAdd(&blockVotes, 1u);
                if (dist < (double)fMul(contributeMargin, scoreV_))
                {
                    key = v;
                    atomicAdd(&cnt[v], 1u);
                    atomicAdd(&blockContrib, 1u);
                }
            }
        }
        keys[k] = key;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (blockVotes) atomicAdd(&counts[0], (unsigned long long)blockVotes);
        if (blockContrib) atomicAdd(&counts[1], (unsigned long long)blockContrib);
    }
}

// inclusive scan of s[0 .. kScanBlock) in shared memory (Hillis-Steele); every thread must call it
__device__ __forceinline__ unsigned int blockInclusiveScan(unsigned int* s, unsigned int x)
{
    s[threadIdx.x] = x;
    __syncthreads();
    for (int off = 1; off < kScanBlock; off <<= 1)
    {
        const unsigned int t = ((int)threadIdx.x >= off) ? s[threadIdx.x - off] : 0u;
        __syncthreads();
        s[threadIdx.x] += t;
        __syncthreads();
    }
    return s[threadIdx.x];
}

// the exclusive prefix of cnt[0 .. n) in three launches: tile totals, their prefix (one block), the tiles
__global__ void scanTotalsKernel(const unsigned int* __restrict__ cnt, std::uint32_t n, unsigned int* __restrict__ tileSums)
{
    __shared__ unsigned int s[kScanBlock];
    const std::size_t base = (std::size_t)blockIdx.x * kScanTile + (std::size_t)threadIdx.x * kScanItems;
    unsigned int sum = 0;
    for (int i = 0; i < kScanItems; ++i)
        if (base + i < n) sum += cnt[base + i];
    const unsigned int incl = blockInclusiveScan(s, sum);
    if (threadIdx.x == kScanBlock - 1) tileSums[blockIdx.x] = incl;
}

__global__ void scanTileSumsKernel(unsigned int* __restrict__ tileSums, std::uint32_t nTiles)
{
    __shared__ unsigned int s[kScanBlock];
    __shared__ unsigned int carry;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    for (std::uint32_t base = 0; base < nTiles; base += kScanBlock)
    {
        const std::uint32_t i = base + threadIdx.x;
        const unsigned int x = (i < nTiles) ? tileSums[i] : 0u;
        const unsigned int incl = blockInclusiveScan(s, x);
        const unsigned int total = s[kScanBlock - 1];
        if (i < nTiles) tileSums[i] = carry + incl - x;
        __syncthreads();
        if (threadIdx.x == 0) carry += total;
        __syncthreads();
    }
}

__global__ void scanTilesKernel(const unsigned int* __restrict__ cnt, std::uint32_t n, const unsigned int* __restrict__ tileStarts,
                                unsigned int* __restrict__ offsets)
{
    __shared__ unsigned int s[kScanBlock];
    const std::size_t base = (std::size_t)blockIdx.x * kScanTile + (std::size_t)threadIdx.x * kScanItems;
    unsigned int x[kScanItems];
    unsigned int sum = 0;
    for (int i = 0; i < kScanItems; ++i)
    {
        x[i] = (base + i < n) ? cnt[base + i] : 0u;
        sum += x[i];
    }
    const unsigned int incl = blockInclusiveScan(s, sum);
    unsigned int run = tileStarts[blockIdx.x] + incl - sum;
    for (int i = 0; i < kScanItems; ++i)
    {
        if (base + i < n) offsets[base + i] = run;
        run += x[i];
    }
}

// each contribution into its vertex's range; cnt goes back to 0 for the next camera
__global__ void scatterKernel(const std::uint32_t* __restrict__ keys, std::uint32_t nq, std::uint32_t nbVertices, const unsigned int* __restrict__ offsets,
                              unsigned int* __restrict__ cnt, std::uint32_t* __restrict__ slots, std::uint32_t* __restrict__ slotVertex,
                              unsigned long long* __restrict__ counts)
{
    const std::uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nq) return;
    const std::uint32_t v = keys[k];
    if (v >= nbVertices) return;
    const unsigned int c = atomicSub(&cnt[v], 1u);
    const unsigned int pos = offsets[v] + c - 1u;
    if (c == 0u || pos >= nq || pos >= offsets[v + 1])
    {
        atomicAdd(&counts[2], 1ull);
        return;
    }
    slots[pos] = k;
    slotVertex[pos] = v;
}

__device__ void siftDown(std::uint32_t* a, unsigned int root, unsigned int n)
{
    while (true)
    {
        unsigned int child = 2u * root + 1u;
        if (child >= n) return;
        if (child + 1u < n && a[child] < a[child + 1u]) ++child;
        if (a[root] >= a[child]) return;
        const std::uint32_t t = a[root];
        a[root] = a[child];
        a[child] = t;
        root = child;
    }
}

// x = (x * n + q) / (n + 1) per component, as Point3d's operators: one product, one sum, one quotient
__device__ __forceinline__ void foldOne(double& x, double& y, double& z, int n, double qx, double qy, double qz)
{
    const double dn = (double)n;
    const double d1 = (double)(n + 1);
    x = __ddiv_rn(dAdd(dMul(x, dn), qx), d1);
    y = __ddiv_rn(dAdd(dMul(y, dn), qy), d1);
    z = __ddiv_rn(dAdd(dMul(z, dn), qz), d1);
}

// the first slot of each vertex's range: sort the range by query (insertion, heapsort when long), fold it
__global__ void foldRunsKernel(const unsigned int* __restrict__ offsets, std::uint32_t nbVertices, std::uint32_t* __restrict__ slots,
                               const std::uint32_t* __restrict__ slotVertex, std::uint32_t nq, const double* __restrict__ q,
                               double* __restrict__ coords, int* __restrict__ nrc, unsigned long long* __restrict__ counts)
{
    const std::uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = offsets[nbVertices];
    if (p >= nq || p >= total) return;
    const std::uint32_t v = slotVertex[p];
    if (v >= nbVertices || offsets[v] != p) return;
    const unsigned int end = offsets[v + 1];
    if (end > total || end > nq || end <= p)
    {
        atomicAdd(&counts[2], 1ull);
        return;
    }
    std::uint32_t* a = slots + p;
    const unsigned int len = end - p;
    if (len <= 32u)
    {
        for (unsigned int i = 1; i < len; ++i)
        {
            const std::uint32_t key = a[i];
            unsigned int j = i;
            while (j > 0 && a[j - 1] > key)
            {
                a[j] = a[j - 1];
                --j;
            }
            a[j] = key;
        }
    }
    else
    {
        for (unsigned int i = len / 2; i-- > 0;)
            siftDown(a, i, len);
        for (unsigned int e = len - 1; e > 0; --e)
        {
            const std::uint32_t t = a[0];
            a[0] = a[e];
            a[e] = t;
            siftDown(a, 0, e);
        }
    }
    double x = coords[3 * (std::size_t)v], y = coords[3 * (std::size_t)v + 1], z = coords[3 * (std::size_t)v + 2];
    int n = nrc[v];
    for (unsigned int i = 0; i < len; ++i)
    {
        const std::size_t k = a[i];
        if (k >= nq)
        {
            atomicAdd(&counts[2], 1ull);
            return;
        }
        foldOne(x, y, z, n, q[3 * k], q[3 * k + 1], q[3 * k + 2]);
        n += 1;
    }
    coords[3 * (std::size_t)v] = x;
    coords[3 * (std::size_t)v + 1] = y;
    coords[3 * (std::size_t)v + 2] = z;
    nrc[v] = n;
}

__global__ void foldProbeKernel(const double* __restrict__ x, const int* __restrict__ n, const double* __restrict__ q, std::uint32_t count,
                                double* __restrict__ out)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    double a = x[3 * i], b = x[3 * i + 1], c = x[3 * i + 2];
    foldOne(a, b, c, n[i], q[3 * i], q[3 * i + 1], q[3 * i + 2]);
    out[3 * i] = a;
    out[3 * i + 1] = b;
    out[3 * i + 2] = c;
}

// the points in leaf order: slot i gets point perm[i]
__global__ void gatherLeafOrder(const double* __restrict__ pts, const std::uint32_t* __restrict__ perm, std::size_t n, double* __restrict__ out)
{
    const std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::size_t id = perm[i];
    out[3 * i] = pts[3 * id];
    out[3 * i + 1] = pts[3 * id + 1];
    out[3 * i + 2] = pts[3 * id + 2];
}

// cheshire (step 6k): the host's backprojection of one camera, operation by operation.
// MultiViewParams::backproject(c, Point2d(x, y), depth) is CArr + (iCamArr * pix).normalize() * depth
// and getCamPixelSize(p, c) projects p with camArr, moves one pixel right, backprojects that pixel's
// ray and returns the ray's distance to p, |cross(ray, C - p)|. The expressions are Matrix3x3's
// m11*x + m12*y + m13, Matrix3x4's m11*x + m12*y + m13*z + m14, Point3d's x*x + y*y + z*z and
// cross's a.y*b.z - a.z*b.y. A host compiler that contracts within an expression (clang-cl under
// /arch:AVX2) fuses the first product of each sum into the second, and the square sums from the
// left: fma(m11, x, m12*y) + m13, fma(m13, z, fma(m11, x, m12*y)) + m14, fma(z, z, fma(x, x, y*y)),
// fma(a.y, b.z, -(a.z*b.y)). Operator-level steps (C + n*depth, C - p, pix.x + 1.0, the divisions)
// are separate statements or functions and never fused. sqrt and division are correctly rounded on
// both sides. FMA selects the fused forms; a generic x86-64 host (the Linux bundle) computes the
// plain ones.
struct BpCam
{
    double C[3];
    double iK[9];
    double P[12];
};

template<bool FMA>
__device__ __forceinline__ double lin2(double a, double x, double b, double y, double c)  // a*x + b*y + c
{
    return FMA ? dAdd(__fma_rn(a, x, dMul(b, y)), c) : dAdd(dAdd(dMul(a, x), dMul(b, y)), c);
}

template<bool FMA>
__device__ __forceinline__ double lin3(double a, double x, double b, double y, double c, double z, double d)  // a*x + b*y + c*z + d
{
    return FMA ? dAdd(__fma_rn(c, z, __fma_rn(a, x, dMul(b, y))), d)
               : dAdd(dAdd(dAdd(dMul(a, x), dMul(b, y)), dMul(c, z)), d);
}

template<bool FMA>
__device__ __forceinline__ double sumSq(double x, double y, double z)  // x*x + y*y + z*z
{
    return FMA ? __fma_rn(z, z, __fma_rn(x, x, dMul(y, y))) : dAdd(dAdd(dMul(x, x), dMul(y, y)), dMul(z, z));
}

template<bool FMA>
__device__ __forceinline__ double diffProd(double a, double b, double c, double d)  // a*b - c*d
{
    return FMA ? __fma_rn(a, b, -dMul(c, d)) : dSub(dMul(a, b), dMul(c, d));
}

constexpr int kBpBlock = 256;

// One block per row; the row is walked in tiles of kBpBlock pixels, the valid pixels of a tile
// numbered by a block scan, so query k is the host's k: rowStart[y] plus the valid pixels before x.
template<bool FMA>
__global__ void backprojectKernel(const float* __restrict__ depth, int w, int h, const std::uint64_t* __restrict__ rowStart,
                                  BpCam cam, double* __restrict__ q, double* __restrict__ pixSize)
{
    const int y = blockIdx.x;
    if (y >= h) return;
    __shared__ unsigned int scan[kBpBlock];
    std::uint64_t base = rowStart[y];
    for (int x0 = 0; x0 < w; x0 += kBpBlock)
    {
        const int x = x0 + (int)threadIdx.x;
        float d = 0.0f;
        bool valid = false;
        if (x < w)
        {
            d = depth[(std::size_t)y * (std::size_t)w + (std::size_t)x];
            valid = !(d <= 0.0f);  // the host keeps what `depth <= 0.0f` does not reject, NaN included
        }
        scan[threadIdx.x] = valid ? 1u : 0u;
        __syncthreads();
        for (int off = 1; off < kBpBlock; off <<= 1)
        {
            const unsigned int t = ((int)threadIdx.x >= off) ? scan[threadIdx.x - off] : 0u;
            __syncthreads();
            scan[threadIdx.x] += t;
            __syncthreads();
        }
        const unsigned int incl = scan[threadIdx.x];
        const unsigned int total = scan[kBpBlock - 1];
        __syncthreads();
        if (valid)
        {
            const std::uint64_t k = base + incl - 1u;
            // backproject: n = (iK * (x, y)).normalize(); p = C + n * depth
            const double px = (double)x, py = (double)y;
            double vx = lin2<FMA>(cam.iK[0], px, cam.iK[1], py, cam.iK[2]);
            double vy = lin2<FMA>(cam.iK[3], px, cam.iK[4], py, cam.iK[5]);
            double vz = lin2<FMA>(cam.iK[6], px, cam.iK[7], py, cam.iK[8]);
            const double vn = __dsqrt_rn(sumSq<FMA>(vx, vy, vz));
            vx = __ddiv_rn(vx, vn);
            vy = __ddiv_rn(vy, vn);
            vz = __ddiv_rn(vz, vn);
            const double dd = (double)d;
            const double p0 = dAdd(cam.C[0], dMul(vx, dd));
            const double p1 = dAdd(cam.C[1], dMul(vy, dd));
            const double p2 = dAdd(cam.C[2], dMul(vz, dd));
            q[3 * k] = p0;
            q[3 * k + 1] = p1;
            q[3 * k + 2] = p2;
            // getCamPixelSize: getPixelFor3DPoint, pix.x + 1, the ray through it, pointLineDistance3D
            const double X = lin3<FMA>(cam.P[0], p0, cam.P[1], p1, cam.P[2], p2, cam.P[3]);
            const double Y = lin3<FMA>(cam.P[4], p0, cam.P[5], p1, cam.P[6], p2, cam.P[7]);
            const double Z = lin3<FMA>(cam.P[8], p0, cam.P[9], p1, cam.P[10], p2, cam.P[11]);
            double ux = -1.0, uy = -1.0;
            if (!(Z <= 0))
            {
                ux = __ddiv_rn(X, Z);
                uy = __ddiv_rn(Y, Z);
            }
            ux = dAdd(ux, 1.0);
            double wx = lin2<FMA>(cam.iK[0], ux, cam.iK[1], uy, cam.iK[2]);
            double wy = lin2<FMA>(cam.iK[3], ux, cam.iK[4], uy, cam.iK[5]);
            double wz = lin2<FMA>(cam.iK[6], ux, cam.iK[7], uy, cam.iK[8]);
            const double wn = __dsqrt_rn(sumSq<FMA>(wx, wy, wz));
            wx = __ddiv_rn(wx, wn);
            wy = __ddiv_rn(wy, wn);
            wz = __ddiv_rn(wz, wn);
            const double b0 = dSub(cam.C[0], p0);
            const double b1 = dSub(cam.C[1], p1);
            const double b2 = dSub(cam.C[2], p2);
            const double c0 = diffProd<FMA>(wy, b2, wz, b1);
            const double c1 = diffProd<FMA>(wz, b0, wx, b2);
            const double c2 = diffProd<FMA>(wx, b1, wy, b0);
            const double s = sumSq<FMA>(c0, c1, c2);
            pixSize[k] = (s == 0.0) ? 0.0 : __dsqrt_rn(s);
        }
        base += total;
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
    for (int i = 0; i < 6; ++i)
    {
        cudaEvent_t e = nullptr;
        if (!ok(cudaEventCreate(&e), "event")) return false;
        _events[i] = e;
    }
    // step 6s: the leaf-order kernel, unless switched off or the tree is too large for its packed
    // frames (a node id takes 28 bits); its stack is the smallest of 40/64/96 frames that holds the
    // deepest path (one frame per inner node). Children are flattened after their parent, so one
    // forward pass gives every node's level; a tree that breaks that order gets the full stack.
    _leafOrder = ::cheshire::env::flag("CHESHIRE_GPU_KNN_LAYOUT", false) && nbNodes < (std::size_t(1) << 28);
    _stack = kMaxDepth;
    _treeFrames = -1;
    {
        std::vector<int> level(nbNodes, 0);
        level[0] = 1;
        int deepest = 1;
        bool ordered = true;
        for (std::size_t i = 0; i < nbNodes && ordered; ++i)
        {
            if (nodes[i].child1 < 0) continue;
            for (std::int32_t c : {nodes[i].child1, nodes[i].child2})
            {
                if (c <= (std::int32_t)i || (std::size_t)c >= nbNodes)
                {
                    ordered = false;
                    break;
                }
                level[c] = level[i] + 1;
                deepest = std::max(deepest, level[c]);
            }
        }
        if (ordered)
        {
            _treeFrames = deepest - 1;
            if (_leafOrder)
                _stack = (deepest - 1 <= 40) ? 40 : ((deepest - 1 <= 64) ? 64 : kMaxDepth);
        }
    }
    if (!ok(cheshire::devMalloc(&_points, nbPoints * 3 * sizeof(double)), "alloc points")) return false;
    if (!ok(cheshire::devMalloc(&_nodes, nbNodes * sizeof(Node)), "alloc nodes")) return false;
    if (!ok(cheshire::devMalloc(&_perm, nbPoints * sizeof(std::uint32_t)), "alloc perm")) return false;
    if (!ok(cheshire::devMalloc(&_overflowCounter, sizeof(unsigned int)), "alloc counter")) return false;
    _overflowHost = hostAlloc(sizeof(unsigned int));
    if (_overflowHost == nullptr) return false;
    if (!ok(cudaMemcpyAsync(_nodes, nodes, nbNodes * sizeof(Node), cudaMemcpyHostToDevice, s), "upload nodes")) return false;
    if (!ok(cudaMemcpyAsync(_perm, perm, nbPoints * sizeof(std::uint32_t), cudaMemcpyHostToDevice, s), "upload perm")) return false;
    if (_leafOrder)
    {
        // the points in their original order into a temporary, gathered into leaf order; done
        // before the query buffers exist, so the temporary does not add to the pass's peak
        void* original = nullptr;
        if (!ok(cheshire::devMalloc(&original, nbPoints * 3 * sizeof(double)), "alloc points (original order)")) return false;
        bool good = ok(cudaMemcpyAsync(original, points, nbPoints * 3 * sizeof(double), cudaMemcpyHostToDevice, s), "upload points");
        if (good)
        {
            const unsigned int block = 256;
            gatherLeafOrder<<<(unsigned int)((nbPoints + block - 1) / block), block, 0, s>>>((const double*)original, (const std::uint32_t*)_perm, nbPoints,
                                                                                             (double*)_points);
            good = ok(cudaGetLastError(), "launch gather");
        }
        good = ok(cudaStreamSynchronize(s), "gather") && good;
        cheshire::devFree(original);
        if (!good) return false;
    }
    else
    {
        if (!ok(cudaMemcpyAsync(_points, points, nbPoints * 3 * sizeof(double), cudaMemcpyHostToDevice, s), "upload points")) return false;
        if (!ok(cudaStreamSynchronize(s), "upload")) return false;
    }
    if (!reserve(maxQueries)) return false;
    return true;
}

namespace {
struct KnnArgs
{
    const Node* nodes;
    const double* pts;
    const std::uint32_t* perm;
    const double* queries;
    std::uint32_t nbQueries;
    double lo[3];
    double hi[3];
    std::uint32_t* outIndex;
    double* outDist2;
    unsigned int* overflow;
};

template<bool FMA>
void launchFlat(unsigned int grid, unsigned int block, cudaStream_t s, const KnnArgs& a)
{
    knnKernel<FMA><<<grid, block, 0, s>>>(a.nodes, a.pts, a.perm, a.queries, a.nbQueries, a.lo[0], a.lo[1], a.lo[2], a.hi[0], a.hi[1], a.hi[2],
                                          a.outIndex, a.outDist2, a.overflow);
}

template<bool FMA, int DEPTH>
void launchLeaf(unsigned int grid, unsigned int block, cudaStream_t s, const KnnArgs& a)
{
    knnLeafKernel<FMA, DEPTH><<<grid, block, 0, s>>>(a.nodes, a.pts, a.perm, a.queries, a.nbQueries, a.lo[0], a.lo[1], a.lo[2], a.hi[0], a.hi[1],
                                                     a.hi[2], a.outIndex, a.outDist2, a.overflow);
}
}  // namespace

bool Index::launchKnn(std::size_t nbQueries, bool fma)
{
    cudaStream_t s = (cudaStream_t)_stream;
    const KnnArgs a{(const Node*)_nodes, (const double*)_points, (const std::uint32_t*)_perm, (const double*)_queries, (std::uint32_t)nbQueries,
                    {_lo[0], _lo[1], _lo[2]}, {_hi[0], _hi[1], _hi[2]}, (std::uint32_t*)_outIndex, (double*)_outDist,
                    (unsigned int*)_overflowCounter};
    const unsigned int block = 128;
    const unsigned int grid = (unsigned int)((nbQueries + block - 1) / block);
    if (!_leafOrder)
        fma ? launchFlat<true>(grid, block, s, a) : launchFlat<false>(grid, block, s, a);
    else if (_stack == 40)
        fma ? launchLeaf<true, 40>(grid, block, s, a) : launchLeaf<false, 40>(grid, block, s, a);
    else if (_stack == 64)
        fma ? launchLeaf<true, 64>(grid, block, s, a) : launchLeaf<false, 64>(grid, block, s, a);
    else
        fma ? launchLeaf<true, kMaxDepth>(grid, block, s, a) : launchLeaf<false, kMaxDepth>(grid, block, s, a);
    return ok(cudaGetLastError(), "launch");
}

bool Index::reserve(std::size_t maxQueries)
{
    if (_stream == nullptr || maxQueries == 0 || maxQueries > 0xFFFFFFF0u) return false;
    if (maxQueries <= _queryCapacity) return true;
    if (!wait()) return false;
    for (void** p : {&_queries, &_outIndex, &_outDist})
    {
        if (*p) cheshire::devFree(*p);
        *p = nullptr;
    }
    _queryCapacity = 0;
    if (!ok(cheshire::devMalloc(&_queries, maxQueries * 3 * sizeof(double)), "alloc queries")) return false;
    if (!ok(cheshire::devMalloc(&_outIndex, maxQueries * sizeof(std::uint32_t)), "alloc out index")) return false;
    if (!ok(cheshire::devMalloc(&_outDist, maxQueries * sizeof(double)), "alloc out dist")) return false;
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
    if (!launchKnn(nbQueries, fma)) return false;
    if (!ok(cudaEventRecord(ev[2], s), "event 2")) return false;
    if (!ok(cudaMemcpyAsync(outIndex, _outIndex, nbQueries * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, s), "download index")) return false;
    if (!ok(cudaMemcpyAsync(outDist2, _outDist, nbQueries * sizeof(double), cudaMemcpyDeviceToHost, s), "download dist")) return false;
    if (!ok(cudaMemcpyAsync(_overflowHost, _overflowCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost, s), "download counter")) return false;
    if (!ok(cudaEventRecord(ev[3], s), "event 3")) return false;
    _inFlight = true;
    return true;
}

bool Index::reserveBackproject(std::size_t maxPixels, std::size_t maxRows)
{
    if (_stream == nullptr || maxPixels == 0 || maxRows == 0) return false;
    if (maxPixels <= _bpPixelCapacity && maxRows + 1 <= _bpRowCapacity && maxPixels <= _queryCapacity) return true;
    if (!wait()) return false;
    if (!reserve(maxPixels)) return false;
    for (void** p : {&_depth, &_rowStart, &_pixSize})
    {
        if (*p) cheshire::devFree(*p);
        *p = nullptr;
    }
    _bpPixelCapacity = 0;
    _bpRowCapacity = 0;
    if (!ok(cheshire::devMalloc(&_depth, maxPixels * sizeof(float)), "alloc depth")) return false;
    if (!ok(cheshire::devMalloc(&_rowStart, (maxRows + 1) * sizeof(std::uint64_t)), "alloc row starts")) return false;
    if (!ok(cheshire::devMalloc(&_pixSize, maxPixels * sizeof(double)), "alloc pixel sizes")) return false;
    _bpPixelCapacity = maxPixels;
    _bpRowCapacity = maxRows + 1;
    return true;
}

bool Index::backprojectQueryAsync(const float* depth, int w, int h, const std::uint64_t* rowStart, std::size_t nbQueries,
                                  const Camera& cam, bool fmaBackproject, double* outQueries, double* outPixSize,
                                  std::uint32_t* outIndex, double* outDist2, bool fma)
{
    if (_points == nullptr || _inFlight || w <= 0 || h <= 0) return false;
    const std::size_t pixels = (std::size_t)w * (std::size_t)h;
    if (nbQueries > _queryCapacity || nbQueries > pixels || pixels > _bpPixelCapacity || (std::size_t)h + 1 > _bpRowCapacity) return false;
    if (nbQueries == 0) return true;
    cudaStream_t s = (cudaStream_t)_stream;
    cudaEvent_t* ev = (cudaEvent_t*)_events;
    BpCam bc;
    for (int i = 0; i < 3; ++i) bc.C[i] = cam.C[i];
    for (int i = 0; i < 9; ++i) bc.iK[i] = cam.iK[i];
    for (int i = 0; i < 12; ++i) bc.P[i] = cam.P[i];
    if (!ok(cudaEventRecord(ev[0], s), "event 0")) return false;
    if (!ok(cudaMemcpyAsync(_depth, depth, pixels * sizeof(float), cudaMemcpyHostToDevice, s), "upload depth")) return false;
    if (!ok(cudaMemcpyAsync(_rowStart, rowStart, ((std::size_t)h + 1) * sizeof(std::uint64_t), cudaMemcpyHostToDevice, s), "upload row starts")) return false;
    if (!ok(cudaMemsetAsync(_overflowCounter, 0, sizeof(unsigned int), s), "reset counter")) return false;
    if (!ok(cudaEventRecord(ev[1], s), "event 1")) return false;
    if (fmaBackproject)
        backprojectKernel<true><<<(unsigned int)h, kBpBlock, 0, s>>>((const float*)_depth, w, h, (const std::uint64_t*)_rowStart, bc, (double*)_queries, (double*)_pixSize);
    else
        backprojectKernel<false><<<(unsigned int)h, kBpBlock, 0, s>>>((const float*)_depth, w, h, (const std::uint64_t*)_rowStart, bc, (double*)_queries, (double*)_pixSize);
    if (!ok(cudaGetLastError(), "launch backprojection")) return false;
    if (!ok(cudaEventRecord(ev[4], s), "event 4")) return false;
    if (!launchKnn(nbQueries, fma)) return false;
    if (!ok(cudaEventRecord(ev[2], s), "event 2")) return false;
    if (!ok(cudaMemcpyAsync(outQueries, _queries, nbQueries * 3 * sizeof(double), cudaMemcpyDeviceToHost, s), "download queries")) return false;
    if (!ok(cudaMemcpyAsync(outPixSize, _pixSize, nbQueries * sizeof(double), cudaMemcpyDeviceToHost, s), "download pixel sizes")) return false;
    if (!ok(cudaMemcpyAsync(outIndex, _outIndex, nbQueries * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, s), "download index")) return false;
    if (!ok(cudaMemcpyAsync(outDist2, _outDist, nbQueries * sizeof(double), cudaMemcpyDeviceToHost, s), "download dist")) return false;
    if (!ok(cudaMemcpyAsync(_overflowHost, _overflowCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost, s), "download counter")) return false;
    if (!ok(cudaEventRecord(ev[3], s), "event 3")) return false;
    _inFlight = true;
    _bpInFlight = true;
    return true;
}

bool Index::wait()
{
    _overflowed = 0;
    if (!_inFlight) return true;
    _inFlight = false;
    const bool bp = _bpInFlight;
    const bool votes = _votesInFlight;
    _bpInFlight = false;
    _votesInFlight = false;
    if (!ok(cudaStreamSynchronize((cudaStream_t)_stream), "sync")) return false;
    cudaEvent_t* ev = (cudaEvent_t*)_events;
    float ms = 0;
    if (cudaEventElapsedTime(&ms, ev[0], ev[1]) == cudaSuccess) _msUpload += ms;
    if (bp)
    {
        // ev[4] closes the backprojection kernel (step 6k)
        if (cudaEventElapsedTime(&ms, ev[1], ev[4]) == cudaSuccess) _msBackproject += ms;
        if (cudaEventElapsedTime(&ms, ev[4], ev[2]) == cudaSuccess) _msKernel += ms;
    }
    else if (cudaEventElapsedTime(&ms, ev[1], ev[2]) == cudaSuccess)
        _msKernel += ms;
    if (votes)
    {
        // ev[5] closes the fold (step 6t): decide, sort and fold, then the downloads
        if (cudaEventElapsedTime(&ms, ev[2], ev[5]) == cudaSuccess) _msVotes += ms;
        if (cudaEventElapsedTime(&ms, ev[5], ev[3]) == cudaSuccess) _msDownload += ms;
    }
    else if (cudaEventElapsedTime(&ms, ev[2], ev[3]) == cudaSuccess)
        _msDownload += ms;
    _overflowed = *(const unsigned int*)_overflowHost;
    return true;
}

void Index::release()
{
    if (_stream) cudaStreamSynchronize((cudaStream_t)_stream);
    _inFlight = false;
    releaseVotes();
    for (void** p : {&_points, &_nodes, &_perm, &_queries, &_outIndex, &_outDist, &_overflowCounter, &_depth, &_rowStart, &_pixSize})
    {
        if (*p) cheshire::devFree(*p);
        *p = nullptr;
    }
    hostFree(_overflowHost);
    _overflowHost = nullptr;
    for (int i = 0; i < 6; ++i)
    {
        if (_events[i]) cudaEventDestroy((cudaEvent_t)_events[i]);
        _events[i] = nullptr;
    }
    if (_stream) cudaStreamDestroy((cudaStream_t)_stream);
    _stream = nullptr;
    _queryCapacity = 0;
    _bpPixelCapacity = 0;
    _bpRowCapacity = 0;
    _overflowed = 0;
    _bpInFlight = false;
    _votesInFlight = false;
    _treeFrames = -1;
}

// ---- device votes (step 6t) -----------------------------------------------------------------

void Index::releaseVotes()
{
    for (void** p : {&_vCoords, &_vNrc, &_vSim, &_vScoreV, &_vBitmap, &_vKeys, &_vCnt, &_vOffsets, &_vTileSums, &_vSlots, &_vSlotVertex, &_vCounts})
    {
        if (*p) cheshire::devFree(*p);
        *p = nullptr;
    }
    _votesN = 0;
    _vCapacity = 0;
    _vTiles = 0;
}

bool Index::votesBegin(std::size_t nbVertices, const double* coords, const int* nrc, const float* sim, const float* scoreV, float voteMargin,
                       float contributeMargin, std::size_t maxQueries)
{
    if (_stream == nullptr || nbVertices == 0 || nbVertices >= 0xFFFFFFF0u || maxQueries == 0 || maxQueries > _queryCapacity
        || _pixSize == nullptr)
        return false;
    if (!wait()) return false;
    releaseVotes();
    cudaStream_t s = (cudaStream_t)_stream;
    const std::size_t words = (nbVertices + 31) / 32;
    const std::size_t tiles = (nbVertices + 1 + kScanTile - 1) / kScanTile;  // the scan covers nbVertices + 1 counts
    bool good = true;
    good = good && ok(cheshire::devMalloc(&_vCoords, nbVertices * 3 * sizeof(double)), "alloc vote coordinates");
    good = good && ok(cheshire::devMalloc(&_vNrc, nbVertices * sizeof(int)), "alloc vote nrc");
    good = good && ok(cheshire::devMalloc(&_vSim, nbVertices * sizeof(float)), "alloc vote sim");
    good = good && ok(cheshire::devMalloc(&_vScoreV, nbVertices * sizeof(float)), "alloc vote scores");
    good = good && ok(cheshire::devMalloc(&_vBitmap, words * sizeof(std::uint32_t)), "alloc vote bitmap");
    good = good && ok(cheshire::devMalloc(&_vKeys, maxQueries * sizeof(std::uint32_t)), "alloc vote keys");
    good = good && ok(cheshire::devMalloc(&_vCnt, (nbVertices + 1) * sizeof(unsigned int)), "alloc vote counts per vertex");
    good = good && ok(cheshire::devMalloc(&_vOffsets, (nbVertices + 1) * sizeof(unsigned int)), "alloc vote offsets");
    good = good && ok(cheshire::devMalloc(&_vTileSums, tiles * sizeof(unsigned int)), "alloc vote scan");
    good = good && ok(cheshire::devMalloc(&_vSlots, maxQueries * sizeof(std::uint32_t)), "alloc vote slots");
    good = good && ok(cheshire::devMalloc(&_vSlotVertex, maxQueries * sizeof(std::uint32_t)), "alloc vote slot vertices");
    good = good && ok(cheshire::devMalloc(&_vCounts, 3 * sizeof(unsigned long long)), "alloc vote counts");
    good = good && ok(cudaMemcpyAsync(_vCoords, coords, nbVertices * 3 * sizeof(double), cudaMemcpyHostToDevice, s), "upload vote coordinates");
    good = good && ok(cudaMemcpyAsync(_vNrc, nrc, nbVertices * sizeof(int), cudaMemcpyHostToDevice, s), "upload vote nrc");
    good = good && ok(cudaMemcpyAsync(_vSim, sim, nbVertices * sizeof(float), cudaMemcpyHostToDevice, s), "upload vote sim");
    good = good && ok(cudaMemcpyAsync(_vScoreV, scoreV, nbVertices * sizeof(float), cudaMemcpyHostToDevice, s), "upload vote scores");
    good = good && ok(cudaMemsetAsync(_vBitmap, 0, words * sizeof(std::uint32_t), s), "clear vote bitmap");
    good = good && ok(cudaMemsetAsync(_vCnt, 0, (nbVertices + 1) * sizeof(unsigned int), s), "clear vote counts per vertex");
    good = good && ok(cudaStreamSynchronize(s), "vote upload");
    if (!good)
    {
        releaseVotes();
        return false;
    }
    _votesN = nbVertices;
    _vCapacity = maxQueries;
    _vTiles = tiles;
    _voteMargin = voteMargin;
    _contributeMargin = contributeMargin;
    return true;
}

bool Index::backprojectQueryVoteAsync(const float* depth, int w, int h, const std::uint64_t* rowStart, std::size_t nbQueries, const Camera& cam,
                                      bool fmaBackproject, bool fma, std::uint32_t* outBitmap, unsigned long long* outCounts, double* outQueries,
                                      double* outPixSize, std::uint32_t* outIndex, double* outDist2)
{
    if (_points == nullptr || _inFlight || _votesN == 0 || w <= 0 || h <= 0) return false;
    const std::size_t pixels = (std::size_t)w * (std::size_t)h;
    if (nbQueries > _queryCapacity || nbQueries > _vCapacity || nbQueries > pixels || pixels > _bpPixelCapacity || (std::size_t)h + 1 > _bpRowCapacity)
        return false;
    cudaStream_t s = (cudaStream_t)_stream;
    cudaEvent_t* ev = (cudaEvent_t*)_events;
    const std::size_t words = votesBitmapWords();
    if (nbQueries == 0)
    {
        // nothing voted: an empty bitmap and zero counts, as the host would have
        for (std::size_t i = 0; i < words; ++i) outBitmap[i] = 0;
        outCounts[0] = outCounts[1] = outCounts[2] = 0;
        return true;
    }
    BpCam bc;
    for (int i = 0; i < 3; ++i) bc.C[i] = cam.C[i];
    for (int i = 0; i < 9; ++i) bc.iK[i] = cam.iK[i];
    for (int i = 0; i < 12; ++i) bc.P[i] = cam.P[i];
    if (!ok(cudaEventRecord(ev[0], s), "event 0")) return false;
    if (!ok(cudaMemcpyAsync(_depth, depth, pixels * sizeof(float), cudaMemcpyHostToDevice, s), "upload depth")) return false;
    if (!ok(cudaMemcpyAsync(_rowStart, rowStart, ((std::size_t)h + 1) * sizeof(std::uint64_t), cudaMemcpyHostToDevice, s), "upload row starts")) return false;
    if (!ok(cudaMemsetAsync(_overflowCounter, 0, sizeof(unsigned int), s), "reset counter")) return false;
    if (!ok(cudaMemsetAsync(_vCounts, 0, 3 * sizeof(unsigned long long), s), "reset vote counts")) return false;
    if (!ok(cudaEventRecord(ev[1], s), "event 1")) return false;
    if (fmaBackproject)
        backprojectKernel<true><<<(unsigned int)h, kBpBlock, 0, s>>>((const float*)_depth, w, h, (const std::uint64_t*)_rowStart, bc, (double*)_queries, (double*)_pixSize);
    else
        backprojectKernel<false><<<(unsigned int)h, kBpBlock, 0, s>>>((const float*)_depth, w, h, (const std::uint64_t*)_rowStart, bc, (double*)_queries, (double*)_pixSize);
    if (!ok(cudaGetLastError(), "launch backprojection")) return false;
    if (!ok(cudaEventRecord(ev[4], s), "event 4")) return false;
    if (!launchKnn(nbQueries, fma)) return false;
    if (!ok(cudaEventRecord(ev[2], s), "event 2")) return false;
    const unsigned int grid = (unsigned int)((nbQueries + kVoteBlock - 1) / kVoteBlock);
    const std::uint32_t nv = (std::uint32_t)_votesN;
    decideKernel<<<grid, kVoteBlock, 0, s>>>((const std::uint32_t*)_outIndex, (const double*)_outDist, (const double*)_pixSize, (std::uint32_t)nbQueries, nv,
                                             (const float*)_vSim, (const float*)_vScoreV, _voteMargin, _contributeMargin, (std::uint32_t*)_vBitmap,
                                             (std::uint32_t*)_vKeys, (unsigned int*)_vCnt, (unsigned long long*)_vCounts);
    if (!ok(cudaGetLastError(), "launch decide")) return false;
    scanTotalsKernel<<<(unsigned int)_vTiles, kScanBlock, 0, s>>>((const unsigned int*)_vCnt, nv + 1, (unsigned int*)_vTileSums);
    scanTileSumsKernel<<<1, kScanBlock, 0, s>>>((unsigned int*)_vTileSums, (std::uint32_t)_vTiles);
    scanTilesKernel<<<(unsigned int)_vTiles, kScanBlock, 0, s>>>((const unsigned int*)_vCnt, nv + 1, (const unsigned int*)_vTileSums, (unsigned int*)_vOffsets);
    if (!ok(cudaGetLastError(), "launch scan")) return false;
    scatterKernel<<<grid, kVoteBlock, 0, s>>>((const std::uint32_t*)_vKeys, (std::uint32_t)nbQueries, nv, (const unsigned int*)_vOffsets, (unsigned int*)_vCnt,
                                              (std::uint32_t*)_vSlots, (std::uint32_t*)_vSlotVertex, (unsigned long long*)_vCounts);
    foldRunsKernel<<<grid, kVoteBlock, 0, s>>>((const unsigned int*)_vOffsets, nv, (std::uint32_t*)_vSlots, (const std::uint32_t*)_vSlotVertex,
                                               (std::uint32_t)nbQueries, (const double*)_queries, (double*)_vCoords, (int*)_vNrc,
                                               (unsigned long long*)_vCounts);
    if (!ok(cudaGetLastError(), "launch fold")) return false;
    if (!ok(cudaEventRecord(ev[5], s), "event 5")) return false;
    if (!ok(cudaMemcpyAsync(outBitmap, _vBitmap, words * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, s), "download bitmap")) return false;
    if (!ok(cudaMemsetAsync(_vBitmap, 0, words * sizeof(std::uint32_t), s), "clear vote bitmap")) return false;
    if (!ok(cudaMemcpyAsync(outCounts, _vCounts, 3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost, s), "download vote counts")) return false;
    if (outQueries && !ok(cudaMemcpyAsync(outQueries, _queries, nbQueries * 3 * sizeof(double), cudaMemcpyDeviceToHost, s), "download queries")) return false;
    if (outPixSize && !ok(cudaMemcpyAsync(outPixSize, _pixSize, nbQueries * sizeof(double), cudaMemcpyDeviceToHost, s), "download pixel sizes")) return false;
    if (outIndex && !ok(cudaMemcpyAsync(outIndex, _outIndex, nbQueries * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, s), "download index")) return false;
    if (outDist2 && !ok(cudaMemcpyAsync(outDist2, _outDist, nbQueries * sizeof(double), cudaMemcpyDeviceToHost, s), "download dist")) return false;
    if (!ok(cudaMemcpyAsync(_overflowHost, _overflowCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost, s), "download counter")) return false;
    if (!ok(cudaEventRecord(ev[3], s), "event 3")) return false;
    _inFlight = true;
    _bpInFlight = true;
    _votesInFlight = true;
    return true;
}

bool Index::votesEnd(double* coords, int* nrc)
{
    if (_votesN == 0) return false;
    if (!wait()) return false;
    cudaStream_t s = (cudaStream_t)_stream;
    if (!ok(cudaMemcpyAsync(coords, _vCoords, _votesN * 3 * sizeof(double), cudaMemcpyDeviceToHost, s), "download vote coordinates")) return false;
    if (!ok(cudaMemcpyAsync(nrc, _vNrc, _votesN * sizeof(int), cudaMemcpyDeviceToHost, s), "download vote nrc")) return false;
    return ok(cudaStreamSynchronize(s), "vote download");
}

bool Index::foldProbe(const double* x, const int* n, const double* q, std::size_t count, double* out)
{
    if (_stream == nullptr || count == 0 || count > 0xFFFFFFu) return false;
    if (!wait()) return false;
    cudaStream_t s = (cudaStream_t)_stream;
    void *dx = nullptr, *dn = nullptr, *dq = nullptr, *dout = nullptr;
    bool good = ok(cheshire::devMalloc(&dx, count * 3 * sizeof(double)), "alloc probe") && ok(cheshire::devMalloc(&dn, count * sizeof(int)), "alloc probe")
                && ok(cheshire::devMalloc(&dq, count * 3 * sizeof(double)), "alloc probe") && ok(cheshire::devMalloc(&dout, count * 3 * sizeof(double)), "alloc probe");
    good = good && ok(cudaMemcpyAsync(dx, x, count * 3 * sizeof(double), cudaMemcpyHostToDevice, s), "upload probe")
           && ok(cudaMemcpyAsync(dn, n, count * sizeof(int), cudaMemcpyHostToDevice, s), "upload probe")
           && ok(cudaMemcpyAsync(dq, q, count * 3 * sizeof(double), cudaMemcpyHostToDevice, s), "upload probe");
    if (good)
    {
        foldProbeKernel<<<(unsigned int)((count + kVoteBlock - 1) / kVoteBlock), kVoteBlock, 0, s>>>((const double*)dx, (const int*)dn, (const double*)dq,
                                                                                                    (std::uint32_t)count, (double*)dout);
        good = ok(cudaGetLastError(), "launch probe")
               && ok(cudaMemcpyAsync(out, dout, count * 3 * sizeof(double), cudaMemcpyDeviceToHost, s), "download probe");
    }
    good = ok(cudaStreamSynchronize(s), "probe") && good;
    for (void* p : {dx, dn, dq, dout})
        if (p) cheshire::devFree(p);
    return good;
}

}  // namespace knn
}  // namespace cheshire
