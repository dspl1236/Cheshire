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
#include <cfloat>
#include <cstdio>
#include <cstdlib>

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
    for (int i = 0; i < 5; ++i)
    {
        cudaEvent_t e = nullptr;
        if (!ok(cudaEventCreate(&e), "event")) return false;
        _events[i] = e;
    }
    if (!ok(cheshire::devMalloc(&_points, nbPoints * 3 * sizeof(double)), "alloc points")) return false;
    if (!ok(cheshire::devMalloc(&_nodes, nbNodes * sizeof(Node)), "alloc nodes")) return false;
    if (!ok(cheshire::devMalloc(&_perm, nbPoints * sizeof(std::uint32_t)), "alloc perm")) return false;
    if (!ok(cheshire::devMalloc(&_overflowCounter, sizeof(unsigned int)), "alloc counter")) return false;
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
    _bpInFlight = false;
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
    if (cudaEventElapsedTime(&ms, ev[2], ev[3]) == cudaSuccess) _msDownload += ms;
    _overflowed = *(const unsigned int*)_overflowHost;
    return true;
}

void Index::release()
{
    if (_stream) cudaStreamSynchronize((cudaStream_t)_stream);
    _inFlight = false;
    for (void** p : {&_points, &_nodes, &_perm, &_queries, &_outIndex, &_outDist, &_overflowCounter, &_depth, &_rowStart, &_pixSize})
    {
        if (*p) cheshire::devFree(*p);
        *p = nullptr;
    }
    hostFree(_overflowHost);
    _overflowHost = nullptr;
    for (int i = 0; i < 5; ++i)
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
}

}  // namespace knn
}  // namespace cheshire
