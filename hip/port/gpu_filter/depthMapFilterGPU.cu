// Cheshire: DepthMapFilter group pass on the GPU (see depthMapFilterGPU.hpp).
//
// The device functions below are line-for-line transcriptions of the upstream CPU helpers the pass
// calls (Fuser::updateInSurr, MultiViewParams::getPixelFor3DPoint / getCamPixelSize /
// getCamPixelSizeRcTc / getCamPixelSizePlaneSweepAlpha, mvsUtils::getTarEpipolarDirectedLine /
// get2dLineImageIntersection / triangulateMatch, lineLineIntersect, pointLineDistance3D), keeping
// the same operation order, the same double/float conversions (upstream mixes them) and the same
// early-outs. FMA contraction is off for this file so a*b+c rounds twice as it does on the CPU.
#ifdef __clang__
#pragma clang fp contract(off)
#endif
#include "depthMapFilterGPU.hpp"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace aliceVision {
namespace fuseCut {
namespace gpu {

namespace {

struct D3 { double x, y, z; };
struct D2 { double x, y; };

__device__ __forceinline__ D3 sub3(D3 a, D3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ __forceinline__ D3 add3(D3 a, D3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ __forceinline__ D3 mul3(D3 a, double d) { return {a.x * d, a.y * d, a.z * d}; }
__device__ __forceinline__ double size3(D3 a) { const double d = a.x * a.x + a.y * a.y + a.z * a.z; return d == 0.0 ? 0.0 : sqrt(d); }   // Point3d::size
__device__ __forceinline__ D3 norm3(D3 a) { const double d = sqrt(a.x * a.x + a.y * a.y + a.z * a.z); return {a.x / d, a.y / d, a.z / d}; }  // Point3d::normalize
__device__ __forceinline__ D3 cross3(D3 a, D3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
__device__ __forceinline__ D2 sub2(D2 a, D2 b) { return {a.x - b.x, a.y - b.y}; }
__device__ __forceinline__ D2 add2(D2 a, D2 b) { return {a.x + b.x, a.y + b.y}; }
__device__ __forceinline__ D2 mul2(D2 a, double d) { return {a.x * d, a.y * d}; }
__device__ __forceinline__ double size2(D2 a) { return sqrt(a.x * a.x + a.y * a.y); }
__device__ __forceinline__ D2 norm2(D2 a) { const double d = sqrt(a.x * a.x + a.y * a.y); return {a.x / d, a.y / d}; }

// Matrix3x4 * Point3d
__device__ __forceinline__ D3 mulP(const double* P, D3 p) {
    return {P[0] * p.x + P[1] * p.y + P[2] * p.z + P[3], P[4] * p.x + P[5] * p.y + P[6] * p.z + P[7], P[8] * p.x + P[9] * p.y + P[10] * p.z + P[11]};
}
// Matrix3x3 * Point2d
__device__ __forceinline__ D3 mulM2(const double* M, D2 p) {
    return {M[0] * p.x + M[1] * p.y + M[2], M[3] * p.x + M[4] * p.y + M[5], M[6] * p.x + M[7] * p.y + M[8]};
}
__device__ __forceinline__ D3 c3(const double* v) { return {v[0], v[1], v[2]}; }

// MultiViewParams::getPixelFor3DPoint(Point2d*, X, P)
__device__ __forceinline__ D2 pixelFor3D(const double* P, D3 X) {
    const D3 XT = mulP(P, X);
    if (XT.z <= 0) return {-1.0, -1.0};
    return {XT.x / XT.z, XT.y / XT.z};
}
// MultiViewParams::getPixelFor3DPoint(Pixel*, X, rc)
__device__ __forceinline__ void pixelFor3Di(const double* P, D3 X, int& px, int& py) {
    const D3 XT = mulP(P, X);
    if (XT.z <= 0) { px = -1; py = -1; return; }
    px = int(floor(XT.x / XT.z + 0.5));
    py = int(floor(XT.y / XT.z + 0.5));
}

// mvsUtils::get2dLineImageIntersection
__device__ bool lineImageIntersection(D2& pFrom, D2& pTo, D2 l1, D2 l2, double rw, double rh) {
    D2 v = sub2(l2, l1);
    if (size2(v) < FLT_EPSILON) return false;
    v = norm2(v);
    const double a = -v.y, b = v.x;
    const double c = -a * l1.x - b * l1.y;
    int n = 0;
    double x = 0, y = -c / b;
    if ((y >= 0) && (y < rh)) { pFrom = {x, y}; n++; }
    x = rw; y = (-c - a * rw) / b;
    if ((y >= 0) && (y < rh)) { if (n == 0) pFrom = {x, y}; else pTo = {x, y}; n++; }
    x = -c / a; y = 0;
    if ((x >= 0) && (x < rw)) { if (n == 0) pFrom = {x, y}; else pTo = {x, y}; n++; }
    x = (-c - b * rh) / a; y = rh;
    if ((x >= 0) && (x < rw)) { if (n == 0) pFrom = {x, y}; else pTo = {x, y}; n++; }
    if (n == 2) {
        if (size2(sub2(l1, pFrom)) > size2(sub2(l1, pTo))) { const D2 t = pFrom; pFrom = pTo; pTo = t; }
        return true;
    }
    return false;
}

// lineLineIntersect (k, l form); only the midpoint is used by the caller
__device__ bool lineLineIntersect(D3& llis, D3 p1, D3 p2, D3 p3, D3 p4) {
    double p13[3] = {p1.x - p3.x, p1.y - p3.y, p1.z - p3.z};
    double p43[3] = {p4.x - p3.x, p4.y - p3.y, p4.z - p3.z};
    if ((fabs(p43[0]) < FLT_EPSILON) && (fabs(p43[1]) < FLT_EPSILON) && (fabs(p43[2]) < FLT_EPSILON)) return false;
    double p21[3] = {p2.x - p1.x, p2.y - p1.y, p2.z - p1.z};
    if ((fabs(p21[0]) < FLT_EPSILON) && (fabs(p21[1]) < FLT_EPSILON) && (fabs(p21[2]) < FLT_EPSILON)) return false;
    const double d1343 = p13[0] * p43[0] + p13[1] * p43[1] + p13[2] * p43[2];
    const double d4321 = p43[0] * p21[0] + p43[1] * p21[1] + p43[2] * p21[2];
    const double d1321 = p13[0] * p21[0] + p13[1] * p21[1] + p13[2] * p21[2];
    const double d4343 = p43[0] * p43[0] + p43[1] * p43[1] + p43[2] * p43[2];
    const double d2121 = p21[0] * p21[0] + p21[1] * p21[1] + p21[2] * p21[2];
    const double denom = d2121 * d4343 - d4321 * d4321;
    if (fabs(denom) < FLT_EPSILON) return false;
    const double numer = d1343 * d4321 - d1321 * d4343;
    const double mua = numer / denom;
    const double mub = (d1343 + d4321 * mua) / d4343;
    const double pa[3] = {p1.x + mua * p21[0], p1.y + mua * p21[1], p1.z + mua * p21[2]};
    const double pb[3] = {p3.x + mub * p43[0], p3.y + mub * p43[1], p3.z + mub * p43[2]};
    llis = {(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0, (pa[2] + pb[2]) / 2.0};
    return true;
}

// MultiViewParams::getCamPixelSize(x0, cam, float d)
__device__ __forceinline__ double camPixelSize(const CamGeom& cam, D3 x0, float d) {
    if (d == 0.0f) return 0.0f;
    D2 pix = pixelFor3D(cam.P, x0);
    pix.x = pix.x + d;
    const D3 vect = norm3(mulM2(cam.iCam, pix));
    return size3(cross3(vect, sub3(c3(cam.C), x0)));   // pointLineDistance3D(x0, C, vect)
}

// MultiViewParams::getCamPixelSizeRcTc(p, rc, tc, float d)
__device__ double camPixelSizeRcTc(const CamGeom& rc, const CamGeom& tc, D3 p, float d) {
    if (d == 0.0f) return 0.0f;
    D3 p1 = add3(c3(rc.C), mul3(sub3(p, c3(rc.C)), 0.1f));
    const D2 rpix = pixelFor3D(rc.P, p);
    // getTarEpipolarDirectedLine (its return value is ignored upstream; pFrom/pTo default to 0 here)
    D2 pFrom = {0.0, 0.0}, pTo = {0.0, 0.0};
    {
        const D3 refvect = norm3(mulM2(rc.riP, rpix));
        const float dd = float(size3(sub3(c3(rc.rC), c3(tc.rC))));
        D3 X = add3(mul3(refvect, dd), c3(rc.rC));
        const D2 tarpix1 = pixelFor3D(tc.P, X);
        X = add3(mul3(mul3(refvect, dd), 500.0), c3(rc.rC));
        const D2 tarpix2 = pixelFor3D(tc.P, X);
        lineImageIntersection(pFrom, pTo, tarpix1, tarpix2, double(tc.w), double(tc.h));
    }
    const D2 pixelVect = mul2(norm2(sub2(pTo, pFrom)), d);
    const D2 tpix = pixelFor3D(tc.P, p);
    const D2 tpix1 = add2(tpix, mul2(pixelVect, d));
    // triangulateMatch(p1, rpix, tpix1, rc, tc)
    const D3 refpoint = add3(norm3(mulM2(rc.iCam, rpix)), c3(rc.C));
    const D3 tarpoint = add3(norm3(mulM2(tc.iCam, tpix1)), c3(tc.C));
    if (!lineLineIntersect(p1, c3(rc.C), refpoint, c3(tc.C), tarpoint))
        return camPixelSize(rc, p, d);
    return size3(sub3(p, p1));
}

// one thread per tc pixel: Fuser::filterGroupsRC's inner loop + updateInSurr
__global__ void voteKernel(const float* __restrict__ tcDepth, CamGeom tc, CamGeom rc,
                           const float* __restrict__ rcDepth, const float* __restrict__ rcSim,
                           float pixToleranceFactor, int pixSizeBall, int pixSizeBallWSP, int* __restrict__ numOfPts)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= tc.w || y >= tc.h) return;
    const float depth = tcDepth[size_t(y) * tc.w + x];
    if (!(depth > 0.0f)) return;
    const D3 p = add3(c3(tc.C), mul3(norm3(mulM2(tc.iCam, D2{double(x), double(y)})), double(depth)));
    // updateInSurr(pixToleranceFactor, pixSizeBall, pixSizeBallWSP, p, rc, tc, numOfPtsMap, depthMap, simMap, 1)
    int px, py; pixelFor3Di(rc.P, p, px, py);
    if (!((px >= 2) && (px < rc.w - 2) && (py >= 2) && (py < rc.h - 2))) return;   // isPixelInImage, g_border = 2
    const float pixDepth = float(size3(sub3(c3(rc.C), p)));
    int d = pixSizeBall;
    const float sim = rcSim[size_t(py) * rc.w + px];
    if (sim >= 1.0f) d = pixSizeBallWSP;
    // getCamPixelSizePlaneSweepAlpha(p, rc, tc, 1, 1): splaneSweepAlpha = 1.0
    const double avRcTc = camPixelSizeRcTc(rc, tc, p, 1.0f);
    const double avRc = camPixelSize(rc, p, 1.0f);
    const float pixSize = float(double(pixToleranceFactor) * ((avRcTc + avRc) * 0.5));
    const int x0 = max(0, px - d), x1 = min(rc.w - 1, px + d), y0 = max(0, py - d), y1 = min(rc.h - 1, py + d);
    for (int nx = x0; nx <= x1; ++nx)
        for (int ny = y0; ny <= y1; ++ny)
            if (fabsf(pixDepth - rcDepth[size_t(ny) * rc.w + nx]) < pixSize)
                atomicAdd(numOfPts + size_t(ny) * rc.w + nx, 1);
}

// one thread: the intermediate values of voteKernel for a chosen tc pixel
__global__ void probeKernel(const float* __restrict__ tcDepth, CamGeom tc, CamGeom rc, const float* __restrict__ rcDepth, const float* __restrict__ rcSim,
                            float pixToleranceFactor, int pixSizeBall, int pixSizeBallWSP, int x, int y, double* __restrict__ out)
{
    for (int i = 0; i < 10; ++i) out[i] = -999.0;
    const float depth = tcDepth[size_t(y) * tc.w + x];
    if (!(depth > 0.0f)) return;
    const D3 p = add3(c3(tc.C), mul3(norm3(mulM2(tc.iCam, D2{double(x), double(y)})), double(depth)));
    out[6] = p.x; out[7] = p.y; out[8] = p.z;
    int px, py; pixelFor3Di(rc.P, p, px, py);
    out[0] = px; out[1] = py;
    if (!((px >= 2) && (px < rc.w - 2) && (py >= 2) && (py < rc.h - 2))) return;
    out[2] = float(size3(sub3(c3(rc.C), p)));
    const double avRcTc = camPixelSizeRcTc(rc, tc, p, 1.0f);
    const double avRc = camPixelSize(rc, p, 1.0f);
    out[3] = avRcTc; out[4] = avRc;
    out[5] = float(double(pixToleranceFactor) * ((avRcTc + avRc) * 0.5));
    out[9] = rcDepth[size_t(py) * rc.w + px];
}

__global__ void foldKernel(const int* __restrict__ numOfPts, unsigned char* __restrict__ modals, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) modals[i] += (unsigned char)(numOfPts[i] > 0);
}

bool g_checked = false, g_available = false;
std::mutex g_mutex;

}  // namespace

bool available()
{
    std::lock_guard<std::mutex> g(g_mutex);
    if (g_checked) return g_available;
    g_checked = true;
    if (const char* e = std::getenv("CHESHIRE_GPU_FILTER")) if (e[0] == '0') { std::fprintf(stderr, "[cheshire] depth map filter: disabled by CHESHIRE_GPU_FILTER=0, CPU\n"); return g_available = false; }
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { std::fprintf(stderr, "[cheshire] depth map filter: no GPU device, CPU\n"); return g_available = false; }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return g_available = false;
    std::fprintf(stderr, "[cheshire] depth map filter: group votes on %s (CHESHIRE_GPU_FILTER=0 for the CPU pass)\n", p.name);
    return g_available = true;
}

struct GroupFilter::Impl {
    float* rcDepth = nullptr; float* rcSim = nullptr; int* pts = nullptr; unsigned char* modals = nullptr; size_t rcCap = 0;
    float* tcDepth = nullptr; size_t tcCap = 0;
    CamGeom rc{};
    static bool grow(void** p, size_t* cap, size_t need) {
        if (need <= *cap) return true;
        if (*p) cudaFree(*p);
        *p = nullptr; *cap = 0;
        if (cudaMalloc(p, need) != cudaSuccess) return false;
        *cap = need; return true;
    }
    ~Impl() { if (rcDepth) cudaFree(rcDepth); if (rcSim) cudaFree(rcSim); if (pts) cudaFree(pts); if (modals) cudaFree(modals); if (tcDepth) cudaFree(tcDepth); }
};

GroupFilter::GroupFilter() : impl_(new Impl) {}
GroupFilter::~GroupFilter() { delete impl_; }

bool GroupFilter::setRc(const float* depth, const float* sim, const CamGeom& rc)
{
    Impl& m = *impl_;
    m.rc = rc;
    const size_t n = size_t(rc.w) * rc.h;
    if (n > m.rcCap) {
        for (void** p : {(void**)&m.rcDepth, (void**)&m.rcSim, (void**)&m.pts, (void**)&m.modals}) { if (*p) cudaFree(*p); *p = nullptr; }
        m.rcCap = 0;
        if (cudaMalloc((void**)&m.rcDepth, n * 4) != cudaSuccess || cudaMalloc((void**)&m.rcSim, n * 4) != cudaSuccess
            || cudaMalloc((void**)&m.pts, n * 4) != cudaSuccess || cudaMalloc((void**)&m.modals, n) != cudaSuccess) return false;
        m.rcCap = n;
    }
    if (cudaMemcpy(m.rcDepth, depth, n * 4, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (cudaMemcpy(m.rcSim, sim, n * 4, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (cudaMemset(m.pts, 0, n * 4) != cudaSuccess) return false;   // once per rc (see accumulate)
    return cudaMemset(m.modals, 0, n) == cudaSuccess;
}

bool GroupFilter::accumulate(const float* tcDepth, const CamGeom& tc, float pixToleranceFactor, int pixSizeBall, int pixSizeBallWSP)
{
    Impl& m = *impl_;
    const size_t n = size_t(m.rc.w) * m.rc.h, tn = size_t(tc.w) * tc.h;
    if (!Impl::grow((void**)&m.tcDepth, &m.tcCap, tn * 4)) return false;
    if (cudaMemcpy(m.tcDepth, tcDepth, tn * 4, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    // Upstream's per-camera vote buffer is "reset" with StaticVector::resize_with(n, 0), which is
    // std::vector::resize and leaves existing elements alone, so votes accumulate across the
    // neighbour cameras and numOfModals counts "neighbours since the first one that agreed", not
    // "neighbours that agree". Meshroom's consistency thresholds were tuned against that, so the
    // default replicates it (measured: first neighbour identical, then CPU counts only grow).
    // CHESHIRE_GPU_FILTER_STRICT=1 clears the buffer per camera, which is what the code meant.
    static const bool strict = [] { const char* e = std::getenv("CHESHIRE_GPU_FILTER_STRICT"); return e && e[0] == '1'; }();
    if (strict && cudaMemset(m.pts, 0, n * 4) != cudaSuccess) return false;
    const dim3 block(16, 16), grid((tc.w + 15) / 16, (tc.h + 15) / 16);
    voteKernel<<<grid, block>>>(m.tcDepth, tc, m.rc, m.rcDepth, m.rcSim, pixToleranceFactor, pixSizeBall, pixSizeBallWSP, m.pts);
    foldKernel<<<(unsigned)((n + 255) / 256), 256>>>(m.pts, m.modals, int(n));
    return cudaGetLastError() == cudaSuccess;
}

bool GroupFilter::probe(const float* tcDepth, const CamGeom& tc, int x, int y, float pixToleranceFactor, int pixSizeBall, int pixSizeBallWSP, double out[10])
{
    Impl& m = *impl_;
    const size_t tn = size_t(tc.w) * tc.h;
    if (!Impl::grow((void**)&m.tcDepth, &m.tcCap, tn * 4)) return false;
    if (cudaMemcpy(m.tcDepth, tcDepth, tn * 4, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    double* d = nullptr;
    if (cudaMalloc((void**)&d, 10 * sizeof(double)) != cudaSuccess) return false;
    probeKernel<<<1, 1>>>(m.tcDepth, tc, m.rc, m.rcDepth, m.rcSim, pixToleranceFactor, pixSizeBall, pixSizeBallWSP, x, y, d);
    const bool ok = cudaMemcpy(out, d, 10 * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(d);
    return ok;
}

bool GroupFilter::voteCount(long long* count)
{
    Impl& m = *impl_;
    const size_t n = size_t(m.rc.w) * m.rc.h;
    std::vector<int> h(n);
    if (cudaMemcpy(h.data(), m.pts, n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    long long c = 0;
    for (size_t i = 0; i < n; ++i) c += h[i] > 0;
    *count = c;
    return true;
}

bool GroupFilter::result(unsigned char* numOfModals)
{
    Impl& m = *impl_;
    return cudaMemcpy(numOfModals, m.modals, size_t(m.rc.w) * m.rc.h, cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace gpu
}  // namespace fuseCut
}  // namespace aliceVision
