// Cheshire: Texturing's per-camera work on the GPU (see texturingGPU.hpp).
//
// Transcribed, in order, from: OpenImageIO 3.0.9 imagebufalgo_xform.cpp resize_() (separable
// path), filter.cpp FilterGaussian1D/2D, fmath.h fast_exp/fast_exp2 (the polynomial, madd = a*b+c
// as MSVC builds it: no FMA); aliceVision imageAlgo::imageDiff / laplacianPyramid,
// image::getInterpolateColor, Texturing.cpp's rasterisation loop, geogram's
// point_triangle_squared_distance (geometry_nd.h). FMA contraction is off so every product and
// sum rounds where the CPU's does.
#ifdef __clang__
#pragma clang fp contract(off)
#endif
#include "texturingGPU.hpp"
#include <cuda_runtime.h>
#include <aliceVision/depthMap/cuda/hip/cheshire/devalloc.h>  // cheshire: bridge on both backends
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace aliceVision {
namespace mesh {
namespace gpu {

namespace {

constexpr int kBlock = 64;      // threads per triangle
constexpr int kMaxBands = 4;

struct Level { const float* rgb; int w, h; };

// ---------------------------------------------------------------- OpenImageIO pieces

__device__ __forceinline__ float oiioClamp(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

// fmath.h fast_exp2 (float), the non-SIMD scalar path
__device__ __forceinline__ float oiioFastExp2(float xval)
{
    float x = oiioClamp(xval, -126.0f, 126.0f);
    int m = int(x); x -= float(m);
    x = 1.0f - (1.0f - x);
    float r = 1.33336498402e-3f;
    r = x * r + 9.810352697968e-3f;
    r = x * r + 5.551834031939e-2f;
    r = x * r + 0.2401793301105f;
    r = x * r + 0.693144857883f;
    r = x * r + 1.0f;
    return __uint_as_float(__float_as_uint(r) + (unsigned(m) << 23));
}
__device__ __forceinline__ float oiioFastExp(float x) { return oiioFastExp2(x * float(1.0 / 0.69314718055994530942)); }

// FilterGaussian1D::gauss1d; the 2D filter's xfilt/yfilt pass x * (2 / width), width 3
__device__ __forceinline__ float gauss1d(float x)
{
    x = fabsf(x);
    return (x < 1.0f) ? oiioFastExp(-2.0f * (x * x)) : 0.0f;
}
__device__ __forceinline__ float gaussFilt(float x) { return gauss1d(x * (2.0f / 3.0f)); }

// resize_(): one thread per destination pixel, separable gaussian, WrapClamp source reads.
// srcW/srcH -> dstW/dstH, RGB interleaved. Weights recomputed per thread exactly as the
// per-column / per-row precomputation does.
__global__ void resizeGaussianKernel(const float* __restrict__ src, int srcW, int srcH, float* __restrict__ dst, int dstW, int dstH)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= dstW || y >= dstH) return;
    const float srcfw = float(srcW), srcfh = float(srcH);
    const float xratio = float(dstW) / srcfw, yratio = float(dstH) / srcfh;
    const float dstpixelwidth = 1.0f / float(dstW), dstpixelheight = 1.0f / float(dstH);
    const float filterrad = 3.0f / 2.0f;   // gaussian width 3 (fd.width * max(1, ratio) with ratio < 1)
    const int radi = (int)ceilf(filterrad / xratio), radj = (int)ceilf(filterrad / yratio);
    const int xtaps = 2 * radi + 1, ytaps = 2 * radj + 1;
    float xf[64], yf[64];   // radi <= 31 covers every downscale this node uses (4: 13 taps)
    if (xtaps > 64 || ytaps > 64) return;

    // column weights
    const float s = (float(x) - 0.0f + 0.5f) * dstpixelwidth;
    const float src_xf = 0.0f + s * srcfw;
    const float fx = floorf(src_xf);
    const int src_x = int(fx);
    const float src_xf_frac = src_xf - fx;
    float totalweight_x = 0.0f;
    for (int i = 0; i < xtaps; ++i) {
        const float w = gaussFilt(xratio * (float(i - radi) - (src_xf_frac - 0.5f)));
        xf[i] = w; totalweight_x += w;
    }
    if (totalweight_x != 0.0f) for (int i = 0; i < xtaps; ++i) xf[i] /= totalweight_x;
    // row weights
    const float t = (float(y) - 0.0f + 0.5f) * dstpixelheight;
    const float src_yf = 0.0f + t * srcfh;
    const float fy = floorf(src_yf);
    const int src_y = int(fy);
    const float src_yf_frac = src_yf - fy;
    float totalweight_y = 0.0f;
    for (int j = 0; j < ytaps; ++j) {
        const float w = gaussFilt(yratio * (float(j - radj) - (src_yf_frac - 0.5f)));
        yf[j] = w; totalweight_y += w;
    }
    if (totalweight_y != 0.0f) for (int j = 0; j < ytaps; ++j) yf[j] /= totalweight_y;

    float pel[3] = {0.0f, 0.0f, 0.0f};
    float tw = 0.0f;
    for (int i = 0; i < xtaps; ++i) tw += xf[i];
    if (tw != 0.0f) {
        for (int j = -radj; j <= radj; ++j) {
            const float wy = yf[j + radj];
            if (wy == 0.0f) continue;
            int sy = src_y + j; sy = sy < 0 ? 0 : (sy >= srcH ? srcH - 1 : sy);
            for (int i = 0; i < xtaps; ++i) {
                const float w = wy * xf[i];
                if (w != 0.0f) {
                    int sx = src_x - radi + i; sx = sx < 0 ? 0 : (sx >= srcW ? srcW - 1 : sx);
                    const float* p = src + (size_t(sy) * srcW + sx) * 3;
                    pel[0] += w * p[0]; pel[1] += w * p[1]; pel[2] += w * p[2];
                }
            }
        }
    }
    float* o = dst + (size_t(y) * dstW + x) * 3;
    if (totalweight_y == 0.0f) { o[0] = 0.0f; o[1] = 0.0f; o[2] = 0.0f; }
    else { o[0] = pel[0]; o[1] = pel[1]; o[2] = pel[2]; }
}

// ---------------------------------------------------------------- aliceVision pieces

// image::getInterpolateColor: clamped top-left pixel, bilinear in float with the CPU's conversions
__device__ __forceinline__ void interp(const float* __restrict__ img, int w, int h, double y, double x, float out[3])
{
    const int xp = min(int(x), w - 2);
    const int yp = min(int(y), h - 2);
    const float ui = float(x - double(float(xp)));
    const float vi = float(y - double(float(yp)));
    const float* lu = img + (size_t(yp) * w + xp) * 3;
    const float* ru = lu + 3;
    const float* ld = lu + size_t(w) * 3;
    const float* rd = ld + 3;
    for (int c = 0; c < 3; ++c) {
        const float u = lu[c] + (ru[c] - lu[c]) * ui;
        const float d = ld[c] + (rd[c] - ld[c]) * ui;
        out[c] = u + (d - u) * vi;
    }
}

// imageAlgo::imageDiff: out = in - interp(down, iy / downscale, ix / downscale) with integer division
__global__ void imageDiffKernel(const float* __restrict__ in, int w, int h, const float* __restrict__ down, int dw, int dh, unsigned downscale, float* __restrict__ out)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x, iy = blockIdx.y;
    if (ix >= w || iy >= h) return;
    float c[3];
    interp(down, dw, dh, double(unsigned(iy) / downscale), double(unsigned(ix) / downscale), c);
    const size_t o = (size_t(iy) * w + ix) * 3;
    out[o] = in[o] - c[0]; out[o + 1] = in[o + 1] - c[1]; out[o + 2] = in[o + 2] - c[2];
}

// geogram point_segment_squared_distance / point_triangle_squared_distance, 2D, double
struct V2 { double x, y; };
__device__ __forceinline__ V2 sub2(V2 a, V2 b) { return {a.x - b.x, a.y - b.y}; }
__device__ __forceinline__ double dot2(V2 a, V2 b) { return a.x * b.x + a.y * b.y; }
__device__ __forceinline__ double length2(V2 a) { return dot2(a, a); }
__device__ __forceinline__ double distance2(V2 a, V2 b) { return length2(sub2(a, b)); }

__device__ double pointSegmentSquaredDistance(V2 point, V2 V0, V2 V1, double& lambda0, double& lambda1)
{
    const double l2 = distance2(V0, V1);
    const double t = dot2(sub2(point, V0), sub2(V1, V0));
    if (t <= 0.0 || l2 == 0.0) { lambda0 = 1.0; lambda1 = 0.0; return distance2(point, V0); }
    else if (t > l2) { lambda0 = 0.0; lambda1 = 1.0; return distance2(point, V1); }
    lambda1 = t / l2;
    lambda0 = 1.0 - lambda1;
    const V2 closest = {lambda0 * V0.x + lambda1 * V1.x, lambda0 * V0.y + lambda1 * V1.y};
    return distance2(point, closest);
}

__device__ double pointTriangleSquaredDistance(V2 point, V2 V0, V2 V1, V2 V2_, double& lambda0, double& lambda1, double& lambda2)
{
    const V2 diff = sub2(V0, point), edge0 = sub2(V1, V0), edge1 = sub2(V2_, V0);
    const double a00 = length2(edge0), a01 = dot2(edge0, edge1), a11 = length2(edge1);
    const double b0 = dot2(diff, edge0), b1 = dot2(diff, edge1), c = length2(diff);
    const double det = fabs(a00 * a11 - a01 * a01);
    double s = a01 * b1 - a11 * b0;
    double t = a01 * b0 - a00 * b1;
    double sqrDistance;

    if (det < 1e-30) {
        double cur_l1, cur_l2;
        double result = pointSegmentSquaredDistance(point, V0, V1, cur_l1, cur_l2);
        lambda0 = cur_l1; lambda1 = cur_l2; lambda2 = 0.0;
        double cur_dist = pointSegmentSquaredDistance(point, V0, V2_, cur_l1, cur_l2);
        if (cur_dist < result) { result = cur_dist; lambda0 = cur_l1; lambda2 = cur_l2; lambda1 = 0.0; }
        cur_dist = pointSegmentSquaredDistance(point, V1, V2_, cur_l1, cur_l2);
        if (cur_dist < result) { result = cur_dist; lambda1 = cur_l1; lambda2 = cur_l2; lambda0 = 0.0; }
        return result;
    }

    if (s + t <= det) {
        if (s < 0.0) {
            if (t < 0.0) {   // region 4
                if (b0 < 0.0) {
                    t = 0.0;
                    if (-b0 >= a00) { s = 1.0; sqrDistance = a00 + 2.0 * b0 + c; }
                    else { s = -b0 / a00; sqrDistance = b0 * s + c; }
                } else {
                    s = 0.0;
                    if (b1 >= 0.0) { t = 0.0; sqrDistance = c; }
                    else if (-b1 >= a11) { t = 1.0; sqrDistance = a11 + 2.0 * b1 + c; }
                    else { t = -b1 / a11; sqrDistance = b1 * t + c; }
                }
            } else {   // region 3
                s = 0.0;
                if (b1 >= 0.0) { t = 0.0; sqrDistance = c; }
                else if (-b1 >= a11) { t = 1.0; sqrDistance = a11 + 2.0 * b1 + c; }
                else { t = -b1 / a11; sqrDistance = b1 * t + c; }
            }
        } else if (t < 0.0) {   // region 5
            t = 0.0;
            if (b0 >= 0.0) { s = 0.0; sqrDistance = c; }
            else if (-b0 >= a00) { s = 1.0; sqrDistance = a00 + 2.0 * b0 + c; }
            else { s = -b0 / a00; sqrDistance = b0 * s + c; }
        } else {   // region 0
            const double invDet = double(1.0) / det;
            s *= invDet;
            t *= invDet;
            sqrDistance = s * (a00 * s + a01 * t + 2.0 * b0) + t * (a01 * s + a11 * t + 2.0 * b1) + c;
        }
    } else {
        double tmp0, tmp1, numer, denom;
        if (s < 0.0) {   // region 2
            tmp0 = a01 + b0; tmp1 = a11 + b1;
            if (tmp1 > tmp0) {
                numer = tmp1 - tmp0; denom = a00 - 2.0 * a01 + a11;
                if (numer >= denom) { s = 1.0; t = 0.0; sqrDistance = a00 + 2.0 * b0 + c; }
                else { s = numer / denom; t = 1.0 - s; sqrDistance = s * (a00 * s + a01 * t + 2.0 * b0) + t * (a01 * s + a11 * t + 2.0 * b1) + c; }
            } else {
                s = 0.0;
                if (tmp1 <= 0.0) { t = 1.0; sqrDistance = a11 + 2.0 * b1 + c; }
                else if (b1 >= 0.0) { t = 0.0; sqrDistance = c; }
                else { t = -b1 / a11; sqrDistance = b1 * t + c; }
            }
        } else if (t < 0.0) {   // region 6
            tmp0 = a01 + b1; tmp1 = a00 + b0;
            if (tmp1 > tmp0) {
                numer = tmp1 - tmp0; denom = a00 - 2.0 * a01 + a11;
                if (numer >= denom) { t = 1.0; s = 0.0; sqrDistance = a11 + 2.0 * b1 + c; }
                else { t = numer / denom; s = 1.0 - t; sqrDistance = s * (a00 * s + a01 * t + 2.0 * b0) + t * (a01 * s + a11 * t + 2.0 * b1) + c; }
            } else {
                t = 0.0;
                if (tmp1 <= 0.0) { s = 1.0; sqrDistance = a00 + 2.0 * b0 + c; }
                else if (b0 >= 0.0) { s = 0.0; sqrDistance = c; }
                else { s = -b0 / a00; sqrDistance = b0 * s + c; }
            }
        } else {   // region 1
            numer = a11 + b1 - a01 - b0;
            if (numer <= 0.0) { s = 0.0; t = 1.0; sqrDistance = a11 + 2.0 * b1 + c; }
            else {
                denom = a00 - 2.0 * a01 + a11;
                if (numer >= denom) { s = 1.0; t = 0.0; sqrDistance = a00 + 2.0 * b0 + c; }
                else { s = numer / denom; t = 1.0 - s; sqrDistance = s * (a00 * s + a01 * t + 2.0 * b0) + t * (a01 * s + a11 * t + 2.0 * b1) + c; }
            }
        }
    }
    if (sqrDistance < 0.0) sqrDistance = 0.0;
    lambda0 = 1.0 - s - t;
    lambda1 = s;
    lambda2 = t;
    return sqrDistance;
}

struct Cam { const float* img; int w, h; double P[12]; Level level[kMaxBands]; int nbBand; };

// Texturing::generateTexturesSubSet's per-triangle loop: one block per triangle, threads over the
// bounding box; float atomics into the slot's per-band accumulators (rgb: 3 per texel, cnt: 1).
__global__ void rasterKernel(const double* __restrict__ triPts, const double* __restrict__ triPix, const uint32_t* __restrict__ triIds,
                             const float* __restrict__ scores, uint32_t n, int band, unsigned texSide, Cam cam,
                             float* __restrict__ accRgb, float* __restrict__ accCnt, size_t levelStride)
{
    const uint32_t li = blockIdx.x;
    if (li >= n) return;
    const uint32_t tri = triIds[li];
    const float triangleScore = scores[li];
    const double* pp = triPts + size_t(tri) * 9;
    const double* px = triPix + size_t(tri) * 6;
    const V2 T0 = {px[0], px[1]}, T1 = {px[2], px[3]}, T2 = {px[4], px[5]};

    // bounding box in texture pixels, clamped to [0, texSide]
    const int texSideI = int(texSide);
    int LUx = int(floor(fmin(fmin(T0.x, T1.x), T2.x))), LUy = int(floor(fmin(fmin(T0.y, T1.y), T2.y)));
    int RDx = int(ceil(fmax(fmax(T0.x, T1.x), T2.x))), RDy = int(ceil(fmax(fmax(T0.y, T1.y), T2.y)));
    LUx = min(max(LUx, 0), texSideI); LUy = min(max(LUy, 0), texSideI);
    RDx = min(max(RDx, 0), texSideI); RDy = min(max(RDy, 0), texSideI);
    const int bw = RDx - LUx, bh = RDy - LUy;
    if (bw <= 0 || bh <= 0) return;
    const int npix = bw * bh;

    for (int k = threadIdx.x; k < npix; k += blockDim.x) {
        const int x = LUx + k % bw, y = LUy + k / bw;
        // isPixelInTriangle: pixel centre, geogram distance, margin 0.5
        const V2 pc = {double(x) + 0.5, double(y) + 0.5};
        double l1, l2, l3;
        const double dist = pointTriangleSquaredDistance(pc, T0, T1, T2, l1, l2, l3);
        if (!(dist < 0.5 + DBL_EPSILON)) continue;
        const double bx = l3, by = l2;   // barycentricCoords.x = l3, .y = l2
        const unsigned y_ = (texSide - 1) - unsigned(y);
        const size_t off = size_t(y_) * texSide + unsigned(x);
        // barycentricToCartesian: t0 + (t2 - t0) * bx + (t1 - t0) * by
        double pt[3];
        for (int c = 0; c < 3; ++c) pt[c] = pp[c] + (pp[6 + c] - pp[c]) * bx + (pp[3 + c] - pp[c]) * by;
        // getPixelFor3DPoint: P * X, (-1, -1) behind the camera
        const double XTz = cam.P[8] * pt[0] + cam.P[9] * pt[1] + cam.P[10] * pt[2] + cam.P[11];
        double rx, ry;
        if (XTz <= 0) { rx = -1.0; ry = -1.0; }
        else {
            const double XTx = cam.P[0] * pt[0] + cam.P[1] * pt[1] + cam.P[2] * pt[2] + cam.P[3];
            const double XTy = cam.P[4] * pt[0] + cam.P[5] * pt[1] + cam.P[6] * pt[2] + cam.P[7];
            rx = XTx / XTz; ry = XTy / XTz;
        }
        // isPixelInImage(Point2d): Pixel(floor(p + 0.5)), margin g_border = 2
        const int pxi = int(floor(rx + 0.5)), pyi = int(floor(ry + 0.5));
        if (!(pxi >= 2 && pxi < cam.w - 2 && pyi >= 2 && pyi < cam.h - 2)) continue;
        // pure zero in the source image: no contribution
        float col[3];
        interp(cam.img, cam.w, cam.h, ry, rx, col);
        if (col[0] == 0.0f && col[1] == 0.0f && col[2] == 0.0f) continue;
        // each band contributes to itself and every lower frequency (higher level index)
        int coef = 1;
        for (int b = 0; b < band; ++b) coef *= 4;   // std::pow(multiBandDownscale, bandContrib) as int
        for (int bc = band; bc < cam.nbBand; ++bc) {
            const Level& L = cam.level[bc];
            interp(L.rgb, L.w, L.h, ry / double(coef), rx / double(coef), col);
            float* r = accRgb + size_t(bc) * levelStride * 3 + off * 3;
            atomicAdd(r, col[0] * triangleScore);
            atomicAdd(r + 1, col[1] * triangleScore);
            atomicAdd(r + 2, col[2] * triangleScore);
            atomicAdd(accCnt + size_t(bc) * levelStride + off, triangleScore);
            coef *= 4;
        }
    }
}

// "Computing final (average) color" and "Fuse frequency bands": level 0 in place
__global__ void finishKernel(float* __restrict__ accRgb, float* __restrict__ accCnt, size_t n, int nbBand, size_t levelStride)
{
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* r0 = accRgb + i * 3;
    const float c0 = accCnt[i];
    if (c0 != 0.0f) {
        r0[0] /= c0; r0[1] /= c0; r0[2] /= c0;
        accCnt[i] = 1.0f;
        for (int l = 1; l < nbBand; ++l) {
            float* rl = accRgb + size_t(l) * levelStride * 3 + i * 3;
            const float cl = accCnt[size_t(l) * levelStride + i];
            rl[0] /= cl; rl[1] /= cl; rl[2] /= cl;
        }
    }
    for (int l = 1; l < nbBand; ++l) {
        const float* rl = accRgb + size_t(l) * levelStride * 3 + i * 3;
        r0[0] += rl[0]; r0[1] += rl[1]; r0[2] += rl[2];
    }
}

// ---- edge padding: upstream's two sweeps as wavefronts ----------------------------------------
//
// Texturing::writeTexture dilates each chart's gutter with two sequential raster sweeps over the
// atlas: up-left to bottom-right, where a texel reads the left and up neighbours AS ALREADY
// UPDATED in this sweep, then bottom-right to up-left reading right and down likewise (and left and
// up as they stood before this sweep). The dependency of texel (x, y) on (x-1, y) and (x, y-1)
// makes every anti-diagonal x + y = d independent once diagonal d-1 is done, so the sweep is
// 2*side launches of one diagonal each, and the same for the reverse sweep in reversed
// coordinates. Same reads, same writes, same order along every dependency chain: the result is the
// sequential algorithm's, texel for texel, which CHESHIRE_GPU_PAD_CHECK=1 verifies on the host.
// count values are upstream's: 1 valid, -k padded k texels deep, 0 untouched.

__global__ void padInitKernel(const float* __restrict__ cnt, int* __restrict__ pc, size_t n)
{
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) pc[i] = cnt[i] > 0.0f ? 1 : 0;
}

__global__ void padForwardKernel(float* __restrict__ rgb, int* __restrict__ pc, int S, int d, int xlo, int nx, int padding)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx) return;
    const int x = xlo + t, y = d - x;
    const size_t o = size_t(y) * S + x;
    if (pc[o] > 0) return;
    const int upCount = pc[o - S];
    const int leftCount = pc[o - 1];
    size_t src; int val;
    if (leftCount > 0) { src = o - 1; val = -1; }
    else if (upCount > 0) { src = o - S; val = -1; }
    else if (leftCount < 0 && -leftCount < padding && (upCount == 0 || leftCount > upCount)) { src = o - 1; val = leftCount - 1; }
    else if (upCount < 0 && -upCount < padding) { src = o - S; val = upCount - 1; }
    else return;
    rgb[o * 3] = rgb[src * 3]; rgb[o * 3 + 1] = rgb[src * 3 + 1]; rgb[o * 3 + 2] = rgb[src * 3 + 2];
    pc[o] = val;
}

__global__ void padBackwardKernel(float* __restrict__ rgb, int* __restrict__ pc, int S, int d, int ulo, int nu, int padding)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nu) return;
    const int u = ulo + t, v = d - u;          // reversed coordinates: u = S-1-x, v = S-1-y
    const int x = S - 1 - u, y = S - 1 - v;
    const size_t o = size_t(y) * S + x;
    if (pc[o] > 0) return;
    const int upCount = pc[o - S];
    const int downCount = pc[o + S];
    const int rightCount = pc[o + 1];
    const int leftCount = pc[o - 1];
    size_t src; int val;
    if (rightCount > 0) { src = o + 1; val = -1; }
    else if (downCount > 0) { src = o + S; val = -1; }
    else if ((rightCount < 0 && -rightCount < padding) && (leftCount == 0 || rightCount > leftCount) && (downCount == 0 || rightCount >= downCount)) { src = o + 1; val = rightCount - 1; }
    else if ((downCount < 0 && -downCount < padding) && (upCount == 0 || downCount > upCount)) { src = o + S; val = downCount - 1; }
    else return;
    rgb[o * 3] = rgb[src * 3]; rgb[o * 3 + 1] = rgb[src * 3 + 1]; rgb[o * 3 + 2] = rgb[src * 3 + 2];
    pc[o] = val;
}

__global__ void padCountOutKernel(const int* __restrict__ pc, float* __restrict__ cnt, size_t n)
{
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) cnt[i] = float(pc[i]);
}

// Upstream's sweeps verbatim, on the host, for CHESHIRE_GPU_PAD_CHECK.
static void padHostReference(std::vector<float>& rgb, std::vector<int>& pc, int S, int padding)
{
    auto copy = [&](size_t dst, size_t src) { rgb[dst * 3] = rgb[src * 3]; rgb[dst * 3 + 1] = rgb[src * 3 + 1]; rgb[dst * 3 + 2] = rgb[src * 3 + 2]; };
    for (int y = 1; y < S - 1; ++y)
        for (int x = 1; x < S - 1; ++x) {
            const size_t o = size_t(y) * S + x;
            if (pc[o] > 0) continue;
            const int upCount = pc[o - S], leftCount = pc[o - 1];
            if (leftCount > 0) { copy(o, o - 1); pc[o] = -1; }
            else if (upCount > 0) { copy(o, o - S); pc[o] = -1; }
            else if (leftCount < 0 && -leftCount < padding && (upCount == 0 || leftCount > upCount)) { copy(o, o - 1); pc[o] = leftCount - 1; }
            else if (upCount < 0 && -upCount < padding) { copy(o, o - S); pc[o] = upCount - 1; }
        }
    for (int y = 1; y < S - 1; ++y)
        for (int x = 1; x < S - 1; ++x) {
            const size_t o = size_t(S - 1 - y) * S + (S - 1 - x);
            if (pc[o] > 0) continue;
            const int upCount = pc[o - S], downCount = pc[o + S], rightCount = pc[o + 1], leftCount = pc[o - 1];
            if (rightCount > 0) { copy(o, o + 1); pc[o] = -1; }
            else if (downCount > 0) { copy(o, o + S); pc[o] = -1; }
            else if ((rightCount < 0 && -rightCount < padding) && (leftCount == 0 || rightCount > leftCount) && (downCount == 0 || rightCount >= downCount)) { copy(o, o + 1); pc[o] = rightCount - 1; }
            else if ((downCount < 0 && -downCount < padding) && (upCount == 0 || downCount > upCount)) { copy(o, o + S); pc[o] = downCount - 1; }
        }
}

bool g_checked = false, g_available = false;
std::mutex g_mutex;

}  // namespace

bool texAvailable()
{
    std::lock_guard<std::mutex> g(g_mutex);
    if (g_checked) return g_available;
    g_checked = true;
    if (const char* e = std::getenv("CHESHIRE_GPU_TEX")) if (e[0] == '0') { std::fprintf(stderr, "[cheshire] texturing: disabled by CHESHIRE_GPU_TEX=0, CPU\n"); return g_available = false; }
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { std::fprintf(stderr, "[cheshire] texturing: no GPU device, CPU\n"); return g_available = false; }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return g_available = false;
    std::fprintf(stderr, "[cheshire] texturing: pyramid + rasterisation on %s (CHESHIRE_GPU_TEX=0 for the CPU pass)\n", p.name);
    return g_available = true;
}

std::size_t atlasBytes(unsigned textureSide, int nbBand)
{
    return std::size_t(textureSide) * textureSide * std::size_t(nbBand) * 16;
}

int maxAtlasSlots(unsigned textureSide, int nbBand, int imgW, int imgH, std::uint32_t nbTris)
{
    if (!texAvailable()) return 0;
    size_t freeB = 0, totalB = 0;
    if (cudaMemGetInfo(&freeB, &totalB) != cudaSuccess) return 0;
    // image + pyramid (levels + downscaled scratch, < 2 images) + triangle tables + margin
    const size_t imgB = size_t(imgW) * imgH * 12;
    const size_t need = imgB * 3 + size_t(nbTris) * 15 * 8 + (512u << 20);
    if (freeB <= need) return 0;
    const size_t slots = (freeB - need) / atlasBytes(textureSide, nbBand);
    return int(slots > 64 ? 64 : slots);
}

struct Texturer::Impl {
    int nbSlots = 0, nbBand = 0, downscale = 4, maxW = 0, maxH = 0;
    unsigned texSide = 0;
    size_t levelStride = 0;              // texels per level
    float* accRgb = nullptr;             // [slot][level][texel][3]
    float* accCnt = nullptr;             // [slot][level][texel]
    double* triPts = nullptr; double* triPix = nullptr; uint32_t nbTris = 0;
    float* img = nullptr;                // current camera image
    std::vector<float*> levels;          // Laplacian levels (nbBand), level i has dims of image / downscale^i
    std::vector<float*> down;            // downscaled scratch per level (nbBand - 1)
    uint32_t* dTri = nullptr; float* dScore = nullptr; uint32_t listCap = 0;
    int* padCnt = nullptr;               // edge padding state, one int per texel of one level
    Cam cam{};
    bool camSet = false;
    std::vector<float> hostTmp;

    ~Impl() {
        for (void* p : {(void*)accRgb, (void*)accCnt, (void*)triPts, (void*)triPix, (void*)img, (void*)dTri, (void*)dScore, (void*)padCnt}) if (p) cheshire::devFree(p);
        for (float* p : levels) if (p) cheshire::devFree(p);
        for (float* p : down) if (p) cheshire::devFree(p);
    }
    static int levelDim(int d, int downscale, int level) { for (int i = 0; i < level; ++i) d /= downscale; return d; }
    size_t slotRgb(int slot) const { return size_t(slot) * nbBand * levelStride * 3; }
    size_t slotCnt(int slot) const { return size_t(slot) * nbBand * levelStride; }
};

Texturer::Texturer() : p(new Impl) {}
Texturer::~Texturer() { delete p; }

static double secsSince(std::chrono::steady_clock::time_point a) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count(); }

bool Texturer::init(int nbSlots, unsigned textureSide, int nbBand, int downscale, int maxImgW, int maxImgH)
{
    if (nbSlots < 1 || nbBand < 1 || nbBand > kMaxBands) return false;
    p->nbSlots = nbSlots; p->texSide = textureSide; p->nbBand = nbBand; p->downscale = downscale; p->maxW = maxImgW; p->maxH = maxImgH;
    p->levelStride = size_t(textureSide) * textureSide;
    const size_t rgbN = size_t(nbSlots) * nbBand * p->levelStride * 3, cntN = size_t(nbSlots) * nbBand * p->levelStride;
    if (cheshire::devMalloc((void**)&p->accRgb, rgbN * sizeof(float)) != cudaSuccess) return false;
    if (cheshire::devMalloc((void**)&p->accCnt, cntN * sizeof(float)) != cudaSuccess) return false;
    if (cudaMemset(p->accRgb, 0, rgbN * sizeof(float)) != cudaSuccess || cudaMemset(p->accCnt, 0, cntN * sizeof(float)) != cudaSuccess) return false;
    if (cheshire::devMalloc((void**)&p->img, size_t(maxImgW) * maxImgH * 12) != cudaSuccess) return false;
    p->levels.assign(nbBand, nullptr); p->down.assign(nbBand > 1 ? nbBand - 1 : 0, nullptr);
    for (int l = 0; l < nbBand; ++l) {
        const size_t n = size_t(Impl::levelDim(maxImgW, downscale, l)) * Impl::levelDim(maxImgH, downscale, l) * 12;
        if (cheshire::devMalloc((void**)&p->levels[l], n) != cudaSuccess) return false;
        if (l + 1 < nbBand && cheshire::devMalloc((void**)&p->down[l], size_t(Impl::levelDim(maxImgW, downscale, l + 1)) * Impl::levelDim(maxImgH, downscale, l + 1) * 12) != cudaSuccess) return false;
    }
    return true;
}

bool Texturer::setTriangles(std::uint32_t nbTris, const double* pts, const double* texPix)
{
    p->nbTris = nbTris;
    if (cheshire::devMalloc((void**)&p->triPts, size_t(nbTris) * 9 * sizeof(double)) != cudaSuccess) return false;
    if (cheshire::devMalloc((void**)&p->triPix, size_t(nbTris) * 6 * sizeof(double)) != cudaSuccess) return false;
    return cudaMemcpy(p->triPts, pts, size_t(nbTris) * 9 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess
        && cudaMemcpy(p->triPix, texPix, size_t(nbTris) * 6 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess;
}

bool Texturer::setCamera(const float* rgb, int w, int h, const double* P)
{
    if (w > p->maxW || h > p->maxH) return false;
    const auto t0 = std::chrono::steady_clock::now();
    if (cudaMemcpy(p->img, rgb, size_t(w) * h * 12, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    cudaDeviceSynchronize();
    uploadSec += secsSince(t0);
    const auto t1 = std::chrono::steady_clock::now();
    // laplacianPyramid: img -> (down, diff) per band, the last level is the last downscaled image
    const float* cur = p->img; int cw = w, ch = h;
    for (int b = 0; b < p->nbBand; ++b) {
        p->cam.level[b].w = cw; p->cam.level[b].h = ch;
        if (b + 1 < p->nbBand) {
            const int dw = cw / p->downscale, dh = ch / p->downscale;
            resizeGaussianKernel<<<dim3((dw + 127) / 128, dh), 128>>>(cur, cw, ch, p->down[b], dw, dh);
            imageDiffKernel<<<dim3((cw + 127) / 128, ch), 128>>>(cur, cw, ch, p->down[b], dw, dh, unsigned(p->downscale), p->levels[b]);
            p->cam.level[b].rgb = p->levels[b];
            cur = p->down[b]; cw = dw; ch = dh;
        } else {
            p->cam.level[b].rgb = cur;   // the last downscaled image itself
        }
    }
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return false;
    pyramidSec += secsSince(t1);
    p->cam.img = p->img; p->cam.w = w; p->cam.h = h; p->cam.nbBand = p->nbBand;
    for (int i = 0; i < 12; ++i) p->cam.P[i] = P[i];
    p->camSet = true;
    return true;
}

bool Texturer::raster(int slot, int band, const std::uint32_t* triIds, const float* scores, std::uint32_t n)
{
    if (!p->camSet || slot < 0 || slot >= p->nbSlots || band < 0 || band >= p->nbBand) return false;
    if (n == 0) return true;
    const auto t0 = std::chrono::steady_clock::now();
    if (n > p->listCap) {
        if (p->dTri) cheshire::devFree(p->dTri); if (p->dScore) cheshire::devFree(p->dScore);
        p->listCap = n < 65536 ? 65536 : n;
        if (cheshire::devMalloc((void**)&p->dTri, size_t(p->listCap) * 4) != cudaSuccess || cheshire::devMalloc((void**)&p->dScore, size_t(p->listCap) * 4) != cudaSuccess) return false;
    }
    if (cudaMemcpy(p->dTri, triIds, size_t(n) * 4, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (cudaMemcpy(p->dScore, scores, size_t(n) * 4, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    rasterKernel<<<n, kBlock>>>(p->triPts, p->triPix, p->dTri, p->dScore, n, band, p->texSide, p->cam,
                                p->accRgb + p->slotRgb(slot), p->accCnt + p->slotCnt(slot), p->levelStride);
    const bool ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    rasterSec += secsSince(t0);
    return ok;
}

bool Texturer::finish(int slot, float* rgb, float* count, int padding)
{
    if (slot < 0 || slot >= p->nbSlots) return false;
    const auto t0 = std::chrono::steady_clock::now();
    float* r = p->accRgb + p->slotRgb(slot); float* c = p->accCnt + p->slotCnt(slot);
    const size_t n = p->levelStride;
    finishKernel<<<unsigned((n + 255) / 256), 256>>>(r, c, n, p->nbBand, p->levelStride);
    bool ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    if (ok && padding > 0) {
        const int S = int(p->texSide);
        static const bool check = std::getenv("CHESHIRE_GPU_PAD_CHECK") != nullptr;
        std::vector<float> refRgb; std::vector<int> refPc;
        if (check) {   // the un-padded atlas, for the host reference below
            refRgb.resize(n * 3); refPc.resize(n);
            std::vector<float> cnt(n);
            ok = cudaMemcpy(refRgb.data(), r, n * 3 * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess
              && cudaMemcpy(cnt.data(), c, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
            for (size_t i = 0; i < n; ++i) refPc[i] = cnt[i] > 0.0f ? 1 : 0;
        }
        if (ok && !p->padCnt) ok = cheshire::devMalloc((void**)&p->padCnt, n * sizeof(int)) == cudaSuccess;
        if (ok) {
            padInitKernel<<<unsigned((n + 255) / 256), 256>>>(c, p->padCnt, n);
            // forward: diagonals d = x + y over 1 <= x, y <= S-2
            for (int d = 2; d <= 2 * (S - 2); ++d) {
                const int xlo = d - (S - 2) > 1 ? d - (S - 2) : 1;
                const int xhi = d - 1 < S - 2 ? d - 1 : S - 2;
                const int nx = xhi - xlo + 1;
                if (nx > 0) padForwardKernel<<<unsigned((nx + 255) / 256), 256>>>(r, p->padCnt, S, d, xlo, nx, padding);
            }
            // backward: the same diagonals in reversed coordinates
            for (int d = 2; d <= 2 * (S - 2); ++d) {
                const int ulo = d - (S - 2) > 1 ? d - (S - 2) : 1;
                const int uhi = d - 1 < S - 2 ? d - 1 : S - 2;
                const int nu = uhi - ulo + 1;
                if (nu > 0) padBackwardKernel<<<unsigned((nu + 255) / 256), 256>>>(r, p->padCnt, S, d, ulo, nu, padding);
            }
            padCountOutKernel<<<unsigned((n + 255) / 256), 256>>>(p->padCnt, c, n);
            ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
        }
        if (ok && check) {
            padHostReference(refRgb, refPc, S, padding);
            std::vector<float> gotRgb(n * 3), gotCnt(n);
            ok = cudaMemcpy(gotRgb.data(), r, n * 3 * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess
              && cudaMemcpy(gotCnt.data(), c, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
            size_t bad = 0;
            for (size_t i = 0; ok && i < n; ++i)
                if (gotCnt[i] != float(refPc[i]) || gotRgb[i * 3] != refRgb[i * 3] || gotRgb[i * 3 + 1] != refRgb[i * 3 + 1] || gotRgb[i * 3 + 2] != refRgb[i * 3 + 2]) ++bad;
            std::fprintf(stderr, "[cheshire] GPU padding check: texels differing from the sequential sweeps: %zu of %zu\n", bad, n);
        }
    }
    if (ok) ok = cudaMemcpy(rgb, r, n * 3 * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess
              && cudaMemcpy(count, c, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    if (ok) ok = cudaMemset(r, 0, size_t(p->nbBand) * n * 3 * sizeof(float)) == cudaSuccess && cudaMemset(c, 0, size_t(p->nbBand) * n * sizeof(float)) == cudaSuccess;
    finishSec += secsSince(t0);
    return ok;
}

}  // namespace gpu
}  // namespace mesh
}  // namespace aliceVision
