// cheshire: the colour optimisation's angles without double precision (device/matrix.cuh, angleBetwABandAC).
//
// Upstream computes an angle in degrees as float(fabs(acos(double(x))) / (CUDART_PI / 180.0)): OCML's double acos and
// a correctly rounded double division, about 65 FP64 instructions, twice per pixel and iteration of the colour
// optimisation, which runs at the card's FP64 rate.
//
// cheshireAngleDegrees computes acos(x) * 180/pi in float-float arithmetic from IEEE operations only (adds, multiplies,
// FMAs, a correctly rounded sqrtf and division): no hardware approximation, so the same bits on every card. It
// evaluates OCML's own asin polynomial (its double coefficients split into float pairs): asin(s) = s + s z P(z), with
// (s, z) = (x, x^2) for |x| < 0.5, where acos = pi/2 - asin(x), and (sqrt(t), t), t = (1 - |x|)/2, otherwise, where
// acos = 2 asin(s) or pi - 2 asin(s); one path for both, no branch. P's terms from z^6 on are summed in float, the rest
// in float-float. The estimate g = g.hi + g.lo is returned as g.hi only when every value within 2^-40 g.hi of it rounds
// to g.hi; x = -1 and 1 give 180 and +0, the values OCML's special cases make of them; otherwise - and for |x| > 1 and
// NaN - upstream's double path runs. The reference is within a few double ulps of the true value on any card (OpenCL's
// bound for a double acos is 4, so two cards' references differ by under 2^-49 relative), and the estimate's error
// against it, measured below over every input, is far inside the margin: a returned g.hi is the reference's float on
// every card.
//
// This test runs every one of the 2^32 bit patterns through both, compares the bits, measures the estimate's largest
// relative error against the double reference, counts the double path's runs, and times both.
//   build.cmd acos_test && acos_test.exe
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

#define CUDART_PI 3.14159265358979323846  // as cuda_to_hip.h defines it

struct CheshireFF
{
    float hi, lo;
};

__device__ __forceinline__ CheshireFF cheshireFFTwoSum(float a, float b)
{
#pragma clang fp contract(off)
    const float s = a + b;
    const float bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}

__device__ __forceinline__ CheshireFF cheshireFFQuickSum(float a, float b)  // |a| >= |b|, or a == 0
{
#pragma clang fp contract(off)
    const float s = a + b;
    return {s, b - (s - a)};
}

__device__ __forceinline__ CheshireFF cheshireFFAdd(CheshireFF a, CheshireFF b)
{
#pragma clang fp contract(off)
    const CheshireFF s = cheshireFFTwoSum(a.hi, b.hi);
    return cheshireFFQuickSum(s.hi, s.lo + (a.lo + b.lo));
}

__device__ __forceinline__ CheshireFF cheshireFFMul(CheshireFF a, CheshireFF b)
{
#pragma clang fp contract(off)
    const float p = a.hi * b.hi;
    return cheshireFFQuickSum(p, fmaf(a.lo, b.hi, fmaf(a.hi, b.lo, fmaf(a.hi, b.hi, -p))));
}

// true, and the degrees in out, when every value within the margin of the float-float estimate rounds to the same float
__device__ __forceinline__ bool cheshireAngleDegreesFast(float x, float& out, CheshireFF& estimate)
{
#pragma clang fp contract(off)
    const float ax = fabsf(x);
    const bool wide = ax >= 0.5f;
    const float t = (1.0f - ax) * 0.5f;  // exact for |x| in [0.5, 1]
    const float sh = sqrtf(t);           // correctly rounded (HIP's default)
    const float sl = sh > 0.0f ? fmaf(-sh, sh, t) / (sh + sh) : 0.0f;
    const float x2 = x * x;
    const CheshireFF z = wide ? CheshireFF{t, 0.0f} : CheshireFF{x2, fmaf(x, x, -x2)};
    const CheshireFF s = wide ? CheshireFF{sh, sl} : CheshireFF{x, 0.0f};
    // OCML's polynomial for asin(s) = s + s z P(z), z in [0, 1/4]: c11 .. c6 in float, c5 .. c0 in float-float
    float q = fmaf(z.hi, 0x1.05985ap-5f, -0x1.0a5a38p-6f);
    q = fmaf(z.hi, q, 0x1.405214p-6f);
    q = fmaf(z.hi, q, 0x1.ab3a0ap-8f);
    q = fmaf(z.hi, q, 0x1.8ed60ap-7f);
    q = fmaf(z.hi, q, 0x1.c6fa84p-7f);
    CheshireFF p = cheshireFFAdd({0x1.1c6c12p-6f, -0x1.c46692p-31f}, cheshireFFMul(z, {q, 0.0f}));
    p = cheshireFFAdd({0x1.6e89f0p-6f, 0x1.415b5ap-31f}, cheshireFFMul(z, p));
    p = cheshireFFAdd({0x1.f1c72cp-6f, 0x1.9a2590p-32f}, cheshireFFMul(z, p));
    p = cheshireFFAdd({0x1.6db6dcp-5f, -0x1.7c6368p-30f}, cheshireFFMul(z, p));
    p = cheshireFFAdd({0x1.333334p-4f, -0x1.992054p-29f}, cheshireFFMul(z, p));
    p = cheshireFFAdd({0x1.555556p-3f, -0x1.555590p-28f}, cheshireFFMul(z, p));
    const CheshireFF a = cheshireFFAdd(s, cheshireFFMul(s, cheshireFFMul(z, p)));
    // acos: pi/2 - asin(x) near 0; 2 asin(s) for x >= 0.5; pi - 2 asin(s) for x <= -0.5
    const CheshireFF base = wide ? (x > 0.0f ? CheshireFF{0.0f, 0.0f} : CheshireFF{0x1.921fb6p+1f, -0x1.777a5cp-24f})
                                : CheshireFF{0x1.921fb6p+0f, -0x1.777a5cp-25f};
    const float m = wide ? (x > 0.0f ? 2.0f : -2.0f) : -1.0f;
    const CheshireFF r = cheshireFFAdd(base, {m * a.hi, m * a.lo});
    const CheshireFF g = cheshireFFMul(r, {0x1.ca5dc2p+5f, -0x1.670f82p-21f});  // 180 / pi
    estimate = g;
    // the float g.hi stands for every value in (g.hi - below, g.hi + half): half an ulp each way, a quarter below a power
    // of two
    const float half = __uint_as_float(__float_as_uint(g.hi) & 0x7f800000u) * 0x1p-24f;
    const float below = (__float_as_uint(g.hi) & 0x7fffffu) == 0u ? 0.5f * half : half;
    const float margin = g.hi * 0x1p-40f;
    // x = -1 and 1: OCML returns pi (its double) and 0 for them, so the reference is 180 and +0 on every card
    out = x == -1.0f ? 180.0f : (x == 1.0f ? 0.0f : g.hi);
    return ((ax < 1.0f) & (g.lo + margin < half) & (g.lo - margin > -below)) | (ax == 1.0f);
}

__device__ __forceinline__ float upstreamAngleDegrees(float xf)
{
    const double x = double(xf);
    double a = acos(x);
    a = isinf(a) ? 0.0 : a;
    return float(fabs(a) / (CUDART_PI / 180.0));
}

__device__ __forceinline__ float cheshireAngleDegrees(float x)
{
    float d;
    CheshireFF g;
    if (cheshireAngleDegreesFast(x, d, g))
        return d;
    return upstreamAngleDegrees(x);
}

// counters: 0 mismatches, 1 fast, 2 double path within (-1, 1), 3 double path outside (|x| >= 1, NaN)
// maxima (float bits): 0 largest relative error of the estimate against the double reference, near; 1 far
__global__ void exhaustive_kernel(unsigned base, unsigned long long* counts, unsigned* maxErr, unsigned* firstBad)
{
    const unsigned u = base + blockIdx.x * blockDim.x + threadIdx.x;
    const float x = __uint_as_float(u);
    float d;
    CheshireFF g;
    const bool fast = cheshireAngleDegreesFast(x, d, g);
    const double xd = double(x);
    double ad = acos(xd);
    ad = isinf(ad) ? 0.0 : ad;
    const double qref = fabs(ad) / (CUDART_PI / 180.0);
    const float ref = float(qref);
    const float got = fast ? d : upstreamAngleDegrees(x);
    if (__float_as_uint(got) != __float_as_uint(ref))
    {
        atomicAdd(&counts[0], 1ull);
        atomicMin(firstBad, u);
    }
    const bool inside = fabsf(x) < 1.0f;
    atomicAdd(&counts[fast ? 1 : (inside ? 2 : 3)], 1ull);
    if (inside && qref > 0.0)
    {
        const double err = fabs((double(g.hi) + double(g.lo)) - qref) / qref;
        atomicMax(&maxErr[fabsf(x) >= 0.5f ? 1 : 0], __float_as_uint(float(err)));
    }
}

// throughput over x = cos of angles spread over [0, 180] degrees (kind 0) or within 2 degrees of 180, flat surfaces
// seen from a pixel's two neighbours (kind 1)
template <bool TOurs>
__global__ void bench_kernel(float* out, int kind, float seed)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int k = 0; k < 64; ++k)
    {
        const float f = float((t * 64 + k) & 0xffff) * (1.0f / 65536.0f) + seed;
        const float ang = kind == 0 ? f * 3.14159f : 3.14159f - f * 0.0349f;
        const float x = __cosf(ang);
        acc += TOurs ? cheshireAngleDegrees(x) : upstreamAngleDegrees(x);
    }
    out[t] = acc;
}

#define CHECK(c) do { hipError_t e_ = (c); if (e_ != hipSuccess) { std::printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); return 1; } } while (0)

int main()
{
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device: %s (%s)\n", prop.name, prop.gcnArchName);
    unsigned long long* counts;
    unsigned *maxErr, *firstBad;
    CHECK(hipMalloc(&counts, 4 * 8));
    CHECK(hipMalloc(&maxErr, 2 * 4));
    CHECK(hipMalloc(&firstBad, 4));
    CHECK(hipMemset(counts, 0, 4 * 8));
    CHECK(hipMemset(maxErr, 0, 2 * 4));
    const unsigned initBad = 0xffffffffu;
    CHECK(hipMemcpy(firstBad, &initBad, 4, hipMemcpyHostToDevice));
    for (unsigned b = 0; b < 256; ++b)
        exhaustive_kernel<<<1u << 16, 256>>>(b << 24, counts, maxErr, firstBad);
    CHECK(hipDeviceSynchronize());
    unsigned long long c[4];
    unsigned me[2], fb;
    CHECK(hipMemcpy(c, counts, sizeof c, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(me, maxErr, sizeof me, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&fb, firstBad, 4, hipMemcpyDeviceToHost));
    float e0, e1;
    std::memcpy(&e0, &me[0], 4);
    std::memcpy(&e1, &me[1], 4);
    std::printf("all 2^32 bit patterns: %llu mismatches%s", c[0], c[0] ? "" : "\n");
    if (c[0])
        std::printf(", first 0x%08x\n", fb);
    std::printf("  fast path %llu; double path %llu within (-1, 1) (%.2e of them), %llu for |x| >= 1 or NaN\n", c[1], c[2],
                double(c[2]) / double(c[1] + c[2]), c[3]);
    std::printf("  the estimate's largest relative error against the double reference: %.3g (2^%.1f) for |x| < 0.5, %.3g (2^%.1f)"
                " otherwise; the margin is 2^-40\n", double(e0), e0 > 0 ? std::log2(double(e0)) : -999.0, double(e1),
                e1 > 0 ? std::log2(double(e1)) : -999.0);

    float* dOut;
    const int grid = 1 << 14, block = 256;
    CHECK(hipMalloc(&dOut, sizeof(float) * grid * block));
    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));
    const char* kinds[2] = {"angles over [0, 180]", "angles within 2 degrees of 180"};
    for (int kind = 0; kind < 2; ++kind)
        for (int rep = 0; rep < 2; ++rep)
            for (int ours = 0; ours < 2; ++ours)
            {
                CHECK(hipEventRecord(t0));
                if (ours) bench_kernel<true><<<grid, block>>>(dOut, kind, 0.0f);
                else bench_kernel<false><<<grid, block>>>(dOut, kind, 0.0f);
                CHECK(hipEventRecord(t1));
                CHECK(hipEventSynchronize(t1));
                float ms = 0;
                CHECK(hipEventElapsedTime(&ms, t0, t1));
                std::printf("%-32s %s: %8.2f ms, %7.1f G angles/s\n", kinds[kind], ours ? "cheshire" : "upstream", ms,
                            double(grid) * block * 64 / (ms * 1e6));
            }
    return 0;
}
