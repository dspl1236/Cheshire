// cheshire: an exact reciprocal for the depth map's projections (project3DPoint: __fdividef(1.0f, p.z)).
//
// On HIP __fdividef(x, y) is x / y, and HIP divides correctly rounded by default (-fhip-fp32-correctly-rounded-divide-
// sqrt): v_div_scale twice, v_rcp_f32, five FMAs, v_div_fmas and v_div_fixup. A refine sample projects its point into
// both cameras, so two of those.
//
// cheshireRcp is Markstein's correction of the hardware reciprocal: y = v_rcp_f32(b), r = fma(-b, y, 1), y' = fma(y, r, y).
// When y is within one ulp of 1/b, y' is the correctly rounded 1/b - the division's result - except when b's significand
// is all ones (b = 2 - 2^-23 times a power of two), whose reciprocal lies within 2^-49 of a midpoint: there y = 0.5
// gives a tie that rounds to even, the wrong side. So that significand, and every b whose reciprocal or residual could
// leave the normal range (zero, denormal, |b| >= 2^126, inf, NaN), takes the division itself.
//
// The proof has two parts, both run here:
//  1. architecture independent: for every significand of b in [1, 2) and every float y within three steps of RN(1/b),
//     the correction's result is compared with the division's, and y's distance from 1/b recorded (exactly, in
//     double: y*b is a 48-bit product). The FMAs are IEEE operations, the same on any hardware. The claim is that every
//     y within one ulp gives the exact result outside the all-ones significand; scaling b by a power of two scales y,
//     r's operands and y' alike, so the binade [1, 2) stands for every exponent the fast path accepts. AMD documents
//     v_rcp_f32 as accurate to one ulp, so any RDNA or GCN reciprocal satisfies the premise.
//  2. this card: every one of the 2^32 bit patterns through cheshireRcp against __fdividef(1.0f, z).
//   build.cmd rcp_test && rcp_test.exe
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

// as in device/matrix.cuh (step 11b): the range by two compares (NaN fails both), the significand by its bits, all
// three evaluated (& rather than &&: one branch)
__device__ __forceinline__ float cheshireRcp(float z)
{
    const float az = fabsf(z);
    if ((az >= 0x1p-126f) & (az < 0x1p126f) & ((__float_as_uint(z) & 0x7fffffu) != 0x7fffffu))
    {
        const float y = __builtin_amdgcn_rcpf(z);
        const float r = fmaf(-z, y, 1.0f);
        return fmaf(y, r, y);
    }
    return __fdividef(1.0f, z);
}

// part 1: m = significand index (b = 1 + m * 2^-23); k = candidate offset -3..3 steps from RN(1/b)
// out: per offset, how many candidates within one ulp failed and the smallest error (in ulps of 2^-24) of a failure
__global__ void proof_kernel(unsigned long long* within, unsigned long long* failedWithin, unsigned* failedAllOnes,
                             unsigned long long* failedOutside, unsigned* minFailUlpBits, unsigned* firstFailM)
{
    const unsigned m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= (1u << 23))
        return;
    const float b = __uint_as_float(0x3f800000u | m);
    const float ref = __fdividef(1.0f, b);
    for (int k = -3; k <= 3; ++k)
    {
        const float y = __uint_as_float(__float_as_uint(ref) + k);
        // |y - 1/b| in units of 2^-24, exactly: (y*b - 1) / b, y*b exact in double
        const double err = fabs(double(y) * double(b) - 1.0) / double(b) * 16777216.0;
        const float r = fmaf(-b, y, 1.0f);
        const float y2 = fmaf(y, r, y);
        const bool ok = __float_as_uint(y2) == __float_as_uint(ref);
        if (err <= 1.0)
        {
            atomicAdd(within, 1ull);
            if (!ok)
            {
                if (m == 0x7fffffu)
                    atomicAdd(failedAllOnes, 1u);
                else
                {
                    atomicAdd(failedWithin, 1ull);
                    atomicMin(firstFailM, m);
                }
            }
        }
        else if (!ok)
            atomicAdd(failedOutside, 1ull);
        if (!ok && m != 0x7fffffu)
            atomicMin(minFailUlpBits, __float_as_uint(float(err)));  // positive floats order like their bits
    }
}

// part 2: every bit pattern
__global__ void exhaustive_kernel(unsigned base, unsigned long long* mismatches, unsigned long long* fast, unsigned* firstBad)
{
    const unsigned u = base + blockIdx.x * blockDim.x + threadIdx.x;
    const float b = __uint_as_float(u);
    const float ref = __fdividef(1.0f, b);
    const float got = cheshireRcp(b);
    const unsigned biased = (u >> 23) & 0xffu;
    if (!(biased - 1u >= 252u || (u & 0x7fffffu) == 0x7fffffu))
        atomicAdd(fast, 1ull);
    if (__float_as_uint(ref) != __float_as_uint(got))
    {
        atomicAdd(mismatches, 1ull);
        atomicMin(firstBad, u);
    }
}

// throughput: four independent chains per thread, so the card overlaps them as it overlaps a kernel's other work
template <bool TOurs>
__global__ void bench_kernel(float* out, float seed)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    float z[4], acc[4];
    for (int j = 0; j < 4; ++j)
    {
        z[j] = seed + float((t + 257 * j) & 1023) * 0.37f;
        acc[j] = 0.0f;
    }
#pragma unroll 2
    for (int k = 0; k < 64; ++k)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const float q = TOurs ? cheshireRcp(z[j]) : __fdividef(1.0f, z[j]);
            acc[j] += q;
            z[j] = z[j] + 0.25f;
        }
    }
    out[t] = acc[0] + acc[1] + acc[2] + acc[3];
}

#define CHECK(c) do { hipError_t e_ = (c); if (e_ != hipSuccess) { std::printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); return 1; } } while (0)

int main()
{
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device: %s (%s)\n", prop.name, prop.gcnArchName);

    unsigned long long* d64;
    unsigned* d32;
    CHECK(hipMalloc(&d64, 4 * 8));
    CHECK(hipMalloc(&d32, 4 * 4));
    CHECK(hipMemset(d64, 0, 4 * 8));
    const unsigned init32[4] = {0u, 0x7f800000u, 0xffffffffu, 0xffffffffu};  // all-ones failures, min fail error, first m, -
    CHECK(hipMemcpy(d32, init32, sizeof init32, hipMemcpyHostToDevice));
    proof_kernel<<<(1u << 23) / 256, 256>>>(d64, d64 + 1, d32, d64 + 2, d32 + 1, d32 + 2);
    CHECK(hipDeviceSynchronize());
    unsigned long long h64[4];
    unsigned h32[4];
    CHECK(hipMemcpy(h64, d64, sizeof h64, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(h32, d32, sizeof h32, hipMemcpyDeviceToHost));
    float minFail;
    std::memcpy(&minFail, &h32[1], 4);
    std::printf("part 1, every significand in [1, 2), y within 3 steps of RN(1/b):\n");
    std::printf("  %llu candidates within one ulp of 1/b: %llu wrong outside the all-ones significand (first m %s0x%x),"
                " %u wrong at the all-ones significand\n", h64[0], h64[1], h64[1] ? "" : "none, ", h64[1] ? h32[2] : 0u, h32[0]);
    std::printf("  candidates beyond one ulp that went wrong: %llu; outside the all-ones significand the closest wrong"
                " candidate was %.6f ulp from 1/b\n", h64[2], double(minFail));

    unsigned long long* e64;
    unsigned* e32;
    CHECK(hipMalloc(&e64, 2 * 8));
    CHECK(hipMalloc(&e32, 4));
    CHECK(hipMemset(e64, 0, 2 * 8));
    const unsigned initBad = 0xffffffffu;
    CHECK(hipMemcpy(e32, &initBad, 4, hipMemcpyHostToDevice));
    const unsigned block = 256, grid = 1u << 16;  // 2^24 per launch
    for (unsigned b = 0; b < 256; ++b)
        exhaustive_kernel<<<grid, block>>>(b << 24, e64, e64 + 1, e32);
    CHECK(hipDeviceSynchronize());
    unsigned long long eh[2];
    unsigned firstBad;
    CHECK(hipMemcpy(eh, e64, sizeof eh, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&firstBad, e32, 4, hipMemcpyDeviceToHost));
    std::printf("part 2, all 2^32 bit patterns on this card: %llu mismatches (%llu on the fast path)", eh[0], eh[1]);
    if (eh[0])
        std::printf(", first 0x%08x", firstBad);
    std::printf("\n");

    float* dOut;
    const int bgrid = 1 << 14, bblock = 256;
    CHECK(hipMalloc(&dOut, sizeof(float) * bgrid * bblock));
    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0)); CHECK(hipEventCreate(&t1));
    for (int rep = 0; rep < 3; ++rep)
        for (int ours = 0; ours < 2; ++ours)
        {
            CHECK(hipEventRecord(t0));
            for (int it = 0; it < 10; ++it)
            {
                if (ours) bench_kernel<true><<<bgrid, bblock>>>(dOut, 1.0f + it);
                else bench_kernel<false><<<bgrid, bblock>>>(dOut, 1.0f + it);
            }
            CHECK(hipEventRecord(t1)); CHECK(hipEventSynchronize(t1));
            float ms = 0; CHECK(hipEventElapsedTime(&ms, t0, t1));
            std::printf("%s: %.2f ms, %.1f G reciprocals/s\n", ours ? "cheshireRcp       " : "__fdividef(1, z)  ", ms,
                        10.0 * bgrid * bblock * 256 / (ms * 1e6));
        }
    return 0;
}
