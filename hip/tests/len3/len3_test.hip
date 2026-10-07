// cheshire: an exact len3 for the depth map's colour distances (euclideanDist3 -> norm3df).
//
// OCML's __ocml_len3_f32 (ROCm device libs, ocml.bc) sorts |x|, |y|, |z| by bit pattern, takes the largest's frexp
// exponent through a double (fpext + llvm.frexp.f64, so a denormal float gets its true exponent), scales the three by
// ldexp(-e), sums the squares with fmuladd, takes llvm.amdgcn.sqrt and scales back; an infinite input gives +inf.
// The double conversion and frexp are two FP64 instructions per call, and a refine sample makes two calls.
//
// cheshireLen3 is the same function with one change: when the largest magnitude is a normal float its frexp exponent
// is its biased exponent minus 126, read from the bits - the same integer, so the same scaling, sums, root and result.
// When all three are zeros the same steps give +0, the library's result (every scaled value is +0), so zeros stay on
// this path - a patch's centre pixel against itself and flat image regions give them often. Denormal, infinite or NaN
// largest magnitudes call __ocml_len3_f32 itself.
//
// This test compares the two bit for bit over random inputs of four kinds and times them.
//   build.cmd len3_test && len3_test.exe [batches]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" __device__ float __ocml_len3_f32(float, float, float);
extern "C" __device__ float __ocml_fmuladd_f32(float, float, float);  // the library's own, so the same lowering on every target
extern "C" __device__ float __ocml_ldexp_f32(float, int);

__device__ __forceinline__ float cheshireLen3(float x, float y, float z)
{
    const unsigned ua = __float_as_uint(fabsf(x));
    const unsigned ub = __float_as_uint(fabsf(y));
    const unsigned uc = __float_as_uint(fabsf(z));
    const unsigned m1 = max(ua, ub);
    const unsigned n1 = min(ua, ub);
    const unsigned a = max(m1, uc);
    const unsigned n2 = min(m1, uc);
    const unsigned b = max(n1, n2);
    const unsigned c = min(n1, n2);
    const unsigned biased = a >> 23;  // the sign bit is clear
    if ((biased == 0u && a != 0u) || biased == 255u)  // denormal largest magnitude, or inf / NaN: the library's own path
        return __ocml_len3_f32(x, y, z);
    const int e = int(biased) - 126;  // frexp's exponent of a normal float (all zeros: any e gives +0)
    const float as = __ocml_ldexp_f32(__uint_as_float(a), -e);
    const float bs = __ocml_ldexp_f32(__uint_as_float(b), -e);
    const float cs = __ocml_ldexp_f32(__uint_as_float(c), -e);
    const float t = __ocml_fmuladd_f32(as, as, __ocml_fmuladd_f32(bs, bs, cs * cs));
    return __ocml_ldexp_f32(__builtin_amdgcn_sqrtf(t), e);
}

__device__ __forceinline__ uint32_t mix(uint64_t v)
{
    v ^= v >> 33; v *= 0xff51afd7ed558ccdull; v ^= v >> 33; v *= 0xc4ceb9fe1a85ec53ull; v ^= v >> 33;
    return uint32_t(v);
}

// kind 0: colour differences, uniform in [-255, 255]; kind 1: any 32-bit pattern; kind 2: magnitudes 2^-160..2^-100
// and their neighbours (denormal and tiny normal largest values), signs random; kind 3: each value +0 or -0 with
// probability 1/2, else a colour difference (all three zeros in an eighth of the inputs, every sign combination)
__device__ __forceinline__ float input(int kind, uint64_t i, int k)
{
    const uint32_t r = mix(i * 3u + uint64_t(k) + 0x9e3779b97f4a7c15ull);
    if (kind == 0)
        return (float(r >> 8) * (1.0f / 16777216.0f)) * 510.0f - 255.0f;
    if (kind == 1)
        return __uint_as_float(r);
    if (kind == 3)
        return (r & 1u) ? __uint_as_float(r & 0x80000000u) : (float(r >> 8) * (1.0f / 16777216.0f)) * 510.0f - 255.0f;
    const uint32_t expo = (r >> 23) % 40u;  // 0 .. 39 -> biased exponents 0 .. 39 (denormal and tiny normal)
    return __uint_as_float((r & 0x807fffffu) | (expo << 23));
}

__global__ void compare_kernel(uint64_t base, int kind, unsigned long long* mismatches, unsigned long long* fallback, float* firstBad)
{
    const uint64_t i = base + uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const float x = input(kind, i, 0), y = input(kind, i, 1), z = input(kind, i, 2);
    const float ref = __ocml_len3_f32(x, y, z);
    const float got = cheshireLen3(x, y, z);
    const unsigned ua = __float_as_uint(fabsf(x)), ub = __float_as_uint(fabsf(y)), uc = __float_as_uint(fabsf(z));
    const unsigned a = max(max(ua, ub), uc), biased = a >> 23;
    if ((biased == 0u && a != 0u) || biased == 255u)
        atomicAdd(fallback, 1ull);
    if (a == 0u)
        atomicAdd(fallback + 1, 1ull);  // all three zeros, on our own path
    if (__float_as_uint(ref) != __float_as_uint(got))
    {
        if (atomicAdd(mismatches, 1ull) == 0ull)
        {
            firstBad[0] = x; firstBad[1] = y; firstBad[2] = z; firstBad[3] = ref; firstBad[4] = got;
        }
    }
}

// throughput: each thread evaluates 256 len3 of colour-like values derived from registers
template <bool TOurs>
__global__ void bench_kernel(float* out, float seed)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    float x = seed + float(t & 255), y = 3.0f - float(t & 127) * 0.5f, z = float(t & 63) * 1.7f;
    float acc = 0.0f;
#pragma unroll 4
    for (int k = 0; k < 256; ++k)
    {
        const float d = TOurs ? cheshireLen3(x, y, z) : __ocml_len3_f32(x, y, z);
        acc += d;
        x = y - d * 0.001f; y = z + 0.75f; z = x * 0.5f - 1.0f;
    }
    out[t] = acc;
}

#define CHECK(c) do { hipError_t e_ = (c); if (e_ != hipSuccess) { std::printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); return 1; } } while (0)

int main(int argc, char** argv)
{
    const int batches = argc > 1 ? std::atoi(argv[1]) : 64;
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device: %s (%s)\n", prop.name, prop.gcnArchName);
    unsigned long long *dMis, *dFb;
    float* dBad;
    CHECK(hipMalloc(&dMis, 8)); CHECK(hipMalloc(&dFb, 16)); CHECK(hipMalloc(&dBad, 5 * sizeof(float)));
    const int block = 256, grid = 1 << 18;  // 2^26 inputs per launch
    const char* names[4] = {"colour differences [-255, 255]", "any 32-bit pattern", "denormal and tiny largest magnitudes",
                            "zeros among colour differences"};
    for (int kind = 0; kind < 4; ++kind)
    {
        CHECK(hipMemset(dMis, 0, 8)); CHECK(hipMemset(dFb, 0, 16));
        for (int b = 0; b < batches; ++b)
            compare_kernel<<<grid, block>>>(uint64_t(b) * uint64_t(grid) * block + uint64_t(kind) * (1ull << 40), kind, dMis, dFb, dBad);
        CHECK(hipDeviceSynchronize());
        unsigned long long mis = 0, fb[2] = {};
        float bad[5] = {};
        CHECK(hipMemcpy(&mis, dMis, 8, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(fb, dFb, 16, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(bad, dBad, sizeof bad, hipMemcpyDeviceToHost));
        const double n = double(batches) * grid * block;
        std::printf("%-38s %.3g inputs: %llu mismatches, %llu took the library path, %llu all zeros\n", names[kind], n, mis, fb[0], fb[1]);
        if (mis)
            std::printf("    first: len3(%a, %a, %a) = %a, ours %a\n", bad[0], bad[1], bad[2], bad[3], bad[4]);
    }
    float* dOut;
    const int bgrid = 1 << 14, bblock = 256;
    CHECK(hipMalloc(&dOut, sizeof(float) * bgrid * bblock));
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    for (int rep = 0; rep < 3; ++rep)
    {
        for (int ours = 0; ours < 2; ++ours)
        {
            CHECK(hipEventRecord(e0));
            for (int it = 0; it < 10; ++it)
            {
                if (ours) bench_kernel<true><<<bgrid, bblock>>>(dOut, 1.0f + it);
                else bench_kernel<false><<<bgrid, bblock>>>(dOut, 1.0f + it);
            }
            CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
            float ms = 0; CHECK(hipEventElapsedTime(&ms, e0, e1));
            const double evals = 10.0 * bgrid * bblock * 256;
            std::printf("%s: %.2f ms, %.1f G len3/s\n", ours ? "cheshireLen3   " : "__ocml_len3_f32", ms, evals / (ms * 1e6));
        }
    }
    return 0;
}
