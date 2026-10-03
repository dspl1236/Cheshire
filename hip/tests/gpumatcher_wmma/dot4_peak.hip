// Sustained packed uint8 dot-product rate (v_dot4_u32_u8, 4 multiply-adds per lane) on this card: CHAINS independent
// accumulators per thread, each a sequence of dependent dot4s like knn2_u8's 32 per candidate, to see whether the
// instruction's throughput or its dependency latency bounds the matcher's dot4 kernel.
//   dot4_peak.exe [iterations=4000]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

template<int CHAINS>
__global__ void __launch_bounds__(256) dot4chains(int iters, unsigned int* out)
{
    unsigned int a[CHAINS], acc[CHAINS], b[32];
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) { a[c] = (threadIdx.x + 7u * c) * 0x9E3779B1u; acc[c] = 0u; }
#pragma unroll
    for (int k = 0; k < 32; ++k) b[k] = (threadIdx.x ^ (k * 0x01010101u)) * 0x85EBCA6Bu;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int k = 0; k < 32; ++k)
#pragma unroll
            for (int c = 0; c < CHAINS; ++c) acc[c] = __builtin_amdgcn_udot4(a[c], b[k], acc[c], false);
#pragma unroll
        for (int c = 0; c < CHAINS; ++c) a[c] += 1u;   // keeps the products changing
    }
    unsigned int s = 0u;
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) s ^= acc[c];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template<int CHAINS>
void run(int iters, unsigned int* dOut, hipEvent_t e0, hipEvent_t e1, double peakGuess)
{
    const int blocks = 4096;
    dot4chains<CHAINS><<<blocks, 256>>>(iters / 10, dOut);
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    dot4chains<CHAINS><<<blocks, 256>>>(iters, dOut);
    CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
    float ms = 0; CHECK(hipEventElapsedTime(&ms, e0, e1));
    const double macs = double(blocks) * 256 * iters * 32.0 * CHAINS * 4.0;
    const double tmac = macs / ms * 1e-9;
    std::printf("chains %d: %.1f ms, %.2f TMAC/s (%.0f %% of %.1f, one dot4 per lane per clock)\n", CHAINS, ms, tmac,
                100.0 * tmac / peakGuess, peakGuess);
}

int main(int argc, char** argv)
{
    const int iters = argc > 1 ? std::atoi(argv[1]) : 4000;
    hipDeviceProp_t p{}; CHECK(hipGetDeviceProperties(&p, 0));
    // a full-rate guess: 2 SIMD32 per compute unit, 32 lanes, 4 MACs per dot4, at 2.52 GHz (the boost wmma_peak reaches)
    const double peakGuess = 2.0 * p.multiProcessorCount * 2 * 32 * 4 * 2.52e9 * 1e-12;
    std::printf("%s (%s), %d WGPs reported\n", p.name, p.gcnArchName, p.multiProcessorCount);
    unsigned int* dOut = nullptr; CHECK(hipMalloc(&dOut, 4096 * 256 * 4));
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    run<1>(iters, dOut, e0, e1, peakGuess); run<2>(iters, dOut, e0, e1, peakGuess);
    run<4>(iters, dOut, e0, e1, peakGuess); run<8>(iters, dOut, e0, e1, peakGuess);
    return 0;
}
