// Sustained INT8 WMMA rate on this card, alone and with the matcher's per-candidate VALU work (one shift-add, a min-max
// and a max per output value), so the matcher kernels can be compared with what the hardware delivers under load.
//   wmma_peak.exe [iterations=4000]
#include <hip/hip_runtime.h>
#include <climits>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));

// CH independent accumulation chains of 8 WMMAs per iteration; VALU = 1 adds the matcher's insertion on every result.
template<int CH, int VALU>
__global__ void __launch_bounds__(256) peak(int iters, int* out)
{
    const int lane = threadIdx.x;
    v2i a[CH], b;
#pragma unroll
    for (int c = 0; c < CH; ++c) { a[c][0] = lane * 7 + c; a[c][1] = lane * 13 + c; }
    b[0] = lane * 3; b[1] = lane * 5;
    int p0[CH][8], p1[CH][8];
#pragma unroll
    for (int c = 0; c < CH; ++c)
#pragma unroll
        for (int v = 0; v < 8; ++v) { p0[c][v] = INT_MIN; p1[c][v] = INT_MIN; }
    v8i keep = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) {
        v8i acc[CH];
#pragma unroll
        for (int c = 0; c < CH; ++c) acc[c] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int c = 0; c < CH; ++c) acc[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a[c], false, b, acc[c], false);
        if (VALU) {
#pragma unroll
            for (int c = 0; c < CH; ++c)
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const int y = int((unsigned(acc[c][v]) << 9) + unsigned(it));
                    p1[c][v] = max(min(p0[c][v], y), p1[c][v]);
                    p0[c][v] = max(p0[c][v], y);
                }
        } else {
#pragma unroll
            for (int c = 0; c < CH; ++c) keep ^= acc[c];
        }
        b[0] += 1;   // keeps the products changing, one VALU per iteration
    }
    int s = keep[0] ^ keep[7];
#pragma unroll
    for (int c = 0; c < CH; ++c)
#pragma unroll
        for (int v = 0; v < 8; ++v) s ^= p0[c][v] + p1[c][v];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template<int CH, int VALU>
void run(int iters, int* dOut, hipEvent_t e0, hipEvent_t e1)
{
    const int blocks = 1024;
    peak<CH, VALU><<<blocks, 256>>>(iters / 10, dOut);   // warm-up
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    peak<CH, VALU><<<blocks, 256>>>(iters, dOut);
    CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
    float ms = 0; CHECK(hipEventElapsedTime(&ms, e0, e1));
    const double macs = double(blocks) * 8 * iters * 8 * CH * 4096.0;
    std::printf("chains %d, %s: %.1f ms, %.1f TMAC/s\n", CH, VALU ? "WMMA + 3 VALU per value" : "WMMA only            ", ms, macs / ms * 1e-9);
}

int main(int argc, char** argv)
{
    const int iters = argc > 1 ? std::atoi(argv[1]) : 4000;
    hipDeviceProp_t p{}; CHECK(hipGetDeviceProperties(&p, 0));
    std::printf("%s, %d CUs, clock %d MHz (reported max)\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    int* dOut = nullptr; CHECK(hipMalloc(&dOut, 1024 * 256 * 4));
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    run<1, 0>(iters, dOut, e0, e1); run<2, 0>(iters, dOut, e0, e1); run<3, 0>(iters, dOut, e0, e1);
    run<1, 1>(iters, dOut, e0, e1); run<2, 1>(iters, dOut, e0, e1); run<3, 1>(iters, dOut, e0, e1);
    return 0;
}
