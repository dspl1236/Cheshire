// Cheshire: does the ballot PopSIFT's extrema compaction relies on report the right lanes?
//
// extrema_count() reserves output slots with
//     mask = ballot(indicator); ct = __popc(mask); if (lane==0) base = atomicAdd(counter, ct);
// so a ballot that reports every lane as set makes each warp reserve 32 slots however many extrema
// it actually found. The slots nothing writes keep their initial contents, which is how a keypoint
// ends up with sigma == 0. Per-octave extrema counts coming back as exact multiples of 32 is the
// symptom; this is the direct check.
#include "popsift_hip.h"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

// through the shim, exactly as popsift/common/assist.h spells it
__device__ static inline unsigned int balletViaShim(int pred) { return __ballot_sync(0xffffffff, pred); }

__global__ void ballotKernel(const int* preds, unsigned int* shimMask, unsigned int* nativeMask, int* shimCount)
{
    const int lane = threadIdx.x;
    const int pred = preds[lane];
    const unsigned int viaShim = balletViaShim(pred);
    const unsigned long long viaNative = __ballot(pred);
    if (lane == 0)
    {
        *shimMask = viaShim;
        *nativeMask = (unsigned int)viaNative;
        *shimCount = __popc(viaShim);
    }
}

int main()
{
    int warp = 0;
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    warp = prop.warpSize;
    std::printf("warpSize %d%c", warp, 10);

    // three predicate patterns, each with a known correct mask over lanes 0..31
    struct { const char* name; int mod; } cases[] = {{"every 7th lane", 7}, {"every 4th lane", 4}, {"lane 0 only", 32}};

    int* devPreds = nullptr;
    unsigned int* devShim = nullptr;
    unsigned int* devNative = nullptr;
    int* devCount = nullptr;
    hipMalloc(&devPreds, sizeof(int) * 32);
    hipMalloc(&devShim, sizeof(unsigned int));
    hipMalloc(&devNative, sizeof(unsigned int));
    hipMalloc(&devCount, sizeof(int));

    for (auto& c : cases)
    {
        std::vector<int> preds(32);
        unsigned int want = 0;
        int wantCount = 0;
        for (int i = 0; i < 32; ++i)
        {
            preds[i] = (i % c.mod == 0) ? 1 : 0;
            if (preds[i]) { want |= (1u << i); ++wantCount; }
        }
        hipMemcpy(devPreds, preds.data(), sizeof(int) * 32, hipMemcpyHostToDevice);
        hipLaunchKernelGGL(ballotKernel, dim3(1), dim3(32), 0, 0, devPreds, devShim, devNative, devCount);
        hipDeviceSynchronize();
        unsigned int shim = 0, native = 0;
        int count = 0;
        hipMemcpy(&shim, devShim, sizeof(unsigned int), hipMemcpyDeviceToHost);
        hipMemcpy(&native, devNative, sizeof(unsigned int), hipMemcpyDeviceToHost);
        hipMemcpy(&count, devCount, sizeof(int), hipMemcpyDeviceToHost);
        std::printf("%-16s want %08x (%2d set) | shim %08x (popc %2d) %s | native __ballot %08x %s%c",
                    c.name, want, wantCount, shim, count, shim == want ? "ok" : "WRONG",
                    native, native == want ? "ok" : "WRONG", 10);
    }
    return 0;
}
