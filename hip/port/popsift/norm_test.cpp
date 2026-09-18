// Cheshire: what does PopSIFT's RootSift normaliser actually compute on HIP?
//
// normalize_histogram launches a (32,32) block: each row of 32 threads owns one descriptor and
// reduces its 128 bins with warp shuffles that carry no width argument. That is exact on CUDA,
// where a warp is 32 lanes and a row is a warp. This replicates the kernel on known positive input
// and prints the sum each row arrives at, so a wrong reduction shows up directly.
#include "popsift_hip.h"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

struct Desc { float features[128]; };

__device__ static inline float shflDown(float v, int delta) { return __shfl_down_sync(0xffffffff, v, delta); }
__device__ static inline float shflFrom(float v, int src) { return __shfl_sync(0xffffffff, v, src); }

// the body of NormalizeRootSift::normalize, with the sum reported instead of consumed
__global__ void rootSiftSum(const Desc* descs, float* sums, int count)
{
    const int offset = blockIdx.x * 32 + threadIdx.y;
    if (offset >= count) return;
    const float4* ptr4 = (const float4*)descs[offset].features;
    float4 descr = ptr4[threadIdx.x];

    float sum = descr.x + descr.y + descr.z + descr.w;
    sum += shflDown(sum, 16);
    sum += shflDown(sum, 8);
    sum += shflDown(sum, 4);
    sum += shflDown(sum, 2);
    sum += shflDown(sum, 1);
    sum = shflFrom(sum, 0);

    if (threadIdx.x == 0) sums[offset] = sum;
}

// the same reduction with the width pinned to 32, which is what CUDA's warp gives it
__global__ void rootSiftSumWidth32(const Desc* descs, float* sums, int count)
{
    const int offset = blockIdx.x * 32 + threadIdx.y;
    if (offset >= count) return;
    const float4* ptr4 = (const float4*)descs[offset].features;
    float4 descr = ptr4[threadIdx.x];

    float sum = descr.x + descr.y + descr.z + descr.w;
    sum += __shfl_down_sync(0xffffffff, sum, 16, 32);
    sum += __shfl_down_sync(0xffffffff, sum, 8, 32);
    sum += __shfl_down_sync(0xffffffff, sum, 4, 32);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 32);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 32);
    sum = __shfl_sync(0xffffffff, sum, 0, 32);

    if (threadIdx.x == 0) sums[offset] = sum;
}

__global__ void reportWarpSize(int* out)
{
    if (threadIdx.x == 0) *out = warpSize;
}

int main()
{
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::printf("device %s, host-side warpSize %d%c", prop.name, prop.warpSize, 10);

    int* devWarp = nullptr;
    hipMalloc(&devWarp, sizeof(int));
    hipLaunchKernelGGL(reportWarpSize, dim3(1), dim3(32), 0, 0, devWarp);
    int devWarpSize = 0;
    hipMemcpy(&devWarpSize, devWarp, sizeof(int), hipMemcpyDeviceToHost);
    std::printf("device-side warpSize %d%c", devWarpSize, 10);

    // every descriptor is all ones, so the correct sum is exactly 128 for each
    const int count = 64;
    std::vector<Desc> host(count);
    for (int d = 0; d < count; ++d)
        for (int k = 0; k < 128; ++k) host[d].features[k] = 1.0f;

    Desc* devDescs = nullptr;
    float* devSums = nullptr;
    hipMalloc(&devDescs, sizeof(Desc) * count);
    hipMalloc(&devSums, sizeof(float) * count);
    hipMemcpy(devDescs, host.data(), sizeof(Desc) * count, hipMemcpyHostToDevice);

    std::vector<float> sums(count);
    const dim3 block(32, 32), grid((count + 31) / 32);
    for (int variant = 0; variant < 2; ++variant)
    {
        hipMemset(devSums, 0, sizeof(float) * count);
        if (variant == 0) hipLaunchKernelGGL(rootSiftSum, grid, block, 0, 0, devDescs, devSums, count);
        else              hipLaunchKernelGGL(rootSiftSumWidth32, grid, block, 0, 0, devDescs, devSums, count);
        hipDeviceSynchronize();
        hipMemcpy(sums.data(), devSums, sizeof(float) * count, hipMemcpyDeviceToHost);
        int bad = 0;
        for (float s : sums) if (s != 128.0f) ++bad;
        std::printf("%-22s %d of %d rows wrong; first eight sums:",
                    variant == 0 ? "as popsift writes it" : "width pinned to 32", bad, count);
        for (int d = 0; d < 8; ++d) std::printf(" %.0f", sums[d]);
        std::printf("   (each should be 128)%c", 10);
    }
    return 0;
}
