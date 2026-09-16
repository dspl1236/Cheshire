// Cheshire: where does hipHostMalloc(Mapped | NonCoherent) memory actually live on this platform?
//
// The memory bridge spills volumes and maps into that kind of allocation and assumes it is system
// RAM behind the PCIe link. On Windows a capped run showed 14 GB of "dedicated GPU memory" in Task
// Manager for a process whose bridge accounting said 0 MB of VRAM, so: allocate N GB of it, watch
// hipMemGetInfo, and time a bandwidth kernel over it against real VRAM. VRAM-backed mapped memory
// would read at VRAM speed and shrink the free VRAM; system-memory-backed reads at link speed.
//
//   host_backing [GB, default 4]
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("%s -> %s\n", #x, hipGetErrorString(e_)); return 1; } } while (0)

__global__ void sumKernel(const float4* __restrict__ p, size_t n, float* out)
{
    float acc = 0.f;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const float4 v = p[i]; acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.f) *out = acc;   // keep the loads alive
}

static double readGBs(const void* p, size_t bytes, float* out)
{
    const size_t n = bytes / sizeof(float4);
    sumKernel<<<4096, 256>>>((const float4*)p, n, out);
    (void)hipDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 3; ++i) sumKernel<<<4096, 256>>>((const float4*)p, n, out);
    (void)hipDeviceSynchronize();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 3;
    return double(bytes) / s / 1e9;
}

int main(int argc, char** argv)
{
    const size_t gb = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 4;
    const size_t bytes = gb << 30;
    hipDeviceProp_t prop{}; CK(hipGetDeviceProperties(&prop, 0));
    size_t free0 = 0, total = 0; CK(hipMemGetInfo(&free0, &total));
    std::printf("device: %s, VRAM total %zu MB, free %zu MB\n", prop.name, total >> 20, free0 >> 20);
    float* out = nullptr; CK(hipMalloc(&out, sizeof(float)));

    void* v = nullptr; CK(hipMalloc(&v, bytes >> 2));               // a VRAM reference of gb/4
    CK(hipMemset(v, 0, bytes >> 2));
    std::printf("VRAM   %zu MB: read %.0f GB/s\n", bytes >> 22, readGBs(v, bytes >> 2, out));
    size_t free1 = 0; CK(hipMemGetInfo(&free1, &total));
    std::printf("  free VRAM after the VRAM allocation: %zu MB (delta %zd MB)\n", free1 >> 20, (ptrdiff_t(free0) - ptrdiff_t(free1)) >> 20);
    CK(hipFree(v));

    void* h = nullptr;
    CK(hipHostMalloc(&h, bytes, hipHostMallocMapped | hipHostMallocNonCoherent));
    void* d = nullptr; CK(hipHostGetDevicePointer(&d, h, 0));
    CK(hipMemset(d, 0, bytes));
    size_t free2 = 0; CK(hipMemGetInfo(&free2, &total));
    std::printf("mapped host non-coherent %zu MB: read %.0f GB/s\n", bytes >> 20, readGBs(d, bytes, out));
    std::printf("  free VRAM after the host allocation: %zu MB (delta %zd MB vs start)\n", free2 >> 20, (ptrdiff_t(free0) - ptrdiff_t(free2)) >> 20);
    CK(hipHostFree(h));

    CK(hipHostMalloc(&h, bytes, hipHostMallocMapped));
    CK(hipHostGetDevicePointer(&d, h, 0));
    CK(hipMemset(d, 0, bytes));
    size_t free3 = 0; CK(hipMemGetInfo(&free3, &total));
    std::printf("mapped host coherent     %zu MB: read %.0f GB/s\n", bytes >> 20, readGBs(d, bytes, out));
    std::printf("  free VRAM after the host allocation: %zu MB (delta %zd MB vs start)\n", free3 >> 20, (ptrdiff_t(free0) - ptrdiff_t(free3)) >> 20);
    CK(hipHostFree(h));
    std::printf("Verdict: if the mapped reads run near VRAM speed or free VRAM drops by the allocation, the platform backs mapped host memory in VRAM.\n");
    return 0;
}
