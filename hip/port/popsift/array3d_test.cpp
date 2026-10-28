// Cheshire: is a non-layered 3D array wider than the reported maxTexture3D (2048 on RDNA) real?
//
// Since 0.10.1 upstream PopSIFT keeps each octave's blur levels in a plain 3D array on HIP (the levels
// are its z), written with surf3Dwrite and read with surf3Dread or through a texture, and it checks the
// image against the device's reported limits. HIP reports maxTexture3D as 2048 x 2048 x 2048 on the
// RX 9070, so PopSIFT would refuse every photograph wider than 2048. This allocates the pyramid's array
// at photograph sizes, writes a distinct value to every cell, and reads all of it back three ways: a
// surface read and a point-filtered texture fetch in a later kernel launch, and a copy to the host.
// Build: hipcc (or clang-cl -x hip) with --offload-arch for the card; run with no arguments.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

#define CHK(call)                                                                                  \
    do {                                                                                           \
        hipError_t e = (call);                                                                     \
        if (e != hipSuccess) {                                                                     \
            std::printf("  %s: %s%c", #call, hipGetErrorString(e), 10);                            \
            return false;                                                                          \
        }                                                                                          \
    } while (0)

__host__ __device__ static inline float expected(int x, int y, int z)
{
    return (float)(((unsigned)x * 73856093u) ^ ((unsigned)y * 19349663u) ^ ((unsigned)z * 83492791u)) / 4294967296.0f;
}

__global__ void fill(hipSurfaceObject_t surf, int w, int h, int d)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    for (int z = 0; z < d; ++z)
        surf3Dwrite(expected(x, y, z), surf, x * 4, y, z);
}

__global__ void check(hipSurfaceObject_t surf, hipTextureObject_t tex, int w, int h, int d, unsigned* bad)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    for (int z = 0; z < d; ++z) {
        float s;
        surf3Dread(&s, surf, x * 4, y, z);
        const float t = tex3D<float>(tex, x + 0.5f, y + 0.5f, z + 0.5f);
        const float e = expected(x, y, z);
        if (s != e) atomicAdd(&bad[0], 1u);
        if (t != e) atomicAdd(&bad[1], 1u);
    }
}

static bool run(int w, int h, int d)
{
    std::printf("%5d x %5d x %d%c", w, h, d, 10);
    hipArray_t arr = nullptr;
    hipChannelFormatDesc fd = hipCreateChannelDesc<float>();
    CHK(hipMalloc3DArray(&arr, &fd, make_hipExtent(w, h, d), hipArraySurfaceLoadStore));
    hipResourceDesc rd{};
    rd.resType = hipResourceTypeArray;
    rd.res.array.array = arr;
    hipSurfaceObject_t surf = 0;
    CHK(hipCreateSurfaceObject(&surf, &rd));
    hipTextureDesc td{};
    td.normalizedCoords = 0;
    td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = hipAddressModeClamp;
    td.readMode = hipReadModeElementType;
    td.filterMode = hipFilterModePoint;
    hipTextureObject_t tex = 0;
    CHK(hipCreateTextureObject(&tex, &rd, &td, nullptr));
    unsigned* bad = nullptr;
    CHK(hipMalloc(&bad, 2 * sizeof(unsigned)));
    CHK(hipMemset(bad, 0, 2 * sizeof(unsigned)));
    dim3 b(16, 16), g((w + 15) / 16, (h + 15) / 16);
    fill<<<g, b>>>(surf, w, h, d);
    CHK(hipGetLastError());
    CHK(hipDeviceSynchronize());
    check<<<g, b>>>(surf, tex, w, h, d, bad);   // a later launch, as PopSIFT's reads are
    CHK(hipGetLastError());
    CHK(hipDeviceSynchronize());
    unsigned hb[2];
    CHK(hipMemcpy(hb, bad, sizeof(hb), hipMemcpyDeviceToHost));
    // and the host copy of the whole array
    std::vector<float> host((size_t)w * h * d);
    hipMemcpy3DParms p{};
    p.srcArray = arr;
    p.dstPtr = make_hipPitchedPtr(host.data(), (size_t)w * sizeof(float), w, h);
    p.extent = make_hipExtent(w, h, d);
    p.kind = hipMemcpyDeviceToHost;
    CHK(hipMemcpy3D(&p));
    size_t hbad = 0;
    for (int z = 0; z < d; ++z)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                hbad += host[((size_t)z * h + y) * w + x] != expected(x, y, z);
    const size_t cells = (size_t)w * h * d;
    std::printf("  %zu cells: surface read %u wrong, texture fetch %u wrong, host copy %zu wrong%c", cells, hb[0], hb[1], hbad, 10);
    hipFree(bad);
    hipDestroyTextureObject(tex);
    hipDestroySurfaceObject(surf);
    hipFreeArray(arr);
    return hb[0] == 0 && hb[1] == 0 && hbad == 0;
}

int main()
{
    hipDeviceProp_t p;
    hipGetDeviceProperties(&p, 0);
    std::printf("%s (%s): maxTexture3D %d %d %d, maxTexture2D %d %d%c", p.name, p.gcnArchName, p.maxTexture3D[0],
                p.maxTexture3D[1], p.maxTexture3D[2], p.maxTexture2D[0], p.maxTexture2D[1], 10);
    const int cases[][3] = {{2048, 2048, 6}, {6000, 3376, 6}, {8192, 8192, 6}, {16384, 2048, 6}};
    bool ok = true;
    for (auto& c : cases)
        ok = run(c[0], c[1], c[2]) && ok;
    std::printf("%s%c", ok ? "all correct" : "SOME WRONG", 10);
    return ok ? 0 : 1;
}
