// Standalone reproducer: HIP's layered surface accessors address the mip level, not the array layer.
//
// Build and run:
//   hipcc --offload-arch=gfx1201 bugreport_layered_surface.hip -o layered && ./layered
//
// Cause, in include/hip/amd_detail/amd_surface_functions.h (ROCm 7.2.1, lines 271-298):
//   surf2DLayeredread  -> __ockl_image_load_lod_2D (i, coords, layer)
//   surf2DLayeredwrite -> __ockl_image_store_lod_2D(i, coords, layer, tmp)
// Both pass the array layer to the LOD parameter of a plain 2D image, so the access lands on a mip
// level that does not exist and is silently dropped. The 2D-array entry points
// __ockl_image_load_2Da / __ockl_image_store_2Da take the layer in coords.z instead. The same two
// functions also compute the pixel address with __ockl_image_channel_data_type_2D / _order_2D
// rather than the _2Da variants; that is a second discrepancy, and part B's partial failure on
// layer 0 (32 of 128 cells) is consistent with it, though this reproducer does not prove that.
//
// Part A populates the array with hipMemcpy3D, so the read accessors are tested on their own.
// Part B writes with the surface accessor and reads back with the texture accessor, which part A
// has just shown to be correct, so it tests the write accessor on its own.
// Part C shows the reported layered limit is not the real one.
#include <hip/hip_runtime.h>
#include <cstdio>

static const int W = 32, H = 4, L = 4;

__host__ __device__ static inline float expected(int layer, int x, int y)
{
    return (float)(layer * 10000 + y * 100 + x);
}

__global__ void writeWithSurface(hipSurfaceObject_t surf)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    for (int layer = 0; layer < L; ++layer)
        surf2DLayeredwrite(expected(layer, x, y), surf, x * (int)sizeof(float), y, layer);
}

__global__ void readWithSurface(hipSurfaceObject_t surf, float* out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    for (int layer = 0; layer < L; ++layer)
    {
        float v = -1.0f;
        surf2DLayeredread(&v, surf, x * (int)sizeof(float), y, layer);
        out[(layer * H + y) * W + x] = v;
    }
}

__global__ void readWithTexture(hipTextureObject_t tex, float* out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    for (int layer = 0; layer < L; ++layer)
        out[(layer * H + y) * W + x] = tex2DLayered<float>(tex, x + 0.5f, y + 0.5f, layer);
}

static void report(const char* what, const float* got)
{
    int wrongPerLayer[L] = {0};
    int wrong = 0;
    for (int layer = 0; layer < L; ++layer)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (got[(layer * H + y) * W + x] != expected(layer, x, y))
                {
                    ++wrong;
                    ++wrongPerLayer[layer];
                }
    std::printf("%-22s %4d of %d wrong  (per layer:", what, wrong, L * W * H);
    for (int layer = 0; layer < L; ++layer) std::printf(" %d", wrongPerLayer[layer]);
    std::printf(")\n");
}

static void putIntoArray(hipArray_t arr, bool values)
{
    static float src[L][H][W];
    for (int l = 0; l < L; ++l)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) src[l][y][x] = values ? expected(l, x, y) : 0.0f;

    hipMemcpy3DParms parms{};
    parms.srcPtr = make_hipPitchedPtr(src, W * sizeof(float), W, H);
    parms.dstArray = arr;
    parms.extent = make_hipExtent(W, H, L);
    parms.kind = hipMemcpyHostToDevice;
    const hipError_t e = hipMemcpy3D(&parms);
    if (e != hipSuccess) std::printf("hipMemcpy3D failed: %s\n", hipGetErrorString(e));
}

int main()
{
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::printf("device %s\n", prop.name);
    std::printf("reported maxSurface2DLayered %d x %d x %d, maxTexture2D %d x %d\n\n",
                prop.maxSurface2DLayered[0], prop.maxSurface2DLayered[1], prop.maxSurface2DLayered[2],
                prop.maxTexture2D[0], prop.maxTexture2D[1]);

    hipArray_t arr = nullptr;
    hipChannelFormatDesc fd = hipCreateChannelDesc<float>();
    if (hipMalloc3DArray(&arr, &fd, make_hipExtent(W, H, L),
                         hipArrayLayered | hipArraySurfaceLoadStore) != hipSuccess)
    {
        std::printf("hipMalloc3DArray failed\n");
        return 1;
    }

    hipResourceDesc res{};
    res.resType = hipResourceTypeArray;
    res.res.array.array = arr;

    hipSurfaceObject_t surf;
    hipCreateSurfaceObject(&surf, &res);

    hipTextureDesc td{};
    td.normalizedCoords = 0;
    td.addressMode[0] = hipAddressModeClamp;
    td.addressMode[1] = hipAddressModeClamp;
    td.readMode = hipReadModeElementType;
    td.filterMode = hipFilterModePoint;
    hipTextureObject_t tex;
    hipCreateTextureObject(&tex, &res, &td, nullptr);

    float* dev = nullptr;
    hipMalloc(&dev, sizeof(float) * L * W * H);
    static float host[L * W * H];
    const dim3 block(32, 4), grid(1, 1);

    std::printf("A. array filled with hipMemcpy3D, read back two ways\n");
    putIntoArray(arr, true);
    hipMemset(dev, 0, sizeof(float) * L * W * H);
    hipLaunchKernelGGL(readWithTexture, grid, block, 0, 0, tex, dev);
    hipDeviceSynchronize();
    hipMemcpy(host, dev, sizeof(host), hipMemcpyDeviceToHost);
    report("   tex2DLayered", host);
    hipMemset(dev, 0, sizeof(float) * L * W * H);
    hipLaunchKernelGGL(readWithSurface, grid, block, 0, 0, surf, dev);
    hipDeviceSynchronize();
    hipMemcpy(host, dev, sizeof(host), hipMemcpyDeviceToHost);
    report("   surf2DLayeredread", host);

    std::printf("\nB. array zeroed, written with surf2DLayeredwrite, read with tex2DLayered\n");
    putIntoArray(arr, false);
    hipLaunchKernelGGL(writeWithSurface, grid, block, 0, 0, surf);
    hipDeviceSynchronize();
    hipMemset(dev, 0, sizeof(float) * L * W * H);
    hipLaunchKernelGGL(readWithTexture, grid, block, 0, 0, tex, dev);
    hipDeviceSynchronize();
    hipMemcpy(host, dev, sizeof(host), hipMemcpyDeviceToHost);
    report("   tex2DLayered", host);

    hipArray_t big = nullptr;
    const hipError_t e = hipMalloc3DArray(&big, &fd, make_hipExtent(8192, 8192, 3), hipArrayLayered);
    std::printf("\nC. 8192 x 8192 x 3 layered array, far past the reported limit: %s\n",
                hipGetErrorString(e));
    if (e == hipSuccess) hipFreeArray(big);
    return 0;
}
