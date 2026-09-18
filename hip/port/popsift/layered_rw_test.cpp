// Cheshire: does a value written to layer L of a layered array come back from a read of layer L?
//
// PopSIFT stores each octave's gaussian levels as the layers of one layered array. It writes them
// with surf2DLayeredwrite and reads them back with tex2DLayered, and the descriptor stage reads the
// level its keypoint was found at. HIP's layered surface WRITE was already found to target a mip
// level instead of an array layer (see popsift_hip.h); this checks the read side of the same pair,
// using exactly the texture descriptors sift_octave.cu builds.
#include "popsift_hip.h"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

#define CHK(call)                                                                                  \
    do {                                                                                           \
        hipError_t e = (call);                                                                     \
        if (e != hipSuccess) {                                                                     \
            std::printf("%s failed at line %d: %s%c", #call, __LINE__, hipGetErrorString(e), 10);   \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

static const int W = 64, H = 8, L = 4;

// the value that layer/x/y should hold; distinct per cell and per layer
__host__ __device__ static inline float expected(int layer, int x, int y)
{
    return (float)(layer * 100000 + y * 1000 + x);
}

__global__ void fill(hipSurfaceObject_t surf)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    for (int layer = 0; layer < L; ++layer)
        surf2DLayeredwrite(expected(layer, x, y), surf, x * 4, y, layer, cudaBoundaryModeZero);
}

// read exactly as popsift's readTex does: pixel coordinates plus 0.5, layer passed as a float
__device__ static inline float readTexLikePopsift(hipTextureObject_t tex, float x, float y, float z)
{
    return tex2DLayered<float>(tex, x + 0.5f, y + 0.5f, z);
}

__global__ void readBack(hipTextureObject_t tex, float* out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    for (int layer = 0; layer < L; ++layer)
        out[(size_t)layer * W * H + (size_t)y * W + x] = readTexLikePopsift(tex, (float)x, (float)y, (float)layer);
}

// the same read with the layer as an int, in case the float conversion is what differs
__global__ void readBackInt(hipTextureObject_t tex, float* out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    for (int layer = 0; layer < L; ++layer)
        out[(size_t)layer * W * H + (size_t)y * W + x] = tex2DLayered<float>(tex, x + 0.5f, y + 0.5f, layer);
}

// and straight off the surface, which bypasses the sampler entirely
__global__ void readBackSurf(hipSurfaceObject_t surf, float* out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    for (int layer = 0; layer < L; ++layer)
    {
        float v = 0.0f;
        surf2DLayeredread(&v, surf, x * 4, y, layer);  // HIP takes no boundary mode
        out[(size_t)layer * W * H + (size_t)y * W + x] = v;
    }
}

static void report(const char* what, const std::vector<float>& got)
{
    int wrong = 0, zero = 0;
    int perLayerWrong[L] = {0};
    float firstGot = 0.0f, firstWant = 0.0f;
    int firstLayer = -1;
    for (int layer = 0; layer < L; ++layer)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
            {
                const float want = expected(layer, x, y);
                const float have = got[(size_t)layer * W * H + (size_t)y * W + x];
                if (have != want)
                {
                    ++wrong;
                    ++perLayerWrong[layer];
                    if (firstLayer < 0) { firstLayer = layer; firstGot = have; firstWant = want; }
                }
                if (have == 0.0f) ++zero;
            }
    std::printf("%-14s wrong %5d of %d, zeros %5d, per layer", what, wrong, L * W * H, zero);
    for (int layer = 0; layer < L; ++layer) std::printf(" %d", perLayerWrong[layer]);
    if (wrong) std::printf("  first: layer %d got %.0f want %.0f", firstLayer, firstGot, firstWant);
    std::printf("%c", 10);
}

int main()
{
    hipArray_t arr = nullptr;
    hipChannelFormatDesc fd = hipCreateChannelDesc<float>();
    CHK(hipMalloc3DArray(&arr, &fd, make_hipExtent(W, H, L), hipArrayLayered | hipArraySurfaceLoadStore));

    hipResourceDesc res{};
    res.resType = hipResourceTypeArray;
    res.res.array.array = arr;

    hipSurfaceObject_t surf;
    CHK(hipCreateSurfaceObject(&surf, &res));

    // sift_octave.cu's descriptors, both filter modes
    hipTextureDesc td{};
    td.normalizedCoords = 0;
    td.addressMode[0] = hipAddressModeClamp;
    td.addressMode[1] = hipAddressModeClamp;
    td.addressMode[2] = hipAddressModeClamp;
    td.readMode = hipReadModeElementType;
    td.filterMode = hipFilterModePoint;
    hipTextureObject_t texPoint;
    CHK(hipCreateTextureObject(&texPoint, &res, &td, nullptr));
    td.filterMode = hipFilterModeLinear;
    hipTextureObject_t texLinear;
    CHK(hipCreateTextureObject(&texLinear, &res, &td, nullptr));

    float* dev = nullptr;
    CHK(hipMalloc(&dev, sizeof(float) * L * W * H));

    const dim3 block(16, 4), grid((W + 15) / 16, (H + 3) / 4);
    hipLaunchKernelGGL(fill, grid, block, 0, 0, surf);
    CHK(hipDeviceSynchronize());

    std::vector<float> got((size_t)L * W * H);
    struct { const char* name; int kind; hipTextureObject_t tex; } runs[] = {
        {"surface read", 2, 0}, {"tex point", 0, texPoint}, {"tex point int", 1, texPoint},
        {"tex linear", 0, texLinear}, {"tex linear int", 1, texLinear},
    };
    for (auto& r : runs)
    {
        CHK(hipMemset(dev, 0, sizeof(float) * L * W * H));
        if (r.kind == 0)      hipLaunchKernelGGL(readBack, grid, block, 0, 0, r.tex, dev);
        else if (r.kind == 1) hipLaunchKernelGGL(readBackInt, grid, block, 0, 0, r.tex, dev);
        else                  hipLaunchKernelGGL(readBackSurf, grid, block, 0, 0, surf, dev);
        CHK(hipDeviceSynchronize());
        CHK(hipMemcpy(got.data(), dev, sizeof(float) * L * W * H, hipMemcpyDeviceToHost));
        report(r.name, got);
    }
    std::printf("done%c", 10);
    return 0;
}
