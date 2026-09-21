// Cheshire: see simBlurGPU.hpp. CUDA dialect; the HIP build force-includes cheshire/cuda_to_hip.h.
#include "aliceVision/fuseCut/gpu/simBlurGPU.hpp"

#include <cuda_runtime.h>
#include <aliceVision/depthMap/cuda/hip/cheshire/devalloc.h>  // cheshire: bridge on both backends

// FMA contraction is off for this file so a*b+c rounds twice, as OIIO's does on the CPU. Without it
// the device fuses and 74.3 M of 244.6 M pixels land a few ULP away - the same 30 % the harness
// measures for a fused CPU variant. __fmul_rn and __fadd_rn do not prevent it: HIP defines them as
// the plain operators, so the compiler contracts them right back.
#pragma clang fp contract(off)

#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace aliceVision {
namespace fuseCut {
namespace gpu {

namespace {

// One thread per output pixel, in the loop order OIIO's result was matched with: kernel-major,
// float accumulator, the row bound tested once outside the column loop, and a true divide by the
// weight that landed inside the image. __fmul_rn and __fadd_rn keep the multiply and the add
// separately rounded, so the device's FMA cannot quietly change the arithmetic.
__global__ void blurKernel(const float* __restrict__ src,
                           float* __restrict__ dst,
                           int w,
                           int h,
                           const float* __restrict__ kern,
                           int kw,
                           int kh,
                           int kx0,
                           int ky0)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;

    float acc = 0.0f;
    float wsum = 0.0f;
    for (int ky = 0; ky < kh; ++ky)
    {
        const int sy = y + ky0 + ky;
        if (sy < 0 || sy >= h)
            continue;
        for (int kx = 0; kx < kw; ++kx)
        {
            const int sx = x + kx0 + kx;
            if (sx < 0 || sx >= w)
                continue;
            const float k = kern[ky * kw + kx];
            acc += src[(size_t)sy * w + sx] * k;
            wsum += k;
        }
    }
    // __fdiv_rn, not '/': the device's default float division is allowed to be approximate, and it
    // is - leaving it to the compiler put 74.3 M of 244.6 M pixels a few ULP away from OIIO, which
    // is what CHESHIRE_GPU_BLUR_CHECK=1 was built to notice.
    dst[(size_t)y * w + x] = wsum != 0.0f ? __fdiv_rn(acc, wsum) : 0.0f;
}

std::once_flag g_once;
bool g_available = false;

void probe()
{
    if (const char* e = std::getenv("CHESHIRE_GPU_BLUR"))
        if (e[0] == '0')
        {
            std::fprintf(stderr, "[cheshire] sim blur: disabled by CHESHIRE_GPU_BLUR=0, OIIO\n");
            return;
        }
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1)
    {
        std::fprintf(stderr, "[cheshire] sim blur: no GPU device, OIIO\n");
        return;
    }
    // Announced on the success path as well: a port that prints only when it is NOT used cannot
    // be shown to have run (the end-to-end gate's marker rule, docs/04).
    cudaDeviceProp p{};
    const char* name = (cudaGetDeviceProperties(&p, 0) == cudaSuccess) ? p.name : "device 0";
    std::fprintf(stderr, "[cheshire] sim blur: Gaussian on the GPU (%s; CHESHIRE_GPU_BLUR=0 for OIIO)\n", name);
    g_available = true;
}

}  // namespace

bool blurAvailable()
{
    std::call_once(g_once, probe);
    return g_available;
}

bool blurGaussian(const float* src, float* dst, int w, int h,
                  const float* kern, int kw, int kh, int kx0, int ky0)
{
    if (!blurAvailable() || w <= 0 || h <= 0 || kw <= 0 || kh <= 0)
        return false;

    const size_t nPix = (size_t)w * h;
    const size_t nKer = (size_t)kw * kh;
    float *dSrc = nullptr, *dDst = nullptr, *dKer = nullptr;
    bool ok = false;

    // Fusion calls this from one OpenMP thread per camera. Each call keeps its own stream so the
    // twelve do not serialise on the null stream, and its own buffers: a map is about 9 MB, so a
    // dozen in flight is a rounding error against the card's memory.
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return false;

    do
    {
        if (cheshire::devMalloc(&dSrc, nPix * sizeof(float)) != cudaSuccess) break;
        if (cheshire::devMalloc(&dDst, nPix * sizeof(float)) != cudaSuccess) break;
        if (cheshire::devMalloc(&dKer, nKer * sizeof(float)) != cudaSuccess) break;
        if (cudaMemcpyAsync(dSrc, src, nPix * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess) break;
        if (cudaMemcpyAsync(dKer, kern, nKer * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess) break;

        const dim3 block(16, 16);
        const dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
        blurKernel<<<grid, block, 0, stream>>>(dSrc, dDst, w, h, dKer, kw, kh, kx0, ky0);
        if (cudaGetLastError() != cudaSuccess) break;
        if (cudaMemcpyAsync(dst, dDst, nPix * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess) break;
        if (cudaStreamSynchronize(stream) != cudaSuccess) break;
        ok = true;
    } while (false);

    cheshire::devFree(dSrc);
    cheshire::devFree(dDst);
    cheshire::devFree(dKer);
    cudaStreamDestroy(stream);
    return ok;
}

}  // namespace gpu
}  // namespace fuseCut
}  // namespace aliceVision
