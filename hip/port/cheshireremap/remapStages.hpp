// CheshireRemap: the per-pixel functor, compiled for the device (remapGPU.cu launches it through
// cheshire::gpu::forEachKernel) and for the host (the CPU backend of hip/tests/cheshireremap).
//
// It is AliceVision's Sampler2d<SamplerLinear>::operator()(src, y, x) on an Image<RGBAfColor>
// (image/Sampler.hpp), operation for operation:
//   dx = double(x) - floor(x), dy likewise                      exact
//   coefs {1 - dx, dx} and {1 - dy, dy}; w = coefX[j] * coefY[i]  rounded products
//   for the neighbours (gridY + i, gridX + j), i then j, those inside the image:
//     res += double(pixel) * w per channel, totalWeight += w     each product and each sum rounded
//   totalWeight <= 0.2: the pixel at (floor(y), floor(x)) clamped into the image, unchanged
//   else res /= totalWeight when it is not 1, and each channel rounded to float
// No operation is fused into a multiply-add, as on the host (each is its own statement there, and the
// Linux build targets no FMA): the pragma below for clang - the HIP build, and the host pass of the
// checks - and --fmad=false for nvcc (CMakeLists.txt). Not HIP's __dadd_rn and friends: they are plain
// operators in HIP's own headers, outside this pragma, and clang's default for HIP fuses across them.
#pragma once

#include <cmath>
#include <cstddef>

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define CHESHIRE_REMAP_HD __host__ __device__
#else
#define CHESHIRE_REMAP_HD
#endif

namespace cheshire {
namespace remap {

struct UndistortPixel
{
    const float* src;  // srcWidth x srcHeight x 4
    int srcWidth;
    int srcHeight;
    const float* map;  // 2 per output pixel: x, y; x NaN outside the source
    float* dst;        // 4 per output pixel
    float fill[4];

    CHESHIRE_REMAP_HD void operator()(size_t i) const
    {
        float* o = dst + 4 * i;
        const float x = map[2 * i];
        const float y = map[2 * i + 1];
        if (x != x)
        {
            o[0] = fill[0];
            o[1] = fill[1];
            o[2] = fill[2];
            o[3] = fill[3];
            return;
        }
        const float fx = floorf(x);
        const float fy = floorf(y);
        const double dx = static_cast<double>(x) - static_cast<double>(fx);
        const double dy = static_cast<double>(y) - static_cast<double>(fy);
        const double coefX[2] = {1.0 - dx, dx};
        const double coefY[2] = {1.0 - dy, dy};
        const int gridX = static_cast<int>(fx);
        const int gridY = static_cast<int>(fy);
        double r = 0.0, g = 0.0, b = 0.0, a = 0.0, total = 0.0;
        for (int k = 0; k < 2; ++k)
        {
            const int row = gridY + k;
            if (row < 0 || row >= srcHeight)
                continue;
            for (int j = 0; j < 2; ++j)
            {
                const int col = gridX + j;
                if (col < 0 || col >= srcWidth)
                    continue;
                const double w = coefX[j] * coefY[k];
                const float* p = src + 4 * (static_cast<size_t>(row) * static_cast<size_t>(srcWidth) + static_cast<size_t>(col));
                const double pr = static_cast<double>(p[0]) * w;
                const double pg = static_cast<double>(p[1]) * w;
                const double pb = static_cast<double>(p[2]) * w;
                const double pa = static_cast<double>(p[3]) * w;
                r = r + pr;
                g = g + pg;
                b = b + pb;
                a = a + pa;
                total = total + w;
            }
        }
        if (total <= 0.2)
        {
            int row = static_cast<int>(fy);
            int col = static_cast<int>(fx);
            if (row < 0)
                row = 0;
            if (col < 0)
                col = 0;
            if (row >= srcHeight)
                row = srcHeight - 1;
            if (col >= srcWidth)
                col = srcWidth - 1;
            const float* p = src + 4 * (static_cast<size_t>(row) * static_cast<size_t>(srcWidth) + static_cast<size_t>(col));
            o[0] = p[0];
            o[1] = p[1];
            o[2] = p[2];
            o[3] = p[3];
            return;
        }
        if (total != 1.0)
        {
            r = r / total;
            g = g / total;
            b = b / total;
            a = a / total;
        }
        o[0] = static_cast<float>(r);
        o[1] = static_cast<float>(g);
        o[2] = static_cast<float>(b);
        o[3] = static_cast<float>(a);
    }
};

}  // namespace remap
}  // namespace cheshire
