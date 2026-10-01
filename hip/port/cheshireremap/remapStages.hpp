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
        sample(src, srcWidth, srcHeight, x, y, o);
    }

    // The sampler at (x, y) into o (4 floats); RemapInterPixel's bilinear branch calls it too (step 9i)
    CHESHIRE_REMAP_HD static void sample(const float* src, int srcWidth, int srcHeight, float x, float y, float* o)
    {
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

// ExportImages' warp (step 9i): AliceVision's remapInter (image/remap.hpp) on an Image<RGBAfColor>, operation for
// operation. Per output pixel it measures how far the map moves between the pixel's neighbours, in double, as the loop
// does. Where the map shrinks the source in both directions it averages the source pixels the output pixel covers,
// weighted by their overlap: each product in double rounded to float (Rgba's operator*), summed per channel in float
// (Eigen's +=), the sum divided by the total weight rounded to float (Eigen's /= takes the pixel's scalar type). Elsewhere
// it is the bilinear sampler above at the coordinates rounded to float - Sampler2d takes floats - or the fill colour
// where Image::contains, which takes the coordinates truncated to int, says the pixel is outside; a NaN or out-of-range
// coordinate, which the host's conversion to int puts outside, takes the fill colour too. std::min and std::max are
// written out as the comparisons they are.
struct RemapInterPixel
{
    const float* src;   // srcWidth x srcHeight x 4
    int srcWidth;
    int srcHeight;
    const double* map;  // 2 per output pixel (x, y), width x height, row-major
    int width;
    int height;
    float* dst;         // 4 per output pixel
    float fill[4];

    CHESHIRE_REMAP_HD static double dmax(double a, double b) { return (a < b) ? b : a; }  // std::max(a, b)
    CHESHIRE_REMAP_HD static double dmin(double a, double b) { return (b < a) ? b : a; }  // std::min(a, b)

    CHESHIRE_REMAP_HD void operator()(size_t idx) const
    {
        const size_t w = static_cast<size_t>(width);
        const int i = static_cast<int>(idx / w);
        const int j = static_cast<int>(idx - static_cast<size_t>(i) * w);
        float* o = dst + 4 * idx;
        const double x = map[2 * idx];
        const double y = map[2 * idx + 1];

        double scaleX = 1.0;
        if (j < width - 1 && j > 0)
            scaleX = fabs(map[2 * (idx + 1)] - map[2 * (idx - 1)]) * 0.5;
        else if (j < width - 1)
            scaleX = fabs(map[2 * (idx + 1)] - x);
        else if (j > 0)
            scaleX = fabs(x - map[2 * (idx - 1)]);

        double scaleY = 1.0;
        if (i < height - 1 && i > 0)
            scaleY = fabs(map[2 * (idx + w) + 1] - map[2 * (idx - w) + 1]) * 0.5;
        else if (i < height - 1)
            scaleY = fabs(map[2 * (idx + w) + 1] - y);
        else if (i > 0)
            scaleY = fabs(y - map[2 * (idx - w) + 1]);

        if (scaleX > 1.0 && scaleY > 1.0)
        {
            // downsampling: the area the output pixel covers, each source pixel weighted by its overlap
            const double halfWidth = scaleX * 0.5;
            const double halfHeight = scaleY * 0.5;
            const double x1 = x - halfWidth;
            const double x2 = x + halfWidth;
            const double y1 = y - halfHeight;
            const double y2 = y + halfHeight;
            const int ix1 = static_cast<int>(x1);
            const int ix2 = static_cast<int>(ceil(x2));
            const int iy1 = static_cast<int>(y1);
            const int iy2 = static_cast<int>(ceil(y2));
            float sr = 0.0f, sg = 0.0f, sb = 0.0f, sa = 0.0f;
            double wsum = 0.0;
            for (int by = iy1; by < iy2; by++)
            {
                if (by < 0 || by >= srcHeight)
                    continue;
                const double yocc = dmax(0.0, dmin(static_cast<double>(by + 1), y2) - dmax(static_cast<double>(by), y1));
                for (int bx = ix1; bx < ix2; bx++)
                {
                    if (bx < 0 || bx >= srcWidth)
                        continue;
                    const double xocc = dmax(0.0, dmin(static_cast<double>(bx + 1), x2) - dmax(static_cast<double>(bx), x1));
                    const float* p = src + 4 * (static_cast<size_t>(by) * static_cast<size_t>(srcWidth) + static_cast<size_t>(bx));
                    const double weight = yocc * xocc;
                    const float pr = static_cast<float>(static_cast<double>(p[0]) * weight);
                    const float pg = static_cast<float>(static_cast<double>(p[1]) * weight);
                    const float pb = static_cast<float>(static_cast<double>(p[2]) * weight);
                    const float pa = static_cast<float>(static_cast<double>(p[3]) * weight);
                    sr = sr + pr;
                    sg = sg + pg;
                    sb = sb + pb;
                    sa = sa + pa;
                    wsum = wsum + weight;
                }
            }
            if (wsum > 0.0)
            {
                const float ws = static_cast<float>(wsum);
                sr = sr / ws;
                sg = sg / ws;
                sb = sb / ws;
                sa = sa / ws;
            }
            o[0] = sr;
            o[1] = sg;
            o[2] = sb;
            o[3] = sa;
            return;
        }

        // not downsampling: the bilinear sampler where Image::contains(int(y), int(x)), the fill colour elsewhere
        const bool inRange = x > -2147483648.0 && x < 2147483648.0 && y > -2147483648.0 && y < 2147483648.0;  // false for NaN
        const int cx = inRange ? static_cast<int>(x) : -1;
        const int cy = inRange ? static_cast<int>(y) : -1;
        if (!(0 <= cx && cx < srcWidth && 0 <= cy && cy < srcHeight))
        {
            o[0] = fill[0];
            o[1] = fill[1];
            o[2] = fill[2];
            o[3] = fill[3];
            return;
        }
        UndistortPixel::sample(src, srcWidth, srcHeight, static_cast<float>(x), static_cast<float>(y), o);
    }
};

}  // namespace remap
}  // namespace cheshire
