// Cheshire: Texturing's per-camera work on the GPU (Texturing::generateTexturesSubSet).
//
// For every camera upstream loads the image, builds a Laplacian pyramid (OpenImageIO gaussian
// resize + bilinear difference) and rasterises every triangle that camera contributes to into
// per-atlas, per-band accumulators (weighted colour sums + weights). Here the host still reads the
// image (EXR, exposure and colour space are OpenImageIO / OCIO work); the pyramid, the rasterisation
// and the final normalise-and-fuse run on the GPU with the accumulators resident in VRAM, one
// thread block per triangle. The arithmetic is upstream's, transcribed: OpenImageIO 3.0's
// separable resize with its Gaussian filter and fast_exp polynomial, geogram's
// point_triangle_squared_distance for the inside test, the same bilinear interpolation and the same
// float accumulation per camera, so texels touched by one triangle per camera come out bit-identical
// and chart-edge texels (two triangles in one camera, racy on the CPU too) differ by add order.
//
// CUDA dialect; the HIP build force-includes cheshire/cuda_to_hip.h. CHESHIRE_GPU_TEX=0 disables.
#pragma once
#include <cstddef>
#include <cstdint>

namespace aliceVision {
namespace mesh {
namespace gpu {

bool texAvailable();

// VRAM one atlas slot takes: textureSide^2 texels x nbBand levels x (3 colour floats + 1 weight).
std::size_t atlasBytes(unsigned textureSide, int nbBand);
// Atlas slots that fit in the device memory next to one image, its pyramid and the triangle
// tables (0: no GPU).
int maxAtlasSlots(unsigned textureSide, int nbBand, int imgW, int imgH, std::uint32_t nbTris);

class Texturer
{
public:
    Texturer();
    ~Texturer();
    Texturer(const Texturer&) = delete;
    Texturer& operator=(const Texturer&) = delete;

    bool init(int nbSlots, unsigned textureSide, int nbBand, int downscale, int maxImgW, int maxImgH);
    // Per mesh triangle: the three 3D vertices (9 doubles) and the three texture-pixel positions
    // (6 doubles: uv remapped into the UDIM tile times textureSide), indexed by triangle id.
    bool setTriangles(std::uint32_t nbTris, const double* pts, const double* texPix);
    // The camera's image (RGB float, interleaved, row-major) and its 3x4 projection matrix
    // (m11 m12 m13 m14 m21 ... m34); builds the Laplacian pyramid.
    bool setCamera(const float* rgb, int w, int h, const double* P);
    // Rasterise this camera's contributions to one atlas slot at one frequency band.
    bool raster(int slot, int band, const std::uint32_t* triIds, const float* scores, std::uint32_t n);
    // Final (average) colour and band fusion for a slot, downloaded into rgb (3 per texel) and
    // count (1 where a texel has colour, else 0), then the slot is cleared for the next chunk.
    bool finish(int slot, float* rgb, float* count);

    // seconds spent, for CHESHIRE_GPU_TEX_LOG
    double uploadSec = 0, pyramidSec = 0, rasterSec = 0, finishSec = 0;

private:
    struct Impl;
    Impl* p;
};

}  // namespace gpu
}  // namespace mesh
}  // namespace aliceVision
