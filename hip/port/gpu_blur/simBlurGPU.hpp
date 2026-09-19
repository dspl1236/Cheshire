// Cheshire: the similarity-map gaussian of Meshing's point-cloud fusion, on the GPU.
//
// PointCloud::createDensePointCloud calls it once per camera through
// imageAlgo::convolveImage(simMap, tmp, "gaussian", simGaussianSizeInit, simGaussianSizeInit),
// which hands OIIO a make_kernel("gaussian", 10, 10) - an 11x11 kernel, 121 taps - and calls the
// general ImageBufAlgo::convolve. OIIO does not exploit separability, so that is 121 multiply-adds
// per pixel over 2016x1134 maps, 107 times.
//
// Profiling the block it lives in (CHESHIRE_FUSION_PROFILE=1) put 108.7 of its 180.8 thread-seconds
// here - 60 %, against 65.5 for all three EXR reads together. The block is called "Load depth maps
// and add points" and the loading is not the cost. Isolated, one call is 0.144 s single-threaded;
// in the pipeline it costs about 1.0 thread-second per camera, because twelve threads each sweep a
// 121-tap stencil over a 9 MB image at once and the CPU's memory system is the limit. Moving it to
// the device removes that contention as much as it removes the arithmetic.
//
// What the kernel computes is OIIO's result, not an approximation of it. Three things had to be
// matched, the first two established against OIIO's own output on real similarity maps
// (scratchpad/gaussbench.cpp, which tries each candidate and counts the pixels that differ):
//   - at the borders OIIO divides by the kernel weight that actually landed inside the image, so a
//     window of constant 1.0 comes back as exactly 1.0 rather than 0.343151. Without this, 25,136
//     border pixels are wrong and no interior pixel is.
//   - the accumulation is float, in kernel-major order, with a true divide. Against 2,286,144
//     pixels: that formulation leaves 2 differing, a reciprocal multiply 4,269, a fused
//     multiply-add 731,102, and a double accumulator 1,845,868. OIIO does none of the last three.
//   - FMA contraction has to be off, which is what the pragma in the .cu is for. HIP defines
//     __fmul_rn and __fadd_rn as the plain operators, so writing those does NOT prevent it; the
//     device fused anyway and 74.3 M of 244.6 M pixels came out a few ULP off - 30 %, which is the
//     same share the harness measures for a fused CPU variant, and how it was identified.
//
// With all three, 244,617,279 of 244,617,408 pixels across the 107-photo engine bay match bit for
// bit; the other 129 differ by at most 4.77e-07 and none of them changes a decision. The point
// cloud is identical: 6,896,447 points after the first filter as OIIO gives, 3,388,702 after the
// second, and the tetrahedralisation input checksums af3a3cce376e3f12 either way. The block goes
// from 17.69 s to 3.06 s.
//
// CHESHIRE_GPU_BLUR_CHECK=1 runs OIIO as well and reports how far apart they were, so that claim is
// checkable on other data rather than taken on trust. It roughly doubles the block, since it does
// the work twice.
//
// CUDA dialect; the HIP build force-includes cheshire/cuda_to_hip.h. CHESHIRE_GPU_BLUR=0 disables.
#pragma once

namespace aliceVision {
namespace fuseCut {
namespace gpu {

// True if a device is present and CHESHIRE_GPU_BLUR is not 0. Checked once.
bool blurAvailable();

// dst = src convolved with kern, each output divided by the covered kernel weight.
// src and dst are w*h floats, row-major; kern is kw*kh floats whose origin sits at (kx0, ky0)
// relative to the output pixel (OIIO centres it, so those are negative). dst may not alias src.
// Returns false if the work did not run, in which case the caller keeps OIIO's path.
bool blurGaussian(const float* src, float* dst, int w, int h,
                  const float* kern, int kw, int kh, int kx0, int ky0);

}  // namespace gpu
}  // namespace fuseCut
}  // namespace aliceVision
