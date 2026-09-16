// Cheshire: DepthMapFilter's "precomputing groups" pass on the GPU (see Fuser::filterGroupsRC).
//
// For a reference camera rc, every pixel of every neighbour camera tc's depth map is back-projected
// to 3D, projected into rc, and if its depth agrees with rc's depth at that pixel (within a pixel-size
// tolerance computed from the two cameras' geometry) the rc pixel gets a vote from tc. The output is
// the number of neighbour cameras that voted for each rc pixel (numOfModalsMap). Upstream does it on
// the CPU per pixel, including an RQ decomposition of both projection matrices per pixel; that was
// 93 % of a 115 s node on 107 photos. Here the per-camera decompositions happen once on the host and
// the per-pixel work runs as one thread per tc pixel, in double precision with FMA contraction off,
// replicating upstream's expression order so the counts are bit-identical.
//
// CUDA dialect; the HIP build force-includes cheshire/cuda_to_hip.h. CHESHIRE_GPU_FILTER=0 disables.
#pragma once

namespace aliceVision {
namespace fuseCut {
namespace gpu {

bool available();

// Everything the kernel needs about one camera, at the depth map's scale (as MultiViewParams holds
// it): the projection matrix, its inverse rotation-intrinsics product for back-projection, the
// centre, and the values decomposeProjectionMatrix() produces for the epipolar step (upstream
// recomputes them per pixel from the same matrix; computed once here, same code, same bits).
struct CamGeom {
    double P[12];    // camArr[cam]: m11 m12 m13 m14 / m21 .. / m31 ..
    double iCam[9];  // iCamArr[cam]: m11 .. m33
    double C[3];     // CArr[cam]
    double riP[9];   // decomposeProjectionMatrix(P).iPo (= iR * iK)
    double rC[3];    // decomposeProjectionMatrix(P).Co
    int w, h;        // getWidth(cam), getHeight(cam)
};

class GroupFilter
{
  public:
    GroupFilter();
    ~GroupFilter();
    GroupFilter(const GroupFilter&) = delete;
    GroupFilter& operator=(const GroupFilter&) = delete;

    // rc's depth and similarity maps (w*h floats each) and geometry; resets the vote counts.
    bool setRc(const float* depth, const float* sim, const CamGeom& rc);
    // one neighbour camera: its depth map (tw*th floats) and geometry; adds its votes.
    bool accumulate(const float* tcDepth, const CamGeom& tc, float pixToleranceFactor, int pixSizeBall, int pixSizeBallWSP);
    // numOfModalsMap, w*h bytes (number of neighbour cameras that voted for the pixel).
    bool result(unsigned char* numOfModals);
    // Diagnostics: the intermediate values for one tc pixel, to line up against the CPU helpers.
    // out = {px, py, pixDepth, avRcTc, avRc, pixSize, p.x, p.y, p.z, rcDepthAtCell}
    bool probe(const float* tcDepth, const CamGeom& tc, int x, int y, float pixToleranceFactor, int pixSizeBall, int pixSizeBallWSP, double out[10]);
    // Diagnostics: number of rc pixels that received at least one vote from the last accumulate().
    bool voteCount(long long* count);

  private:
    struct Impl;
    Impl* impl_;
};

}  // namespace gpu
}  // namespace fuseCut
}  // namespace aliceVision
