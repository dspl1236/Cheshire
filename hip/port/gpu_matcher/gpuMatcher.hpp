// Cheshire: GPU brute-force k-NN (k = 2) for descriptor matching. Host-facing API, no GPU types.
//
// AliceVision's FeatureMatching spends >90 % of its time in "Regions Matching": for every image
// pair, the 2 nearest neighbours of each query descriptor in the other image's descriptors (SIFT /
// DSP-SIFT: 128 x uint8; AKAZE and friends: 64/128 x float), then Lowe's ratio test. Upstream does
// it on the CPU with a FLANN kd-tree (ANN_L2, approximate, one pair at a time) or an exact brute
// force; on a 107-photo scan that stage was 181 s of a 196 s chunk. This is the exact search on the
// GPU: squared L2, best two per query. Same numbers as BRUTE_FORCE_L2, not the kd-tree's
// approximation.
//
// Compiled as CUDA (nvcc) or as HIP (cheshire/cuda_to_hip.h force-included by the HIP build), so the
// same source serves an NVIDIA and an AMD build. CHESHIRE_GPU_MATCHER=0 disables it at run time.
#pragma once
#include <cstddef>

namespace aliceVision {
namespace matching {
namespace gpu {

// A usable device exists and the matcher is not disabled (cached after the first call).
bool available();
// Descriptor lengths the kernels are instantiated for.
bool supportsDim(int dim);

class KnnMatcher
{
  public:
    KnnMatcher();
    ~KnnMatcher();
    KnnMatcher(const KnnMatcher&) = delete;
    KnnMatcher& operator=(const KnnMatcher&) = delete;

    // Upload the database: `rows` descriptors of `dim` scalars, uint8 or float32, row-major.
    bool build(const void* data, int rows, int dim, bool isFloat);
    int rows() const;
    // 2 nearest neighbours (squared L2) of each of `nbQuery` descriptors of the database's type/dim.
    // idx[2q], idx[2q+1]: database row of the best and second best; dist likewise (squared L2 as
    // float). Ties resolve to the lower row index, like a stable CPU scan.
    bool search2(const void* queries, int nbQuery, int* idx, float* dist);

  private:
    struct Impl;
    Impl* impl_;
};

}  // namespace gpu
}  // namespace matching
}  // namespace aliceVision
