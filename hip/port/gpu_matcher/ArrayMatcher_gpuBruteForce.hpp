// Cheshire: AliceVision ArrayMatcher backed by the GPU brute-force 2-NN (gpuMatcher.hpp).
// Drop-in for ArrayMatcher_bruteForce / ArrayMatcher_kdtreeFlann in RegionsMatcher: exact squared-L2
// neighbours, so the ratio test sees the same numbers as BRUTE_FORCE_L2 (the kd-tree is approximate).
#pragma once
#include <aliceVision/matching/ArrayMatcher.hpp>
#include <aliceVision/matching/gpu/gpuMatcher.hpp>
#include <type_traits>
#include <vector>

namespace aliceVision {
namespace matching {

template<typename Scalar, typename Metric>
class ArrayMatcher_gpuBruteForce : public ArrayMatcher<Scalar, Metric>
{
  public:
    typedef typename Metric::ResultType DistanceType;
    static_assert(std::is_same<Scalar, unsigned char>::value || std::is_same<Scalar, float>::value, "uint8 or float32 descriptors");

    ArrayMatcher_gpuBruteForce() {}
    virtual ~ArrayMatcher_gpuBruteForce() {}

    bool Build(std::mt19937& /*rng*/, const Scalar* dataset, int nbRows, int dimension)
    {
        return matcher_.build(dataset, nbRows, dimension, std::is_same<Scalar, float>::value);
    }

    bool SearchNeighbour(const Scalar* query, int* indice, DistanceType* distance)
    {
        int idx[2]; float d[2];
        if (!matcher_.search2(query, 1, idx, d)) return false;
        *indice = idx[0]; *distance = DistanceType(d[0]);
        return true;
    }

    bool SearchNeighbours(const Scalar* query, int nbQuery, IndMatches* pvec_indices, std::vector<DistanceType>* pvec_distances, size_t NN)
    {
        if (NN != 2 || nbQuery < 1 || matcher_.rows() < 2) return false;
        idx_.resize(size_t(nbQuery) * 2); dist_.resize(size_t(nbQuery) * 2);
        if (!matcher_.search2(query, nbQuery, idx_.data(), dist_.data())) return false;
        pvec_indices->resize(size_t(nbQuery) * 2);
        pvec_distances->resize(size_t(nbQuery) * 2);
        for (int q = 0; q < nbQuery; ++q)
            for (int k = 0; k < 2; ++k) {
                (*pvec_indices)[size_t(q) * 2 + k] = IndMatch(q, idx_[size_t(q) * 2 + k]);
                (*pvec_distances)[size_t(q) * 2 + k] = DistanceType(dist_[size_t(q) * 2 + k]);
            }
        return true;
    }

  private:
    gpu::KnnMatcher matcher_;
    std::vector<int> idx_;
    std::vector<float> dist_;
};

}  // namespace matching
}  // namespace aliceVision
