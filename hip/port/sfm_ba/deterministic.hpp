// This file is part of the Cheshire project (MPL-2.0), a patch set over AliceVision.
// Copied by scripts/apply_hip_patch.py (step 5n) to src/aliceVision/sfm/deterministic.hpp.
//
// Incremental SfM, reproducible. Upstream seeds one std::mt19937 (--randomSeed, default 5489) and
// then draws from it inside OpenMP loops - resection localises views in parallel and triangulation
// runs LO-RANSAC per track in parallel, both through the shared generator - so the draw each task
// gets follows thread scheduling, and two runs of the same input differ. Removing that is not
// enough: two single-threaded runs still differ in the last bits, because Ceres orders the
// parameter blocks of an elimination group by pointer (ParameterBlockOrdering is
// std::map<int, std::set<double*>>), the blocks live in std::map nodes, and the Windows heap hands
// out addresses in an order that changes from run to run; the elimination order changes the
// rounding of the reduced camera system, and the trajectory drifts from there.
//
// Three things, always on unless switched off:
//   * a generator per task, derived from (seed, task kind, task id, resection pass) - the resection
//     of a view and the triangulation of a track draw the same samples whichever thread runs them
//     and whenever it runs them (CHESHIRE_SFM_TASK_SEED=0 restores the shared generator);
//   * landmark parameter blocks in one contiguous array in key order (OrderedBlocks), and one
//     ordering group per pose, rig sub-pose, intrinsic and distortion block numbered by key, so
//     Ceres' order is the key order, never the address order;
//   * a total order where a sort broke ties by arrival (the next-best-views ranking).
// What is left is Ceres' own multi-threading: with num_threads > 1 the Schur eliminator adds the
// chunks' contributions to the reduced matrix in arrival order. CHESHIRE_SFM_DETERMINISTIC=1 runs
// bundle adjustment on one thread (CHESHIRE_BA_THREADS=n sets it explicitly); the OpenMP loops of
// SfM itself keep their threads, since the points above make them order-independent.
#pragma once

#include <aliceVision/types.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aliceVision {
namespace sfm {
namespace cheshire {

inline bool sfmDeterministic()
{
    static const bool on = [] {
        const char* v = std::getenv("CHESHIRE_SFM_DETERMINISTIC");
        return v != nullptr && std::string(v) != "0";
    }();
    return on;
}

inline bool taskSeedEnabled()
{
    static const bool on = [] {
        const char* v = std::getenv("CHESHIRE_SFM_TASK_SEED");
        return v == nullptr || std::string(v) != "0";
    }();
    return on;
}

// The thread count bundle adjustment hands to Ceres: CHESHIRE_BA_THREADS if set, else 1 in the
// deterministic mode, else what the caller asked for.
inline unsigned baThreads(unsigned requested)
{
    static const long forced = [] {
        const char* v = std::getenv("CHESHIRE_BA_THREADS");
        return v != nullptr ? std::strtol(v, nullptr, 10) : 0L;
    }();
    if (forced > 0)
        return static_cast<unsigned>(forced);
    if (sfmDeterministic())
        return 1u;
    return requested;
}

inline std::uint64_t mix64(std::uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// A generator for one task: the same (seed, kind, id, pass) gives the same draws on any thread.
inline std::mt19937 taskGenerator(unsigned seed, unsigned kind, std::uint64_t id, std::uint64_t pass)
{
    std::uint64_t h = mix64(seed);
    h = mix64(h ^ (static_cast<std::uint64_t>(kind) << 56) ^ id);
    h = mix64(h ^ pass);
    return std::mt19937(static_cast<std::uint32_t>(h ^ (h >> 32)));
}

// Parameter blocks in one contiguous array, in insertion (key) order, with a map for lookup. The
// interface is the part of std::map<IndexT, Block> BundleAdjustmentCeres uses, and the elements are
// std::pair<const IndexT, Block> as a map's are, so `for (auto& [id, block] : blocks)` reads the
// same. reserve() must be called with an upper bound first: the array never reallocates after that,
// so the pointers Ceres holds stay valid and increase with insertion order.
template<typename Block>
class OrderedBlocks
{
  public:
    using value_type = std::pair<const IndexT, Block>;
    using iterator = typename std::vector<value_type>::iterator;
    using const_iterator = typename std::vector<value_type>::const_iterator;

    void reserve(std::size_t n) { _data.reserve(n); }
    void clear()
    {
        _data.clear();
        _index.clear();
    }
    std::size_t size() const { return _data.size(); }
    bool empty() const { return _data.empty(); }

    Block& operator[](IndexT id)
    {
        auto it = _index.find(id);
        if (it == _index.end())
        {
            if (_data.size() == _data.capacity())
                throw std::runtime_error("cheshire: OrderedBlocks holds more blocks than were reserved");
            _data.emplace_back(id, Block{});
            it = _index.emplace(id, _data.size() - 1).first;
        }
        return _data[it->second].second;
    }
    Block& at(IndexT id) { return _data[_index.at(id)].second; }
    const Block& at(IndexT id) const { return _data[_index.at(id)].second; }
    iterator find(IndexT id)
    {
        auto it = _index.find(id);
        return it == _index.end() ? _data.end() : _data.begin() + static_cast<std::ptrdiff_t>(it->second);
    }
    const_iterator find(IndexT id) const
    {
        auto it = _index.find(id);
        return it == _index.end() ? _data.end() : _data.begin() + static_cast<std::ptrdiff_t>(it->second);
    }
    iterator begin() { return _data.begin(); }
    iterator end() { return _data.end(); }
    const_iterator begin() const { return _data.begin(); }
    const_iterator end() const { return _data.end(); }

  private:
    std::vector<value_type> _data;
    std::map<IndexT, std::size_t> _index;
};

}  // namespace cheshire
}  // namespace sfm
}  // namespace aliceVision
