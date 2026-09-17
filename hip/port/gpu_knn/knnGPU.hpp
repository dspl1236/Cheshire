// Cheshire: nearest-neighbour queries for Meshing's visibility passes on the GPU.
//
// createVerticesWithVisibilities looks up, for every valid depth-map pixel of every camera, the
// nearest vertex of the point cloud (152 M queries against a 6.9 M point kd-tree on the 107-photo
// engine bay job, ~29 s per pass in 12 threads, twice per Meshing). The tree is nanoflann's own,
// built on the host as upstream builds it and copied node for node; the device walks it exactly as
// nanoflann::KDTreeSingleIndexAdaptor::searchLevel does (same child order, same pruning test, same
// double arithmetic), so the answer is the one upstream's search would give, ties included.
// Queries are asynchronous on one stream over pinned host buffers, so the host backprojects the
// next camera and applies the previous camera's votes while the device works.
// CUDA dialect; the HIP build force-includes cheshire/cuda_to_hip.h.
#pragma once
#include <cstddef>
#include <cstdint>

namespace cheshire {
namespace knn {

// One nanoflann node. Leaf: child1 == child2 == -1, a/b = the [left, right) range in the point
// permutation. Inner: a = the split dimension, lo/hi = divlow/divhigh, child1/child2 = node ids.
struct Node
{
    std::int32_t child1 = -1;
    std::int32_t child2 = -1;
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    double lo = 0;
    double hi = 0;
};

bool available();

// Pinned host memory for the query and result buffers (nullptr when unavailable).
void* hostAlloc(std::size_t bytes);
void hostFree(void* p);

class Index
{
  public:
    Index() = default;
    ~Index() { release(); }
    Index(const Index&) = delete;
    Index& operator=(const Index&) = delete;

    // points: nbPoints * 3 doubles (x, y, z interleaved); nodes: the flattened tree, node 0 the
    // root; perm: nanoflann's vAcc_ (nbPoints entries); bboxLo/bboxHi: root_bbox_.
    // maxQueries sizes the device-side query buffers.
    bool build(const double* points, std::size_t nbPoints, const Node* nodes, std::size_t nbNodes,
               const std::uint32_t* perm, const double bboxLo[3], const double bboxHi[3], std::size_t maxQueries);

    // Resizes the device-side query buffers (waits for the query in flight first).
    bool reserve(std::size_t maxQueries);

    // Enqueues one camera's queries (nbQueries * 3 doubles, pinned) and their answers (pinned):
    // outIndex gets the nearest point's id (UINT32_MAX when the walk overflowed its stack, which
    // the caller answers on the host), outDist2 the squared distance as nanoflann computes it.
    // fma selects the contracted a*b+c form of the metric, matching a host build whose compiler
    // fuses `result += diff * diff`. Returns false without enqueuing when nbQueries exceeds
    // maxQueries or a previous query is still in flight.
    bool queryAsync(const double* queries, std::size_t nbQueries, std::uint32_t* outIndex, double* outDist2, bool fma);

    // Waits for the query in flight; returns false on a device error. overflowed() then tells
    // how many of its answers are UINT32_MAX.
    bool wait();
    std::size_t overflowed() const { return _overflowed; }

    // device-side timing of the queries so far, milliseconds
    double msUpload() const { return _msUpload; }
    double msKernel() const { return _msKernel; }
    double msDownload() const { return _msDownload; }

    void release();

  private:
    void* _points = nullptr;
    void* _nodes = nullptr;
    void* _perm = nullptr;
    void* _queries = nullptr;
    void* _outIndex = nullptr;
    void* _outDist = nullptr;
    void* _overflowCounter = nullptr;
    void* _overflowHost = nullptr;
    void* _stream = nullptr;
    void* _events[4] = {nullptr, nullptr, nullptr, nullptr};
    std::size_t _queryCapacity = 0;
    std::size_t _overflowed = 0;
    bool _inFlight = false;
    double _msUpload = 0, _msKernel = 0, _msDownload = 0;
    double _lo[3] = {0, 0, 0};
    double _hi[3] = {0, 0, 0};
};

}  // namespace knn
}  // namespace cheshire
