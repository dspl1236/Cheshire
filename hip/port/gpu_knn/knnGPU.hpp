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

// One camera as MultiViewParams holds it, for the backprojection on the device: C = CArr[c],
// iK = iCamArr[c] (m11 .. m33, row by row), P = camArr[c] (m11 .. m34, row by row).
struct Camera
{
    double C[3] = {0, 0, 0};
    double iK[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    double P[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};

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

    // Device buffers for backprojectQueryAsync: a depth map of up to maxPixels floats and maxRows
    // rows (waits for the query in flight first).
    bool reserveBackproject(std::size_t maxPixels, std::size_t maxRows);

    // One camera's queries built on the device (cheshire step 6k): uploads its depth map (w * h
    // floats, pinned) and rowStart (h + 1 counts: the valid pixels, !(depth <= 0), before each row;
    // rowStart[h] == nbQueries, pinned), backprojects every valid pixel in pixel order into the query
    // buffer with MultiViewParams::backproject's and getCamPixelSize's arithmetic operation by
    // operation (fmaBackproject selects the fused forms a host compiler that contracts emits for
    // them), answers the queries as queryAsync does, and downloads the queries, their pixel sizes and
    // the answers into the pinned outputs. Same return rules as queryAsync.
    bool backprojectQueryAsync(const float* depth, int w, int h, const std::uint64_t* rowStart, std::size_t nbQueries,
                               const Camera& cam, bool fmaBackproject, double* outQueries, double* outPixSize,
                               std::uint32_t* outIndex, double* outDist2, bool fma);

    // ---- Device votes (cheshire step 6t) ----------------------------------------------------------
    // The pass's vertices live on the device: coords (nbVertices * 3 doubles, the pass-start
    // coordinates the votes move), nrc, sim (simScorePrepare) and scoreV (sim * pixSize^2 as
    // floats). Allocates everything the per-camera pipeline needs for up to maxQueries queries; false
    // (nothing kept) when a buffer cannot be had. treeFrames() must fit the stack for the pass to run
    // here, since an overflowed query cannot be answered in order later.
    bool votesBegin(std::size_t nbVertices, const double* coords, const int* nrc, const float* sim, const float* scoreV,
                    float voteMargin, float contributeMargin, std::size_t maxQueries);
    // Words of the voted-vertex bitmap (bit v of word v / 32), the size of backprojectQueryVoteAsync's outBitmap.
    std::size_t votesBitmapWords() const { return (_votesN + 31) / 32; }
    // backprojectQueryAsync's pipeline, then on the device: every query's vote and contribute decision
    // (the host's float/double expressions), the voted vertices' bits, the contributions grouped by
    // vertex and each vertex's folded in query order. Downloads
    // the bitmap (pinned, votesBitmapWords() words) and the camera's vote and contribution counts
    // (pinned, 3 values: votes, contributions, and indices found out of range, which must be 0 for the
    // camera to count). The out* query buffers are downloaded too when given (CHECK), else nullptr.
    bool backprojectQueryVoteAsync(const float* depth, int w, int h, const std::uint64_t* rowStart, std::size_t nbQueries, const Camera& cam,
                                   bool fmaBackproject, bool fma, std::uint32_t* outBitmap, unsigned long long* outCounts, double* outQueries,
                                   double* outPixSize, std::uint32_t* outIndex, double* outDist2);
    // Waits, then downloads the moved coordinates and nrc (the host copies stay untouched before).
    bool votesEnd(double* coords, int* nrc);
    // The device's fold of count (x, n, q) triples, x = (x * n + q) / (n + 1) per component, for the
    // probe that compares it with the host's before a pass uses it. x and q: 3 doubles each.
    bool foldProbe(const double* x, const int* n, const double* q, std::size_t count, double* out);
    // The deepest path's inner nodes (the frames a query needs), -1 when build() could not tell.
    int treeFrames() const { return _treeFrames; }

    // Waits for the query in flight; returns false on a device error. overflowed() then tells
    // how many of its answers are UINT32_MAX.
    bool wait();
    std::size_t overflowed() const { return _overflowed; }

    // the kernel build() chose (step 6s): points in leaf order (CHESHIRE_GPU_KNN_LAYOUT=1; the default keeps the
    // original order and 96-frame stack), and the stack frames per query
    bool leafOrder() const { return _leafOrder; }
    int stackFrames() const { return _stack; }

    // device-side timing of the queries so far, milliseconds
    double msUpload() const { return _msUpload; }
    double msBackproject() const { return _msBackproject; }
    double msKernel() const { return _msKernel; }
    double msDownload() const { return _msDownload; }
    double msVotes() const { return _msVotes; }  // decide, sort and fold (device votes)

    void release();

  private:
    // enqueues the knn kernel build() chose over the queries in the device buffer
    bool launchKnn(std::size_t nbQueries, bool fma);

    void* _points = nullptr;  // leaf order when _leafOrder
    void* _nodes = nullptr;
    void* _perm = nullptr;
    void* _queries = nullptr;
    void* _outIndex = nullptr;
    void* _outDist = nullptr;
    void* _overflowCounter = nullptr;
    void* _overflowHost = nullptr;
    void* _depth = nullptr;
    void* _rowStart = nullptr;
    void* _pixSize = nullptr;
    void* _stream = nullptr;
    void* _events[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    std::size_t _queryCapacity = 0;
    std::size_t _bpPixelCapacity = 0;
    std::size_t _bpRowCapacity = 0;
    std::size_t _overflowed = 0;
    bool _inFlight = false;
    bool _bpInFlight = false;
    bool _leafOrder = false;
    int _stack = 96;
    int _treeFrames = -1;
    // device votes (step 6t)
    void* _vCoords = nullptr;
    void* _vNrc = nullptr;
    void* _vSim = nullptr;
    void* _vScoreV = nullptr;
    void* _vBitmap = nullptr;
    void* _vKeys = nullptr;        // per query: its vertex when it contributes, else nbVertices
    void* _vCnt = nullptr;         // per vertex (+1): this camera's contributions, back to 0 after the scatter
    void* _vOffsets = nullptr;     // per vertex (+1): the first slot of its range, then the total
    void* _vTileSums = nullptr;    // the scan's tile totals, then their prefix
    void* _vSlots = nullptr;       // the contributions' queries, by vertex
    void* _vSlotVertex = nullptr;  // and their vertex
    void* _vCounts = nullptr;      // votes, contributions, out-of-range indices
    std::size_t _votesN = 0;
    std::size_t _vCapacity = 0;
    std::size_t _vTiles = 0;
    float _voteMargin = 0.0f;
    float _contributeMargin = 0.0f;
    void releaseVotes();
    double _msUpload = 0, _msBackproject = 0, _msKernel = 0, _msDownload = 0, _msVotes = 0;
    bool _votesInFlight = false;
    double _lo[3] = {0, 0, 0};
    double _hi[3] = {0, 0, 0};
};

}  // namespace knn
}  // namespace cheshire
