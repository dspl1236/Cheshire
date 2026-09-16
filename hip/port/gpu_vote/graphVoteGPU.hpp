// Cheshire: Meshing's s-t graph weight voting (GraphFiller::fillGraph) on the GPU.
//
// For every real vertex and every camera that sees it, upstream marches a ray through the Delaunay
// tetrahedralisation twice: towards the camera, voting "empty" on every cell crossed and pinning the
// cells near the camera to the source; and a few pixel sizes behind the vertex, voting "full". The
// per-ray work is independent and the votes are float atomics, which is why upstream runs it with
// OpenMP in a randomised vertex order and why the CPU result is not bit-reproducible run to run
// (measured: two identical runs, 239,547 vs 239,045 mesh vertices). This port keeps the marching
// itself exact (one thread per ray, the same double-precision predicates and the same quirks: the
// ambiguity and forward checks compare against Eigen's Vector3d::size(), which is 3) so each ray
// votes on the same cells as the CPU; only the accumulation order differs, as it does on the CPU.
//
// CUDA dialect; the HIP build force-includes cheshire/cuda_to_hip.h. CHESHIRE_GPU_VOTE=0 disables.
#pragma once
#include <cstddef>
#include <cstdint>

namespace aliceVision {
namespace fuseCut {
namespace gpu {

bool voteAvailable();

// Flat copy of what the marching reads. Indices are uint32 with 0xffffffff for "none"
// (GEO::NO_CELL / NO_VERTEX are index_t(-1)).
struct VoteInput {
    const double* vertices = nullptr;        // 3 per vertex
    uint32_t nbVertices = 0;
    const uint32_t* cellVertices = nullptr;  // 4 per cell
    const uint32_t* cellAdjacent = nullptr;  // 4 per cell
    uint32_t nbCells = 0;
    const uint32_t* vertexCellsOffset = nullptr;  // CSR over vertices: cells around each vertex, ascending
    const uint32_t* vertexCells = nullptr;
    const uint32_t* rayVertex = nullptr;     // per ray: the vertex
    const uint32_t* rayCamVertex = nullptr;  // per ray: the camera's vertex in the tetrahedralisation
    const float* rayWeight = nullptr;        // per ray: the empty-vote weight (nrc or the forced weight)
    const double* rayMaxDist = nullptr;      // per ray: nPixelSizeBehind * pixSize (<= 0: no full pass)
    const double* rayCamCenter = nullptr;    // 3 per ray: CArr[cam]
    uint32_t nbRays = 0;
    float fullWeight = 1.0f;
};

// cellAttr: 8 floats per cell in GC_cellInfo order (cellSWeight, cellTWeight, gEdgeVisWeight[4],
// emptinessScore, on), read in, voted on, written back.
bool fillGraph(const VoteInput& in, float* cellAttr);

}  // namespace gpu
}  // namespace fuseCut
}  // namespace aliceVision
