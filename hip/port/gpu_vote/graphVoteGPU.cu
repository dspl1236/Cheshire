// Cheshire: Meshing's graph-weight voting on the GPU (see graphVoteGPU.hpp).
//
// TetrahedronsRayMarching (Intersections.cpp), GraphFiller::rayMarchingGraphEmpty / Full and
// GraphFiller::forceTedgesByGradientIJCV transcribed for one thread per ray. Double precision, FMA
// contraction off, the same expression order as upstream, and upstream's quirks kept on purpose:
//   * "is the new point farther than the best ambiguous one" and "did we move at all" compare
//     Eigen::Vector3d::size(), which is the element count 3, not a length: so the first hit wins and
//     the "too close" test never fires;
//   * Point3d::size() returns 0 for a zero vector (no sqrt) where Eigen's norm() does not.
#ifdef __clang__
#pragma clang fp contract(off)
#endif
#include "graphVoteGPU.hpp"
#include <cuda_runtime.h>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace aliceVision {
namespace fuseCut {
namespace gpu {

namespace {

constexpr uint32_t NONE = 0xffffffffu;
constexpr int kBlock = 128;
constexpr uint32_t kRayChunk = 4u << 20;   // rays per upload

struct V3 { double x, y, z; };
__device__ __forceinline__ V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ __forceinline__ V3 neg(V3 a) { return {-a.x, -a.y, -a.z}; }
__device__ __forceinline__ double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ double enorm(V3 a) { return sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }              // Eigen norm()
__device__ __forceinline__ V3 enormalized(V3 a) { const double n = enorm(a); return {a.x / n, a.y / n, a.z / n}; }   // Eigen normalized()
__device__ __forceinline__ double psize(V3 a) { const double d = a.x * a.x + a.y * a.y + a.z * a.z; return d == 0.0 ? 0.0 : sqrt(d); }  // Point3d::size()
__device__ __forceinline__ bool isnormald(double x) { return x == x && fabs(x) != HUGE_VAL && fabs(x) >= DBL_MIN; }

enum GType : int { GVertex = 0, GEdge = 1, GFacet = 2, GNone = 3 };
struct Geo { int type; uint32_t a, b; };   // Facet: a = cell, b = local vertex; Vertex: a; Edge: a = v0, b = v1
__device__ __forceinline__ Geo gNone() { return {GNone, NONE, NONE}; }
__device__ __forceinline__ Geo gFacet(uint32_t c, uint32_t lv) { return {GFacet, c, lv}; }
__device__ __forceinline__ Geo gVertex(uint32_t v) { return {GVertex, v, NONE}; }
__device__ __forceinline__ Geo gEdge(uint32_t v0, uint32_t v1) { return {GEdge, v0, v1}; }

struct Mesh {
    const double* verts; const uint32_t* cv; const uint32_t* ca; const uint32_t* vco; const uint32_t* vcl;
    __device__ __forceinline__ V3 P(uint32_t v) const { return {verts[3 * size_t(v)], verts[3 * size_t(v) + 1], verts[3 * size_t(v) + 2]}; }
    __device__ __forceinline__ uint32_t cellVertex(uint32_t c, uint32_t lv) const { return cv[4 * size_t(c) + lv]; }
    __device__ __forceinline__ uint32_t cellAdjacent(uint32_t c, uint32_t lv) const { return ca[4 * size_t(c) + lv]; }
    __device__ __forceinline__ bool isInfinite(uint32_t c) const { for (int i = 0; i < 4; ++i) if (cv[4 * size_t(c) + i] == NONE) return true; return false; }
    __device__ __forceinline__ bool invalidOrInfinite(uint32_t c) const { return c == NONE || isInfinite(c); }
    __device__ __forceinline__ uint32_t index(uint32_t c, uint32_t v) const { for (uint32_t i = 0; i < 4; ++i) if (cv[4 * size_t(c) + i] == v) return i; return NONE; }
    // Tetrahedralization::mirrorFacet
    __device__ Geo mirrorFacet(uint32_t c, uint32_t lv) const {
        const uint32_t fv[3] = {cellVertex(c, (lv + 1) % 4), cellVertex(c, (lv + 2) % 4), cellVertex(c, (lv + 3) % 4)};
        uint32_t oc = cellAdjacent(c, lv), olv = NONE;
        if (oc != NONE)
            for (uint32_t k = 0; k < 4; ++k) {
                const uint32_t ov = cellVertex(oc, k);
                if (ov != fv[0] && ov != fv[1] && ov != fv[2]) { olv = k; break; }
            }
        return gFacet(oc, olv);
    }
};

struct Marching {
    const Mesh& m;
    V3 origin, dir, ip, prevIp;
    Geo inter, prevInter;
    unsigned facetCount = 0, vertexCount = 0, edgeCount = 0;
    static constexpr double epsilonFactor = 1e-4;

    __device__ Marching(const Mesh& mesh, uint32_t originId, uint32_t destinationId, bool away) : m(mesh) {
        origin = m.P(originId); ip = origin; prevIp = origin; inter = gVertex(originId); prevInter = gNone();
        dir = enormalized(sub(m.P(destinationId), origin));
        if (away) dir = neg(dir);
    }

    // getLineTriangleIntersectBarycCoords
    __device__ void bary(V3& P, V3 A, V3 B, V3 C, double& u, double& v) const {
        const double v0_x = C.x - A.x, v0_y = C.y - A.y, v0_z = C.z - A.z;
        const double v1_x = B.x - A.x, v1_y = B.y - A.y, v1_z = B.z - A.z;
        const double n_x = v0_y * v1_z - v0_z * v1_y, n_y = v0_z * v1_x - v0_x * v1_z, n_z = v0_x * v1_y - v0_y * v1_x;
        const double k = ((A.x * n_x + A.y * n_y + A.z * n_z) - (n_x * origin.x + n_y * origin.y + n_z * origin.z)) / (n_x * dir.x + n_y * dir.y + n_z * dir.z);
        P.x = origin.x + dir.x * k; P.y = origin.y + dir.y * k; P.z = origin.z + dir.z * k;
        const double v2_x = P.x - A.x, v2_y = P.y - A.y, v2_z = P.z - A.z;
        const double dot00 = (v0_x * v0_x + v0_y * v0_y + v0_z * v0_z), dot01 = (v0_x * v1_x + v0_y * v1_y + v0_z * v1_z);
        const double dot02 = (v0_x * v2_x + v0_y * v2_y + v0_z * v2_z), dot11 = (v1_x * v1_x + v1_y * v1_y + v1_z * v1_z);
        const double dot12 = (v1_x * v2_x + v1_y * v2_y + v1_z * v2_z);
        const double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
        u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    }

    // rayIntersectTriangle: writes `ip` only past the direction check, as upstream writes _intersectionPoint
    __device__ Geo rayIntersectTriangle(uint32_t cell, uint32_t lv, V3 lastIp, bool& ambiguous) {
        ambiguous = false;
        const uint32_t Ai = m.cellVertex(cell, (lv + 1) % 4), Bi = m.cellVertex(cell, (lv + 2) % 4), Ci = m.cellVertex(cell, (lv + 3) % 4);
        const V3 A = m.P(Ai), B = m.P(Bi), C = m.P(Ci);
        const double ABSize = psize(sub(A, B)), BCSize = psize(sub(B, C)), ACSize = psize(sub(A, C));
        const double marginEpsilon = fmin(fmin(ABSize, BCSize), ACSize) * epsilonFactor;
        const double ambiguityEpsilon = (ABSize + BCSize + ACSize) / 3.0 * 1.0e-2;
        V3 P; double u, v;
        bary(P, A, B, C, u, v);
        if (!isnormald(P.x) || !isnormald(P.y) || !isnormald(P.z)) return gNone();
        if (!(u == u && fabs(u) != HUGE_VAL) || !(v == v && fabs(v) != HUGE_VAL)) return gNone();
        if (u < -marginEpsilon || v < -marginEpsilon || (u + v) > (1.0 + marginEpsilon)) return gNone();
        const V3 diff = sub(P, lastIp);
        const double dotValue = dot(dir, enormalized(diff));
        const double diffSize = 3.0;   // Eigen::Vector3d::size() is the element count (upstream quirk)
        if (dotValue < marginEpsilon || diffSize < 100 * DBL_MIN) return gNone();
        if (diffSize < ambiguityEpsilon) ambiguous = true;
        ip = P;
        if (v < marginEpsilon) {
            if (u < marginEpsilon) { ip = A; return gVertex(Ai); }
            if (u > 1.0 - marginEpsilon) { ip = C; return gVertex(Ci); }
            return gEdge(Ai, Ci);
        }
        if (u < marginEpsilon) {
            if (v > 1.0 - marginEpsilon) { ip = B; return gVertex(Bi); }
            return gEdge(Ai, Bi);
        }
        if (u + v > 1.0 - marginEpsilon) return gEdge(Bi, Ci);
        return gFacet(cell, lv);
    }

    // the "best ambiguous match" bookkeeping shared by the three next* functions; with size() == 3 the
    // comparison "(origin - ip).size() > (origin - bestIp).size()" is 3 > 3, so only the first is kept
    __device__ __forceinline__ bool consider(Geo res, bool ambiguous, Geo& best, V3& bestIp, Geo& out) {
        if (res.type == GNone) return false;
        if (!ambiguous) { out = res; return true; }
        if (best.type == GNone || false) { bestIp = ip; best = res; }
        return false;
    }

    __device__ Geo nextFacet() {
        Geo best = gNone(); V3 bestIp = ip; Geo out;
        const uint32_t c = inter.a;
        for (uint32_t i = 0; i < 4; ++i) {
            if (i == inter.b) continue;
            bool amb; const Geo res = rayIntersectTriangle(c, i, prevIp, amb);
            if (consider(res, amb, best, bestIp, out)) return out;
        }
        ip = bestIp; return best;
    }

    __device__ Geo nextEdge() {
        Geo best = gNone(); V3 bestIp = ip; Geo out;
        const uint32_t v0 = inter.a, v1 = inter.b;
        // set_intersection of the two ascending cell lists
        uint32_t i = m.vco[v0], ie = m.vco[v0 + 1], j = m.vco[v1], je = m.vco[v1 + 1];
        while (i < ie && j < je) {
            const uint32_t ci = m.vcl[i], cj = m.vcl[j];
            if (ci < cj) { ++i; continue; }
            if (cj < ci) { ++j; continue; }
            ++i; ++j;
            const uint32_t adj = ci;
            if (m.invalidOrInfinite(adj)) continue;
            const uint32_t lvi0 = m.index(adj, v0), lvi1 = m.index(adj, v1);
            const uint32_t lvs[2] = {lvi0, lvi1};
            for (int k = 0; k < 2; ++k) {
                bool amb; const Geo res = rayIntersectTriangle(adj, lvs[k], prevIp, amb);
                if (res.type == GEdge && ((res.a == v0 && res.b == v1) || (res.a == v1 && res.b == v0))) continue;
                if (consider(res, amb, best, bestIp, out)) return out;
            }
        }
        ip = bestIp; return best;
    }

    __device__ Geo nextVertex() {
        Geo best = gNone(); V3 bestIp = ip; Geo out;
        const uint32_t v = inter.a;
        for (uint32_t i = m.vco[v]; i < m.vco[v + 1]; ++i) {
            const uint32_t adj = m.vcl[i];
            if (m.invalidOrInfinite(adj)) continue;
            const uint32_t lv = m.index(adj, v);
            bool amb; const Geo res = rayIntersectTriangle(adj, lv, prevIp, amb);
            if (consider(res, amb, best, bestIp, out)) return out;
        }
        ip = bestIp; return best;
    }

    // intersectNextGeom
    __device__ Geo next() {
        prevInter = inter; prevIp = ip;
        switch (prevInter.type) {
            case GFacet: inter = nextFacet(); facetCount++; break;
            case GEdge: inter = nextEdge(); edgeCount++; break;
            case GVertex: inter = nextVertex(); vertexCount++; break;
            default: return gNone();
        }
        if (facetCount > 10000 && prevInter.type == GFacet) return gNone();
        if (vertexCount > 1000 && prevInter.type == GVertex) return gNone();
        if (edgeCount > 1000 && prevInter.type == GEdge) return gNone();
        if (enorm(sub(ip, origin)) <= enorm(sub(prevIp, origin))) return gNone();
        if (inter.type == GFacet) {
            const Geo f = m.mirrorFacet(inter.a, inter.b);
            if (m.invalidOrInfinite(f.a)) return gNone();
            prevInter = inter; inter = f;
        }
        return inter;
    }
};

// cell attribute slots
enum { A_SW = 0, A_TW = 1, A_G0 = 2, A_EMPT = 6, A_ON = 7, A_STRIDE = 8 };

__global__ void voteKernel(Mesh mesh, const uint32_t* __restrict__ rayVertex, const uint32_t* __restrict__ rayCam, const float* __restrict__ rayWeight,
                           const double* __restrict__ rayMaxDist, const double* __restrict__ rayCamCenter, uint32_t nbRays, float fullWeightParam,
                           float* __restrict__ attr)
{
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nbRays) return;
    const uint32_t vi = rayVertex[r], camV = rayCam[r];
    if (camV == NONE) return;
    const float weight = rayWeight[r];
    const V3 C = {rayCamCenter[3 * size_t(r)], rayCamCenter[3 * size_t(r) + 1], rayCamCenter[3 * size_t(r) + 2]};
    const float maxint = float(INT_MAX);

    // rayMarchingGraphEmpty
    {
        const V3 originPt = mesh.P(vi);
        const double pointCamDistance = psize(sub(C, originPt));
        Marching mr(mesh, vi, camV, false);
        Geo geometry = gVertex(vi); V3 ip = originPt;
        uint32_t lastCell = NONE, lastLv = NONE;
        while (geometry.type != GVertex || psize(sub(C, ip)) >= 1.0e-3) {
            const Geo previousGeometry = geometry;
            geometry = mr.next(); ip = mr.ip;
            if (geometry.type == GNone) break;
            if (geometry.type == GFacet) {
                const Geo pg = mr.prevInter;   // the facet we left through, on the cell we crossed
                atomicAdd(attr + size_t(pg.a) * A_STRIDE + A_EMPT, weight);
                atomicAdd(attr + size_t(pg.a) * A_STRIDE + A_G0 + pg.b, weight);
                lastCell = geometry.a; lastLv = geometry.b;
            } else if (previousGeometry.type == GFacet) {
                atomicAdd(attr + size_t(previousGeometry.a) * A_STRIDE + A_EMPT, weight);
            }
            if (lastCell != NONE && psize(sub(C, ip)) < 0.2 * pointCamDistance)
                atomicExch(attr + size_t(lastCell) * A_STRIDE + A_SW, maxint);
        }
        if (lastCell != NONE) atomicExch(attr + size_t(lastCell) * A_STRIDE + A_SW, maxint);
    }

    // rayMarchingGraphFull
    const double maxDist = rayMaxDist[r];
    if (maxDist > 0.0) {
        const float fullWeight = weight * fullWeightParam;
        const V3 originPt = mesh.P(vi);
        Marching mr(mesh, vi, camV, true);
        Geo geometry = gVertex(vi); V3 ip = originPt;
        uint32_t lastCell = NONE;
        while (psize(sub(originPt, ip)) < maxDist) {
            geometry = mr.next(); ip = mr.ip;
            if (geometry.type == GNone) break;
            if (geometry.type == GFacet) {
                lastCell = geometry.a;
                atomicAdd(attr + size_t(geometry.a) * A_STRIDE + A_G0 + geometry.b, fullWeight);
            }
        }
        if (lastCell != NONE) atomicAdd(attr + size_t(lastCell) * A_STRIDE + A_TW, fullWeight);
    }
}

// std::max(a, b): b only when a < b (a NaN stays)
__device__ __forceinline__ float smax(float a, float b) { return a < b ? b : a; }

// Whether the cells around vertex `v` and the cells around geometry `g` (a vertex, or an edge: the cells
// around both of its vertices) share a cell: the set_intersection upstream builds on the first step
// behind the vertex, whose only effect is whether the (quirky) midSilent read happens at all.
__device__ bool sharesCell(const Mesh& m, uint32_t v, Geo g)
{
    uint32_t i = m.vco[v], ie = m.vco[v + 1];
    uint32_t j = m.vco[g.a], je = m.vco[g.a + 1];
    uint32_t k = 0, ke = 0;
    if (g.type == GEdge) { k = m.vco[g.b]; ke = m.vco[g.b + 1]; }
    while (i < ie && j < je) {
        const uint32_t ci = m.vcl[i], cj = m.vcl[j];
        if (ci < cj) { ++i; continue; }
        if (cj < ci) { ++j; continue; }
        if (g.type != GEdge) return true;
        while (k < ke && m.vcl[k] < ci) ++k;
        if (k < ke && m.vcl[k] == ci) return true;
        ++i; ++j;
    }
    return false;
}

// forceTedgesByGradientIJCV, one thread per ray: read-only over the (final) emptiness scores, one
// atomic add on the last cell's `on`. The loop bounds lag one step behind the march exactly as
// upstream's lastIntersectPt does, and the sigma products are float like the upstream constants.
__global__ void tedgeKernel(Mesh mesh, const uint32_t* __restrict__ rayVertex, const uint32_t* __restrict__ rayCam, const float* __restrict__ rayDist,
                            const double* __restrict__ rayCamCenter, uint32_t nbRays, uint32_t nbCells, float* __restrict__ attr)
{
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nbRays) return;
    const uint32_t vi = rayVertex[r], camV = rayCam[r];
    if (camV == NONE) return;
    const float maxDist = rayDist[r];
    const V3 C = {rayCamCenter[3 * size_t(r)], rayCamCenter[3 * size_t(r) + 1], rayCamCenter[3 * size_t(r) + 2]};
    const V3 originPt = mesh.P(vi);
    float maxJump = 0.0f, maxSilent = 0.0f, midSilent = 10000000.0f;

    // towards the camera: emptiness before the point (jump part) and around it (front silent part)
    {
        Marching mr(mesh, vi, camV, false);
        Geo geometry = gVertex(vi); V3 ip = originPt, lastIp = originPt;
        const double frontRange = double((4.0f + 2.0f) * maxDist), silentRange = double(2.0f * maxDist);
        while ((geometry.type != GVertex || psize(sub(C, ip)) > 1.0e-3) && psize(sub(lastIp, originPt)) <= frontRange) {
            lastIp = ip;
            geometry = mr.next(); ip = mr.ip;
            if (geometry.type == GNone) break;
            if (geometry.type == GFacet) {
                const float e = attr[size_t(mr.prevInter.a) * A_STRIDE + A_EMPT];
                if (psize(sub(lastIp, originPt)) > silentRange) maxJump = smax(maxJump, e);
                else maxSilent = smax(maxSilent, e);
            }
        }
    }
    // behind the point: the first cell's emptiness (mid) and the max around it (back silent part)
    uint32_t lastCell = NONE;
    {
        Marching mr(mesh, vi, camV, true);
        Geo geometry = gVertex(vi); V3 ip = originPt, lastIp = originPt;
        const double backRange = double(2.0f * maxDist);
        bool first = true;
        while (psize(sub(lastIp, originPt)) <= backRange) {
            const Geo previousGeometry = geometry;
            lastIp = ip;
            geometry = mr.next(); ip = mr.ip;
            if (geometry.type == GNone) break;
            if (geometry.type == GFacet) {
                const float e = attr[size_t(mr.prevInter.a) * A_STRIDE + A_EMPT];
                if (first) { midSilent = e; first = false; }
                maxSilent = smax(maxSilent, e);
                lastCell = geometry.a;
            } else if (first) {
                // upstream quirk: when the first step lands on a vertex or an edge it reads
                // _cellsAttr[geometry.facet.cellIndex] through the union, i.e. cell index = that vertex
                // index (or the edge's first vertex), once the two neighbourhoods are known to intersect
                if (previousGeometry.type == GVertex && sharesCell(mesh, previousGeometry.a, geometry) && geometry.a < nbCells)
                    midSilent = attr[size_t(geometry.a) * A_STRIDE + A_EMPT];
                first = false;
            }
        }
    }
    if (lastCell != NONE) {
        // equation 6: (g / B) < k_rel, (B - g) > k_abs, g < k_outl
        if ((midSilent / maxJump < 0.1f) && (maxJump - midSilent > 10000.0f) && (maxSilent < 100.0f))
            atomicAdd(attr + size_t(lastCell) * A_STRIDE + A_ON, maxJump - midSilent);
    }
}

bool g_checked = false, g_available = false, g_tedgesDone = false;
std::mutex g_mutex;

template<class T> bool up(T** d, const T* h, size_t n) {
    if (cudaMalloc((void**)d, n * sizeof(T)) != cudaSuccess) return false;
    return cudaMemcpy(*d, h, n * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}

}  // namespace

bool voteAvailable()
{
    std::lock_guard<std::mutex> g(g_mutex);
    if (g_checked) return g_available;
    g_checked = true;
    if (const char* e = std::getenv("CHESHIRE_GPU_VOTE")) if (e[0] == '0') { std::fprintf(stderr, "[cheshire] meshing votes: disabled by CHESHIRE_GPU_VOTE=0, CPU\n"); return g_available = false; }
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { std::fprintf(stderr, "[cheshire] meshing votes: no GPU device, CPU\n"); return g_available = false; }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return g_available = false;
    std::fprintf(stderr, "[cheshire] meshing votes: ray marching on %s (CHESHIRE_GPU_VOTE=0 for the CPU pass)\n", p.name);
    return g_available = true;
}

bool fillGraph(const VoteInput& in, float* cellAttr)
{
    // CHESHIRE_GPU_VOTE_LOG=1: where the time goes (uploads, kernels, download)
    const bool log = std::getenv("CHESHIRE_GPU_VOTE_LOG") != nullptr;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto secs = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    const auto t0 = now();
    double* dv = nullptr; uint32_t *dcv = nullptr, *dca = nullptr, *dvco = nullptr, *dvcl = nullptr; float* dattr = nullptr;
    uint32_t *drv = nullptr, *drc = nullptr; float *drw = nullptr, *drt = nullptr; double *drd = nullptr, *drcc = nullptr;
    bool ok = up(&dv, in.vertices, 3 * size_t(in.nbVertices)) && up(&dcv, in.cellVertices, 4 * size_t(in.nbCells)) && up(&dca, in.cellAdjacent, 4 * size_t(in.nbCells))
           && up(&dvco, in.vertexCellsOffset, size_t(in.nbVertices) + 1) && up(&dvcl, in.vertexCells, size_t(in.vertexCellsOffset[in.nbVertices]))
           && up(&dattr, cellAttr, size_t(in.nbCells) * A_STRIDE);
    const uint32_t chunk = kRayChunk;
    if (ok) ok = cudaMalloc((void**)&drv, chunk * 4) == cudaSuccess && cudaMalloc((void**)&drc, chunk * 4) == cudaSuccess && cudaMalloc((void**)&drw, chunk * 4) == cudaSuccess
              && cudaMalloc((void**)&drt, chunk * 4) == cudaSuccess && cudaMalloc((void**)&drd, chunk * 8) == cudaSuccess && cudaMalloc((void**)&drcc, size_t(chunk) * 24) == cudaSuccess;
    const auto t1 = now();
    const bool tedges = in.rayTedgeDist != nullptr;
    double voteSec = 0, tedgeSec = 0;
    if (ok) {
        Mesh mesh{dv, dcv, dca, dvco, dvcl};
        // the ray arrays go up in chunks; each pass is complete (and synchronised) before the next starts,
        // so the tedge pass reads the final emptiness scores as upstream does
        for (int pass = 0; ok && pass < (tedges ? 2 : 1); ++pass) {
            for (uint32_t r0 = 0; ok && r0 < in.nbRays; r0 += chunk) {
                const uint32_t n = in.nbRays - r0 < chunk ? in.nbRays - r0 : chunk;
                ok = cudaMemcpy(drv, in.rayVertex + r0, n * 4, cudaMemcpyHostToDevice) == cudaSuccess
                  && cudaMemcpy(drc, in.rayCamVertex + r0, n * 4, cudaMemcpyHostToDevice) == cudaSuccess
                  && cudaMemcpy(drcc, in.rayCamCenter + 3 * size_t(r0), size_t(n) * 24, cudaMemcpyHostToDevice) == cudaSuccess;
                if (ok && pass == 0)
                    ok = cudaMemcpy(drw, in.rayWeight + r0, n * 4, cudaMemcpyHostToDevice) == cudaSuccess
                      && cudaMemcpy(drd, in.rayMaxDist + r0, n * 8, cudaMemcpyHostToDevice) == cudaSuccess;
                if (ok && pass == 1) ok = cudaMemcpy(drt, in.rayTedgeDist + r0, n * 4, cudaMemcpyHostToDevice) == cudaSuccess;
                if (!ok) break;
                const auto tk = now();
                if (pass == 0) voteKernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(mesh, drv, drc, drw, drd, drcc, n, in.fullWeight, dattr);
                else tedgeKernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(mesh, drv, drc, drt, drcc, n, in.nbCells, dattr);
                ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
                (pass == 0 ? voteSec : tedgeSec) += secs(tk, now());
            }
        }
        const auto t2 = now();
        if (ok) ok = cudaMemcpy(cellAttr, dattr, size_t(in.nbCells) * A_STRIDE * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
        if (log) std::fprintf(stderr, "[cheshire] meshing votes profile: %u rays, %u cells: uploads %.2f s, vote kernels %.2f s, tedge kernels %.2f s (ray uploads incl.), download %.2f s\n",
                              in.nbRays, in.nbCells, secs(t0, t1), voteSec, tedgeSec, secs(t2, now()));
    }
    for (void* p : {(void*)dv, (void*)dcv, (void*)dca, (void*)dvco, (void*)dvcl, (void*)dattr, (void*)drv, (void*)drc, (void*)drw, (void*)drt, (void*)drd, (void*)drcc}) if (p) cudaFree(p);
    {
        std::lock_guard<std::mutex> g(g_mutex);
        g_tedgesDone = ok && tedges;
    }
    return ok;
}

bool tedgesDone()
{
    std::lock_guard<std::mutex> g(g_mutex);
    const bool d = g_tedgesDone;
    g_tedgesDone = false;
    return d;
}

}  // namespace gpu
}  // namespace fuseCut
}  // namespace aliceVision
