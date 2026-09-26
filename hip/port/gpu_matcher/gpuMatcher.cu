// Cheshire: GPU brute-force 2-NN for descriptor matching (see gpuMatcher.hpp).
//
// One thread per query descriptor; the database is streamed through shared memory in tiles, and
// every thread scans the whole tile. Within a wavefront all lanes read the same database row at
// the same time, so the tile reads are broadcasts and the loop is compute-bound: 128 multiply-adds
// per (query, row). uint8 descriptors are kept packed (4 per uint32) in shared memory and in
// registers and unpacked on the fly; distances are exact integers (max 128 * 255^2 < 2^31), so
// the uint8 path is bit-exact against the CPU brute force. Float descriptors use plain FMA.
//
// CUDA dialect. Under HIP the build force-includes cheshire/cuda_to_hip.h, which maps the runtime
// calls; __global__/__shared__/__syncthreads are the same in both.
#include "gpuMatcher.hpp"
#include <cuda_runtime.h>
#include <aliceVision/depthMap/cuda/hip/cheshire/devalloc.h>  // cheshire: bridge on both backends
#include <aliceVision/depthMap/cuda/hip/cheshire/env.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <mutex>

namespace aliceVision {
namespace matching {
namespace gpu {

namespace {

constexpr int kQueriesPerBlock = 256;   // threads per block = queries per block
constexpr int kTileRowsU8 = 256;        // 256 rows x 128 B = 32 KB of shared memory
constexpr int kTileRowsF32 = 64;        // 64 rows x 128 x 4 B = 32 KB

struct Best2 {
    int i0 = -1, i1 = -1;
    float d0 = 3.4e38f, d1 = 3.4e38f;
    __device__ inline void push(int i, float d) {
        // strict '<' against the current best keeps the lower row index on ties (rows arrive in order)
        if (d < d0) { d1 = d0; i1 = i0; d0 = d; i0 = i; }
        else if (d < d1) { d1 = d; i1 = i; }
    }
};

// Dot product of two words of four uint8 each, accumulated into acc. One hardware instruction
// (v_dot4_u32_u8 on RDNA2+/Vega 20, dp4a on sm_61+), else four multiply-adds. Exact in every case.
__device__ __forceinline__ unsigned int dot4u8(unsigned int a, unsigned int b, unsigned int acc)
{
#if defined(CHESHIRE_MATCHER_NO_DOT4)
    // fallback path forced at build time (what gfx1010 / gfx900 / pre-Pascal get), for testing it on any card
    return acc + (a & 0xffu) * (b & 0xffu) + ((a >> 8) & 0xffu) * ((b >> 8) & 0xffu)
               + ((a >> 16) & 0xffu) * ((b >> 16) & 0xffu) + (a >> 24) * (b >> 24);
#elif defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx900__) && !defined(__gfx1010__)
    return __builtin_amdgcn_udot4(a, b, acc, false);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, acc);
#else
    return acc + (a & 0xffu) * (b & 0xffu) + ((a >> 8) & 0xffu) * ((b >> 8) & 0xffu)
               + ((a >> 16) & 0xffu) * ((b >> 16) & 0xffu) + (a >> 24) * (b >> 24);
#endif
}

// Squared norm of one packed uint8 row (DIM scalars): |r|^2, exact (max 128 * 255^2).
template<int DIM>
__global__ void rowNormsU8(const unsigned int* __restrict__ db, int rows, unsigned int* __restrict__ norms)
{
    constexpr int W = DIM / 4;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    unsigned int n = 0;
    for (int k = 0; k < W; ++k) { const unsigned int a = db[size_t(r) * W + k]; n = dot4u8(a, a, n); }
    norms[r] = n;
}

// DIM scalars per descriptor, uint8, packed as DIM/4 uint32 per row. Squared L2 as
// |q|^2 + |r|^2 - 2 q.r with q.r from the packed dot product: the same integer as the
// difference-of-squares form, at a quarter of the instructions.
template<int DIM>
__global__ void knn2_u8(const unsigned int* __restrict__ db, const unsigned int* __restrict__ dbNorm, int rows,
                        const unsigned int* __restrict__ q, int nbQuery, int* __restrict__ outIdx, float* __restrict__ outDist)
{
    constexpr int W = DIM / 4;
    __shared__ unsigned int tile[kTileRowsU8 * W];
    __shared__ unsigned int tileNorm[kTileRowsU8];
    const int qi = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int qreg[W];
    unsigned int qnorm = 0;
    if (qi < nbQuery) {
        for (int k = 0; k < W; ++k) { qreg[k] = q[size_t(qi) * W + k]; qnorm = dot4u8(qreg[k], qreg[k], qnorm); }
    } else {
        for (int k = 0; k < W; ++k) qreg[k] = 0u;
    }
    Best2 best;
    for (int base = 0; base < rows; base += kTileRowsU8) {
        const int n = min(kTileRowsU8, rows - base);
        // cooperative tile load: n * W words plus the row norms
        for (int t = threadIdx.x; t < n * W; t += blockDim.x) tile[t] = db[size_t(base) * W + t];
        for (int t = threadIdx.x; t < n; t += blockDim.x) tileNorm[t] = dbNorm[base + t];
        __syncthreads();
        if (qi < nbQuery) {
            for (int r = 0; r < n; ++r) {
                const unsigned int* row = tile + r * W;
                unsigned int dot = 0;
#pragma unroll
                for (int k = 0; k < W; ++k) dot = dot4u8(qreg[k], row[k], dot);
                const int d2 = int(qnorm + tileNorm[r]) - 2 * int(dot);   // exact, >= 0
                best.push(base + r, float(d2));
            }
        }
        __syncthreads();
    }
    if (qi < nbQuery) {
        outIdx[2 * qi] = best.i0; outIdx[2 * qi + 1] = best.i1;
        outDist[2 * qi] = best.d0; outDist[2 * qi + 1] = best.d1;
    }
}

// The same search with two structural changes, kept as an opt-in experiment (0.3.3): each thread
// carries Q queries and reads each database word once for all Q (uint4 reads, Q dot4 per read),
// and the database is cut into slices along blockIdx.y so a search is hundreds of blocks rather
// than tens; each slice yields its own 2-NN per query and merge2 picks the global two.
//
// Measured on the engine bay (1930 searches, 38.6 M query descriptors, RX 9070): knn2_u8 12.6 s,
// this with Q=4 23.0 s (192 VGPRs, 48 B/lane of scratch, occupancy 8) and with Q=2 13.8 s
// (119 VGPRs, no scratch, occupancy 12). So the original is NOT LDS-bound - its tile reads are
// wavefront broadcasts, as its header says - and register reuse buys nothing; the original sits
// at roughly a quarter of the card's dot4 rate with about 1 ms of transfer and launch per search,
// and closing that gap is a different kind of work (2-D register tiles, tuned occupancy). Off by
// default; CHESHIRE_MATCHER_SLICED=1 selects it, for measuring on other cards.
//
// Byte-identical to knn2_u8 by construction: distances are the same integers (dot4 is exact and
// the order of accumulation over the 32 words is the same), rows within a slice are pushed in
// index order with the same strict '<', and the merge takes the two smallest (distance, index)
// pairs, which is what a sequential push over all rows produces - on a tie the lower index was
// pushed first and kept. Verified byte-identical on the engine bay chunk (165,162 match lines).
constexpr int kSliceRows = 2048;        // rows per slice (8 tiles); slices = ceil(rows / this)
constexpr int kQ = 2;                   // queries per thread (4 spilled: 192 VGPRs, 48 B scratch, half the occupancy; 2 = 119 VGPRs, none)

template<int DIM, int Q>
__global__ void knn2_u8_sliced(const unsigned int* __restrict__ db, const unsigned int* __restrict__ dbNorm, int rows,
                               const unsigned int* __restrict__ q, int nbQuery,
                               int* __restrict__ partIdx, float* __restrict__ partDist)
{
    constexpr int W = DIM / 4;
    __shared__ __align__(16) unsigned int tile[kTileRowsU8 * W];
    __shared__ unsigned int tileNorm[kTileRowsU8];
    const int slice = blockIdx.y;
    const int rBegin = slice * kSliceRows;
    const int rEnd = min(rows, rBegin + kSliceRows);
    const int q0 = (blockIdx.x * blockDim.x + threadIdx.x) * Q;
    unsigned int qreg[Q][W];
    unsigned int qnorm[Q];
    Best2 best[Q];
#pragma unroll
    for (int j = 0; j < Q; ++j) {
        qnorm[j] = 0;
        const int qi = q0 + j;
        if (qi < nbQuery) {
#pragma unroll
            for (int k = 0; k < W; ++k) { qreg[j][k] = q[size_t(qi) * W + k]; qnorm[j] = dot4u8(qreg[j][k], qreg[j][k], qnorm[j]); }
        } else {
#pragma unroll
            for (int k = 0; k < W; ++k) qreg[j][k] = 0u;
        }
    }
    for (int base = rBegin; base < rEnd; base += kTileRowsU8) {
        const int n = min(kTileRowsU8, rEnd - base);
        for (int t = threadIdx.x; t < n * W; t += blockDim.x) tile[t] = db[size_t(base) * W + t];
        for (int t = threadIdx.x; t < n; t += blockDim.x) tileNorm[t] = dbNorm[base + t];
        __syncthreads();
        if (q0 < nbQuery) {
            for (int r = 0; r < n; ++r) {
                const uint4* row4 = reinterpret_cast<const uint4*>(tile + r * W);
                unsigned int dot[Q];
#pragma unroll
                for (int j = 0; j < Q; ++j) dot[j] = 0;
#pragma unroll
                for (int k = 0; k < W / 4; ++k) {
                    const uint4 w = row4[k];
#pragma unroll
                    for (int j = 0; j < Q; ++j) {
                        dot[j] = dot4u8(qreg[j][4 * k], w.x, dot[j]);
                        dot[j] = dot4u8(qreg[j][4 * k + 1], w.y, dot[j]);
                        dot[j] = dot4u8(qreg[j][4 * k + 2], w.z, dot[j]);
                        dot[j] = dot4u8(qreg[j][4 * k + 3], w.w, dot[j]);
                    }
                }
                const unsigned int rn = tileNorm[r];
#pragma unroll
                for (int j = 0; j < Q; ++j) {
                    const int d2 = int(qnorm[j] + rn) - 2 * int(dot[j]);   // exact, >= 0
                    best[j].push(base + r, float(d2));
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < Q; ++j) {
        const int qi = q0 + j;
        if (qi < nbQuery) {
            const size_t o = (size_t(slice) * nbQuery + qi) * 2;
            partIdx[o] = best[j].i0; partIdx[o + 1] = best[j].i1;
            partDist[o] = best[j].d0; partDist[o + 1] = best[j].d1;
        }
    }
}

// One thread per query: the two smallest (distance, index) pairs over every slice's two.
__global__ void merge2(const int* __restrict__ partIdx, const float* __restrict__ partDist, int slices, int nbQuery,
                       int* __restrict__ outIdx, float* __restrict__ outDist)
{
    const int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nbQuery) return;
    int i0 = -1, i1 = -1; float d0 = 3.4e38f, d1 = 3.4e38f;
    for (int s = 0; s < slices; ++s) {
        const size_t o = (size_t(s) * nbQuery + qi) * 2;
        for (int t = 0; t < 2; ++t) {
            const int i = partIdx[o + t]; const float d = partDist[o + t];
            if (i < 0) continue;
            // (d, i) lexicographic: a sequential push keeps the lower index on equal distances
            if (d < d0 || (d == d0 && i < i0)) { d1 = d0; i1 = i0; d0 = d; i0 = i; }
            else if (d < d1 || (d == d1 && i < i1)) { d1 = d; i1 = i; }
        }
    }
    outIdx[2 * qi] = i0; outIdx[2 * qi + 1] = i1;
    outDist[2 * qi] = d0; outDist[2 * qi + 1] = d1;
}

template<int DIM>
__global__ void knn2_f32(const float* __restrict__ db, int rows, const float* __restrict__ q, int nbQuery,
                         int* __restrict__ outIdx, float* __restrict__ outDist)
{
    __shared__ float tile[kTileRowsF32 * DIM];
    const int qi = blockIdx.x * blockDim.x + threadIdx.x;
    float qreg[DIM];
    if (qi < nbQuery)
        for (int k = 0; k < DIM; ++k) qreg[k] = q[size_t(qi) * DIM + k];
    else
        for (int k = 0; k < DIM; ++k) qreg[k] = 0.f;
    Best2 best;
    for (int base = 0; base < rows; base += kTileRowsF32) {
        const int n = min(kTileRowsF32, rows - base);
        for (int t = threadIdx.x; t < n * DIM; t += blockDim.x) tile[t] = db[size_t(base) * DIM + t];
        __syncthreads();
        if (qi < nbQuery) {
            for (int r = 0; r < n; ++r) {
                const float* row = tile + r * DIM;
                float acc = 0.f;
#pragma unroll
                for (int k = 0; k < DIM; ++k) { const float d = qreg[k] - row[k]; acc = fmaf(d, d, acc); }
                best.push(base + r, acc);
            }
        }
        __syncthreads();
    }
    if (qi < nbQuery) {
        outIdx[2 * qi] = best.i0; outIdx[2 * qi + 1] = best.i1;
        outDist[2 * qi] = best.d0; outDist[2 * qi + 1] = best.d1;
    }
}

bool g_checked = false, g_available = false;
std::mutex g_mutex;

// CHESHIRE_GPU_MATCHER_LOG=1: cumulative time inside build()/search2() (uploads, kernel, downloads)
// printed at exit, to split the matcher's share of "Regions Matching" from the CPU work around it.
struct Profile {
    bool on = ::cheshire::env::flag("CHESHIRE_GPU_MATCHER_LOG");
    double buildSec = 0, searchSec = 0; size_t builds = 0, searches = 0, queries = 0;
    ~Profile() {
        if (on) std::fprintf(stderr, "[cheshire] matcher profile: %zu builds %.2f s, %zu searches %.2f s (%zu query descriptors)\n",
                             builds, buildSec, searches, searchSec, queries);
    }
} g_profile;
struct ScopedTimer {
    double& acc; std::chrono::steady_clock::time_point t0;
    explicit ScopedTimer(double& a) : acc(a), t0(std::chrono::steady_clock::now()) {}
    ~ScopedTimer() { acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};

void logOnce(const char* what) {
    static bool done = false;
    if (done) return;
    done = true;
    std::fprintf(stderr, "[cheshire] matcher: %s\n", what);
}

}  // namespace

bool available()
{
    std::lock_guard<std::mutex> g(g_mutex);
    if (g_checked) return g_available;
    g_checked = true;
    if (!::cheshire::env::flag("CHESHIRE_GPU_MATCHER", true)) { logOnce("GPU brute-force disabled by CHESHIRE_GPU_MATCHER=0"); return g_available = false; }
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { logOnce("no GPU device, CPU matcher"); return g_available = false; }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { logOnce("cannot query device 0, CPU matcher"); return g_available = false; }
    char line[320];
    std::snprintf(line, sizeof line, "GPU brute-force L2 2-NN on %s (exact; CHESHIRE_GPU_MATCHER=0 for the CPU matcher)", p.name);
    logOnce(line);
    return g_available = true;
}

bool supportsDim(int dim) { return dim == 128 || dim == 64; }

namespace {
struct CheckStats {
    std::mutex m;
    long long checked[2] = {0, 0}, identical[2] = {0, 0}, nearest[2] = {0, 0}, distance[2] = {0, 0};
    ~CheckStats() {
        for (int f = 0; f < 2; ++f)
            if (checked[f] > 0)
                std::fprintf(stderr, "[cheshire] GPU matcher check%s: %lld of %lld sampled queries identical to upstream's brute force (nearest row differs %lld, distances differ %lld)\n",
                             f ? " (float descriptors)" : "", identical[f], checked[f], nearest[f], distance[f]);
    }
} g_check;
}  // namespace

bool checkEnabled()
{
    static const bool on = ::cheshire::env::flag("CHESHIRE_GPU_MATCHER_CHECK");
    return on;
}

int checkSample()
{
    static const int n = static_cast<int>(std::max(1LL, ::cheshire::env::integer("CHESHIRE_GPU_MATCHER_CHECK_SAMPLE", 64)));
    return n;
}

void checkRecord(bool isFloat, long long checked, long long identical, long long nearestDiffers, long long distanceDiffers)
{
    std::lock_guard<std::mutex> lock(g_check.m);
    const int f = isFloat ? 1 : 0;
    g_check.checked[f] += checked;
    g_check.identical[f] += identical;
    g_check.nearest[f] += nearestDiffers;
    g_check.distance[f] += distanceDiffers;
}

struct KnnMatcher::Impl {
    void* db = nullptr; size_t dbCap = 0;
    void* dbNorm = nullptr; size_t dbNormCap = 0;   // uint8 path: |row|^2 per database row
    void* q = nullptr; size_t qCap = 0;
    int* idx = nullptr; float* dist = nullptr; size_t outCap = 0;   // in queries
    void* pIdx = nullptr; size_t pIdxCap = 0;       // per-slice partial 2-NN (sliced kernel)
    void* pDist = nullptr; size_t pDistCap = 0;
    int rows = 0, dim = 0; bool isFloat = false;
    size_t rowBytes() const { return size_t(dim) * (isFloat ? 4 : 1); }
    static bool grow(void** p, size_t* cap, size_t need) {
        if (need <= *cap) return true;
        if (*p) cheshire::devFree(*p);
        *p = nullptr; *cap = 0;
        if (cheshire::devMalloc(p, need) != cudaSuccess) return false;
        *cap = need; return true;
    }
    ~Impl() { if (db) cheshire::devFree(db); if (dbNorm) cheshire::devFree(dbNorm); if (q) cheshire::devFree(q); if (idx) cheshire::devFree(idx); if (dist) cheshire::devFree(dist);
              if (pIdx) cheshire::devFree(pIdx); if (pDist) cheshire::devFree(pDist); }
};

KnnMatcher::KnnMatcher() : impl_(new Impl) {}
KnnMatcher::~KnnMatcher() { delete impl_; }
int KnnMatcher::rows() const { return impl_->rows; }

bool KnnMatcher::build(const void* data, int rows, int dim, bool isFloat)
{
    Impl& m = *impl_;
    if (rows < 1 || !supportsDim(dim)) { m.rows = 0; return false; }
    ScopedTimer timer(g_profile.buildSec); g_profile.builds++;
    m.rows = rows; m.dim = dim; m.isFloat = isFloat;
    const size_t bytes = size_t(rows) * m.rowBytes();
    if (!Impl::grow(&m.db, &m.dbCap, bytes)) return false;
    if (cudaMemcpy(m.db, data, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (!isFloat) {
        if (!Impl::grow(&m.dbNorm, &m.dbNormCap, size_t(rows) * sizeof(unsigned int))) return false;
        const dim3 block(256), grid((rows + 255) / 256);
        if (dim == 128) rowNormsU8<128><<<grid, block>>>((const unsigned int*)m.db, rows, (unsigned int*)m.dbNorm);
        else            rowNormsU8<64><<<grid, block>>>((const unsigned int*)m.db, rows, (unsigned int*)m.dbNorm);
        if (cudaGetLastError() != cudaSuccess) return false;
    }
    return true;
}

bool KnnMatcher::search2(const void* queries, int nbQuery, int* idx, float* dist)
{
    Impl& m = *impl_;
    if (m.rows < 2 || nbQuery < 1) return false;
    ScopedTimer timer(g_profile.searchSec); g_profile.searches++; g_profile.queries += size_t(nbQuery);
    const size_t qBytes = size_t(nbQuery) * m.rowBytes();
    if (!Impl::grow(&m.q, &m.qCap, qBytes)) return false;
    if (size_t(nbQuery) > m.outCap) {
        if (m.idx) cheshire::devFree(m.idx); if (m.dist) cheshire::devFree(m.dist);
        m.idx = nullptr; m.dist = nullptr; m.outCap = 0;
        if (cheshire::devMalloc((void**)&m.idx, size_t(nbQuery) * 2 * sizeof(int)) != cudaSuccess) return false;
        if (cheshire::devMalloc((void**)&m.dist, size_t(nbQuery) * 2 * sizeof(float)) != cudaSuccess) return false;
        m.outCap = size_t(nbQuery);
    }
    if (cudaMemcpy(m.q, queries, qBytes, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    const dim3 block(kQueriesPerBlock), grid((nbQuery + kQueriesPerBlock - 1) / kQueriesPerBlock);
    if (m.isFloat) {
        if (m.dim == 128) knn2_f32<128><<<grid, block>>>((const float*)m.db, m.rows, (const float*)m.q, nbQuery, m.idx, m.dist);
        else              knn2_f32<64><<<grid, block>>>((const float*)m.db, m.rows, (const float*)m.q, nbQuery, m.idx, m.dist);
    } else {
        static const bool sliced = ::cheshire::env::flag("CHESHIRE_MATCHER_SLICED");
        if (!sliced) {
            if (m.dim == 128) knn2_u8<128><<<grid, block>>>((const unsigned int*)m.db, (const unsigned int*)m.dbNorm, m.rows, (const unsigned int*)m.q, nbQuery, m.idx, m.dist);
            else              knn2_u8<64><<<grid, block>>>((const unsigned int*)m.db, (const unsigned int*)m.dbNorm, m.rows, (const unsigned int*)m.q, nbQuery, m.idx, m.dist);
        } else {
            // sliced: Q queries per thread, the database in slices along y, partials merged after
            const int slices = (m.rows + kSliceRows - 1) / kSliceRows;
            const size_t partN = size_t(slices) * nbQuery * 2;
            if (!Impl::grow(&m.pIdx, &m.pIdxCap, partN * sizeof(int))) return false;
            if (!Impl::grow(&m.pDist, &m.pDistCap, partN * sizeof(float))) return false;
            const int threads = 128;  // x 2 queries = 256 queries per block
            const dim3 sblock(threads), sgrid((nbQuery + threads * kQ - 1) / (threads * kQ), slices);
            if (m.dim == 128) knn2_u8_sliced<128, kQ><<<sgrid, sblock>>>((const unsigned int*)m.db, (const unsigned int*)m.dbNorm, m.rows, (const unsigned int*)m.q, nbQuery, (int*)m.pIdx, (float*)m.pDist);
            else              knn2_u8_sliced<64, kQ><<<sgrid, sblock>>>((const unsigned int*)m.db, (const unsigned int*)m.dbNorm, m.rows, (const unsigned int*)m.q, nbQuery, (int*)m.pIdx, (float*)m.pDist);
            if (cudaGetLastError() != cudaSuccess) return false;
            merge2<<<grid, block>>>((const int*)m.pIdx, (const float*)m.pDist, slices, nbQuery, m.idx, m.dist);
        }
    }
    if (cudaGetLastError() != cudaSuccess) return false;
    if (cudaMemcpy(idx, m.idx, size_t(nbQuery) * 2 * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    if (cudaMemcpy(dist, m.dist, size_t(nbQuery) * 2 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    return true;
}

}  // namespace gpu
}  // namespace matching
}  // namespace aliceVision
