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
//
// For 128-byte uint8 descriptors two faster kernels give the same rows and distances, bit for bit:
// knn2_wmma on RDNA4 (gfx120x: the 8-bit matrix instructions, 9.4x knn2_u8 on the RX 9070) and
// knn2_dot4q on NVIDIA sm_61+ (2.0x on the GTX 1080 Ti) and RDNA2 (gfx103x: 1.9x on the RX 6750 XT);
// everything else keeps knn2_u8 until it is measured. CHESHIRE_MATCHER_KERNEL=u8|sliced|dot4q|wmma
// picks one; the log line names the choice.
#include "gpuMatcher.hpp"
#include <cuda_runtime.h>
#include <aliceVision/depthMap/cuda/hip/cheshire/devalloc.h>  // cheshire: bridge on both backends
#include <aliceVision/depthMap/cuda/hip/cheshire/env.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <string>

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
// (v_dot4_u32_u8 on Vega 20, RDNA1's gfx1011/gfx1012 and RDNA2 on; dp4a on sm_61+), else four
// multiply-adds (gfx1010, gfx900). Exact in every case.
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

// ---------------------------------------------------------------------------------------------------------- 0.4.x
// knn2_wmma and knn2_dot4q keep a query's two best as packed integer keys over windows of 240 rows:
// key = 256 (2 dot - rowNorm) + f. 2 dot - rowNorm is queryNorm - distance, a per-query constant away from
// the distance, and |2 dot - rowNorm| <= 128 * 255^2 < 2^23, so a key fits an int32. The tie field f is
// 254 - rowInWindow for the window's rows and 255 for the two carried in from earlier windows (lower rows,
// so they win ties). Per candidate that is one shift-add, a max and a min-max. At the end of a window f
// says whether each of the two is new (row base + 254 - f) or carried (rows i0, then i1), and both are
// marked carried; equal keys always hold rows i0 < i1. Integer sums in any grouping are the same integers,
// so both kernels give knn2_u8's rows and distances, bit for bit (hip/tests/gpumatcher_wmma: every query
// of 8 engine-bay pairs of 20000 x 20000, and wmma_edge.hip's sizes 0 to 7201, heavy ties and extremes).
constexpr int kKeyWindowRows = 240;       // f = 255 stays free for the carried two
constexpr int kEmptyKey = INT_MIN + 255;  // carried, with rows INT_MAX: nothing yet

__device__ __forceinline__ void carryWindow(int& p0, int& p1, int& i0, int& i1, int base)
{
    const int top = base + 254;
    const int f0 = p0 & 255, f1 = p1 & 255;
    const bool c0 = f0 == 255, c1 = f1 == 255;
    const int n1 = c1 ? (c0 ? i1 : i0) : top - f1;
    if (!c0) i0 = top - f0;
    i1 = n1;
    p0 |= 255; p1 |= 255;
}

// knn2_dot4q: Q queries per thread, so every shared-memory read of a database word feeds Q dot4, one per
// query, instead of one. On Pascal a warp's broadcast read costs about what the 32 dp4a it feeds for one
// query cost, so knn2_u8 is bound by those reads: Q = 2 in blocks of 128 is 2.03x knn2_u8 on the GTX
// 1080 Ti (cuda_matcher.cu). On the RX 9070 it is 1.13x (blocks of 256), on the RX 6750 XT (RDNA2,
// gfx1031) 1.9x in FeatureMatching's own profile (41 views: kernels 13.46 s to 7.05 s); RDNA1/3 are unmeasured.
template<int Q>
__global__ void knn2_dot4q(const unsigned int* __restrict__ db, const unsigned int* __restrict__ dbNorm, int rows,
                           const unsigned int* __restrict__ q, const unsigned int* __restrict__ qNorm, int nbQuery,
                           int* __restrict__ outIdx, float* __restrict__ outDist)
{
    constexpr int W = 32;
    __shared__ __align__(16) unsigned int tile[kKeyWindowRows * W];
    __shared__ int tileC[kKeyWindowRows];
    const int q0 = (blockIdx.x * blockDim.x + threadIdx.x) * Q;
    unsigned int qreg[Q][W];
    int p0[Q], p1[Q], i0[Q], i1[Q];
#pragma unroll
    for (int j = 0; j < Q; ++j) {
        const int qi = q0 + j;
#pragma unroll
        for (int k = 0; k < W; ++k) qreg[j][k] = qi < nbQuery ? q[size_t(qi) * W + k] : 0u;
        p0[j] = kEmptyKey; p1[j] = kEmptyKey; i0[j] = INT_MAX; i1[j] = INT_MAX;
    }
    for (int base = 0; base < rows; base += kKeyWindowRows) {
        const int n = min(kKeyWindowRows, rows - base);
        for (int t = threadIdx.x; t < n * W; t += blockDim.x) tile[t] = db[size_t(base) * W + t];
        for (int t = threadIdx.x; t < n; t += blockDim.x) tileC[t] = int((254u - unsigned(t)) - (dbNorm[base + t] << 8));
        __syncthreads();
        for (int r = 0; r < n; ++r) {
            unsigned int acc[Q];
#pragma unroll
            for (int j = 0; j < Q; ++j) acc[j] = 0u;
#pragma unroll
            for (int c = 0; c < W / 4; ++c) {
                const uint4 w = reinterpret_cast<const uint4*>(tile + r * W)[c];
#pragma unroll
                for (int j = 0; j < Q; ++j) {
                    acc[j] = dot4u8(qreg[j][4 * c], w.x, acc[j]);
                    acc[j] = dot4u8(qreg[j][4 * c + 1], w.y, acc[j]);
                    acc[j] = dot4u8(qreg[j][4 * c + 2], w.z, acc[j]);
                    acc[j] = dot4u8(qreg[j][4 * c + 3], w.w, acc[j]);
                }
            }
            const unsigned int cr = unsigned(tileC[r]);
#pragma unroll
            for (int j = 0; j < Q; ++j) {
                const int y = int((acc[j] << 9) + cr);   // wrapping shift-add; the key itself is in range
                p1[j] = max(min(p0[j], y), p1[j]);
                p0[j] = max(p0[j], y);
            }
        }
#pragma unroll
        for (int j = 0; j < Q; ++j) carryWindow(p0[j], p1[j], i0[j], i1[j], base);
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < Q; ++j) {
        const int qi = q0 + j;
        if (qi < nbQuery) {
            const int qn = int(qNorm[qi]);
            outIdx[2 * qi] = i0[j] == INT_MAX ? -1 : i0[j]; outIdx[2 * qi + 1] = i1[j] == INT_MAX ? -1 : i1[j];
            outDist[2 * qi] = i0[j] == INT_MAX ? 3.4e38f : float(qn - (p0[j] >> 8));
            outDist[2 * qi + 1] = i1[j] == INT_MAX ? 3.4e38f : float(qn - (p1[j] >> 8));
        }
    }
}

#if defined(__HIP_PLATFORM_AMD__)
// knn2_wmma: RDNA4's v_wmma_i32_16x16x16_iu8 (gfx120x only; other code objects get an empty body and the
// host never launches it there). A wave takes kWmmaTiles 16-query tiles (A, rows m) and walks the
// database 16 rows at a time (B, columns n); eight instructions cover the 128 bytes, the dot products
// exact int32. Layout, measured by hip/tests/gpumatcher_wmma/wmma_probe.hip: lane l holds A row l % 16
// and B column l % 16, bytes 8 (l / 16) .. +7 of each 16-byte step, and receives D for column l % 16,
// rows 8 (l / 16) + v. Each lane keeps a best two for its 8 queries over the rows n = l % 16 (mod 16);
// a butterfly over the 16 lanes of a half merges them lexicographically on (distance, row). The database
// goes through LDS in the 240-row key windows, transposed (k-group, row) so the B reads of a k-group are
// consecutive 8-byte words, and filled with 16-byte global reads written as two 8-byte LDS stores that
// never share a bank. RX 9070, 8 engine-bay pairs: 21.4 ms against knn2_u8's 197 ms.
typedef int cheshireV2i __attribute__((ext_vector_type(2)));
typedef int cheshireV8i __attribute__((ext_vector_type(8)));
constexpr int kWmmaWaves = 8;   // waves per block
constexpr int kWmmaTiles = 2;   // 16-query tiles per wave (3 is 2 % faster at 206 VGPRs; 4 spills)

struct KeyBest2 { int d0, i0, d1, i1; };

__device__ __forceinline__ bool lexLess(int d, int i, int e, int j) { return d < e || (d == e && i < j); }

__device__ __forceinline__ void mergeBest2(KeyBest2& a, int b0, int bi0, int b1, int bi1)
{
    // both sorted on (distance, row); keep the first two of their merge
    if (lexLess(b0, bi0, a.d0, a.i0)) {
        if (lexLess(b1, bi1, a.d0, a.i0)) { a.d1 = b1; a.i1 = bi1; } else { a.d1 = a.d0; a.i1 = a.i0; }
        a.d0 = b0; a.i0 = bi0;
    } else if (lexLess(b0, bi0, a.d1, a.i1)) {
        a.d1 = b0; a.i1 = bi0;
    }
}

__global__ void __launch_bounds__(kWmmaWaves * 32)
knn2_wmma(const unsigned int* __restrict__ db, const unsigned int* __restrict__ dbNorm, int rows,
          const unsigned int* __restrict__ q, const unsigned int* __restrict__ qNorm, int nbQuery,
          int* __restrict__ outIdx, float* __restrict__ outDist)
{
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12_generic__))
    constexpr int W = 32;
    constexpr int MT = kWmmaTiles;
    __shared__ __align__(16) unsigned int tile[kKeyWindowRows * W];
    __shared__ int tileC[kKeyWindowRows];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int col = lane & 15, h = lane >> 4;
    const int qbase = (blockIdx.x * kWmmaWaves + wave) * 16 * MT;

    cheshireV2i a[MT][8];
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        const int qr = qbase + 16 * t + col;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (qr < nbQuery) { a[t][s][0] = int(q[size_t(qr) * W + 4 * s + 2 * h]); a[t][s][1] = int(q[size_t(qr) * W + 4 * s + 2 * h + 1]); }
            else { a[t][s][0] = 0; a[t][s][1] = 0; }
        }
    }
    int p0[MT][8], p1[MT][8], i0[MT][8], i1[MT][8];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int v = 0; v < 8; ++v) { p0[t][v] = kEmptyKey; p1[t][v] = kEmptyKey; i0[t][v] = INT_MAX; i1[t][v] = INT_MAX; }

    for (int base = 0; base < rows; base += kKeyWindowRows) {
        const int n = min(kKeyWindowRows, rows - base);
        const int ntiles = (n + 15) >> 4;
        // wave w fills n-tiles w, w + 8; lane (col, h) moves 16-byte chunks 2j + h of row col: k-groups 2i, 2i + 1
        for (int nt = wave; nt < ntiles; nt += kWmmaWaves) {
            const int r = nt * 16 + col;
            unsigned int* T = tile + nt * 512;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int i = 2 * j + h;
                uint4 w = make_uint4(0u, 0u, 0u, 0u);
                if (r < n) w = reinterpret_cast<const uint4*>(db + size_t(base + r) * W)[i];
                *reinterpret_cast<uint2*>(T + ((2 * i) * 16 + col) * 2) = make_uint2(w.x, w.y);
                *reinterpret_cast<uint2*>(T + ((2 * i + 1) * 16 + col) * 2) = make_uint2(w.z, w.w);
            }
        }
        // past the end: zero descriptors and the lowest key, never chosen
        for (int t = threadIdx.x; t < ntiles * 16; t += blockDim.x)
            tileC[t] = t < n ? int((254u - unsigned(t)) - (dbNorm[base + t] << 8)) : INT_MIN;
        __syncthreads();
        for (int nt = 0; nt < ntiles; ++nt) {
            const unsigned int* T = tile + nt * 512;
            cheshireV2i b[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint2 w = *reinterpret_cast<const uint2*>(T + ((2 * s + h) * 16 + col) * 2);
                b[s][0] = int(w.x); b[s][1] = int(w.y);
            }
            const unsigned int c = unsigned(tileC[nt * 16 + col]);
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                cheshireV8i acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int s = 0; s < 8; ++s) acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a[t][s], false, b[s], acc, false);
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const int y = int((unsigned(acc[v]) << 9) + c);
                    p1[t][v] = max(min(p0[t][v], y), p1[t][v]);
                    p0[t][v] = max(p0[t][v], y);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int v = 0; v < 8; ++v) carryWindow(p0[t][v], p1[t][v], i0[t][v], i1[t][v], base);
        __syncthreads();
    }
    KeyBest2 best[MT][8];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int m = qbase + 16 * t + 8 * h + v;
            const int qn = m < nbQuery ? int(qNorm[m]) : 0;
            best[t][v].i0 = i0[t][v]; best[t][v].d0 = i0[t][v] == INT_MAX ? INT_MAX : qn - (p0[t][v] >> 8);
            best[t][v].i1 = i1[t][v]; best[t][v].d1 = i1[t][v] == INT_MAX ? INT_MAX : qn - (p1[t][v] >> 8);
        }
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int v = 0; v < 8; ++v)
#pragma unroll
            for (int x = 1; x < 16; x <<= 1) {
                const int b0 = __shfl_xor(best[t][v].d0, x, 16), bi0 = __shfl_xor(best[t][v].i0, x, 16);
                const int b1 = __shfl_xor(best[t][v].d1, x, 16), bi1 = __shfl_xor(best[t][v].i1, x, 16);
                mergeBest2(best[t][v], b0, bi0, b1, bi1);
            }
    if (col == 0) {
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int m = qbase + 16 * t + 8 * h + v;
                if (m < nbQuery) {
                    const KeyBest2& b = best[t][v];
                    outIdx[2 * m] = b.i0 == INT_MAX ? -1 : b.i0; outIdx[2 * m + 1] = b.i1 == INT_MAX ? -1 : b.i1;
                    outDist[2 * m] = b.i0 == INT_MAX ? 3.4e38f : float(b.d0);
                    outDist[2 * m + 1] = b.i1 == INT_MAX ? 3.4e38f : float(b.d1);
                }
            }
    }
#endif
}
#endif  // __HIP_PLATFORM_AMD__

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

// The uint8 kernel for 128-byte descriptors, chosen once in available(); 64-byte ones always take knn2_u8.
enum U8Kernel { kU8, kSliced, kDot4q, kWmma };
const char* const kU8KernelNames[] = {"knn2_u8", "knn2_u8_sliced", "knn2_dot4q", "knn2_wmma"};
int g_u8Kernel = kU8;

// CHESHIRE_GPU_MATCHER_LOG=1: cumulative time inside build()/search2() (uploads, kernel, downloads)
// printed at exit, to split the matcher's share of "Regions Matching" from the CPU work around it.
struct Profile {
    bool on = ::cheshire::env::flag("CHESHIRE_GPU_MATCHER_LOG");
    double buildSec = 0, searchSec = 0; size_t builds = 0, searches = 0, queries = 0;
    double uploadSec = 0, kernelSec = 0, downloadSec = 0;   // the searches' three phases (the kernel phase synchronized)
    ~Profile() {
        if (on) std::fprintf(stderr, "[cheshire] matcher profile: %zu builds %.2f s, %zu searches %.2f s (%zu query descriptors; "
                             "upload %.2f s, kernels %.2f s, download %.2f s)\n",
                             builds, buildSec, searches, searchSec, queries, uploadSec, kernelSec, downloadSec);
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
    // The uint8 kernel: the matrix instructions on gfx120x, the dot4 kernel on NVIDIA sm_61+ and on RDNA2
    // (gfx103x, measured on gfx1031; the family shares its ISA), knn2_u8 on the cards the new kernels are
    // unmeasured on. CHESHIRE_MATCHER_KERNEL=u8|sliced|dot4q|wmma overrides it (wmma only where the device has
    // gfx120x code); CHESHIRE_MATCHER_SLICED=1 is the older switch for "sliced".
#if defined(__HIP_PLATFORM_AMD__)
    const bool gfx120 = std::strncmp(p.gcnArchName, "gfx120", 6) == 0;
    const bool gfx103 = std::strncmp(p.gcnArchName, "gfx103", 6) == 0;
    g_u8Kernel = gfx120 ? kWmma : gfx103 ? kDot4q : kU8;
#else
    const bool gfx120 = false;
    g_u8Kernel = (p.major * 10 + p.minor >= 61) ? kDot4q : kU8;
#endif
    if (::cheshire::env::flag("CHESHIRE_MATCHER_SLICED")) g_u8Kernel = kSliced;
    const std::string want = ::cheshire::env::text("CHESHIRE_MATCHER_KERNEL");
    if (want == "u8") g_u8Kernel = kU8;
    else if (want == "sliced") g_u8Kernel = kSliced;
    else if (want == "dot4q") g_u8Kernel = kDot4q;
    else if (want == "wmma" && gfx120) g_u8Kernel = kWmma;
    else if (!want.empty())
        std::fprintf(stderr, "[cheshire] matcher: CHESHIRE_MATCHER_KERNEL=%s not usable here, keeping %s\n", want.c_str(), kU8KernelNames[g_u8Kernel]);
    char line[400];
    std::snprintf(line, sizeof line, "GPU brute-force L2 2-NN on %s, %s for 128-byte uint8 (exact; CHESHIRE_GPU_MATCHER=0 for the CPU matcher)",
                  p.name, kU8KernelNames[g_u8Kernel]);
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
    // A search's results. One contiguous buffer copied once into pinned memory was slower on the RX 9070 (FeatureMatching
    // engine bay 26.0 s against 24.7: HIP's default pinned memory is coherent, slow for the CPU to read), so two copies.
    int* idx = nullptr; float* dist = nullptr; size_t outCap = 0;   // in queries
    void* pIdx = nullptr; size_t pIdxCap = 0;       // per-slice partial 2-NN (sliced kernel)
    void* pDist = nullptr; size_t pDistCap = 0;
    void* qNorm = nullptr; size_t qNormCap = 0;     // |query|^2 for knn2_dot4q / knn2_wmma
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
              if (pIdx) cheshire::devFree(pIdx); if (pDist) cheshire::devFree(pDist); if (qNorm) cheshire::devFree(qNorm); }
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
    const bool prof = g_profile.on;
    std::chrono::steady_clock::time_point t0, t1, t2;
    if (prof) t0 = std::chrono::steady_clock::now();
    if (cudaMemcpy(m.q, queries, qBytes, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (prof) t1 = std::chrono::steady_clock::now();
    const dim3 block(kQueriesPerBlock), grid((nbQuery + kQueriesPerBlock - 1) / kQueriesPerBlock);
    if (m.isFloat) {
        if (m.dim == 128) knn2_f32<128><<<grid, block>>>((const float*)m.db, m.rows, (const float*)m.q, nbQuery, m.idx, m.dist);
        else              knn2_f32<64><<<grid, block>>>((const float*)m.db, m.rows, (const float*)m.q, nbQuery, m.idx, m.dist);
    } else {
        int kernel = g_u8Kernel;
        if (m.dim != 128 && (kernel == kDot4q || kernel == kWmma)) kernel = kU8;
        if (kernel == kDot4q || kernel == kWmma) {
            if (!Impl::grow(&m.qNorm, &m.qNormCap, size_t(nbQuery) * sizeof(unsigned int))) return false;
            rowNormsU8<128><<<dim3((nbQuery + 255) / 256), dim3(256)>>>((const unsigned int*)m.q, nbQuery, (unsigned int*)m.qNorm);
#if defined(__HIP_PLATFORM_AMD__)
            if (kernel == kWmma) {
                constexpr int perBlock = kWmmaWaves * 16 * kWmmaTiles;
                knn2_wmma<<<dim3((nbQuery + perBlock - 1) / perBlock), dim3(kWmmaWaves * 32)>>>(
                    (const unsigned int*)m.db, (const unsigned int*)m.dbNorm, m.rows, (const unsigned int*)m.q,
                    (const unsigned int*)m.qNorm, nbQuery, m.idx, m.dist);
            } else
#endif
            {
#if defined(__HIP_PLATFORM_AMD__)
                constexpr int threads = 256;   // 1.13x knn2_u8 on the RX 9070; 128 was 1.03x
#else
                constexpr int threads = 128;   // 2.03x on the GTX 1080 Ti; 256 was 1.92x
#endif
                knn2_dot4q<2><<<dim3((nbQuery + 2 * threads - 1) / (2 * threads)), dim3(threads)>>>(
                    (const unsigned int*)m.db, (const unsigned int*)m.dbNorm, m.rows, (const unsigned int*)m.q,
                    (const unsigned int*)m.qNorm, nbQuery, m.idx, m.dist);
            }
        } else if (kernel != kSliced) {
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
    if (prof) {
        // profiling only: wait for the kernels so the three phases can be told apart
        if (cudaDeviceSynchronize() != cudaSuccess) return false;
        t2 = std::chrono::steady_clock::now();
        g_profile.uploadSec += std::chrono::duration<double>(t1 - t0).count();
        g_profile.kernelSec += std::chrono::duration<double>(t2 - t1).count();
    }
    if (cudaMemcpy(idx, m.idx, size_t(nbQuery) * 2 * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    if (cudaMemcpy(dist, m.dist, size_t(nbQuery) * 2 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    if (prof) g_profile.downloadSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
    return true;
}

}  // namespace gpu
}  // namespace matching
}  // namespace aliceVision
