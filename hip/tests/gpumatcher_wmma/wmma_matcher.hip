// Cheshire experiment: exact uint8 2-NN on RDNA4's INT8 matrix units (v_wmma_i32_16x16x16_iu8), against the shipped
// kernel (knn2_u8, copied verbatim from hip/port/gpu_matcher/gpuMatcher.cu). Real descriptors: pairs of consecutive
// .desc files (u64 count, then count x 128 bytes) from a FeatureExtraction folder, image i's as queries against
// image i+1's as the database, as FeatureMatching does per pair.
//
//   wmma_matcher.exe <folder with *.desc> [pairs=8] [reps=5] [query tiles per wave=1] [kernel=1] [waves per block=8]
//
// Kernels (arg 5; every one gave all queries identical to knn2_u8 here, v5/v6 also in wmma_edge.hip):
//   0  v2  knn2_wmma128   per-candidate push into an exact best-two
//   1  v3  knn2_wmma128p  packed (distance << 8 | rowInTile) keys, min/max insertion, merged per 256-row tile
//   2  v4  knn2_wmma128s  keys without the query norm and no per-candidate predicate, 128-row windows
//   3  v5  knn2_wmma_c    keys valid across 240-row windows (carried tie field): 3 VALU per candidate
//   4  v6  knn2_wmma_c    v5 with a conflict-free LDS fill (integrated in gpuMatcher.cu as knn2_wmma)
// and the dot4 kernel restructured, for cards without matrix instructions (arg 4 is then rows per iteration or block
// size; on the RX 9070 none passed 1.23x, see docs/07; cuda_matcher.cu measures them on NVIDIA):
//   5  dot4b  RPI rows per iteration, all of a row's LDS reads together, v6's insertion   1.02x / 1.01x / 1.17x
//   6  dot4s  the wave-uniform row by scalar loads into SGPRs, no LDS                     1.23x / 1.18x / 1.14x
//   7  dot4q  two queries per thread (blocks of 64 / 128 / 256)                           0.56x / 1.03x / 1.13x
// RX 9070, 8 engine-bay pairs of 20000 x 20000, 4 timed reps each: knn2_u8 about 197 ms (8.3 TMAC/s); v3 MT=2 38.8 ms;
// v6 MT=2 21.4 ms (9.25x), MT=3 21.0 ms (9.43x). wmma_peak.hip: the card sustains 144 TMAC/s of WMMA alone and about
// 112 with v6's 3 VALU per value, so v6 runs at about 70 % of what this design can reach.
//
// Both kernels must give the same nearest and second-nearest rows and distances for every query - the same integers,
// and on equal distances the lower row, as a sequential scan with a strict '<' keeps (knn2_u8's Best2).
//
// The WMMA kernel (layout measured by wmma_probe.hip): a wave takes 16 queries (the A operand, rows m) and walks the
// database 16 rows at a time (B, columns n); eight instructions cover the 128 bytes, the dot products exact int32.
// Lane l receives D for column n = l % 16 and rows m = 8 * (l / 16) + v (v = 0..7), so each lane keeps a best-two for
// its 8 queries over the rows n = l % 16 (mod 16) - pushed in index order - and a butterfly over the 16 lanes of a half
// merges them lexicographically on (distance, row) at the end: what one sequential push over all rows yields.
// The LDS tile is stored transposed, (k-group, row) major, so the 16 lanes reading one k-group read consecutive 8-byte
// words and never share a bank.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));

// ---------------------------------------------------------------- the shipped kernel, verbatim
constexpr int kQueriesPerBlock = 256;
constexpr int kTileRowsU8 = 256;

struct Best2 {
    int i0 = -1, i1 = -1;
    float d0 = 3.4e38f, d1 = 3.4e38f;
    __device__ inline void push(int i, float d) {
        if (d < d0) { d1 = d0; i1 = i0; d0 = d; i0 = i; }
        else if (d < d1) { d1 = d; i1 = i; }
    }
};

__device__ __forceinline__ unsigned int dot4u8(unsigned int a, unsigned int b, unsigned int acc)
{
    return __builtin_amdgcn_udot4(a, b, acc, false);
}

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
        for (int t = threadIdx.x; t < n * W; t += blockDim.x) tile[t] = db[size_t(base) * W + t];
        for (int t = threadIdx.x; t < n; t += blockDim.x) tileNorm[t] = dbNorm[base + t];
        __syncthreads();
        if (qi < nbQuery) {
            for (int r = 0; r < n; ++r) {
                const unsigned int* row = tile + r * W;
                unsigned int dot = 0;
#pragma unroll
                for (int k = 0; k < W; ++k) dot = dot4u8(qreg[k], row[k], dot);
                const int d2 = int(qnorm + tileNorm[r]) - 2 * int(dot);
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

// ---------------------------------------------------------------- the WMMA kernel (128-byte descriptors)
constexpr int kWaves = 8;                 // waves per block = 16-query tiles per block
constexpr int kTileRowsW = 256;           // database rows per LDS tile (16 n-tiles)

struct B2 {   // best two as exact ints; empty = (INT_MAX, INT_MAX) so it sorts after every real (distance, row)
    int d0 = INT_MAX, i0 = INT_MAX, d1 = INT_MAX, i1 = INT_MAX;
};
__device__ __forceinline__ void pushB(B2& b, int i, int d)
{
    if (d < b.d0) { b.d1 = b.d0; b.i1 = b.i0; b.d0 = d; b.i0 = i; }
    else if (d < b.d1) { b.d1 = d; b.i1 = i; }
}
__device__ __forceinline__ bool lexLess(int d, int i, int e, int j) { return d < e || (d == e && i < j); }
__device__ __forceinline__ void mergeB(B2& a, int b0, int bi0, int b1, int bi1)
{
    // both lists sorted lexicographically; keep the first two of their merge
    if (lexLess(b0, bi0, a.d0, a.i0)) {
        if (lexLess(b1, bi1, a.d0, a.i0)) { a.d1 = b1; a.i1 = bi1; } else { a.d1 = a.d0; a.i1 = a.i0; }
        a.d0 = b0; a.i0 = bi0;
    } else if (lexLess(b0, bi0, a.d1, a.i1)) {
        a.d1 = b0; a.i1 = bi0;
    }
}

// MT 16-query tiles per wave share every database read; the pushed key is rowNorm - 2 dot (the query's norm is the same
// for every row, so it orders and ties exactly as the distance does) and the norm is added at the end.
template<int MT>
__global__ void __launch_bounds__(kWaves * 32)
knn2_wmma128(const unsigned int* __restrict__ db, const unsigned int* __restrict__ dbNorm, int rows,
             const unsigned int* __restrict__ q, const unsigned int* __restrict__ qNorm, int nbQuery,
             int* __restrict__ outIdx, float* __restrict__ outDist)
{
    constexpr int W = 32;                                // words per descriptor
    __shared__ __align__(16) unsigned int tile[kTileRowsW * W];   // transposed: [ntile][kgroup 0..15][row 0..15][2 words]
    __shared__ int tileNorm[kTileRowsW];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int col = lane & 15, h = lane >> 4;
    const int qbase = (blockIdx.x * kWaves + wave) * 16 * MT;

    v2i a[MT][8];
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        const int qr = qbase + 16 * t + col;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (qr < nbQuery) { a[t][s][0] = int(q[size_t(qr) * W + 4 * s + 2 * h]); a[t][s][1] = int(q[size_t(qr) * W + 4 * s + 2 * h + 1]); }
            else { a[t][s][0] = 0; a[t][s][1] = 0; }
        }
    }
    B2 best[MT][8];

    for (int base = 0; base < rows; base += kTileRowsW) {
        const int n = min(kTileRowsW, rows - base);
        for (int t = threadIdx.x; t < kTileRowsW * W; t += blockDim.x) {
            const int r = t / W, k = t % W;
            const unsigned int w = r < n ? db[size_t(base + r) * W + k] : 0u;
            tile[(r >> 4) * 512 + ((k >> 1) * 16 + (r & 15)) * 2 + (k & 1)] = w;
        }
        for (int t = threadIdx.x; t < kTileRowsW; t += blockDim.x) tileNorm[t] = t < n ? int(dbNorm[base + t]) : 0;
        __syncthreads();
        const int ntiles = (n + 15) >> 4;
        for (int nt = 0; nt < ntiles; ++nt) {
            const unsigned int* T = tile + nt * 512;
            v2i b[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint2 w = *reinterpret_cast<const uint2*>(T + ((2 * s + h) * 16 + col) * 2);
                b[s][0] = int(w.x); b[s][1] = int(w.y);
            }
            const int r = nt * 16 + col;
            const int rn = tileNorm[r], ri = base + r;
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                v8i acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int s = 0; s < 8; ++s) acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a[t][s], false, b[s], acc, false);
                if (r < n) {
#if defined(CHESHIRE_DIAG_NOPUSH)
#pragma unroll
                    for (int v = 0; v < 8; ++v) best[t][v].d0 ^= (rn - 2 * acc[v]) + ri;
#else
#pragma unroll
                    for (int v = 0; v < 8; ++v) pushB(best[t][v], ri, rn - 2 * acc[v]);
#endif
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int v = 0; v < 8; ++v)
#pragma unroll
            for (int x = 1; x < 16; x <<= 1) {
                const int b0 = __shfl_xor(best[t][v].d0, x, 16), bi0 = __shfl_xor(best[t][v].i0, x, 16);
                const int b1 = __shfl_xor(best[t][v].d1, x, 16), bi1 = __shfl_xor(best[t][v].i1, x, 16);
                mergeB(best[t][v], b0, bi0, b1, bi1);
            }
    if (col == 0) {
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int m = qbase + 16 * t + 8 * h + v;
                if (m < nbQuery) {
                    const B2& b = best[t][v];
                    const int qn = int(qNorm[m]);
                    outIdx[2 * m] = b.i0 == INT_MAX ? -1 : b.i0; outIdx[2 * m + 1] = b.i1 == INT_MAX ? -1 : b.i1;
                    outDist[2 * m] = b.i0 == INT_MAX ? 3.4e38f : float(b.d0 + qn); outDist[2 * m + 1] = b.i1 == INT_MAX ? 3.4e38f : float(b.d1 + qn);
                }
            }
    }
}

// v3: the insertion as unsigned min/max on packed keys. Within one LDS tile a row is identified by its position in the
// tile (0..255) and a distance is < 2^23 (128 * 255^2), so (d << 8) | rowInTile is one uint32 whose order is exactly
// (distance, row) - the lower row on equal distances, as a sequential push keeps. Per candidate:
//   p1 = min(p1, max(p0, x)); p0 = min(p0, x)
// and at the end of each tile its two (sorted) are unpacked and merged into the exact best two, whose rows all precede
// the next tile's. Same result as v2 and knn2_u8, bit for bit.
template<int MT>
__global__ void __launch_bounds__(kWaves * 32)
knn2_wmma128p(const unsigned int* __restrict__ db, const unsigned int* __restrict__ dbNorm, int rows,
              const unsigned int* __restrict__ q, const unsigned int* __restrict__ qNorm, int nbQuery,
              int* __restrict__ outIdx, float* __restrict__ outDist)
{
    constexpr int W = 32;
    __shared__ __align__(16) unsigned int tile[kTileRowsW * W];
    __shared__ int tileNorm[kTileRowsW];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int col = lane & 15, h = lane >> 4;
    const int qbase = (blockIdx.x * kWaves + wave) * 16 * MT;

    v2i a[MT][8];
    int qn[MT][8];
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        const int qr = qbase + 16 * t + col;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (qr < nbQuery) { a[t][s][0] = int(q[size_t(qr) * W + 4 * s + 2 * h]); a[t][s][1] = int(q[size_t(qr) * W + 4 * s + 2 * h + 1]); }
            else { a[t][s][0] = 0; a[t][s][1] = 0; }
        }
#pragma unroll
        for (int v = 0; v < 8; ++v) { const int m = qbase + 16 * t + 8 * h + v; qn[t][v] = m < nbQuery ? int(qNorm[m]) : 0; }
    }
    B2 best[MT][8];

    for (int base = 0; base < rows; base += kTileRowsW) {
        const int n = min(kTileRowsW, rows - base);
        for (int t = threadIdx.x; t < kTileRowsW * W; t += blockDim.x) {
            const int r = t / W, k = t % W;
            const unsigned int w = r < n ? db[size_t(base + r) * W + k] : 0u;
            tile[(r >> 4) * 512 + ((k >> 1) * 16 + (r & 15)) * 2 + (k & 1)] = w;
        }
        for (int t = threadIdx.x; t < kTileRowsW; t += blockDim.x) tileNorm[t] = t < n ? int(dbNorm[base + t]) : 0;
        __syncthreads();
        unsigned int p0[MT][8], p1[MT][8];
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int v = 0; v < 8; ++v) { p0[t][v] = 0xffffffffu; p1[t][v] = 0xffffffffu; }
        const int ntiles = (n + 15) >> 4;
        for (int nt = 0; nt < ntiles; ++nt) {
            const unsigned int* T = tile + nt * 512;
            v2i b[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint2 w = *reinterpret_cast<const uint2*>(T + ((2 * s + h) * 16 + col) * 2);
                b[s][0] = int(w.x); b[s][1] = int(w.y);
            }
            const int r = nt * 16 + col;                 // row within the tile
            const int rn = tileNorm[r];
            const bool live = r < n;
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                v8i acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int s = 0; s < 8; ++s) acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a[t][s], false, b[s], acc, false);
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const unsigned int d = unsigned(qn[t][v] + rn - 2 * acc[v]);   // exact, >= 0, < 2^23
                    const unsigned int x = live ? ((d << 8) | unsigned(r)) : 0xffffffffu;
                    p1[t][v] = min(p1[t][v], max(p0[t][v], x));
                    p0[t][v] = min(p0[t][v], x);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const unsigned int x0 = p0[t][v], x1 = p1[t][v];
                const int d0 = x0 == 0xffffffffu ? INT_MAX : int(x0 >> 8), i0 = x0 == 0xffffffffu ? INT_MAX : base + int(x0 & 255u);
                const int d1 = x1 == 0xffffffffu ? INT_MAX : int(x1 >> 8), i1 = x1 == 0xffffffffu ? INT_MAX : base + int(x1 & 255u);
                mergeB(best[t][v], d0, i0, d1, i1);
            }
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int v = 0; v < 8; ++v)
#pragma unroll
            for (int x = 1; x < 16; x <<= 1) {
                const int b0 = __shfl_xor(best[t][v].d0, x, 16), bi0 = __shfl_xor(best[t][v].i0, x, 16);
                const int b1 = __shfl_xor(best[t][v].d1, x, 16), bi1 = __shfl_xor(best[t][v].i1, x, 16);
                mergeB(best[t][v], b0, bi0, b1, bi1);
            }
    if (col == 0) {
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int m = qbase + 16 * t + 8 * h + v;
                if (m < nbQuery) {
                    const B2& b = best[t][v];
                    outIdx[2 * m] = b.i0 == INT_MAX ? -1 : b.i0; outIdx[2 * m + 1] = b.i1 == INT_MAX ? -1 : b.i1;
                    outDist[2 * m] = b.i0 == INT_MAX ? 3.4e38f : float(b.d0); outDist[2 * m + 1] = b.i1 == INT_MAX ? 3.4e38f : float(b.d1);
                }
            }
    }
}

// v4: per candidate one multiply-add and a max-min pair. The key is (rowNorm - 2 dot) * 128 + rowInWindow over windows
// of 128 rows: rowNorm - 2 dot = distance - queryNorm, a per-query constant away from the distance, so it orders and ties
// as the distance does; in [-2 * 128 * 255^2, 2^23] times 128 plus 127 it fits a signed int32. rr = rowNorm * 128 + r is
// computed once per 16 rows, and the key is rr - 256 * dot. Rows past the end are zero with a norm of 2^23, which no real
// row's key reaches (max 128 * 255^2 < 2^23), so they need no predicate and are dropped when a window is unpacked
// (keys >= 2^30). Per window the two best are unpacked (arithmetic shift), given back the query's norm and merged into
// the exact best two. Same result as knn2_u8, bit for bit.
constexpr int kWinRows = 128;
template<int MT>
__global__ void __launch_bounds__(kWaves * 32)
knn2_wmma128s(const unsigned int* __restrict__ db, const unsigned int* __restrict__ dbNorm, int rows,
              const unsigned int* __restrict__ q, const unsigned int* __restrict__ qNorm, int nbQuery,
              int* __restrict__ outIdx, float* __restrict__ outDist)
{
    constexpr int W = 32;
    __shared__ __align__(16) unsigned int tile[kWinRows * W];
    __shared__ int tileRR[kWinRows];                     // rowNorm * 128 + row in the window
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int col = lane & 15, h = lane >> 4;
    const int qbase = (blockIdx.x * kWaves + wave) * 16 * MT;

    v2i a[MT][8];
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        const int qr = qbase + 16 * t + col;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (qr < nbQuery) { a[t][s][0] = int(q[size_t(qr) * W + 4 * s + 2 * h]); a[t][s][1] = int(q[size_t(qr) * W + 4 * s + 2 * h + 1]); }
            else { a[t][s][0] = 0; a[t][s][1] = 0; }
        }
    }
    B2 best[MT][8];

    for (int base = 0; base < rows; base += kWinRows) {
        const int n = min(kWinRows, rows - base);
        for (int t = threadIdx.x; t < kWinRows * W; t += blockDim.x) {
            const int r = t / W, k = t % W;
            const unsigned int w = r < n ? db[size_t(base + r) * W + k] : 0u;
            tile[(r >> 4) * 512 + ((k >> 1) * 16 + (r & 15)) * 2 + (k & 1)] = w;
        }
        for (int t = threadIdx.x; t < kWinRows; t += blockDim.x) tileRR[t] = (t < n ? int(dbNorm[base + t]) : (1 << 23)) * 128 + t;
        __syncthreads();
        int p0[MT][8], p1[MT][8];
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int v = 0; v < 8; ++v) { p0[t][v] = INT_MAX; p1[t][v] = INT_MAX; }
        const int ntiles = (n + 15) >> 4;
        for (int nt = 0; nt < ntiles; ++nt) {
            const unsigned int* T = tile + nt * 512;
            v2i b[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint2 w = *reinterpret_cast<const uint2*>(T + ((2 * s + h) * 16 + col) * 2);
                b[s][0] = int(w.x); b[s][1] = int(w.y);
            }
            const int rr = tileRR[nt * 16 + col];
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                v8i acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (int s = 0; s < 8; ++s) acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a[t][s], false, b[s], acc, false);
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const int x = rr - acc[v] * 256;
                    p1[t][v] = min(p1[t][v], max(p0[t][v], x));
                    p0[t][v] = min(p0[t][v], x);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int m = qbase + 16 * t + 8 * h + v;
                const int qn = m < nbQuery ? int(qNorm[m]) : 0;
                const int x0 = p0[t][v], x1 = p1[t][v];
                const bool e0 = x0 >= (1 << 30), e1 = x1 >= (1 << 30);
                mergeB(best[t][v], e0 ? INT_MAX : (x0 >> 7) + qn, e0 ? INT_MAX : base + (x0 & 127),
                                   e1 ? INT_MAX : (x1 >> 7) + qn, e1 ? INT_MAX : base + (x1 & 127));
            }
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int v = 0; v < 8; ++v)
#pragma unroll
            for (int x = 1; x < 16; x <<= 1) {
                const int b0 = __shfl_xor(best[t][v].d0, x, 16), bi0 = __shfl_xor(best[t][v].i0, x, 16);
                const int b1 = __shfl_xor(best[t][v].d1, x, 16), bi1 = __shfl_xor(best[t][v].i1, x, 16);
                mergeB(best[t][v], b0, bi0, b1, bi1);
            }
    if (col == 0) {
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int m = qbase + 16 * t + 8 * h + v;
                if (m < nbQuery) {
                    const B2& b = best[t][v];
                    outIdx[2 * m] = b.i0 == INT_MAX ? -1 : b.i0; outIdx[2 * m + 1] = b.i1 == INT_MAX ? -1 : b.i1;
                    outDist[2 * m] = b.i0 == INT_MAX ? 3.4e38f : float(b.d0); outDist[2 * m + 1] = b.i1 == INT_MAX ? 3.4e38f : float(b.d1);
                }
            }
    }
}

// v5/v6: keys that stay valid across windows. The key is 256 (2 dot - rowNorm) + f, where 2 dot - rowNorm = queryNorm -
// distance (one query's keys order as its distances do, reversed) and f breaks ties: 254 - rowInWindow for this window's
// rows, 255 for the two carried in from earlier windows (lower rows, so they win ties). |2 dot - rowNorm| <= 128 * 255^2
// < 2^23, so a key fits a signed int32. Per candidate it is one shift-add, y = (dot << 9) + c with c = 254 - r -
// 256 rowNorm per row (wrapping; the result is in range), and the two largest are kept with a max and a min-max. Padding
// rows have c = INT_MIN, below the empty key. At the end of each window f tells whether each of the two is new (row
// base + 254 - f) or carried (rows i0 then i1), and both are then marked carried (f = 255); equal keys always hold rows
// i0 < i1. Same result as knn2_u8, bit for bit. STORE picks the LDS fill: false = one word per thread (a 16-way bank
// conflict on the transposed store), true = 16 bytes per thread written as two conflict-free 8-byte stores.
constexpr int kWinTilesC = 15, kWinRowsC = kWinTilesC * 16;   // window rows leave f = 255 free
template<int MT, bool STORE, int WV = kWaves>
__global__ void __launch_bounds__(WV * 32)
knn2_wmma_c(const unsigned int* __restrict__ db, const unsigned int* __restrict__ dbNorm, int rows,
            const unsigned int* __restrict__ q, const unsigned int* __restrict__ qNorm, int nbQuery,
            int* __restrict__ outIdx, float* __restrict__ outDist)
{
    constexpr int W = 32;
    constexpr int kEmpty = INT_MIN + 255;                 // carried, rows INT_MAX
    __shared__ __align__(16) unsigned int tile[kWinRowsC * W];
    __shared__ int tileC[kWinRowsC];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int col = lane & 15, h = lane >> 4;
    const int qbase = (blockIdx.x * WV + wave) * 16 * MT;

    v2i a[MT][8];
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
        for (int v = 0; v < 8; ++v) { p0[t][v] = kEmpty; p1[t][v] = kEmpty; i0[t][v] = INT_MAX; i1[t][v] = INT_MAX; }

    for (int base = 0; base < rows; base += kWinRowsC) {
        const int n = min(kWinRowsC, rows - base);
        const int ntiles = (n + 15) >> 4;
        if (STORE) {
            // wave w fills n-tiles w, w + 8; lane (col, h) moves 16-byte chunks 2j + h of row col: kgroups 2i and 2i + 1
            for (int nt = wave; nt < ntiles; nt += WV) {
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
        } else {
            for (int t = threadIdx.x; t < kWinRowsC * W; t += blockDim.x) {
                const int r = t / W, k = t % W;
                const unsigned int w = r < n ? db[size_t(base + r) * W + k] : 0u;
                tile[(r >> 4) * 512 + ((k >> 1) * 16 + (r & 15)) * 2 + (k & 1)] = w;
            }
        }
        for (int t = threadIdx.x; t < kWinRowsC; t += blockDim.x)
            tileC[t] = t < n ? int((254u - unsigned(t)) - (dbNorm[base + t] << 8)) : INT_MIN;
        __syncthreads();
        for (int nt = 0; nt < ntiles; ++nt) {
            const unsigned int* T = tile + nt * 512;
            v2i b[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint2 w = *reinterpret_cast<const uint2*>(T + ((2 * s + h) * 16 + col) * 2);
                b[s][0] = int(w.x); b[s][1] = int(w.y);
            }
            const unsigned int c = unsigned(tileC[nt * 16 + col]);
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                v8i acc = {0, 0, 0, 0, 0, 0, 0, 0};
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
        const int top = base + 254;
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int f0 = p0[t][v] & 255, f1 = p1[t][v] & 255;
                const bool c0 = f0 == 255, c1 = f1 == 255;
                const int n1 = c1 ? (c0 ? i1[t][v] : i0[t][v]) : top - f1;
                if (!c0) i0[t][v] = top - f0;
                i1[t][v] = n1;
                p0[t][v] |= 255; p1[t][v] |= 255;
            }
        __syncthreads();
    }
    B2 best[MT][8];
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
                mergeB(best[t][v], b0, bi0, b1, bi1);
            }
    if (col == 0) {
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int m = qbase + 16 * t + 8 * h + v;
                if (m < nbQuery) {
                    const B2& b = best[t][v];
                    outIdx[2 * m] = b.i0 == INT_MAX ? -1 : b.i0; outIdx[2 * m + 1] = b.i1 == INT_MAX ? -1 : b.i1;
                    outDist[2 * m] = b.i0 == INT_MAX ? 3.4e38f : float(b.d0); outDist[2 * m + 1] = b.i1 == INT_MAX ? 3.4e38f : float(b.d1);
                }
            }
    }
}

template<int MT, bool STORE, int WV>
void launchC(const unsigned int* db, const unsigned int* dbNorm, int rows, const unsigned int* q, const unsigned int* qNorm,
             int nbQuery, int* outIdx, float* outDist)
{
    knn2_wmma_c<MT, STORE, WV><<<(nbQuery + WV * 16 * MT - 1) / (WV * 16 * MT), WV * 32>>>(db, dbNorm, rows, q, qNorm, nbQuery, outIdx, outDist);
}

// dot4b: knn2_u8 restructured, still one query per thread. The shipped loop is one dependent chain of 32 dot4 per row
// (66 % of the instruction's rate at best, dot4_peak.hip), with the row's eight 16-byte LDS reads issued two at a time and
// a branchy insertion. Here each thread takes RPI rows per iteration - RPI independent chains reusing the query's
// registers - with all of a row's reads issued together, and the insertion is v6's: one shift-add, a max and a min-max
// on keys 256 (2 dot - rowNorm) + f over 240-row windows, the two best carried from window to window with f = 255.
// Integer sums in any grouping are the same integers, so the result is knn2_u8's, bit for bit.
constexpr int kDot4bWinRows = 240;
template<int RPI>
__global__ void __launch_bounds__(256)
knn2_dot4b(const unsigned int* __restrict__ db, const unsigned int* __restrict__ dbNorm, int rows,
           const unsigned int* __restrict__ q, const unsigned int* __restrict__ qNorm, int nbQuery,
           int* __restrict__ outIdx, float* __restrict__ outDist)
{
    constexpr int W = 32;
    constexpr int kEmpty = INT_MIN + 255;
    __shared__ __align__(16) unsigned int tile[kDot4bWinRows * W];
    __shared__ int tileC[kDot4bWinRows];
    const int qi = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int qreg[W];
    for (int k = 0; k < W; ++k) qreg[k] = qi < nbQuery ? q[size_t(qi) * W + k] : 0u;
    int p0 = kEmpty, p1 = kEmpty, i0 = INT_MAX, i1 = INT_MAX;

    for (int base = 0; base < rows; base += kDot4bWinRows) {
        const int n = min(kDot4bWinRows, rows - base);
        const int nPad = (n + RPI - 1) / RPI * RPI;   // padding rows: zero descriptors, never selected
        for (int t = threadIdx.x; t < nPad * W; t += blockDim.x) {
            const int r = t / W;
            tile[t] = r < n ? db[size_t(base) * W + t] : 0u;
        }
        for (int t = threadIdx.x; t < nPad; t += blockDim.x)
            tileC[t] = t < n ? int((254u - unsigned(t)) - (dbNorm[base + t] << 8)) : INT_MIN;
        __syncthreads();
        for (int r = 0; r < nPad; r += RPI) {
            unsigned int acc[RPI];
#pragma unroll
            for (int j = 0; j < RPI; ++j) acc[j] = 0u;
#pragma unroll
            for (int c = 0; c < W / 4; ++c) {
                uint4 w[RPI];
#pragma unroll
                for (int j = 0; j < RPI; ++j) w[j] = reinterpret_cast<const uint4*>(tile + (r + j) * W)[c];
#pragma unroll
                for (int j = 0; j < RPI; ++j) {
                    acc[j] = __builtin_amdgcn_udot4(qreg[4 * c], w[j].x, acc[j], false);
                    acc[j] = __builtin_amdgcn_udot4(qreg[4 * c + 1], w[j].y, acc[j], false);
                    acc[j] = __builtin_amdgcn_udot4(qreg[4 * c + 2], w[j].z, acc[j], false);
                    acc[j] = __builtin_amdgcn_udot4(qreg[4 * c + 3], w[j].w, acc[j], false);
                }
            }
#pragma unroll
            for (int j = 0; j < RPI; ++j) {
                const int y = int((acc[j] << 9) + unsigned(tileC[r + j]));
                p1 = max(min(p0, y), p1);
                p0 = max(p0, y);
            }
        }
        const int top = base + 254;
        const int f0 = p0 & 255, f1 = p1 & 255;
        const bool c0 = f0 == 255, c1 = f1 == 255;
        const int n1 = c1 ? (c0 ? i1 : i0) : top - f1;
        if (!c0) i0 = top - f0;
        i1 = n1;
        p0 |= 255; p1 |= 255;
        __syncthreads();
    }
    if (qi < nbQuery) {
        const int qn = int(qNorm[qi]);
        outIdx[2 * qi] = i0 == INT_MAX ? -1 : i0; outIdx[2 * qi + 1] = i1 == INT_MAX ? -1 : i1;
        outDist[2 * qi] = i0 == INT_MAX ? 3.4e38f : float(qn - (p0 >> 8));
        outDist[2 * qi + 1] = i1 == INT_MAX ? 3.4e38f : float(qn - (p1 >> 8));
    }
}

// dot4s: no LDS. All lanes of a wave compare their own query with the same database row, so the row is wave-uniform:
// read through the constant address space it comes in by scalar loads into SGPRs, and v_dot4 takes each word as an
// SGPR operand, broadcast to every lane for free, instead of eight 16-byte LDS reads per row per wave. With no LDS
// tile, occupancy is bounded by registers only. RPI rows per iteration give RPI independent chains. Insertion and
// windows as dot4b (240 rows per key window, the two best carried with f = 255). Same result as knn2_u8, bit for bit.
typedef const __attribute__((address_space(4))) unsigned int ConstWord;
template<int RPI>
__global__ void __launch_bounds__(256)
knn2_dot4s(const unsigned int* __restrict__ db, const unsigned int* __restrict__ dbNorm, int rows,
           const unsigned int* __restrict__ q, const unsigned int* __restrict__ qNorm, int nbQuery,
           int* __restrict__ outIdx, float* __restrict__ outDist)
{
    constexpr int W = 32;
    constexpr int kEmpty = INT_MIN + 255;
    ConstWord* cdb = (ConstWord*)(uintptr_t)db;
    ConstWord* cnorm = (ConstWord*)(uintptr_t)dbNorm;
    const int qi = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int qreg[W];
    for (int k = 0; k < W; ++k) qreg[k] = qi < nbQuery ? q[size_t(qi) * W + k] : 0u;
    int p0 = kEmpty, p1 = kEmpty, i0 = INT_MAX, i1 = INT_MAX;

    for (int base = 0; base < rows; base += kDot4bWinRows) {
        const int n = min(kDot4bWinRows, rows - base);
        for (int r = 0; r < n; r += RPI) {
            unsigned int acc[RPI];
            int rowOf[RPI];
#pragma unroll
            for (int j = 0; j < RPI; ++j) { acc[j] = 0u; rowOf[j] = min(r + j, n - 1); }   // past the end: a real row, ignored
#pragma unroll
            for (int k = 0; k < W; ++k)
#pragma unroll
                for (int j = 0; j < RPI; ++j)
                    acc[j] = __builtin_amdgcn_udot4(qreg[k], cdb[size_t(base + rowOf[j]) * W + k], acc[j], false);
#pragma unroll
            for (int j = 0; j < RPI; ++j) {
                const unsigned int c = (254u - unsigned(rowOf[j])) - (cnorm[base + rowOf[j]] << 8);
                const int y = r + j < n ? int((acc[j] << 9) + c) : INT_MIN;
                p1 = max(min(p0, y), p1);
                p0 = max(p0, y);
            }
        }
        const int top = base + 254;
        const int f0 = p0 & 255, f1 = p1 & 255;
        const bool c0 = f0 == 255, c1 = f1 == 255;
        const int n1 = c1 ? (c0 ? i1 : i0) : top - f1;
        if (!c0) i0 = top - f0;
        i1 = n1;
        p0 |= 255; p1 |= 255;
    }
    if (qi < nbQuery) {
        const int qn = int(qNorm[qi]);
        outIdx[2 * qi] = i0 == INT_MAX ? -1 : i0; outIdx[2 * qi + 1] = i1 == INT_MAX ? -1 : i1;
        outDist[2 * qi] = i0 == INT_MAX ? 3.4e38f : float(qn - (p0 >> 8));
        outDist[2 * qi + 1] = i1 == INT_MAX ? 3.4e38f : float(qn - (p1 >> 8));
    }
}

// dot4q: Q queries per thread sharing every LDS read of the database row (Q dot4 per word read). The CUDA harness's best
// on the GTX 1080 Ti (Q = 2, 128 threads: 2.03x knn2_u8). Insertion and windows as dot4b, per query.
template<int Q, int BLOCK>
__global__ void __launch_bounds__(BLOCK)
knn2_dot4q(const unsigned int* __restrict__ db, const unsigned int* __restrict__ dbNorm, int rows,
           const unsigned int* __restrict__ q, const unsigned int* __restrict__ qNorm, int nbQuery,
           int* __restrict__ outIdx, float* __restrict__ outDist)
{
    constexpr int W = 32;
    constexpr int kEmpty = INT_MIN + 255;
    __shared__ __align__(16) unsigned int tile[kDot4bWinRows * W];
    __shared__ int tileC[kDot4bWinRows];
    const int q0 = (blockIdx.x * blockDim.x + threadIdx.x) * Q;
    unsigned int qreg[Q][W];
    int p0[Q], p1[Q], i0[Q], i1[Q];
#pragma unroll
    for (int j = 0; j < Q; ++j) {
        const int qi = q0 + j;
#pragma unroll
        for (int k = 0; k < W; ++k) qreg[j][k] = qi < nbQuery ? q[size_t(qi) * W + k] : 0u;
        p0[j] = kEmpty; p1[j] = kEmpty; i0[j] = INT_MAX; i1[j] = INT_MAX;
    }
    for (int base = 0; base < rows; base += kDot4bWinRows) {
        const int n = min(kDot4bWinRows, rows - base);
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
                    acc[j] = __builtin_amdgcn_udot4(qreg[j][4 * c], w.x, acc[j], false);
                    acc[j] = __builtin_amdgcn_udot4(qreg[j][4 * c + 1], w.y, acc[j], false);
                    acc[j] = __builtin_amdgcn_udot4(qreg[j][4 * c + 2], w.z, acc[j], false);
                    acc[j] = __builtin_amdgcn_udot4(qreg[j][4 * c + 3], w.w, acc[j], false);
                }
            }
            const unsigned int cr = unsigned(tileC[r]);
#pragma unroll
            for (int j = 0; j < Q; ++j) {
                const int y = int((acc[j] << 9) + cr);
                p1[j] = max(min(p0[j], y), p1[j]);
                p0[j] = max(p0[j], y);
            }
        }
        const int top = base + 254;
#pragma unroll
        for (int j = 0; j < Q; ++j) {
            const int f0 = p0[j] & 255, f1 = p1[j] & 255;
            const bool c0 = f0 == 255, c1 = f1 == 255;
            const int n1 = c1 ? (c0 ? i1[j] : i0[j]) : top - f1;
            if (!c0) i0[j] = top - f0;
            i1[j] = n1;
            p0[j] |= 255; p1[j] |= 255;
        }
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

// ---------------------------------------------------------------- harness
static std::vector<uint8_t> loadDesc(const std::string& path, int& count)
{
    std::ifstream f(path, std::ios::binary);
    uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    std::vector<uint8_t> d(size_t(n) * 128);
    f.read(reinterpret_cast<char*>(d.data()), std::streamsize(d.size()));
    count = int(n);
    return d;
}

#if !defined(WMMA_NO_MAIN)
int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: wmma_matcher <folder with *.desc> [pairs] [reps]\n"); return 2; }
    const int pairs = argc > 2 ? std::atoi(argv[2]) : 8, reps = argc > 3 ? std::atoi(argv[3]) : 5;
    const int MT = argc > 4 ? std::atoi(argv[4]) : 1;
    const int VER = argc > 5 ? std::atoi(argv[5]) : 1;
    const int WV = argc > 6 ? std::atoi(argv[6]) : kWaves;
    std::vector<std::string> files;
    for (auto& e : std::filesystem::directory_iterator(argv[1]))
        if (e.path().extension() == ".desc") files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    hipDeviceProp_t p{}; CHECK(hipGetDeviceProperties(&p, 0));
    std::printf("%s, %zu descriptor files, %d pairs, %d reps, %d query tiles per wave, kernel v%d, %d waves per block\n", p.name, files.size(), pairs, reps, MT, VER + 2, WV);

    unsigned int *dDb = nullptr, *dQ = nullptr, *dDbN = nullptr, *dQN = nullptr;
    int *dI1 = nullptr, *dI2 = nullptr; float *dD1 = nullptr, *dD2 = nullptr;
    size_t cap = 0;
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    double tRef = 0, tW = 0; long long queries = 0, same = 0, diffIdx = 0, diffDist = 0; double macs = 0;
    for (int pi = 0; pi < pairs && pi + 1 < int(files.size()); ++pi) {
        int nq = 0, nr = 0;
        const auto Q = loadDesc(files[pi], nq), D = loadDesc(files[pi + 1], nr);
        const size_t need = std::max(Q.size(), D.size());
        if (need > cap) {
            if (cap) { hipFree(dDb); hipFree(dQ); hipFree(dDbN); hipFree(dQN); hipFree(dI1); hipFree(dI2); hipFree(dD1); hipFree(dD2); }
            const size_t rowsMax = need / 128;
            CHECK(hipMalloc(&dDb, need)); CHECK(hipMalloc(&dQ, need));
            CHECK(hipMalloc(&dDbN, rowsMax * 4)); CHECK(hipMalloc(&dQN, rowsMax * 4));
            CHECK(hipMalloc(&dI1, rowsMax * 8)); CHECK(hipMalloc(&dI2, rowsMax * 8));
            CHECK(hipMalloc(&dD1, rowsMax * 8)); CHECK(hipMalloc(&dD2, rowsMax * 8));
            cap = need;
        }
        CHECK(hipMemcpy(dQ, Q.data(), Q.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dDb, D.data(), D.size(), hipMemcpyHostToDevice));
        rowNormsU8<128><<<(nr + 255) / 256, 256>>>(dDb, nr, dDbN);
        rowNormsU8<128><<<(nq + 255) / 256, 256>>>(dQ, nq, dQN);
        CHECK(hipDeviceSynchronize());
        for (int r = 0; r < reps; ++r) {
            float ms = 0;
            CHECK(hipEventRecord(e0));
            knn2_u8<128><<<(nq + kQueriesPerBlock - 1) / kQueriesPerBlock, kQueriesPerBlock>>>(dDb, dDbN, nr, dQ, nq, dI1, dD1);
            CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); CHECK(hipEventElapsedTime(&ms, e0, e1));
            if (r) tRef += ms;   // the first rep warms up
            CHECK(hipEventRecord(e0));
            if (VER == 7) {
                if (MT == 64)       knn2_dot4q<2, 64><<<(nq + 127) / 128, 64>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                else if (MT == 128) knn2_dot4q<2, 128><<<(nq + 255) / 256, 128>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                else                knn2_dot4q<2, 256><<<(nq + 511) / 512, 256>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
            } else if (VER == 6) {
                const int g = (nq + 255) / 256;
                if (MT == 1)      knn2_dot4s<1><<<g, 256>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                else if (MT == 2) knn2_dot4s<2><<<g, 256>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                else              knn2_dot4s<4><<<g, 256>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
            } else if (VER == 5) {
                const int g = (nq + 255) / 256;
                if (MT == 1)      knn2_dot4b<1><<<g, 256>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                else if (MT == 2) knn2_dot4b<2><<<g, 256>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                else              knn2_dot4b<4><<<g, 256>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
            } else if (VER == 3) {
                if (MT == 1) launchC<1, false, kWaves>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                else         launchC<2, false, kWaves>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
            } else if (VER == 4) {
                if (WV == 4) {
                    if (MT == 1)      launchC<1, true, 4>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                    else if (MT == 2) launchC<2, true, 4>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                    else if (MT == 3) launchC<3, true, 4>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                    else              launchC<4, true, 4>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                } else {
                    if (MT == 1)      launchC<1, true, 8>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                    else if (MT == 2) launchC<2, true, 8>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                    else if (MT == 3) launchC<3, true, 8>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                    else              launchC<4, true, 8>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                }
            } else if (VER == 2) {
                if (MT == 1) knn2_wmma128s<1><<<(nq + kWaves * 16 - 1) / (kWaves * 16), kWaves * 32>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                else         knn2_wmma128s<2><<<(nq + kWaves * 32 - 1) / (kWaves * 32), kWaves * 32>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
            } else if (VER == 1) {
                if (MT == 1) knn2_wmma128p<1><<<(nq + kWaves * 16 - 1) / (kWaves * 16), kWaves * 32>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
                else         knn2_wmma128p<2><<<(nq + kWaves * 32 - 1) / (kWaves * 32), kWaves * 32>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
            } else if (MT == 1) knn2_wmma128<1><<<(nq + kWaves * 16 - 1) / (kWaves * 16), kWaves * 32>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
            else if (MT == 2) knn2_wmma128<2><<<(nq + kWaves * 32 - 1) / (kWaves * 32), kWaves * 32>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
            else if (MT == 3) knn2_wmma128<3><<<(nq + kWaves * 48 - 1) / (kWaves * 48), kWaves * 32>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
            else              knn2_wmma128<4><<<(nq + kWaves * 64 - 1) / (kWaves * 64), kWaves * 32>>>(dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
            CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); CHECK(hipEventElapsedTime(&ms, e0, e1));
            if (r) tW += ms;
        }
        CHECK(hipGetLastError());
        std::vector<int> i1(size_t(nq) * 2), i2(size_t(nq) * 2);
        std::vector<float> d1(size_t(nq) * 2), d2(size_t(nq) * 2);
        CHECK(hipMemcpy(i1.data(), dI1, i1.size() * 4, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(i2.data(), dI2, i2.size() * 4, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(d1.data(), dD1, d1.size() * 4, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(d2.data(), dD2, d2.size() * 4, hipMemcpyDeviceToHost));
        long long s = 0;
        for (int k = 0; k < nq; ++k) {
            const bool idxSame = i1[2 * k] == i2[2 * k] && i1[2 * k + 1] == i2[2 * k + 1];
            const bool distSame = std::memcmp(&d1[2 * k], &d2[2 * k], 8) == 0;
            s += idxSame && distSame; diffIdx += !idxSame; diffDist += !distSame;
        }
        same += s; queries += nq; macs += double(nq) * nr * 128 * (reps - 1);
        std::printf("pair %d: %d x %d, identical %lld of %d\n", pi, nq, nr, s, nq);
    }
    std::printf("identical %lld of %lld queries (nearest/second rows differ %lld, distances differ %lld)\n", same, queries, diffIdx, diffDist);
    std::printf("knn2_u8  %.1f ms (%.2f TMAC/s)\nknn2_wmma %.1f ms (%.2f TMAC/s), %.2fx\n", tRef, macs / tRef * 1e-9, tW, macs / tW * 1e-9, tRef / tW);
    return same == queries ? 0 : 1;
}
#endif
