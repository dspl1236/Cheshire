// CUDA counterpart of wmma_matcher.hip, for cards without matrix units (house-pc's GTX 1080 Ti, sm_61): the shipped
// knn2_u8 (copied from hip/port/gpu_matcher/gpuMatcher.cu, its dp4a path) against candidate dp4a kernels on the same
// descriptor pairs, every query compared exactly (rows and distances); and a dp4a throughput probe.
//   cuda_matcher <folder with *.desc> [pairs=8] [reps=5] [kernel=1] [param=1]
//   cuda_matcher --peak [iterations=2000]
// Build (WSL): nvcc -O3 -arch=sm_61 -std=c++17 cuda_matcher.cu -o cuda_matcher
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

// ---------------------------------------------------------------- the shipped kernel, verbatim (CUDA path)
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
    return __dp4a(a, b, acc);
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

// ---------------------------------------------------------------- candidates
// dot4b: one query per thread, RPI rows per iteration (RPI independent dp4a chains reusing the query registers),
// v6's insertion: keys 256 (2 dot - rowNorm) + f over 240-row windows, a max and a min-max per candidate, the two best
// carried from window to window with f = 255. Same result as knn2_u8, bit for bit (integer sums in any grouping).
constexpr int kWinRows = 240;
template<int RPI>
__global__ void __launch_bounds__(256)
knn2_dot4b(const unsigned int* __restrict__ db, const unsigned int* __restrict__ dbNorm, int rows,
           const unsigned int* __restrict__ q, const unsigned int* __restrict__ qNorm, int nbQuery,
           int* __restrict__ outIdx, float* __restrict__ outDist)
{
    constexpr int W = 32;
    constexpr int kEmpty = INT_MIN + 255;
    __shared__ __align__(16) unsigned int tile[kWinRows * W];
    __shared__ int tileC[kWinRows];
    const int qi = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int qreg[W];
    for (int k = 0; k < W; ++k) qreg[k] = qi < nbQuery ? q[size_t(qi) * W + k] : 0u;
    int p0 = kEmpty, p1 = kEmpty, i0 = INT_MAX, i1 = INT_MAX;
    for (int base = 0; base < rows; base += kWinRows) {
        const int n = min(kWinRows, rows - base);
        const int nPad = (n + RPI - 1) / RPI * RPI;
        for (int t = threadIdx.x; t < nPad * W; t += blockDim.x) tile[t] = t / W < n ? db[size_t(base) * W + t] : 0u;
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
                    acc[j] = __dp4a(qreg[4 * c], w[j].x, acc[j]);
                    acc[j] = __dp4a(qreg[4 * c + 1], w[j].y, acc[j]);
                    acc[j] = __dp4a(qreg[4 * c + 2], w[j].z, acc[j]);
                    acc[j] = __dp4a(qreg[4 * c + 3], w[j].w, acc[j]);
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

// dot4q: Q queries per thread. On Pascal a warp's 16-byte broadcast read of shared memory costs about as much as the
// 32 dp4a it feeds for one query, so the shipped kernel and dot4b are bound by those reads; here every word read feeds Q
// dp4a, one per query (Q independent chains). Insertion and windows as dot4b, per query. Same result as knn2_u8.
template<int Q, int BLOCK = 256>
__global__ void __launch_bounds__(BLOCK)
knn2_dot4q(const unsigned int* __restrict__ db, const unsigned int* __restrict__ dbNorm, int rows,
           const unsigned int* __restrict__ q, const unsigned int* __restrict__ qNorm, int nbQuery,
           int* __restrict__ outIdx, float* __restrict__ outDist)
{
    constexpr int W = 32;
    constexpr int kEmpty = INT_MIN + 255;
    __shared__ __align__(16) unsigned int tile[kWinRows * W];
    __shared__ int tileC[kWinRows];
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
    for (int base = 0; base < rows; base += kWinRows) {
        const int n = min(kWinRows, rows - base);
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
                    acc[j] = __dp4a(qreg[j][4 * c], w.x, acc[j]);
                    acc[j] = __dp4a(qreg[j][4 * c + 1], w.y, acc[j]);
                    acc[j] = __dp4a(qreg[j][4 * c + 2], w.z, acc[j]);
                    acc[j] = __dp4a(qreg[j][4 * c + 3], w.w, acc[j]);
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

// ---------------------------------------------------------------- dp4a throughput probe
template<int CHAINS>
__global__ void __launch_bounds__(256) dp4aChains(int iters, unsigned int* out)
{
    unsigned int a[CHAINS], acc[CHAINS], b[32];
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) { a[c] = (threadIdx.x + 7u * c) * 0x9E3779B1u; acc[c] = 0u; }
#pragma unroll
    for (int k = 0; k < 32; ++k) b[k] = (threadIdx.x ^ (k * 0x01010101u)) * 0x85EBCA6Bu;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int k = 0; k < 32; ++k)
#pragma unroll
            for (int c = 0; c < CHAINS; ++c) acc[c] = __dp4a(a[c], b[k], acc[c]);
#pragma unroll
        for (int c = 0; c < CHAINS; ++c) a[c] += 1u;
    }
    unsigned int s = 0u;
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) s ^= acc[c];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template<int CHAINS>
void peakRun(int iters, unsigned int* dOut, cudaEvent_t e0, cudaEvent_t e1)
{
    const int blocks = 4096;
    dp4aChains<CHAINS><<<blocks, 256>>>(iters / 10, dOut);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(e0));
    dp4aChains<CHAINS><<<blocks, 256>>>(iters, dOut);
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, e0, e1));
    const double macs = double(blocks) * 256 * iters * 32.0 * CHAINS * 4.0;
    std::printf("chains %d: %.1f ms, %.2f TMAC/s\n", CHAINS, ms, macs / ms * 1e-9);
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

static void launch(int kernel, int param, const unsigned int* dDb, const unsigned int* dDbN, int nr, const unsigned int* dQ,
                   const unsigned int* dQN, int nq, int* dI, float* dD)
{
    const int g = (nq + 255) / 256;
    if (kernel == 1) {
        if (param == 1)      knn2_dot4b<1><<<g, 256>>>(dDb, dDbN, nr, dQ, dQN, nq, dI, dD);
        else if (param == 2) knn2_dot4b<2><<<g, 256>>>(dDb, dDbN, nr, dQ, dQN, nq, dI, dD);
        else                 knn2_dot4b<4><<<g, 256>>>(dDb, dDbN, nr, dQ, dQN, nq, dI, dD);
    } else if (kernel == 2) {
        if (param == 2) knn2_dot4q<2><<<(nq + 511) / 512, 256>>>(dDb, dDbN, nr, dQ, dQN, nq, dI, dD);
        else            knn2_dot4q<4><<<(nq + 1023) / 1024, 256>>>(dDb, dDbN, nr, dQ, dQN, nq, dI, dD);
    } else if (kernel == 3) {   // dot4q Q=2 with smaller blocks: param = threads per block
        if (param == 64)       knn2_dot4q<2, 64><<<(nq + 127) / 128, 64>>>(dDb, dDbN, nr, dQ, dQN, nq, dI, dD);
        else if (param == 128) knn2_dot4q<2, 128><<<(nq + 255) / 256, 128>>>(dDb, dDbN, nr, dQ, dQN, nq, dI, dD);
        else                   knn2_dot4q<2, 256><<<(nq + 511) / 512, 256>>>(dDb, dDbN, nr, dQ, dQN, nq, dI, dD);
    } else if (kernel == 4) {   // dot4q Q=1 (one query per thread, no row reuse) with smaller blocks
        if (param == 64)       knn2_dot4q<1, 64><<<(nq + 63) / 64, 64>>>(dDb, dDbN, nr, dQ, dQN, nq, dI, dD);
        else if (param == 128) knn2_dot4q<1, 128><<<(nq + 127) / 128, 128>>>(dDb, dDbN, nr, dQ, dQN, nq, dI, dD);
        else                   knn2_dot4q<1, 256><<<(nq + 255) / 256, 256>>>(dDb, dDbN, nr, dQ, dQN, nq, dI, dD);
    }
}

int main(int argc, char** argv)
{
    cudaDeviceProp p{}; CHECK(cudaGetDeviceProperties(&p, 0));
    if (argc > 1 && std::string(argv[1]) == "--peak") {
        const int iters = argc > 2 ? std::atoi(argv[2]) : 2000;
        std::printf("%s (sm_%d%d), %d SMs, %d MHz\n", p.name, p.major, p.minor, p.multiProcessorCount, p.clockRate / 1000);
        unsigned int* dOut = nullptr; CHECK(cudaMalloc(&dOut, 4096 * 256 * 4));
        cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
        peakRun<1>(iters, dOut, e0, e1); peakRun<2>(iters, dOut, e0, e1); peakRun<4>(iters, dOut, e0, e1); peakRun<8>(iters, dOut, e0, e1);
        return 0;
    }
    if (argc < 2) { std::printf("usage: cuda_matcher <folder with *.desc> [pairs] [reps] [kernel] [param] | --peak\n"); return 2; }
    const int pairs = argc > 2 ? std::atoi(argv[2]) : 8, reps = argc > 3 ? std::atoi(argv[3]) : 5;
    const int kernel = argc > 4 ? std::atoi(argv[4]) : 1, param = argc > 5 ? std::atoi(argv[5]) : 1;
    std::vector<std::string> files;
    for (auto& e : std::filesystem::directory_iterator(argv[1]))
        if (e.path().extension() == ".desc") files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    std::printf("%s, %zu descriptor files, %d pairs, %d reps, kernel %d, param %d\n", p.name, files.size(), pairs, reps, kernel, param);
    unsigned int *dDb = nullptr, *dQ = nullptr, *dDbN = nullptr, *dQN = nullptr;
    int *dI1 = nullptr, *dI2 = nullptr; float *dD1 = nullptr, *dD2 = nullptr;
    size_t cap = 0;
    cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    double tRef = 0, tNew = 0, macs = 0; long long queries = 0, same = 0;
    for (int pi = 0; pi < pairs && pi + 1 < int(files.size()); ++pi) {
        int nq = 0, nr = 0;
        const auto Q = loadDesc(files[pi], nq), D = loadDesc(files[pi + 1], nr);
        const size_t need = std::max(Q.size(), D.size());
        if (need > cap) {
            if (cap) { cudaFree(dDb); cudaFree(dQ); cudaFree(dDbN); cudaFree(dQN); cudaFree(dI1); cudaFree(dI2); cudaFree(dD1); cudaFree(dD2); }
            const size_t rowsMax = need / 128;
            CHECK(cudaMalloc(&dDb, need)); CHECK(cudaMalloc(&dQ, need));
            CHECK(cudaMalloc(&dDbN, rowsMax * 4)); CHECK(cudaMalloc(&dQN, rowsMax * 4));
            CHECK(cudaMalloc(&dI1, rowsMax * 8)); CHECK(cudaMalloc(&dI2, rowsMax * 8));
            CHECK(cudaMalloc(&dD1, rowsMax * 8)); CHECK(cudaMalloc(&dD2, rowsMax * 8));
            cap = need;
        }
        CHECK(cudaMemcpy(dQ, Q.data(), Q.size(), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dDb, D.data(), D.size(), cudaMemcpyHostToDevice));
        rowNormsU8<128><<<(nr + 255) / 256, 256>>>(dDb, nr, dDbN);
        rowNormsU8<128><<<(nq + 255) / 256, 256>>>(dQ, nq, dQN);
        CHECK(cudaDeviceSynchronize());
        for (int r = 0; r < reps; ++r) {
            float ms = 0;
            CHECK(cudaEventRecord(e0));
            knn2_u8<128><<<(nq + kQueriesPerBlock - 1) / kQueriesPerBlock, kQueriesPerBlock>>>(dDb, dDbN, nr, dQ, nq, dI1, dD1);
            CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1)); CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (r) tRef += ms;
            CHECK(cudaEventRecord(e0));
            launch(kernel, param, dDb, dDbN, nr, dQ, dQN, nq, dI2, dD2);
            CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1)); CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (r) tNew += ms;
        }
        CHECK(cudaGetLastError());
        std::vector<int> i1(size_t(nq) * 2), i2(size_t(nq) * 2);
        std::vector<float> d1(size_t(nq) * 2), d2(size_t(nq) * 2);
        CHECK(cudaMemcpy(i1.data(), dI1, i1.size() * 4, cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(i2.data(), dI2, i2.size() * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(d1.data(), dD1, d1.size() * 4, cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(d2.data(), dD2, d2.size() * 4, cudaMemcpyDeviceToHost));
        long long s = 0;
        for (int k = 0; k < nq; ++k)
            s += i1[2 * k] == i2[2 * k] && i1[2 * k + 1] == i2[2 * k + 1] && std::memcmp(&d1[2 * k], &d2[2 * k], 8) == 0;
        same += s; queries += nq; macs += double(nq) * nr * 128 * (reps - 1);
    }
    std::printf("identical %lld of %lld queries\n", same, queries);
    std::printf("knn2_u8 %.1f ms (%.2f TMAC/s); candidate %.1f ms (%.2f TMAC/s), %.2fx\n", tRef, macs / tRef * 1e-9, tNew,
                macs / tNew * 1e-9, tRef / tNew);
    return same == queries ? 0 : 1;
}
