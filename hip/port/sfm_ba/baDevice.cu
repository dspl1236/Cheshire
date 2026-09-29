// This file is part of the Cheshire patch set for AliceVision (https://github.com/dspl1236/Cheshire).
// Copied by scripts/apply_hip_patch.py (step 7f) to src/aliceVision/sfm/bundle/gpu/.
//
// 0.3.7, step 4c: the bundle-adjustment solver's work per row and per landmark on the device (baDevice.hpp).
// The host solver's result byte for byte:
//   - every value is computed by the host's arithmetic: baArith.hpp's functions, and the few loops of
//     ownSolver.hpp restated below with the same operations in the same order;
//   - no contraction: the pragma below, before any of this file's code (HIP), and --fmad=false (CUDA, the
//     build). Only plain operators, no math intrinsics but sqrt and division, which round correctly;
//   - every sum in the host's order: per row, per landmark (chunk), per (partition, camera block), per
//     (group, block of the reduced system, row of the block), each by one thread in order, and the
//     partials over partitions and groups summed in order.
// CUDA dialect; the HIP build force-includes cheshire/cuda_to_hip.h.
#ifdef __clang__
#pragma clang fp contract(off)
#endif

#include "baDevice.hpp"

#include <aliceVision/depthMap/cuda/hip/cheshire/env.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace aliceVision {
namespace sfm {
namespace cheshire {
namespace device {

using arith::CameraValues;
using arith::Corrector;
using arith::PoseRotation;

namespace {

constexpr int kMaxT = 8;   // the largest camera block tangent size the device takes (poses 6)

bool g_checked = false, g_available = false;
std::string g_name;
std::mutex g_mutex;

// The device memory kept across solves: the k-th allocation of a solve reuses the k-th buffer when it is large
// enough, else the buffer is replaced by one a quarter larger than asked. One problem at a time holds it (another
// alive at the same time allocates for itself). The buffers stay until the process ends.
struct Pool
{
    std::mutex m;
    bool busy = false;
    std::vector<std::pair<void*, size_t>> slots;
};

Pool& pool()
{
    static Pool* p = new Pool();   // never destroyed: the runtime may be gone at exit
    return *p;
}

bool poolEnabled()
{
    static const bool on = ::cheshire::env::flag("CHESHIRE_BA_DEVICE_POOL", true);
    return on;
}

// ---- the kernels ------------------------------------------------------------------------------------------

struct DeviceInner
{
    const CameraValues* cam;
    double ox, oy, os;
    __device__ bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
    {
        return arith::fusedProjectValues(*cam, ox, oy, os, parameters, residuals, jacobians);
    }
};

struct EvalArgs
{
    const Row* rows;
    int nRows;
    int mode;                 // Mode
    const double* x;          // the free landmarks, 3 per E block
    const double* constLm;
    const PoseEntry* poses;
    const CameraValues* cams;
    const Camera* camInfo;
    const double* plusJ;
    const std::int64_t* manifOff;
    const std::int32_t* manifSize;
    const std::int32_t* manifTsize;
    const std::int32_t* fTsize;
    double* J;                // the row's Jacobian goes here (Full: the kept one; Gradient: the temporary one)
    double* r;                // likewise the residuals
    double* rowCost;
    char* rowOk;
    int loss;
    double hubA, hubB;
};

// rowRho (ownSolver.hpp) with the Huber loss: false when the row has no loss at all
__device__ bool rowRho(int loss, double a, double b, double scale, double s, double rho[3])
{
    if (scale == 1.0)
    {
        if (!loss)
            return false;
        arith::huber(a, b, s, rho);
        return true;
    }
    if (!loss)
    {
        rho[0] = scale * s;
        rho[1] = scale;
        rho[2] = 0.0;
        return true;
    }
    arith::huber(a, b, s, rho);
    rho[0] *= scale;
    rho[1] *= scale;
    rho[2] *= scale;
    return true;
}

// Solver::evalRow, one thread per row
__global__ void kEval(EvalArgs A)
{
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= A.nRows)
        return;
    const Row& row = A.rows[i];
    const bool wantJ = A.mode != int(Mode::Cost);
    const double* X = row.lm >= 0 ? A.x + 3 * std::int64_t(row.lm) : A.constLm + 3 * std::int64_t(-1 - row.lm);
    const PoseEntry& pe = A.poses[row.pose];
    const double pose[6] = {pe.rot.aa[0], pe.rot.aa[1], pe.rot.aa[2], pe.center[0], pe.center[1], pe.center[2]};
    const double unused[1] = {0.0};   // the fused projection reads the intrinsics and distortion from the camera values
    const double* params[4] = {unused, unused, pose, X};
    double r[2];
    double amb[4][2 * kMaxT];
    double* Jrow = wantJ ? A.J + row.j : nullptr;
    double* jac[4] = {nullptr, nullptr, nullptr, nullptr};
    int tsize[4] = {0, 0, 0, 3};
    for (int k = 0; k < 4; ++k)
    {
        if (row.pj[k] < 0)
            continue;
        if (k < 3)
            tsize[k] = A.fTsize[row.blk[k]];
        if (wantJ)
            jac[k] = row.manif[k] >= 0 ? amb[k] : Jrow + row.pj[k];
    }
    const Camera& ci = A.camInfo[row.camera];
    const DeviceInner inner{&A.cams[row.camera], row.ox, row.oy, row.os};
    if (!arith::projectSimpleRot(inner, pe.rot, ci.hasDistortion != 0, ci.nd, params, r, wantJ ? jac : nullptr))
    {
        A.rowOk[i] = 0;
        return;
    }
    double s = 0.0;
    for (int q = 0; q < 2; ++q)
    {
        if (!arith::baFinite(r[q]))
        {
            A.rowOk[i] = 0;
            return;
        }
        s += r[q] * r[q];
    }
    if (wantJ)
        for (int k = 0; k < 4; ++k)
        {
            if (row.pj[k] < 0)
                continue;
            double* J = Jrow + row.pj[k];
            const int t = tsize[k];
            if (row.manif[k] >= 0)
            {
                const int m = row.manif[k];
                const int size = A.manifSize[m];
                const double* Ja = amb[k];
                const double* PJ = A.plusJ + A.manifOff[m];
                for (int q = 0; q < 2; ++q)
                    for (int tt = 0; tt < t; ++tt)
                    {
                        double acc = 0.0;
                        for (int a = 0; a < size; ++a)
                            acc += Ja[q * size + a] * PJ[a * t + tt];
                        J[q * t + tt] = acc;
                    }
            }
            for (int q = 0; q < 2 * t; ++q)
                if (!arith::baFinite(J[q]))
                {
                    A.rowOk[i] = 0;
                    return;
                }
        }
    double rho[3];
    double cost;
    if (!rowRho(A.loss, A.hubA, A.hubB, row.weight, s, rho))
        cost = 0.5 * s;
    else
    {
        cost = 0.5 * rho[0];
        if (wantJ)
        {
            const Corrector c(s, rho);
            for (int k = 0; k < 4; ++k)
                if (row.pj[k] >= 0)
                    c.correctJacobian(2, tsize[k], r, Jrow + row.pj[k]);
            c.correctResiduals(2, r);
        }
    }
    if (wantJ)
    {
        A.r[2 * std::int64_t(i)] = r[0];
        A.r[2 * std::int64_t(i) + 1] = r[1];
    }
    A.rowCost[i] = cost;
    A.rowOk[i] = 1;
}

// per partition: the rows' costs summed in order, and whether every row evaluated (the host stops at the
// first failure; the cost is then unused)
__global__ void kPartCost(const std::int32_t* parts, int P, const double* rowCost, const char* rowOk, double* partCost, char* partOk)
{
    const int p = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= P)
        return;
    // the host stops at a failed row and the cost is then unused: the flags and the sum apart, so the loads of
    // the sum do not wait on the flags
    char ok = 1;
    for (int ri = parts[p]; ri < parts[p + 1]; ++ri)
        ok &= rowOk[ri];
    // eight loads in flight, then their adds in order
    double total = 0.0;
    int ri = parts[p];
    const int end = parts[p + 1];
    for (; ri + 8 <= end; ri += 8)
    {
        double x[8];
        for (int u = 0; u < 8; ++u)
            x[u] = rowCost[ri + u];
        for (int u = 0; u < 8; ++u)
            total += x[u];
    }
    for (; ri < end; ++ri)
        total += rowCost[ri];
    partCost[p] = total;
    partOk[p] = ok;
}

// the E part of J^T r (norms == 0) or of the squared column norms (norms == 1), one thread per landmark
__global__ void kColumnsE(const Row* rows, const std::int32_t* chunkStart, int nE, const double* J, const double* r, int norms, double* out)
{
    const int c = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= nE)
        return;
    double v[3] = {0.0, 0.0, 0.0};
    for (int ri = chunkStart[c]; ri < chunkStart[c + 1]; ++ri)
    {
        const Row& row = rows[ri];
        const double* E = J + row.j + row.pj[3];
        if (norms)
        {
            for (int q = 0; q < 2; ++q)
                for (int t = 0; t < 3; ++t)
                    v[t] += E[q * 3 + t] * E[q * 3 + t];
        }
        else
        {
            const double* b = r + 2 * std::int64_t(ri);
            for (int t = 0; t < 3; ++t)
            {
                double acc = 0.0;
                for (int q = 0; q < 2; ++q)
                    acc += E[q * 3 + t] * b[q];
                v[t] += acc;
            }
        }
    }
    for (int t = 0; t < 3; ++t)
        out[3 * std::int64_t(c) + t] = v[t];
}

// the F part per (partition, camera block, column): one thread each, over the partition's rows on the block in
// order, eight rows' loads in flight; into the partition's accumulator (P x nf, zero elsewhere)
__global__ void kColumnsF(const std::int64_t* pairStart, const std::int32_t* pairPart, const std::int32_t* pairF, const std::int32_t* listRow,
                          const std::int64_t* listJ, const std::int64_t* colStart, int nPairs, const std::int32_t* fTsize, const std::int32_t* fCol,
                          int nf, const double* J, const double* r, int norms, double* partF)
{
    const std::int64_t tid = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= colStart[nPairs])
        return;
    int lo = 0, hi = nPairs;   // the pair: the last with colStart <= tid
    while (hi - lo > 1)
    {
        const int mid = (lo + hi) / 2;
        if (colStart[mid] <= tid)
            lo = mid;
        else
            hi = mid;
    }
    const int pr = lo, tt = int(tid - colStart[pr]);
    const int f = pairF[pr], t = fTsize[f];
    double v = 0.0;
    std::int64_t e = pairStart[pr];
    const std::int64_t end = pairStart[pr + 1];
    for (; e + 8 <= end; e += 8)
    {
        double j0[8], j1[8], b0[8], b1[8];
        for (int u = 0; u < 8; ++u)
        {
            const double* F = J + listJ[e + u];
            j0[u] = F[tt];
            j1[u] = F[t + tt];
            if (!norms)
            {
                const double* b = r + 2 * std::int64_t(listRow[e + u]);
                b0[u] = b[0];
                b1[u] = b[1];
            }
        }
        for (int u = 0; u < 8; ++u)
        {
            if (norms)
            {
                v += j0[u] * j0[u];
                v += j1[u] * j1[u];
            }
            else
            {
                double acc = 0.0;
                acc += j0[u] * b0[u];
                acc += j1[u] * b1[u];
                v += acc;
            }
        }
    }
    for (; e < end; ++e)
    {
        const double* F = J + listJ[e];
        if (norms)
        {
            v += F[tt] * F[tt];
            v += F[t + tt] * F[t + tt];
        }
        else
        {
            const double* b = r + 2 * std::int64_t(listRow[e]);
            double acc = 0.0;
            acc += F[tt] * b[0];
            acc += F[t + tt] * b[1];
            v += acc;
        }
    }
    partF[std::int64_t(pairPart[pr]) * nf + fCol[f] + tt] = v;
}

// reduceF: per camera column, the partitions' accumulators summed in partition order
__global__ void kReduceF(const double* partF, int P, int nf, double* dst)
{
    const int c = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= nf)
        return;
    double s = 0.0;
    for (int p = 0; p < P; ++p)
        s += partF[std::int64_t(p) * nf + c];
    dst[c] = s;
}

__device__ inline std::int64_t slotCol(const Row& row, int k, int ne, const std::int32_t* fCol)
{
    return k == 3 ? 3 * std::int64_t(row.blk[3]) : std::int64_t(ne) + fCol[row.blk[k]];
}

// scaleColumns: J's columns times the Jacobi scale
__global__ void kScale(const Row* rows, int nRows, const std::int32_t* fTsize, const std::int32_t* fCol, int ne, const double* scale, double* J)
{
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= nRows)
        return;
    const Row& row = rows[i];
    for (int k = 0; k < 4; ++k)
    {
        if (row.pj[k] < 0)
            continue;
        const int t = k == 3 ? 3 : fTsize[row.blk[k]];
        const double* c = scale + slotCol(row, k, ne, fCol);
        double* Jb = J + row.j + row.pj[k];
        for (int q = 0; q < 2; ++q)
            for (int tt = 0; tt < t; ++tt)
                Jb[q * t + tt] *= c[tt];
    }
}

// modelCostChange's two terms per row: f = J step, f_i (r_i + f_i / 2)
__global__ void kModel(const Row* rows, int nRows, const std::int32_t* fTsize, const std::int32_t* fCol, int ne, const double* step,
                       const double* J, const double* r, double* terms)
{
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= nRows)
        return;
    const Row& row = rows[i];
    double fr[2] = {0.0, 0.0};
    for (int k = 0; k < 4; ++k)
    {
        if (row.pj[k] < 0)
            continue;
        const int t = k == 3 ? 3 : fTsize[row.blk[k]];
        const double* yy = step + slotCol(row, k, ne, fCol);
        const double* Jb = J + row.j + row.pj[k];
        for (int q = 0; q < 2; ++q)
        {
            double acc = 0.0;
            for (int tt = 0; tt < t; ++tt)
                acc += Jb[q * t + tt] * yy[tt];
            fr[q] += acc;
        }
    }
    const double* b = r + 2 * std::int64_t(i);
    terms[2 * std::int64_t(i)] = fr[0] * (b[0] + fr[0] / 2.0);
    terms[2 * std::int64_t(i) + 1] = fr[1] * (b[1] + fr[1] / 2.0);
}

__global__ void kPartDot(const std::int32_t* parts, int P, const double* terms, double* partDot)
{
    const int p = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= P)
        return;
    // the two terms of each row, in order; sixteen loads in flight
    double dot = 0.0;
    std::int64_t q = 2 * std::int64_t(parts[p]);
    const std::int64_t end = 2 * std::int64_t(parts[p + 1]);
    for (; q + 16 <= end; q += 16)
    {
        double x[16];
        for (int u = 0; u < 16; ++u)
            x[u] = terms[q + u];
        for (int u = 0; u < 16; ++u)
            dot += x[u];
    }
    for (; q < end; ++q)
        dot += terms[q];
    partDot[p] = dot;
}

// addAtB (ownSolver.hpp): out (t1 x t2) += A^T B, A 2 x t1, B 2 x t2, per entry the sum over the rows from zero
__device__ inline void addAtB(int t1, int t2, const double* A, const double* B, double* out)
{
    for (int a = 0; a < t1; ++a)
    {
        double acc[kMaxT];
        for (int b = 0; b < t2; ++b)
            acc[b] = 0.0;
        for (int q = 0; q < 2; ++q)
        {
            const double s = A[q * t1 + a];
            for (int b = 0; b < t2; ++b)
                acc[b] += s * B[q * t2 + b];
        }
        for (int b = 0; b < t2; ++b)
            out[a * t2 + b] += acc[b];
    }
}

// the elimination, per landmark: E^T E + D_e^2 and its inverse, E^T b, the E^T F blocks
__global__ void kElimChunk(const Row* rows, const std::int32_t* chunkStart, int nE, const std::int64_t* cfStart, const std::int32_t* cf,
                           const std::int64_t* cfM, const std::int32_t* fTsize, const double* D, const double* J, const double* r,
                           double* m, double* mU, double* inv, double* ge, int* notFinite)
{
    const int c = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= nE)
        return;
    const double* De = D + 3 * std::int64_t(c);
    double ete[9] = {De[0] * De[0], 0, 0, 0, De[1] * De[1], 0, 0, 0, De[2] * De[2]};
    double g[3] = {0, 0, 0};
    const std::int64_t l0 = cfStart[c], l1 = cfStart[c + 1];
    if (l1 > l0)
    {
        const std::int64_t e = cfM[l1 - 1] + 3 * std::int64_t(fTsize[cf[l1 - 1]]);
        for (std::int64_t q = cfM[l0]; q < e; ++q)
            m[q] = 0.0;
    }
    for (int ri = chunkStart[c]; ri < chunkStart[c + 1]; ++ri)
    {
        const Row& row = rows[ri];
        const double* b = r + 2 * std::int64_t(ri);
        const double* E = J + row.j + row.pj[3];
        for (int p = 0; p < 3; ++p)
        {
            for (int q = 0; q < 3; ++q)
            {
                double acc = 0.0;
                for (int i = 0; i < 2; ++i)
                    acc += E[i * 3 + p] * E[i * 3 + q];
                ete[p * 3 + q] += acc;
            }
            double acc = 0.0;
            for (int i = 0; i < 2; ++i)
                acc += E[i * 3 + p] * b[i];
            g[p] += acc;
        }
        for (int k = 0; k < 3; ++k)
        {
            if (row.pj[k] < 0)
                continue;
            const int f = row.blk[k];
            std::int64_t l = l0;
            while (cf[l] != f)
                ++l;
            addAtB(3, fTsize[f], E, J + row.j + row.pj[k], m + cfM[l]);
        }
    }
    double* iv = inv + 9 * std::int64_t(c);
    arith::inverse3(ete, iv);
    for (int p = 0; p < 9; ++p)
        if (!arith::baFinite(iv[p]))
            *notFinite = 1;
    for (int p = 0; p < 3; ++p)
        ge[3 * std::int64_t(c) + p] = g[p];
    // u = m^T inv per camera block (t x 3, beside m), as the host computes it for the block's Schur terms
    for (std::int64_t l = l0; l < l1; ++l)
    {
        const int t = fTsize[cf[l]];
        const double* mm = m + cfM[l];
        double* U = mU + cfM[l];
        for (int a = 0; a < t; ++a)
            for (int q = 0; q < 3; ++q)
                U[a * 3 + q] = mm[0 * t + a] * iv[0 * 3 + q] + mm[1 * t + a] * iv[1 * 3 + q] + mm[2 * t + a] * iv[2 * 3 + q];
    }
}

// ---- the reduced camera system --------------------------------------------------------------------------
// The host adds, per assembly group, landmark after landmark, each row's F^T F terms and then the landmark's
// Schur term into its copy of (S, rhs). So every entry of a copy is a sum in a fixed order: per landmark of the
// group holding both of the entry's camera blocks, its rows on both blocks, then its Schur term; in the last
// group then the rows without a landmark. kAssembleRec follows that order, one thread per entry, through the
// host-built records of its (group, block); u = m^T inv comes from kElimChunk, as the host computes it once per
// landmark and block. The blocks inside one camera's intrinsics and distortion, whose lists are long, go by
// kHeavyEvents / kHeavyFold instead.
struct AsmArgs
{
    const std::int64_t* recStart;     // per (group, block): its records
    const std::int32_t* recChunk;     // the landmark, or -1: a row without one
    const std::int32_t* recU;         // f1's u (t1 x 3) and f2's m (3 x t2), at the landmark's offsets for them
    const std::int32_t* recM;
    const std::int32_t* recRowStart;  // its rows on both blocks
    const std::int32_t* recRowCount;
    const std::int32_t* rowJ1;        // per such row: the F1 and F2 blocks' offsets in J, the row
    const std::int32_t* rowJ2;
    const std::int32_t* rowR;
    const std::int32_t* itemBlock;
    const std::int32_t* itemRow;
    const std::int32_t* itemCol;      // t2: the right-hand side (diagonal blocks)
    const std::int32_t* sRow;
    const std::int32_t* sCol;
    const std::int64_t* sOff;
    const std::int32_t* fTsize;
    const std::int32_t* fCol;
    int nS, nItems, G, ne;
    std::int64_t sValues, stride;
    const double* D;
    const double* J;
    const double* r;
    const double* mU;
    const double* m;
    const double* ge;
    double* sg;
};

__global__ void kAssembleRec(AsmArgs A)
{
    const std::int64_t tid = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= std::int64_t(A.G) * A.nItems)
        return;
    const int g = int(tid / A.nItems);
    const int item = int(tid % A.nItems);
    const int s = A.itemBlock[item], a = A.itemRow[item], col = A.itemCol[item];
    const int f1 = A.sRow[s], f2 = A.sCol[s];
    const int t1 = A.fTsize[f1], t2 = A.fTsize[f2];
    double acc = 0.0;
    if (g == 0 && f1 == f2 && col == a)
    {
        const double d = A.D[A.ne + A.fCol[f1] + a];
        acc = d * d;   // the host puts the diagonal in first (SchurEliminator::Eliminate)
    }
    const std::int64_t e0 = A.recStart[std::int64_t(g) * A.nS + s], e1 = A.recStart[std::int64_t(g) * A.nS + s + 1];
    for (std::int64_t e = e0; e < e1; ++e)
    {
        // the rows on both blocks: rowTerms' addAtB (or F^T b), summed over the residuals from zero, then added
        const int rs = A.recRowStart[e], n = A.recRowCount[e];
        for (int k = 0; k < n; ++k)
        {
            const double* F1 = A.J + A.rowJ1[rs + k];
            double v = 0.0;
            if (col < t2)
            {
                const double* F2 = A.J + A.rowJ2[rs + k];
                for (int q = 0; q < 2; ++q)
                    v += F1[q * t1 + a] * F2[q * t2 + col];
            }
            else
            {
                const double* b = A.r + 2 * std::int64_t(A.rowR[rs + k]);
                for (int q = 0; q < 2; ++q)
                    v += F1[q * t1 + a] * b[q];
            }
            acc += v;
        }
        // the landmark's Schur term: subU3M, and the right-hand side's u g
        const int c = A.recChunk[e];
        if (c >= 0)
        {
            const double* U = A.mU + A.recU[e] + a * 3;
            if (col < t2)
            {
                const double* M = A.m + A.recM[e];
                acc -= U[0] * M[0 * t2 + col] + U[1] * M[1 * t2 + col] + U[2] * M[2 * t2 + col];
            }
            else
            {
                const double* gc = A.ge + 3 * std::int64_t(c);
                acc -= U[0] * gc[0] + U[1] * gc[1] + U[2] * gc[2];
            }
        }
    }
    double* S = A.sg + std::int64_t(g) * A.stride;
    if (col < t2)
        S[A.sOff[s] + std::int64_t(a) * t2 + col] = acc;
    else
        S[A.sValues + A.fCol[f1] + a] = acc;
}

// The blocks inside one camera's intrinsics and distortion collect a term from every row and every landmark
// of the camera: chains of up to a hundred thousand terms per group, too long for one thread walking the
// structure. They are split in two passes that keep the host's order. Per group and camera, its terms
// ("events") in the host's order: per landmark of the camera, its rows on the camera, then the landmark's
// Schur term; in the last group then the rows without a landmark. kHeavyEvents computes every term of every
// such entry in parallel into a scratch array (per entry its terms contiguous; a Schur term negated, as
// x - y is x + (-y) exactly); kHeavyFold adds them in order, one thread per entry.
struct HeavyVal
{
    std::int32_t f1, f2, a, b;   // the block (f1, f2), its row a and column b; a right-hand side value: f1 = f2, b = 0
    std::int64_t dst;            // its place in a group's copy of (S, rhs)
    std::int32_t rhs, diag;      // a right-hand side value; the diagonal of S (group 0 starts at D^2)
};

struct HeavyArgs
{
    const Row* rows;
    const double* J;
    const double* r;
    const double* m;
    const double* inv;
    const double* ge;
    const double* D;
    const std::int64_t* cfStart;
    const std::int32_t* cf;
    const std::int64_t* cfM;
    const std::int32_t* fTsize;
    const std::int32_t* fCol;
    const HeavyVal* vals;
    const std::int32_t* vStart;      // per camera, its values
    const std::int32_t* evKey;       // a row (>= 0) or a landmark's Schur term (-1 - landmark)
    const std::int64_t* evStart;     // per segment (group * nCam + camera), its events
    const std::int32_t* segCam;
    const std::int32_t* segGroup;
    const std::int64_t* scrBase;     // per segment, its first scratch value
    const std::int64_t* foldStart;   // per segment, its first fold thread
    int nSeg;
    int ne;
    std::int64_t stride;
    double* scratch;
    double* sg;
};

__device__ inline int segOf(const std::int64_t* start, int n, std::int64_t x)
{
    int lo = 0, hi = n;   // the last segment with start <= x
    while (hi - lo > 1)
    {
        const int mid = (lo + hi) / 2;
        if (start[mid] <= x)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

__device__ inline const double* rowSlot(const Row& row, const double* J, int f)
{
    for (int k = 0; k < 2; ++k)
        if (row.pj[k] >= 0 && row.blk[k] == f)
            return J + row.j + row.pj[k];
    return nullptr;
}

__device__ inline const double* chunkM(const std::int64_t* cfStart, const std::int32_t* cf, const std::int64_t* cfM, const double* m, int ch, int f)
{
    for (std::int64_t l = cfStart[ch]; l < cfStart[ch + 1]; ++l)
        if (cf[l] == f)
            return m + cfM[l];
    return nullptr;
}

__global__ void kHeavyEvents(HeavyArgs H, std::int64_t e0, std::int64_t e1, std::int64_t scrOrigin, int seg0, int seg1)
{
    const std::int64_t e = e0 + std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= e1)
        return;
    const int seg = seg0 + segOf(H.evStart + seg0, seg1 - seg0, e);
    const int c = H.segCam[seg];
    const std::int64_t k = e - H.evStart[seg], nEv = H.evStart[seg + 1] - H.evStart[seg];
    const HeavyVal* V = H.vals + H.vStart[c];
    const int nv = H.vStart[c + 1] - H.vStart[c];
    double* out = H.scratch + (H.scrBase[seg] - scrOrigin) + k;
    const int key = H.evKey[e];
    if (key >= 0)
    {
        const Row& row = H.rows[key];
        const double* b = H.r + 2 * std::int64_t(key);
        for (int v = 0; v < nv; ++v)
        {
            const HeavyVal& hv = V[v];
            const int t1 = H.fTsize[hv.f1];
            const double* F1 = rowSlot(row, H.J, hv.f1);
            double s = 0.0;
            if (hv.rhs)
            {
                for (int q = 0; q < 2; ++q)
                    s += F1[q * t1 + hv.a] * b[q];
            }
            else
            {
                const int t2 = H.fTsize[hv.f2];
                const double* F2 = rowSlot(row, H.J, hv.f2);
                for (int q = 0; q < 2; ++q)
                    s += F1[q * t1 + hv.a] * F2[q * t2 + hv.b];
            }
            out[std::int64_t(v) * nEv] = s;
        }
    }
    else
    {
        const int ch = -1 - key;
        const double* iv = H.inv + 9 * std::int64_t(ch);
        const double* g = H.ge + 3 * std::int64_t(ch);
        for (int v = 0; v < nv; ++v)
        {
            const HeavyVal& hv = V[v];
            const int t1 = H.fTsize[hv.f1];
            const double* m1 = chunkM(H.cfStart, H.cf, H.cfM, H.m, ch, hv.f1);
            double u[3];
            for (int q = 0; q < 3; ++q)
                u[q] = m1[0 * t1 + hv.a] * iv[0 * 3 + q] + m1[1 * t1 + hv.a] * iv[1 * 3 + q] + m1[2 * t1 + hv.a] * iv[2 * 3 + q];
            double val;
            if (hv.rhs)
                val = u[0] * g[0] + u[1] * g[1] + u[2] * g[2];
            else
            {
                const int t2 = H.fTsize[hv.f2];
                const double* M = chunkM(H.cfStart, H.cf, H.cfM, H.m, ch, hv.f2);
                val = u[0] * M[0 * t2 + hv.b] + u[1] * M[1 * t2 + hv.b] + u[2] * M[2 * t2 + hv.b];
            }
            out[std::int64_t(v) * nEv] = -val;
        }
    }
}

__global__ void kHeavyFold(HeavyArgs H, std::int64_t t0, std::int64_t t1, std::int64_t scrOrigin, int seg0, int seg1)
{
    const std::int64_t tid = t0 + std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= t1)
        return;
    const int seg = seg0 + segOf(H.foldStart + seg0, seg1 - seg0, tid);
    const int c = H.segCam[seg], g = H.segGroup[seg];
    const int v = int(tid - H.foldStart[seg]);
    const HeavyVal& hv = H.vals[H.vStart[c] + v];
    const std::int64_t nEv = H.evStart[seg + 1] - H.evStart[seg];
    double acc = 0.0;
    if (g == 0 && hv.diag)
    {
        const double d = H.D[H.ne + H.fCol[hv.f1] + hv.a];
        acc = d * d;
    }
    const double* src = H.scratch + (H.scrBase[seg] - scrOrigin) + std::int64_t(v) * nEv;
    std::int64_t k = 0;
    for (; k + 4 <= nEv; k += 4)
    {
        const double x0 = src[k], x1 = src[k + 1], x2 = src[k + 2], x3 = src[k + 3];
        acc += x0;
        acc += x1;
        acc += x2;
        acc += x3;
    }
    for (; k < nEv; ++k)
        acc += src[k];
    H.sg[std::int64_t(g) * H.stride + hv.dst] = acc;
}

// the groups' copies summed in group order
__global__ void kSumGroups(const double* sg, int G, std::int64_t stride, double* out)
{
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= stride)
        return;
    double s = 0.0;
    for (int g = 0; g < G; ++g)
        s += sg[std::int64_t(g) * stride + i];
    out[i] = s;
}

// the back-substitution, per landmark: y_e = (E^T E + D_e^2)^-1 (E^T b - sum_f E^T F y_f)
__global__ void kBack(const std::int64_t* cfStart, const std::int32_t* cf, const std::int64_t* cfM, const std::int32_t* fTsize,
                      const std::int32_t* fCol, int nE, const double* m, const double* inv, const double* ge, const double* yf, double* yE,
                      int* notFinite)
{
    const int c = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= nE)
        return;
    const double* g = ge + 3 * std::int64_t(c);
    double q[3] = {g[0], g[1], g[2]};
    for (std::int64_t l = cfStart[c]; l < cfStart[c + 1]; ++l)
    {
        const int f = cf[l], t = fTsize[f];
        const double* mm = m + cfM[l];
        const double* yy = yf + fCol[f];
        for (int p = 0; p < 3; ++p)
        {
            double acc = 0.0;
            for (int bb = 0; bb < t; ++bb)
                acc += mm[p * t + bb] * yy[bb];
            q[p] -= acc;
        }
    }
    const double* iv = inv + 9 * std::int64_t(c);
    for (int p = 0; p < 3; ++p)
    {
        const double v = iv[p * 3 + 0] * q[0] + iv[p * 3 + 1] * q[1] + iv[p * 3 + 2] * q[2];
        yE[3 * std::int64_t(c) + p] = v;
        if (!arith::baFinite(v))
            *notFinite = 1;
    }
}

// ---- the minimiser's vectors (step 4d) -------------------------------------------------------------------------
// the elementwise ops, each the host loop's expression (std::min / std::max as their comparisons)
__global__ void kVecOp(int o, std::int64_t n, double* dst, const double* a, const double* b, double s, double t)
{
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    switch (o)
    {
        case int(Problem::Op::Copy):
            dst[i] = a[i];
            break;
        case int(Problem::Op::Scale):
            dst[i] = s * a[i];
            break;
        case int(Problem::Op::Neg):
            dst[i] = -a[i];
            break;
        case int(Problem::Op::Mul):
            dst[i] = a[i] * b[i];
            break;
        case int(Problem::Op::Jacobi):
            dst[i] = 1.0 / (1.0 + arith::baSqrt(a[i]));
            break;
        case int(Problem::Op::Clamp):
        {
            const double m = a[i] < s ? s : a[i];   // std::max(a, s)
            dst[i] = t < m ? t : m;                 // std::min(m, t)
            break;
        }
        case int(Problem::Op::SqrtDiv):
            dst[i] = arith::baSqrt(a[i] / s);
            break;
        case int(Problem::Op::ScaleInPlace):
            dst[i] = dst[i] * s;
            break;
        case int(Problem::Op::Add):
            dst[i] = a[i] + b[i];
            break;
        case int(Problem::Op::Fill):
            dst[i] = s;
            break;
        default:
            break;
    }
}

// psum / pmax per block of `blk` values (the host's blocks), one workgroup per block; the host combines the blocks
// in order. A sum is the host's chain, in order from zero: the workgroup stages a tile of the values in shared
// memory and one thread adds it in order. A maximum does not depend on the order (from zero, std::max skips a NaN,
// as the comparison here does), so every thread takes a share and the shares meet in a tree.
constexpr int kRedThreads = 256, kRedTile = 2048;

__device__ inline double redValue(int r, const double* a, const double* b, std::int64_t i)
{
    switch (r)
    {
        case int(Problem::Red::Dot):
            return a[i] * b[i];
        case int(Problem::Red::Sq):
            return a[i] * a[i];
        case int(Problem::Red::SqDiff):
            return (a[i] - b[i]) * (a[i] - b[i]);
        case int(Problem::Red::MaxAbs):
            return fabs(a[i]);
        default:
            return fabs(a[i] - b[i]);
    }
}

__global__ void __launch_bounds__(kRedThreads) kVecReduce(int r, std::int64_t n, std::int64_t blk, const double* a, const double* b,
                                                           double* partials)
{
    __shared__ double tile[kRedTile];
    const std::int64_t k = blockIdx.x, s = k * blk, e = s + blk < n ? s + blk : n;
    const int tid = int(threadIdx.x);
    if (r == int(Problem::Red::MaxAbs) || r == int(Problem::Red::MaxAbsDiff))
    {
        double m = 0.0;
        for (std::int64_t i = s + tid; i < e; i += kRedThreads)
        {
            const double v = redValue(r, a, b, i);
            m = m < v ? v : m;   // std::max(m, v)
        }
        tile[tid] = m;
        __syncthreads();
        for (int w = kRedThreads / 2; w > 0; w >>= 1)
        {
            if (tid < w)
            {
                const double q = tile[tid + w];
                tile[tid] = tile[tid] < q ? q : tile[tid];
            }
            __syncthreads();
        }
        if (tid == 0)
            partials[k] = tile[0];
        return;
    }
    double acc = 0.0;   // thread 0's
    for (std::int64_t t0 = s; t0 < e; t0 += kRedTile)
    {
        const int cnt = int(e - t0 < kRedTile ? e - t0 : kRedTile);
        for (int j = tid; j < cnt; j += kRedThreads)
            tile[j] = redValue(r, a, b, t0 + j);
        __syncthreads();
        if (tid == 0)
            for (int j = 0; j < cnt; ++j)
                acc += tile[j];
        __syncthreads();
    }
    if (tid == 0)
        partials[k] = acc;
}

// S's values in the order the host's factorisation reads them (its sparse matrix's, column-major)
__global__ void kOrder(const double* sv, const std::int64_t* order, std::int64_t n, double* out)
{
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = sv[order[i]];
}

unsigned gridOf(std::int64_t n, unsigned block)
{
    return unsigned((n + block - 1) / block);
}

// fn(k) for k < n on the host's threads (the setup; every k writes only its own outputs)
template <class F>
void hostParallel(int n, F&& fn)
{
    const int nt = int(std::max(1u, std::min(std::thread::hardware_concurrency(), unsigned(std::max(1, n)))));
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i)
        pool.emplace_back([&] {
            for (int k = next++; k < n; k = next++)
                fn(k);
        });
    for (auto& th : pool)
        th.join();
}

}  // namespace

// ---- the problem on the device ------------------------------------------------------------------------------
struct Problem::Impl
{
    std::string err;
    std::vector<void*> allocs;
    double secs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // CHESHIRE_BA_DEVICE_PROFILE=1: a sync after every kernel and copy, the time per slot, one line per solve
    enum Slot
    {
        kCreate, kUpload, kEvalK, kPartCostK, kColE, kColF, kReduce, kScaleK, kModelK, kPartDotK, kElimK, kTermsK, kFoldK, kHeavyK, kSumK, kBackK,
        kDownload, kSlots
    };
    bool prof = false;
    double kt[kSlots] = {};
    int calls[kSlots] = {};
    std::chrono::steady_clock::time_point last;
    void tick()
    {
        if (!prof)
            return;
        cudaDeviceSynchronize();
        last = std::chrono::steady_clock::now();
    }
    void tock(int slot)
    {
        if (!prof)
            return;
        cudaDeviceSynchronize();
        const auto now = std::chrono::steady_clock::now();
        kt[slot] += std::chrono::duration<double>(now - last).count();
        ++calls[slot];
        last = now;
    }

    int nRows = 0, nE = 0, ne = 0, nf = 0, n = 0, P = 0, G = 0, nPairs = 0, nItems = 0, nCam = 0, nPoses = 0, nManif = 0;
    std::int64_t jValues = 0, mValues = 0, sValues = 0, stride = 0, plusJValues = 0;
    int loss = 1;
    double hubA = 0, hubB = 0;
    std::vector<std::int32_t> hostParts;

    Row* rows = nullptr;
    std::int32_t *chunkStart = nullptr, *parts = nullptr, *groups = nullptr, *fTsize = nullptr, *fCol = nullptr;
    std::int64_t *cfStart = nullptr, *cfM = nullptr, *fcStart = nullptr, *sOff = nullptr, *manifOff = nullptr, *pairStart = nullptr;
    std::int32_t *cf = nullptr, *fc = nullptr, *fcPos = nullptr, *fcChunks = nullptr, *sRow = nullptr, *sCol = nullptr, *itemBlock = nullptr,
                 *itemRow = nullptr, *manifSize = nullptr, *manifTsize = nullptr, *pairPart = nullptr, *pairF = nullptr, *listRow = nullptr;
    std::int64_t* listJ = nullptr;
    std::int64_t* colStart = nullptr;
    std::int64_t nPairCols = 0;
    // the long chains (kHeavyEvents / kHeavyFold), processed in batches of groups so the scratch stays bounded
    struct HeavyBatch
    {
        int seg0, seg1;
        std::int64_t e0, e1, scr0, scr1, f0, f1;
    };
    std::vector<HeavyBatch> heavyBatches;
    int nSeg = 0;
    HeavyVal* hVals = nullptr;
    std::int32_t *hvStart = nullptr, *evKey = nullptr, *segCam = nullptr, *segGroup = nullptr;
    std::int64_t *evStart = nullptr, *scrBase = nullptr, *foldStart = nullptr;
    double* hScratch = nullptr;
    // the reduced system's other blocks (kAssembleRec): per (group, block) its records in the host's order
    int nS = 0;
    std::int64_t* recStart = nullptr;
    std::int32_t *recChunk = nullptr, *recU = nullptr, *recM = nullptr, *recRowStart = nullptr, *recRowCount = nullptr, *rowJ1 = nullptr,
                 *rowJ2 = nullptr, *rowR = nullptr, *itemCol = nullptr;
    double* mU = nullptr;
    // the minimiser's vectors (step 4d)
    std::vector<double*> vecs;
    std::vector<std::int64_t> vecN;
    double* partials = nullptr;
    std::int64_t partialsN = 0;
    // S in the host factorisation's order (setOrder)
    std::int64_t* order = nullptr;
    double* ordered = nullptr;
    std::int64_t nOrder = -1;
    double *constLm = nullptr, *plusJ = nullptr, *J = nullptr, *r = nullptr, *Jtmp = nullptr, *rtmp = nullptr, *rowCost = nullptr,
           *partCost = nullptr, *partF = nullptr, *terms = nullptr, *partDot = nullptr, *m = nullptr,
           *inv = nullptr, *ge = nullptr, *sg = nullptr, *sv = nullptr;
    char *rowOk = nullptr, *partOk = nullptr;
    PoseEntry* poses = nullptr;
    CameraValues* cams = nullptr;
    Camera* camInfo = nullptr;
    int* flag = nullptr;

    ~Impl()
    {
        if (prof)
        {
            static const char* names[kSlots] = {"create", "upload", "eval", "partCost", "colE", "colF", "reduceF", "scale", "model", "partDot",
                                                "elimChunk", "assemble", "unused", "heavy", "sumGroups", "back", "download"};
            std::string line = "[cheshire] BA device profile: " + std::to_string(nRows) + " rows:";
            char buf[96];
            for (int k = 0; k < kSlots; ++k)
                if (calls[k])
                {
                    std::snprintf(buf, sizeof buf, " %s %.4f (%d)", names[k], kt[k], calls[k]);
                    line += buf;
                }
            std::fprintf(stderr, "%s\n", line.c_str());
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (void* p : allocs)
            cudaFree(p);
        if (partials)
            cudaFree(partials);
        if (pooled)
        {
            cudaDeviceSynchronize();   // the pool's next holder may reuse the buffers at once
            std::lock_guard<std::mutex> lock(pool().m);
            pool().busy = false;
        }
        if (prof)
            std::fprintf(stderr, "[cheshire] BA device free: %.4f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    bool ok(cudaError_t e, const char* what)
    {
        if (e == cudaSuccess)
            return true;
        if (err.empty())
            err = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }
    double tAlloc = 0.0, tCopy = 0.0, tFree = 0.0;   // CHESHIRE_BA_DEVICE_PROFILE: the setup's allocations and copies
    bool pooled = false;   // this problem holds the pool
    size_t slot = 0;       // its next buffer in the pool
    void acquirePool()
    {
        if (!poolEnabled())
            return;
        std::lock_guard<std::mutex> lock(pool().m);
        if (!pool().busy)
            pooled = pool().busy = true;
    }
    template <typename T>
    bool alloc(T** p, std::int64_t count, const char* what)
    {
        void* q = nullptr;
        const size_t bytes = size_t(std::max<std::int64_t>(count, 1)) * sizeof(T);
        const auto t0 = std::chrono::steady_clock::now();
        if (pooled)
        {
            auto& slots = pool().slots;
            if (slot == slots.size())
                slots.emplace_back(nullptr, 0);
            auto& sl = slots[slot++];
            if (sl.second < bytes)
            {
                if (sl.first)
                    cudaFree(sl.first);
                sl.first = nullptr;
                sl.second = 0;
                const size_t grown = bytes + bytes / 4;
                if (!ok(cudaMalloc(&sl.first, grown), what))
                {
                    sl.first = nullptr;
                    return false;
                }
                sl.second = grown;
            }
            *p = static_cast<T*>(sl.first);
            tAlloc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            return true;
        }
        const bool good = ok(cudaMalloc(&q, bytes), what);
        tAlloc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!good)
            return false;
        allocs.push_back(q);
        *p = static_cast<T*>(q);
        return true;
    }
    template <typename T>
    bool upload(T** p, const T* v, size_t count, const char* what)
    {
        if (!alloc(p, std::int64_t(count), what))
            return false;
        const auto t0 = std::chrono::steady_clock::now();
        const bool good = count == 0 || ok(cudaMemcpy(*p, v, count * sizeof(T), cudaMemcpyHostToDevice), what);
        tCopy += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return good;
    }
    template <typename T>
    bool upload(T** p, const std::vector<T>& v, const char* what)
    {
        if (!alloc(p, std::int64_t(v.size()), what))
            return false;
        const auto t0 = std::chrono::steady_clock::now();
        const bool good = v.empty() || ok(cudaMemcpy(*p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), what);
        tCopy += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return good;
    }
    bool sync(const char* what) { return ok(cudaGetLastError(), what) && ok(cudaDeviceSynchronize(), what); }
};

bool available(std::string* name)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_checked)
    {
        g_checked = true;
        g_available = false;
        if (::cheshire::env::text("CHESHIRE_BA_DEVICE", "1") != "0")
        {
            int n = 0;
            cudaDeviceProp p{};
            if (cudaGetDeviceCount(&n) == cudaSuccess && n > 0 && cudaGetDeviceProperties(&p, 0) == cudaSuccess)
            {
                g_available = true;
                g_name = p.name;
            }
        }
    }
    if (name)
        *name = g_name;
    return g_available;
}

Problem::Problem(std::unique_ptr<Impl> impl)
  : _impl(std::move(impl))
{}

Problem::~Problem() = default;

std::string Problem::error() const { return _impl->err; }

double Problem::seconds(int phase) const { return phase >= 0 && phase < 8 ? _impl->secs[phase] : 0.0; }

std::unique_ptr<Problem> Problem::create(const Structure& s, std::string* why)
{
    auto im = std::make_unique<Impl>();
    Impl& I = *im;
    I.prof = ::cheshire::env::flag("CHESHIRE_BA_DEVICE_PROFILE");
    I.acquirePool();
    const auto tc = std::chrono::steady_clock::now();
    I.nRows = int(s.rows.size());
    I.nE = s.nE;
    I.ne = s.ne;
    I.nf = s.nf;
    I.n = s.n;
    I.P = int(s.parts.size()) - 1;
    I.G = int(s.groups.size()) - 1;
    I.jValues = s.jValues;
    I.mValues = s.mValues;
    I.sValues = s.sValues;
    I.stride = s.sValues + s.nf;
    I.loss = s.loss ? 1 : 0;
    I.hubA = s.hubA;
    I.hubB = s.hubB;
    I.nCam = int(s.cameras.size());
    I.nPoses = s.nPoses;
    I.nManif = int(s.manifSize.size());
    I.hostParts = s.parts;
    for (const auto& f : s.fTsize)
        if (f > kMaxT)
        {
            *why = "a camera block's tangent size is above " + std::to_string(kMaxT);
            return nullptr;
        }
    for (int q = 0; q < I.nManif; ++q)
        if (s.manifSize[q] > kMaxT)
        {
            *why = "a manifold's size is above " + std::to_string(kMaxT);
            return nullptr;
        }

    // the manifolds' Jacobians, concatenated
    std::vector<std::int64_t> manifOff(size_t(I.nManif) + 1, 0);
    for (int q = 0; q < I.nManif; ++q)
        manifOff[q + 1] = manifOff[q] + std::int64_t(s.manifSize[q]) * s.manifTsize[q];
    I.plusJValues = manifOff[I.nManif];

    // the rows (the largest upload) go up on a thread of their own while the host builds the rest
    if (!I.alloc(&I.rows, std::int64_t(s.rows.size()), "rows"))
    {
        *why = I.err;
        return nullptr;
    }
    cudaError_t rowsCopy = cudaSuccess;
    std::thread rowsUp([&] {
        if (s.rows.size())
            rowsCopy = cudaMemcpy(I.rows, s.rows.data(), s.rows.size() * sizeof(Row), cudaMemcpyHostToDevice);
    });
    struct Joiner
    {
        std::thread& th;
        ~Joiner()
        {
            if (th.joinable())
                th.join();
        }
    } joinRows{rowsUp};
    double stage[5] = {0, 0, 0, 0, 0};   // CHESHIRE_BA_DEVICE_PROFILE: create's stages
    auto stamp = [&](int k) { stage[k] = std::chrono::duration<double>(std::chrono::steady_clock::now() - tc).count(); };
    // per (partition, camera block), the partition's rows on the block, in order: each partition on its own,
    // then concatenated in partition order
    std::vector<std::int64_t> pairStart(1, 0);
    std::vector<std::int32_t> pairPart, pairF, listRow;
    std::vector<std::int64_t> listJ;
    {
        struct Entry
        {
            std::int32_t f, ri;
            std::int64_t j;
        };
        std::vector<std::vector<Entry>> per(size_t(I.P));
        const int nFb = int(s.fTsize.size());
        hostParallel(I.P, [&](int p) {
            // a counting sort by block over the blocks this partition touches; within a block, row order
            std::vector<std::int32_t> cnt(size_t(nFb) + 1, 0), touched;
            std::int64_t n = 0;
            for (int ri = s.parts[p]; ri < s.parts[p + 1]; ++ri)
            {
                const Row& row = s.rows[size_t(ri)];
                for (int k = 0; k < 3; ++k)
                    if (row.pj[k] >= 0)
                    {
                        if (cnt[size_t(row.blk[k])]++ == 0)
                            touched.push_back(row.blk[k]);
                        ++n;
                    }
            }
            std::sort(touched.begin(), touched.end());
            std::int32_t acc = 0;
            for (int f : touched)
            {
                const std::int32_t c = cnt[size_t(f)];
                cnt[size_t(f)] = acc;
                acc += c;
            }
            std::vector<Entry>& e = per[size_t(p)];
            e.resize(size_t(n));
            for (int ri = s.parts[p]; ri < s.parts[p + 1]; ++ri)
            {
                const Row& row = s.rows[size_t(ri)];
                for (int k = 0; k < 3; ++k)
                    if (row.pj[k] >= 0)
                        e[size_t(cnt[size_t(row.blk[k])]++)] = Entry{row.blk[k], ri, row.j + row.pj[k]};
            }
        });
        // per partition its pairs and entries, then everything placed in parallel at the prefix offsets
        std::vector<std::int64_t> pairOff(size_t(I.P) + 1, 0), entryOff(size_t(I.P) + 1, 0);
        for (int p = 0; p < I.P; ++p)
        {
            const std::vector<Entry>& e = per[size_t(p)];
            std::int64_t np = 0;
            for (size_t q = 0; q < e.size(); ++q)
                np += q == 0 || e[q].f != e[q - 1].f ? 1 : 0;
            pairOff[size_t(p) + 1] = pairOff[size_t(p)] + np;
            entryOff[size_t(p) + 1] = entryOff[size_t(p)] + std::int64_t(e.size());
        }
        const size_t nPairs = size_t(pairOff[size_t(I.P)]), nEntries = size_t(entryOff[size_t(I.P)]);
        pairPart.resize(nPairs);
        pairF.resize(nPairs);
        pairStart.resize(nPairs + 1);
        listRow.resize(nEntries);
        listJ.resize(nEntries);
        hostParallel(I.P, [&](int p) {
            const std::vector<Entry>& e = per[size_t(p)];
            std::int64_t pr = pairOff[size_t(p)], at = entryOff[size_t(p)];
            for (size_t q = 0; q < e.size(); ++q, ++at)
            {
                if (q == 0 || e[q].f != e[q - 1].f)
                {
                    pairPart[size_t(pr)] = p;
                    pairF[size_t(pr)] = e[q].f;
                    pairStart[size_t(pr)] = at;
                    ++pr;
                }
                listRow[size_t(at)] = e[q].ri;
                listJ[size_t(at)] = e[q].j;
            }
            std::vector<Entry>().swap(per[size_t(p)]);
        });
        pairStart[nPairs] = std::int64_t(nEntries);
        I.nPairs = int(pairPart.size());
    }
    std::vector<std::int64_t> colStart(size_t(I.nPairs) + 1, 0);
    for (int q = 0; q < I.nPairs; ++q)
        colStart[size_t(q) + 1] = colStart[size_t(q)] + s.fTsize[size_t(pairF[size_t(q)])];
    I.nPairCols = colStart[size_t(I.nPairs)];

    const int nF = int(s.fTsize.size());
    stamp(0);
    // the blocks inside one camera's intrinsics and distortion (the long chains): the camera blocks' kinds from
    // the rows' slots (0 intrinsics, 1 distortion, 2 pose)
    std::vector<int> kindOf(size_t(nF), -1), camOf(size_t(nF), -1);
    for (const Row& row : s.rows)
        for (int k = 0; k < 3; ++k)
            if (row.pj[k] >= 0)
            {
                kindOf[size_t(row.blk[k])] = k;
                if (k < 2)
                    camOf[size_t(row.blk[k])] = row.camera;
            }
    std::vector<std::vector<int>> internal(size_t(I.nCam));
    for (int f = 0; f < nF; ++f)
        if (kindOf[size_t(f)] == 0 || kindOf[size_t(f)] == 1)
            internal[size_t(camOf[size_t(f)])].push_back(f);
    auto blockOfH = [&](int f1, int f2) -> std::int64_t {
        const auto b = s.sCol.begin() + std::ptrdiff_t(s.sStart[f1]), e = s.sCol.begin() + std::ptrdiff_t(s.sStart[f1 + 1]);
        const auto it = std::lower_bound(b, e, f2);
        return it != e && *it == f2 ? std::int64_t(it - s.sCol.begin()) : -1;
    };
    std::vector<char> heavyBlock(s.sCol.size(), 0);
    std::vector<HeavyVal> vals;
    std::vector<std::int32_t> vStart(size_t(I.nCam) + 1, 0);
    for (int c = 0; c < I.nCam; ++c)
    {
        auto& B = internal[size_t(c)];
        std::sort(B.begin(), B.end());
        for (size_t i = 0; i < B.size(); ++i)
            for (size_t j = i; j < B.size(); ++j)
            {
                const int f1 = B[i], f2 = B[j];
                const std::int64_t sb = blockOfH(f1, f2);
                if (sb < 0)
                {
                    *why = "a camera's intrinsics blocks without their block of the reduced system";
                    return nullptr;
                }
                heavyBlock[size_t(sb)] = 1;
                for (int a = 0; a < s.fTsize[f1]; ++a)
                    for (int b = 0; b < s.fTsize[f2]; ++b)
                        vals.push_back(HeavyVal{f1, f2, a, b, s.sOff[size_t(sb)] + std::int64_t(a) * s.fTsize[f2] + b, 0, f1 == f2 && a == b ? 1 : 0});
            }
        for (int f : B)
            for (int a = 0; a < s.fTsize[f]; ++a)
                vals.push_back(HeavyVal{f, f, a, 0, s.sValues + s.fCol[f] + a, 1, 0});
        vStart[size_t(c) + 1] = std::int32_t(vals.size());
    }
    // the events per (group, camera), in the host's order: each group on its own, then concatenated in order
    std::vector<std::int32_t> evKey, segCam, segGroup;
    std::vector<std::int64_t> evStart, scrBase, foldStart;
    {
        std::int64_t scr = 0, fold = 0;
        std::vector<std::vector<std::int32_t>> keys(size_t(I.G) * size_t(I.nCam));
        hostParallel(I.G, [&](int g) {
            for (int c = 0; c < I.nCam; ++c)
            {
                const auto& B = internal[size_t(c)];
                if (B.empty())
                    continue;
                std::vector<std::int32_t>& key = keys[size_t(g) * size_t(I.nCam) + size_t(c)];
                const int anchor = B[0];
                auto hasAnchor = [&](const Row& row) {
                    for (int k = 0; k < 2; ++k)
                        if (row.pj[k] >= 0 && row.blk[k] == anchor)
                            return true;
                    return false;
                };
                const auto first = s.fc.begin() + std::ptrdiff_t(s.fcStart[anchor]);
                const auto last = first + s.fcChunks[anchor];
                for (auto it = std::lower_bound(first, last, s.groups[size_t(g)]); it != last && *it < s.groups[size_t(g) + 1]; ++it)
                {
                    const int ch = *it;
                    for (int ri = s.chunkStart[ch]; ri < s.chunkStart[ch + 1]; ++ri)
                        if (hasAnchor(s.rows[size_t(ri)]))
                            key.push_back(ri);
                    key.push_back(-1 - ch);
                }
                if (g == I.G - 1)
                    for (std::int64_t e = s.fcStart[anchor] + s.fcChunks[anchor]; e < s.fcStart[anchor + 1]; ++e)
                        key.push_back(-1 - s.fc[size_t(e)]);
            }
        });
        for (int g = 0; g < I.G; ++g)
            for (int c = 0; c < I.nCam; ++c)
            {
                evStart.push_back(std::int64_t(evKey.size()));
                segCam.push_back(c);
                segGroup.push_back(g);
                std::vector<std::int32_t>& key = keys[size_t(g) * size_t(I.nCam) + size_t(c)];
                evKey.insert(evKey.end(), key.begin(), key.end());
                std::vector<std::int32_t>().swap(key);
                const std::int64_t nEv = std::int64_t(evKey.size()) - evStart.back();
                const int nv = vStart[size_t(c) + 1] - vStart[size_t(c)];
                scrBase.push_back(scr);
                foldStart.push_back(fold);
                scr += nEv * nv;
                fold += nv;
            }
        evStart.push_back(std::int64_t(evKey.size()));
        scrBase.push_back(scr);
        foldStart.push_back(fold);
        I.nSeg = int(segCam.size());
        // batches of whole groups, each at most 64 M scratch values (one group always fits in one batch)
        const std::int64_t budget = std::int64_t(1) << 26;
        std::int64_t most = 0;
        int seg = 0;
        while (seg < I.nSeg)
        {
            Impl::HeavyBatch hb;
            hb.seg0 = seg;
            int end = seg;
            while (end < I.nSeg)
            {
                const int next = end + I.nCam;   // one more group
                if (end > seg && scrBase[size_t(next)] - scrBase[size_t(seg)] > budget)
                    break;
                end = next;
            }
            hb.seg1 = end;
            hb.e0 = evStart[size_t(seg)];
            hb.e1 = evStart[size_t(end)];
            hb.scr0 = scrBase[size_t(seg)];
            hb.scr1 = scrBase[size_t(end)];
            hb.f0 = foldStart[size_t(seg)];
            hb.f1 = foldStart[size_t(end)];
            most = std::max(most, hb.scr1 - hb.scr0);
            I.heavyBatches.push_back(hb);
            seg = end;
        }
        if (!I.alloc(&I.hScratch, most, "the long chains' terms"))
        {
            *why = I.err;
            return nullptr;
        }
    }
    stamp(1);
    // the reduced system's other blocks: per (group, block), in the host's order, its records - a landmark holding
    // both blocks (its rows on both, then its Schur term), then in the last group the rows without a landmark
    if (s.jValues >= (std::int64_t(1) << 31) || s.mValues >= (std::int64_t(1) << 31))
    {
        *why = "a Jacobian or an E^T F store above 2^31 values";
        return nullptr;
    }
    I.nS = int(s.sCol.size());
    std::vector<std::int32_t> sRow(s.sCol.size(), 0);
    for (int f1 = 0; f1 < nF; ++f1)
        for (std::int64_t b = s.sStart[f1]; b < s.sStart[f1 + 1]; ++b)
            sRow[size_t(b)] = f1;
    std::vector<std::int32_t> pairTab;
    const bool tab = std::int64_t(nF) * nF <= (std::int64_t(1) << 24);
    if (tab)
    {
        pairTab.assign(size_t(nF) * size_t(nF), -1);
        for (int f1 = 0; f1 < nF; ++f1)
            for (std::int64_t b = s.sStart[f1]; b < s.sStart[f1 + 1]; ++b)
                pairTab[size_t(f1) * size_t(nF) + size_t(s.sCol[size_t(b)])] = std::int32_t(b);
    }
    auto blockOf = [&](int f1, int f2) -> std::int32_t {
        if (tab)
            return pairTab[size_t(f1) * size_t(nF) + size_t(f2)];
        const auto b = s.sCol.begin() + std::ptrdiff_t(s.sStart[f1]), e = s.sCol.begin() + std::ptrdiff_t(s.sStart[f1 + 1]);
        return std::int32_t(std::lower_bound(b, e, f2) - s.sCol.begin());
    };
    std::vector<std::int32_t> elessRow;
    for (int ri = s.chunkStart[size_t(I.nE)]; ri < s.chunkStart[size_t(I.nE) + 1]; ++ri)
        elessRow.push_back(ri);
    struct RowOn
    {
        std::int32_t ri, j1, j2;
    };
    // visit(c0, c1, eless, f): the records of the landmarks [c0, c1) in the host's order (then, when eless, the rows
    // without a landmark), f(landmark or -1, f1's l, f2's l, block, rows on both)
    auto visit = [&](int c0, int c1, bool eless, auto&& blockOfPair, auto&& f) {
        std::vector<std::int32_t> lStart, lRow, lPj, slotPos, cur;
        std::vector<RowOn> both;
        for (int c = c0; c < c1; ++c)
        {
            const int r0 = s.chunkStart[size_t(c)], nr = s.chunkStart[size_t(c) + 1] - r0;
            const std::int64_t l0 = s.cfStart[size_t(c)], l1 = s.cfStart[size_t(c) + 1];
            const int K = int(l1 - l0);
            lStart.assign(size_t(K) + 1, 0);
            slotPos.assign(3 * size_t(nr), -1);
            for (int q = 0; q < nr; ++q)
            {
                const Row& row = s.rows[size_t(r0 + q)];
                for (int k = 0; k < 3; ++k)
                    if (row.pj[k] >= 0)
                    {
                        const auto it = std::lower_bound(s.cf.begin() + std::ptrdiff_t(l0), s.cf.begin() + std::ptrdiff_t(l1), row.blk[k]);
                        const int l = int(it - (s.cf.begin() + std::ptrdiff_t(l0)));
                        slotPos[3 * size_t(q) + size_t(k)] = l;
                        ++lStart[size_t(l) + 1];
                    }
            }
            for (int l = 0; l < K; ++l)
                lStart[size_t(l) + 1] += lStart[size_t(l)];
            lRow.assign(size_t(lStart[size_t(K)]), 0);
            lPj.assign(size_t(lStart[size_t(K)]), 0);
            {
                cur.assign(lStart.begin(), lStart.end() - 1);
                for (int q = 0; q < nr; ++q)
                    for (int k = 0; k < 3; ++k)
                    {
                        const int l = slotPos[3 * size_t(q) + size_t(k)];
                        if (l < 0)
                            continue;
                        const int at = cur[size_t(l)]++;
                        lRow[size_t(at)] = q;
                        lPj[size_t(at)] = s.rows[size_t(r0 + q)].pj[k];
                    }
            }
            for (int la = 0; la < K; ++la)
                for (int lb = la; lb < K; ++lb)
                {
                    const int f1 = s.cf[size_t(l0 + la)], f2 = s.cf[size_t(l0 + lb)];
                    const std::int32_t blk = blockOfPair(f1, f2);
                    if (heavyBlock[size_t(blk)])
                        continue;
                    both.clear();
                    for (int i1 = lStart[size_t(la)], i2 = lStart[size_t(lb)]; i1 < lStart[size_t(la) + 1] && i2 < lStart[size_t(lb) + 1];)
                    {
                        if (lRow[size_t(i1)] < lRow[size_t(i2)])
                            ++i1;
                        else if (lRow[size_t(i2)] < lRow[size_t(i1)])
                            ++i2;
                        else
                        {
                            const std::int64_t j = s.rows[size_t(r0 + lRow[size_t(i1)])].j;
                            both.push_back(RowOn{r0 + lRow[size_t(i1)], std::int32_t(j + lPj[size_t(i1)]), std::int32_t(j + lPj[size_t(i2)])});
                            ++i1;
                            ++i2;
                        }
                    }
                    f(c, int(l0 + la), int(l0 + lb), blk, both);
                }
        }
        if (eless)
            for (const int ri : elessRow)
            {
                const Row& row = s.rows[size_t(ri)];
                for (int k1 = 0; k1 < 3; ++k1)
                {
                    if (row.pj[k1] < 0)
                        continue;
                    for (int k2 = 0; k2 < 3; ++k2)
                    {
                        if (row.pj[k2] < 0 || row.blk[k2] < row.blk[k1] || (row.blk[k2] == row.blk[k1] && k2 != k1))
                            continue;
                        const std::int32_t blk = blockOfPair(row.blk[k1], row.blk[k2]);
                        if (heavyBlock[size_t(blk)])
                            continue;
                        both.assign(1, RowOn{ri, std::int32_t(row.j + row.pj[k1]), std::int32_t(row.j + row.pj[k2])});
                        f(-1, -1, -1, blk, both);
                    }
                }
            }
    };
    // over the groups on the host's threads; a group's records and rows are its own
    auto eachGroup = [&](auto&& fn) {
        const int nt = int(std::max(1u, std::min(std::thread::hardware_concurrency(), unsigned(I.G))));
        std::atomic<int> next{0};
        std::vector<std::thread> pool;
        for (int i = 0; i < nt; ++i)
            pool.emplace_back([&] {
                for (int g = next++; g < I.G; g = next++)
                    fn(g);
            });
        for (auto& th : pool)
            th.join();
    };
    // two passes over units of work: each group split into kSub landmark ranges (the rows without a landmark in the
    // last unit), so the host's threads stay busy; a unit's records go to (group, block) offsets after the earlier
    // units of its group, which keeps every (group, block)'s records in the host's order
    constexpr int kSub = 8;
    const int nU = I.G * kSub;
    std::vector<int> uC0(static_cast<size_t>(nU)), uC1(static_cast<size_t>(nU));
    for (int g = 0; g < I.G; ++g)
    {
        const int c0 = s.groups[size_t(g)], n = s.groups[size_t(g) + 1] - c0;
        for (int k = 0; k < kSub; ++k)
        {
            uC0[size_t(g * kSub + k)] = c0 + int(std::int64_t(n) * k / kSub);
            uC1[size_t(g * kSub + k)] = c0 + int(std::int64_t(n) * (k + 1) / kSub);
        }
    }
    // pass 1 looks each pair's block up (a random access into a table of nF^2) and keeps the answers in visit
    // order; pass 2 reads them back
    std::vector<std::vector<std::int32_t>> unitBlk(static_cast<size_t>(nU));
    auto visitUnit = [&](int u, auto&& blockOfPair, auto&& f) { visit(uC0[size_t(u)], uC1[size_t(u)], u == nU - 1, blockOfPair, f); };
    // pass 1: per unit, its records per block and its rows
    std::vector<std::int32_t> unitCnt(size_t(nU) * size_t(I.nS), 0);
    std::vector<std::int64_t> unitRows(size_t(nU) + 1, 0);
    hostParallel(nU, [&](int u) {
        std::int32_t* cnt = unitCnt.data() + size_t(u) * size_t(I.nS);
        std::int64_t rows = 0;
        std::vector<std::int32_t>& kept = unitBlk[size_t(u)];
        auto lookUp = [&](int f1, int f2) {
            const std::int32_t b = blockOf(f1, f2);
            kept.push_back(b);
            return b;
        };
        visitUnit(u, lookUp, [&](int, int, int, std::int32_t blk, const std::vector<RowOn>& on) {
            ++cnt[blk];
            rows += std::int64_t(on.size());
        });
        unitRows[size_t(u) + 1] = rows;
    });
    for (int u = 0; u < nU; ++u)
        unitRows[size_t(u) + 1] += unitRows[size_t(u)];
    if (unitRows[size_t(nU)] >= (std::int64_t(1) << 31))
    {
        *why = "more than 2^31 row terms";
        return nullptr;
    }
    // per (group, block) its records, and per (unit, block) where its records start
    std::vector<std::int64_t> recStart(size_t(I.G) * size_t(I.nS) + 1, 0);
    for (int g = 0; g < I.G; ++g)
        for (int b = 0; b < I.nS; ++b)
        {
            std::int64_t n = 0;
            for (int k = 0; k < kSub; ++k)
                n += unitCnt[size_t(g * kSub + k) * size_t(I.nS) + size_t(b)];
            recStart[size_t(g) * size_t(I.nS) + size_t(b) + 1] = n;
        }
    for (size_t q = 1; q < recStart.size(); ++q)
        recStart[q] += recStart[q - 1];
    std::vector<std::int64_t> unitPos(size_t(nU) * size_t(I.nS));
    hostParallel(I.G, [&](int g) {
        for (int b = 0; b < I.nS; ++b)
        {
            std::int64_t at = recStart[size_t(g) * size_t(I.nS) + size_t(b)];
            for (int k = 0; k < kSub; ++k)
            {
                const size_t q = size_t(g * kSub + k) * size_t(I.nS) + size_t(b);
                unitPos[q] = at;
                at += unitCnt[q];
            }
        }
    });
    std::vector<std::int32_t>().swap(unitCnt);
    // pass 2: the records and their rows
    const size_t nRec = size_t(recStart.back()), nOn = size_t(unitRows[size_t(nU)]);
    std::vector<std::int32_t> recChunk(nRec), recU(nRec), recM(nRec), recRowStart(nRec), recRowCount(nRec), rowJ1(nOn), rowJ2(nOn), rowR(nOn);
    hostParallel(nU, [&](int u) {
        std::int64_t* pos = unitPos.data() + size_t(u) * size_t(I.nS);
        std::int64_t at = unitRows[size_t(u)];
        const std::vector<std::int32_t>& kept = unitBlk[size_t(u)];
        size_t next = 0;
        auto readBack = [&](int, int) { return kept[next++]; };
        visitUnit(u, readBack, [&](int c, int la, int lb, std::int32_t blk, const std::vector<RowOn>& on) {
            const size_t q = size_t(pos[blk]++);
            recChunk[q] = c;
            recU[q] = c >= 0 ? std::int32_t(s.cfM[size_t(la)]) : 0;
            recM[q] = c >= 0 ? std::int32_t(s.cfM[size_t(lb)]) : 0;
            recRowStart[q] = std::int32_t(at);
            recRowCount[q] = std::int32_t(on.size());
            for (const RowOn& o : on)
            {
                rowJ1[size_t(at)] = o.j1;
                rowJ2[size_t(at)] = o.j2;
                rowR[size_t(at)] = o.ri;
                ++at;
            }
        });
    });
    stamp(2);
    // the assembly's items: (block, row of the block, column; t2 for the right-hand side of a diagonal block)
    std::vector<std::int32_t> itemBlock, itemRow, itemCol;
    for (int b = 0; b < I.nS; ++b)
    {
        if (heavyBlock[size_t(b)])
            continue;
        const int f1 = sRow[size_t(b)], f2 = s.sCol[size_t(b)];
        const int cols = s.fTsize[f2] + (f1 == f2 ? 1 : 0);
        for (int a = 0; a < s.fTsize[f1]; ++a)
            for (int col = 0; col < cols; ++col)
            {
                itemBlock.push_back(b);
                itemRow.push_back(a);
                itemCol.push_back(col);
            }
    }
    I.nItems = int(itemBlock.size());

    stamp(3);
    const bool ok = I.upload(&I.chunkStart, s.chunkStart, "chunks") && I.upload(&I.parts, s.parts, "parts") &&
                    I.upload(&I.groups, s.groups, "groups") && I.upload(&I.fTsize, s.fTsize, "fTsize") && I.upload(&I.fCol, s.fCol, "fCol") &&
                    I.upload(&I.cfStart, s.cfStart, "cfStart") && I.upload(&I.cfM, s.cfM, "cfM") && I.upload(&I.cf, s.cf, "cf") &&
                    I.upload(&I.fcStart, s.fcStart, "fcStart") && I.upload(&I.fc, s.fc, "fc") && I.upload(&I.fcPos, s.fcPos, "fcPos") &&
                    I.upload(&I.fcChunks, s.fcChunks, "fcChunks") && I.upload(&I.sOff, s.sOff, "sOff") && I.upload(&I.sCol, s.sCol, "sCol") &&
                    I.upload(&I.sRow, sRow, "sRow") && I.upload(&I.itemBlock, itemBlock, "items") && I.upload(&I.itemRow, itemRow, "items") &&
                    I.upload(&I.manifOff, manifOff, "manifolds") && I.upload(&I.manifSize, s.manifSize, "manifolds") &&
                    I.upload(&I.manifTsize, s.manifTsize, "manifolds") && I.upload(&I.pairStart, pairStart, "pairs") &&
                    I.upload(&I.pairPart, pairPart, "pairs") && I.upload(&I.pairF, pairF, "pairs") && I.upload(&I.listRow, listRow, "pairs") &&
                    I.upload(&I.listJ, listJ, "pairs") && I.upload(&I.colStart, colStart, "pairs") && I.upload(&I.constLm, s.constLandmarks, "landmarks") &&
                    I.upload(&I.camInfo, s.cameras, "cameras") && I.upload(&I.hVals, vals, "heavy") && I.upload(&I.hvStart, vStart, "heavy") &&
                    I.upload(&I.evKey, evKey, "heavy") && I.upload(&I.evStart, evStart, "heavy") && I.upload(&I.segCam, segCam, "heavy") &&
                    I.upload(&I.segGroup, segGroup, "heavy") && I.upload(&I.scrBase, scrBase, "heavy") && I.upload(&I.foldStart, foldStart, "heavy") &&
                    I.upload(&I.recStart, recStart, "records") && I.upload(&I.recChunk, recChunk, "records") &&
                    I.upload(&I.recU, recU, "records") && I.upload(&I.recM, recM, "records") && I.upload(&I.recRowStart, recRowStart, "records") &&
                    I.upload(&I.recRowCount, recRowCount, "records") && I.upload(&I.rowJ1, rowJ1, "records") && I.upload(&I.rowJ2, rowJ2, "records") &&
                    I.upload(&I.rowR, rowR, "records") && I.upload(&I.itemCol, itemCol, "items") && I.alloc(&I.mU, I.mValues, "u") &&
                    I.alloc(&I.plusJ, I.plusJValues, "plusJ") && I.alloc(&I.poses, std::int64_t(I.nPoses), "poses") &&
                    I.alloc(&I.cams, std::int64_t(I.nCam), "cameras") && I.alloc(&I.J, I.jValues, "the Jacobian") &&
                    I.alloc(&I.r, 2 * std::int64_t(I.nRows), "residuals") && I.alloc(&I.rowCost, std::int64_t(I.nRows), "rows") &&
                    I.alloc(&I.rowOk, std::int64_t(I.nRows), "rows") && I.alloc(&I.partCost, std::int64_t(I.P), "parts") &&
                    I.alloc(&I.partOk, std::int64_t(I.P), "parts") && I.alloc(&I.partF, std::int64_t(I.P) * I.nf, "parts") &&
                    I.alloc(&I.terms, 2 * std::int64_t(I.nRows), "model") && I.alloc(&I.partDot, std::int64_t(I.P), "model") &&
                    I.alloc(&I.m, I.mValues, "E^T F") && I.alloc(&I.inv, 9 * std::int64_t(I.nE), "inverses") &&
                    I.alloc(&I.ge, 3 * std::int64_t(I.nE), "E^T b") && I.alloc(&I.sg, std::int64_t(I.G) * I.stride, "the groups' systems") &&
                    I.alloc(&I.sv, I.stride, "the reduced system") && I.alloc(&I.flag, 1, "flag");
    rowsUp.join();
    if (!ok || !I.ok(rowsCopy, "rows"))
    {
        *why = I.err;
        return nullptr;
    }
    if (I.prof)
    {
        cudaDeviceSynchronize();
        stamp(4);
        std::fprintf(stderr, "[cheshire] BA device create: %d rows: columns %.4f, long chains %.4f, records %.4f, items %.4f, uploads %.4f (alloc %.4f, copy %.4f) s\n",
                     I.nRows, stage[0], stage[1] - stage[0], stage[2] - stage[1], stage[3] - stage[2], stage[4] - stage[3], I.tAlloc, I.tCopy);
        I.kt[Impl::kCreate] += std::chrono::duration<double>(std::chrono::steady_clock::now() - tc).count();
        ++I.calls[Impl::kCreate];
    }
    return std::unique_ptr<Problem>(new Problem(std::move(im)));
}

bool Problem::setOrder(const std::int64_t* src, std::int64_t count)
{
    Impl& I = *_impl;
    if (!I.alloc(&I.order, count, "S's order") || !I.alloc(&I.ordered, count, "S's order"))
        return false;
    I.nOrder = count;
    return count == 0 || I.ok(cudaMemcpy(I.order, src, size_t(count) * sizeof(std::int64_t), cudaMemcpyHostToDevice), "S's order");
}

int Problem::vector(std::int64_t n)
{
    Impl& I = *_impl;
    double* p = nullptr;
    if (!I.alloc(&p, n, "a vector"))
        return -1;
    I.vecs.push_back(p);
    I.vecN.push_back(n);
    return int(I.vecs.size()) - 1;
}

bool Problem::upload(int v, const double* h, std::int64_t offset, std::int64_t count)
{
    Impl& I = *_impl;
    return count <= 0 || I.ok(cudaMemcpy(I.vecs[size_t(v)] + offset, h, size_t(count) * sizeof(double), cudaMemcpyHostToDevice), "vector up");
}

bool Problem::download(int v, double* h, std::int64_t offset, std::int64_t count)
{
    Impl& I = *_impl;
    return count <= 0 || I.ok(cudaMemcpy(h, I.vecs[size_t(v)] + offset, size_t(count) * sizeof(double), cudaMemcpyDeviceToHost), "vector down");
}

bool Problem::op(Op o, int dst, int a, int b, double s, double t, std::int64_t n)
{
    Impl& I = *_impl;
    if (n <= 0)
        return true;
    kVecOp<<<gridOf(n, 256), 256>>>(int(o), n, I.vecs[size_t(dst)], a >= 0 ? I.vecs[size_t(a)] : nullptr, b >= 0 ? I.vecs[size_t(b)] : nullptr, s,
                                     t);
    return I.ok(cudaGetLastError(), "vector op");
}

bool Problem::reduce(Red r, int a, int b, std::int64_t n, std::int64_t block, std::vector<double>* partials)
{
    Impl& I = *_impl;
    const std::int64_t nb = (n + block - 1) / block;
    partials->assign(size_t(nb), 0.0);
    if (nb == 0)
        return true;
    if (I.partialsN < nb)
    {
        if (I.partials)
            cudaFree(I.partials);
        I.partials = nullptr;
        I.partialsN = 0;
        if (!I.ok(cudaMalloc(&I.partials, size_t(nb) * sizeof(double)), "partials"))
            return false;
        I.partialsN = nb;
    }
    kVecReduce<<<unsigned(nb), kRedThreads>>>(int(r), n, block, I.vecs[size_t(a)], b >= 0 ? I.vecs[size_t(b)] : nullptr, I.partials);
    return I.sync("vector reduce") && I.ok(cudaMemcpy(partials->data(), I.partials, size_t(nb) * sizeof(double), cudaMemcpyDeviceToHost), "partials");
}

bool Problem::evaluate(Mode mode, int xVec, const PoseEntry* poses, const CameraValues* cameras, const double* plusJacobians,
                       double* partCost, char* partOk, int gradVec)
{
    Impl& I = *_impl;
    const auto t0 = std::chrono::steady_clock::now();
    if (mode == Mode::Gradient && !I.Jtmp)
    {
        if (!I.alloc(&I.Jtmp, I.jValues, "the line search's Jacobian") || !I.alloc(&I.rtmp, 2 * std::int64_t(I.nRows), "the line search's residuals"))
            return false;
    }
    I.tick();
    double* const gradient = gradVec >= 0 ? I.vecs[size_t(gradVec)] : nullptr;
    if (!I.ok(cudaMemcpy(I.poses, poses, size_t(I.nPoses) * sizeof(PoseEntry), cudaMemcpyHostToDevice), "poses") ||
        !I.ok(cudaMemcpy(I.cams, cameras, size_t(I.nCam) * sizeof(CameraValues), cudaMemcpyHostToDevice), "cameras") ||
        (I.plusJValues > 0 && !I.ok(cudaMemcpy(I.plusJ, plusJacobians, size_t(I.plusJValues) * sizeof(double), cudaMemcpyHostToDevice), "plusJ")))
        return false;
    double* J = mode == Mode::Gradient ? I.Jtmp : I.J;
    double* r = mode == Mode::Gradient ? I.rtmp : I.r;
    EvalArgs A{I.rows, I.nRows, int(mode), I.vecs[size_t(xVec)], I.constLm, I.poses, I.cams, I.camInfo, I.plusJ, I.manifOff, I.manifSize, I.manifTsize,
               I.fTsize, J, r, I.rowCost, I.rowOk, I.loss, I.hubA, I.hubB};
    const unsigned B = 128;
    I.tock(Impl::kUpload);
    kEval<<<gridOf(I.nRows, B), B>>>(A);
    I.tock(Impl::kEvalK);
    kPartCost<<<gridOf(I.P, 64), 64>>>(I.parts, I.P, I.rowCost, I.rowOk, I.partCost, I.partOk);
    I.tock(Impl::kPartCostK);
    if (mode != Mode::Cost && gradient)
    {
        kColumnsE<<<gridOf(I.nE, B), B>>>(I.rows, I.chunkStart, I.nE, J, r, 0, gradient);
        I.tock(Impl::kColE);
        cudaMemsetAsync(I.partF, 0, size_t(I.P) * I.nf * sizeof(double));
        kColumnsF<<<gridOf(I.nPairCols, 64), 64>>>(I.pairStart, I.pairPart, I.pairF, I.listRow, I.listJ, I.colStart, I.nPairs, I.fTsize, I.fCol, I.nf,
                                                     J, r, 0, I.partF);
        I.tock(Impl::kColF);
        kReduceF<<<gridOf(I.nf, 64), 64>>>(I.partF, I.P, I.nf, gradient + I.ne);
        I.tock(Impl::kReduce);
    }
    if (!I.sync("evaluation"))
        return false;
    if (!I.ok(cudaMemcpy(partCost, I.partCost, size_t(I.P) * sizeof(double), cudaMemcpyDeviceToHost), "costs") ||
        !I.ok(cudaMemcpy(partOk, I.partOk, size_t(I.P), cudaMemcpyDeviceToHost), "costs"))
        return false;
    I.tock(Impl::kDownload);
    I.secs[0] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

bool Problem::squaredColumnNorms(int outVec)
{
    Impl& I = *_impl;
    double* const out = I.vecs[size_t(outVec)];
    const unsigned B = 128;
    I.tick();
    kColumnsE<<<gridOf(I.nE, B), B>>>(I.rows, I.chunkStart, I.nE, I.J, I.r, 1, out);
    I.tock(Impl::kColE);
    cudaMemsetAsync(I.partF, 0, size_t(I.P) * I.nf * sizeof(double));
    kColumnsF<<<gridOf(I.nPairCols, 64), 64>>>(I.pairStart, I.pairPart, I.pairF, I.listRow, I.listJ, I.colStart, I.nPairs, I.fTsize, I.fCol, I.nf,
                                                 I.J, I.r, 1, I.partF);
    I.tock(Impl::kColF);
    kReduceF<<<gridOf(I.nf, 64), 64>>>(I.partF, I.P, I.nf, out + I.ne);
    I.tock(Impl::kReduce);
    return I.sync("column norms");
}

bool Problem::scaleColumns(int scaleVec)
{
    Impl& I = *_impl;
    I.tick();
    kScale<<<gridOf(I.nRows, 128), 128>>>(I.rows, I.nRows, I.fTsize, I.fCol, I.ne, I.vecs[size_t(scaleVec)], I.J);
    I.tock(Impl::kScaleK);
    return I.sync("scale");
}

bool Problem::modelCostChange(int stepVec, double* partDot)
{
    Impl& I = *_impl;
    I.tick();
    kModel<<<gridOf(I.nRows, 128), 128>>>(I.rows, I.nRows, I.fTsize, I.fCol, I.ne, I.vecs[size_t(stepVec)], I.J, I.r, I.terms);
    I.tock(Impl::kModelK);
    kPartDot<<<gridOf(I.P, 64), 64>>>(I.parts, I.P, I.terms, I.partDot);
    I.tock(Impl::kPartDotK);
    const bool ok = I.sync("model cost") && I.ok(cudaMemcpy(partDot, I.partDot, size_t(I.P) * sizeof(double), cudaMemcpyDeviceToHost), "model cost");
    I.tock(Impl::kDownload);
    return ok;
}

bool Problem::eliminate(int DVec, double* sValues, double* rhs, bool* finite)
{
    Impl& I = *_impl;
    const auto t0 = std::chrono::steady_clock::now();
    I.tick();
    double* const D = I.vecs[size_t(DVec)];
    if (!I.ok(cudaMemset(I.flag, 0, sizeof(int)), "flag"))
        return false;
    I.tock(Impl::kUpload);
    kElimChunk<<<gridOf(I.nE, 128), 128>>>(I.rows, I.chunkStart, I.nE, I.cfStart, I.cf, I.cfM, I.fTsize, D, I.J, I.r, I.m, I.mU, I.inv, I.ge, I.flag);
    I.tock(Impl::kElimK);
    const AsmArgs As{I.recStart, I.recChunk, I.recU, I.recM, I.recRowStart, I.recRowCount, I.rowJ1, I.rowJ2, I.rowR, I.itemBlock, I.itemRow,
                     I.itemCol, I.sRow, I.sCol, I.sOff, I.fTsize, I.fCol, I.nS, I.nItems, I.G, I.ne, I.sValues, I.stride, D, I.J, I.r,
                     I.mU, I.m, I.ge, I.sg};
    const std::int64_t na = std::int64_t(I.G) * I.nItems;
    if (na > 0)
        kAssembleRec<<<gridOf(na, 128), 128>>>(As);
    I.tock(Impl::kTermsK);
    HeavyArgs H{I.rows, I.J, I.r, I.m, I.inv, I.ge, D, I.cfStart, I.cf, I.cfM, I.fTsize, I.fCol, I.hVals, I.hvStart, I.evKey, I.evStart,
                I.segCam, I.segGroup, I.scrBase, I.foldStart, I.nSeg, I.ne, I.stride, I.hScratch, I.sg};
    for (const auto& hb : I.heavyBatches)
    {
        if (hb.e1 > hb.e0)
            kHeavyEvents<<<gridOf(hb.e1 - hb.e0, 128), 128>>>(H, hb.e0, hb.e1, hb.scr0, hb.seg0, hb.seg1);
        if (hb.f1 > hb.f0)
            kHeavyFold<<<gridOf(hb.f1 - hb.f0, 64), 64>>>(H, hb.f0, hb.f1, hb.scr0, hb.seg0, hb.seg1);
    }
    I.tock(Impl::kHeavyK);
    kSumGroups<<<gridOf(I.stride, 128), 128>>>(I.sg, I.G, I.stride, I.sv);
    if (I.nOrder >= 0)
        kOrder<<<gridOf(I.nOrder, 256), 256>>>(I.sv, I.order, I.nOrder, I.ordered);
    I.tock(Impl::kSumK);
    int nf = 0;
    const bool ordered = I.nOrder >= 0;
    if (!I.sync("elimination") || !I.ok(cudaMemcpy(&nf, I.flag, sizeof(int), cudaMemcpyDeviceToHost), "flag") ||
        !I.ok(cudaMemcpy(sValues, ordered ? I.ordered : I.sv, size_t(ordered ? I.nOrder : I.sValues) * sizeof(double), cudaMemcpyDeviceToHost),
              "S") ||
        !I.ok(cudaMemcpy(rhs, I.sv + I.sValues, size_t(I.nf) * sizeof(double), cudaMemcpyDeviceToHost), "rhs"))
        return false;
    I.tock(Impl::kDownload);
    *finite = nf == 0;
    I.secs[1] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

bool Problem::backSubstitute(const double* yf, int yVec, bool* finite)
{
    Impl& I = *_impl;
    I.tick();
    double* const y = I.vecs[size_t(yVec)];
    // yf goes to y's camera part, and the landmarks' part is solved from it
    if (!I.ok(cudaMemcpy(y + I.ne, yf, size_t(I.nf) * sizeof(double), cudaMemcpyHostToDevice), "y") || !I.ok(cudaMemset(I.flag, 0, sizeof(int)), "flag"))
        return false;
    I.tock(Impl::kUpload);
    kBack<<<gridOf(I.nE, 128), 128>>>(I.cfStart, I.cf, I.cfM, I.fTsize, I.fCol, I.nE, I.m, I.inv, I.ge, y + I.ne, y, I.flag);
    I.tock(Impl::kBackK);
    int nf = 0;
    if (!I.sync("back-substitution") || !I.ok(cudaMemcpy(&nf, I.flag, sizeof(int), cudaMemcpyDeviceToHost), "flag"))
        return false;
    I.tock(Impl::kDownload);
    *finite = nf == 0;
    return true;
}

}  // namespace device
}  // namespace cheshire
}  // namespace sfm
}  // namespace aliceVision
