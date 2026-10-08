// This file is part of the Cheshire project (MPL-2.0), a patch set over AliceVision.
// Copied into src/aliceVision/numeric/ by scripts/apply_hip_patch.py (step 13a).
//
// cheshire (step 13a): the 7-point solver's nullspace, Nullspace2(A) = Eigen::JacobiSVD<Mat9>(A, ComputeFullV), for four
// matrices at once in the four lanes of AVX2 packets, bit for bit. The fit is a Jacobi sweep whose 2x2 steps are a chain
// of divisions and square roots, about three quarters of the geometric filter's time; one chain of packets serves four
// fits. Every lane runs Eigen 3.4.1's operations for a real square 9x9 matrix (JacobiSVD::compute, real_2x2_jacobi_svd,
// JacobiRotation::makeJacobi and operator*, apply_rotation_in_the_plane), all of them IEEE operations that AVX2 rounds
// per lane as the scalar unit does. Where the compiler contracts Eigen's scalar "a * b + c * d" (into fma(a, b, c * d),
// the left product first: clang's -ffp-contract=on with FMA), the lanes use that fma; where it does not, they do not
// (probed at startup, in this translation unit's flags). Branches become per-lane masks and blends; a lane whose
// rotation is the identity keeps its values (Eigen returns early) and a lane whose sweep rotated nothing stops.
// hip/tests/jacobi/jacobibench.cpp is the harness (millions of 7-point systems, degenerate, scaled and zero inputs); a
// self-test on first use compares fixed inputs with Eigen again and keeps Eigen if a bit differs.
// CHESHIRE_ACR_SVD_LANES=0: Eigen for every fit; CHESHIRE_ACR_SVD_LANES_CHECK=1: Eigen beside every lane, mismatches
// counted (cheshireSvdLanesReport).
#pragma once
#include <aliceVision/depthMap/cuda/hip/cheshire/env.h>  // CHESHIRE_* switches

#include <Eigen/Core>
#include <Eigen/SVD>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64)
#define CHESHIRE_SVD_LANES_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#if defined(__clang__) || defined(__GNUC__)
#define CHESHIRE_SVD_LANES_TARGET __attribute__((target("avx2,fma")))
#else
#define CHESHIRE_SVD_LANES_TARGET
#endif
#endif

namespace aliceVision {
namespace cheshireSvdLanes {

using Mat9 = Eigen::Matrix<double, 9, 9>;
using Vec9 = Eigen::Matrix<double, 9, 1>;

/// one lane's result: V column-major and the singular values, as JacobiSVD leaves them (sorted)
struct Result
{
    double V[81];
    double sv[9];
    bool finite;  // false: a NaN or an infinity in A, which JacobiSVD reports as InvalidInput; the caller asks Eigen
};

// Does the compiler contract Eigen's scalar statement "c * xi + conj(s) * yi" here? Inputs where fused and unfused differ.
inline double contractProbe(double c, double xi, double s, double yi) { return c * xi + s * yi; }
inline bool compilerContracts()
{
    volatile double c = 1.0 + 0x1p-30, xi = 1.0 - 0x1p-30, s = -1.0, yi = 1.0;
    const double got = contractProbe(c, xi, s, yi);
    const double fused = std::fma(double(c), double(xi), double(s) * double(yi));
    volatile double a = double(c) * double(xi);
    volatile double b = double(s) * double(yi);
    const double plain = a + b;
    return got == fused && got != plain;
}

#if defined(CHESHIRE_SVD_LANES_X86)
inline bool cpuHasAvx2Fma()
{
#if defined(__AVX2__) && defined(__FMA__)
    return true;  // the build requires it already
#else
    unsigned int a = 0, b = 0, c = 0, d = 0;
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 1);
    c = unsigned(r[2]);
#else
    if (!__get_cpuid(1, &a, &b, &c, &d))
        return false;
#endif
    const bool osxsave = (c >> 27) & 1, avx = (c >> 28) & 1, fma = (c >> 12) & 1;
    if (!(osxsave && avx && fma))
        return false;
#if defined(_MSC_VER)
    const unsigned long long xcr0 = _xgetbv(0);
#else
    unsigned int lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    const unsigned long long xcr0 = (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
    if ((xcr0 & 6) != 6)
        return false;  // the OS saves no YMM state
#if defined(_MSC_VER)
    __cpuidex(r, 7, 0);
    b = unsigned(r[1]);
#else
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d))
        return false;
#endif
    return (b >> 5) & 1;
#endif
}

template<bool Fused>
struct Lanes
{
    CHESHIRE_SVD_LANES_TARGET static inline __m256d fmaLike(__m256d a, __m256d b, __m256d c)
    {
        if constexpr (Fused)
            return _mm256_fmadd_pd(a, b, c);
        else
            return _mm256_add_pd(_mm256_mul_pd(a, b), c);
    }
    CHESHIRE_SVD_LANES_TARGET static inline __m256d absv(__m256d x) { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), x); }
    CHESHIRE_SVD_LANES_TARGET static inline __m256d neg(__m256d x) { return _mm256_xor_pd(_mm256_set1_pd(-0.0), x); }
    // apply_rotation_in_the_plane on one element pair: x' = c x + s y, y' = -s x + c y, where m
    CHESHIRE_SVD_LANES_TARGET static inline void rot(__m256d& x, __m256d& y, __m256d c, __m256d s, __m256d m)
    {
        const __m256d nx = fmaLike(c, x, _mm256_mul_pd(s, y));
        const __m256d ny = fmaLike(neg(s), x, _mm256_mul_pd(c, y));
        x = _mm256_blendv_pd(x, nx, m);
        y = _mm256_blendv_pd(y, ny, m);
    }
    // apply_rotation_in_the_plane returns early on c == 1 && s == 0
    CHESHIRE_SVD_LANES_TARGET static inline __m256d notIdentity(__m256d c, __m256d s)
    {
        const __m256d id = _mm256_and_pd(_mm256_cmp_pd(c, _mm256_set1_pd(1.0), _CMP_EQ_OQ), _mm256_cmp_pd(s, _mm256_setzero_pd(), _CMP_EQ_OQ));
        return _mm256_andnot_pd(id, _mm256_castsi256_pd(_mm256_set1_epi64x(-1)));
    }
    // std::max(a, b) = (a < b) ? b : a
    CHESHIRE_SVD_LANES_TARGET static inline __m256d stdMax(__m256d a, __m256d b) { return _mm256_blendv_pd(a, b, _mm256_cmp_pd(a, b, _CMP_LT_OQ)); }

    CHESHIRE_SVD_LANES_TARGET static void svd4(const Mat9* const* in, Result* out)
    {
        const __m256d one = _mm256_set1_pd(1.0), zero = _mm256_setzero_pd();
        const __m256d tiny = _mm256_set1_pd((std::numeric_limits<double>::min)());
        const __m256d precision = _mm256_set1_pd(2.0 * std::numeric_limits<double>::epsilon());
        __m256d W[81], V[81];
        alignas(32) double sc[4];
        bool finite[4];
        for (int l = 0; l < 4; ++l)
        {
            // JacobiSVD: scale = max |a| (NaN propagated); not finite -> InvalidInput; 0 -> 1
            const double* A = in[l]->data();
            double s = 0.0;
            bool nan = false;
            for (int k = 0; k < 81; ++k)
            {
                const double a = std::abs(A[k]);
                nan = nan || std::isnan(a);
                s = a > s ? a : s;
            }
            finite[l] = !nan && std::isfinite(s);
            out[l].finite = finite[l];
            sc[l] = (!finite[l] || s == 0.0) ? 1.0 : s;
        }
        const __m256d scale = _mm256_load_pd(sc);
        for (int k = 0; k < 81; ++k)
        {
            W[k] = _mm256_div_pd(_mm256_setr_pd(in[0]->data()[k], in[1]->data()[k], in[2]->data()[k], in[3]->data()[k]), scale);
            V[k] = zero;
        }
        for (int i = 0; i < 9; ++i)
            V[i * 9 + i] = one;
        __m256d maxDiag = absv(W[0]);
        for (int i = 1; i < 9; ++i)
            maxDiag = stdMax(maxDiag, absv(W[i * 9 + i]));
        __m256d live = _mm256_castsi256_pd(
          _mm256_setr_epi64x(finite[0] ? -1 : 0, finite[1] ? -1 : 0, finite[2] ? -1 : 0, finite[3] ? -1 : 0));
        while (_mm256_movemask_pd(live))
        {
            __m256d rotated = zero;
            for (int p = 1; p < 9; ++p)
            {
                for (int q = 0; q < p; ++q)
                {
                    const __m256d th = stdMax(tiny, _mm256_mul_pd(precision, maxDiag));
                    __m256d act = _mm256_or_pd(_mm256_cmp_pd(absv(W[p + q * 9]), th, _CMP_GT_OQ), _mm256_cmp_pd(absv(W[q + p * 9]), th, _CMP_GT_OQ));
                    act = _mm256_and_pd(act, live);
                    if (!_mm256_movemask_pd(act))
                        continue;
                    rotated = _mm256_or_pd(rotated, act);
                    // real_2x2_jacobi_svd
                    __m256d m00 = W[p + p * 9], m10 = W[q + p * 9], m01 = W[p + q * 9], m11 = W[q + q * 9];
                    const __m256d t = _mm256_add_pd(m00, m11);
                    const __m256d d = _mm256_sub_pd(m10, m01);
                    const __m256d dSmall = _mm256_cmp_pd(absv(d), tiny, _CMP_LT_OQ);
                    const __m256d u = _mm256_div_pd(t, d);
                    const __m256d tmp = _mm256_sqrt_pd(_mm256_add_pd(one, _mm256_mul_pd(u, u)));  // abs2: its own function, never fused
                    const __m256d c1 = _mm256_blendv_pd(_mm256_div_pd(u, tmp), one, dSmall);
                    const __m256d s1 = _mm256_blendv_pd(_mm256_div_pd(one, tmp), zero, dSmall);
                    const __m256d r1 = notIdentity(c1, s1);
                    rot(m00, m10, c1, s1, r1);
                    rot(m01, m11, c1, s1, r1);
                    // JacobiRotation::makeJacobi(m00, m01, m11)
                    const __m256d deno = _mm256_mul_pd(_mm256_set1_pd(2.0), absv(m01));
                    const __m256d jSmall = _mm256_cmp_pd(deno, tiny, _CMP_LT_OQ);
                    const __m256d tau = _mm256_div_pd(_mm256_sub_pd(m00, m11), deno);
                    const __m256d w = _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(tau, tau), one));
                    const __m256d tauPos = _mm256_cmp_pd(tau, zero, _CMP_GT_OQ);
                    const __m256d tt = _mm256_div_pd(one, _mm256_blendv_pd(_mm256_sub_pd(tau, w), _mm256_add_pd(tau, w), tauPos));
                    const __m256d signT = _mm256_blendv_pd(_mm256_set1_pd(-1.0), one, _mm256_cmp_pd(tt, zero, _CMP_GT_OQ));
                    const __m256d n = _mm256_div_pd(one, _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(tt, tt), one)));
                    __m256d sr = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(neg(signT), _mm256_div_pd(m01, absv(m01))), absv(tt)), n);
                    const __m256d cr = _mm256_blendv_pd(n, one, jSmall);
                    sr = _mm256_blendv_pd(sr, zero, jSmall);
                    // j_left = rot1 * j_right.transpose(); j_right.transpose() = (cr, -sr)
                    const __m256d srT = neg(sr);
                    const __m256d cl = fmaLike(c1, cr, neg(_mm256_mul_pd(s1, srT)));
                    const __m256d sl = fmaLike(c1, srT, _mm256_mul_pd(s1, cr));
                    // rows p, q by j_left; columns p, q of the work matrix and of V by j_right.transpose()
                    const __m256d ml = _mm256_and_pd(act, notIdentity(cl, sl));
                    for (int j = 0; j < 9; ++j)
                        rot(W[p + j * 9], W[q + j * 9], cl, sl, ml);
                    const __m256d mr = _mm256_and_pd(act, notIdentity(cr, srT));
                    for (int i = 0; i < 9; ++i)
                        rot(W[i + p * 9], W[i + q * 9], cr, srT, mr);
                    for (int i = 0; i < 9; ++i)
                        rot(V[i + p * 9], V[i + q * 9], cr, srT, mr);
                    const __m256d nm = stdMax(maxDiag, stdMax(absv(W[p * 9 + p]), absv(W[q * 9 + q])));
                    maxDiag = _mm256_blendv_pd(maxDiag, nm, act);
                }
            }
            live = _mm256_and_pd(live, rotated);  // a sweep without a rotation ends that lane, as Eigen's loop
        }
        alignas(32) double buf[4];
        for (int i = 0; i < 9; ++i)
        {
            _mm256_store_pd(buf, absv(W[i * 9 + i]));
            for (int l = 0; l < 4; ++l)
                out[l].sv[i] = buf[l];
        }
        for (int k = 0; k < 81; ++k)
        {
            _mm256_store_pd(buf, V[k]);
            for (int l = 0; l < 4; ++l)
                out[l].V[k] = buf[l];
        }
        for (int l = 0; l < 4; ++l)
        {
            if (!finite[l])
                continue;
            double* sv = out[l].sv;
            double* Vo = out[l].V;
            for (int i = 0; i < 9; ++i)
                sv[i] *= sc[l];
            // step 4: the largest remaining value first (maxCoeff: the first of equals), stopping at a zero
            for (int i = 0; i < 9; ++i)
            {
                int pos = 0;
                double mx = sv[i];
                for (int k = 1; k < 9 - i; ++k)
                    if (sv[i + k] > mx)
                    {
                        mx = sv[i + k];
                        pos = k;
                    }
                if (mx == 0.0)
                    break;
                if (pos)
                {
                    pos += i;
                    std::swap(sv[i], sv[pos]);
                    for (int r = 0; r < 9; ++r)
                        std::swap(Vo[pos * 9 + r], Vo[i * 9 + r]);
                }
            }
        }
    }
};
#endif

/// four matrices (repeat one to fill), the lanes' results; false when this build or CPU has no lanes
inline bool svd4(const Mat9* const* in, Result* out, bool fused)
{
#if defined(CHESHIRE_SVD_LANES_X86)
    if (fused)
        Lanes<true>::svd4(in, out);
    else
        Lanes<false>::svd4(in, out);
    return true;
#else
    (void)in;
    (void)out;
    (void)fused;
    return false;
#endif
}

/// Eigen's answer for one matrix, as Nullspace2 reads it
inline void eigenSvd(const Mat9& A, Result& r)
{
    Eigen::JacobiSVD<Mat9> svd(A, Eigen::ComputeFullV);
    const Mat9& V = svd.matrixV();
    for (int j = 0; j < 9; ++j)
        for (int i = 0; i < 9; ++i)
            r.V[j * 9 + i] = V(i, j);
    for (int i = 0; i < 9; ++i)
        r.sv[i] = svd.singularValues()(i);
    r.finite = true;
}

/// fixed pseudo-random 7-point systems (encodeEpipolarEquation's rows) and degenerate, scaled and zero ones
inline Mat9 selfTestMatrix(std::uint64_t& state, int kind)
{
    auto next = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(state >> 11) * 0x1p-53 * 3.0 - 1.5;
    };
    double x1[7][2], x2[7][2];
    for (int i = 0; i < 7; ++i)
    {
        x1[i][0] = next();
        x1[i][1] = next();
        x2[i][0] = 0.98 * x1[i][0] - 0.05 * x1[i][1] + 0.1 + 0.01 * next();
        x2[i][1] = 0.05 * x1[i][0] + 0.98 * x1[i][1] - 0.03 + 0.01 * next();
    }
    if (kind == 1)
    {
        x1[3][0] = x1[1][0];
        x1[3][1] = x1[1][1];
        x2[3][0] = x2[1][0];
        x2[3][1] = x2[1][1];
    }
    if (kind == 2)
        for (int i = 0; i < 7; ++i)
            for (int c = 0; c < 2; ++c)
            {
                x1[i][c] = std::round(x1[i][c] * 1500.0);
                x2[i][c] = std::round(x2[i][c] * 1500.0);
            }
    Mat9 A = Mat9::Zero();
    for (int i = 0; i < 7; ++i)
    {
        const double a0 = x1[i][0], a1 = x1[i][1], b0 = x2[i][0], b1 = x2[i][1];
        A.row(i) << b0 * a0, b0 * a1, b0, b1 * a0, b1 * a1, b1, a0, a1, 1.0;
    }
    if (kind == 3)
        A *= 1e-150;
    if (kind == 4)
        A *= 1e150;
    if (kind == 5)
        A.setZero();
    if (kind == 6)
        A.row(2).setZero();
    return A;
}

inline bool selfTest(bool fused)
{
    std::uint64_t state = 0x13a0cafe;
    for (int k = 0; k < 256; k += 4)
    {
        Mat9 A[4];
        const Mat9* in[4];
        for (int l = 0; l < 4; ++l)
        {
            A[l] = selfTestMatrix(state, (k + l) % 16 < 7 ? (k + l) % 16 : 0);
            in[l] = &A[l];
        }
        Result lanes[4];
        if (!svd4(in, lanes, fused))
            return false;
        for (int l = 0; l < 4; ++l)
        {
            Result e;
            eigenSvd(A[l], e);
            if (!lanes[l].finite || std::memcmp(lanes[l].V, e.V, sizeof e.V) != 0 || std::memcmp(lanes[l].sv, e.sv, sizeof e.sv) != 0)
                return false;
        }
    }
    return true;
}

struct State
{
    bool on = false;
    bool fused = false;
    bool check = false;
};

inline const State& state()
{
    static const State s = []() {
        State r;
        r.check = ::cheshire::env::flag("CHESHIRE_ACR_SVD_LANES_CHECK");
        if (!::cheshire::env::flag("CHESHIRE_ACR_SVD_LANES", true))
        {
            std::fprintf(stderr, "[cheshire] 7-point fits one at a time through Eigen (CHESHIRE_ACR_SVD_LANES=0)\n");
            return r;
        }
#if defined(CHESHIRE_SVD_LANES_X86)
        if (!cpuHasAvx2Fma())
        {
            std::fprintf(stderr, "[cheshire] 7-point fits one at a time through Eigen (no AVX2 and FMA on this CPU)\n");
            return r;
        }
        r.fused = compilerContracts();
        if (!selfTest(r.fused))
        {
            std::fprintf(stderr, "[cheshire] 7-point fits one at a time through Eigen (the four-lane SVD's self-test differs from Eigen)\n");
            return r;
        }
        r.on = true;
        std::fprintf(stderr, "[cheshire] 7-point fits four at a time, Eigen's SVD in AVX2 lanes (exact; CHESHIRE_ACR_SVD_LANES=0 for one at a time)\n");
#else
        std::fprintf(stderr, "[cheshire] 7-point fits one at a time through Eigen (no AVX2 lanes on this architecture)\n");
#endif
        return r;
    }();
    return s;
}

struct Stats
{
    std::atomic<long long> fits{0}, checked{0}, differ{0};
};
inline Stats& stats()
{
    static Stats s;
    return s;
}

/// CHESHIRE_ACR_SVD_LANES_CHECK=1: what the check found, once the caller's estimations are done
inline void report()
{
    const Stats& s = stats();
    if (state().check && s.checked.load() > 0)
        std::fprintf(stderr, "[cheshire] 7-point four-lane SVD check: %lld of %lld fits identical to Eigen's\n", s.checked.load() - s.differ.load(),
                     s.checked.load());
}

}  // namespace cheshireSvdLanes

/// cheshire (step 13a): whether cheshireNullspace2Lanes runs four fits at once (switch, CPU and self-test, announced once)
inline bool cheshireSvdLanesOn() { return cheshireSvdLanes::state().on; }

/// cheshire (step 13a): Nullspace2(A[l], x1[l], x2[l]) for l < n, n <= 4, the same bits; requires cheshireSvdLanesOn()
inline void cheshireNullspace2Lanes(const Eigen::Matrix<double, 9, 9>* const* A, int n, Eigen::Matrix<double, 9, 1>* x1, Eigen::Matrix<double, 9, 1>* x2)
{
    using namespace cheshireSvdLanes;
    const Mat9* in[4];
    for (int l = 0; l < 4; ++l)
        in[l] = A[l < n ? l : 0];
    Result r[4];
    svd4(in, r, state().fused);
    for (int l = 0; l < n; ++l)
    {
        if (!r[l].finite)
            eigenSvd(*A[l], r[l]);  // InvalidInput: whatever Eigen leaves, as upstream reads it
        for (int i = 0; i < 9; ++i)
        {
            x1[l](i) = r[l].V[8 * 9 + i];
            x2[l](i) = r[l].V[7 * 9 + i];
        }
        if (state().check)
        {
            Result e;
            eigenSvd(*A[l], e);
            ++stats().checked;
            if (std::memcmp(r[l].V, e.V, sizeof e.V) != 0 || std::memcmp(r[l].sv, e.sv, sizeof e.sv) != 0)
                ++stats().differ;
        }
    }
    stats().fits += n;
}

}  // namespace aliceVision
