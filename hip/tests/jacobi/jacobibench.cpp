// The 7-point solver's nullspace (Fundamental7PSolver.cpp -> Nullspace2 -> Eigen::JacobiSVD<Mat9>(A, ComputeFullV)) against
// replicas of Eigen 3.4.1's JacobiSVD for a square 9x9 real matrix, bit for bit: V (all 81 entries) and the singular
// values, on 7-point systems built as encodeEpipolarEquation builds them (seven rows of normalised correspondences, two
// zero rows) plus degenerate, scaled and zero inputs; then timed.
//   replica 0: Eigen's statements written out in the same source form (so the compiler contracts them, or not, the same)
//   replica 1: the same operations, the work matrix row-major (rows padded to 12) so a left rotation is three AVX2
//              packets, V column-major padded to 12 so its rotation is three packets; the work matrix's column rotation
//              stays scalar. Lane-wise identical operations; contraction matched to what the compiler does to Eigen's
//              scalar statements (probed at startup).
// Build it with the flags of the build it vouches for (build.cmd: clang-cl /O2 /arch:AVX2 as AliceVision on Windows).
// jacobibench [count] [seed]; exit code 1 if any output differs.
#include <Eigen/Core>
#include <Eigen/SVD>
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

using Mat9 = Eigen::Matrix<double, 9, 9>;

struct Out
{
    double V[81];  // column-major
    double sv[9];
    bool ok;
};

// ---------------------------------------------------------------------------------------------------------------------
// reference
NOINLINE void eigenSvd(const Mat9& A, Out& o)
{
    Eigen::JacobiSVD<Mat9> svd(A, Eigen::ComputeFullV);
    const Mat9& V = svd.matrixV();
    for (int j = 0; j < 9; ++j)
        for (int i = 0; i < 9; ++i)
            o.V[j * 9 + i] = V(i, j);
    for (int i = 0; i < 9; ++i)
        o.sv[i] = svd.singularValues()(i);
    o.ok = true;
}

// ---------------------------------------------------------------------------------------------------------------------
// replica 0: Eigen's own statements, column-major 9x9
struct Rot
{
    double c, s;
};

static inline double conjd(double x) { return x; }
static inline double abs2d(double x) { return x * x; }  // numext::abs2: its own function, so never fused with an add

// apply_rotation_in_the_plane's scalar loop
static inline void rotPlane(double* x, int incx, double* y, int incy, int size, double c, double s)
{
    if (c == 1.0 && s == 0.0)
        return;
    for (int i = 0; i < size; ++i)
    {
        double xi = *x;
        double yi = *y;
        *x = c * xi + conjd(s) * yi;
        *y = -s * xi + conjd(c) * yi;
        x += incx;
        y += incy;
    }
}

static inline Rot makeJacobi(double x, double y, double z)
{
    using std::abs;
    using std::sqrt;
    Rot r;
    double deno = 2.0 * abs(y);
    if (deno < (std::numeric_limits<double>::min)())
    {
        r.c = 1.0;
        r.s = 0.0;
    }
    else
    {
        double tau = (x - z) / deno;
        double w = sqrt(abs2d(tau) + 1.0);
        double t;
        if (tau > 0.0)
            t = 1.0 / (tau + w);
        else
            t = 1.0 / (tau - w);
        double sign_t = t > 0.0 ? 1.0 : -1.0;
        double n = 1.0 / sqrt(abs2d(t) + 1.0);
        r.s = -sign_t * (conjd(y) / abs(y)) * abs(t) * n;
        r.c = n;
    }
    return r;
}

static inline Rot rotMul(const Rot& a, const Rot& b)  // JacobiRotation::operator*
{
    return Rot{a.c * b.c - conjd(a.s) * b.s, conjd(a.c * conjd(b.s) + conjd(a.s) * conjd(b.c))};
}

// real_2x2_jacobi_svd on the 2x2 block (p,q); w column-major with leading dimension ld
static inline void real2x2(const double* w, int ld, int p, int q, Rot& jl, Rot& jr)
{
    using std::abs;
    using std::sqrt;
    double m[4] = {w[p + p * ld], w[q + p * ld], w[p + q * ld], w[q + q * ld]};  // column-major 2x2: m00 m10 m01 m11
    Rot rot1;
    double t = m[0] + m[3];
    double d = m[1] - m[2];
    if (abs(d) < (std::numeric_limits<double>::min)())
    {
        rot1.s = 0.0;
        rot1.c = 1.0;
    }
    else
    {
        double u = t / d;
        double tmp = sqrt(1.0 + abs2d(u));
        rot1.s = 1.0 / tmp;
        rot1.c = u / tmp;
    }
    rotPlane(&m[0], 2, &m[1], 2, 2, rot1.c, rot1.s);  // m.applyOnTheLeft(0,1,rot1): rows 0 and 1, stride 2
    jr = makeJacobi(m[0], m[2], m[3]);
    jl = rotMul(rot1, Rot{jr.c, -conjd(jr.s)});
}

NOINLINE void replica0(const Mat9& Ain, Out& o)
{
    using std::abs;
    const double* A = Ain.data();
    const double precision = 2.0 * std::numeric_limits<double>::epsilon();
    const double considerAsZero = (std::numeric_limits<double>::min)();
    double scale = 0.0;
    for (int k = 0; k < 81; ++k)
    {
        const double a = abs(A[k]);
        if (std::isnan(a))
        {
            o.ok = false;
            return;
        }
        scale = a > scale ? a : scale;
    }
    if (!std::isfinite(scale))
    {
        o.ok = false;
        return;
    }
    if (scale == 0.0)
        scale = 1.0;
    double W[81];
    for (int k = 0; k < 81; ++k)
        W[k] = A[k] / scale;
    double* V = o.V;
    for (int k = 0; k < 81; ++k)
        V[k] = 0.0;
    for (int i = 0; i < 9; ++i)
        V[i * 9 + i] = 1.0;
    double maxDiag = abs(W[0]);
    for (int i = 1; i < 9; ++i)
        maxDiag = std::max(maxDiag, abs(W[i * 9 + i]));  // maxCoeff: the first of equals, no NaN here
    bool finished = false;
    while (!finished)
    {
        finished = true;
        for (int p = 1; p < 9; ++p)
        {
            for (int q = 0; q < p; ++q)
            {
                double threshold = std::max(considerAsZero, precision * maxDiag);
                if (abs(W[p + q * 9]) > threshold || abs(W[q + p * 9]) > threshold)
                {
                    finished = false;
                    Rot jl, jr;
                    real2x2(W, 9, p, q, jl, jr);
                    rotPlane(&W[p], 9, &W[q], 9, 9, jl.c, jl.s);                    // applyOnTheLeft: rows p, q
                    rotPlane(&W[p * 9], 1, &W[q * 9], 1, 9, jr.c, -conjd(jr.s));    // applyOnTheRight: j.transpose()
                    rotPlane(&V[p * 9], 1, &V[q * 9], 1, 9, jr.c, -conjd(jr.s));
                    maxDiag = std::max(maxDiag, std::max(abs(W[p * 9 + p]), abs(W[q * 9 + q])));
                }
            }
        }
    }
    for (int i = 0; i < 9; ++i)
        o.sv[i] = abs(W[i * 9 + i]);
    for (int i = 0; i < 9; ++i)
        o.sv[i] *= scale;
    for (int i = 0; i < 9; ++i)
    {
        int pos = 0;
        double mx = o.sv[i];
        for (int k = 1; k < 9 - i; ++k)
            if (o.sv[i + k] > mx)
            {
                mx = o.sv[i + k];
                pos = k;
            }
        if (mx == 0.0)
            break;
        if (pos)
        {
            pos += i;
            std::swap(o.sv[i], o.sv[pos]);
            for (int r = 0; r < 9; ++r)
                std::swap(V[pos * 9 + r], V[i * 9 + r]);
        }
    }
    o.ok = true;
}

// ---------------------------------------------------------------------------------------------------------------------
// replica 1: packets for the work matrix's row rotation and V's column rotation
static bool g_fused = false;  // the compiler contracts "c * xi + s * yi" into fma(c, xi, s * yi)

NOINLINE double probeContract(double c, double xi, double s, double yi) { return c * xi + conjd(s) * yi; }

static inline __m256d rotX(__m256d c, __m256d s, __m256d x, __m256d y)  // c * x + s * y
{
    return g_fused ? _mm256_fmadd_pd(c, x, _mm256_mul_pd(s, y)) : _mm256_add_pd(_mm256_mul_pd(c, x), _mm256_mul_pd(s, y));
}
static inline __m256d rotY(__m256d ns, __m256d c, __m256d x, __m256d y)  // -s * x + c * y, with ns = -s
{
    return g_fused ? _mm256_fmadd_pd(ns, x, _mm256_mul_pd(c, y)) : _mm256_add_pd(_mm256_mul_pd(ns, x), _mm256_mul_pd(c, y));
}

static inline void rotPacked(double* x, double* y, double c, double s)  // 12 doubles each (9 used)
{
    if (c == 1.0 && s == 0.0)
        return;
    const __m256d pc = _mm256_set1_pd(c), ps = _mm256_set1_pd(s), pns = _mm256_set1_pd(-s);
    for (int k = 0; k < 12; k += 4)
    {
        const __m256d xi = _mm256_loadu_pd(x + k), yi = _mm256_loadu_pd(y + k);
        _mm256_storeu_pd(x + k, rotX(pc, ps, xi, yi));
        _mm256_storeu_pd(y + k, rotY(pns, pc, xi, yi));
    }
}

NOINLINE void replica1(const Mat9& Ain, Out& o)
{
    using std::abs;
    const double* A = Ain.data();
    const double precision = 2.0 * std::numeric_limits<double>::epsilon();
    const double considerAsZero = (std::numeric_limits<double>::min)();
    double scale = 0.0;
    for (int k = 0; k < 81; ++k)
    {
        const double a = abs(A[k]);
        if (std::isnan(a))
        {
            o.ok = false;
            return;
        }
        scale = a > scale ? a : scale;
    }
    if (!std::isfinite(scale))
    {
        o.ok = false;
        return;
    }
    if (scale == 0.0)
        scale = 1.0;
    alignas(32) double R[9 * 12];  // work matrix, row-major, rows padded to 12 (zeros)
    alignas(32) double V[9 * 12];  // V, column-major, columns padded to 12 (zeros)
    for (int i = 0; i < 9; ++i)
    {
        for (int j = 0; j < 9; ++j)
            R[i * 12 + j] = A[i + j * 9] / scale;
        R[i * 12 + 9] = R[i * 12 + 10] = R[i * 12 + 11] = 0.0;
    }
    for (int k = 0; k < 9 * 12; ++k)
        V[k] = 0.0;
    for (int i = 0; i < 9; ++i)
        V[i * 12 + i] = 1.0;
    double maxDiag = abs(R[0]);
    for (int i = 1; i < 9; ++i)
        maxDiag = std::max(maxDiag, abs(R[i * 12 + i]));
    bool finished = false;
    while (!finished)
    {
        finished = true;
        for (int p = 1; p < 9; ++p)
        {
            for (int q = 0; q < p; ++q)
            {
                double threshold = std::max(considerAsZero, precision * maxDiag);
                if (abs(R[p * 12 + q]) > threshold || abs(R[q * 12 + p]) > threshold)
                {
                    finished = false;
                    // real2x2 reads W(p,p), W(q,p), W(p,q), W(q,q); here W(i,j) = R[i * 12 + j], i.e. column-major
                    // with leading dimension 1 and column stride 12 is not expressible - read the four directly
                    Rot jl, jr;
                    {
                        double m[4] = {R[p * 12 + p], R[q * 12 + p], R[p * 12 + q], R[q * 12 + q]};
                        Rot rot1;
                        double t = m[0] + m[3];
                        double d = m[1] - m[2];
                        if (abs(d) < (std::numeric_limits<double>::min)())
                        {
                            rot1.s = 0.0;
                            rot1.c = 1.0;
                        }
                        else
                        {
                            double u = t / d;
                            double tmp = std::sqrt(1.0 + abs2d(u));
                            rot1.s = 1.0 / tmp;
                            rot1.c = u / tmp;
                        }
                        rotPlane(&m[0], 2, &m[1], 2, 2, rot1.c, rot1.s);
                        jr = makeJacobi(m[0], m[2], m[3]);
                        jl = rotMul(rot1, Rot{jr.c, -conjd(jr.s)});
                    }
                    rotPacked(&R[p * 12], &R[q * 12], jl.c, jl.s);                // rows p, q: contiguous here
                    rotPlane(&R[p], 12, &R[q], 12, 9, jr.c, -conjd(jr.s));      // columns p, q: stride 12
                    rotPacked(&V[p * 12], &V[q * 12], jr.c, -conjd(jr.s));
                    maxDiag = std::max(maxDiag, std::max(abs(R[p * 12 + p]), abs(R[q * 12 + q])));
                }
            }
        }
    }
    for (int i = 0; i < 9; ++i)
        o.sv[i] = abs(R[i * 12 + i]);
    for (int i = 0; i < 9; ++i)
        o.sv[i] *= scale;
    int perm[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    for (int i = 0; i < 9; ++i)
    {
        int pos = 0;
        double mx = o.sv[i];
        for (int k = 1; k < 9 - i; ++k)
            if (o.sv[i + k] > mx)
            {
                mx = o.sv[i + k];
                pos = k;
            }
        if (mx == 0.0)
            break;
        if (pos)
        {
            pos += i;
            std::swap(o.sv[i], o.sv[pos]);
            std::swap(perm[i], perm[pos]);
        }
    }
    for (int j = 0; j < 9; ++j)
        for (int i = 0; i < 9; ++i)
            o.V[j * 9 + i] = V[perm[j] * 12 + i];
    o.ok = true;
}

// ---------------------------------------------------------------------------------------------------------------------
// replica 2: K independent SVDs interleaved, each lane replica 0's statements (so its rounding and contraction), the lanes
// in step so the processor overlaps their dependency chains (the 2x2 steps are a chain of divisions and square roots)
template<int K>
NOINLINE void replica2(const Mat9* const* Ain, Out* o)
{
    using std::abs;
    const double precision = 2.0 * std::numeric_limits<double>::epsilon();
    const double considerAsZero = (std::numeric_limits<double>::min)();
    double W[K][81];
    double scale[K];
    double maxDiag[K];
    bool live[K];  // still sweeping
    for (int l = 0; l < K; ++l)
    {
        const double* A = Ain[l]->data();
        double sc = 0.0;
        bool bad = false;
        for (int k = 0; k < 81; ++k)
        {
            const double a = abs(A[k]);
            if (std::isnan(a))
                bad = true;
            sc = a > sc ? a : sc;
        }
        o[l].ok = !bad && std::isfinite(sc);
        live[l] = o[l].ok;
        if (!o[l].ok)
            continue;
        if (sc == 0.0)
            sc = 1.0;
        scale[l] = sc;
        for (int k = 0; k < 81; ++k)
            W[l][k] = A[k] / sc;
        double* V = o[l].V;
        for (int k = 0; k < 81; ++k)
            V[k] = 0.0;
        for (int i = 0; i < 9; ++i)
            V[i * 9 + i] = 1.0;
        double md = abs(W[l][0]);
        for (int i = 1; i < 9; ++i)
            md = std::max(md, abs(W[l][i * 9 + i]));
        maxDiag[l] = md;
    }
    bool any = false;
    for (int l = 0; l < K; ++l)
        any = any || live[l];
    while (any)
    {
        bool rotated[K] = {};
        for (int p = 1; p < 9; ++p)
        {
            for (int q = 0; q < p; ++q)
            {
                bool act[K];
                Rot jl[K], jr[K];
                for (int l = 0; l < K; ++l)
                {
                    act[l] = false;
                    if (!live[l])
                        continue;
                    const double threshold = std::max(considerAsZero, precision * maxDiag[l]);
                    act[l] = abs(W[l][p + q * 9]) > threshold || abs(W[l][q + p * 9]) > threshold;
                }
                for (int l = 0; l < K; ++l)
                    if (act[l])
                        real2x2(W[l], 9, p, q, jl[l], jr[l]);
                for (int l = 0; l < K; ++l)
                {
                    if (!act[l])
                        continue;
                    rotated[l] = true;
                    double* w = W[l];
                    double* V = o[l].V;
                    rotPlane(&w[p], 9, &w[q], 9, 9, jl[l].c, jl[l].s);
                    rotPlane(&w[p * 9], 1, &w[q * 9], 1, 9, jr[l].c, -conjd(jr[l].s));
                    rotPlane(&V[p * 9], 1, &V[q * 9], 1, 9, jr[l].c, -conjd(jr[l].s));
                    maxDiag[l] = std::max(maxDiag[l], std::max(abs(w[p * 9 + p]), abs(w[q * 9 + q])));
                }
            }
        }
        any = false;
        for (int l = 0; l < K; ++l)
        {
            live[l] = live[l] && rotated[l];  // a sweep without a rotation ends this lane, as Eigen's loop
            any = any || live[l];
        }
    }
    for (int l = 0; l < K; ++l)
    {
        if (!o[l].ok)
            continue;
        double* sv = o[l].sv;
        double* V = o[l].V;
        for (int i = 0; i < 9; ++i)
            sv[i] = abs(W[l][i * 9 + i]);
        for (int i = 0; i < 9; ++i)
            sv[i] *= scale[l];
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
                    std::swap(V[pos * 9 + r], V[i * 9 + r]);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// replica 3: four SVDs in the four lanes of AVX2 packets. Every lane runs replica 0's operations, IEEE ones (add, sub,
// mul, div and sqrt are correctly rounded per lane); where the compiler contracts Eigen's "a * b + c * d" (the left
// product into fma(a, b, c * d)) the lanes use that fma (Fused). Branches become per-lane masks and blends; a lane whose
// rotation is the identity keeps its values (Eigen returns early), a finished lane does nothing.
template<bool Fused>
struct Lanes
{
    static inline __m256d fmaLike(__m256d a, __m256d b, __m256d c) { return Fused ? _mm256_fmadd_pd(a, b, c) : _mm256_add_pd(_mm256_mul_pd(a, b), c); }
    static inline __m256d abs(__m256d x) { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), x); }
    static inline __m256d neg(__m256d x) { return _mm256_xor_pd(_mm256_set1_pd(-0.0), x); }
    // x' = c x + s y, y' = -s x + c y where m (and the rotation is not the identity)
    static inline void rot(__m256d& x, __m256d& y, __m256d c, __m256d s, __m256d m)
    {
        const __m256d nx = fmaLike(c, x, _mm256_mul_pd(s, y));
        const __m256d ny = fmaLike(neg(s), x, _mm256_mul_pd(c, y));
        x = _mm256_blendv_pd(x, nx, m);
        y = _mm256_blendv_pd(y, ny, m);
    }
    static inline __m256d notIdentity(__m256d c, __m256d s)
    {
        const __m256d id = _mm256_and_pd(_mm256_cmp_pd(c, _mm256_set1_pd(1.0), _CMP_EQ_OQ), _mm256_cmp_pd(s, _mm256_setzero_pd(), _CMP_EQ_OQ));
        return _mm256_andnot_pd(id, _mm256_castsi256_pd(_mm256_set1_epi64x(-1)));
    }

    static NOINLINE void svd4(const Mat9* const* Ain, Out* o)
    {
        const __m256d one = _mm256_set1_pd(1.0), zero = _mm256_setzero_pd();
        const __m256d tiny = _mm256_set1_pd((std::numeric_limits<double>::min)());
        const __m256d precision = _mm256_set1_pd(2.0 * std::numeric_limits<double>::epsilon());
        __m256d W[81], V[81];
        alignas(32) double sc[4];
        bool okLane[4];
        for (int l = 0; l < 4; ++l)
        {
            const double* A = Ain[l]->data();
            double s = 0.0;
            bool bad = false;
            for (int k = 0; k < 81; ++k)
            {
                const double a = std::abs(A[k]);
                if (std::isnan(a))
                    bad = true;
                s = a > s ? a : s;
            }
            okLane[l] = !bad && std::isfinite(s);
            o[l].ok = okLane[l];
            sc[l] = (s == 0.0 || !okLane[l]) ? 1.0 : s;
        }
        const __m256d scale = _mm256_load_pd(sc);
        for (int k = 0; k < 81; ++k)
        {
            W[k] = _mm256_div_pd(_mm256_setr_pd(Ain[0]->data()[k], Ain[1]->data()[k], Ain[2]->data()[k], Ain[3]->data()[k]), scale);
            V[k] = zero;
        }
        for (int i = 0; i < 9; ++i)
            V[i * 9 + i] = one;
        __m256d maxDiag = abs(W[0]);
        for (int i = 1; i < 9; ++i)
        {
            const __m256d a = abs(W[i * 9 + i]);
            maxDiag = _mm256_blendv_pd(maxDiag, a, _mm256_cmp_pd(maxDiag, a, _CMP_LT_OQ));  // std::max: b if a < b
        }
        __m256d live = _mm256_setr_pd(okLane[0] ? -1.0 : 0.0, okLane[1] ? -1.0 : 0.0, okLane[2] ? -1.0 : 0.0, okLane[3] ? -1.0 : 0.0);
        live = _mm256_cmp_pd(live, zero, _CMP_NEQ_OQ);
        while (_mm256_movemask_pd(live))
        {
            __m256d rotated = zero;
            for (int p = 1; p < 9; ++p)
            {
                for (int q = 0; q < p; ++q)
                {
                    const __m256d pm = _mm256_mul_pd(precision, maxDiag);
                    const __m256d th = _mm256_blendv_pd(tiny, pm, _mm256_cmp_pd(tiny, pm, _CMP_LT_OQ));
                    __m256d act = _mm256_or_pd(_mm256_cmp_pd(abs(W[p + q * 9]), th, _CMP_GT_OQ), _mm256_cmp_pd(abs(W[q + p * 9]), th, _CMP_GT_OQ));
                    act = _mm256_and_pd(act, live);
                    if (!_mm256_movemask_pd(act))
                        continue;
                    rotated = _mm256_or_pd(rotated, act);
                    // real_2x2_jacobi_svd
                    __m256d m00 = W[p + p * 9], m10 = W[q + p * 9], m01 = W[p + q * 9], m11 = W[q + q * 9];
                    const __m256d t = _mm256_add_pd(m00, m11);
                    const __m256d d = _mm256_sub_pd(m10, m01);
                    const __m256d dSmall = _mm256_cmp_pd(abs(d), tiny, _CMP_LT_OQ);
                    const __m256d u = _mm256_div_pd(t, d);
                    const __m256d tmp = _mm256_sqrt_pd(_mm256_add_pd(one, _mm256_mul_pd(u, u)));
                    const __m256d c1 = _mm256_blendv_pd(_mm256_div_pd(u, tmp), one, dSmall);
                    const __m256d s1 = _mm256_blendv_pd(_mm256_div_pd(one, tmp), zero, dSmall);
                    const __m256d r1 = notIdentity(c1, s1);
                    rot(m00, m10, c1, s1, r1);
                    rot(m01, m11, c1, s1, r1);
                    // makeJacobi(m00, m01, m11)
                    const __m256d deno = _mm256_mul_pd(_mm256_set1_pd(2.0), abs(m01));
                    const __m256d jSmall = _mm256_cmp_pd(deno, tiny, _CMP_LT_OQ);
                    const __m256d tau = _mm256_div_pd(_mm256_sub_pd(m00, m11), deno);
                    const __m256d w = _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(tau, tau), one));
                    const __m256d tauPos = _mm256_cmp_pd(tau, zero, _CMP_GT_OQ);
                    const __m256d tt = _mm256_div_pd(one, _mm256_blendv_pd(_mm256_sub_pd(tau, w), _mm256_add_pd(tau, w), tauPos));
                    const __m256d signT = _mm256_blendv_pd(_mm256_set1_pd(-1.0), one, _mm256_cmp_pd(tt, zero, _CMP_GT_OQ));
                    const __m256d n = _mm256_div_pd(one, _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(tt, tt), one)));
                    __m256d sr = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(neg(signT), _mm256_div_pd(m01, abs(m01))), abs(tt)), n);
                    const __m256d cr = _mm256_blendv_pd(n, one, jSmall);
                    sr = _mm256_blendv_pd(sr, zero, jSmall);
                    // j_left = rot1 * j_right.transpose()
                    const __m256d srT = neg(sr);
                    const __m256d cl = fmaLike(c1, cr, neg(_mm256_mul_pd(s1, srT)));
                    const __m256d sl = fmaLike(c1, srT, _mm256_mul_pd(s1, cr));
                    // rows p and q by j_left, columns p and q (W and V) by j_right.transpose() = (cr, -sr)
                    const __m256d ml = _mm256_and_pd(act, notIdentity(cl, sl));
                    for (int j = 0; j < 9; ++j)
                        rot(W[p + j * 9], W[q + j * 9], cl, sl, ml);
                    const __m256d mr = _mm256_and_pd(act, notIdentity(cr, srT));
                    for (int i = 0; i < 9; ++i)
                        rot(W[i + p * 9], W[i + q * 9], cr, srT, mr);
                    for (int i = 0; i < 9; ++i)
                        rot(V[i + p * 9], V[i + q * 9], cr, srT, mr);
                    const __m256d app = abs(W[p * 9 + p]), aqq = abs(W[q * 9 + q]);
                    const __m256d mpq = _mm256_blendv_pd(app, aqq, _mm256_cmp_pd(app, aqq, _CMP_LT_OQ));
                    const __m256d nm = _mm256_blendv_pd(maxDiag, mpq, _mm256_cmp_pd(maxDiag, mpq, _CMP_LT_OQ));
                    maxDiag = _mm256_blendv_pd(maxDiag, nm, act);
                }
            }
            live = _mm256_and_pd(live, rotated);
        }
        alignas(32) double buf[4];
        for (int l = 0; l < 4; ++l)
        {
            if (!okLane[l])
                continue;
            double* sv = o[l].sv;
            double* Vo = o[l].V;
            for (int i = 0; i < 9; ++i)
            {
                _mm256_store_pd(buf, abs(W[i * 9 + i]));
                sv[i] = buf[l];
            }
            for (int i = 0; i < 9; ++i)
                sv[i] *= sc[l];
            for (int k = 0; k < 81; ++k)
            {
                _mm256_store_pd(buf, V[k]);
                Vo[k] = buf[l];
            }
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

static void replica3(const Mat9* const* in, Out* o)
{
    if (g_fused)
        Lanes<true>::svd4(in, o);
    else
        Lanes<false>::svd4(in, o);
}

// ---------------------------------------------------------------------------------------------------------------------
// inputs
static Mat9 sevenPoint(std::mt19937_64& rng, int kind)
{
    std::uniform_real_distribution<double> u(-1.5, 1.5);
    double x1[7][2], x2[7][2];
    for (int i = 0; i < 7; ++i)
    {
        x1[i][0] = u(rng);
        x1[i][1] = u(rng);
        // a loose epipolar relation: the second view shifted and rotated a little, plus noise
        x2[i][0] = 0.98 * x1[i][0] - 0.05 * x1[i][1] + 0.1 + 0.01 * u(rng);
        x2[i][1] = 0.05 * x1[i][0] + 0.98 * x1[i][1] - 0.03 + 0.01 * u(rng);
    }
    if (kind == 1)  // two identical correspondences (rank 6)
    {
        x1[3][0] = x1[1][0];
        x1[3][1] = x1[1][1];
        x2[3][0] = x2[1][0];
        x2[3][1] = x2[1][1];
    }
    if (kind == 2)  // integer pixel-like coordinates, unnormalised
        for (int i = 0; i < 7; ++i)
        {
            x1[i][0] = std::round(x1[i][0] * 2000.0);
            x1[i][1] = std::round(x1[i][1] * 1300.0);
            x2[i][0] = std::round(x2[i][0] * 2000.0);
            x2[i][1] = std::round(x2[i][1] * 1300.0);
        }
    if (kind == 3)  // all points on a line
        for (int i = 0; i < 7; ++i)
        {
            x1[i][1] = 0.5 * x1[i][0] + 0.25;
            x2[i][1] = 0.5 * x2[i][0] + 0.2;
        }
    Mat9 A = Mat9::Zero();
    for (int i = 0; i < 7; ++i)
    {
        const double a0 = x1[i][0], a1 = x1[i][1], b0 = x2[i][0], b1 = x2[i][1];
        A.row(i) << b0 * a0, b0 * a1, b0, b1 * a0, b1 * a1, b1, a0, a1, 1.0;
    }
    if (kind == 4)
        A *= 1e-150;
    if (kind == 5)
        A *= 1e150;
    if (kind == 6)
        A.setZero();
    if (kind == 7)
        A.row(2).setZero();
    return A;
}

static bool same(const Out& a, const Out& b)
{
    if (a.ok != b.ok)
        return false;
    if (!a.ok)
        return true;
    return std::memcmp(a.V, b.V, sizeof a.V) == 0 && std::memcmp(a.sv, b.sv, sizeof a.sv) == 0;
}

int main(int argc, char** argv)
{
    const long long count = argc > 1 ? std::atoll(argv[1]) : 2000000;
    const unsigned long long seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;
    {
        // does the compiler contract Eigen's scalar statement form? (inputs where fused and unfused differ)
        volatile double c = 1.0 + 0x1p-30, xi = 1.0 - 0x1p-30, s = -1.0, yi = 1.0;
        const double r = probeContract(c, xi, s, yi);
        const double fused = std::fma((double)c, (double)xi, (double)s * (double)yi);
        const double plain = [&] {
            volatile double a = (double)c * (double)xi;
            volatile double b = (double)s * (double)yi;
            return a + b;
        }();
        g_fused = (r == fused && r != plain);
        std::printf("compiler contracts \"c * xi + s * yi\": %s (fused %a, unfused %a, got %a)\n", g_fused ? "yes" : "no", fused, plain, r);
    }
    std::mt19937_64 rng(seed);
    std::vector<Mat9> inputs;
    inputs.reserve(count);
    for (long long k = 0; k < count; ++k)
    {
        int kind = 0;
        const long long m = k % 1000;
        if (m < 8)
            kind = static_cast<int>(m);
        inputs.push_back(sevenPoint(rng, kind));
    }
    long long diff0 = 0, diff1 = 0, notOk = 0;
    Out e, r0, r1;
    for (long long k = 0; k < count; ++k)
    {
        eigenSvd(inputs[k], e);
        replica0(inputs[k], r0);
        replica1(inputs[k], r1);
        if (!r0.ok)
            ++notOk;
        if (r0.ok && !same(e, r0))
        {
            if (diff0 < 3)
                std::printf("replica 0 differs at %lld\n", k);
            ++diff0;
        }
        if (r1.ok && !same(e, r1))
        {
            if (diff1 < 3)
                std::printf("replica 1 differs at %lld\n", k);
            ++diff1;
        }
    }
    std::printf("%lld inputs (seed %llu): replica 0 differs on %lld, replica 1 on %lld; %lld left to Eigen (not finite)\n", count, seed, diff0, diff1,
                notOk);
    // replica 2, four lanes and two
    long long diff2 = 0, diff2b = 0, diff3 = 0;
    for (long long k = 0; k + 4 <= count; k += 4)
    {
        const Mat9* in[4] = {&inputs[k], &inputs[k + 1], &inputs[k + 2], &inputs[k + 3]};
        Out o4[4];
        replica2<4>(in, o4);
        Out o2[2];
        replica2<2>(in, o2);
        Out o3[4];
        replica3(in, o3);
        for (int l = 0; l < 4; ++l)
        {
            eigenSvd(inputs[k + l], e);
            if (o4[l].ok && !same(e, o4[l]))
                ++diff2;
            if (l < 2 && o2[l].ok && !same(e, o2[l]))
                ++diff2b;
            if (o3[l].ok && !same(e, o3[l]))
            {
                if (diff3 < 3)
                    std::printf("replica 3 differs at %lld\n", k + l);
                ++diff3;
            }
        }
    }
    std::printf("replica 2: 4 lanes differ on %lld, 2 lanes on %lld; replica 3 (AVX2 lanes) on %lld\n", diff2, diff2b, diff3);
    diff1 += diff2 + diff2b + diff3;
    // timing on the plain 7-point systems
    const long long n = std::min<long long>(count, 400000);
    double sink = 0.0;
    auto time = [&](auto f) {
        double best = 1e30;
        for (int rep = 0; rep < 3; ++rep)
        {
            const auto t0 = std::chrono::steady_clock::now();
            for (long long k = 0; k < n; ++k)
            {
                Out o;
                f(inputs[k], o);
                sink += o.V[80];
            }
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        return best / double(n) * 1e9;
    };
    const double te = time(eigenSvd), t0 = time(replica0), t1 = time(replica1);
    std::printf("ns per SVD: Eigen %.0f, replica 0 %.0f (%.2fx), replica 1 %.0f (%.2fx)  [%g]\n", te, t0, te / t0, t1, te / t1, sink);
    auto timeLanes = [&](auto f, int K) {
        double best = 1e30;
        for (int rep = 0; rep < 3; ++rep)
        {
            const auto t0 = std::chrono::steady_clock::now();
            for (long long k = 0; k + K <= n; k += K)
            {
                const Mat9* in[8];
                for (int l = 0; l < K; ++l)
                    in[l] = &inputs[k + l];
                Out o[8];
                f(in, o);
                sink += o[0].V[80];
            }
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        return best / double(n) * 1e9;
    };
    const double t22 = timeLanes(replica2<2>, 2), t24 = timeLanes(replica2<4>, 4), t28 = timeLanes(replica2<8>, 8);
    const double t3 = timeLanes(replica3, 4);
    std::printf("ns per SVD, AVX2 lanes: %.0f (%.2fx)\n", t3, te / t3);
    std::printf("ns per SVD, lanes: 2 %.0f (%.2fx), 4 %.0f (%.2fx), 8 %.0f (%.2fx)  [%g]\n", t22, te / t22, t24, te / t24, t28, te / t28, sink);
    return (diff0 || diff1) ? 1 : 0;
}
