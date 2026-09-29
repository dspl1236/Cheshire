// This file is part of the Cheshire patch set for AliceVision (https://github.com/dspl1236/Cheshire).
//
// 0.3.7: a bundle-adjustment solver of Cheshire's own (docs/notes/ba-own-solver.md): step 1 shadow mode,
// step 2 the parallel, block-sparse implementation, step 3 in SfM in place of Ceres (the default;
// CHESHIRE_BA_SOLVER=ceres for Ceres).
//
// It walks the ceres::Problem BundleAdjustmentCeres has built, through Ceres' public API, and solves it
// with Ceres' own algorithm specialised for this shape: Levenberg-Marquardt in a trust region, Jacobi
// scaling, the Huber corrector, landmarks eliminated by Schur complement and the reduced camera system
// solved by dense LLT (DENSE_SCHUR) or Eigen's SimplicialLDLT (SPARSE_SCHUR, the EIGEN_SPARSE path
// Cheshire's GPL-free Ceres takes: the camera blocks in the block AMD order Ceres gives them, then the
// natural order in the factorisation). Every residual block is evaluated by its own
// CostFunction, so the numbers are the ones Ceres sees; what this replaces is the machinery around them.
//
// In shadow mode (CHESHIRE_BA_SHADOW=1) it runs in place on the problem's parameter blocks before Ceres,
// records its trajectory and its result, and puts the starting values back, so Ceres then solves exactly
// what it would have and the reconstruction is today's. The two trajectories are compared per solve.
//
// Bounds: AliceVision bounds the focal length (with a prior) and the principal point. With any bound on a
// free block Ceres projects every Plus onto the box and runs a projected Armijo line search (cubic
// interpolation) along each trust-region step; both are ported here, the polynomial helpers from
// polynomial.cc (BSD-3-Clause, Copyright 2023 Google Inc.).
//
// Threads (step 2): the evaluation, the elimination and the back-substitution run on OpenMP over
// partitions fixed by the problem alone, and every sum across them is taken in partition order, so the
// result does not depend on the thread count. The reduced camera system is kept as its upper block
// triangle, assembled row by row of blocks; SPARSE_SCHUR analyses its pattern once per solve, as Ceres'
// EigenSparseCholesky does.
//
// The references to Ceres below are to the 2.2.0 source Cheshire builds.
#pragma once

#include <aliceVision/depthMap/cuda/hip/cheshire/env.h>
#include <aliceVision/system/Logger.hpp>

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace aliceVision {
namespace sfm {
namespace cheshire {
namespace own {

struct Options
{
    bool sparse = false;               // SPARSE_SCHUR (Eigen SimplicialLDLT) or DENSE_SCHUR (LLT)
    int threads = 1;                   // Ceres' num_threads; the result does not depend on it
    int maxIterations = 50;
    int maxConsecutiveInvalidSteps = 10;
    double functionTolerance = 1e-6;
    double gradientTolerance = 1e-10;
    double parameterTolerance = 1e-8;
    double initialRadius = 1e4;
    double maxRadius = 1e16;
    double minRadius = 1e-32;
    double minRelativeDecrease = 1e-3;
    double minDiagonal = 1e-6;
    double maxDiagonal = 1e32;
    bool jacobiScaling = true;
    // the projected line search Ceres runs when the problem has bounds (Solver::Options defaults)
    int lineSearchMaxIterations = 20;
    double lineSearchSufficientDecrease = 1e-4;
    double lineSearchMaxContraction = 1e-3;
    double lineSearchMinContraction = 0.6;
    double lineSearchMinStepSize = 1e-9;
};

// ---- polynomial.cc, ported ---------------------------------------------------------------------------
namespace poly {
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

inline double evaluate(const Vec& p, double x)
{
    double v = 0.0;
    for (int i = 0; i < p.size(); ++i)
        v = v * x + p(i);
    return v;
}

inline void balanceCompanion(Mat* m)
{
    Mat& cm = *m;
    Mat off = cm;
    off.diagonal().setZero();
    const int degree = int(cm.rows());
    const double gamma = 0.9;
    bool changed;
    do
    {
        changed = false;
        for (int i = 0; i < degree; ++i)
        {
            const double rowNorm = off.row(i).lpNorm<1>();
            const double colNorm = off.col(i).lpNorm<1>();
            int exponent = 0;
            std::frexp(rowNorm / colNorm, &exponent);
            exponent /= 2;
            if (exponent != 0)
            {
                const double sc = std::ldexp(colNorm, exponent);
                const double sr = std::ldexp(rowNorm, -exponent);
                if (sc + sr < gamma * (colNorm + rowNorm))
                {
                    changed = true;
                    off.row(i) *= std::ldexp(1.0, -exponent);
                    off.col(i) *= std::ldexp(1.0, exponent);
                }
            }
        }
    } while (changed);
    off.diagonal() = cm.diagonal();
    cm = off;
}

inline Vec removeLeadingZeros(const Vec& in)
{
    int i = 0;
    while (i < (in.size() - 1) && in(i) == 0.0)
        ++i;
    return in.tail(in.size() - i);
}

inline bool roots(const Vec& in, Vec* real)
{
    if (in.size() == 0)
        return false;
    Vec p = removeLeadingZeros(in);
    const int degree = int(p.size()) - 1;
    if (degree == 0)
        return true;   // Ceres leaves *real untouched here
    if (degree == 1)
    {
        real->resize(1);
        (*real)(0) = -p(1) / p(0);
        return true;
    }
    if (degree == 2)
    {
        const double a = p(0), b = p(1), c = p(2);
        const double D = b * b - 4 * a * c;
        const double sqrtD = std::sqrt(std::fabs(D));
        real->setZero(2);
        if (D >= 0)
        {
            if (b >= 0)
            {
                (*real)(0) = (-b - sqrtD) / (2.0 * a);
                (*real)(1) = (2.0 * c) / (-b - sqrtD);
            }
            else
            {
                (*real)(0) = (2.0 * c) / (-b + sqrtD);
                (*real)(1) = (-b + sqrtD) / (2.0 * a);
            }
            return true;
        }
        (*real)(0) = -b / (2.0 * a);
        (*real)(1) = -b / (2.0 * a);
        return true;
    }
    p /= p(0);
    Mat cm = Mat::Zero(degree, degree);
    cm.diagonal(-1).setOnes();
    cm.col(degree - 1) = -p.reverse().head(degree);
    balanceCompanion(&cm);
    Eigen::EigenSolver<Mat> solver(cm, false);
    if (solver.info() != Eigen::Success)
        return false;
    *real = solver.eigenvalues().real();
    return true;
}

inline Vec differentiate(const Vec& p)
{
    const int degree = int(p.rows()) - 1;
    if (degree == 0)
        return Vec::Zero(1);
    Vec d(degree);
    for (int i = 0; i < degree; ++i)
        d(i) = (degree - i) * p(i);
    return d;
}

struct Sample
{
    double x = 0.0, value = 0.0, gradient = 0.0;
    bool valueValid = false, gradientValid = false;
};

inline void minimize(const Vec& p, double xmin, double xmax, double* ox, double* ov)
{
    *ox = (xmin + xmax) / 2.0;
    *ov = evaluate(p, *ox);
    const double vmin = evaluate(p, xmin);
    if (vmin < *ov)
    {
        *ov = vmin;
        *ox = xmin;
    }
    const double vmax = evaluate(p, xmax);
    if (vmax < *ov)
    {
        *ov = vmax;
        *ox = xmax;
    }
    if (p.rows() <= 2)
        return;
    const Vec d = differentiate(p);
    Vec r;
    if (!roots(d, &r))
        return;
    for (int i = 0; i < r.rows(); ++i)
    {
        const double root = r(i);
        if (root < xmin || root > xmax)
            continue;
        const double v = evaluate(p, root);
        if (v < *ov)
        {
            *ov = v;
            *ox = root;
        }
    }
}

inline Vec interpolate(const std::vector<Sample>& s)
{
    int n = 0;
    for (const auto& x : s)
        n += int(x.valueValid) + int(x.gradientValid);
    const int degree = n - 1;
    Mat lhs = Mat::Zero(n, n);
    Vec rhs = Vec::Zero(n);
    int row = 0;
    for (const auto& x : s)
    {
        if (x.valueValid)
        {
            for (int j = 0; j <= degree; ++j)
                lhs(row, j) = std::pow(x.x, degree - j);
            rhs(row) = x.value;
            ++row;
        }
        if (x.gradientValid)
        {
            for (int j = 0; j < degree; ++j)
                lhs(row, j) = (degree - j) * std::pow(x.x, degree - j - 1);
            rhs(row) = x.gradient;
            ++row;
        }
    }
    Eigen::FullPivLU<Mat> lu(lhs);
    return lu.setThreshold(0.0).solve(rhs);
}

// LineSearch::InterpolatingPolynomialMinimizingStepSize, CUBIC
inline double stepSize(const Sample& lower, const Sample& previous, const Sample& current, double minStep, double maxStep)
{
    if (!current.valueValid)
        return std::min(std::max(current.x * 0.5, minStep), maxStep);
    std::vector<Sample> s{lower, current};
    if (previous.valueValid)
        s.push_back(previous);
    const Vec p = interpolate(s);
    double ox = 0.0, ov = 0.0;
    minimize(p, minStep, maxStep, &ox, &ov);
    for (const auto& x : s)   // MinimizeInterpolatingPolynomial: the samples themselves
    {
        if (x.x < minStep || x.x > maxStep)
            continue;
        const double v = evaluate(p, x.x);
        if (v < ov)
        {
            ox = x.x;
            ov = v;
        }
    }
    return ox;
}
}  // namespace poly

// One iteration, as Ceres' IterationSummary reports it.
struct Iteration
{
    int iteration = 0;
    double cost = 0.0;              // reduced program's cost plus the fixed cost, as Ceres reports
    double costChange = 0.0;
    double gradientMaxNorm = 0.0;
    double stepNorm = 0.0;
    double relativeDecrease = 0.0;
    double radius = 0.0;
    bool stepIsValid = false;
    bool stepIsSuccessful = false;
    // ours only: the line search along this iteration's step (samples taken, gradients taken, the step
    // size it returned, whether it succeeded)
    int lsSamples = 0, lsGradients = 0;
    double lsStep = 1.0;
    bool lsOk = false;
};

// Where our time goes, per solve (the shadow line prints it).
struct Times
{
    double setup = 0.0, evalJ = 0.0, evalCost = 0.0, evalGrad = 0.0, elim = 0.0, schur = 0.0, factor = 0.0, back = 0.0;
    // the passes around them (step 4a): column norms, Jacobi scaling, the model cost change, Plus, and the
    // state copies
    double norms = 0.0, scale = 0.0, model = 0.0, plus = 0.0, state = 0.0;
    int nJ = 0, nCost = 0, nGrad = 0, nSolves = 0, threads = 1;
};

struct Result
{
    bool ok = false;               // false: the problem is outside the solver's scope, or it failed
    std::string why;
    std::vector<Iteration> iterations;
    double initialCost = 0.0, finalCost = 0.0, fixedCost = 0.0;
    std::string termination;
    int eBlocks = 0, fBlocks = 0, fColumns = 0, rows = 0;
    size_t sBlocks = 0, sValues = 0, jValues = 0;   // the reduced camera system's blocks and values; the Jacobian's
    int groups = 0, parts = 0;
    bool denseQR = false;          // no eliminated block: Ceres' DENSE_QR in place of DENSE_SCHUR
    size_t residuals = 0;          // scalar residuals of the reduced program
    bool constrained = false;
    double seconds = 0.0;
    Times time;
    std::uint64_t digest = 0;      // FNV-1a over the final state's bytes: the thread-count independence check
    std::uint64_t digestIn = 0;    // the same over the starting state (CHESHIRE_BA_DIGEST)
};

// FNV-1a over a state vector's bytes
inline std::uint64_t digestOf(const std::vector<double>& x)
{
    std::uint64_t h = 0xcbf29ce484222325ull;
    const unsigned char* b = reinterpret_cast<const unsigned char*>(x.data());
    for (size_t i = 0; i < x.size() * sizeof(double); ++i)
        h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

// CHESHIRE_BA_DIGEST=1: one line per solve with the digests of its starting and final states
// CHESHIRE_BA_PROFILE: also one line per solve with the solver's time per phase (step 4a)
inline bool profileEnabled()
{
    static const bool on = ::cheshire::env::flag("CHESHIRE_BA_PROFILE");
    return on;
}

inline bool digestEnabled()
{
    static const bool on = ::cheshire::env::flag("CHESHIRE_BA_DIGEST");
    return on;
}

// CHESHIRE_BA_DIRECT (step 7e, direct.inc): 1 the solve built from the SfM data (the default), 0 off, check
// both on every solve and compare. Returns 0, 1 or 2.
inline int directMode()
{
    static const int mode = [] {
        const std::string s = ::cheshire::env::text("CHESHIRE_BA_DIRECT", "1");
        if (s == "check")
            return 2;
        return ::cheshire::env::flag("CHESHIRE_BA_DIRECT", true) ? 1 : 0;
    }();
    return mode;
}

// the last solve's account on this thread, kept in the direct check mode (it compares two solves)
inline Result& lastResult()
{
    thread_local Result r;
    return r;
}

// Ceres' corrector (corrector.cc): scale by sqrt(rho'), and the rank-one curvature correction in the
// inlier region only when rho'' > 0 (never for Huber).
struct Corrector
{
    double sqrtRho1 = 1.0, residualScaling = 1.0, alphaSqNorm = 0.0;
    Corrector(double sqNorm, const double rho[3])
    {
        sqrtRho1 = std::sqrt(rho[1]);
        if (sqNorm == 0.0 || rho[2] <= 0.0)
        {
            residualScaling = sqrtRho1;
            alphaSqNorm = 0.0;
            return;
        }
        const double D = 1.0 + 2.0 * sqNorm * rho[2] / rho[1];
        const double alpha = 1.0 - std::sqrt(D);
        residualScaling = sqrtRho1 / (1 - alpha);
        alphaSqNorm = alpha / sqNorm;
    }
    void correctResiduals(int n, double* r) const
    {
        for (int i = 0; i < n; ++i)
            r[i] *= residualScaling;
    }
    // J := sqrt(rho1) * (J - alpha/|r|^2 * r r^T J), row-major n x c (corrector.cc CorrectJacobian)
    void correctJacobian(int n, int c, const double* r, double* J) const
    {
        if (alphaSqNorm == 0.0)
        {
            for (int i = 0; i < n * c; ++i)
                J[i] *= sqrtRho1;
            return;
        }
        for (int j = 0; j < c; ++j)
        {
            double rtj = 0.0;
            for (int i = 0; i < n; ++i)
                rtj += r[i] * J[i * c + j];
            for (int i = 0; i < n; ++i)
                J[i * c + j] = sqrtRho1 * (J[i * c + j] - alphaSqNorm * r[i] * rtj);
        }
    }
};

// ---- small dense kernels -------------------------------------------------------------------------
// out (t1 x t2) += A^T B, A n x t1 and B n x t2, all row-major. Per entry the sum over i in order from
// zero, then added to out; the column loop innermost, so fixed sizes vectorise.
template <int T1, int T2>
inline void addAtBFixed(int n, const double* A, const double* B, double* out)
{
    for (int a = 0; a < T1; ++a)
    {
        double acc[T2];
        for (int b = 0; b < T2; ++b)
            acc[b] = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double s = A[i * T1 + a];
            for (int b = 0; b < T2; ++b)
                acc[b] += s * B[i * T2 + b];
        }
        for (int b = 0; b < T2; ++b)
            out[a * T2 + b] += acc[b];
    }
}

inline void addAtBDynamic(int n, int t1, int t2, const double* A, const double* B, double* out)
{
    for (int a = 0; a < t1; ++a)
    {
        double acc[64];
        for (int b = 0; b < t2; ++b)
            acc[b] = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double s = A[i * t1 + a];
            for (int b = 0; b < t2; ++b)
                acc[b] += s * B[i * t2 + b];
        }
        for (int b = 0; b < t2; ++b)
            out[a * t2 + b] += acc[b];
    }
}

inline void addAtB(int n, int t1, int t2, const double* A, const double* B, double* out)
{
    if (t1 == 6 && t2 == 6)
        addAtBFixed<6, 6>(n, A, B, out);
    else if (t1 == 3 && t2 == 6)
        addAtBFixed<3, 6>(n, A, B, out);
    else
        addAtBDynamic(n, t1, t2, A, B, out);
}

// out (t1 x t2) -= U M, U t1 x 3 and M 3 x t2 row-major, per entry the three products summed in order
template <int T1, int T2>
inline void subU3MFixed(const double* U, const double* M, double* out)
{
    for (int a = 0; a < T1; ++a)
    {
        const double u0 = U[a * 3 + 0], u1 = U[a * 3 + 1], u2 = U[a * 3 + 2];
        for (int b = 0; b < T2; ++b)
            out[a * T2 + b] -= u0 * M[0 * T2 + b] + u1 * M[1 * T2 + b] + u2 * M[2 * T2 + b];
    }
}

inline void subU3M(int t1, int t2, const double* U, const double* M, double* out)
{
    if (t1 == 6 && t2 == 6)
    {
        subU3MFixed<6, 6>(U, M, out);
        return;
    }
    for (int a = 0; a < t1; ++a)
    {
        const double u0 = U[a * 3 + 0], u1 = U[a * 3 + 1], u2 = U[a * 3 + 2];
        for (int b = 0; b < t2; ++b)
            out[a * t2 + b] -= u0 * M[0 * t2 + b] + u1 * M[1 * t2 + b] + u2 * M[2 * t2 + b];
    }
}

inline bool shadowEnabled();

// A pointer -> int map for the setup (open addressing, Fibonacci hashing): Ceres' own lookups are std::map.
class PtrIndex
{
  public:
    void reserve(size_t n)
    {
        size_t cap = 16;
        int bits = 4;
        while (cap < 2 * n + 16)
        {
            cap <<= 1;
            ++bits;
        }
        _keys.assign(cap, nullptr);
        _vals.assign(cap, -1);
        _shift = 64 - bits;
        _mask = cap - 1;
    }
    void set(const double* p, int v)
    {
        size_t i = slot(p);
        while (_keys[i] && _keys[i] != p)
            i = (i + 1) & _mask;
        _keys[i] = p;
        _vals[i] = v;
    }
    // -1 when absent
    int get(const double* p) const
    {
        size_t i = slot(p);
        while (_keys[i])
        {
            if (_keys[i] == p)
                return _vals[i];
            i = (i + 1) & _mask;
        }
        return -1;
    }

  private:
    size_t slot(const double* p) const
    {
        return size_t((std::uint64_t(reinterpret_cast<std::uintptr_t>(p)) * 0x9E3779B97F4A7C15ull) >> _shift);
    }
    std::vector<const double*> _keys;
    std::vector<int> _vals;
    int _shift = 60;
    size_t _mask = 15;
};

// An uninitialised double buffer (the Jacobian runs to hundreds of MB; every entry is written before use).
struct Buffer
{
    std::unique_ptr<double[]> p;
    size_t n = 0;
    void reset(size_t size)
    {
        p.reset(size ? new double[size] : nullptr);
        n = size;
    }
    double* data() { return p.get(); }
    const double* data() const { return p.get(); }
};

// ---- what the solver solves ------------------------------------------------------------------------
// A Source lays a problem out as Ceres' preprocessor sees it: its parameter blocks as candidates in
// Ceres' order (the first elimination group, then the others, each in the order Ceres keeps it), with
// their properties, and its residual blocks in the program's order; it evaluates a residual block.
// ProblemSource reads a ceres::Problem (steps 1-3); the direct source builds from the SfM data (3b).
struct BlockInfo
{
    double* ptr = nullptr;
    int size = 0, tsize = 0;
    const ceres::Manifold* manifold = nullptr;
    bool constant = false;                 // or missing: not a free block
    bool eliminate = false;                // in the first elimination group
    std::vector<double> lower, upper;      // ambient bounds, empty when the block has none
};

class Source
{
  public:
    virtual ~Source() = default;
    virtual bool blocks(std::vector<BlockInfo>* out, int threads, std::string* why) const = 0;
    virtual int numRows() const = 0;
    virtual int rowNumParams(int i) const = 0;
    virtual void rowParams(int i, double** out) const = 0;
    virtual int rowNumResiduals(int i) const = 0;
    virtual const ceres::LossFunction* rowLoss(int i) const = 0;
    // ScaledLoss's factor on top of rowLoss (ScaledLoss::Evaluate's arithmetic); 1 for none
    virtual double rowLossScale(int) const { return 1.0; }
    virtual bool evaluate(int i, double const* const* params, double* residuals, double** jacobians) const = 0;
    // what Solver::Summary reports of the problem as given
    virtual void counts(int* parameterBlocks, int* parameters, int* residualBlocks, int* residuals) const = 0;
};

// rho of a row's loss, with ScaledLoss's arithmetic for a scale; false when the row has no loss at all
inline bool rowRho(const ceres::LossFunction* loss, double scale, double s, double rho[3])
{
    if (scale == 1.0)
    {
        if (!loss)
            return false;
        loss->Evaluate(s, rho);
        return true;
    }
    if (!loss)
    {
        rho[0] = scale * s;
        rho[1] = scale;
        rho[2] = 0.0;
        return true;
    }
    loss->Evaluate(s, rho);
    rho[0] *= scale;
    rho[1] *= scale;
    rho[2] *= scale;
    return true;
}

// A ceres::Problem as the solver's source: the candidates from the elimination ordering (the groups in
// order, a group's blocks in its set's order, then the blocks the ordering does not name), the residual
// blocks in the program's order. The per-block queries are std::map lookups in Ceres; they are const,
// so in parallel.
class ProblemSource final : public Source
{
  public:
    ProblemSource(ceres::Problem& problem, const ceres::ParameterBlockOrdering& ordering)
      : _problem(problem), _ordering(ordering)
    {
        _problem.GetResidualBlocks(&_rids);
        _cost.resize(_rids.size());
        _loss.resize(_rids.size());
        for (size_t i = 0; i < _rids.size(); ++i)
        {
            _cost[i] = _problem.GetCostFunctionForResidualBlock(_rids[i]);
            _loss[i] = _problem.GetLossFunctionForResidualBlock(_rids[i]);
        }
    }

    bool blocks(std::vector<BlockInfo>* out, int threads, std::string* why) const override
    {
        const auto& groups = _ordering.group_to_elements();
        if (groups.empty())
        {
            *why = "no ordering";
            return false;
        }
        const int eGroup = groups.begin()->first;
        std::vector<double*> cand;
        std::vector<char> candE;
        cand.reserve(size_t(_ordering.NumElements()));
        candE.reserve(size_t(_ordering.NumElements()));
        for (const auto& [group, elements] : groups)
            for (double* p : elements)
            {
                cand.push_back(p);
                candE.push_back(char(group == eGroup));
            }
        if (_ordering.NumElements() != _problem.NumParameterBlocks())
        {
            // blocks the ordering does not name (Ceres puts them in the last group)
            PtrIndex member;
            member.reserve(cand.size());
            for (double* p : cand)
                member.set(p, 0);
            std::vector<double*> blocks;
            _problem.GetParameterBlocks(&blocks);
            for (double* p : blocks)
                if (member.get(p) < 0)
                {
                    cand.push_back(p);
                    candE.push_back(0);
                }
        }
        const int nc = int(cand.size());
        out->assign(static_cast<size_t>(nc), BlockInfo());
        const double big = std::numeric_limits<double>::max();
#pragma omp parallel for schedule(dynamic, 4096) num_threads(threads)
        for (int i = 0; i < nc; ++i)
        {
            double* p = cand[i];
            BlockInfo& b = (*out)[i];
            b.ptr = p;
            b.eliminate = candE[i] != 0;
            if (!_problem.HasParameterBlock(p) || _problem.IsParameterBlockConstant(p))
            {
                b.constant = true;
                continue;
            }
            b.size = _problem.ParameterBlockSize(p);
            b.tsize = _problem.ParameterBlockTangentSize(p);
            b.manifold = _problem.GetManifold(p);
            if (b.tsize == 0)
                continue;
            // bounds (Program::IsBoundsConstrained): -max / +max when a coordinate has none
            bool any = false;
            std::vector<double> lo(static_cast<size_t>(b.size)), hi(static_cast<size_t>(b.size));
            for (int k = 0; k < b.size; ++k)
            {
                lo[k] = _problem.GetParameterLowerBound(p, k);
                hi[k] = _problem.GetParameterUpperBound(p, k);
                any = any || lo[k] > -big || hi[k] < big;
            }
            if (any)
            {
                b.lower = std::move(lo);
                b.upper = std::move(hi);
            }
        }
        return true;
    }
    int numRows() const override { return int(_rids.size()); }
    int rowNumParams(int i) const override { return int(_cost[i]->parameter_block_sizes().size()); }
    void rowParams(int i, double** out) const override
    {
        thread_local std::vector<double*> tmp;
        _problem.GetParameterBlocksForResidualBlock(_rids[i], &tmp);
        std::copy(tmp.begin(), tmp.end(), out);
    }
    int rowNumResiduals(int i) const override { return _cost[i]->num_residuals(); }
    const ceres::LossFunction* rowLoss(int i) const override { return _loss[i]; }
    bool evaluate(int i, double const* const* params, double* residuals, double** jacobians) const override
    {
        return _cost[i]->Evaluate(params, residuals, jacobians);
    }
    void counts(int* parameterBlocks, int* parameters, int* residualBlocks, int* residuals) const override
    {
        *parameterBlocks = _problem.NumParameterBlocks();
        *parameters = _problem.NumParameters();
        *residualBlocks = _problem.NumResidualBlocks();
        *residuals = _problem.NumResiduals();
    }

  private:
    ceres::Problem& _problem;
    const ceres::ParameterBlockOrdering& _ordering;
    std::vector<ceres::ResidualBlockId> _rids;
    std::vector<const ceres::CostFunction*> _cost;
    std::vector<const ceres::LossFunction*> _loss;
};

class Solver
{
  public:
    // prepare(): what Ceres' EvaluationCallback does before a new evaluation point (AliceVision pushes
    // the intrinsics and distortion blocks into its camera objects, which the cost functions read).
    Solver(const Source& source, const Options& options, std::function<void()> prepare)
      : _src(source), _opt(options), _prepare(std::move(prepare)), _threads(std::max(1, options.threads))
    {
        // CHESHIRE_BA_OWN_THREADS: this solver's threads alone, Ceres' unchanged (the independence check)
        static const long own = static_cast<long>(::cheshire::env::integer("CHESHIRE_BA_OWN_THREADS", 0));
        if (own > 0)
            _threads = int(own);
    }

    // Solve in place: the problem's parameter blocks end at the solution (the caller restores them in
    // shadow mode).
    Result solve()
    {
        const auto t0 = Clock::now();
        Result res;
        const bool ok = setup(&res.why);
        _times.setup = since(t0);
        if (ok)
        {
            res.eBlocks = int(_e.size());
            res.fBlocks = int(_f.size());
            res.fColumns = _nf;
            res.rows = int(_rows.size());
            res.residuals = _nr;
            res.denseQR = _qr;
            res.constrained = _constrained;
            res.fixedCost = _fixedCost;
            res.sBlocks = _sCol.size();
            res.sValues = _sv.size();
            res.jValues = _J.n;
            res.groups = int(_groups.size()) - 1;
            res.parts = nParts();
            minimize(&res);
            if (res.ok && (shadowEnabled() || digestEnabled() || directMode() == 2))
            {
                res.digest = digestOf(stateVector());
                res.digestIn = digestOf(_x0);
            }
        }
        res.seconds = since(t0);
        _times.threads = _threads;
        res.time = _times;
        return res;
    }

  private:
    using Clock = std::chrono::steady_clock;
    static double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
    static constexpr int kMaxT = 32;         // the largest F tangent size handled (poses 6, intrinsics a few)

    struct PBlock
    {
        double* ptr = nullptr;
        int size = 0, tsize = 0;
        const ceres::Manifold* manifold = nullptr;
        int col = 0;                         // tangent column in the full vector: E blocks first, then F
        size_t x = 0;                        // ambient offset in the state vector, same order
        std::vector<double> plusJacobian;    // size x tsize, row-major, at the current state
        std::vector<double> lower, upper;    // ambient bounds, empty when the block has none
    };
    // A residual block. Its parameters are _pp/_pv/_pj[p .. p + np): the pointer, the free block (an E
    // index >= 0, an F index f as -2 - f, or -1 when constant) and the tangent Jacobian block's offset
    // from j (or -1).
    struct Row
    {
        int src = 0;                         // the source's residual block
        const ceres::LossFunction* loss = nullptr;
        double lossScale = 1.0;
        int nres = 0;
        int np = 0;
        size_t p = 0;
        size_t r = 0;                        // residuals in _r
        size_t j = 0;                        // Jacobian blocks in _J, contiguous from here
        int jSize = 0;
    };
    enum class Mode
    {
        Full,       // _r (corrected), _J (tangent, corrected, unscaled) and _grad at the current state
        Cost,       // the cost alone (a candidate point); _r and _J untouched
        Gradient    // the cost and the unscaled tangent gradient into *grad (the line search); _r, _J untouched
    };
    const Source& _src;
    Options _opt;
    std::function<void()> _prepare;
    int _threads = 1;
    mutable Times _times;

    std::vector<PBlock> _e, _f;              // free blocks, in Ceres' order
    int _ne = 0, _nf = 0, _n = 0;            // tangent sizes: E, F, both
    size_t _nx = 0;                          // ambient size of the state
    std::vector<int> _mb;                    // blocks with a manifold (E index, or -2 - F index)
    std::vector<double*> _pp;
    std::vector<int> _pv, _pj;
    std::vector<Row> _rows;                  // rows with an E block first, grouped by it, then the rest
    std::vector<int> _chunkStart;            // per E block its first row; [nE] the first row without one; [nE + 1] the end
    std::vector<int> _parts;                 // evaluation partitions (row boundaries, whole chunks)
    double _fixedCost = 0.0;
    std::vector<double> _r, _scale, _diag, _grad;
    Buffer _J;
    size_t _nr = 0;
    bool _constrained = false;               // any bound on a free block: Ceres' is_constrained
    std::vector<double> _x0;                 // the starting point (Ceres puts it back on FAILURE)
    bool _qr = false;                        // no eliminated block: Ceres' DENSE_QR (LinearSolverForZeroEBlocks)
    std::vector<double> _partF;              // per partition, an F-sized accumulator (gradient, column norms)

    // Schur elimination: per chunk, its F blocks (sorted) with their E^T F blocks in _m
    std::vector<size_t> _cfStart, _cfM;
    std::vector<int> _cf;
    Buffer _m;
    std::vector<double> _inv, _ge;           // per chunk: (E^T E + D_e^2)^-1 (3x3) and E^T b
    // per F block, the chunks (c >= 0) and the rows without an E block (-1 - row) that touch it, in order
    std::vector<size_t> _fcStart;
    std::vector<int> _fc, _fcPos;
    // S, upper block triangle: row f1 holds the blocks (f1, f2 >= f1), each t1 x t2 row-major in _sv
    std::vector<size_t> _sStart, _sOff, _sRow, _sRowSize;
    std::vector<int> _sCol;
    std::vector<double> _sv, _rhs;
    // the assembly: fixed groups of whole chunks (the last also takes the rows without an E block), each
    // adding into its own copy of (S, rhs); the copies are summed in group order
    std::vector<int> _groups;                // chunk boundaries, [G] = nE
    std::vector<double> _sg;                 // G copies of (S, rhs), _sgStride apart
    size_t _sgStride = 0;
    std::vector<int> _pairIdx;               // nF x nF: the block (f1 <= f2) as an index into _sCol/_sOff, when nF is small
    // the factorisation
    Eigen::MatrixXd _A;                      // DENSE_SCHUR: lower triangle
    Eigen::SparseMatrix<double> _S;          // SPARSE_SCHUR: upper triangle, column-major, pattern fixed at setup
    std::vector<size_t> _cscSrc;
    // the camera blocks are already in Ceres' AMD order (setup), so the factorisation keeps it, as Ceres'
    // does (AreJacobianColumnsOrdered: NATURAL for SPARSE_SCHUR with EIGEN_SPARSE)
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper, Eigen::NaturalOrdering<int>> _ldlt;
    bool _analyzed = false;

    PBlock& block(int v) { return v >= 0 ? _e[v] : _f[-2 - v]; }
    const PBlock& block(int v) const { return v >= 0 ? _e[v] : _f[-2 - v]; }
    int nParts() const { return int(_parts.size()) - 1; }

    // ---- setup -------------------------------------------------------------------------------------
    // The reduced program Ceres' preprocessor builds, in its order: the free parameter blocks that some
    // residual block uses (the E group first, then the rest by group, within a group by address; for
    // SPARSE_SCHUR the camera blocks then reordered as ReorderSchurComplementColumnsUsingEigen does), and
    // the residual blocks on at least one free block, grouped by their E block.
    bool setup(std::string* why)
    {
        std::vector<BlockInfo> info;
        if (!_src.blocks(&info, _threads, why))
            return false;
        const int nc = int(info.size());
        // provisional indices, in candidate order
        std::vector<PBlock> tE, tF;
        PtrIndex index;
        index.reserve(info.size());
        for (int i = 0; i < nc; ++i)
        {
            BlockInfo& bi = info[i];
            if (bi.constant || bi.tsize == 0)
                continue;
            PBlock b;
            b.ptr = bi.ptr;
            b.size = bi.size;
            b.tsize = bi.tsize;
            b.manifold = bi.manifold;
            b.lower = std::move(bi.lower);
            b.upper = std::move(bi.upper);
            if (bi.eliminate)
            {
                index.set(b.ptr, int(tE.size()));
                tE.push_back(std::move(b));
            }
            else
            {
                index.set(b.ptr, -2 - int(tF.size()));
                tF.push_back(std::move(b));
            }
        }
        info.clear();

        // residual blocks
        const int nR = _src.numRows();
        std::vector<size_t> pStart(size_t(nR) + 1, 0);
        for (int i = 0; i < nR; ++i)
            pStart[i + 1] = pStart[i] + size_t(_src.rowNumParams(i));
        _pp.resize(pStart[nR]);
        _pv.resize(pStart[nR]);
        _pj.assign(pStart[nR], -1);
        std::vector<int> rowE(size_t(nR), -1);
        std::vector<char> rowFree(size_t(nR), 0), rowBad(size_t(nR), 0);
#pragma omp parallel num_threads(_threads)
        {
#pragma omp for schedule(dynamic, 4096)
            for (int i = 0; i < nR; ++i)
            {
                _src.rowParams(i, _pp.data() + pStart[i]);
                for (size_t p = pStart[i]; p < pStart[i + 1]; ++p)
                {
                    const int v = index.get(_pp[p]);
                    _pv[p] = v;
                    if (v >= 0)
                    {
                        if (rowE[i] >= 0)
                            rowBad[i] = 1;
                        rowE[i] = v;
                    }
                    if (v != -1)
                        rowFree[i] = 1;
                }
            }
        }
        for (int i = 0; i < nR; ++i)
            if (rowBad[i])
            {
                *why = "a residual block touches two eliminated blocks";
                return false;
            }
        // residual blocks on constant parameters only: Ceres' preprocessor removes them and reports their
        // cost as fixed_cost
        for (int i = 0; i < nR; ++i)
        {
            if (rowFree[i])
                continue;
            std::vector<double> r(static_cast<size_t>(_src.rowNumResiduals(i)));
            if (!_src.evaluate(i, _pp.data() + pStart[i], r.data(), nullptr))
            {
                *why = "a fixed residual block failed to evaluate";
                return false;
            }
            double s = 0.0;
            for (double v : r)
                s += v * v;
            double rho[3];
            if (rowRho(_src.rowLoss(i), _src.rowLossScale(i), s, rho))
                _fixedCost += 0.5 * rho[0];
            else
                _fixedCost += 0.5 * s;
        }
        // free blocks no residual block uses: Ceres' reduced program drops them
        std::vector<char> usedE(tE.size(), 0), usedF(tF.size(), 0);
        for (int i = 0; i < nR; ++i)
        {
            if (!rowFree[i])
                continue;
            for (size_t p = pStart[i]; p < pStart[i + 1]; ++p)
            {
                if (_pv[p] >= 0)
                    usedE[_pv[p]] = 1;
                else if (_pv[p] <= -2)
                    usedF[-2 - _pv[p]] = 1;
            }
        }
        std::vector<int> mapE(tE.size(), -1), mapF(tF.size(), -1);
        for (size_t i = 0; i < tE.size(); ++i)
            if (usedE[i])
            {
                if (tE[i].tsize != 3)
                {
                    *why = "an eliminated block is not of size 3";
                    return false;
                }
                mapE[i] = int(_e.size());
                _e.push_back(std::move(tE[i]));
            }
        for (size_t i = 0; i < tF.size(); ++i)
            if (usedF[i])
            {
                if (tF[i].tsize > kMaxT)
                {
                    *why = "a camera block's tangent size is above " + std::to_string(kMaxT);
                    return false;
                }
                mapF[i] = int(_f.size());
                _f.push_back(std::move(tF[i]));
            }
        tE.clear();
        tF.clear();
        {
            const std::ptrdiff_t np = std::ptrdiff_t(_pv.size());
#pragma omp parallel for schedule(static) num_threads(_threads)
            for (std::ptrdiff_t p = 0; p < np; ++p)
            {
                const int v = _pv[p];
                if (v >= 0)
                    _pv[p] = mapE[v];
                else if (v <= -2)
                    _pv[p] = -2 - mapF[-2 - v];
            }
            for (int i = 0; i < nR; ++i)
                if (rowE[i] >= 0)
                    rowE[i] = mapE[rowE[i]];
        }
        // No eliminated block left: Ceres gives up the Schur solver (LinearSolverForZeroEBlocks), and without
        // the Schur reordering its program keeps the order the blocks were added in. DENSE_SCHUR becomes
        // DENSE_QR, done here; SPARSE_SCHUR becomes SPARSE_NORMAL_CHOLESKY, left to Ceres.
        // The column order matters to the QR's rounding. The public API does not give the program order
        // (GetParameterBlocks is in address order, which changes from run to run: it once made two False
        // Door runs part at a new camera's first view). The ordering groups do: BundleAdjustmentCeres adds
        // the poses, then each camera's intrinsics and distortion, and numbers the groups the same way, so
        // for the resection refinements (one pose, one camera) _f is already Ceres' order; elsewhere it is
        // at least the same order every run.
        if (_e.empty())
        {
            if (_opt.sparse)
            {
                *why = "no eliminated block (Ceres' SPARSE_NORMAL_CHOLESKY)";
                return false;
            }
            _qr = true;
        }

        // rows grouped by E block in residual order (a stable counting sort), then the rows without one
        const int nE = int(_e.size());
        _chunkStart.assign(size_t(nE) + 2, 0);
        for (int i = 0; i < nR; ++i)
            if (rowFree[i])
                ++_chunkStart[size_t(rowE[i] >= 0 ? rowE[i] : nE) + 1];
        for (int c = 0; c <= nE; ++c)
            _chunkStart[c + 1] += _chunkStart[c];
        const int nRows = _chunkStart[nE + 1];
        std::vector<int> order(static_cast<size_t>(nRows)), next(_chunkStart.begin(), _chunkStart.end() - 1);
        for (int i = 0; i < nR; ++i)
            if (rowFree[i])
                order[next[rowE[i] >= 0 ? rowE[i] : nE]++] = i;
        _rows.resize(size_t(nRows));
        size_t nj = 0;
        for (int ri = 0; ri < nRows; ++ri)
        {
            const int i = order[ri];
            Row& row = _rows[ri];
            row.src = i;
            row.loss = _src.rowLoss(i);
            row.lossScale = _src.rowLossScale(i);
            row.nres = _src.rowNumResiduals(i);
            row.p = pStart[i];
            row.np = int(pStart[i + 1] - pStart[i]);
            row.r = _nr;
            _nr += size_t(row.nres);
            row.j = nj;
            int jo = 0;
            for (int k = 0; k < row.np; ++k)
            {
                const int v = _pv[row.p + k];
                if (v == -1)
                    continue;
                _pj[row.p + k] = jo;
                jo += row.nres * block(v).tsize;
            }
            row.jSize = jo;
            nj += size_t(jo);
        }
        _r.assign(_nr, 0.0);
        _J.reset(nj);

        buildStructure();
        if (_opt.sparse && _f.size() > 1)
        {
            // ReorderSchurComplementColumnsUsingEigen: AMD on the block pattern of the Schur complement,
            // F^T F - F^T E E^T F (both triangles, the diagonal included); the camera block i of the new
            // order is block perm.indices()[i] of the old. The factorisation then keeps the natural order.
            const int nF = int(_f.size());
            std::vector<Eigen::Triplet<int>> trip;
            for (int f1 = 0; f1 < nF; ++f1)
                for (size_t s = _sStart[f1]; s < _sStart[f1 + 1]; ++s)
                {
                    trip.emplace_back(f1, _sCol[s], 1);
                    if (_sCol[s] != f1)
                        trip.emplace_back(_sCol[s], f1, 1);
                }
            Eigen::SparseMatrix<int> pattern(nF, nF);
            pattern.setFromTriplets(trip.begin(), trip.end());
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm;
            Eigen::AMDOrdering<int> amd;
            amd(pattern, perm);
            std::vector<int> inv(size_t(nF), -1);
            bool identity = true;
            for (int i = 0; i < nF; ++i)
            {
                inv[perm.indices()[i]] = i;
                identity = identity && perm.indices()[i] == i;
            }
            if (!identity)
            {
                std::vector<PBlock> nf(static_cast<size_t>(nF));
                for (int i = 0; i < nF; ++i)
                    nf[i] = std::move(_f[perm.indices()[i]]);
                _f.swap(nf);
                const std::ptrdiff_t np = std::ptrdiff_t(_pv.size());
#pragma omp parallel for schedule(static) num_threads(_threads)
                for (std::ptrdiff_t p = 0; p < np; ++p)
                    if (_pv[p] <= -2)
                        _pv[p] = -2 - inv[-2 - _pv[p]];
                buildStructure();
            }
        }

        // columns and state offsets, E then F; bounds; manifolds
        for (auto& b : _e)
        {
            b.col = _ne;
            _ne += 3;
        }
        for (auto& b : _f)
        {
            b.col = _ne + _nf;
            _nf += b.tsize;
        }
        _n = _ne + _nf;
        for (int pass = 0; pass < 2; ++pass)
            for (size_t i = 0; i < (pass == 0 ? _e : _f).size(); ++i)
            {
                PBlock& b = pass == 0 ? _e[i] : _f[i];
                b.x = _nx;
                _nx += size_t(b.size);
                if (!b.lower.empty() || !b.upper.empty())
                    _constrained = true;
                if (b.manifold)
                {
                    b.plusJacobian.assign(size_t(b.size) * b.tsize, 0.0);
                    _mb.push_back(pass == 0 ? int(i) : -2 - int(i));
                }
            }
        _rhs.assign(size_t(_nf), 0.0);
        _inv.assign(size_t(nE) * 9, 0.0);
        _ge.assign(size_t(nE) * 3, 0.0);

        // evaluation partitions: whole chunks, then the rows without an E block, about target rows each; the
        // boundaries depend on the problem only, so every reduction over them is thread-count independent
        const int target = std::max(512, nRows / 256 + 1);
        _parts.assign(1, 0);
        for (int c = 0, acc = 0; c < nE; ++c)
        {
            acc += _chunkStart[c + 1] - _chunkStart[c];
            if (acc >= target)
            {
                _parts.push_back(_chunkStart[c + 1]);
                acc = 0;
            }
        }
        if (_parts.back() != _chunkStart[nE])
            _parts.push_back(_chunkStart[nE]);
        for (int ri = _chunkStart[nE]; ri < nRows;)
        {
            ri = std::min(nRows, ri + target);
            _parts.push_back(ri);
        }
        _partF.assign(size_t(nParts()) * size_t(_nf), 0.0);

        const int nF = int(_f.size());
        if (_nf > 0 && !_opt.sparse)
            _A.setZero(_nf, _nf);
        if (_nf > 0 && _opt.sparse)
        {
            // the CSC pattern of the upper triangle, fixed for the solve: column j of block f2 takes the rows
            // of every f1 <= f2 with a block (f1, f2), only a <= b inside a diagonal block (Ceres' matrix
            // also has the rest of its diagonal blocks, which the up-looking factorisation skips)
            std::vector<std::vector<std::pair<int, size_t>>> colBlocks(static_cast<size_t>(nF));
            for (int f1 = 0; f1 < nF; ++f1)
                for (size_t s = _sStart[f1]; s < _sStart[f1 + 1]; ++s)
                    colBlocks[_sCol[s]].emplace_back(f1, _sOff[s]);
            std::vector<int> outer(size_t(_nf) + 1, 0);
            size_t nnz = 0;
            for (int f2 = 0; f2 < nF; ++f2)
            {
                const int t2 = _f[f2].tsize, c2 = _f[f2].col - _ne;
                for (int bb = 0; bb < t2; ++bb)
                {
                    for (const auto& [f1, off] : colBlocks[f2])
                        nnz += size_t(f1 == f2 ? bb + 1 : _f[f1].tsize);
                    if (nnz > size_t(std::numeric_limits<int>::max()))
                    {
                        *why = "the reduced camera system is too large for 32-bit indices";
                        return false;
                    }
                    outer[c2 + bb + 1] = int(nnz);
                }
            }
            std::vector<int> inner(nnz);
            _cscSrc.resize(nnz);
            size_t q = 0;
            for (int f2 = 0; f2 < nF; ++f2)
            {
                const int t2 = _f[f2].tsize;
                for (int bb = 0; bb < t2; ++bb)
                    for (const auto& [f1, off] : colBlocks[f2])
                    {
                        const int t1 = _f[f1].tsize, c1 = _f[f1].col - _ne;
                        const int na = f1 == f2 ? bb + 1 : t1;
                        for (int a = 0; a < na; ++a)
                        {
                            inner[q] = c1 + a;
                            _cscSrc[q] = off + size_t(a) * size_t(t2) + size_t(bb);
                            ++q;
                        }
                    }
            }
            _S.resize(_nf, _nf);
            _S.resizeNonZeros(Eigen::Index(nnz));
            std::copy(outer.begin(), outer.end(), _S.outerIndexPtr());
            std::copy(inner.begin(), inner.end(), _S.innerIndexPtr());
            std::fill(_S.valuePtr(), _S.valuePtr() + nnz, 0.0);
        }
        return true;
    }

    // The elimination structure from the rows and the current F order: per chunk its F blocks with their
    // E^T F storage, per F block its contributors, the block structure of S, and the assembly tasks.
    void buildStructure()
    {
        const int nE = int(_e.size()), nF = int(_f.size());
        const int nRows = int(_rows.size());
        std::vector<int> listLen(size_t(nE), 0);
        auto chunkF = [&](int c, std::vector<int>* out) {
            out->clear();
            for (int ri = _chunkStart[c]; ri < _chunkStart[c + 1]; ++ri)
            {
                const Row& row = _rows[ri];
                for (int k = 0; k < row.np; ++k)
                    if (_pv[row.p + k] <= -2)
                        out->push_back(-2 - _pv[row.p + k]);
            }
            std::sort(out->begin(), out->end());
            out->erase(std::unique(out->begin(), out->end()), out->end());
        };
#pragma omp parallel num_threads(_threads)
        {
            std::vector<int> tmp;
#pragma omp for schedule(dynamic, 1024)
            for (int c = 0; c < nE; ++c)
            {
                chunkF(c, &tmp);
                listLen[c] = int(tmp.size());
            }
        }
        _cfStart.assign(size_t(nE) + 1, 0);
        for (int c = 0; c < nE; ++c)
            _cfStart[c + 1] = _cfStart[c] + size_t(listLen[c]);
        _cf.assign(_cfStart[nE], 0);
#pragma omp parallel num_threads(_threads)
        {
            std::vector<int> tmp;
#pragma omp for schedule(dynamic, 1024)
            for (int c = 0; c < nE; ++c)
            {
                chunkF(c, &tmp);
                std::copy(tmp.begin(), tmp.end(), _cf.begin() + std::ptrdiff_t(_cfStart[c]));
            }
        }
        _cfM.resize(_cf.size());
        size_t mTotal = 0;
        for (size_t l = 0; l < _cf.size(); ++l)
        {
            _cfM[l] = mTotal;
            mTotal += 3 * size_t(_f[_cf[l]].tsize);
        }
        _m.reset(mTotal);

        // per F block, its contributors in order: the chunks, then the rows without an E block
        _fcStart.assign(size_t(nF) + 1, 0);
        for (size_t l = 0; l < _cf.size(); ++l)
            ++_fcStart[size_t(_cf[l]) + 1];
        for (int ri = _chunkStart[nE]; ri < nRows; ++ri)
        {
            const Row& row = _rows[ri];
            for (int k = 0; k < row.np; ++k)
                if (_pv[row.p + k] <= -2)
                    ++_fcStart[size_t(-2 - _pv[row.p + k]) + 1];
        }
        for (int f = 0; f < nF; ++f)
            _fcStart[f + 1] += _fcStart[f];
        _fc.assign(_fcStart[nF], 0);
        _fcPos.assign(_fcStart[nF], 0);
        {
            std::vector<size_t> at(_fcStart.begin(), _fcStart.end() - 1);
            for (int c = 0; c < nE; ++c)
                for (size_t l = _cfStart[c]; l < _cfStart[c + 1]; ++l)
                {
                    const size_t k = at[_cf[l]]++;
                    _fc[k] = c;
                    _fcPos[k] = int(l - _cfStart[c]);
                }
            for (int ri = _chunkStart[nE]; ri < nRows; ++ri)
            {
                const Row& row = _rows[ri];
                for (int k = 0; k < row.np; ++k)
                    if (_pv[row.p + k] <= -2)
                    {
                        const size_t q = at[size_t(-2 - _pv[row.p + k])]++;
                        _fc[q] = -1 - ri;
                        _fcPos[q] = -1;
                    }
            }
        }

        // the block structure of S (upper), as SparseSchurComplementSolver::InitStorage: the diagonal, the
        // pairs of F blocks sharing a chunk, the pairs sharing a row without an E block
        std::vector<std::vector<int>> rowsF(static_cast<size_t>(nF));
#pragma omp parallel num_threads(_threads)
        {
            std::vector<int> mark(size_t(nF), -1);
#pragma omp for schedule(dynamic, 16)
            for (int f1 = 0; f1 < nF; ++f1)
            {
                std::vector<int>& list = rowsF[f1];
                list.push_back(f1);
                mark[f1] = f1;
                for (size_t k = _fcStart[f1]; k < _fcStart[f1 + 1]; ++k)
                {
                    const int entry = _fc[k];
                    if (entry >= 0)
                    {
                        for (size_t l = _cfStart[entry] + size_t(_fcPos[k]); l < _cfStart[entry + 1]; ++l)
                            if (mark[_cf[l]] != f1)
                            {
                                mark[_cf[l]] = f1;
                                list.push_back(_cf[l]);
                            }
                    }
                    else
                    {
                        const Row& row = _rows[-1 - entry];
                        for (int q = 0; q < row.np; ++q)
                        {
                            const int v = _pv[row.p + q];
                            if (v > -2 || -2 - v < f1 || mark[-2 - v] == f1)
                                continue;
                            mark[-2 - v] = f1;
                            list.push_back(-2 - v);
                        }
                    }
                }
                std::sort(list.begin(), list.end());
            }
        }
        _sStart.assign(size_t(nF) + 1, 0);
        _sRow.assign(size_t(nF), 0);
        _sRowSize.assign(size_t(nF), 0);
        for (int f1 = 0; f1 < nF; ++f1)
            _sStart[f1 + 1] = _sStart[f1] + rowsF[f1].size();
        _sCol.assign(_sStart[nF], 0);
        _sOff.assign(_sStart[nF], 0);
        size_t sTotal = 0;
        for (int f1 = 0; f1 < nF; ++f1)
        {
            _sRow[f1] = sTotal;
            size_t s = _sStart[f1];
            for (int f2 : rowsF[f1])
            {
                _sCol[s] = f2;
                _sOff[s] = sTotal;
                sTotal += size_t(_f[f1].tsize) * size_t(_f[f2].tsize);
                ++s;
            }
            _sRowSize[f1] = sTotal - _sRow[f1];
        }
        _sv.assign(sTotal, 0.0);

        // the block lookup for the assembly: a table when the camera blocks are few, else a binary search
        _pairIdx.clear();
        if (nF <= 2048)
        {
            _pairIdx.assign(size_t(nF) * size_t(nF), -1);
            for (int f1 = 0; f1 < nF; ++f1)
                for (size_t s = _sStart[f1]; s < _sStart[f1 + 1]; ++s)
                    _pairIdx[size_t(f1) * size_t(nF) + size_t(_sCol[s])] = int(s);
        }
        // assembly groups: whole chunks of about equal rows; their number depends on the problem alone, and
        // their copies of S stay under 256 MB
        size_t rhsSize = 0;
        for (const auto& b : _f)
            rhsSize += size_t(b.tsize);
        _sgStride = sTotal + rhsSize;
        const int rowsE = _chunkStart[nE];
        int G = std::max(1, std::min(32, rowsE / 8192));
        G = int(std::max<size_t>(1, std::min<size_t>(size_t(G), (size_t(1) << 25) / std::max<size_t>(1, _sgStride))));
        _groups.assign(1, 0);
        for (int c = 0, acc = 0; c < nE && int(_groups.size()) < G; ++c)
        {
            acc += _chunkStart[c + 1] - _chunkStart[c];
            if (acc >= (rowsE + G - 1) / G)
            {
                _groups.push_back(c + 1);
                acc = 0;
            }
        }
        if (_groups.back() != nE || _groups.size() == 1)
            _groups.push_back(nE);
        _sg.assign((_groups.size() - 1) * _sgStride, 0.0);
    }

    // x: every free block's ambient values, E first then F (for the norms and the step)
    std::vector<double> stateVector() const
    {
        const auto t0 = Clock::now();
        std::vector<double> x(_nx);
        const int nb = int(_e.size() + _f.size()), nE = int(_e.size());
#pragma omp parallel for schedule(static, 4096) num_threads(_threads)
        for (int i = 0; i < nb; ++i)
        {
            const PBlock& b = i < nE ? _e[i] : _f[i - nE];
            std::copy(b.ptr, b.ptr + b.size, x.begin() + std::ptrdiff_t(b.x));
        }
        _times.state += since(t0);
        return x;
    }
    void setState(const std::vector<double>& x)
    {
        const auto t0 = Clock::now();
        const int nb = int(_e.size() + _f.size()), nE = int(_e.size());
#pragma omp parallel for schedule(static, 4096) num_threads(_threads)
        for (int i = 0; i < nb; ++i)
        {
            PBlock& b = i < nE ? _e[i] : _f[i - nE];
            std::copy(x.begin() + std::ptrdiff_t(b.x), x.begin() + std::ptrdiff_t(b.x + size_t(b.size)), b.ptr);
        }
        _times.state += since(t0);
    }

    // ---- vector passes on the threads ----------------------------------------------------------------
    static constexpr std::ptrdiff_t kBlock = 1 << 15;
    template <typename F>
    void pfor(size_t n, F f) const
    {
        const std::ptrdiff_t nn = std::ptrdiff_t(n);
#pragma omp parallel for schedule(static, 8192) num_threads(_threads)
        for (std::ptrdiff_t i = 0; i < nn; ++i)
            f(size_t(i));
    }
    // sum of f(i), i < n: per fixed block in order, then the blocks in order (thread-count independent)
    template <typename F>
    double psum(size_t n, F f) const
    {
        const int nb = int((std::ptrdiff_t(n) + kBlock - 1) / kBlock);
        std::vector<double> part(size_t(nb), 0.0);
#pragma omp parallel for schedule(static) num_threads(_threads)
        for (int b = 0; b < nb; ++b)
        {
            const size_t e = std::min(n, size_t(b + 1) * size_t(kBlock));
            double s = 0.0;
            for (size_t i = size_t(b) * size_t(kBlock); i < e; ++i)
                s += f(i);
            part[b] = s;
        }
        double s = 0.0;
        for (double v : part)
            s += v;
        return s;
    }
    template <typename F>
    double pmax(size_t n, F f) const
    {
        const int nb = int((std::ptrdiff_t(n) + kBlock - 1) / kBlock);
        std::vector<double> part(size_t(nb), 0.0);
#pragma omp parallel for schedule(static) num_threads(_threads)
        for (int b = 0; b < nb; ++b)
        {
            const size_t e = std::min(n, size_t(b + 1) * size_t(kBlock));
            double m = 0.0;
            for (size_t i = size_t(b) * size_t(kBlock); i < e; ++i)
                m = std::max(m, f(i));
            part[b] = m;
        }
        double m = 0.0;
        for (double v : part)
            m = std::max(m, v);
        return m;
    }

    // ---- evaluation ----------------------------------------------------------------------------------
    // One residual block, in Ceres' order (residual_block.cc): Evaluate, the manifold products, then the
    // loss correction of the tangent Jacobian (from the uncorrected residual), then of the residual.
    // jb == nullptr: the cost alone.
    bool evalRow(const Row& row, double* r, double* jb, std::vector<double>& amb, std::vector<double*>& jptr, double* cost) const
    {
        const size_t p = row.p;
        if (jb)
        {
            jptr.assign(size_t(row.np), nullptr);
            size_t ambTotal = 0;
            for (int k = 0; k < row.np; ++k)
                if (_pj[p + k] >= 0 && block(_pv[p + k]).manifold)
                    ambTotal += size_t(row.nres) * size_t(block(_pv[p + k]).size);
            amb.assign(ambTotal, 0.0);
            size_t ao = 0;
            for (int k = 0; k < row.np; ++k)
            {
                if (_pj[p + k] < 0)
                    continue;
                const PBlock& b = block(_pv[p + k]);
                if (b.manifold)
                {
                    jptr[k] = amb.data() + ao;
                    ao += size_t(row.nres) * size_t(b.size);
                }
                else
                    jptr[k] = jb + _pj[p + k];
            }
        }
        if (!_src.evaluate(row.src, _pp.data() + p, r, jb ? jptr.data() : nullptr))
            return false;
        double s = 0.0;
        for (int i = 0; i < row.nres; ++i)
        {
            if (!std::isfinite(r[i]))
                return false;
            s += r[i] * r[i];
        }
        if (jb)
            for (int k = 0; k < row.np; ++k)
            {
                if (_pj[p + k] < 0)
                    continue;
                const PBlock& b = block(_pv[p + k]);
                double* J = jb + _pj[p + k];
                if (b.manifold)
                {
                    const double* Ja = jptr[k];
                    for (int i = 0; i < row.nres; ++i)
                        for (int tt = 0; tt < b.tsize; ++tt)
                        {
                            double acc = 0.0;
                            for (int a = 0; a < b.size; ++a)
                                acc += Ja[i * b.size + a] * b.plusJacobian[size_t(a) * b.tsize + tt];
                            J[i * b.tsize + tt] = acc;
                        }
                }
                for (int i = 0; i < row.nres * b.tsize; ++i)
                    if (!std::isfinite(J[i]))
                        return false;
            }
        double rho[3];
        if (!rowRho(row.loss, row.lossScale, s, rho))
        {
            *cost = 0.5 * s;
            return true;
        }
        *cost = 0.5 * rho[0];
        if (jb)
        {
            Corrector c(s, rho);
            for (int k = 0; k < row.np; ++k)
                if (_pj[p + k] >= 0)
                    c.correctJacobian(row.nres, block(_pv[p + k]).tsize, r, jb + _pj[p + k]);
            c.correctResiduals(row.nres, r);
        }
        return true;
    }

    // dst[c] = sum over the partitions of _partF[part][c], in partition order
    void reduceF(double* dst) const
    {
        const int P = nParts();
        const size_t nf = size_t(_nf);
#pragma omp parallel for schedule(static) num_threads(_threads)
        for (int c = 0; c < _nf; ++c)
        {
            double s = 0.0;
            for (int part = 0; part < P; ++part)
                s += _partF[size_t(part) * nf + size_t(c)];
            dst[c] = s;
        }
    }

    // The cost of the reduced program at the current state, and per mode the rest (see Mode). The cost
    // and the F part of the gradient are summed per partition, then over the partitions in order.
    bool evaluate(Mode mode, double* cost, std::vector<double>* gradOut = nullptr)
    {
        const auto t0 = Clock::now();
        _prepare();
        const bool wantJ = mode != Mode::Cost;
        if (wantJ)
        {
            const int nm = int(_mb.size());
#pragma omp parallel for schedule(static) num_threads(_threads)
            for (int i = 0; i < nm; ++i)
            {
                PBlock& b = block(_mb[i]);
                b.manifold->PlusJacobian(b.ptr, b.plusJacobian.data());
            }
        }
        std::vector<double>* g = nullptr;
        if (wantJ)
        {
            g = mode == Mode::Full ? &_grad : gradOut;
            g->assign(size_t(_n), 0.0);
        }
        const int P = nParts();
        std::vector<double> partCost(size_t(P), 0.0);
        std::vector<char> partOk(size_t(P), 1);
#pragma omp parallel num_threads(_threads)
        {
            std::vector<double> amb, rLocal, jLocal;
            std::vector<double*> jptr;
#pragma omp for schedule(dynamic, 1)
            for (int part = 0; part < P; ++part)
            {
                double* gf = wantJ ? _partF.data() + size_t(part) * size_t(_nf) : nullptr;
                if (gf)
                    std::fill(gf, gf + _nf, 0.0);
                double total = 0.0;
                for (int ri = _parts[part]; ri < _parts[part + 1]; ++ri)
                {
                    const Row& row = _rows[ri];
                    double* r;
                    if (mode == Mode::Full)
                        r = _r.data() + row.r;
                    else
                    {
                        rLocal.resize(size_t(row.nres));
                        r = rLocal.data();
                    }
                    double* jb = nullptr;
                    if (wantJ)
                    {
                        if (mode == Mode::Full)
                            jb = _J.data() + row.j;
                        else
                        {
                            jLocal.resize(size_t(row.jSize));
                            jb = jLocal.data();
                        }
                    }
                    double c = 0.0;
                    if (!evalRow(row, r, jb, amb, jptr, &c))
                    {
                        partOk[part] = 0;
                        break;
                    }
                    total += c;
                    if (!wantJ)
                        continue;
                    // the gradient: an E block's rows are all in this partition, the F part goes to gf
                    for (int k = 0; k < row.np; ++k)
                    {
                        if (_pj[row.p + k] < 0)
                            continue;
                        const int v = _pv[row.p + k];
                        const PBlock& b = block(v);
                        const double* J = jb + _pj[row.p + k];
                        double* gg = v >= 0 ? g->data() + b.col : gf + (b.col - _ne);
                        for (int tt = 0; tt < b.tsize; ++tt)
                        {
                            double acc = 0.0;
                            for (int i = 0; i < row.nres; ++i)
                                acc += J[i * b.tsize + tt] * r[i];
                            gg[tt] += acc;
                        }
                    }
                }
                partCost[part] = total;
            }
        }
        bool ok = true;
        for (char o : partOk)
            ok = ok && o;
        double total = 0.0;
        if (ok)
        {
            for (double c : partCost)
                total += c;
            if (wantJ)
                reduceF(g->data() + _ne);
            *cost = total;
        }
        const double dt = since(t0);
        if (mode == Mode::Full)
        {
            _times.evalJ += dt;
            ++_times.nJ;
        }
        else if (mode == Mode::Cost)
        {
            _times.evalCost += dt;
            ++_times.nCost;
        }
        else
        {
            _times.evalGrad += dt;
            ++_times.nGrad;
        }
        return ok;
    }

    // Plus(x, delta) per block, delta in the tangent space (E then F); out: ambient, E then F
    void plus(const std::vector<double>& x, const std::vector<double>& delta, std::vector<double>* out) const
    {
        const auto t0 = Clock::now();
        out->resize(x.size());
        const int nb = int(_e.size() + _f.size()), nE = int(_e.size());
#pragma omp parallel for schedule(static, 4096) num_threads(_threads)
        for (int i = 0; i < nb; ++i)
        {
            const PBlock& b = i < nE ? _e[i] : _f[i - nE];
            const double* d = delta.data() + b.col;
            const double* xi = x.data() + b.x;
            double* o = out->data() + b.x;
            if (b.manifold)
                b.manifold->Plus(xi, d, o);
            else
                for (int a = 0; a < b.size; ++a)
                    o[a] = xi[a] + d[a];
            // ParameterBlock::Plus: project onto the box constraints
            if (!b.lower.empty())
                for (int a = 0; a < b.size; ++a)
                    o[a] = std::max(o[a], b.lower[a]);
            if (!b.upper.empty())
                for (int a = 0; a < b.size; ++a)
                    o[a] = std::min(o[a], b.upper[a]);
        }
        _times.plus += since(t0);
    }

    // Squared column norms of the (corrected, current) Jacobian, tangent space E then F.
    std::vector<double> squaredColumnNorms()
    {
        const auto t0 = Clock::now();
        std::vector<double> n(size_t(_n), 0.0);
        const int P = nParts();
#pragma omp parallel for schedule(dynamic, 1) num_threads(_threads)
        for (int part = 0; part < P; ++part)
        {
            double* gf = _partF.data() + size_t(part) * size_t(_nf);
            std::fill(gf, gf + _nf, 0.0);
            for (int ri = _parts[part]; ri < _parts[part + 1]; ++ri)
            {
                const Row& row = _rows[ri];
                for (int k = 0; k < row.np; ++k)
                {
                    if (_pj[row.p + k] < 0)
                        continue;
                    const int v = _pv[row.p + k];
                    const PBlock& b = block(v);
                    double* c = v >= 0 ? n.data() + b.col : gf + (b.col - _ne);
                    const double* J = _J.data() + row.j + _pj[row.p + k];
                    for (int i = 0; i < row.nres; ++i)
                        for (int t = 0; t < b.tsize; ++t)
                            c[t] += J[i * b.tsize + t] * J[i * b.tsize + t];
                }
            }
        }
        reduceF(n.data() + _ne);
        _times.norms += since(t0);
        return n;
    }

    void scaleColumns(const std::vector<double>& s)
    {
        const auto t0 = Clock::now();
        const int P = nParts();
#pragma omp parallel for schedule(dynamic, 1) num_threads(_threads)
        for (int part = 0; part < P; ++part)
            for (int ri = _parts[part]; ri < _parts[part + 1]; ++ri)
            {
                const Row& row = _rows[ri];
                for (int k = 0; k < row.np; ++k)
                {
                    if (_pj[row.p + k] < 0)
                        continue;
                    const PBlock& b = block(_pv[row.p + k]);
                    const double* c = s.data() + b.col;
                    double* J = _J.data() + row.j + _pj[row.p + k];
                    for (int i = 0; i < row.nres; ++i)
                        for (int t = 0; t < b.tsize; ++t)
                            J[i * b.tsize + t] *= c[t];
                }
            }
        _times.scale += since(t0);
    }

    // The model cost change of a step: -(f . (r + f / 2)) with f = J step, summed per partition then in
    // order (LevenbergMarquardtStrategy / TrustRegionMinimizer::ComputeTrustRegionStep).
    double modelCostChange(const std::vector<double>& step) const
    {
        const auto t0 = Clock::now();
        const int P = nParts();
        std::vector<double> partDot(size_t(P), 0.0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(_threads)
        for (int part = 0; part < P; ++part)
        {
            double dot = 0.0;
            for (int ri = _parts[part]; ri < _parts[part + 1]; ++ri)
            {
                const Row& row = _rows[ri];
                double f[16];
                std::vector<double> big;
                double* fr = f;
                if (row.nres > 16)
                {
                    big.assign(size_t(row.nres), 0.0);
                    fr = big.data();
                }
                else
                    std::fill(f, f + row.nres, 0.0);
                for (int k = 0; k < row.np; ++k)
                {
                    if (_pj[row.p + k] < 0)
                        continue;
                    const PBlock& b = block(_pv[row.p + k]);
                    const double* yy = step.data() + b.col;
                    const double* J = _J.data() + row.j + _pj[row.p + k];
                    for (int i = 0; i < row.nres; ++i)
                    {
                        double acc = 0.0;
                        for (int t = 0; t < b.tsize; ++t)
                            acc += J[i * b.tsize + t] * yy[t];
                        fr[i] += acc;
                    }
                }
                const double* r = _r.data() + row.r;
                for (int i = 0; i < row.nres; ++i)
                    dot += fr[i] * (r[i] + fr[i] / 2.0);
            }
            partDot[part] = dot;
        }
        double dot = 0.0;
        for (double d : partDot)
            dot += d;
        _times.model += since(t0);
        return -dot;
    }

    // ---- the linear step: (J^T J + D^2) y = J^T r by Schur elimination of the E blocks ---------------
    // J is the scaled Jacobian; D the LM diagonal (E then F). Returns false on a numerical failure.

    // the block (f1, f2), f1 <= f2, in _sv
    size_t blockOff(int f1, int f2) const
    {
        if (!_pairIdx.empty())
            return _sOff[size_t(_pairIdx[size_t(f1) * _f.size() + size_t(f2)])];
        const auto b = _sCol.begin() + std::ptrdiff_t(_sStart[f1]), e = _sCol.begin() + std::ptrdiff_t(_sStart[f1 + 1]);
        return _sOff[size_t(std::lower_bound(b, e, f2) - _sCol.begin())];
    }

    // a row's F^T F blocks (f1 <= f2) into S and F^T b into rhs
    void rowTerms(const Row& row, double* S, double* rhs) const
    {
        const size_t p = row.p;
        const int nres = row.nres;
        const double* b = _r.data() + row.r;
        for (int k1 = 0; k1 < row.np; ++k1)
        {
            const int v1 = _pv[p + k1];
            if (v1 > -2)
                continue;
            const int f1 = -2 - v1, t1 = _f[f1].tsize;
            const double* F1 = _J.data() + row.j + _pj[p + k1];
            for (int k2 = 0; k2 < row.np; ++k2)
            {
                const int v2 = _pv[p + k2];
                if (v2 > -2 || -2 - v2 < f1)
                    continue;
                const int f2 = -2 - v2;
                addAtB(nres, t1, _f[f2].tsize, F1, _J.data() + row.j + _pj[p + k2], S + blockOff(f1, f2));
            }
            double* rr = rhs + (_f[f1].col - _ne);
            for (int a = 0; a < t1; ++a)
            {
                double acc = 0.0;
                for (int i = 0; i < nres; ++i)
                    acc += F1[i * t1 + a] * b[i];
                rr[a] += acc;
            }
        }
    }

    // DenseQRSolver: Eigen's HouseholderQR of [J; diag(D)] (column-major, the problem's order) against
    // [r; 0], the dense_linear_algebra_library_type EIGEN that AliceVision leaves as it is
    bool solveStepQR(const std::vector<double>& D, std::vector<double>* y)
    {
        const auto t0 = Clock::now();
        const Eigen::Index m = Eigen::Index(_nr), nc = _nf;
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m + nc, nc);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(m + nc);
        for (const Row& row : _rows)
        {
            for (int i = 0; i < row.nres; ++i)
                b(Eigen::Index(row.r) + i) = _r[row.r + size_t(i)];
            for (int k = 0; k < row.np; ++k)
            {
                if (_pj[row.p + k] < 0)
                    continue;
                const PBlock& fb = block(_pv[row.p + k]);
                const double* J = _J.data() + row.j + _pj[row.p + k];
                const int c = fb.col - _ne;
                for (int i = 0; i < row.nres; ++i)
                    for (int tt = 0; tt < fb.tsize; ++tt)
                        A(Eigen::Index(row.r) + i, c + tt) = J[i * fb.tsize + tt];
            }
        }
        for (Eigen::Index i = 0; i < nc; ++i)
            A(m + i, i) = D[size_t(_ne + i)];
        const Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
        const Eigen::VectorXd yf = qr.solve(b);
        y->assign(size_t(_n), 0.0);
        bool ok = true;
        for (Eigen::Index i = 0; i < nc; ++i)
        {
            (*y)[size_t(_ne + i)] = yf(i);
            ok = ok && std::isfinite(yf(i));
        }
        _times.factor += since(t0);
        return ok;
    }

    bool solveStep(const std::vector<double>& D, std::vector<double>* y)
    {
        const auto t0 = Clock::now();
        ++_times.nSolves;
        if (_qr)
            return solveStepQR(D, y);
        const int nE = int(_e.size()), nF = int(_f.size());
        // 1. per group of chunks, into the group's copy of (S, rhs): per chunk E^T E + D_e^2 and its inverse,
        // E^T b and E^T F (kept for the back-substitution), each row's F^T F and F^T b, then the chunk's Schur
        // terms -m1^T inv m2 and -m1^T inv E^T b; the rows without an E block go with the last group
        const int G = int(_groups.size()) - 1;
        const size_t svn = _sv.size();
        bool finite = true;
#pragma omp parallel for schedule(dynamic, 1) num_threads(_threads) reduction(&& : finite)
        for (int gi = 0; gi < G; ++gi)
        {
            double* S = _sg.data() + size_t(gi) * _sgStride;
            double* rhs = S + svn;
            std::fill(S, S + _sgStride, 0.0);
            if (gi == 0)
                // Ceres adds the diagonal first (SchurEliminator::Eliminate); (f1, f1) is a row's first block
                for (int f1 = 0; f1 < nF; ++f1)
                {
                    const int t1 = _f[f1].tsize;
                    const double* Df = D.data() + _f[f1].col;
                    double* blk = S + _sOff[_sStart[f1]];
                    for (int a = 0; a < t1; ++a)
                        blk[a * t1 + a] = Df[a] * Df[a];
                }
            for (int c = _groups[gi]; c < _groups[gi + 1]; ++c)
            {
                const double* De = D.data() + _e[c].col;
                double ete[9] = {De[0] * De[0], 0, 0, 0, De[1] * De[1], 0, 0, 0, De[2] * De[2]};
                double g[3] = {0, 0, 0};
                const size_t l0 = _cfStart[c], l1 = _cfStart[c + 1];
                if (l1 > l0)
                    std::fill(_m.data() + _cfM[l0], _m.data() + _cfM[l1 - 1] + 3 * size_t(_f[_cf[l1 - 1]].tsize), 0.0);
                for (int ri = _chunkStart[c]; ri < _chunkStart[c + 1]; ++ri)
                {
                    const Row& row = _rows[ri];
                    const int nres = row.nres;
                    const double* b = _r.data() + row.r;
                    const double* E = nullptr;
                    for (int k = 0; k < row.np; ++k)
                        if (_pv[row.p + k] >= 0)
                            E = _J.data() + row.j + _pj[row.p + k];
                    for (int p = 0; p < 3; ++p)
                    {
                        for (int q = 0; q < 3; ++q)
                        {
                            double acc = 0.0;
                            for (int i = 0; i < nres; ++i)
                                acc += E[i * 3 + p] * E[i * 3 + q];
                            ete[p * 3 + q] += acc;
                        }
                        double acc = 0.0;
                        for (int i = 0; i < nres; ++i)
                            acc += E[i * 3 + p] * b[i];
                        g[p] += acc;
                    }
                    for (int k = 0; k < row.np; ++k)
                    {
                        const int v = _pv[row.p + k];
                        if (v > -2)
                            continue;
                        const int f = -2 - v, tt = _f[f].tsize;
                        size_t l = l0;
                        while (_cf[l] != f)
                            ++l;
                        addAtB(nres, 3, tt, E, _J.data() + row.j + _pj[row.p + k], _m.data() + _cfM[l]);
                    }
                    rowTerms(row, S, rhs);
                }
                const Eigen::Matrix3d invM = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(ete).inverse();   // InvertPSDMatrix<3>, full rank
                double* inv = _inv.data() + size_t(c) * 9;
                for (int p = 0; p < 3; ++p)
                    for (int q = 0; q < 3; ++q)
                    {
                        inv[p * 3 + q] = invM(p, q);
                        finite = finite && std::isfinite(invM(p, q));
                    }
                std::copy(g, g + 3, _ge.data() + size_t(c) * 3);
                for (size_t la = l0; la < l1; ++la)
                {
                    const int f1 = _cf[la], t1 = _f[f1].tsize;
                    const double* m1 = _m.data() + _cfM[la];
                    double u[kMaxT * 3];   // m1^T inv, t1 x 3
                    for (int a = 0; a < t1; ++a)
                        for (int q = 0; q < 3; ++q)
                            u[a * 3 + q] = m1[0 * t1 + a] * inv[0 * 3 + q] + m1[1 * t1 + a] * inv[1 * 3 + q] + m1[2 * t1 + a] * inv[2 * 3 + q];
                    for (size_t lb = la; lb < l1; ++lb)
                        subU3M(t1, _f[_cf[lb]].tsize, u, _m.data() + _cfM[lb], S + blockOff(f1, _cf[lb]));
                    double* rr = rhs + (_f[f1].col - _ne);
                    for (int a = 0; a < t1; ++a)
                        rr[a] -= u[a * 3 + 0] * g[0] + u[a * 3 + 1] * g[1] + u[a * 3 + 2] * g[2];
                }
            }
            if (gi == G - 1)
                for (int ri = _chunkStart[nE]; ri < int(_rows.size()); ++ri)
                    rowTerms(_rows[ri], S, rhs);
        }
        _times.elim += since(t0);
        if (!finite)
            return false;
        // 2. S and rhs: the groups' copies summed in group order
        const auto t0s = Clock::now();
        pfor(_sgStride, [&](size_t i) {
            double s = 0.0;
            for (int gi = 0; gi < G; ++gi)
                s += _sg[size_t(gi) * _sgStride + i];
            if (i < svn)
                _sv[i] = s;
            else
                _rhs[i - svn] = s;
        });
        _times.schur += since(t0s);

        // 3. the reduced camera system
        const auto t1c = Clock::now();
        Eigen::VectorXd yf(_nf);
        if (_nf > 0)
        {
            const Eigen::Map<const Eigen::VectorXd> rhs(_rhs.data(), _nf);
            if (!_opt.sparse)
            {
#pragma omp parallel for schedule(dynamic, 4) num_threads(_threads)
                for (int f1 = 0; f1 < nF; ++f1)
                {
                    const int t1 = _f[f1].tsize, c1 = _f[f1].col - _ne;
                    for (size_t s = _sStart[f1]; s < _sStart[f1 + 1]; ++s)
                    {
                        const int f2 = _sCol[s], t2 = _f[f2].tsize, c2 = _f[f2].col - _ne;
                        const double* blk = _sv.data() + _sOff[s];
                        for (int a = 0; a < t1; ++a)
                            for (int bb = 0; bb < t2; ++bb)
                                _A(c2 + bb, c1 + a) = blk[a * t2 + bb];   // lower: (f2, f1) = (f1, f2)^T
                    }
                }
                Eigen::LLT<Eigen::MatrixXd> llt(_A);
                if (llt.info() != Eigen::Success)
                    return false;
                yf = llt.solve(rhs);
            }
            else
            {
                double* val = _S.valuePtr();
                const int nnz = int(_cscSrc.size());
#pragma omp parallel for schedule(static) num_threads(_threads)
                for (int q = 0; q < nnz; ++q)
                    val[q] = _sv[_cscSrc[q]];
                if (!_analyzed)
                {
                    // EigenSparseCholesky: the symbolic analysis once per solve, the numeric one per iteration
                    _ldlt.analyzePattern(_S);
                    _analyzed = true;
                }
                _ldlt.factorize(_S);
                if (_ldlt.info() != Eigen::Success)
                    return false;
                yf = _ldlt.solve(rhs);
                if (_ldlt.info() != Eigen::Success)
                    return false;
            }
        }
        _times.factor += since(t1c);

        // 4. back-substitution: y_e = (E^T E + D_e^2)^-1 (E^T b - sum_f E^T F y_f)
        const auto t2c = Clock::now();
        y->assign(size_t(_n), 0.0);
        std::copy(yf.data(), yf.data() + _nf, y->begin() + _ne);
        bool ok = true;
        for (int i = 0; i < _nf; ++i)
            ok = ok && std::isfinite(yf[i]);
#pragma omp parallel for schedule(dynamic, 1024) num_threads(_threads) reduction(&& : ok)
        for (int c = 0; c < nE; ++c)
        {
            const double* g = _ge.data() + size_t(c) * 3;
            double q[3] = {g[0], g[1], g[2]};
            for (size_t l = _cfStart[c]; l < _cfStart[c + 1]; ++l)
            {
                const int f = _cf[l], t = _f[f].tsize;
                const double* m = _m.data() + _cfM[l];
                const double* yy = yf.data() + (_f[f].col - _ne);
                for (int p = 0; p < 3; ++p)
                {
                    double acc = 0.0;
                    for (int bb = 0; bb < t; ++bb)
                        acc += m[p * t + bb] * yy[bb];
                    q[p] -= acc;
                }
            }
            const double* inv = _inv.data() + size_t(c) * 9;
            double* ye = y->data() + _e[c].col;
            for (int p = 0; p < 3; ++p)
            {
                ye[p] = inv[p * 3 + 0] * q[0] + inv[p * 3 + 1] * q[1] + inv[p * 3 + 2] * q[2];
                ok = ok && std::isfinite(ye[p]);
            }
        }
        _times.back += since(t2c);
        return ok;
    }

    // ---- the trust-region loop, in Ceres' order (trust_region_minimizer.cc) ---------------------------
    // TrustRegionMinimizer::DoLineSearch with ArmijoLineSearch and CUBIC interpolation: scales *delta (the
    // unscaled tangent step) by the accepted step size, or leaves it when the search fails. The function
    // is x -> cost(Plus(x, a * delta)) with its derivative delta . gradient.
    //
    // Returns the cost at Plus(x, *delta) on return, which is what Ceres' ComputeCandidatePointAndEvaluate-
    // Cost evaluates next (max when invalid): the same point, so the minimizer reuses it. Ceres evaluates
    // each sample's gradient with its value; it is read only when the sample fails the Armijo test (the
    // cubic step), so it is taken then. One difference follows: Ceres' evaluation fails, and the sample is
    // invalid, when a Jacobian is not finite, and a sample that passes here is never asked.
    double lineSearch(const std::vector<double>& x, double cost, std::vector<double>* delta, Iteration* info)
    {
        // CHESHIRE_BA_OWN_EAGER=1: every sample with its gradient, as Ceres takes them
        static const bool eager = ::cheshire::env::flag("CHESHIRE_BA_OWN_EAGER");
        const size_t n = delta->size();
        const std::vector<double>& dl = *delta;
        const double initialGradient = psum(n, [&](size_t i) { return _grad[i] * dl[i]; });
        const double dirInf = pmax(n, [&](size_t i) { return std::abs(dl[i]); });
        poly::Sample initial;
        initial.x = 0.0;
        initial.value = cost;
        initial.gradient = initialGradient;
        initial.valueValid = initial.gradientValid = true;
        std::vector<double> scaled(n), xp, gg;
        auto sample = [&](double a, bool withGradient, poly::Sample* s) {
            withGradient = withGradient || eager;
            ++info->lsSamples;
            if (withGradient)
                ++info->lsGradients;
            s->x = a;
            s->valueValid = s->gradientValid = false;
            pfor(n, [&](size_t i) { scaled[i] = a * dl[i]; });
            plus(x, scaled, &xp);
            setState(xp);
            double c = 0.0;
            const bool ok = withGradient ? evaluate(Mode::Gradient, &c, &gg) : evaluate(Mode::Cost, &c);
            setState(x);
            if (!ok || !std::isfinite(c))
                return;
            s->value = c;
            s->valueValid = true;
            if (!withGradient)
                return;
            const double d = psum(n, [&](size_t i) { return dl[i] * gg[i]; });
            if (!std::isfinite(d))
                return;
            s->gradient = d;
            s->gradientValid = true;
        };
        const double big = std::numeric_limits<double>::max();
        poly::Sample previous, current;
        sample(1.0, false, &current);
        const double firstCost = current.valueValid ? current.value : big;
        int iterations = 0;
        while (!current.valueValid || current.value > (cost + _opt.lineSearchSufficientDecrease * initialGradient * current.x))
        {
            ++iterations;
            if (iterations >= _opt.lineSearchMaxIterations)
            {
                info->lsStep = current.x;
                return firstCost;
            }
            if (current.valueValid && !current.gradientValid && !eager)
                sample(current.x, true, &current);   // the value again, identical, with the gradient
            const double step = poly::stepSize(initial, previous, current, _opt.lineSearchMaxContraction * current.x,
                                               _opt.lineSearchMinContraction * current.x);
            if (step * dirInf < _opt.lineSearchMinStepSize)
            {
                info->lsStep = step;
                return firstCost;
            }
            previous = current;
            sample(step, false, &current);
        }
        const double a = current.x;
        pfor(n, [&](size_t i) { (*delta)[i] *= a; });
        info->lsStep = current.x;
        info->lsOk = true;
        return current.value;
    }

    void minimize(Result* res)
    {
        std::vector<double> x = stateVector();
        _x0 = x;
        if (_constrained)
        {
            // IterationZero: project the starting point onto the feasible set
            std::vector<double> zero(size_t(_n), 0.0), xp;
            plus(x, zero, &xp);
            x = xp;
            setState(x);
        }
        double xCost = 0.0;
        if (!evaluate(Mode::Full, &xCost))
        {
            res->why = "initial evaluation failed";
            setState(_x0);
            _prepare();
            return;
        }
        const int n = _n;
        // Jacobi scaling, from iteration 0's Jacobian
        _scale.assign(size_t(n), 1.0);
        if (_opt.jacobiScaling)
        {
            const auto c = squaredColumnNorms();
            for (int i = 0; i < n; ++i)
                _scale[i] = 1.0 / (1.0 + std::sqrt(c[i]));
            scaleColumns(_scale);
        }
        auto gradientNorms = [&](double* maxNorm) {
            std::vector<double> neg(_grad.size());
            pfor(neg.size(), [&](size_t i) { neg[i] = -_grad[i]; });
            std::vector<double> xg;
            plus(x, neg, &xg);
            *maxNorm = pmax(x.size(), [&](size_t i) { return std::abs(x[i] - xg[i]); });
        };

        res->initialCost = xCost + _fixedCost;
        double radius = _opt.initialRadius, decreaseFactor = 2.0;
        bool reuseDiagonal = false;
        // TrustRegionStepEvaluator with no nonmonotonic steps
        double referenceCost = xCost, candidateCostRef = xCost, currentCost = xCost, minimumCost = xCost;
        double accumulatedReference = 0.0, accumulatedCandidate = 0.0;
        int consecutiveNonmonotonic = 0;

        Iteration it0;
        it0.iteration = 0;
        it0.cost = xCost + _fixedCost;
        gradientNorms(&it0.gradientMaxNorm);
        it0.stepIsValid = true;
        it0.stepIsSuccessful = true;
        it0.radius = radius;
        res->iterations.push_back(it0);
        std::vector<double> best = x;
        double bestCost = xCost;
        int consecutiveInvalid = 0;
        bool atLeastOneSuccessful = false;
        std::vector<double> y, step, delta, candidate, D(static_cast<size_t>(n));
        Iteration cur = it0;
        auto finish = [&](const std::string& why) {
            res->termination = why;
            // Solver::Solve: the minimizer's best point when the solution is usable, else the starting point
            setState(why.rfind("FAILURE", 0) == 0 ? _x0 : best);
            _prepare();
            res->finalCost = bestCost + _fixedCost;
            res->ok = true;
        };

        while (true)
        {
            // FinalizeIterationAndCheckIfMinimizerCanContinue (the entry for the finished iteration is
            // already in the trace)
            if (cur.iteration >= _opt.maxIterations)
                return finish("NO_CONVERGENCE (max iterations)");
            if (cur.stepIsSuccessful && cur.gradientMaxNorm <= _opt.gradientTolerance)
                return finish("CONVERGENCE (gradient tolerance)");
            if (cur.radius <= _opt.minRadius)
                return finish("CONVERGENCE (min trust region radius)");

            const double prevGradMax = cur.gradientMaxNorm;
            Iteration nx;
            nx.iteration = cur.iteration + 1;

            // ComputeTrustRegionStep: the LM diagonal from the scaled Jacobian
            if (!reuseDiagonal)
            {
                _diag = squaredColumnNorms();
                pfor(_diag.size(), [&](size_t i) { _diag[i] = std::min(std::max(_diag[i], _opt.minDiagonal), _opt.maxDiagonal); });
            }
            pfor(size_t(n), [&](size_t i) { D[i] = std::sqrt(_diag[i] / radius); });
            const bool solved = solveStep(D, &y);
            reuseDiagonal = true;
            double modelChange = 0.0;
            nx.stepIsValid = false;
            if (solved)
            {
                step.resize(size_t(n));
                pfor(size_t(n), [&](size_t i) { step[i] = -y[i]; });
                modelChange = modelCostChange(step);
                nx.stepIsValid = modelChange > 0.0;
            }
            if (!nx.stepIsValid)
            {
                // HandleInvalidStep
                if (++consecutiveInvalid >= _opt.maxConsecutiveInvalidSteps)
                    return finish("FAILURE (consecutive invalid steps)");
                radius = radius / decreaseFactor;   // StepIsInvalid = StepRejected(0)
                decreaseFactor *= 2.0;
                reuseDiagonal = true;
                nx.cost = xCost + _fixedCost;
                nx.gradientMaxNorm = prevGradMax;
                nx.radius = radius;
                res->iterations.push_back(nx);
                cur = nx;
                continue;
            }
            consecutiveInvalid = 0;
            delta.resize(size_t(n));
            pfor(size_t(n), [&](size_t i) { delta[i] = step[i] * _scale[i]; });
            // ComputeCandidatePointAndEvaluateCost; with bounds, Ceres first searches along the step
            // (max_num_line_search_step_size_iterations > 0), and the search has the candidate's cost
            double candidateCost = std::numeric_limits<double>::max();
            const bool searched = _constrained && _opt.lineSearchMaxIterations > 0;
            if (searched)
                candidateCost = lineSearch(x, xCost, &delta, &nx);
            plus(x, delta, &candidate);
            if (!searched)
            {
                setState(candidate);
                double c = 0.0;
                if (evaluate(Mode::Cost, &c))
                    candidateCost = c;
                setState(x);
            }

            // ParameterToleranceReached (after at least one successful step)
            const double xNorm = std::sqrt(psum(x.size(), [&](size_t i) { return x[i] * x[i]; }));
            nx.stepNorm = std::sqrt(psum(x.size(), [&](size_t i) { return (x[i] - candidate[i]) * (x[i] - candidate[i]); }));
            if (atLeastOneSuccessful && nx.stepNorm <= _opt.parameterTolerance * (xNorm + _opt.parameterTolerance))
                return finish("CONVERGENCE (parameter tolerance)");
            // FunctionToleranceReached
            nx.costChange = xCost - candidateCost;
            if (std::abs(nx.costChange) <= _opt.functionTolerance * xCost)
                return finish("CONVERGENCE (function tolerance)");

            // IsStepSuccessful
            double relativeDecrease;
            if (candidateCost >= std::numeric_limits<double>::max())
                relativeDecrease = std::numeric_limits<double>::lowest();
            else
            {
                const double rd = (currentCost - candidateCost) / modelChange;
                const double hd = (referenceCost - candidateCost) / (accumulatedReference + modelChange);
                relativeDecrease = std::max(rd, hd);
            }
            nx.relativeDecrease = relativeDecrease;
            if (relativeDecrease > _opt.minRelativeDecrease)
            {
                atLeastOneSuccessful = true;
                // HandleSuccessfulStep
                x = candidate;
                setState(x);
                if (!evaluate(Mode::Full, &xCost))
                    return finish("FAILURE (evaluation at the accepted point)");
                if (_opt.jacobiScaling)
                    scaleColumns(_scale);
                nx.cost = xCost + _fixedCost;
                gradientNorms(&nx.gradientMaxNorm);
                nx.stepIsSuccessful = true;
                // LevenbergMarquardtStrategy::StepAccepted
                radius = radius / std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * relativeDecrease - 1.0, 3));
                radius = std::min(_opt.maxRadius, radius);
                decreaseFactor = 2.0;
                reuseDiagonal = false;
                // TrustRegionStepEvaluator::StepAccepted, max_consecutive_nonmonotonic_steps = 0
                currentCost = candidateCost;
                accumulatedCandidate += modelChange;
                accumulatedReference += modelChange;
                if (currentCost < minimumCost)
                {
                    minimumCost = currentCost;
                    consecutiveNonmonotonic = 0;
                    candidateCostRef = currentCost;
                    accumulatedCandidate = 0.0;
                }
                else
                {
                    ++consecutiveNonmonotonic;
                    if (currentCost > candidateCostRef)
                    {
                        candidateCostRef = currentCost;
                        accumulatedCandidate = 0.0;
                    }
                }
                if (consecutiveNonmonotonic == 0)               // == max_consecutive_nonmonotonic_steps (0)
                {
                    referenceCost = candidateCostRef;
                    accumulatedReference = accumulatedCandidate;
                }
                // FinalizeIteration: keep the best point
                if (xCost < bestCost)
                {
                    bestCost = xCost;
                    best = x;
                }
            }
            else
            {
                nx.stepIsSuccessful = false;
                nx.cost = candidateCost + _fixedCost;
                nx.gradientMaxNorm = prevGradMax;
                radius = radius / decreaseFactor;               // StepRejected
                decreaseFactor *= 2.0;
                reuseDiagonal = true;
            }
            nx.radius = radius;
            res->iterations.push_back(nx);
            cur = nx;
        }
    }
};

// ---- step 3: in place of Ceres ---------------------------------------------------------------------------
// CHESHIRE_BA_SOLVER: Cheshire's solver by default (since 0.3.7), Ceres with "ceres"
inline bool solverEnabled()
{
    static const bool on = ::cheshire::env::text("CHESHIRE_BA_SOLVER", "cheshire") != "ceres";
    return on;
}

// The solver's options from the Ceres options BundleAdjustmentCeres set, or false (and why) when those ask
// for something the solver does not do.
inline bool optionsFrom(const ceres::Solver::Options& c, Options* o, std::string* why)
{
    if (c.linear_solver_type != ceres::DENSE_SCHUR && c.linear_solver_type != ceres::SPARSE_SCHUR)
        *why = std::string("linear solver ") + ceres::LinearSolverTypeToString(c.linear_solver_type);
    else if (!c.linear_solver_ordering)
        *why = "no elimination ordering";
    else if (c.minimizer_type != ceres::TRUST_REGION || c.trust_region_strategy_type != ceres::LEVENBERG_MARQUARDT)
        *why = "not Levenberg-Marquardt";
    else if (c.use_nonmonotonic_steps || c.use_inner_iterations || c.dynamic_sparsity)
        *why = "nonmonotonic steps, inner iterations or dynamic sparsity";
    else if (c.line_search_interpolation_type != ceres::CUBIC)
        *why = "line search interpolation other than CUBIC";
    else if (c.max_solver_time_in_seconds < 1e8)
        *why = "a solver time limit";
    else if (c.min_lm_diagonal <= 0.0 || c.max_num_iterations < 0)
        *why = "unusual limits";
    if (!why->empty())
        return false;
    o->sparse = c.linear_solver_type == ceres::SPARSE_SCHUR;
    o->maxIterations = c.max_num_iterations;
    o->maxConsecutiveInvalidSteps = c.max_num_consecutive_invalid_steps;
    o->functionTolerance = c.function_tolerance;
    o->gradientTolerance = c.gradient_tolerance;
    o->parameterTolerance = c.parameter_tolerance;
    o->initialRadius = c.initial_trust_region_radius;
    o->maxRadius = c.max_trust_region_radius;
    o->minRadius = c.min_trust_region_radius;
    o->minRelativeDecrease = c.min_relative_decrease;
    o->minDiagonal = c.min_lm_diagonal;
    o->maxDiagonal = c.max_lm_diagonal;
    o->jacobiScaling = c.jacobi_scaling;
    o->lineSearchMaxIterations = c.max_num_line_search_step_size_iterations;
    o->lineSearchSufficientDecrease = c.line_search_sufficient_function_decrease;
    o->lineSearchMaxContraction = c.max_line_search_step_contraction;
    o->lineSearchMinContraction = c.min_line_search_step_contraction;
    o->lineSearchMinStepSize = c.min_line_search_step_size;
    o->threads = c.num_threads;
    return true;
}

// Each reason a solve goes to Ceres, logged the first time.
inline void noteFallback(const std::string& why)
{
    static std::mutex m;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lock(m);
    if (seen.insert(why).second)
        ALICEVISION_LOG_INFO("cheshire: BA solver: this solve goes to Ceres (" << why << "); further ones for the same reason are not logged");
}

// The solver's options from Ceres', with its threads: requestedThreads is what BundleAdjustmentCeres asked
// for; the deterministic mode's single thread is Ceres' need, not this solver's (its result does not
// depend on the count), so CHESHIRE_BA_THREADS alone lowers it. False (and why) when Ceres was asked for
// something this solver does not do.
inline bool solverOptions(const ceres::Solver::Options& options, int requestedThreads, Options* o, std::string* why)
{
    if (!optionsFrom(options, o, why))
        return false;
    static const long forced = static_cast<long>(::cheshire::env::integer("CHESHIRE_BA_THREADS", 0));
    o->threads = forced > 0 ? int(forced) : std::max(1, requestedThreads);
    return true;
}

// Solve a Source with this solver and fill the Summary as Ceres would; false (the reason noted, the blocks
// untouched) when it cannot. *out, when given, receives the solver's own account.
inline bool solveSource(const Source& src, const ceres::Solver::Options& options, const Options& o, std::function<void()> prepare,
                        ceres::Solver::Summary* summary, Result* out = nullptr)
{
    Solver solver(src, o, std::move(prepare));
    const Result r = solver.solve();
    if (!r.ok)
    {
        noteFallback(r.why);
        return false;
    }
    *summary = ceres::Solver::Summary();
    ceres::Solver::Summary& s = *summary;
    s.minimizer_type = ceres::TRUST_REGION;
    s.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    s.linear_solver_type_given = options.linear_solver_type;
    s.linear_solver_type_used = r.denseQR ? ceres::DENSE_QR : options.linear_solver_type;
    s.sparse_linear_algebra_library_type = options.sparse_linear_algebra_library_type;
    s.dense_linear_algebra_library_type = options.dense_linear_algebra_library_type;
    s.num_threads_given = options.num_threads;
    s.num_threads_used = r.time.threads;
    s.termination_type = r.termination.rfind("CONVERGENCE", 0) == 0      ? ceres::CONVERGENCE
                         : r.termination.rfind("NO_CONVERGENCE", 0) == 0 ? ceres::NO_CONVERGENCE
                                                                         : ceres::FAILURE;
    s.message = "Cheshire's solver: " + r.termination;
    s.initial_cost = r.initialCost;
    s.final_cost = r.finalCost;
    s.fixed_cost = r.fixedCost;
    s.num_successful_steps = s.num_unsuccessful_steps = s.num_line_search_steps = 0;
    for (const Iteration& it : r.iterations)
    {
        ceres::IterationSummary i;
        i.iteration = it.iteration;
        i.step_is_valid = it.stepIsValid;
        i.step_is_successful = it.stepIsSuccessful;
        i.cost = it.cost;
        i.cost_change = it.costChange;
        i.gradient_max_norm = it.gradientMaxNorm;
        i.step_norm = it.stepNorm;
        i.relative_decrease = it.relativeDecrease;
        i.trust_region_radius = it.radius;
        i.line_search_iterations = std::max(0, it.lsSamples - 1);
        s.iterations.push_back(i);
        (it.stepIsSuccessful ? s.num_successful_steps : s.num_unsuccessful_steps) += 1;   // iteration 0 counts, as in Ceres
        s.num_line_search_steps += std::max(0, it.lsSamples - 1);
    }
    s.is_constrained = r.constrained;
    src.counts(&s.num_parameter_blocks, &s.num_parameters, &s.num_residual_blocks, &s.num_residuals);
    s.num_parameter_blocks_reduced = r.eBlocks + r.fBlocks;
    s.num_residual_blocks_reduced = r.rows;
    s.num_residuals_reduced = int(r.residuals);
    s.num_effective_parameters_reduced = 3 * r.eBlocks + r.fColumns;
    s.num_linear_solves = r.time.nSolves;
    s.num_residual_evaluations = r.time.nJ + r.time.nCost + r.time.nGrad;
    s.num_jacobian_evaluations = r.time.nJ + r.time.nGrad;
    s.preprocessor_time_in_seconds = r.time.setup;
    s.minimizer_time_in_seconds = r.seconds - r.time.setup;
    s.postprocessor_time_in_seconds = 0.0;
    s.total_time_in_seconds = r.seconds;
    s.jacobian_evaluation_time_in_seconds = r.time.evalJ + r.time.evalGrad;
    s.residual_evaluation_time_in_seconds = r.time.evalCost;
    s.linear_solver_time_in_seconds = r.time.elim + r.time.schur + r.time.factor + r.time.back;
    if (profileEnabled())
    {
        const Times& t = r.time;
        const double known = t.setup + t.evalJ + t.evalCost + t.evalGrad + t.elim + t.schur + t.factor + t.back + t.norms + t.scale + t.model +
                             t.plus + t.state;
        ALICEVISION_LOG_INFO("cheshire: BA own profile: " << r.seconds << " s = setup " << t.setup << " + evalJ " << t.evalJ << " (" << t.nJ
                             << ") + evalCost " << t.evalCost << " (" << t.nCost << ") + evalGrad " << t.evalGrad << " (" << t.nGrad
                             << ") + elim " << t.elim << " + schur " << t.schur << " + factor " << t.factor << " + back " << t.back << " ("
                             << t.nSolves << ") + norms " << t.norms << " + scale " << t.scale << " + model " << t.model << " + plus "
                             << t.plus << " + state " << t.state << " + other " << r.seconds - known << "; rows " << r.rows << ", E "
                             << r.eBlocks << ", F " << r.fBlocks << " (" << r.fColumns << " columns), J " << r.jValues << ", S "
                             << r.sBlocks << " blocks " << r.sValues << " values, groups " << r.groups << ", parts " << r.parts << ", "
                             << (r.denseQR ? "dense QR" : o.sparse ? "sparse" : "dense") << ", " << r.iterations.size() - 1
                             << " iterations, " << t.threads << " threads");
    }
    if (digestEnabled())
        ALICEVISION_LOG_INFO("cheshire: BA digest: " << r.rows << " rows, " << r.fColumns << " columns, in " << std::hex << r.digestIn << " out "
                                                     << r.digest << std::dec << ", " << r.iterations.size() - 1 << " iterations");
    // the callbacks see the iterations after the fact (AliceVision's only logs them)
    for (const auto& i : s.iterations)
        for (ceres::IterationCallback* cb : options.callbacks)
            (*cb)(i);
    if (out)
        *out = r;
    if (directMode() == 2)
        lastResult() = r;
    return true;
}

// Solve AliceVision's ceres::Problem with this solver in place of ceres::Solve, or return false (the problem
// untouched) for Ceres to solve it.
inline bool solveInstead(ceres::Problem& problem, const ceres::Solver::Options& options, int requestedThreads,
                         std::function<void()> prepare, ceres::Solver::Summary* summary)
{
    if (!solverEnabled())
        return false;
    static std::once_flag said;
    std::call_once(said, [] {
        ALICEVISION_LOG_INFO("cheshire: BA solver: Cheshire's own for the Schur solves (CHESHIRE_BA_SOLVER=ceres for Ceres)");
    });
    Options o;
    std::string why;
    if (!solverOptions(options, requestedThreads, &o, &why))
    {
        noteFallback(why);
        return false;
    }
    const ProblemSource src(problem, *options.linear_solver_ordering);
    return solveSource(src, options, o, std::move(prepare), summary);
}

// Ceres' trajectory, for the comparison.
class Recorder : public ceres::IterationCallback
{
  public:
    std::vector<Iteration> iterations;
    ceres::CallbackReturnType operator()(const ceres::IterationSummary& s) override
    {
        Iteration it;
        it.iteration = s.iteration;
        it.cost = s.cost;
        it.costChange = s.cost_change;
        it.gradientMaxNorm = s.gradient_max_norm;
        it.stepNorm = s.step_norm;
        it.relativeDecrease = s.relative_decrease;
        it.radius = s.trust_region_radius;
        it.stepIsValid = s.step_is_valid;
        it.stepIsSuccessful = s.step_is_successful;
        iterations.push_back(it);
        return ceres::SOLVER_CONTINUE;
    }
};

inline bool shadowEnabled()
{
    static const bool on = ::cheshire::env::flag("CHESHIRE_BA_SHADOW");
    return on;
}

// CHESHIRE_BA_SHADOW=2: both trajectories in full when they part
inline bool shadowVerbose()
{
    static const bool on = ::cheshire::env::integer("CHESHIRE_BA_SHADOW", 0) >= 2;
    return on;
}

inline std::string traces(const Result& ours, const std::vector<Iteration>& ceresIts)
{
    std::ostringstream o;
    o.precision(12);
    const size_t n = std::max(ours.iterations.size(), ceresIts.size());
    for (size_t i = 0; i < n; ++i)
    {
        o << "\n    it " << i;
        for (int side = 0; side < 2; ++side)
        {
            const auto& v = side == 0 ? ours.iterations : ceresIts;
            o << (side == 0 ? " | ours " : " | Ceres ");
            if (i >= v.size())
            {
                o << "-";
                continue;
            }
            const Iteration& it = v[i];
            o << "cost " << it.cost << " rho " << it.relativeDecrease << " radius " << it.radius << " step " << it.stepNorm
              << (it.stepIsValid ? "" : " INVALID") << (it.stepIsSuccessful ? " ok" : " rejected");
            if (side == 0 && it.lsSamples > 0)
                o << " [ls " << it.lsSamples << " samples, " << it.lsGradients << " gradients, a " << it.lsStep
                  << (it.lsOk ? "" : " FAILED") << "]";
        }
    }
    return o.str();
}

inline double relDiff(double a, double b)
{
    const double d = std::abs(a - b), m = std::max(std::abs(a), std::abs(b));
    return m > 0.0 ? d / m : d;
}

// One line comparing the two trajectories: iterations, costs, the first iteration where they part.
inline std::string compare(const Result& ours, const std::vector<Iteration>& ceresIts, const ceres::Solver::Summary& summary,
                           double paramMaxAbs, double paramMaxRel, double ceresSeconds)
{
    std::ostringstream o;
    o.precision(10);
    if (!ours.ok)
    {
        o << "not run (" << ours.why << ")";
        return o.str();
    }
    int firstDiff = -1;
    double worst = 0.0;
    const size_t nIt = std::min(ours.iterations.size(), ceresIts.size());
    for (size_t i = 0; i < nIt; ++i)
    {
        const double d = relDiff(ours.iterations[i].cost, ceresIts[i].cost);
        worst = std::max(worst, d);
        if (firstDiff < 0 && (d > 1e-9 || ours.iterations[i].stepIsSuccessful != ceresIts[i].stepIsSuccessful))
            firstDiff = int(i);
    }
    o << ours.eBlocks << " landmarks, " << ours.fBlocks << " camera blocks (" << ours.fColumns << " columns), " << ours.rows
      << " rows | ours: " << ours.iterations.size() - 1 << " iterations, cost " << ours.initialCost << " -> " << ours.finalCost
      << ", " << ours.termination << ", " << ours.seconds << " s | Ceres: " << ceresIts.size() - 1 << " iterations, cost "
      << summary.initial_cost << " -> " << summary.final_cost << ", " << ceres::TerminationTypeToString(summary.termination_type)
      << ", " << ceresSeconds << " s | final cost rel diff " << relDiff(ours.finalCost, summary.final_cost)
      << ", per-iteration cost worst rel diff " << worst << ", first parting at iteration "
      << (firstDiff < 0 ? std::string("none") : std::to_string(firstDiff)) << " | parameters max abs diff " << paramMaxAbs
      << ", max rel diff " << paramMaxRel << " | Ceres says: " << summary.message << " (" << summary.num_line_search_steps
      << " line search steps)";
    const Times& t = ours.time;
    o.precision(4);
    o << " | ours split: setup " << t.setup << " s, jacobians " << t.nJ << "x " << t.evalJ << " s, costs " << t.nCost << "x "
      << t.evalCost << " s, gradients " << t.nGrad << "x " << t.evalGrad << " s, elim " << t.elim << " s, schur " << t.schur << " s, factor " << t.factor
      << " s, back " << t.back << " s, " << t.threads << " threads, digest " << std::hex << ours.digest << std::dec;
    return o.str();
}

}  // namespace own
}  // namespace cheshire
}  // namespace sfm
}  // namespace aliceVision
