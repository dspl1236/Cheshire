// This file is part of the Cheshire patch set for AliceVision (https://github.com/dspl1236/Cheshire).
//
// 0.3.7: a bundle-adjustment solver of Cheshire's own (docs/notes/ba-own-solver.md), step 1: shadow mode.
//
// It walks the ceres::Problem BundleAdjustmentCeres has built, through Ceres' public API, and solves it
// with Ceres' own algorithm specialised for this shape: Levenberg-Marquardt in a trust region, Jacobi
// scaling, the Huber corrector, landmarks eliminated by Schur complement and the reduced camera system
// solved by dense LLT (DENSE_SCHUR) or Eigen's SimplicialLDLT with AMD ordering (SPARSE_SCHUR, the
// EIGEN_SPARSE path Cheshire's GPL-free Ceres takes). Every residual block is evaluated by its own
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
// The references to Ceres below are to the 2.2.0 source Cheshire builds.
#pragma once

#include <aliceVision/depthMap/cuda/hip/cheshire/env.h>

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace aliceVision {
namespace sfm {
namespace cheshire {
namespace own {

struct Options
{
    bool sparse = false;               // SPARSE_SCHUR (Eigen SimplicialLDLT, AMD) or DENSE_SCHUR (LLT)
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
};

struct Result
{
    bool ok = false;               // false: the problem is outside the solver's scope, or it failed
    std::string why;
    std::vector<Iteration> iterations;
    double initialCost = 0.0, finalCost = 0.0, fixedCost = 0.0;
    std::string termination;
    int eBlocks = 0, fBlocks = 0, fColumns = 0, rows = 0;
    double seconds = 0.0;
};

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

class Solver
{
  public:
    // prepare(): what Ceres' EvaluationCallback does before a new evaluation point (AliceVision pushes
    // the intrinsics and distortion blocks into its camera objects, which the cost functions read).
    Solver(ceres::Problem& problem, const ceres::ParameterBlockOrdering& ordering, const Options& options,
           std::function<void()> prepare)
      : _problem(problem), _ordering(ordering), _opt(options), _prepare(std::move(prepare))
    {}

    // Solve in place: the problem's parameter blocks end at the solution (the caller restores them in
    // shadow mode).
    Result solve()
    {
        const auto t0 = std::chrono::steady_clock::now();
        Result res;
        if (!setup(&res.why))
            return res;
        res.eBlocks = int(_e.size());
        res.fBlocks = int(_f.size());
        res.fColumns = _nf;
        res.rows = int(_rows.size());
        res.fixedCost = _fixedCost;
        minimize(&res);
        res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return res;
    }

  private:
    struct PBlock
    {
        double* ptr = nullptr;
        int size = 0, tsize = 0;
        const ceres::Manifold* manifold = nullptr;
        int offset = 0;                      // tangent offset: in the e vector (E) or the f vector (F)
        std::vector<double> plusJacobian;    // size x tsize, row-major, at the current state
        std::vector<double> lower, upper;    // ambient bounds, empty when the block has none
    };
    struct Row
    {
        const ceres::CostFunction* cost = nullptr;
        const ceres::LossFunction* loss = nullptr;
        std::vector<double*> params;        // every parameter block, constant ones included
        std::vector<int> pe;                 // per parameter: index into _e, or -1
        std::vector<int> pf;                 // per parameter: index into _f, or -1
        int e = -1;                          // the row's eliminated block, or -1
        int nres = 0;
        int rOffset = 0;                     // into _r
        std::vector<int> jOffset;            // per parameter: into _J (tangent, row-major nres x tsize), or -1
        int jBase = 0, jSize = 0;            // the row's Jacobian blocks are contiguous from jBase
    };
    enum class Mode
    {
        Full,       // _r (corrected), _J (tangent, corrected, unscaled) and _grad at the current state
        Cost,       // the cost alone (a candidate point); _r and _J untouched
        Gradient    // the cost and the unscaled tangent gradient into *grad (the line search); _r, _J untouched
    };

    ceres::Problem& _problem;
    const ceres::ParameterBlockOrdering& _ordering;
    Options _opt;
    std::function<void()> _prepare;

    std::vector<PBlock> _e, _f;             // free blocks, in Ceres' order
    int _ne = 0, _nf = 0;                    // tangent sizes of the e and f vectors
    std::vector<Row> _rows;                  // rows with an e block first, grouped by it, then the rest
    std::vector<int> _chunkStart;            // per e block: its first row; _chunkStart[ne] = first row without one
    double _fixedCost = 0.0;
    std::vector<double> _r, _J, _scale, _diag, _grad;
    int _nr = 0, _nJ = 0;
    bool _constrained = false;               // any bound on a free block: Ceres' is_constrained

    // ---- setup -------------------------------------------------------------------------------------
    bool setup(std::string* why)
    {
        std::vector<double*> blocks;
        _problem.GetParameterBlocks(&blocks);
        const auto& groups = _ordering.group_to_elements();
        if (groups.empty())
        {
            *why = "no ordering";
            return false;
        }
        const int eGroup = groups.begin()->first;
        std::unordered_map<double*, int> eIndex, fIndex;
        // Ceres' Schur reordering: by group, within a group by the set's order (the block's address)
        for (const auto& [group, elements] : groups)
        {
            for (double* p : elements)
            {
                if (!_problem.HasParameterBlock(p) || _problem.IsParameterBlockConstant(p))
                    continue;
                PBlock b;
                b.ptr = p;
                b.size = _problem.ParameterBlockSize(p);
                b.tsize = _problem.ParameterBlockTangentSize(p);
                b.manifold = _problem.GetManifold(p);
                if (b.tsize == 0)
                    continue;
                if (group == eGroup)
                {
                    b.offset = _ne;
                    _ne += b.tsize;
                    eIndex[p] = int(_e.size());
                    _e.push_back(std::move(b));
                }
                else
                {
                    b.offset = _nf;
                    _nf += b.tsize;
                    fIndex[p] = int(_f.size());
                    _f.push_back(std::move(b));
                }
            }
        }
        // blocks the ordering does not name (Ceres puts them in the last group)
        for (double* p : blocks)
        {
            if (eIndex.count(p) || fIndex.count(p) || _problem.IsParameterBlockConstant(p) || _ordering.IsMember(p))
                continue;
            PBlock b;
            b.ptr = p;
            b.size = _problem.ParameterBlockSize(p);
            b.tsize = _problem.ParameterBlockTangentSize(p);
            b.manifold = _problem.GetManifold(p);
            if (b.tsize == 0)
                continue;
            b.offset = _nf;
            _nf += b.tsize;
            fIndex[p] = int(_f.size());
            _f.push_back(std::move(b));
        }
        for (auto& b : _e)
            if (b.tsize != 3)
            {
                *why = "an eliminated block is not of size 3";
                return false;
            }
        // bounds (Program::IsBoundsConstrained): -max / +max when a coordinate has none
        const double big = std::numeric_limits<double>::max();
        for (auto* v : {&_e, &_f})
            for (auto& b : *v)
            {
                bool any = false;
                std::vector<double> lo(b.size), hi(b.size);
                for (int i = 0; i < b.size; ++i)
                {
                    lo[i] = _problem.GetParameterLowerBound(b.ptr, i);
                    hi[i] = _problem.GetParameterUpperBound(b.ptr, i);
                    any = any || lo[i] > -big || hi[i] < big;
                }
                if (any)
                {
                    b.lower = std::move(lo);
                    b.upper = std::move(hi);
                    _constrained = true;
                }
            }

        std::vector<ceres::ResidualBlockId> rids;
        _problem.GetResidualBlocks(&rids);
        std::vector<std::vector<Row>> byE(_e.size());
        std::vector<Row> noE;
        for (ceres::ResidualBlockId id : rids)
        {
            Row row;
            row.cost = _problem.GetCostFunctionForResidualBlock(id);
            row.loss = _problem.GetLossFunctionForResidualBlock(id);
            _problem.GetParameterBlocksForResidualBlock(id, &row.params);
            row.nres = row.cost->num_residuals();
            bool anyFree = false;
            for (double* p : row.params)
            {
                const auto ie = eIndex.find(p);
                const auto jf = fIndex.find(p);
                row.pe.push_back(ie == eIndex.end() ? -1 : ie->second);
                row.pf.push_back(jf == fIndex.end() ? -1 : jf->second);
                if (ie != eIndex.end())
                {
                    if (row.e >= 0)
                    {
                        *why = "a residual block touches two eliminated blocks";
                        return false;
                    }
                    row.e = ie->second;
                }
                anyFree = anyFree || ie != eIndex.end() || jf != fIndex.end();
            }
            if (!anyFree)
            {
                // Ceres' preprocessor removes it and reports its cost as fixed_cost
                std::vector<double> r(row.nres);
                if (!row.cost->Evaluate(row.params.data(), r.data(), nullptr))
                {
                    *why = "a fixed residual block failed to evaluate";
                    return false;
                }
                double s = 0.0;
                for (double v : r)
                    s += v * v;
                if (row.loss)
                {
                    double rho[3];
                    row.loss->Evaluate(s, rho);
                    _fixedCost += 0.5 * rho[0];
                }
                else
                    _fixedCost += 0.5 * s;
                continue;
            }
            if (row.e >= 0)
                byE[row.e].push_back(std::move(row));
            else
                noE.push_back(std::move(row));
        }
        _chunkStart.assign(_e.size() + 1, 0);
        for (size_t e = 0; e < _e.size(); ++e)
        {
            _chunkStart[e] = int(_rows.size());
            for (auto& row : byE[e])
                _rows.push_back(std::move(row));
        }
        _chunkStart[_e.size()] = int(_rows.size());
        for (auto& row : noE)
            _rows.push_back(std::move(row));

        // storage
        for (auto& row : _rows)
        {
            row.rOffset = _nr;
            _nr += row.nres;
            row.jBase = _nJ;
            row.jOffset.assign(row.params.size(), -1);
            for (size_t k = 0; k < row.params.size(); ++k)
            {
                const PBlock* b = row.pe[k] >= 0 ? &_e[row.pe[k]] : row.pf[k] >= 0 ? &_f[row.pf[k]] : nullptr;
                if (!b)
                    continue;
                row.jOffset[k] = _nJ;
                _nJ += row.nres * b->tsize;
            }
            row.jSize = _nJ - row.jBase;
        }
        _r.assign(_nr, 0.0);
        _J.assign(_nJ, 0.0);
        return true;
    }

    PBlock& blockOf(const Row& row, size_t k) { return row.pe[k] >= 0 ? _e[row.pe[k]] : _f[row.pf[k]]; }

    // x: every free block's ambient values, E first then F (for the norms and the step)
    std::vector<double> stateVector() const
    {
        std::vector<double> x;
        for (const auto* v : {&_e, &_f})
            for (const auto& b : *v)
                x.insert(x.end(), b.ptr, b.ptr + b.size);
        return x;
    }
    void setState(const std::vector<double>& x)
    {
        size_t o = 0;
        for (auto* v : {&_e, &_f})
            for (auto& b : *v)
            {
                std::copy(x.begin() + o, x.begin() + o + b.size, b.ptr);
                o += b.size;
            }
    }

    // ---- evaluation ----------------------------------------------------------------------------------
    // The cost of the reduced program at the current state, and per mode the rest (see Mode). Ceres'
    // order per residual block (residual_block.cc): Evaluate, the manifold product, then the loss
    // correction of the tangent Jacobian (from the uncorrected residual), then of the residual.
    bool evaluate(Mode mode, double* cost, std::vector<double>* gradOut = nullptr)
    {
        _prepare();
        const bool wantJ = mode != Mode::Cost;
        if (wantJ)
        {
            for (auto* v : {&_e, &_f})
                for (auto& b : *v)
                    if (b.manifold)
                    {
                        b.plusJacobian.assign(size_t(b.size) * b.tsize, 0.0);
                        b.manifold->PlusJacobian(b.ptr, b.plusJacobian.data());
                    }
        }
        std::vector<double>& g = mode == Mode::Full ? _grad : *gradOut;
        if (wantJ)
            g.assign(size_t(_ne + _nf), 0.0);
        double total = 0.0;
        std::vector<double> amb, rLocal, jLocal;
        std::vector<double*> jptr;
        for (auto& row : _rows)
        {
            double* r;
            if (mode == Mode::Full)
                r = _r.data() + row.rOffset;
            else
            {
                rLocal.resize(size_t(row.nres));
                r = rLocal.data();
            }
            double* jBase = nullptr;
            if (wantJ)
            {
                if (mode == Mode::Full)
                    jBase = _J.data() + row.jBase;
                else
                {
                    jLocal.resize(size_t(row.jSize));
                    jBase = jLocal.data();
                }
            }
            auto Jt = [&](size_t k) { return jBase + (row.jOffset[k] - row.jBase); };
            jptr.assign(row.params.size(), nullptr);
            if (wantJ)
            {
                size_t ambTotal = 0;
                for (size_t k = 0; k < row.params.size(); ++k)
                    if (row.jOffset[k] >= 0 && blockOf(row, k).manifold)
                        ambTotal += size_t(row.nres) * blockOf(row, k).size;
                amb.assign(ambTotal, 0.0);
                size_t ao = 0;
                for (size_t k = 0; k < row.params.size(); ++k)
                {
                    if (row.jOffset[k] < 0)
                        continue;
                    if (blockOf(row, k).manifold)
                    {
                        jptr[k] = amb.data() + ao;
                        ao += size_t(row.nres) * blockOf(row, k).size;
                    }
                    else
                        jptr[k] = Jt(k);
                }
            }
            if (!row.cost->Evaluate(row.params.data(), r, wantJ ? jptr.data() : nullptr))
                return false;
            for (int i = 0; i < row.nres; ++i)
                if (!std::isfinite(r[i]))
                    return false;
            double s = 0.0;
            for (int i = 0; i < row.nres; ++i)
                s += r[i] * r[i];
            if (wantJ)
            {
                for (size_t k = 0; k < row.params.size(); ++k)
                {
                    if (row.jOffset[k] < 0)
                        continue;
                    const PBlock& b = blockOf(row, k);
                    double* J = Jt(k);
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
            }
            if (!row.loss)
                total += 0.5 * s;
            else
            {
                double rho[3];
                row.loss->Evaluate(s, rho);
                total += 0.5 * rho[0];
                if (wantJ)
                {
                    Corrector c(s, rho);
                    for (size_t k = 0; k < row.params.size(); ++k)
                        if (row.jOffset[k] >= 0)
                            c.correctJacobian(row.nres, blockOf(row, k).tsize, r, Jt(k));
                    c.correctResiduals(row.nres, r);
                }
            }
            if (wantJ)
            {
                for (size_t k = 0; k < row.params.size(); ++k)
                {
                    if (row.jOffset[k] < 0)
                        continue;
                    const PBlock& b = blockOf(row, k);
                    const double* J = Jt(k);
                    double* gg = g.data() + (row.pe[k] >= 0 ? b.offset : _ne + b.offset);
                    for (int tt = 0; tt < b.tsize; ++tt)
                    {
                        double acc = 0.0;
                        for (int i = 0; i < row.nres; ++i)
                            acc += J[i * b.tsize + tt] * r[i];
                        gg[tt] += acc;
                    }
                }
            }
        }
        *cost = total;
        return true;
    }

    // Plus(x, delta) per block, delta in the tangent space (E then F); out: ambient, E then F
    void plus(const std::vector<double>& x, const std::vector<double>& delta, std::vector<double>* out) const
    {
        out->resize(x.size());
        size_t xo = 0;
        for (int pass = 0; pass < 2; ++pass)
        {
            const auto& v = pass == 0 ? _e : _f;
            for (const auto& b : v)
            {
                const double* d = delta.data() + (pass == 0 ? b.offset : _ne + b.offset);
                double* o = out->data() + xo;
                if (b.manifold)
                    b.manifold->Plus(x.data() + xo, d, o);
                else
                    for (int a = 0; a < b.size; ++a)
                        o[a] = x[xo + a] + d[a];
                // ParameterBlock::Plus: project onto the box constraints
                if (!b.lower.empty())
                    for (int a = 0; a < b.size; ++a)
                        o[a] = std::max(o[a], b.lower[a]);
                if (!b.upper.empty())
                    for (int a = 0; a < b.size; ++a)
                        o[a] = std::min(o[a], b.upper[a]);
                xo += b.size;
            }
        }
    }

    // Squared column norms of the (corrected, current) Jacobian, tangent space E then F.
    std::vector<double> squaredColumnNorms() const
    {
        std::vector<double> n(size_t(_ne + _nf), 0.0);
        for (const auto& row : _rows)
            for (size_t k = 0; k < row.params.size(); ++k)
            {
                if (row.jOffset[k] < 0)
                    continue;
                const PBlock& b = row.pe[k] >= 0 ? _e[row.pe[k]] : _f[row.pf[k]];
                double* c = n.data() + (row.pe[k] >= 0 ? b.offset : _ne + b.offset);
                const double* J = _J.data() + row.jOffset[k];
                for (int i = 0; i < row.nres; ++i)
                    for (int t = 0; t < b.tsize; ++t)
                        c[t] += J[i * b.tsize + t] * J[i * b.tsize + t];
            }
        return n;
    }

    void scaleColumns(const std::vector<double>& s)
    {
        for (auto& row : _rows)
            for (size_t k = 0; k < row.params.size(); ++k)
            {
                if (row.jOffset[k] < 0)
                    continue;
                const PBlock& b = row.pe[k] >= 0 ? _e[row.pe[k]] : _f[row.pf[k]];
                const double* c = s.data() + (row.pe[k] >= 0 ? b.offset : _ne + b.offset);
                double* J = _J.data() + row.jOffset[k];
                for (int i = 0; i < row.nres; ++i)
                    for (int t = 0; t < b.tsize; ++t)
                        J[i * b.tsize + t] *= c[t];
            }
    }

    // J y (tangent y, E then F) into out (size _nr)
    void jacobianTimes(const std::vector<double>& y, std::vector<double>* out) const
    {
        out->assign(size_t(_nr), 0.0);
        for (const auto& row : _rows)
            for (size_t k = 0; k < row.params.size(); ++k)
            {
                if (row.jOffset[k] < 0)
                    continue;
                const PBlock& b = row.pe[k] >= 0 ? _e[row.pe[k]] : _f[row.pf[k]];
                const double* yy = y.data() + (row.pe[k] >= 0 ? b.offset : _ne + b.offset);
                const double* J = _J.data() + row.jOffset[k];
                for (int i = 0; i < row.nres; ++i)
                {
                    double acc = 0.0;
                    for (int t = 0; t < b.tsize; ++t)
                        acc += J[i * b.tsize + t] * yy[t];
                    (*out)[row.rOffset + i] += acc;
                }
            }
    }

    // ---- the linear step: (J^T J + D^2) y = J^T r by Schur elimination of the E blocks ---------------
    // J is the scaled Jacobian; D the LM diagonal (E then F). Returns false on a numerical failure.
    bool solveStep(const std::vector<double>& D, std::vector<double>* y)
    {
        const int nf = _nf;
        using MatF = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        // S and rhs over the F columns. Dense storage for DENSE_SCHUR; for SPARSE_SCHUR the same
        // arithmetic accumulates into block pairs, then goes to Eigen's SimplicialLDLT.
        Eigen::MatrixXd S = Eigen::MatrixXd::Zero(nf, nf);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(nf);
        for (const auto& b : _f)
            for (int t = 0; t < b.tsize; ++t)
                S(b.offset + t, b.offset + t) += D[_ne + b.offset + t] * D[_ne + b.offset + t];

        const size_t nE = _e.size();
        std::vector<Eigen::Matrix3d> invEtE(nE);
        std::vector<Eigen::Vector3d> gE(nE);
        std::vector<std::vector<std::pair<int, Eigen::MatrixXd>>> bufE(nE);   // per chunk: (f block, E^T F 3 x tf)
        for (size_t e = 0; e < nE; ++e)
        {
            const PBlock& eb = _e[e];
            Eigen::Matrix3d ete = Eigen::Matrix3d::Zero();
            for (int t = 0; t < 3; ++t)
                ete(t, t) = D[eb.offset + t] * D[eb.offset + t];
            Eigen::Vector3d g = Eigen::Vector3d::Zero();
            std::map<int, Eigen::MatrixXd> buf;
            for (int ri = _chunkStart[e]; ri < _chunkStart[e + 1]; ++ri)
            {
                const Row& row = _rows[ri];
                const double* b = _r.data() + row.rOffset;
                int ek = -1;
                for (size_t k = 0; k < row.params.size(); ++k)
                    if (row.pe[k] >= 0)
                        ek = int(k);
                Eigen::Map<const MatF> E(_J.data() + row.jOffset[ek], row.nres, 3);
                Eigen::Map<const Eigen::VectorXd> bv(b, row.nres);
                ete.noalias() += E.transpose() * E;
                g.noalias() += E.transpose() * bv;
                for (size_t k = 0; k < row.params.size(); ++k)
                {
                    if (row.pf[k] < 0 || row.jOffset[k] < 0)
                        continue;
                    const PBlock& fb = _f[row.pf[k]];
                    Eigen::Map<const MatF> F(_J.data() + row.jOffset[k], row.nres, fb.tsize);
                    auto it = buf.find(row.pf[k]);
                    if (it == buf.end())
                        it = buf.emplace(row.pf[k], Eigen::MatrixXd::Zero(3, fb.tsize)).first;
                    it->second.noalias() += E.transpose() * F;
                    rhs.segment(fb.offset, fb.tsize).noalias() += F.transpose() * bv;
                    for (size_t k2 = 0; k2 < row.params.size(); ++k2)
                    {
                        if (row.pf[k2] < 0 || row.jOffset[k2] < 0)
                            continue;
                        const PBlock& fb2 = _f[row.pf[k2]];
                        Eigen::Map<const MatF> F2(_J.data() + row.jOffset[k2], row.nres, fb2.tsize);
                        S.block(fb.offset, fb2.offset, fb.tsize, fb2.tsize).noalias() += F.transpose() * F2;
                    }
                }
            }
            const Eigen::Matrix3d inv = ete.inverse();   // InvertPSDMatrix<3>, full rank: Eigen's inverse()
            invEtE[e] = inv;
            gE[e] = g;
            const Eigen::Vector3d invg = inv * g;
            for (auto& [f, m] : buf)
            {
                const PBlock& fb = _f[f];
                rhs.segment(fb.offset, fb.tsize).noalias() -= m.transpose() * invg;
                for (auto& [f2, m2] : buf)
                {
                    const PBlock& fb2 = _f[f2];
                    S.block(fb.offset, fb2.offset, fb.tsize, fb2.tsize).noalias() -= m.transpose() * inv * m2;
                }
            }
            bufE[e].assign(buf.begin(), buf.end());
        }
        // rows without an E block
        for (int ri = _chunkStart[nE]; ri < int(_rows.size()); ++ri)
        {
            const Row& row = _rows[ri];
            Eigen::Map<const Eigen::VectorXd> bv(_r.data() + row.rOffset, row.nres);
            for (size_t k = 0; k < row.params.size(); ++k)
            {
                if (row.pf[k] < 0 || row.jOffset[k] < 0)
                    continue;
                const PBlock& fb = _f[row.pf[k]];
                Eigen::Map<const MatF> F(_J.data() + row.jOffset[k], row.nres, fb.tsize);
                rhs.segment(fb.offset, fb.tsize).noalias() += F.transpose() * bv;
                for (size_t k2 = 0; k2 < row.params.size(); ++k2)
                {
                    if (row.pf[k2] < 0 || row.jOffset[k2] < 0)
                        continue;
                    const PBlock& fb2 = _f[row.pf[k2]];
                    Eigen::Map<const MatF> F2(_J.data() + row.jOffset[k2], row.nres, fb2.tsize);
                    S.block(fb.offset, fb2.offset, fb.tsize, fb2.tsize).noalias() += F.transpose() * F2;
                }
            }
        }

        Eigen::VectorXd yf(nf);
        if (nf > 0)
        {
            if (!_opt.sparse)
            {
                Eigen::LLT<Eigen::MatrixXd> llt(S);
                if (llt.info() != Eigen::Success)
                    return false;
                yf = llt.solve(rhs);
            }
            else
            {
                Eigen::SparseMatrix<double> Ss = S.sparseView();
                Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper, Eigen::AMDOrdering<int>> ldlt;
                ldlt.compute(Ss);
                if (ldlt.info() != Eigen::Success)
                    return false;
                yf = ldlt.solve(rhs);
                if (ldlt.info() != Eigen::Success)
                    return false;
            }
        }
        y->assign(size_t(_ne + _nf), 0.0);
        for (int i = 0; i < nf; ++i)
            (*y)[_ne + i] = yf[i];
        // back-substitution: y_e = (E^T E + D_e^2)^-1 (E^T b - sum_f E^T F y_f)
        for (size_t e = 0; e < nE; ++e)
        {
            Eigen::Vector3d q = gE[e];
            for (auto& [f, m] : bufE[e])
                q.noalias() -= m * yf.segment(_f[f].offset, _f[f].tsize);
            const Eigen::Vector3d ye = invEtE[e] * q;
            for (int t = 0; t < 3; ++t)
                (*y)[_e[e].offset + t] = ye[t];
        }
        for (double v : *y)
            if (!std::isfinite(v))
                return false;
        return true;
    }

    // ---- the trust-region loop, in Ceres' order (trust_region_minimizer.cc) ---------------------------
    // TrustRegionMinimizer::DoLineSearch with ArmijoLineSearch and CUBIC interpolation: scales *delta
    // (the unscaled tangent step) by the accepted step size, or leaves it when the search fails. The
    // function is x -> cost(Plus(x, a * delta)) with its derivative delta . gradient.
    void lineSearch(const std::vector<double>& x, double cost, std::vector<double>* delta)
    {
        const size_t n = delta->size();
        double initialGradient = 0.0, dirInf = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            initialGradient += _grad[i] * (*delta)[i];
            dirInf = std::max(dirInf, std::abs((*delta)[i]));
        }
        poly::Sample initial;
        initial.x = 0.0;
        initial.value = cost;
        initial.gradient = initialGradient;
        initial.valueValid = initial.gradientValid = true;
        auto evalAt = [&](double a, poly::Sample* s) {
            *s = poly::Sample();
            s->x = a;
            std::vector<double> scaled(n), xp, gg;
            for (size_t i = 0; i < n; ++i)
                scaled[i] = a * (*delta)[i];
            plus(x, scaled, &xp);
            setState(xp);
            double c = 0.0;
            const bool ok = evaluate(Mode::Gradient, &c, &gg);
            setState(x);
            if (!ok || !std::isfinite(c))
                return;
            s->value = c;
            s->valueValid = true;
            double d = 0.0;
            for (size_t i = 0; i < n; ++i)
                d += (*delta)[i] * gg[i];
            if (!std::isfinite(d))
                return;
            s->gradient = d;
            s->gradientValid = true;
        };
        poly::Sample previous, current;
        evalAt(1.0, &current);
        int iterations = 0;
        while (!current.valueValid || current.value > (cost + _opt.lineSearchSufficientDecrease * initialGradient * current.x))
        {
            ++iterations;
            if (iterations >= _opt.lineSearchMaxIterations)
                return;
            const double step = poly::stepSize(initial, previous, current, _opt.lineSearchMaxContraction * current.x,
                                               _opt.lineSearchMinContraction * current.x);
            if (step * dirInf < _opt.lineSearchMinStepSize)
                return;
            previous = current;
            evalAt(step, &current);
        }
        for (auto& d : *delta)
            d *= current.x;
    }

    void minimize(Result* res)
    {
        std::vector<double> x = stateVector();
        if (_constrained)
        {
            // IterationZero: project the starting point onto the feasible set
            std::vector<double> zero(size_t(_ne + _nf), 0.0), xp;
            plus(x, zero, &xp);
            x = xp;
            setState(x);
        }
        double xCost = 0.0;
        if (!evaluate(Mode::Full, &xCost))
        {
            res->why = "initial evaluation failed";
            return;
        }
        const int n = _ne + _nf;
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
            for (size_t i = 0; i < neg.size(); ++i)
                neg[i] = -_grad[i];
            std::vector<double> xg;
            plus(x, neg, &xg);
            double m = 0.0;
            for (size_t i = 0; i < x.size(); ++i)
                m = std::max(m, std::abs(x[i] - xg[i]));
            *maxNorm = m;
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
        std::vector<double> y, step, delta, candidate, modelResiduals;
        Iteration cur = it0;
        auto finish = [&](const std::string& why) {
            res->termination = why;
            setState(best);
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
                for (auto& d : _diag)
                    d = std::min(std::max(d, _opt.minDiagonal), _opt.maxDiagonal);
            }
            std::vector<double> D(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i)
                D[i] = std::sqrt(_diag[i] / radius);
            const bool solved = solveStep(D, &y);
            reuseDiagonal = true;
            double modelCostChange = 0.0;
            nx.stepIsValid = false;
            if (solved)
            {
                step.resize(size_t(n));
                for (int i = 0; i < n; ++i)
                    step[i] = -y[i];
                jacobianTimes(step, &modelResiduals);
                double dot = 0.0;
                for (int i = 0; i < _nr; ++i)
                    dot += modelResiduals[i] * (_r[i] + modelResiduals[i] / 2.0);
                modelCostChange = -dot;
                nx.stepIsValid = modelCostChange > 0.0;
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
            for (int i = 0; i < n; ++i)
                delta[i] = step[i] * _scale[i];
            // with bounds, Ceres searches along the step (max_num_line_search_step_size_iterations > 0)
            if (_constrained && _opt.lineSearchMaxIterations > 0)
                lineSearch(x, xCost, &delta);

            // ComputeCandidatePointAndEvaluateCost
            plus(x, delta, &candidate);
            double candidateCost = std::numeric_limits<double>::max();
            {
                const std::vector<double> keep = x;
                setState(candidate);
                double c = 0.0;
                if (evaluate(Mode::Cost, &c))
                    candidateCost = c;
                setState(keep);
            }

            // ParameterToleranceReached (after at least one successful step)
            double xNorm = 0.0, stepNorm = 0.0;
            for (size_t i = 0; i < x.size(); ++i)
            {
                xNorm += x[i] * x[i];
                stepNorm += (x[i] - candidate[i]) * (x[i] - candidate[i]);
            }
            xNorm = std::sqrt(xNorm);
            nx.stepNorm = std::sqrt(stepNorm);
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
                const double rd = (currentCost - candidateCost) / modelCostChange;
                const double hd = (referenceCost - candidateCost) / (accumulatedReference + modelCostChange);
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
                accumulatedCandidate += modelCostChange;
                accumulatedReference += modelCostChange;
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
      << ", max rel diff " << paramMaxRel;
    return o.str();
}

}  // namespace own
}  // namespace cheshire
}  // namespace sfm
}  // namespace aliceVision
