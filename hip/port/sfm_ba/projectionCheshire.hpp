// This file is part of the Cheshire project (MPL-2.0), a patch set over AliceVision.
// Copied by scripts/apply_hip_patch.py (step 5m) to src/aliceVision/sfm/bundle/costfunctions/.
//
// Bundle adjustment: the projection residual's Jacobians without the autodiff passes.
//
// Upstream's ProjectionSimpleErrorFunctor (projection.hpp) is a DynamicAutoDiffCostFunction whose
// functor moves the point into the camera frame with Jets and hands it to CostIntrinsicsProject
// through DynamicCostFunctionToFunctorTmp. CostIntrinsicsProject is already analytic - the
// projection and its derivatives with respect to the intrinsics, the distortion and the camera-frame
// point - but Ceres evaluates a dynamic autodiff functor in passes of Stride (4) derivative
// components, and every pass calls CostIntrinsicsProject::Evaluate with all three Jacobian blocks
// again. With the point, the pose and the intrinsics free that is three to five full analytic
// evaluations per residual block per Jacobian, plus the Jet arithmetic and a heap allocation per
// pass (dynamic_cost_function_to_functor.h:127-131).
//
// Three ways to the same numbers, chosen by CHESHIRE_BA_JACOBIANS:
//   autodiff   upstream, unchanged
//   stride     upstream's functor with Stride 32, so one pass (the default). A derivative component
//              is computed by the same operations whatever the stride, so the Jacobian is
//              bit-identical to upstream's: on the 41-view set, 0 of 67,949,760 Jacobian values and
//              0 of 7,089,684 residuals differed over 68 solves, for a Jacobian phase 1.9x faster
//              (13.0 s to 6.9 s; docs/04).
//   analytic   this file: one CostIntrinsicsProject::Evaluate, then the chain rule by hand through
//              the pose (angle-axis + centre) and, for rigs, the sub-pose. The rotation's derivative
//              comes from Ceres' AngleAxisRotatePoint on 3-component Jets - the arithmetic autodiff
//              itself uses for it - so what differs from upstream is only the order of the chain-rule
//              sums, i.e. rounding: on the same set 18.7 % of Jacobian values differ, by at most
//              9.1e-13 absolute and 4.4e-10 relative, residuals identical; the Jacobian phase is
//              3.0x faster (13.0 s to 4.3 s). Opt-in until the quality gate of 0.3.5 exists.
// CHESHIRE_BA_CHECK=1 evaluates a reference cost function next to the selected one on every call and
// reports after each solve how many residual and Jacobian values differed and by how much. The
// reference is upstream's autodiff; when autodiff itself is selected the reference is the analytic
// function, so the report then measures the analytic path while the solver sees upstream's numbers.
#pragma once

#include <aliceVision/sfm/bundle/costfunctions/projection.hpp>
#include <aliceVision/system/Logger.hpp>

#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <ceres/rotation.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace aliceVision {
namespace sfm {
namespace cheshire {

enum class BaJacobians
{
    Autodiff,
    Stride,
    Analytic
};

inline BaJacobians baJacobiansMode()
{
    static const BaJacobians mode = [] {
        const char* v = std::getenv("CHESHIRE_BA_JACOBIANS");
        if (v == nullptr)
            return BaJacobians::Stride;
        const std::string s(v);
        if (s == "autodiff")
            return BaJacobians::Autodiff;
        if (s == "analytic")
            return BaJacobians::Analytic;
        return BaJacobians::Stride;
    }();
    return mode;
}

inline const char* baJacobiansName(BaJacobians m)
{
    switch (m)
    {
        case BaJacobians::Stride:
            return "stride";
        case BaJacobians::Analytic:
            return "analytic";
        default:
            return "autodiff";
    }
}

inline bool baCheckEnabled()
{
    static const bool on = std::getenv("CHESHIRE_BA_CHECK") != nullptr;
    return on;
}

// ---------------------------------------------------------------------------------------------
// A rotation by an angle-axis vector, with the rotated point's derivative w.r.t. that vector.

struct RotationWithDerivative
{
    double R[9];    // column-major, R[i + 3 * j] = R(i, j)
    double Rv[3];   // R v
    double dRv[9];  // dRv[3 * k + j] = d(R v)_k / d theta_j
};

inline void rotateWithDerivative(const double* theta, const double* v, RotationWithDerivative& out)
{
    ceres::AngleAxisToRotationMatrix(theta, out.R);
    using J3 = ceres::Jet<double, 3>;
    J3 th[3], vv[3], res[3];
    for (int k = 0; k < 3; ++k)
    {
        th[k] = J3(theta[k], k);
        vv[k] = J3(v[k]);
    }
    ceres::AngleAxisRotatePoint(th, vv, res);
    for (int k = 0; k < 3; ++k)
    {
        out.Rv[k] = res[k].a;
        for (int j = 0; j < 3; ++j)
            out.dRv[3 * k + j] = res[k].v[j];
    }
}

// out[r][j] = sum_k A[r][k] * M(k, j), A 2x3 row-major, M column-major 3x3 (M[k + 3 * j])
inline void mul2x3ColMajor(const double* A, const double* M, double* out, double sign = 1.0)
{
    for (int r = 0; r < 2; ++r)
        for (int j = 0; j < 3; ++j)
            out[3 * r + j] = sign * (A[3 * r + 0] * M[0 + 3 * j] + A[3 * r + 1] * M[1 + 3 * j] + A[3 * r + 2] * M[2 + 3 * j]);
}

// out[r][j] = sum_k A[r][k] * D[3 * k + j], D row-major 3x3
inline void mul2x3RowMajor(const double* A, const double* D, double* out)
{
    for (int r = 0; r < 2; ++r)
        for (int j = 0; j < 3; ++j)
            out[3 * r + j] = A[3 * r + 0] * D[0 + j] + A[3 * r + 1] * D[3 + j] + A[3 * r + 2] * D[6 + j];
}

inline bool intrinsicHasDistortion(const std::shared_ptr<camera::IntrinsicBase>& intrinsic)
{
    auto isod = camera::IntrinsicScaleOffsetDisto::cast(intrinsic);
    return isod && isod->getDistortion();
}

// ---------------------------------------------------------------------------------------------
// The single-camera projection residual: blocks intrinsics, distortion, pose (angle-axis, centre),
// point. Residual and inner Jacobians from CostIntrinsicsProject, exactly as upstream's functor
// uses it; the pose and point Jacobians by the chain rule.
//
//   v = X - c,  Xc = R(theta) v,  residual = inner(intrinsics, distortion, Xc)
//   d res / d theta = Jp . d(R v)/d theta      d res / d c = -Jp . R      d res / d X = Jp . R

class CostProjectionSimpleAnalytic final : public ceres::CostFunction
{
  public:
    CostProjectionSimpleAnalytic(const sfmData::Observation& obs, const std::shared_ptr<camera::IntrinsicBase>& intrinsic)
      : _inner(obs, intrinsic),
        _hasDistortion(intrinsicHasDistortion(intrinsic))
    {
        const auto& s = _inner.parameter_block_sizes();  // intrinsics, distortion, point
        mutable_parameter_block_sizes()->push_back(s[0]);
        mutable_parameter_block_sizes()->push_back(s[1]);
        mutable_parameter_block_sizes()->push_back(6);
        mutable_parameter_block_sizes()->push_back(3);
        set_num_residuals(2);
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const double* pose = parameters[2];
        const double* X = parameters[3];
        const double v[3] = {X[0] - pose[3], X[1] - pose[4], X[2] - pose[5]};

        if (jacobians == nullptr)
        {
            double Xc[3];
            ceres::AngleAxisRotatePoint(pose, v, Xc);
            const double* inner[3] = {parameters[0], parameters[1], Xc};
            return _inner.Evaluate(inner, residuals, nullptr);
        }

        const bool needPoint = jacobians[2] != nullptr || jacobians[3] != nullptr;
        RotationWithDerivative rot;
        if (needPoint)
            rotateWithDerivative(pose, v, rot);
        else
            ceres::AngleAxisRotatePoint(pose, v, rot.Rv);

        const double* inner[3] = {parameters[0], parameters[1], rot.Rv};
        double Jp[6];
        double* innerJac[3] = {jacobians[0], _hasDistortion ? jacobians[1] : nullptr, needPoint ? Jp : nullptr};
        if (!_inner.Evaluate(inner, residuals, innerJac))
            return false;
        if (jacobians[1] != nullptr && !_hasDistortion)
            std::fill(jacobians[1], jacobians[1] + 2 * parameter_block_sizes()[1], 0.0);

        if (jacobians[2] != nullptr)
        {
            double a[6], b[6];
            mul2x3RowMajor(Jp, rot.dRv, a);
            mul2x3ColMajor(Jp, rot.R, b, -1.0);
            for (int r = 0; r < 2; ++r)
                for (int j = 0; j < 3; ++j)
                {
                    jacobians[2][6 * r + j] = a[3 * r + j];
                    jacobians[2][6 * r + 3 + j] = b[3 * r + j];
                }
        }
        if (jacobians[3] != nullptr)
            mul2x3ColMajor(Jp, rot.R, jacobians[3]);
        return true;
    }

  private:
    CostIntrinsicsProject _inner;
    bool _hasDistortion;
};

// The rig projection residual: blocks intrinsics, distortion, pose, sub-pose, point.
//
//   v = X - c,  w = R1 v,  u = w - c2,  Xc = R2 u
//   d res / d theta1 = Jp . R2 . d(R1 v)/d theta1     d res / d c  = -Jp . R2 . R1
//   d res / d theta2 = Jp . d(R2 u)/d theta2          d res / d c2 = -Jp . R2
//   d res / d X = Jp . R2 . R1

class CostProjectionRigAnalytic final : public ceres::CostFunction
{
  public:
    CostProjectionRigAnalytic(const sfmData::Observation& obs, const std::shared_ptr<camera::IntrinsicBase>& intrinsic)
      : _inner(obs, intrinsic),
        _hasDistortion(intrinsicHasDistortion(intrinsic))
    {
        const auto& s = _inner.parameter_block_sizes();
        mutable_parameter_block_sizes()->push_back(s[0]);
        mutable_parameter_block_sizes()->push_back(s[1]);
        mutable_parameter_block_sizes()->push_back(6);
        mutable_parameter_block_sizes()->push_back(6);
        mutable_parameter_block_sizes()->push_back(3);
        set_num_residuals(2);
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const double* pose = parameters[2];
        const double* sub = parameters[3];
        const double* X = parameters[4];
        const double v[3] = {X[0] - pose[3], X[1] - pose[4], X[2] - pose[5]};

        if (jacobians == nullptr)
        {
            double w[3], u[3], Xc[3];
            ceres::AngleAxisRotatePoint(pose, v, w);
            for (int k = 0; k < 3; ++k)
                u[k] = w[k] - sub[3 + k];
            ceres::AngleAxisRotatePoint(sub, u, Xc);
            const double* inner[3] = {parameters[0], parameters[1], Xc};
            return _inner.Evaluate(inner, residuals, nullptr);
        }

        RotationWithDerivative r1, r2;
        rotateWithDerivative(pose, v, r1);
        double u[3];
        for (int k = 0; k < 3; ++k)
            u[k] = r1.Rv[k] - sub[3 + k];
        rotateWithDerivative(sub, u, r2);

        const double* inner[3] = {parameters[0], parameters[1], r2.Rv};
        const bool needPoint = jacobians[2] != nullptr || jacobians[3] != nullptr || jacobians[4] != nullptr;
        double Jp[6];
        double* innerJac[3] = {jacobians[0], _hasDistortion ? jacobians[1] : nullptr, needPoint ? Jp : nullptr};
        if (!_inner.Evaluate(inner, residuals, innerJac))
            return false;
        if (jacobians[1] != nullptr && !_hasDistortion)
            std::fill(jacobians[1], jacobians[1] + 2 * parameter_block_sizes()[1], 0.0);
        if (!needPoint)
            return true;

        double JpR2[6];
        mul2x3ColMajor(Jp, r2.R, JpR2);
        if (jacobians[2] != nullptr)
        {
            double a[6], b[6];
            mul2x3RowMajor(JpR2, r1.dRv, a);
            mul2x3ColMajor(JpR2, r1.R, b, -1.0);
            for (int r = 0; r < 2; ++r)
                for (int j = 0; j < 3; ++j)
                {
                    jacobians[2][6 * r + j] = a[3 * r + j];
                    jacobians[2][6 * r + 3 + j] = b[3 * r + j];
                }
        }
        if (jacobians[3] != nullptr)
        {
            double a[6];
            mul2x3RowMajor(Jp, r2.dRv, a);
            for (int r = 0; r < 2; ++r)
                for (int j = 0; j < 3; ++j)
                {
                    jacobians[3][6 * r + j] = a[3 * r + j];
                    jacobians[3][6 * r + 3 + j] = -JpR2[3 * r + j];
                }
        }
        if (jacobians[4] != nullptr)
            mul2x3ColMajor(JpR2, r1.R, jacobians[4]);
        return true;
    }

  private:
    CostIntrinsicsProject _inner;
    bool _hasDistortion;
};

// ---------------------------------------------------------------------------------------------
// Upstream's functors at a stride of 32: one derivative pass instead of three to five.

template<typename Functor, int Stride>
ceres::CostFunction* createAutodiffStrided(bool rig, const std::shared_ptr<camera::IntrinsicBase>& intrinsic, const sfmData::Observation& observation)
{
    auto* cost = new ceres::DynamicAutoDiffCostFunction<Functor, Stride>(new Functor(observation, intrinsic));
    int distortionSize = 1;
    auto isod = camera::IntrinsicScaleOffsetDisto::cast(intrinsic);
    if (isod && isod->getDistortion())
        distortionSize = static_cast<int>(isod->getDistortion()->getParameters().size());
    cost->AddParameterBlock(static_cast<int>(intrinsic->getParameters().size()));
    cost->AddParameterBlock(distortionSize);
    cost->AddParameterBlock(6);
    if (rig)
        cost->AddParameterBlock(6);
    cost->AddParameterBlock(3);
    cost->SetNumResiduals(2);
    return cost;
}

// ---------------------------------------------------------------------------------------------
// The check: a reference cost function evaluated next to the selected one.

struct BaCheckStats
{
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> jacobianCalls{0};
    std::atomic<std::uint64_t> failures{0};
    std::atomic<std::uint64_t> residualValues{0};
    std::atomic<std::uint64_t> residualDiffer{0};
    std::atomic<std::uint64_t> jacobianValues{0};
    std::atomic<std::uint64_t> jacobianDiffer{0};
    std::atomic<double> residualMaxAbs{0.0};
    std::atomic<double> residualMaxRel{0.0};
    std::atomic<double> jacobianMaxAbs{0.0};
    std::atomic<double> jacobianMaxRel{0.0};

    void reset()
    {
        calls = jacobianCalls = failures = residualValues = residualDiffer = jacobianValues = jacobianDiffer = 0;
        residualMaxAbs = residualMaxRel = jacobianMaxAbs = jacobianMaxRel = 0.0;
    }
};

inline BaCheckStats& baCheckStats()
{
    static BaCheckStats stats;
    return stats;
}

inline void atomicMax(std::atomic<double>& m, double value)
{
    double cur = m.load(std::memory_order_relaxed);
    while (value > cur && !m.compare_exchange_weak(cur, value, std::memory_order_relaxed))
    {
    }
}

inline void compareValues(const double* a, const double* b, std::size_t n, std::atomic<std::uint64_t>& values, std::atomic<std::uint64_t>& differ,
                          std::atomic<double>& maxAbs, std::atomic<double>& maxRel)
{
    std::uint64_t d = 0;
    double mAbs = 0.0, mRel = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (a[i] == b[i] || (std::isnan(a[i]) && std::isnan(b[i])))
            continue;
        ++d;
        const double diff = std::fabs(a[i] - b[i]);
        const double scale = std::max(std::fabs(a[i]), std::fabs(b[i]));
        mAbs = std::max(mAbs, diff);
        mRel = std::max(mRel, scale > 0.0 ? diff / scale : diff);
    }
    values.fetch_add(n, std::memory_order_relaxed);
    if (d != 0)
    {
        differ.fetch_add(d, std::memory_order_relaxed);
        atomicMax(maxAbs, mAbs);
        atomicMax(maxRel, mRel);
    }
}

class CheckedCostFunction final : public ceres::CostFunction
{
  public:
    CheckedCostFunction(ceres::CostFunction* selected, ceres::CostFunction* reference)
      : _selected(selected),
        _reference(reference)
    {
        *mutable_parameter_block_sizes() = selected->parameter_block_sizes();
        set_num_residuals(selected->num_residuals());
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const bool ok = _selected->Evaluate(parameters, residuals, jacobians);
        BaCheckStats& st = baCheckStats();
        st.calls.fetch_add(1, std::memory_order_relaxed);

        const auto& sizes = parameter_block_sizes();
        const int nr = num_residuals();
        std::vector<double> refResiduals(nr);
        std::vector<std::vector<double>> refJac(sizes.size());
        std::vector<double*> refJacPtr(sizes.size(), nullptr);
        if (jacobians != nullptr)
        {
            st.jacobianCalls.fetch_add(1, std::memory_order_relaxed);
            for (std::size_t i = 0; i < sizes.size(); ++i)
                if (jacobians[i] != nullptr)
                {
                    refJac[i].resize(static_cast<std::size_t>(nr) * sizes[i]);
                    refJacPtr[i] = refJac[i].data();
                }
        }
        const bool refOk = _reference->Evaluate(parameters, refResiduals.data(), jacobians != nullptr ? refJacPtr.data() : nullptr);
        if (ok != refOk)
        {
            st.failures.fetch_add(1, std::memory_order_relaxed);
            return ok;
        }
        if (!ok)
            return false;
        compareValues(residuals, refResiduals.data(), nr, st.residualValues, st.residualDiffer, st.residualMaxAbs, st.residualMaxRel);
        if (jacobians != nullptr)
            for (std::size_t i = 0; i < sizes.size(); ++i)
                if (jacobians[i] != nullptr)
                    compareValues(jacobians[i], refJac[i].data(), refJac[i].size(), st.jacobianValues, st.jacobianDiffer, st.jacobianMaxAbs,
                                  st.jacobianMaxRel);
        return true;
    }

  private:
    std::unique_ptr<ceres::CostFunction> _selected;
    std::unique_ptr<ceres::CostFunction> _reference;
};

// One line per solve, then the counters start again.
inline std::string baCheckReport()
{
    BaCheckStats& st = baCheckStats();
    std::ostringstream os;
    const std::uint64_t rv = st.residualValues.load(), rd = st.residualDiffer.load(), jv = st.jacobianValues.load(), jd = st.jacobianDiffer.load();
    os << st.calls.load() << " evaluations (" << st.jacobianCalls.load() << " with Jacobians): residuals " << rd << " of " << rv << " values differ";
    if (rd != 0)
        os << " (max abs " << st.residualMaxAbs.load() << ", max rel " << st.residualMaxRel.load() << ")";
    os << "; Jacobians " << jd << " of " << jv << " values differ";
    if (jd != 0)
        os << " (max abs " << st.jacobianMaxAbs.load() << ", max rel " << st.jacobianMaxRel.load() << ")";
    if (st.failures.load() != 0)
        os << "; " << st.failures.load() << " evaluations where only one side failed";
    os << (rd == 0 && jd == 0 && st.failures.load() == 0 ? " - identical" : "");
    st.reset();
    return os.str();
}

// ---------------------------------------------------------------------------------------------

inline ceres::CostFunction* makeProjectionCost(BaJacobians mode, bool rig, const std::shared_ptr<camera::IntrinsicBase>& intrinsic,
                                               const sfmData::Observation& observation)
{
    switch (mode)
    {
        case BaJacobians::Stride:
            return rig ? createAutodiffStrided<ProjectionErrorFunctor, 32>(true, intrinsic, observation)
                       : createAutodiffStrided<ProjectionSimpleErrorFunctor, 32>(false, intrinsic, observation);
        case BaJacobians::Analytic:
            return rig ? static_cast<ceres::CostFunction*>(new CostProjectionRigAnalytic(observation, intrinsic))
                       : static_cast<ceres::CostFunction*>(new CostProjectionSimpleAnalytic(observation, intrinsic));
        default:
            return rig ? ProjectionErrorFunctor::createCostFunction(intrinsic, observation)
                       : ProjectionSimpleErrorFunctor::createCostFunction(intrinsic, observation);
    }
}

// What BundleAdjustmentCeres calls in place of the two createCostFunction calls.
inline ceres::CostFunction* createProjectionCost(bool rig, const std::shared_ptr<camera::IntrinsicBase>& intrinsic, const sfmData::Observation& observation)
{
    const BaJacobians mode = baJacobiansMode();
    static const bool announced = [mode] {
        ALICEVISION_LOG_INFO("cheshire: BA jacobians: " << baJacobiansName(mode) << " (CHESHIRE_BA_JACOBIANS=stride|analytic|autodiff, stride is the default), check "
                                                        << (baCheckEnabled() ? "on" : "off") << " (CHESHIRE_BA_CHECK=1)");
        return true;
    }();
    (void)announced;
    ceres::CostFunction* selected = makeProjectionCost(mode, rig, intrinsic, observation);
    if (!baCheckEnabled())
        return selected;
    const BaJacobians reference = (mode == BaJacobians::Autodiff) ? BaJacobians::Analytic : BaJacobians::Autodiff;
    return new CheckedCostFunction(selected, makeProjectionCost(reference, rig, intrinsic, observation));
}

inline std::string baCheckReferenceName()
{
    return baJacobiansMode() == BaJacobians::Autodiff ? "analytic" : "autodiff";
}

}  // namespace cheshire
}  // namespace sfm
}  // namespace aliceVision
