// This file is part of the Cheshire patch set for AliceVision (https://github.com/dspl1236/Cheshire).
// Copied by scripts/apply_hip_patch.py (step 5m) to src/aliceVision/sfm/bundle/costfunctions/.
//
// 0.3.7, step 4b of Cheshire's own bundle-adjustment solver (docs/notes/ba-own-solver.md): the arithmetic
// of a projection residual and of the landmark elimination, in one place for the host solver and the
// device. Every function here is plain arithmetic with no contraction (the pragma below; CUDA builds the
// device side with --fmad=false), so the host and the device compute the same bytes, and so do the
// Windows (clang-cl, AVX2) and Linux (GCC, generic x86-64) hosts.
//
// Before 4b the rotation came from Ceres' rotation.h and the 3x3 inverse from Eigen, both parsed before
// the projection's pragma: the Windows build rounded AngleAxisRotatePoint<double>,
// AngleAxisToRotationMatrix and Eigen's cofactors with FMA, the Linux build did not. The formulas here
// are theirs, operation by operation, without contraction.
//
// The angle-axis rotation has a part per pose (hypot, sin, cos, the Jets of the angle) and a part per
// row. The per-pose part uses libm, so only the host computes it (poseRotation, in the host headers);
// the device receives it. Everything per row is here.
#pragma once

#ifdef __clang__
#pragma clang fp contract(off)
#endif

#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define CHESHIRE_BA_HD __host__ __device__
#else
#define CHESHIRE_BA_HD
#endif

namespace aliceVision {
namespace sfm {
namespace cheshire {
namespace arith {

CHESHIRE_BA_HD inline double baSqrt(double x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::sqrt(x);
#else
    return std::sqrt(x);
#endif
}

CHESHIRE_BA_HD inline bool baFinite(double x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return isfinite(x);
#else
    return std::isfinite(x);
#endif
}

// ---- ceres::Jet<double, 3>, operator by operator (jet.h) -------------------------------------------
struct J3
{
    double a;
    double v[3];
};

CHESHIRE_BA_HD inline J3 j3(double a)
{
    J3 r;
    r.a = a;
    r.v[0] = 0.0;
    r.v[1] = 0.0;
    r.v[2] = 0.0;
    return r;
}

CHESHIRE_BA_HD inline J3 j3(double a, int k)
{
    J3 r = j3(a);
    r.v[k] = 1.0;
    return r;
}

// f + g: Jet(f.a + g.a, f.v + g.v)
CHESHIRE_BA_HD inline J3 jadd(const J3& f, const J3& g)
{
    J3 r;
    r.a = f.a + g.a;
    for (int i = 0; i < 3; ++i)
        r.v[i] = f.v[i] + g.v[i];
    return r;
}

// f - g: Jet(f.a - g.a, f.v - g.v)
CHESHIRE_BA_HD inline J3 jsub(const J3& f, const J3& g)
{
    J3 r;
    r.a = f.a - g.a;
    for (int i = 0; i < 3; ++i)
        r.v[i] = f.v[i] - g.v[i];
    return r;
}

// f * g: Jet(f.a * g.a, f.a * g.v + f.v * g.a)
CHESHIRE_BA_HD inline J3 jmul(const J3& f, const J3& g)
{
    J3 r;
    r.a = f.a * g.a;
    for (int i = 0; i < 3; ++i)
        r.v[i] = f.a * g.v[i] + f.v[i] * g.a;
    return r;
}

// f / g: Jet(f.a / g.a, (f.v - (f.a / g.a) g.v) / g.a), through g.a's inverse as Ceres writes it
CHESHIRE_BA_HD inline J3 jdiv(const J3& f, const J3& g)
{
    const double gInv = 1.0 / g.a;
    const double fByG = f.a * gInv;
    J3 r;
    r.a = fByG;
    for (int i = 0; i < 3; ++i)
        r.v[i] = (f.v[i] - fByG * g.v[i]) * gInv;
    return r;
}

// ---- the angle-axis rotation --------------------------------------------------------------------------
// What ceres::AngleAxisRotatePoint (the double and the Jet<3> version, the latter with the angle-axis as
// the derivative's variables) and ceres::AngleAxisToRotationMatrix compute from the pose alone.
struct PoseRotation
{
    int nonzero = 0;       // fpclassify(hypot(aa)) != FP_ZERO: the Rodrigues branch, else the first-order one
    double aa[3] = {0, 0, 0};
    double c = 1.0, s = 0.0;   // cos, sin of the angle
    double w[3] = {0, 0, 0};   // the axis: aa * (1 / angle)
    J3 th[3];                  // the angle-axis as Jets, th[k] = (aa[k], e_k)
    J3 cJ, sJ, wJ[3];          // cos, sin and the axis as Jets
    double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // AngleAxisToRotationMatrix, column-major
};

// The part per pose, on the host only (libm): AngleAxisRotatePoint<double> and <Jet<3>> up to the point,
// and AngleAxisToRotationMatrix.
inline void poseRotation(const double* aa, PoseRotation* out)
{
    PoseRotation& p = *out;
    for (int k = 0; k < 3; ++k)
    {
        p.aa[k] = aa[k];
        p.th[k] = j3(aa[k], k);
    }
    const double theta = std::hypot(aa[0], aa[1], aa[2]);
    p.nonzero = std::fpclassify(theta) != FP_ZERO;
    double* R = p.R;   // R(r, c) = R[r + 3 c]
    if (!p.nonzero)
    {
        // AngleAxisToRotationMatrix at zero: the first-order Taylor expansion
        R[0 + 3 * 0] = 1.0;
        R[1 + 3 * 0] = aa[2];
        R[2 + 3 * 0] = -aa[1];
        R[0 + 3 * 1] = -aa[2];
        R[1 + 3 * 1] = 1.0;
        R[2 + 3 * 1] = aa[0];
        R[0 + 3 * 2] = aa[1];
        R[1 + 3 * 2] = -aa[0];
        R[2 + 3 * 2] = 1.0;
        return;
    }
    // every cos and sin below is of the same angle, so each is computed once
    const double ct = std::cos(theta), st = std::sin(theta);
    // AngleAxisRotatePoint<double>
    p.c = ct;
    p.s = st;
    const double thetaInverse = 1.0 / theta;
    for (int k = 0; k < 3; ++k)
        p.w[k] = aa[k] * thetaInverse;
    // AngleAxisRotatePoint<Jet<3>>: theta = hypot(th0, th1, th2) = (tmp, x.a / tmp * x.v + y.a / tmp * y.v + z.a / tmp * z.v)
    J3 t;
    t.a = theta;
    for (int i = 0; i < 3; ++i)
        t.v[i] = aa[0] / theta * p.th[0].v[i] + aa[1] / theta * p.th[1].v[i] + aa[2] / theta * p.th[2].v[i];
    // cos(t) = (cos(t.a), -sin(t.a) * t.v), sin(t) = (sin(t.a), cos(t.a) * t.v)
    p.cJ.a = ct;
    p.sJ.a = st;
    for (int i = 0; i < 3; ++i)
    {
        p.cJ.v[i] = -st * t.v[i];
        p.sJ.v[i] = ct * t.v[i];
    }
    const J3 tInverse = jdiv(j3(1.0), t);   // T(1.0) / theta with T a Jet
    for (int k = 0; k < 3; ++k)
        p.wJ[k] = jmul(p.th[k], tInverse);
    // AngleAxisToRotationMatrix
    const double wx = aa[0] / theta, wy = aa[1] / theta, wz = aa[2] / theta;
    const double kOne = 1.0;
    R[0 + 3 * 0] = ct + wx * wx * (kOne - ct);
    R[1 + 3 * 0] = wz * st + wx * wy * (kOne - ct);
    R[2 + 3 * 0] = -wy * st + wx * wz * (kOne - ct);
    R[0 + 3 * 1] = wx * wy * (kOne - ct) - wz * st;
    R[1 + 3 * 1] = ct + wy * wy * (kOne - ct);
    R[2 + 3 * 1] = wx * st + wy * wz * (kOne - ct);
    R[0 + 3 * 2] = wy * st + wx * wz * (kOne - ct);
    R[1 + 3 * 2] = -wx * st + wy * wz * (kOne - ct);
    R[2 + 3 * 2] = ct + wz * wz * (kOne - ct);
}

// AngleAxisRotatePoint<double>, the part per point
CHESHIRE_BA_HD inline void rotatePoint(const PoseRotation& p, const double pt[3], double result[3])
{
    if (p.nonzero)
    {
        const double* w = p.w;
        const double wc0 = w[1] * pt[2] - w[2] * pt[1];
        const double wc1 = w[2] * pt[0] - w[0] * pt[2];
        const double wc2 = w[0] * pt[1] - w[1] * pt[0];
        const double tmp = (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (1.0 - p.c);
        result[0] = pt[0] * p.c + wc0 * p.s + w[0] * tmp;
        result[1] = pt[1] * p.c + wc1 * p.s + w[1] * tmp;
        result[2] = pt[2] * p.c + wc2 * p.s + w[2] * tmp;
    }
    else
    {
        const double* a = p.aa;
        const double wc0 = a[1] * pt[2] - a[2] * pt[1];
        const double wc1 = a[2] * pt[0] - a[0] * pt[2];
        const double wc2 = a[0] * pt[1] - a[1] * pt[0];
        result[0] = pt[0] + wc0;
        result[1] = pt[1] + wc1;
        result[2] = pt[2] + wc2;
    }
}

struct RotationWithDerivative
{
    double R[9];    // column-major, R[i + 3 * j] = R(i, j)
    double Rv[3];   // R v
    double dRv[9];  // dRv[3 * k + j] = d(R v)_k / d theta_j
};

// AngleAxisRotatePoint<Jet<double, 3>> of a constant point, the part per point; R from the pose
CHESHIRE_BA_HD inline void rotateWithDerivative(const PoseRotation& p, const double v[3], RotationWithDerivative& out)
{
    for (int i = 0; i < 9; ++i)
        out.R[i] = p.R[i];
    const J3 pt[3] = {j3(v[0]), j3(v[1]), j3(v[2])};
    J3 res[3];
    if (p.nonzero)
    {
        const J3* w = p.wJ;
        const J3 wc[3] = {jsub(jmul(w[1], pt[2]), jmul(w[2], pt[1])), jsub(jmul(w[2], pt[0]), jmul(w[0], pt[2])),
                          jsub(jmul(w[0], pt[1]), jmul(w[1], pt[0]))};
        const J3 tmp = jmul(jadd(jadd(jmul(w[0], pt[0]), jmul(w[1], pt[1])), jmul(w[2], pt[2])), jsub(j3(1.0), p.cJ));
        for (int k = 0; k < 3; ++k)
            res[k] = jadd(jadd(jmul(pt[k], p.cJ), jmul(wc[k], p.sJ)), jmul(w[k], tmp));
    }
    else
    {
        const J3* a = p.th;
        const J3 wc[3] = {jsub(jmul(a[1], pt[2]), jmul(a[2], pt[1])), jsub(jmul(a[2], pt[0]), jmul(a[0], pt[2])),
                          jsub(jmul(a[0], pt[1]), jmul(a[1], pt[0]))};
        for (int k = 0; k < 3; ++k)
            res[k] = jadd(pt[k], wc[k]);
    }
    for (int k = 0; k < 3; ++k)
    {
        out.Rv[k] = res[k].a;
        for (int j = 0; j < 3; ++j)
            out.dRv[3 * k + j] = res[k].v[j];
    }
}

// out[r][j] = sum_k A[r][k] * M(k, j), A 2x3 row-major, M column-major 3x3 (M[k + 3 * j])
CHESHIRE_BA_HD inline void mul2x3ColMajor(const double* A, const double* M, double* out, double sign = 1.0)
{
    for (int r = 0; r < 2; ++r)
        for (int j = 0; j < 3; ++j)
            out[3 * r + j] = sign * (A[3 * r + 0] * M[0 + 3 * j] + A[3 * r + 1] * M[1 + 3 * j] + A[3 * r + 2] * M[2 + 3 * j]);
}

// out[r][j] = sum_k A[r][k] * D[3 * k + j], D row-major 3x3
CHESHIRE_BA_HD inline void mul2x3RowMajor(const double* A, const double* D, double* out)
{
    for (int r = 0; r < 2; ++r)
        for (int j = 0; j < 3; ++j)
            out[3 * r + j] = A[3 * r + 0] * D[0 + j] + A[3 * r + 1] * D[3 + j] + A[3 * r + 2] * D[6 + j];
}

// ---- the fused projection -----------------------------------------------------------------------------
// What CheshireIntrinsicsProject reads of its camera for a fused model: the scale and the principal point
// (IntrinsicScaleOffset's getters), the distortion's parameters. The host fills it from the camera
// objects after each PrepareForEvaluation.
struct CameraValues
{
    int model = 0;          // 0 none, 1 radial K1, 3 radial K3, 5 Brown
    int nd = 1;             // the distortion block's size (1: the fake block)
    double s0 = 0, s1 = 0, pp0 = 0, pp1 = 0;
    double k[5] = {0, 0, 0, 0, 0};
};

// CheshireIntrinsicsProject::Evaluate for a fused model: blocks intrinsics, distortion, camera-frame point;
// the observation's x, y and scale
CHESHIRE_BA_HD inline bool fusedProjectValues(const CameraValues& cam, double obsX, double obsY, double obsScaleRaw,
                                              double const* const* parameters, double* residuals, double** jacobians)
{
    const double* pt = parameters[2];
    const double x = pt[0], y = pt[1], z = pt[2];
    const double px = x / z, py = y / z;  // Pinhole::project: pt.head<2>() / pt(2)

    // the distortion: value D, dD/dP (row-major 2x2), dD/dk (row-major 2 x nd)
    double D0 = px, D1 = py;
    double dP00 = 1.0, dP01 = 0.0, dP10 = 0.0, dP11 = 1.0;
    double dK[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const int nd = cam.nd;
    if (cam.model == 1)
    {
        const double k1 = cam.k[0];
        const double r2 = px * px + py * py;
        const double r_coeff = 1. + k1 * r2;
        D0 = px * r_coeff;
        D1 = py * r_coeff;
        // Identity * r_coeff + p * [2 k1 px, 2 k1 py]
        dP00 = 1.0 * r_coeff + px * (2.0 * k1 * px);
        dP01 = 0.0 * r_coeff + px * (2.0 * k1 * py);
        dP10 = 0.0 * r_coeff + py * (2.0 * k1 * px);
        dP11 = 1.0 * r_coeff + py * (2.0 * k1 * py);
        dK[0] = px * r2;
        dK[1] = py * r2;
    }
    else if (cam.model == 3)
    {
        const double k1 = cam.k[0], k2 = cam.k[1], k3 = cam.k[2];
        {
            // addDistortion: r via sqrt, then r2 = r * r
            const double r = baSqrt(px * px + py * py);
            const double r2 = r * r;
            const double r4 = r2 * r2;
            const double r6 = r4 * r2;
            const double r_coeff = (1. + k1 * r2 + k2 * r4 + k3 * r6);
            D0 = px * r_coeff;
            D1 = py * r_coeff;
        }
        const double r2 = px * px + py * py;
        if (!(r2 < 1e-21))
        {
            const double r4 = r2 * r2;
            const double r6 = r4 * r2;
            const double r_coeff = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
            const double d_coeff = 2.0 * k1 + 4.0 * k2 * r2 + 6.0 * k3 * r4;
            dP00 = r_coeff + px * d_coeff * px;
            dP01 = px * d_coeff * py;
            dP10 = py * d_coeff * px;
            dP11 = r_coeff + py * d_coeff * py;
            dK[0] = px * r2;
            dK[1] = px * r4;
            dK[2] = px * r6;
            dK[3] = py * r2;
            dK[4] = py * r4;
            dK[5] = py * r6;
        }
    }
    else if (cam.model == 5)
    {
        const double k1 = cam.k[0], k2 = cam.k[1], k3 = cam.k[2], t1 = cam.k[3], t2 = cam.k[4];
        const double r2 = px * px + py * py;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double k_diff = (k1 * r2 + k2 * r4 + k3 * r6);
        const double t_x = t2 * (r2 + 2 * px * px) + 2 * t1 * px * py;
        const double t_y = t1 * (r2 + 2 * py * py) + 2 * t2 * px * py;
        D0 = px + px * k_diff + t_x;
        D1 = py + py * k_diff + t_y;
        dP00 = k1 * r2 + k2 * r4 + k3 * r6 + 2 * px * px * (k1 + 2 * k2 * r2 + 3 * k3 * r4) + 6 * px * t2 + 2 * py * t1 + 1;
        dP01 = 2 * px * py * (k1 + 2 * k2 * r2 + 3 * k3 * r4) + 2 * px * t1 + 2 * py * t2;
        dP10 = 2 * px * py * (k1 + 2 * k2 * r2 + 3 * k3 * r4) + 2 * px * t1 + 2 * py * t2;
        dP11 = k1 * r2 + k2 * r4 + k3 * r6 + 2 * px * t2 + 2 * py * py * (k1 + 2 * k2 * r2 + 3 * k3 * r4) + 6 * py * t1 + 1;
        dK[0] = px * r2;
        dK[1] = px * r4;
        dK[2] = px * r6;
        dK[3] = 2 * px * py;
        dK[4] = 3 * px * px + py * py;
        dK[5] = py * r2;
        dK[6] = py * r4;
        dK[7] = py * r6;
        dK[8] = px * px + 3 * py * py;
        dK[9] = 2 * px * py;
    }

    // cam2ima: p.cwiseProduct(scale) + principal point
    const double s0 = cam.s0, s1 = cam.s1;
    const double u = D0 * s0 + cam.pp0;
    const double v = D1 * s1 + cam.pp1;
    const double obsScale = (obsScaleRaw > 1e-12) ? obsScaleRaw : 1.0;
    residuals[0] = (u - obsX) / obsScale;
    residuals[1] = (v - obsY) / obsScale;
    if (jacobians == nullptr)
        return true;
    const double inv = 1.0 / obsScale;

    if (jacobians[0] != nullptr)
    {
        // d cam2ima / d [scale0, scale1, offset0, offset1] = [D0 0 1 0; 0 D1 0 1]
        double* J = jacobians[0];
        J[0] = inv * D0;
        J[1] = 0.0;
        J[2] = inv * 1.0;
        J[3] = 0.0;
        J[4] = 0.0;
        J[5] = inv * D1;
        J[6] = 0.0;
        J[7] = inv * 1.0;
    }
    if (jacobians[1] != nullptr)
    {
        double* J = jacobians[1];
        if (nd == 1 && cam.model == 0)
        {
            J[0] = 0.0;
            J[1] = 0.0;
        }
        else
        {
            // diag(scale) * dD/dk
            for (int j = 0; j < nd; ++j)
            {
                J[j] = inv * (s0 * dK[j]);
                J[nd + j] = inv * (s1 * dK[nd + j]);
            }
        }
    }
    if (jacobians[2] != nullptr)
    {
        // diag(scale) * dD/dP * dP/dX, dP/dX = [invz 0 -x invz^2; 0 invz -y invz^2]
        const double invz = 1.0 / z;
        const double invzsq = invz * invz;
        const double mxinvzsq = -x * invzsq;
        const double myinvzsq = -y * invzsq;
        const double a00 = dP00 * invz, a01 = dP01 * invz, a02 = dP00 * mxinvzsq + dP01 * myinvzsq;
        const double a10 = dP10 * invz, a11 = dP11 * invz, a12 = dP10 * mxinvzsq + dP11 * myinvzsq;
        double* J = jacobians[2];
        J[0] = inv * (s0 * a00);
        J[1] = inv * (s0 * a01);
        J[2] = inv * (s0 * a02);
        J[3] = inv * (s1 * a10);
        J[4] = inv * (s1 * a11);
        J[5] = inv * (s1 * a12);
    }
    return true;
}

// ---- the single-camera projection residual --------------------------------------------------------------
// Blocks intrinsics, distortion, pose (angle-axis, centre), point. Residual and inner Jacobians from the
// inner projection (CheshireIntrinsicsProject's Evaluate), the pose and point Jacobians by the chain rule:
//
//   v = X - c,  Xc = R(theta) v,  residual = inner(intrinsics, distortion, Xc)
//   d res / d theta = Jp . d(R v)/d theta      d res / d c = -Jp . R      d res / d X = Jp . R
template <typename Inner>
CHESHIRE_BA_HD inline bool projectSimpleRot(const Inner& proj, const PoseRotation& rot, bool hasDistortion, int distortionSize,
                                            double const* const* parameters, double* residuals, double** jacobians)
{
    const double* pose = parameters[2];
    const double* X = parameters[3];
    const double v[3] = {X[0] - pose[3], X[1] - pose[4], X[2] - pose[5]};

    if (jacobians == nullptr)
    {
        double Xc[3];
        rotatePoint(rot, v, Xc);
        const double* inner[3] = {parameters[0], parameters[1], Xc};
        return proj.Evaluate(inner, residuals, nullptr);
    }

    const bool needPoint = jacobians[2] != nullptr || jacobians[3] != nullptr;
    RotationWithDerivative r;
    if (needPoint)
        rotateWithDerivative(rot, v, r);
    else
        rotatePoint(rot, v, r.Rv);

    const double* inner[3] = {parameters[0], parameters[1], r.Rv};
    double Jp[6];
    double* innerJac[3] = {jacobians[0], hasDistortion ? jacobians[1] : nullptr, needPoint ? Jp : nullptr};
    if (!proj.Evaluate(inner, residuals, innerJac))
        return false;
    if (jacobians[1] != nullptr && !hasDistortion)
        for (int i = 0; i < 2 * distortionSize; ++i)
            jacobians[1][i] = 0.0;

    if (jacobians[2] != nullptr)
    {
        double a[6], b[6];
        mul2x3RowMajor(Jp, r.dRv, a);
        mul2x3ColMajor(Jp, r.R, b, -1.0);
        for (int rr = 0; rr < 2; ++rr)
            for (int j = 0; j < 3; ++j)
            {
                jacobians[2][6 * rr + j] = a[3 * rr + j];
                jacobians[2][6 * rr + 3 + j] = b[3 * rr + j];
            }
    }
    if (jacobians[3] != nullptr)
        mul2x3ColMajor(Jp, r.R, jacobians[3]);
    return true;
}

// ---- the loss --------------------------------------------------------------------------------------------
// ceres::HuberLoss::Evaluate (loss_function.cc), a the scale, b = a * a
CHESHIRE_BA_HD inline void huber(double a, double b, double s, double rho[3])
{
    if (s > b)
    {
        const double r = baSqrt(s);
        rho[0] = 2.0 * a * r - b;
        const double q = a / r;
        rho[1] = q > 2.2250738585072014e-308 ? q : 2.2250738585072014e-308;   // std::max(DBL_MIN, a / r)
        rho[2] = -rho[1] / (2.0 * s);
    }
    else
    {
        rho[0] = s;
        rho[1] = 1.0;
        rho[2] = 0.0;
    }
}

// Ceres' corrector (corrector.cc): scale by sqrt(rho'), and the rank-one curvature correction in the
// inlier region only when rho'' > 0 (never for Huber).
struct Corrector
{
    double sqrtRho1 = 1.0, residualScaling = 1.0, alphaSqNorm = 0.0;
    CHESHIRE_BA_HD Corrector(double sqNorm, const double rho[3])
    {
        sqrtRho1 = baSqrt(rho[1]);
        if (sqNorm == 0.0 || rho[2] <= 0.0)
        {
            residualScaling = sqrtRho1;
            alphaSqNorm = 0.0;
            return;
        }
        const double D = 1.0 + 2.0 * sqNorm * rho[2] / rho[1];
        const double alpha = 1.0 - baSqrt(D);
        residualScaling = sqrtRho1 / (1 - alpha);
        alphaSqNorm = alpha / sqNorm;
    }
    CHESHIRE_BA_HD void correctResiduals(int n, double* r) const
    {
        for (int i = 0; i < n; ++i)
            r[i] *= residualScaling;
    }
    // J := sqrt(rho1) * (J - alpha/|r|^2 * r r^T J), row-major n x c (corrector.cc CorrectJacobian)
    CHESHIRE_BA_HD void correctJacobian(int n, int c, const double* r, double* J) const
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

// ---- the landmark elimination ----------------------------------------------------------------------------
// Eigen's 3x3 inverse (InverseImpl.h, compute_inverse<..., 3>): the cofactors of column 0, the determinant
// from them (Eigen's unrolled sum: c0 m00 + (c1 m10 + c2 m20)), then every cofactor times 1 / det.
// m and out row-major.
CHESHIRE_BA_HD inline double cofactor3(const double* m, int i, int j)
{
    const int i1 = (i + 1) % 3, i2 = (i + 2) % 3, j1 = (j + 1) % 3, j2 = (j + 2) % 3;
    return m[i1 * 3 + j1] * m[i2 * 3 + j2] - m[i1 * 3 + j2] * m[i2 * 3 + j1];
}

CHESHIRE_BA_HD inline void inverse3(const double* m, double* out)
{
    const double c0 = cofactor3(m, 0, 0), c1 = cofactor3(m, 1, 0), c2 = cofactor3(m, 2, 0);
    const double det = c0 * m[0] + (c1 * m[3] + c2 * m[6]);
    const double invdet = 1.0 / det;
    // compute_inverse_size3_helper: the result is the adjugate (cofactor (i, j) at (j, i)) over det
    const double c01 = cofactor3(m, 0, 1) * invdet;
    const double c11 = cofactor3(m, 1, 1) * invdet;
    const double c02 = cofactor3(m, 0, 2) * invdet;
    out[1 * 3 + 2] = cofactor3(m, 2, 1) * invdet;
    out[2 * 3 + 1] = cofactor3(m, 1, 2) * invdet;
    out[2 * 3 + 2] = cofactor3(m, 2, 2) * invdet;
    out[1 * 3 + 0] = c01;
    out[1 * 3 + 1] = c11;
    out[2 * 3 + 0] = c02;
    out[0 * 3 + 0] = c0 * invdet;
    out[0 * 3 + 1] = c1 * invdet;
    out[0 * 3 + 2] = c2 * invdet;
}

}  // namespace arith
}  // namespace cheshire
}  // namespace sfm
}  // namespace aliceVision
