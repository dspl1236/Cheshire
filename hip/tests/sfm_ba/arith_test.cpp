// Step 4b check: baArith.hpp against Ceres' rotation.h / jet.h and Eigen's 3x3 inverse, bit for bit.
// Built twice: with the pragma before every include (Ceres and Eigen then round without FMA, which is what
// baArith.hpp must reproduce exactly), and without it (the Windows build's contraction), to count how many
// values the old host rounded differently.
#ifdef ARITH_NO_CONTRACT
#pragma clang fp contract(off)
#endif
#include <Eigen/Dense>
#include <ceres/jet.h>
#include <ceres/rotation.h>

#include "../../port/sfm_ba/baArith.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

using namespace aliceVision::sfm::cheshire::arith;

static bool same(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

int main()
{
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::uniform_real_distribution<double> Big(-50.0, 50.0);
    long long n = 0, dRot = 0, dJet = 0, dR = 0, dInv = 0, dRotVals = 0, dJetVals = 0, dRVals = 0, dInvVals = 0;
    const int N = 2000000;
    for (int it = 0; it < N; ++it)
    {
        double aa[3], v[3];
        const int kind = it % 10;
        for (int k = 0; k < 3; ++k)
        {
            aa[k] = kind == 0 ? 0.0 : kind == 1 ? U(rng) * 1e-9 : kind == 2 ? U(rng) * 1e-160 : U(rng) * 3.0;
            v[k] = Big(rng);
        }
        PoseRotation p;
        poseRotation(aa, &p);
        // double path
        double ours[3], theirs[3];
        rotatePoint(p, v, ours);
        ceres::AngleAxisRotatePoint(aa, v, theirs);
        bool d = false;
        for (int k = 0; k < 3; ++k)
            if (!same(ours[k], theirs[k]))
            {
                d = true;
                ++dRotVals;
            }
        dRot += d;
        // Jet path and R
        RotationWithDerivative r;
        rotateWithDerivative(p, v, r);
        double R[9];
        ceres::AngleAxisToRotationMatrix(aa, R);
        using J = ceres::Jet<double, 3>;
        J th[3], vv[3], res[3];
        for (int k = 0; k < 3; ++k)
        {
            th[k] = J(aa[k], k);
            vv[k] = J(v[k]);
        }
        ceres::AngleAxisRotatePoint(th, vv, res);
        d = false;
        for (int k = 0; k < 3; ++k)
        {
            if (!same(r.Rv[k], res[k].a))
            {
                d = true;
                ++dJetVals;
            }
            for (int j = 0; j < 3; ++j)
                if (!same(r.dRv[3 * k + j], res[k].v[j]))
                {
                    d = true;
                    ++dJetVals;
                }
        }
        dJet += d;
        d = false;
        for (int i = 0; i < 9; ++i)
            if (!same(r.R[i], R[i]))
            {
                d = true;
                ++dRVals;
            }
        dR += d;
        // 3x3 inverse of a symmetric positive matrix like E^T E + D^2
        double e[9];
        {
            double a[6];
            for (int i = 0; i < 6; ++i)
                a[i] = Big(rng);
            const double diag[3] = {std::abs(U(rng)) * 1e-3, std::abs(U(rng)) * 1e2, std::abs(U(rng))};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    e[i * 3 + j] = a[i] * a[j] + a[3 + i] * a[3 + j] + (i == j ? diag[i] : 0.0);
        }
        double inv[9];
        inverse3(e, inv);
        const Eigen::Matrix3d invM = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(e).inverse();
        d = false;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (!same(inv[i * 3 + j], invM(i, j)))
                {
                    d = true;
                    ++dInvVals;
                }
        dInv += d;
        ++n;
    }
#ifdef ARITH_NO_CONTRACT
    std::printf("contraction off in Ceres and Eigen (must all be 0):\n");
#else
    std::printf("contraction as the Windows build (Ceres and Eigen with FMA):\n");
#endif
    std::printf("  cases %lld\n  rotatePoint differs in %lld cases (%lld values)\n  rotateWithDerivative Rv/dRv differs in %lld (%lld)\n"
                "  R differs in %lld (%lld)\n  inverse3 differs in %lld (%lld)\n",
                n, dRot, dRotVals, dJet, dJetVals, dR, dRVals, dInv, dInvVals);
    return (dRot + dJet + dR + dInv) ? 1 : 0;
}
