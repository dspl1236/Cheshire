// Step 8e: FundamentalEpipolarDistanceError::error through Eigen, per correspondence as the kernels called it, against
// cheshireEpipolarDistanceErrors (the vectorised loop step 8e puts into FundamentalError.hpp, copied below and kept in
// step with it by hand), bit for bit on random, tiny and huge, zero and degenerate inputs, and timed at n = 333.
// Build it with the flags of the build it vouches for:
//   Windows (clang-cl /arch:AVX2, Eigen's FMA path): build.cmd;  build.cmd /arch:AVX for the path without FMA
//   Linux (GCC, TARGET_ARCHITECTURE=core):  g++ -O3 -DNDEBUG -march=core2 -msse -msse2 -msse3 -std=c++20
//                                           -I<eigen3> epibench.cpp -o epibench
// epibench [seed]; exit code 1 if any residual differs.
#include <Eigen/Core>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

// AliceVision's types (numeric.hpp, 64-bit, AV_EIGEN_MEMORY_ALIGNMENT on)
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix<double, 3, 3>;
using Mat = Eigen::MatrixXd;

template<typename T>
inline T Square(T x)
{
    return x * x;
}

struct Mat3Model
{
    Mat3 _matrix;
    inline const Mat3& getMatrix() const { return _matrix; }
};

// upstream's functor (multiview/relativePose/FundamentalError.hpp)
struct FundamentalEpipolarDistanceError
{
    double error(const Mat3Model& F, const Vec2& x1, const Vec2& x2) const
    {
        const Vec3 x(x1(0), x1(1), 1.0);
        const Vec3 y(x2(0), x2(1), 1.0);
        const Vec3 F_x = F.getMatrix() * x;
        return Square(F_x.dot(y)) / F_x.head<2>().squaredNorm();
    }
};

// the kernels' loop before step 8e
NOINLINE void upstreamErrors(const Mat3Model& model, const Mat& x1, const Mat& x2, std::vector<double>& errors)
{
    FundamentalEpipolarDistanceError est;
    const std::size_t n = x1.cols();
    errors.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        errors[i] = est.error(model, x1.col(i), x2.col(i));
}

// step 8e's loop, as the generator writes it
NOINLINE inline void cheshireEpipolarDistanceErrors(const Mat3& F, const Mat& x1, const Mat& x2, std::vector<double>& out)
{
#ifdef __clang__
#pragma clang fp contract(off)
#endif
    const double f00 = F(0, 0), f10 = F(1, 0), f20 = F(2, 0), f01 = F(0, 1), f11 = F(1, 1), f21 = F(2, 1), f02 = F(0, 2),
                 f12 = F(1, 2), f22 = F(2, 2);
    const std::size_t n = x1.cols();
    out.resize(n);
    const double* __restrict a = x1.data();
    const double* __restrict b = x2.data();
    double* __restrict e = out.data();
    for (std::size_t i = 0; i < n; ++i)
    {
        const double a0 = a[2 * i], a1 = a[2 * i + 1], b0 = b[2 * i], b1 = b[2 * i + 1];
#ifdef EIGEN_VECTORIZE_FMA
        const double u0 = std::fma(a1, f01, f00 * a0);
        const double u1 = std::fma(a1, f11, f10 * a0);
#else
        const double m00 = f00 * a0, m10 = f10 * a0;
        const double m01 = f01 * a1, m11 = f11 * a1;
        const double u0 = m00 + m01;
        const double u1 = m10 + m11;
#endif
        const double fx0 = u0 + f02;
        const double fx1 = u1 + f12;
        const double m21 = f21 * a1;
        const double m20 = f20 * a0;
        const double s2 = m21 + f22;
        const double fx2 = m20 + s2;
        const double p0 = fx0 * b0;
        const double p1 = fx1 * b1;
        const double s = p0 + p1;
        const double d = s + fx2;
        const double q0 = fx0 * fx0;
        const double q1 = fx1 * fx1;
        const double nn = q0 + q1;
        const double dd = d * d;
        e[i] = dd / nn;
    }
}

int main(int argc, char** argv)
{
    std::printf("Eigen %d.%d.%d, %s\n", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION,
#ifdef EIGEN_VECTORIZE_FMA
                "FMA path"
#else
                "path without FMA"
#endif
    );
    std::mt19937_64 rng(argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1);
    std::uniform_real_distribution<double> u(-1.0, 1.0), lg(-12.0, 3.0);
    long long total = 0, differ = 0;
    std::vector<double> e1, e2;
    for (int trial = 0; trial < 20000; ++trial)
    {
        const int n = 8 + int(rng() % 600);
        Mat x1(2, n), x2(2, n);
        const int mode = trial % 4;  // 0 plain, 1 tiny and huge points, 2 zeros and zero rows of F, 3 tiny and huge F
        for (int i = 0; i < n; ++i)
            for (int r = 0; r < 2; ++r)
            {
                x1(r, i) = u(rng);
                x2(r, i) = u(rng);
                if (mode == 1)
                    x1(r, i) *= std::pow(10.0, lg(rng)), x2(r, i) *= std::pow(10.0, lg(rng));
                if (mode == 2 && (rng() % 16) == 0)
                    x1(r, i) = 0.0;
            }
        Mat3Model m;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                m._matrix(r, c) = mode == 3 ? u(rng) * std::pow(10.0, lg(rng)) : u(rng);
        if (mode == 2 && (rng() % 8) == 0)  // 0/0 and x/0
            m._matrix.row(0).setZero(), m._matrix.row(1).setZero();
        upstreamErrors(m, x1, x2, e1);
        cheshireEpipolarDistanceErrors(m.getMatrix(), x1, x2, e2);
        for (int i = 0; i < n; ++i, ++total)
            if (std::bit_cast<std::uint64_t>(e1[i]) != std::bit_cast<std::uint64_t>(e2[i]))
            {
                if (differ < 5)
                    std::printf("differ: mode %d i %d upstream %.17g step 8e %.17g\n", mode, i, e1[i], e2[i]);
                ++differ;
            }
    }
    std::printf("%lld residuals, %lld differ\n", total, differ);

    const int n = 333;  // geometric filtering's mean on 41 views
    Mat x1(2, n), x2(2, n);
    for (int i = 0; i < n; ++i)
        x1(0, i) = u(rng), x1(1, i) = u(rng), x2(0, i) = u(rng), x2(1, i) = u(rng);
    std::vector<Mat3Model> models(1024);
    for (auto& m : models)
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                m._matrix(r, c) = u(rng);
    double sink = 0, tU = 1e30, tC = 1e30;
    for (int rep = 0; rep < 5; ++rep)
    {
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < 200000; ++k)
        {
            upstreamErrors(models[k & 1023], x1, x2, e1);
            sink += e1[k % n];
        }
        auto t1 = std::chrono::steady_clock::now();
        for (int k = 0; k < 200000; ++k)
        {
            cheshireEpipolarDistanceErrors(models[k & 1023].getMatrix(), x1, x2, e2);
            sink += e2[k % n];
        }
        auto t2 = std::chrono::steady_clock::now();
        tU = std::min(tU, std::chrono::duration<double, std::nano>(t1 - t0).count() / (200000.0 * n));
        tC = std::min(tC, std::chrono::duration<double, std::nano>(t2 - t1).count() / (200000.0 * n));
    }
    std::printf("per residual: upstream %.3f ns, step 8e %.3f ns (%.2fx) (sink %g)\n", tU, tC, tU / tC, sink);
    return differ != 0;
}
