// Step 13a's header (hip/port/acr_lanes/cheshireJacobiSvd9.hpp) as the solver uses it: cheshireNullspace2Lanes against
// AliceVision's Nullspace2 (Eigen::JacobiSVD<Mat9>) on 7-point systems, n = 1..4 per call, bit for bit; then timed.
// Build: build-lanestest.cmd. lanestest [count] [seed]; exit code 1 if anything differs.
#include "../../port/acr_lanes/cheshireJacobiSvd9.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using Mat9 = Eigen::Matrix<double, 9, 9>;
using Vec9 = Eigen::Matrix<double, 9, 1>;

// numeric/algebra.hpp's Nullspace2, as upstream
static void nullspace2(const Mat9& A, Vec9& x1, Vec9& x2)
{
    Eigen::JacobiSVD<Mat9> svd(A, Eigen::ComputeFullV);
    const auto& V = svd.matrixV();
    x1 = V.col(V.cols() - 1);
    x2 = V.col(V.cols() - 2);
}

int main(int argc, char** argv)
{
    const long long count = argc > 1 ? std::atoll(argv[1]) : 400000;
    std::uint64_t st = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 7;
    if (!aliceVision::cheshireSvdLanesOn())
    {
        std::printf("lanes off\n");
        return 1;
    }
    std::vector<Mat9> in(count);
    for (long long k = 0; k < count; ++k)
        in[k] = aliceVision::cheshireSvdLanes::selfTestMatrix(st, k % 500 < 7 ? int(k % 500) : 0);
    long long differ = 0;
    for (long long k = 0; k + 4 <= count; k += 4)
    {
        const int n = 1 + int(k / 4 % 4);
        const Mat9* A[4] = {&in[k], &in[k + 1], &in[k + 2], &in[k + 3]};
        Vec9 a1[4], a2[4];
        aliceVision::cheshireNullspace2Lanes(A, n, a1, a2);
        for (int l = 0; l < n; ++l)
        {
            Vec9 e1, e2;
            nullspace2(in[k + l], e1, e2);
            if (std::memcmp(e1.data(), a1[l].data(), sizeof(double) * 9) != 0 || std::memcmp(e2.data(), a2[l].data(), sizeof(double) * 9) != 0)
                ++differ;
        }
    }
    std::printf("%lld systems: %lld nullspaces differ\n", count, differ);
    const long long n = std::min<long long>(count, 200000);
    double sink = 0.0;
    auto t0 = std::chrono::steady_clock::now();
    for (long long k = 0; k < n; ++k)
    {
        Vec9 x1, x2;
        nullspace2(in[k], x1, x2);
        sink += x1(8);
    }
    const double te = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / double(n) * 1e9;
    t0 = std::chrono::steady_clock::now();
    for (long long k = 0; k + 4 <= n; k += 4)
    {
        const Mat9* A[4] = {&in[k], &in[k + 1], &in[k + 2], &in[k + 3]};
        Vec9 a1[4], a2[4];
        aliceVision::cheshireNullspace2Lanes(A, 4, a1, a2);
        sink += a1[0](8);
    }
    const double tl = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / double(n) * 1e9;
    std::printf("ns per nullspace: Eigen %.0f, lanes %.0f (%.2fx)  [%g]\n", te, tl, te / tl, sink);
    return differ ? 1 : 0;
}
