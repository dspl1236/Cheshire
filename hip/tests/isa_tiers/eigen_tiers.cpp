// 0.4.5 scoping: does Eigen compute the same bits at x86-64-v1 (SSE2 packets) and x86-64-v3 (AVX packets), with FMA
// contraction off? Prints a hash of the bits of each result; build it twice and compare (run.sh).
#include <Eigen/Dense>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

static uint64_t h = 1469598103934665603ull;
static void mix(const double* p, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        uint64_t b;
        std::memcpy(&b, p + i, 8);
        h = (h ^ b) * 1099511628211ull;
    }
}
template <class M>
static void report(const char* what, const M& m)
{
    h = 1469598103934665603ull;
    mix(m.data(), size_t(m.size()));
    std::printf("%-28s %016llx\n", what, (unsigned long long)h);
}

int main()
{
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> u(-1, 1);
    Eigen::MatrixXd A(97, 61), B(61, 83);
    Eigen::VectorXd x(1001), y(1001);
    for (int i = 0; i < A.size(); ++i) A.data()[i] = u(rng);
    for (int i = 0; i < B.size(); ++i) B.data()[i] = u(rng);
    for (int i = 0; i < x.size(); ++i) { x[i] = u(rng); y[i] = u(rng); }
    Eigen::Matrix<double, 1, 1> d; d(0) = x.dot(y);
    report("VectorXd dot (1001)", d);
    Eigen::Matrix<double, 1, 1> s; s(0) = x.sum();
    report("VectorXd sum", s);
    Eigen::Matrix<double, 1, 1> nn; nn(0) = x.norm();
    report("VectorXd norm", nn);
    Eigen::MatrixXd C = A * B;
    report("MatrixXd product 97x61x83", C);
    Eigen::Matrix3d R = Eigen::Matrix3d::Random(); Eigen::Vector3d v = Eigen::Vector3d::Random();
    Eigen::Vector3d Rv = R * v;
    report("Matrix3d * Vector3d", Rv);
    Eigen::Matrix<double, 9, 9> M9 = Eigen::Matrix<double, 9, 9>::Random();
    Eigen::JacobiSVD<Eigen::Matrix<double, 9, 9>> svd(M9, Eigen::ComputeFullV);
    report("JacobiSVD<Mat9> V", svd.matrixV());
    Eigen::MatrixXd S = A.transpose() * A;
    Eigen::LLT<Eigen::MatrixXd> llt(S);
    report("LLT 61x61", llt.matrixL().toDenseMatrix());
    Eigen::MatrixXd Q = Eigen::HouseholderQR<Eigen::MatrixXd>(A).householderQ();
    report("HouseholderQR Q", Q);
    return 0;
}
