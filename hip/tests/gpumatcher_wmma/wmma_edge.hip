// Edge cases for the WMMA 2-NN kernels against the shipped knn2_u8 (both from wmma_matcher.hip): sizes around the 16-row
// n-tiles, the 240-row windows and the query tiles, tiny and empty databases, heavy exact ties (descriptors drawn from a
// few prototypes, so many rows share a distance and the lower row must win), and the ends of the key range (all-0 and
// all-255 descriptors, distance 128 * 255^2). Every variant must give the same rows and distances for every query.
//   wmma_edge.exe [seed=1]
#define WMMA_NO_MAIN
#include "wmma_matcher.hip"
#include <random>

namespace {

std::vector<uint8_t> make(int n, int pattern, std::mt19937& rng)
{
    std::vector<uint8_t> d(size_t(std::max(n, 1)) * 128, 0);
    std::uniform_int_distribution<int> byte(0, 255);
    static std::vector<uint8_t> protos;
    if (protos.empty()) { protos.resize(3 * 128); for (auto& b : protos) b = uint8_t(byte(rng)); }
    for (int r = 0; r < n; ++r) {
        uint8_t* x = &d[size_t(r) * 128];
        switch (pattern) {
        case 0: for (int k = 0; k < 128; ++k) x[k] = uint8_t(byte(rng)); break;                       // random
        case 1: std::memcpy(x, &protos[size_t(rng() % 3) * 128], 128); break;                          // 3 prototypes
        case 2: { const int kind = int(rng() % 3);                                                     // 0, 255, halves
                  for (int k = 0; k < 128; ++k) x[k] = kind == 0 ? 0 : kind == 1 ? 255 : (k < 64 ? 255 : 0); } break;
        case 3: x[rng() % 128] = uint8_t(byte(rng)); x[rng() % 128] = uint8_t(byte(rng)); break;      // sparse
        default: std::memcpy(x, &protos[size_t(rng() % 3) * 128], 128);                               // near duplicates
                 { const int k = int(rng() % 128); if (rng() & 1) x[k] = uint8_t(std::min(255, x[k] + 1)); else x[k] = uint8_t(std::max(0, x[k] - 1)); }
        }
    }
    return d;
}

struct Variant { const char* name; void (*launch)(const unsigned int*, const unsigned int*, int, const unsigned int*, const unsigned int*, int, int*, float*); };

}  // namespace

int main(int argc, char** argv)
{
    std::mt19937 rng(argc > 1 ? unsigned(std::atoi(argv[1])) : 1u);
    const Variant variants[] = {
        {"v6 MT1 WV8", launchC<1, true, 8>}, {"v6 MT2 WV8", launchC<2, true, 8>}, {"v6 MT3 WV8", launchC<3, true, 8>},
        {"v6 MT2 WV4", launchC<2, true, 4>}, {"v6 MT3 WV4", launchC<3, true, 4>}, {"v5 MT2 WV8", launchC<2, false, 8>},
    };
    const int sizes[][2] = {{1, 0}, {5, 0}, {1, 1}, {1, 2}, {2, 1}, {3, 5}, {16, 16}, {17, 15}, {40, 239}, {33, 240},
                            {100, 241}, {257, 255}, {300, 256}, {513, 257}, {129, 480}, {385, 481}, {1000, 1000},
                            {2049, 4097}, {5000, 7201}};
    const char* patterns[] = {"random", "3 prototypes", "0/255/halves", "sparse", "near duplicates"};
    long long runs = 0, bad = 0, queries = 0, ties = 0;
    for (const auto& sz : sizes)
        for (int pat = 0; pat < 5; ++pat) {
            const int nq = sz[0], nr = sz[1];
            const auto Q = make(nq, pat, rng), D = make(nr, pat, rng);
            unsigned int *dQ, *dD, *dQN, *dDN; int *dI1, *dI2; float *dD1, *dD2;
            CHECK(hipMalloc(&dQ, Q.size())); CHECK(hipMalloc(&dD, D.size()));
            CHECK(hipMalloc(&dQN, size_t(nq) * 4)); CHECK(hipMalloc(&dDN, size_t(std::max(nr, 1)) * 4));
            CHECK(hipMalloc(&dI1, size_t(nq) * 8)); CHECK(hipMalloc(&dI2, size_t(nq) * 8));
            CHECK(hipMalloc(&dD1, size_t(nq) * 8)); CHECK(hipMalloc(&dD2, size_t(nq) * 8));
            CHECK(hipMemcpy(dQ, Q.data(), Q.size(), hipMemcpyHostToDevice)); CHECK(hipMemcpy(dD, D.data(), D.size(), hipMemcpyHostToDevice));
            rowNormsU8<128><<<(nq + 255) / 256, 256>>>(dQ, nq, dQN);
            if (nr) rowNormsU8<128><<<(nr + 255) / 256, 256>>>(dD, nr, dDN);
            knn2_u8<128><<<(nq + kQueriesPerBlock - 1) / kQueriesPerBlock, kQueriesPerBlock>>>(dD, dDN, nr, dQ, nq, dI1, dD1);
            CHECK(hipDeviceSynchronize());
            std::vector<int> i1(size_t(nq) * 2), i2(size_t(nq) * 2);
            std::vector<float> d1(size_t(nq) * 2), d2(size_t(nq) * 2);
            CHECK(hipMemcpy(i1.data(), dI1, i1.size() * 4, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(d1.data(), dD1, d1.size() * 4, hipMemcpyDeviceToHost));
            for (int k = 0; k < nq; ++k) ties += i1[2 * k + 1] >= 0 && d1[2 * k] == d1[2 * k + 1];
            queries += nq;
            for (const auto& v : variants) {
                CHECK(hipMemset(dI2, 0x7f, size_t(nq) * 8)); CHECK(hipMemset(dD2, 0x7f, size_t(nq) * 8));
                v.launch(dD, dDN, nr, dQ, dQN, nq, dI2, dD2);
                CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
                CHECK(hipMemcpy(i2.data(), dI2, i2.size() * 4, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(d2.data(), dD2, d2.size() * 4, hipMemcpyDeviceToHost));
                ++runs;
                int diff = 0, first = -1;
                for (int k = 0; k < nq; ++k)
                    if (i1[2 * k] != i2[2 * k] || i1[2 * k + 1] != i2[2 * k + 1] || std::memcmp(&d1[2 * k], &d2[2 * k], 8)) { if (first < 0) first = k; ++diff; }
                if (diff) {
                    ++bad;
                    std::printf("MISMATCH %s, %d x %d, %s: %d queries; first %d: ref (%d %.0f, %d %.0f) got (%d %.0f, %d %.0f)\n", v.name, nq, nr,
                                patterns[pat], diff, first, i1[2 * first], d1[2 * first], i1[2 * first + 1], d1[2 * first + 1],
                                i2[2 * first], d2[2 * first], i2[2 * first + 1], d2[2 * first + 1]);
                }
            }
            hipFree(dQ); hipFree(dD); hipFree(dQN); hipFree(dDN); hipFree(dI1); hipFree(dI2); hipFree(dD1); hipFree(dD2);
        }
    std::printf("%lld runs, %lld with a mismatch; %lld queries per variant, %lld of them with nearest == second distance\n",
                runs, bad, queries, ties);
    return bad ? 1 : 0;
}
