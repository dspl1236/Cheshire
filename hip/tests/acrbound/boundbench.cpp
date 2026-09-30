// Step 8d: AC-RANSAC's rejection bound (acr_bound.inc) against the sort and the full NFA scan it lets a model skip, on
// residual arrays captured from a real run (capture.py): per model the radix sort of the keys and bestNFA, as
// ACRansac.hpp does them, then whether the result beats the running minNFA. The bound has to settle only models whose
// result does not, and the rest go through the sort and the scan as before. build.cmd builds it with the build's
// toolchain; boundbench.exe <capture file> [reps].
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

using ErrorIndex = std::pair<double, size_t>;
inline double acrResidual(std::uint64_t e) { return std::bit_cast<double>(e); }

#include "acr_bound.inc"

template<typename Type>
void makelogcombi(std::size_t k, std::size_t n, std::vector<Type>& vec_logc_k, std::vector<Type>& vec_logc_n)
{
    std::vector<Type> vec_log10(n + 1);
    for (std::size_t i = 0; i <= n; ++i)
        vec_log10[i] = std::log10((Type)i);
    auto logcombi = [&](std::size_t kk, std::size_t nn) {
        if (kk >= nn || kk <= 0)
            return Type(0);
        if (nn - kk < kk)
            kk = nn - kk;
        Type r = 0;
        for (std::size_t i = 1; i <= kk; ++i)
            r += vec_log10[nn - i + 1] - vec_log10[i];
        return r;
    };
    vec_logc_n.resize(n + 1);
    for (std::size_t kk = 0; kk <= n; ++kk)
        vec_logc_n[kk] = logcombi(kk, n);
    vec_logc_k.resize(n + 1);
    for (std::size_t nn = 0; nn <= n; ++nn)
        vec_logc_k[nn] = logcombi(k, nn);
}

// ACRansac.hpp's cheshireRadixSortResiduals, without the deferred-array count
void radixSort(std::vector<std::uint64_t>& keys, std::vector<std::uint64_t>& scratch)
{
    const std::size_t n = keys.size();
    if (n < 2)
        return;
    std::uint64_t* a = keys.data();
    std::uint32_t hist[8][256] = {};
    bool defer = false;
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::uint64_t k = a[i];
        defer |= (k > 0x7FF0000000000000ULL);
        for (int p = 0; p < 8; ++p)
            ++hist[p][(k >> (p * 8)) & 0xFF];
    }
    if (defer)
    {
        std::sort(keys.begin(), keys.end(), [](std::uint64_t x, std::uint64_t y) { return std::bit_cast<double>(x) < std::bit_cast<double>(y); });
        return;
    }
    scratch.resize(n);
    std::uint64_t* b = scratch.data();
    for (int p = 0; p < 8; ++p)
    {
        if (hist[p][(a[0] >> (p * 8)) & 0xFF] == n)
            continue;
        std::uint32_t off[256];
        std::uint32_t sum = 0;
        for (int d = 0; d < 256; ++d)
        {
            off[d] = sum;
            sum += hist[p][d];
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::uint64_t k = a[i];
            b[off[(k >> (p * 8)) & 0xFF]++] = k;
        }
        std::swap(a, b);
    }
    if (a != keys.data())
        std::copy(a, a + n, keys.data());
}

// ACRansac.hpp's bestNFA on keys
ErrorIndex bestNFA(int startIndex, double logalpha0, const std::vector<std::uint64_t>& e, double loge0, double maxThreshold,
                   const std::vector<float>& logc_n, const std::vector<float>& logc_k, double errorVectorDimension)
{
    ErrorIndex bestIndex(std::numeric_limits<double>::infinity(), startIndex);
    const size_t n = e.size();
    for (size_t k = startIndex + 1; k <= n && acrResidual(e[k - 1]) <= maxThreshold; ++k)
    {
        double squaredResidual = acrResidual(e[k - 1]);
        double residual = sqrt(squaredResidual) + std::numeric_limits<float>::epsilon();
        const double logalpha = logalpha0 + errorVectorDimension * log10(residual);
        const double nfa = loge0 + logalpha * (double)(k - startIndex) + logc_n[k] + logc_k[k];
        if (nfa < bestIndex.first)
            bestIndex = ErrorIndex(nfa, k);
    }
    return bestIndex;
}

struct Call
{
    std::vector<float> logc_n, logc_k;
    CheshireAcrBoundSetup setup;
};

struct Model
{
    std::size_t s;
    double logalpha0, loge0, dim, maxT, bestNfa, minNFA;
    std::uint64_t bestK;
    std::vector<double> residuals;
    const Call* call;
};

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "boundbench <capture file> [reps]\n");
        return 2;
    }
    const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
    FILE* f = std::fopen(argv[1], "rb");
    if (!f)
    {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    std::vector<Model> models;
    std::map<std::tuple<std::size_t, std::size_t, double, double, double>, Call> calls;
    for (;;)
    {
        std::uint32_t hdr[2];
        double par[5], minNFA;
        std::uint64_t k;
        if (std::fread(hdr, sizeof hdr, 1, f) != 1)
            break;
        Model m;
        if (std::fread(par, sizeof par, 1, f) != 1 || std::fread(&k, sizeof k, 1, f) != 1 || std::fread(&minNFA, sizeof minNFA, 1, f) != 1)
            return 3;
        m.s = hdr[1];
        m.logalpha0 = par[0], m.loge0 = par[1], m.dim = par[2], m.maxT = par[3], m.bestNfa = par[4], m.bestK = k, m.minNFA = minNFA;
        m.residuals.resize(hdr[0]);
        if (std::fread(m.residuals.data(), sizeof(double), hdr[0], f) != hdr[0])
            return 3;
        Call& c = calls[{m.s, std::size_t(hdr[0]), m.logalpha0, m.loge0, m.dim}];
        if (c.logc_n.empty())
        {
            makelogcombi(m.s, std::size_t(hdr[0]), c.logc_k, c.logc_n);
            c.setup = cheshireAcrBoundSetup(m.s, m.logalpha0, m.loge0, m.dim, c.logc_n, c.logc_k);
        }
        m.call = &c;
        models.push_back(std::move(m));
    }
    std::fclose(f);

    // correctness: bestNFA reproduces the capture, and the bound settles only models that do not beat minNFA
    std::vector<std::uint64_t> keys, scratch;
    long long settled = 0, settledN = 0, totalN = 0, wrong = 0, capture = 0, better = 0;
    for (const Model& m : models)
    {
        keys.resize(m.residuals.size());
        for (size_t i = 0; i < keys.size(); ++i)
            keys[i] = std::bit_cast<std::uint64_t>(m.residuals[i]);
        radixSort(keys, scratch);
        const ErrorIndex full = bestNFA(int(m.s), m.logalpha0, keys, m.loge0, m.maxT, m.call->logc_n, m.call->logc_k, m.dim);
        capture += !(std::bit_cast<std::uint64_t>(full.first) == std::bit_cast<std::uint64_t>(m.bestNfa) && full.second == m.bestK);
        better += full.first < m.minNFA;
        totalN += m.residuals.size();
        if (cheshireAcrCannotBeat(m.call->setup, m.residuals, m.s, m.logalpha0, m.loge0, m.maxT, m.dim, m.minNFA))
        {
            ++settled;
            settledN += m.residuals.size();
            wrong += full.first < m.minNFA;
        }
    }
    std::printf("%zu models (%zu calls), mean n %.0f: bestNFA differs from the capture in %lld; %lld better than minNFA\n",
                models.size(), calls.size(), double(totalN) / models.size(), capture, better);
    std::printf("bound settles %lld (%.2f %% of models, %.2f %% of residuals); settled but better: %lld\n", settled,
                100.0 * settled / models.size(), 100.0 * settledN / totalN, wrong);

    // time: the key path as it is (fill, sort, scan, compare) against the bound first
    double sink = 0;
    double tOld = 1e30, tNew = 1e30;
    for (int r = 0; r < reps; ++r)
    {
        auto t0 = std::chrono::steady_clock::now();
        for (const Model& m : models)
        {
            keys.resize(m.residuals.size());
            for (size_t i = 0; i < keys.size(); ++i)
                keys[i] = std::bit_cast<std::uint64_t>(m.residuals[i]);
            radixSort(keys, scratch);
            const ErrorIndex b = bestNFA(int(m.s), m.logalpha0, keys, m.loge0, m.maxT, m.call->logc_n, m.call->logc_k, m.dim);
            sink += b.first < m.minNFA ? b.first : 0.0;
        }
        auto t1 = std::chrono::steady_clock::now();
        for (const Model& m : models)
        {
            if (cheshireAcrCannotBeat(m.call->setup, m.residuals, m.s, m.logalpha0, m.loge0, m.maxT, m.dim, m.minNFA))
                continue;
            keys.resize(m.residuals.size());
            for (size_t i = 0; i < keys.size(); ++i)
                keys[i] = std::bit_cast<std::uint64_t>(m.residuals[i]);
            radixSort(keys, scratch);
            const ErrorIndex b = bestNFA(int(m.s), m.logalpha0, keys, m.loge0, m.maxT, m.call->logc_n, m.call->logc_k, m.dim);
            sink -= b.first < m.minNFA ? b.first : 0.0;
        }
        auto t2 = std::chrono::steady_clock::now();
        tOld = std::min(tOld, std::chrono::duration<double, std::milli>(t1 - t0).count());
        tNew = std::min(tNew, std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    std::printf("sort + scan %.1f ms, bound first %.1f ms (%.2fx), per model %.3f us -> %.3f us (sink %g)\n", tOld, tNew, tOld / tNew,
                1e3 * tOld / models.size(), 1e3 * tNew / models.size(), sink);

    return (capture || wrong) ? 1 : 0;
}
