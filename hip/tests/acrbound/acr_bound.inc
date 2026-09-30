// Step 8d's rejection bound as scripts/apply_hip_patch.py (STEP8D_ACR) puts it into ACRansac.hpp, without the switches,
// the announcement and the check mode. Kept in step with it by hand.

/// Buckets of the squared residual: a quarter binade each over [2^-54, 2^36), one below (lower edge 0) and one from 2^36
/// up. A key's bucket comes from its top 14 bits (sign, exponent and two mantissa bits).
struct CheshireAcrBuckets
{
    static constexpr int shift = 50;
    static constexpr std::int64_t u0 = std::int64_t(1023 - 54) * 4;  // key >> shift of 2^-54
    static constexpr std::int64_t u1 = std::int64_t(1023 + 36) * 4;  // key >> shift of 2^36
    static constexpr int count = int(u1 - u0) + 2;
    /// below log10(sqrt(e) + FLT_EPSILON) for every e in the bucket: its value at the lower edge, less 1e-12
    double g[count];
    CheshireAcrBuckets()
    {
        for (int b = 0; b < count; ++b)
        {
            const double edge = b == 0 ? 0.0 : std::bit_cast<double>(std::uint64_t(u0 + b - 1) << shift);
            g[b] = std::log10(std::sqrt(edge) + double(std::numeric_limits<float>::epsilon())) - 1e-12;
        }
    }
};

/// What the bound needs of one ACRANSAC call: logc_n[k] + logc_k[k] per k, and the slack its comparisons carry
struct CheshireAcrBoundSetup
{
    std::vector<double> c;
    double slack = 0.0;
    bool usable = false;
};

inline CheshireAcrBoundSetup cheshireAcrBoundSetup(std::size_t s,
                                                   double logalpha0,
                                                   double loge0,
                                                   double dim,
                                                   const std::vector<float>& logc_n,
                                                   const std::vector<float>& logc_k)
{
    CheshireAcrBoundSetup r;
    if (!(dim > 0.0 && std::isfinite(dim) && std::isfinite(logalpha0) && std::isfinite(loge0)) || logc_n.size() < 2 ||
        logc_n.size() != logc_k.size())
        return r;
    const std::size_t n = logc_n.size() - 1;
    r.c.resize(n + 1);
    // gap: how far the tables are from log10 C(n,k) + log10 C(k,s), which the sums below track within 1e-6 + 1e-15 n^2
    double cn = 0.0, ck = 0.0, maxc = 0.0, gap = 0.0;
    for (std::size_t k = 0; k <= n; ++k)
    {
        r.c[k] = double(logc_n[k]) + double(logc_k[k]);
        maxc = std::max(maxc, std::abs(r.c[k]));
        if (k == 0)
            continue;
        cn += std::log10(double(n - k + 1) / double(k));
        if (k > s)
        {
            ck += std::log10(double(k) / double(k - s));
            gap = std::max(gap, std::abs(r.c[k] - (cn + ck)));
        }
    }
    if (!std::isfinite(maxc) || !std::isfinite(gap))
        return r;
    const double margin = 1e-6 + 1e-9 * (std::abs(loge0) + maxc + double(n) * (std::abs(logalpha0) + 8.0 * dim + 1.0));
    r.slack = margin + 2.0 * (gap + 1e-6 + 1e-15 * double(n) * double(n));
    r.usable = true;
    return r;
}

/// True when bestNFA cannot find an nfa below minNFA among these residuals (as kernel.errors left them, unsorted)
inline bool cheshireAcrCannotBeat(const CheshireAcrBoundSetup& setup,
                                  const std::vector<double>& residuals,
                                  std::size_t s,
                                  double logalpha0,
                                  double loge0,
                                  double maxThreshold,
                                  double dim,
                                  double minNFA)
{
    using B = CheshireAcrBuckets;
    static const B buckets;
    if (!setup.usable || !(minNFA < std::numeric_limits<double>::infinity()) || !(maxThreshold >= 0.0) ||
        residuals.size() + 1 != setup.c.size())
        return false;
    const std::size_t n = residuals.size();
    std::uint32_t hist[B::count] = {};
    const std::uint64_t tkey = std::bit_cast<std::uint64_t>(maxThreshold);
    bool odd = false;
    std::size_t last = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::uint64_t key = std::bit_cast<std::uint64_t>(residuals[i]);
        odd |= key > 0x7FF0000000000000ull;  // negative, -0.0 or NaN
        last += key <= tkey;                 // not above maxThreshold
        std::int64_t b = std::int64_t(key >> B::shift) - B::u0 + 1;
        b = b < 0 ? 0 : (b > B::count - 1 ? B::count - 1 : b);
        ++hist[b];
    }
    if (odd)
        return false;
    // bestNFA evaluates k = s+1 .. last (it stops at the first residual above maxThreshold); in sorted order the ranks of
    // bucket b follow those of bucket b-1
    const double limit = minNFA + setup.slack;
    std::size_t r = 0;
    for (int b = 0; b < B::count && r < last; ++b)
    {
        const std::uint32_t cnt = hist[b];
        if (!cnt)
            continue;
        const std::size_t lo = std::max(r + 1, s + 1), hi = std::min(r + cnt, last);
        r += cnt;
        if (lo > hi)
            continue;
        const double a = logalpha0 + dim * buckets.g[b];
        if (loge0 + a * double(lo - s) + setup.c[lo] < limit || loge0 + a * double(hi - s) + setup.c[hi] < limit)
            return false;
    }
    return true;
}
