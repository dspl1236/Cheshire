#!/usr/bin/env python3
"""Capture AC-RANSAC residual arrays for boundbench: a TEMPORARY instrumentation of the development tree's ACRansac.hpp
(third_party/aliceVision, after scripts/apply_hip_patch.py), not part of any build.

  capture.py apply [--every N] [--out FILE]   every Nth model of the key path (default 499) goes to FILE
                                              (default build/acrbound/capture.bin), with bestNFA's answer for it
  capture.py revert

then build/dev-quick.cmd (an incremental build that does not rerun the generator), a run - scripts/fmbench.py, say -
revert, and dev-quick.cmd again. dev-build.cmd would also drop it, since the generator resets ACRansac.hpp.

A record: u32 n, u32 s, f64 logalpha0, f64 loge0, f64 dim, f64 maxThreshold, f64 bestNFA's nfa, u64 its k, f64 the
running minNFA, then the n residuals as kernel.errors left them.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ACR = ROOT / "third_party/aliceVision/src/aliceVision/robustEstimation/ACRansac.hpp"
TAG = "CHESHIRE_CAPTURE_TMP"
INC = "#include <atomic>  // cheshire\n"
INC_NEW = INC + "#include <mutex>  // " + TAG + "\n"
A1 = "template<typename Kernel>\nstd::pair<double, double> ACRANSAC("
A2 = "                    const bool cheshireSkip = cheshireAcrCannotBeat("


def head(every: int, out: str) -> str:
    return f"""// {TAG} begin
inline void cheshireAcrCapture(const std::vector<double>& res, std::size_t s, double logalpha0, double loge0, double maxT, double dim,
                               const std::vector<float>& logc_n, const std::vector<float>& logc_k, double minNFA)
{{
    static std::atomic<long long> seen{{0}};
    if (seen.fetch_add(1) % {every} != 0)
        return;
    std::vector<std::uint64_t> keys(res.size());
    for (std::size_t i = 0; i < res.size(); ++i)
    {{
        keys[i] = std::bit_cast<std::uint64_t>(res[i]);
        if (keys[i] > 0x7FF0000000000000ull)
            return;
    }}
    std::sort(keys.begin(), keys.end());
    const ErrorIndex best = bestNFA(int(s), logalpha0, keys, loge0, maxT, logc_n, logc_k, dim);
    static std::mutex m;
    static FILE* f = std::fopen("{out}", "wb");
    if (!f)
        return;
    std::lock_guard<std::mutex> lock(m);
    const std::uint32_t hdr[2] = {{std::uint32_t(res.size()), std::uint32_t(s)}};
    const double par[5] = {{logalpha0, loge0, dim, maxT, best.first}};
    const std::uint64_t k = best.second;
    std::fwrite(hdr, sizeof hdr, 1, f);
    std::fwrite(par, sizeof par, 1, f);
    std::fwrite(&k, sizeof k, 1, f);
    std::fwrite(&minNFA, sizeof minNFA, 1, f);
    std::fwrite(res.data(), sizeof(double), res.size(), f);
    std::fflush(f);
}}
// {TAG} end

"""


CALL = ("                    cheshireAcrCapture(vec_residuals_, sizeSample, kernel.logalpha0(), loge0, maxThreshold, "
        "kernel.errorVectorDimension(), vec_logc_n, vec_logc_k, minNFA);  // " + TAG + "\n")


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("apply", "revert"):
        sys.exit(__doc__)
    t = ACR.read_text(encoding="utf-8")
    if argv[0] == "apply":
        every = int(argv[argv.index("--every") + 1]) if "--every" in argv else 499
        out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else ROOT / "build/acrbound/capture.bin"
        out.parent.mkdir(parents=True, exist_ok=True)
        if TAG in t:
            sys.exit("already applied")
        if t.count(A1) != 1 or t.count(A2) != 1 or t.count(INC) != 1:
            sys.exit("anchors not found once: is step 8d in the tree?")
        t = t.replace(A1, head(every, out.resolve().as_posix()) + A1, 1).replace(A2, CALL + A2, 1).replace(INC, INC_NEW, 1)
    else:
        a, b = t.find("// " + TAG + " begin\n"), t.find("// " + TAG + " end\n\n")
        if a >= 0 and b > a:
            t = t[:a] + t[b + len("// " + TAG + " end\n\n"):]
        t = t.replace(CALL, "", 1).replace(INC_NEW, INC, 1)
        if TAG in t:
            sys.exit("could not revert every piece")
    ACR.write_text(t, encoding="utf-8", newline="")
    print(argv[0], "done:", ACR)


if __name__ == "__main__":
    main(sys.argv[1:])
