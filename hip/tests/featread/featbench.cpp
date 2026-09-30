// 0.3.8 SfM: how long does reading .feat files take, the way AliceVision reads them (istream_iterator over
// operator>> of four floats) against one read of the file and std::from_chars, and do the two give the
// same bits? Same toolchain as the build (clang-cl, MSVC STL; build.cmd). Step 8a, docs/04 "0.3.8".
// Usage: featbench <dir with *.feat> [threads]
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

struct Feat { float v[4]; };
static std::istream& operator>>(std::istream& in, Feat& f) { return in >> f.v[0] >> f.v[1] >> f.v[2] >> f.v[3]; }

static std::vector<Feat> viaStream(const std::string& path) {
    std::vector<Feat> out;
    std::ifstream in(path);
    std::copy(std::istream_iterator<Feat>(in), std::istream_iterator<Feat>(), std::back_inserter(out));
    return out;
}

static std::vector<Feat> viaFromChars(const std::string& path) {
    std::vector<Feat> out;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string buf(size_t(n), '\0');
    const size_t got = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    buf.resize(got);
    const char* p = buf.data();
    const char* e = p + buf.size();
    out.reserve(buf.size() / 32);
    for (;;) {
        Feat ft;
        int k = 0;
        for (; k < 4; ++k) {
            while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
            if (p == e) break;
            auto r = std::from_chars(p, e, ft.v[k]);
            if (r.ec != std::errc()) { p = e; break; }
            p = r.ptr;
        }
        if (k < 4) break;
        out.push_back(ft);
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: featbench <dir> [threads]\n"); return 2; }
    std::vector<std::string> files;
    for (auto& de : std::filesystem::directory_iterator(argv[1]))
        if (de.path().extension() == ".feat") files.push_back(de.path().string());
    std::sort(files.begin(), files.end());
    const int threads = argc > 2 ? std::atoi(argv[2]) : int(std::thread::hardware_concurrency());
    std::vector<std::vector<Feat>> a(files.size()), b(files.size());

    auto timed = [&](const char* name, int nt, auto fn, std::vector<std::vector<Feat>>& dst) {
        std::atomic<size_t> next{0};
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        for (int t = 0; t < nt; ++t)
            pool.emplace_back([&] { for (size_t i; (i = next++) < files.size();) dst[i] = fn(files[i]); });
        for (auto& th : pool) th.join();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%-12s %2d thread%s %7.2f s\n", name, nt, nt == 1 ? " " : "s", s);
    };
    // warm the file cache first, so both methods read from memory
    for (auto& f : files) { std::ifstream in(f, std::ios::binary); std::string s((std::istreambuf_iterator<char>(in)), {}); }
    timed("istream", 1, viaStream, a);
    timed("from_chars", 1, viaFromChars, b);
    timed("istream", threads, viaStream, a);
    timed("from_chars", threads, viaFromChars, b);

    size_t feats = 0, differ = 0, filesDiffer = 0;
    for (size_t i = 0; i < files.size(); ++i) {
        feats += a[i].size();
        if (a[i].size() != b[i].size()) { ++filesDiffer; continue; }
        size_t d = 0;
        for (size_t j = 0; j < a[i].size(); ++j) d += std::memcmp(a[i][j].v, b[i][j].v, sizeof(Feat)) != 0;
        differ += d; filesDiffer += d != 0;
    }
    std::printf("%zu files, %zu features: %zu features differ in %zu files -> %s\n", files.size(), feats, differ,
                filesDiffer, differ == 0 && filesDiffer == 0 ? "BIT-IDENTICAL" : "DIFFERENT");
    return differ || filesDiffer;
}
