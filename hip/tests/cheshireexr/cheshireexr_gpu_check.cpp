// CheshireEXR: the decoder on the device, against OpenEXR, with timings.
//
//   cheshireexr_gpu_check [--reps N] [--threads 1,2,4,8,12] [--workdir DIR] [file.exr ...]
//
// Without files it writes a test set with OpenEXR (every supported compression and layout, edge
// sizes, every half bit pattern, and a 6000x3376 half RGBA ZIP file as PrepareDenseScene writes).
// Every file must decode to OpenEXR's floats bit for bit. Then, per file, median end-to-end times:
// CheshireEXR (read the file, decode, floats in host memory), CheshireEXR to device (the same with the
// floats left on the device, what the depth-map node uses; checked against decode first) against OpenEXR's Imf::InputFile on
// the calling thread (how the depth-map and texturing read-ahead threads read) and with its thread
// pool (a lone read). --threads: images and megapixels per second with one Codec per thread against
// OpenEXR on as many threads, each thread decoding every file --reps times. Only successful decodes
// are counted, and a failed one (a GPU out of memory at a high thread count, say) fails the run.
#include "exrCheckCommon.hpp"
#include "exrCodec.hpp"

#include <OpenEXR/ImfThreading.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using namespace cheshire::exr;
using exrcheck::ChannelSpec;
using Clock = std::chrono::steady_clock;

namespace {

template <class F>
double medianMs(int reps, F&& f)
{
    std::vector<double> t;
    for (int i = 0; i < reps; ++i)
    {
        const auto t0 = Clock::now();
        f();
        t.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int channelsFor(const std::string& path)
{
    const std::vector<uint8_t> f = exrcheck::readFile(path);
    Header h;
    if (readHeader(f.data(), f.size(), h) != Status::Ok)
        return 3;
    return h.channels.size() == 1 ? 1 : (h.channels.size() == 4 ? 4 : 3);
}

Status gpuDecode(Codec& codec, const std::string& path, int nch, std::vector<float>& out, DecodeStats* st = nullptr)
{
    const std::vector<uint8_t> f = exrcheck::readFile(path);
    return codec.decode(f.data(), f.size(), nch,
                        [&](int w, int h) {
                            out.resize((size_t)w * h * nch);
                            return out.data();
                        },
                        st);
}

// decodeToDevice, then Codec::download: must give decode's floats
Status gpuDecodeToDevice(Codec& codec, const std::string& path, int nch, std::vector<float>& out)
{
    const std::vector<uint8_t> f = exrcheck::readFile(path);
    DeviceImage di;
    const Status s = codec.decodeToDevice(f.data(), f.size(), nch, di);
    if (s != Status::Ok)
        return s;
    out.assign(di.rowBytes / sizeof(float) * (size_t)di.height, 0.0f);
    return codec.download(di, out.data());
}

}  // namespace

int main(int argc, char** argv)
{
    int reps = 5;
    std::vector<int> threadCounts;
    std::string dir = (std::filesystem::temp_directory_path() / "cheshireexr_gpu_check").string();
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--reps" && i + 1 < argc)
            reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--workdir" && i + 1 < argc)
            dir = argv[++i];
        else if (a == "--threads" && i + 1 < argc)
        {
            for (const char* p = argv[++i]; *p;)
            {
                threadCounts.push_back(std::max(1, std::atoi(p)));
                while (*p && *p != ',')
                    ++p;
                if (*p == ',')
                    ++p;
            }
        }
        else
            files.push_back(a);
    }
    if (!Codec::deviceAvailable())
    {
        std::printf("no GPU device (or CHESHIRE_EXR_GPU=0)\n");
        return 2;
    }
    Codec codec;
    int cases = 0, fails = 0;

    if (files.empty())
    {
        std::filesystem::create_directories(dir);
        const Imf::Compression comps[] = {Imf::NO_COMPRESSION, Imf::RLE_COMPRESSION, Imf::ZIPS_COMPRESSION, Imf::ZIP_COMPRESSION};
        const std::vector<ChannelSpec> layouts[] = {{{"R", true}, {"G", true}, {"B", true}, {"A", true}},
                                                    {{"R", true}, {"G", false}, {"B", true}},
                                                    {{"Y", false}},
                                                    {{"Y", true}}};
        const int sizes[][2] = {{1, 1}, {7, 3}, {17, 33}, {257, 129}, {640, 97}};
        int n = 0;
        for (const auto& sz : sizes)
            for (Imf::Compression c : comps)
                for (const auto& L : layouts)
                {
                    const std::string p = dir + "/s" + std::to_string(n) + ".exr";
                    exrcheck::writeExr(p, sz[0], sz[1], L, c, (n % 5 == 4) ? Imf::DECREASING_Y : Imf::INCREASING_Y,
                                       (exrcheck::Kind)(n % 3), 500u + n);
                    ++n;
                    const int nch = L.size() == 1 ? 1 : (L.size() == 4 ? 4 : 3);
                    ++cases;
                    std::vector<float> ours;
                    const Status s = gpuDecode(codec, p, nch, ours);
                    const exrcheck::RefRead ref = exrcheck::refRead(p, nch);
                    if (s != Status::Ok || !ref.ok || ours.size() != ref.pixels.size() ||
                        std::memcmp(ours.data(), ref.pixels.data(), ours.size() * 4) != 0)
                    {
                        std::printf("FAIL %s (%dx%d %s, %zu channels): %s\n", p.c_str(), sz[0], sz[1], exrcheck::compName(c),
                                    L.size(), statusName(s));
                        ++fails;
                    }
                }
        for (Imf::Compression c : comps)
        {
            const std::string p = dir + "/exhaustive-" + exrcheck::compName(c) + ".exr";
            exrcheck::writeExr(p, 256, 256, layouts[0], c, Imf::INCREASING_Y, exrcheck::kExhaustive, 7);
            files.push_back(p);
        }
        const std::string pds = dir + "/pds-6000x3376-zip.exr";
        exrcheck::writeExr(pds, 6000, 3376, layouts[0], Imf::ZIP_COMPRESSION, Imf::INCREASING_Y, exrcheck::kSmooth, 2026);
        files.push_back(pds);
        std::printf("synthetic: %d cases, %d failed\n", cases, fails);
    }

    const int hw = (int)std::max(1u, std::thread::hardware_concurrency());
    Imf::setGlobalThreadCount(hw);
    std::printf("%-44s %10s %7s | %10s %10s %12s %12s\n", "file", "size", "chunks", "GPU", "to device", "OpenEXR 1t", "OpenEXR pool");
    for (const std::string& path : files)
    {
        const int nch = channelsFor(path);
        std::vector<float> ours;
        DecodeStats st;
        ++cases;
        const Status s = gpuDecode(codec, path, nch, ours, &st);
        const exrcheck::RefRead ref = exrcheck::refRead(path, nch);
        std::string label = path.size() > 44 ? "..." + path.substr(path.size() - 41) : path;
        if (s != Status::Ok)
        {
            std::printf("%-44s %s%s\n", label.c_str(), statusName(s),
                        (s == Status::Unsupported || s == Status::NotExr || (s == Status::Corrupt && !ref.ok)) ? " (OpenEXR path)" : " FAIL");
            fails += (s == Status::Unsupported || s == Status::NotExr || (s == Status::Corrupt && !ref.ok)) ? 0 : 1;
            continue;
        }
        if (!ref.ok || ours.size() != ref.pixels.size() || std::memcmp(ours.data(), ref.pixels.data(), ours.size() * 4) != 0)
        {
            std::printf("%-44s FAIL: differs from OpenEXR\n", label.c_str());
            ++fails;
            continue;
        }
        std::vector<float> viaDevice;
        if (gpuDecodeToDevice(codec, path, nch, viaDevice) != Status::Ok || viaDevice.size() != ours.size() ||
            std::memcmp(viaDevice.data(), ours.data(), ours.size() * 4) != 0)
        {
            std::printf("%-44s FAIL: decodeToDevice differs from decode\n", label.c_str());
            ++fails;
            continue;
        }
        // every timed decode must succeed, or its time is not a decode's time
        int timingFails = 0;
        const double g = medianMs(reps, [&] { timingFails += gpuDecode(codec, path, nch, ours) != Status::Ok; });
        // what the depth-map node pays: read the file, decode, floats left on the device
        const double gd = medianMs(reps, [&] {
            const std::vector<uint8_t> f = exrcheck::readFile(path);
            DeviceImage di;
            timingFails += codec.decodeToDevice(f.data(), f.size(), nch, di) != Status::Ok;
        });
        const double c1 = medianMs(reps, [&] { timingFails += !exrcheck::refRead(path, nch, 0).ok; });
        const double cp = medianMs(reps, [&] { timingFails += !exrcheck::refRead(path, nch, hw).ok; });
        if (timingFails)
        {
            std::printf("%-44s FAIL: %d of the timed decodes failed\n", label.c_str(), timingFails);
            ++fails;
            continue;
        }
        char size[32];
        std::snprintf(size, sizeof(size), "%dx%d", ref.width, ref.height);
        std::printf("%-44s %10s %7u | %8.1fms %8.1fms %10.1fms %10.1fms\n", label.c_str(), size, st.chunks, g, gd, c1, cp);
    }

    if (!threadCounts.empty() && !fails)
    {
        // Per file, outside the timed loops: the channels to request and the pixel count. Images/s
        // is only comparable between files of one size; megapixels/s is comparable across sizes.
        struct FileInfo
        {
            std::string path;
            int nch = 3;
            uint64_t pixels = 0;
        };
        std::vector<FileInfo> infos;
        for (const std::string& path : files)
        {
            const std::vector<uint8_t> f = exrcheck::readFile(path);
            Header h;
            FileInfo fi;
            fi.path = path;
            if (readHeader(f.data(), f.size(), h) == Status::Ok)
            {
                fi.nch = h.channels.size() == 1 ? 1 : (h.channels.size() == 4 ? 4 : 3);
                fi.pixels = (uint64_t)h.width * (uint64_t)h.height;
            }
            infos.push_back(fi);
        }
        std::printf("\nthroughput, %zu files x %d reps per thread; only successful decodes count, and any failure fails the run\n"
                    "%8s %12s %10s %14s %10s\n",
                    files.size(), reps, "threads", "CheshireEXR", "MP/s", "OpenEXR (1t)", "MP/s");
        for (int t : threadCounts)
        {
            struct Result
            {
                double imagesPerSecond = 0, megapixelsPerSecond = 0;
                long failed = 0, total = 0;
            };
            auto run = [&](bool gpu) {
                std::atomic<long> ok{0}, failed{0};
                std::atomic<uint64_t> pixels{0};
                const auto t0 = Clock::now();
                std::vector<std::thread> pool;
                for (int k = 0; k < t; ++k)
                    pool.emplace_back([&, gpu] {
                        Codec local;  // one Codec per thread
                        std::vector<float> out;
                        for (int r = 0; r < reps; ++r)
                            for (const FileInfo& fi : infos)
                            {
                                const bool good = gpu ? gpuDecode(local, fi.path, fi.nch, out) == Status::Ok : exrcheck::refRead(fi.path, fi.nch, 0).ok;
                                if (good)
                                {
                                    ++ok;
                                    pixels += fi.pixels;
                                }
                                else
                                    ++failed;
                            }
                    });
                for (auto& th : pool)
                    th.join();
                const double s = std::chrono::duration<double>(Clock::now() - t0).count();
                Result res;
                res.imagesPerSecond = ok.load() / s;
                res.megapixelsPerSecond = pixels.load() / s / 1e6;
                res.failed = failed.load();
                res.total = ok.load() + failed.load();
                return res;
            };
            const Result g = run(true);
            const Result c = run(false);
            std::printf("%8d %12.1f %10.1f %14.1f %10.1f\n", t, g.imagesPerSecond, g.megapixelsPerSecond, c.imagesPerSecond, c.megapixelsPerSecond);
            if (g.failed || c.failed)
            {
                std::printf("         FAIL: %ld of %ld CheshireEXR and %ld of %ld OpenEXR decodes failed at %d threads\n", g.failed, g.total,
                            c.failed, c.total, t);
                ++fails;
            }
        }
    }
    std::printf("%d cases, %d failed\n%s\n", cases, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
