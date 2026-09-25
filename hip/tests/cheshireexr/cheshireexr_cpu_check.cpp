// CheshireEXR: the device pipeline, run on the host, against OpenEXR.
//
//   cheshireexr_cpu_check [--backend host|async|both] [--large] [--workdir DIR] [extra.exr ...]
//
// Test files are written with OpenEXR itself - every supported compression, both line orders,
// half, float and mixed channels, RGB, RGBA and one-channel, edge-case sizes, noise, gradients,
// constant runs and every half bit pattern - then decoded two ways: by the pipeline on a backend,
// and by OpenEXR's Imf::InputFile into FLOAT slices, as Cheshire's direct reader does. The floats
// must be identical bit for bit, and decodeToDevice (downloaded) must give the same status and
// floats as decode. Mutated files (bytes flipped inside chunks): a decode that
// reports Ok must equal OpenEXR's, and OpenEXR must have succeeded too. The "async" backend is
// the device backend's own code on the deferred runtime (hip/tests/cheshiregpu).
#include "asyncBackend.hpp"
#include "cpuBackend.hpp"
#include "deferredRuntime.hpp"
#include "exrCheckCommon.hpp"
#include "exrPipeline.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace cheshire::exr;
using exrcheck::ChannelSpec;

namespace {

struct Tally
{
    int run = 0, fail = 0, ok = 0, handedBack = 0, device = 0;
};

template <class Pipe>
Status ours(Pipe& pipe, const std::vector<uint8_t>& file, int nch, std::vector<float>& out, int& w, int& h, DecodeStats* st = nullptr)
{
    return pipe.decode(file.data(), file.size(), nch,
                       [&](int ww, int hh) {
                           w = ww;
                           h = hh;
                           out.assign((size_t)ww * hh * nch, 0.0f);
                           return out.data();
                       },
                       st);
}

// The same file through decodeToDevice, then Pipeline::download: the status must be the one decode
// returned and, on Ok, the floats and the layout the ones decode wrote.
template <class Pipe>
bool deviceAgrees(Pipe& pipe, const std::vector<uint8_t>& file, int nch, Status expect, const std::vector<float>& px, int w, int h)
{
    // fresh buffers (garbage-filled on the deferred runtime): the output buffer decode just filled
    // holds the expected floats already, and would hide a result read before the stream ran it
    pipe.trim();
    DeviceImage di;
    const Status s = pipe.decodeToDevice(file.data(), file.size(), nch, di, nullptr);
    if (s != expect)
        return false;
    if (s != Status::Ok)
        return di.data == nullptr;
    if (di.width != w || di.height != h || di.channels != nch || di.rowBytes != (size_t)w * nch * sizeof(float))
        return false;
    // both check backends keep "device" memory on the host: read it directly first, so a result
    // still queued on the deferred stream when decodeToDevice returns shows up as garbage
    if (std::memcmp(di.data, px.data(), px.size() * sizeof(float)) != 0)
        return false;
    std::vector<float> out(px.size(), -1.0f);
    if (pipe.download(di, out.data()) != Status::Ok)
        return false;
    return std::memcmp(out.data(), px.data(), px.size() * sizeof(float)) == 0;
}

// expect Ok: must match OpenEXR bit for bit. Otherwise the status that must come back.
template <class Pipe>
bool checkFile(Pipe& pipe, const std::string& path, int nch, const std::string& name, Tally& t, Status expect = Status::Ok)
{
    ++t.run;
    const std::vector<uint8_t> file = exrcheck::readFile(path);
    std::vector<float> px;
    int w = 0, h = 0;
    const Status s = ours(pipe, file, nch, px, w, h);
    if (!deviceAgrees(pipe, file, nch, s, px, w, h))
    {
        std::printf("FAIL %s nch%d: decodeToDevice differs from decode\n", name.c_str(), nch);
        ++t.fail;
        return false;
    }
    ++t.device;
    if (expect != Status::Ok)
    {
        if (s != expect)
        {
            std::printf("FAIL %s nch%d: expected %s, got %s\n", name.c_str(), nch, statusName(expect), statusName(s));
            ++t.fail;
            return false;
        }
        ++t.handedBack;
        return true;
    }
    if (s != Status::Ok)
    {
        std::printf("FAIL %s nch%d: %s\n", name.c_str(), nch, statusName(s));
        ++t.fail;
        return false;
    }
    // Header::chunkCount (what the depth-map loader's chunk threshold reads) against the chunk table
    {
        Header hd;
        Plan plan;
        if (readHeader(file.data(), file.size(), hd) != Status::Ok || makePlan(file.data(), file.size(), nch, plan) != Status::Ok ||
            hd.chunkCount() != (int)plan.chunks.size())
        {
            std::printf("FAIL %s nch%d: Header::chunkCount %d, chunk table %zu\n", name.c_str(), nch, hd.chunkCount(), plan.chunks.size());
            ++t.fail;
            return false;
        }
    }
    const exrcheck::RefRead ref = exrcheck::refRead(path, nch);
    if (!ref.ok || ref.width != w || ref.height != h || std::memcmp(ref.pixels.data(), px.data(), px.size() * 4) != 0)
    {
        size_t first = 0;
        while (ref.ok && first < px.size() && std::memcmp(&ref.pixels[first], &px[first], 4) == 0)
            ++first;
        uint32_t a = 0, b = 0;
        if (ref.ok && first < px.size())
        {
            std::memcpy(&a, &px[first], 4);
            std::memcpy(&b, &ref.pixels[first], 4);
        }
        std::printf("FAIL %s nch%d: differs from OpenEXR%s at float %zu (%08x vs %08x)\n", name.c_str(), nch,
                    ref.ok ? "" : (" (OpenEXR failed: " + ref.error + ")").c_str(), first, a, b);
        ++t.fail;
        return false;
    }
    ++t.ok;
    return true;
}

template <class Backend>
int runChecks(const char* label, const std::string& dir, bool large, const std::vector<std::string>& extra)
{
    std::printf("== backend: %s\n", label);
    Backend backend;
    Pipeline<Backend> pipe(backend);
    Tally t, fuzz;

    const int sizes[][2] = {{1, 1}, {2, 1}, {1, 17}, {7, 3}, {16, 16}, {17, 33}, {100, 75}, {257, 129}, {640, 97}};
    const Imf::Compression comps[] = {Imf::NO_COMPRESSION, Imf::RLE_COMPRESSION, Imf::ZIPS_COMPRESSION, Imf::ZIP_COMPRESSION};
    struct Layout
    {
        const char* name;
        std::vector<ChannelSpec> chans;
        std::vector<int> requests;
    };
    const Layout layouts[] = {
      {"rgba-half", {{"R", true}, {"G", true}, {"B", true}, {"A", true}}, {3, 4}},
      {"rgb-half", {{"R", true}, {"G", true}, {"B", true}}, {3}},
      {"rgba-float", {{"R", false}, {"G", false}, {"B", false}, {"A", false}}, {4}},
      {"rgb-mixed", {{"R", true}, {"G", false}, {"B", true}}, {3}},
      {"y-float", {{"Y", false}}, {1}},
      {"y-half", {{"Y", true}}, {1}},
    };
    const exrcheck::Kind kinds[] = {exrcheck::kSmooth, exrcheck::kNoise, exrcheck::kConstant};
    const char* kindName[] = {"noise", "smooth", "constant", "exhaustive"};
    int n = 0;
    for (const auto& sz : sizes)
        for (Imf::Compression comp : comps)
            for (const Layout& L : layouts)
            {
                const exrcheck::Kind kind = kinds[n % 3];
                const Imf::LineOrder order = (n % 5 == 4) ? Imf::DECREASING_Y : Imf::INCREASING_Y;
                const std::string path = dir + "/c" + std::to_string(n) + ".exr";
                if (!exrcheck::writeExr(path, sz[0], sz[1], L.chans, comp, order, kind, 1000u + n))
                    return 1;
                char name[160];
                std::snprintf(name, sizeof(name), "%dx%d %s %s %s%s", sz[0], sz[1], exrcheck::compName(comp), L.name,
                              kindName[kind], order == Imf::DECREASING_Y ? " decreasing" : "");
                for (int nch : L.requests)
                    checkFile(pipe, path, nch, name, t);
                ++n;
            }

    pipe.trim();  // the buffers are allocated again on the next decode

    // every half bit pattern through every compression, in all four channels
    for (Imf::Compression comp : comps)
    {
        const std::string path = dir + "/exhaustive.exr";
        exrcheck::writeExr(path, 256, 256, {{"R", true}, {"G", true}, {"B", true}, {"A", true}}, comp, Imf::INCREASING_Y,
                           exrcheck::kExhaustive, 7);
        checkFile(pipe, path, 4, std::string("exhaustive half ") + exrcheck::compName(comp), t);
    }

    // what the decoder hands back
    {
        const std::string path = dir + "/piz.exr";
        exrcheck::writeExr(path, 64, 48, {{"R", true}, {"G", true}, {"B", true}}, Imf::PIZ_COMPRESSION, Imf::INCREASING_Y,
                           exrcheck::kSmooth, 3);
        checkFile(pipe, path, 3, "piz", t, Status::Unsupported);
        const std::string p2 = dir + "/rgb-for-a.exr";
        exrcheck::writeExr(p2, 32, 32, {{"R", true}, {"G", true}, {"B", true}}, Imf::ZIP_COMPRESSION, Imf::INCREASING_Y,
                           exrcheck::kSmooth, 4);
        checkFile(pipe, p2, 4, "alpha requested from RGB", t, Status::Unsupported);
        checkFile(pipe, p2, 1, "one channel requested from RGB", t, Status::Unsupported);
        const std::vector<uint8_t> junk = {'n', 'o', 't', ' ', 'a', 'n', ' ', 'e', 'x', 'r'};
        const std::string p3 = dir + "/junk.exr";
        exrcheck::writeBytes(p3, junk);
        checkFile(pipe, p3, 3, "not an exr", t, Status::NotExr);
    }

    // mutated chunks: Ok must mean OpenEXR's result
    {
        int same = 0, back = 0;
        const Imf::Compression mcomps[] = {Imf::ZIP_COMPRESSION, Imf::ZIPS_COMPRESSION, Imf::RLE_COMPRESSION};
        for (Imf::Compression comp : mcomps)
        {
            const std::string path = dir + "/mut-src.exr";
            exrcheck::writeExr(path, 120, 70, {{"R", true}, {"G", true}, {"B", true}, {"A", true}}, comp, Imf::INCREASING_Y,
                               exrcheck::kSmooth, 11);
            const std::vector<uint8_t> file = exrcheck::readFile(path);
            Plan plan;
            makePlan(file.data(), file.size(), 4, plan);
            const size_t b0 = (size_t)plan.chunks.front().fileOffset;
            uint32_t s = 99u + (uint32_t)comp;
            for (int m = 0; m < 200; ++m)
            {
                std::vector<uint8_t> f = file;
                const int flips = 1 + (int)(exrcheck::lcg(s) % 3);
                for (int j = 0; j < flips; ++j)
                {
                    const size_t at = b0 + exrcheck::lcg(s) % (f.size() - b0);
                    f[at] ^= (uint8_t)(1u << (exrcheck::lcg(s) % 8));
                }
                ++fuzz.run;
                std::vector<float> px;
                int w, h;
                const Status st = ours(pipe, f, 4, px, w, h);
                if (!deviceAgrees(pipe, f, 4, st, px, w, h))
                {
                    std::printf("FAIL mutated %s #%d: decodeToDevice differs from decode\n", exrcheck::compName(comp), m);
                    ++fuzz.fail;
                    continue;
                }
                if (st != Status::Ok)
                {
                    ++back;
                    continue;
                }
                const std::string mp = dir + "/mut.exr";
                exrcheck::writeBytes(mp, f);
                const exrcheck::RefRead ref = exrcheck::refRead(mp, 4);
                if (!ref.ok || std::memcmp(ref.pixels.data(), px.data(), px.size() * 4) != 0)
                {
                    std::printf("FAIL mutated %s #%d: decoded Ok but %s\n", exrcheck::compName(comp), m,
                                ref.ok ? "differs from OpenEXR" : ("OpenEXR failed: " + ref.error).c_str());
                    ++fuzz.fail;
                }
                else
                    ++same;
            }
        }
        std::printf("mutated chunks: %d decoded identically to OpenEXR, %d handed back\n", same, back);
    }

    // PrepareDenseScene's size: 6000x3376 half RGBA, ZIP
    if (large)
    {
        const std::string path = dir + "/pds.exr";
        exrcheck::writeExr(path, 6000, 3376, {{"R", true}, {"G", true}, {"B", true}, {"A", true}}, Imf::ZIP_COMPRESSION,
                           Imf::INCREASING_Y, exrcheck::kSmooth, 2026);
        const auto t0 = std::chrono::steady_clock::now();
        checkFile(pipe, path, 4, "6000x3376 half RGBA zip", t);
        std::printf("large: %.1f s on this backend (host loop, not a timing of the device)\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    for (const std::string& path : extra)
    {
        Header hd;
        const std::vector<uint8_t> f = exrcheck::readFile(path);
        const Status hs = readHeader(f.data(), f.size(), hd);
        const int nch = hd.channels.size() == 1 ? 1 : (hd.channels.size() == 4 ? 4 : 3);
        Plan plan;
        const Status ps = hs == Status::Ok ? makePlan(f.data(), f.size(), nch, plan) : hs;
        checkFile(pipe, path, nch, path, t, ps);
    }

    std::printf("files: %d decodes, %d identical to OpenEXR, %d handed back as expected, %d failed; decodeToDevice agreed on %d\n",
                t.run, t.ok, t.handedBack, t.fail, t.device);
    std::printf("mutated: %d cases, %d failed\n", fuzz.run, fuzz.fail);
    const bool ok = t.fail == 0 && fuzz.fail == 0;
    std::printf("%s: %s\n", label, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv)
{
    std::string which = "both";
    std::string dir = (std::filesystem::temp_directory_path() / "cheshireexr_check").string();
    bool large = false;
    std::vector<std::string> extra;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc)
            which = argv[++i];
        else if (a == "--workdir" && i + 1 < argc)
            dir = argv[++i];
        else if (a == "--large")
            large = true;
        else
            extra.push_back(a);
    }
    std::filesystem::create_directories(dir);
    int rc = 0;
    if (which == "both" || which == "host")
        rc |= runChecks<cheshire::gpu::CpuBackend>("host", dir, large, extra);
    if (which == "both" || which == "async")
        rc |= runChecks<cheshire::gpu::AsyncBackend<check::DeferredRuntime>>("async (deferred)", dir, large, extra);
    std::printf("%s\n", rc ? "FAIL" : "PASS");
    return rc;
}
