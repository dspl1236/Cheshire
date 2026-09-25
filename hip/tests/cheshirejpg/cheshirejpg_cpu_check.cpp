// CheshireJPG: the device pipeline, run on the CPU backend, against libjpeg-turbo.
//
// Every stage functor the kernels run is run here in the same order, so this checks the codec's
// arithmetic and its parallel decomposition (the Huffman synchronisation, the scans, the
// byte-stuffing scatter) without a GPU. What it cannot check is the device compiler and runtime;
// cheshirejpg_gpu_check does that on hardware.
//
//   cheshirejpg_cpu_check [--testimages DIR] [--large] [extra.jpg ...]
//
// Decode: pixels must equal jpeg_read_scanlines' and every real block's coefficients must equal
// jpeg_read_coefficients'. Encode: the file must equal libjpeg's byte for byte. Mutated streams:
// a decode that reports Ok must equal libjpeg's output. Run it again with JSIMD_FORCENONE=1 to
// check against libjpeg-turbo's C paths as well as its SIMD ones.
#include "checkCommon.hpp"
#include "asyncBackend.hpp"
#include "cpuBackend.hpp"
#include "deferredRuntime.hpp"
#include "jpegPipeline.hpp"

#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

using namespace cheshire::jpeg;
using check::EncodeCase;

namespace {

struct Tally
{
    int run = 0;
    int fail = 0;
    uint64_t subseqs = 0;
    uint64_t rounds = 0;
    uint32_t maxRounds = 0;
    int decodes = 0;
    bool verbose = false;
};

uint32_t decBlockToUnit(const DecodeGeom& g, int c, int bx, int by)
{
    if (!g.interleaved)
        return (uint32_t)by * (uint32_t)g.blocksW[0] + (uint32_t)bx;
    int first = 0;
    while (g.blkComp[first] != c)
        ++first;
    const uint32_t mcu = (uint32_t)(by / g.v[c]) * g.mcusX + (uint32_t)(bx / g.h[c]);
    return mcu * (uint32_t)g.bpm + (uint32_t)(first + (by % g.v[c]) * g.h[c] + bx % g.h[c]);
}

// expect: Ok (must match), or the status that must come back.
template <class Pipe>
bool checkDecode(Pipe& pipe, const std::vector<uint8_t>& file, const std::string& name, Tally& t, Status expect = Status::Ok)
{
    ++t.run;
    Image img;
    DecodeStats st;
    const Status s = pipe.decode(file.data(), file.size(), img, &st);
    if (expect != Status::Ok)
    {
        if (s != expect)
        {
            std::printf("FAIL decode %s: expected %s, got %s\n", name.c_str(), statusName(expect), statusName(s));
            ++t.fail;
            return false;
        }
        return true;
    }
    if (s != Status::Ok)
    {
        if (s == Status::Corrupt)
        {
            // the codec hands damaged streams back; fine if libjpeg also finds them damaged
            const check::RefDecode ref = check::refDecode(file, false);
            if (!ref.ok || ref.warnings > 0)
            {
                std::printf("  %-48s Corrupt, handed back (libjpeg: %d warnings%s)\n", name.c_str(), ref.warnings,
                            ref.ok ? "" : ", error");
                return true;
            }
        }
        std::printf("FAIL decode %s: %s\n", name.c_str(), statusName(s));
        ++t.fail;
        return false;
    }
    ++t.decodes;
    if (t.verbose)
        std::printf("  %-48s %5dx%-5d %u subsequences, %u restart segments, %u sync rounds\n", name.c_str(), img.width,
                    img.height, st.subsequences, st.restartSegments, st.syncRounds);
    t.subseqs += st.subsequences;
    t.rounds += st.syncRounds;
    t.maxRounds = st.syncRounds > t.maxRounds ? st.syncRounds : t.maxRounds;

    const check::RefDecode ref = check::refDecode(file, true);
    if (!ref.ok)
    {
        std::printf("FAIL decode %s: libjpeg failed (%s) but we decoded\n", name.c_str(), ref.error.c_str());
        ++t.fail;
        return false;
    }
    if (img.width != ref.width || img.height != ref.height || img.channels != ref.channels)
    {
        std::printf("FAIL decode %s: %dx%dx%d, libjpeg %dx%dx%d\n", name.c_str(), img.width, img.height, img.channels,
                    ref.width, ref.height, ref.channels);
        ++t.fail;
        return false;
    }
    const long d = check::firstDiff(img.pixels, ref.pixels);
    if (d >= 0)
    {
        const long px = d / img.channels;
        std::printf("FAIL decode %s: pixel (%ld,%ld) channel %ld: %d vs libjpeg %d\n", name.c_str(), px % img.width,
                    px / img.width, d % img.channels, img.pixels[d], ref.pixels[d]);
        ++t.fail;
        return false;
    }

    ParsedJpeg p;
    DecodeGeom g;
    parseJpeg(file.data(), file.size(), p);
    makeDecodeGeom(p, g);
    std::vector<int16_t> coefs;
    pipe.downloadCoefs(coefs, (size_t)g.totalUnits * 64);
    for (int c = 0; c < ref.ncomp; ++c)
        for (int by = 0; by < ref.heightInBlocks[c]; ++by)
            for (int bx = 0; bx < ref.widthInBlocks[c]; ++bx)
            {
                const int16_t* a = &coefs[(size_t)decBlockToUnit(g, c, bx, by) * 64];
                const int16_t* b = &ref.coefs[c][((size_t)by * ref.widthInBlocks[c] + bx) * 64];
                if (std::memcmp(a, b, 64 * sizeof(int16_t)))
                {
                    std::printf("FAIL decode %s: coefficients of component %d block (%d,%d) differ\n", name.c_str(), c, bx, by);
                    ++t.fail;
                    return false;
                }
            }
    return true;
}

template <class Pipe>
bool checkEncode(Pipe& pipe,
                 const std::vector<uint8_t>& px,
                 int w,
                 int h,
                 int ch,
                 const EncodeCase& k,
                 const std::string& name,
                 Tally& enc,
                 Tally& dec)
{
    ++enc.run;
    std::vector<uint8_t> ours, ref;
    EncodeOptions o;
    o.quality = k.quality;
    o.subsampling = k.sub;
    o.restartInterval = k.restart;
    const Status s = pipe.encode(px.data(), w, h, ch, (size_t)w * ch, o, ours);
    std::string err;
    if (!check::refEncode(px.data(), w, h, ch, k, ref, &err))
    {
        std::printf("FAIL encode %s: libjpeg failed: %s\n", name.c_str(), err.c_str());
        ++enc.fail;
        return false;
    }
    if (s != Status::Ok)
    {
        std::printf("FAIL encode %s: %s\n", name.c_str(), statusName(s));
        ++enc.fail;
        return false;
    }
    const long d = check::firstDiff(ours, ref);
    if (d >= 0)
    {
        std::printf("FAIL encode %s: %zu bytes vs libjpeg %zu, first difference at byte %ld\n", name.c_str(), ours.size(),
                    ref.size(), d);
        ++enc.fail;
        return false;
    }
    // and our decoder on our file
    return checkDecode(pipe, ours, name + " (roundtrip)", dec);
}

const Subsampling kSubs[] = {Subsampling::S444, Subsampling::S422, Subsampling::S420, Subsampling::S440, Subsampling::Gray};

template <class Backend>
int runChecks(const char* label, const std::string& testimages, bool large, const std::vector<std::string>& extra)
{
    std::printf("== backend: %s\n", label);
    Backend backend;
    Pipeline<Backend> pipe(backend);
    Tally dec, enc, fuzz;

    const int sizes[][2] = {{1, 1},   {2, 2},   {3, 7},    {7, 5},   {8, 8},    {9, 9},     {16, 16},
                            {17, 9},  {33, 31}, {100, 75}, {64, 1},  {1, 64},   {257, 129}, {640, 480},
                            {1023, 67}};

    // ---- encoder: byte-identical files ----------------------------------------------------------
    {
        const int qualities[] = {1, 30, 75, 90, 100};
        const int restarts[] = {0, 1, 3, 0, 7};
        int n = 0;
        for (const auto& sz : sizes)
            for (Subsampling sub : kSubs)
                for (int kind = 0; kind < 4; ++kind)
                {
                    const int w = sz[0], h = sz[1];
                    const int ch = (sub == Subsampling::Gray && (n & 1)) ? 1 : 3;
                    EncodeCase k;
                    k.sub = sub;
                    k.quality = qualities[n % 5];
                    k.restart = restarts[(n / 5) % 5];
                    const std::vector<uint8_t> px = check::synthImage(w, h, ch, 1234u + n, kind);
                    char name[160];
                    std::snprintf(name, sizeof(name), "enc %dx%dx%d %s q%d rst%d kind%d", w, h, ch, check::subName(sub),
                                  k.quality, k.restart, kind);
                    checkEncode(pipe, px, w, h, ch, k, name, enc, dec);
                    ++n;
                }
    }

    // ---- decoder: files libjpeg writes with features our encoder does not use ------------------
    {
        struct Layout
        {
            const char* name;
            int hv[3][2];
        };
        const Layout layouts[] = {
          {"411", {{4, 1}, {1, 1}, {1, 1}}},
          {"mixed", {{2, 2}, {1, 2}, {2, 1}}},
          {"h3", {{3, 1}, {1, 1}, {1, 1}}},
          {"chroma-full", {{1, 1}, {2, 2}, {1, 1}}},
          {"222", {{2, 2}, {2, 2}, {2, 2}}},
        };
        int n = 0;
        for (const auto& sz : sizes)
        {
            const int w = sz[0], h = sz[1];
            for (int opt = 0; opt < 2; ++opt)
                for (Subsampling sub : kSubs)
                {
                    EncodeCase k;
                    k.sub = sub;
                    k.optimize = opt != 0;
                    k.quality = (n * 37) % 100 + 1;
                    k.restart = (n % 3) * 2;
                    const std::vector<uint8_t> px = check::synthImage(w, h, 3, 99u + n, n % 4);
                    std::vector<uint8_t> file;
                    check::refEncode(px.data(), w, h, 3, k, file);
                    char name[160];
                    std::snprintf(name, sizeof(name), "dec %dx%d %s q%d rst%d opt%d", w, h, check::subName(sub), k.quality,
                                  k.restart, opt);
                    checkDecode(pipe, file, name, dec);
                    ++n;
                }
            for (const Layout& L : layouts)
            {
                EncodeCase k;
                std::memcpy(k.customSampling, L.hv, sizeof(L.hv));
                k.quality = 60 + n % 40;
                k.restart = n % 2;
                k.optimize = n % 3 == 0;
                const std::vector<uint8_t> px = check::synthImage(w, h, 3, 7u + n, n % 4);
                std::vector<uint8_t> file;
                std::string err;
                if (!check::refEncode(px.data(), w, h, 3, k, file, &err))
                    continue;  // libjpeg rejects the layout at this size
                char name[160];
                std::snprintf(name, sizeof(name), "dec %dx%d layout %s q%d rst%d", w, h, L.name, k.quality, k.restart);
                checkDecode(pipe, file, name, dec);
                ++n;
            }
            {
                EncodeCase k;
                k.rgbColorspace = true;
                k.quality = 85;
                const std::vector<uint8_t> px = check::synthImage(w, h, 3, 5u + n, 1);
                std::vector<uint8_t> file;
                check::refEncode(px.data(), w, h, 3, k, file);
                checkDecode(pipe, file, "dec " + std::to_string(w) + "x" + std::to_string(h) + " RGB colorspace", dec);
            }
        }

        // formats we hand back to libjpeg
        {
            const std::vector<uint8_t> px = check::synthImage(64, 48, 3, 3, 1);
            jpeg_compress_struct c;
            check::ErrorMgr e;
            c.err = jpeg_std_error(&e.pub);
            jpeg_create_compress(&c);
            unsigned char* mem = nullptr;
            unsigned long memSize = 0;
            jpeg_mem_dest(&c, &mem, &memSize);
            c.image_width = 64;
            c.image_height = 48;
            c.input_components = 3;
            c.in_color_space = JCS_RGB;
            jpeg_set_defaults(&c);
            jpeg_simple_progression(&c);
            jpeg_start_compress(&c, TRUE);
            while (c.next_scanline < c.image_height)
            {
                JSAMPROW row = const_cast<JSAMPROW>(px.data() + (size_t)c.next_scanline * 64 * 3);
                jpeg_write_scanlines(&c, &row, 1);
            }
            jpeg_finish_compress(&c);
            const std::vector<uint8_t> prog(mem, mem + memSize);
            jpeg_destroy_compress(&c);
            std::free(mem);
            checkDecode(pipe, prog, "progressive", dec, Status::Unsupported);
        }
        {
            const std::vector<uint8_t> junk = {0x89, 'P', 'N', 'G', 0, 0, 0, 0};
            checkDecode(pipe, junk, "not a jpeg", dec, Status::NotJpeg);
        }
    }

    // ---- libjpeg-turbo's own test images -------------------------------------------------------
    if (!testimages.empty())
    {
        struct T
        {
            const char* file;
            Status expect;
        };
        const T files[] = {{"testorig.jpg", Status::Ok},
                           {"testimgint.jpg", Status::Ok},
                           {"testimgari.jpg", Status::Unsupported},
                           {"testorig12.jpg", Status::Unsupported}};
        for (const T& f : files)
        {
            const std::vector<uint8_t> d = check::readFile(testimages + "/" + f.file);
            if (d.empty())
            {
                std::printf("skip %s (not found)\n", f.file);
                continue;
            }
            ParsedJpeg p;
            const Status ps = parseJpeg(d.data(), d.size(), p);
            // testimgint.jpg is progressive in some releases; accept whatever the parser says it is
            const Status expect = (f.expect == Status::Ok && ps == Status::Unsupported) ? Status::Unsupported : f.expect;
            checkDecode(pipe, d, f.file, dec, expect);
        }
    }
    Tally files;
    files.verbose = true;
    for (const std::string& path : extra)
    {
        const std::vector<uint8_t> d = check::readFile(path);
        ParsedJpeg p;
        const Status ps = parseJpeg(d.data(), d.size(), p);
        if (ps != Status::Ok)
            std::printf("  %-48s %s\n", path.c_str(), statusName(ps));
        checkDecode(pipe, d, path, files, ps == Status::Ok ? Status::Ok : ps);
    }
    if (files.run)
        std::printf("files: %d, %d failed; %d decoded on the parallel path, mean %.2f / max %u sync rounds\n", files.run,
                    files.fail, files.decodes, files.decodes ? (double)files.rounds / files.decodes : 0.0, files.maxRounds);
    dec.fail += files.fail;

    // ---- mutated streams: an Ok decode must be libjpeg's decode --------------------------------
    {
        int okSame = 0, rejected = 0;
        const int restarts[] = {0, 4};
        for (int ri = 0; ri < 2; ++ri)
            for (Subsampling sub : {Subsampling::S420, Subsampling::S444, Subsampling::Gray})
            {
                const std::vector<uint8_t> px = check::synthImage(96, 80, 3, 42u + ri, 1);
                EncodeCase k;
                k.sub = sub;
                k.restart = restarts[ri];
                std::vector<uint8_t> file;
                check::refEncode(px.data(), 96, 80, 3, k, file);
                ParsedJpeg p;
                parseJpeg(file.data(), file.size(), p);
                const size_t b0 = p.entropyBegin, b1 = file.size() - 2;
                uint32_t s = 777u + ri * 31u + (uint32_t)sub;
                for (int m = 0; m < 150; ++m)
                {
                    std::vector<uint8_t> f = file;
                    const int flips = 1 + (int)(check::lcg(s) % 3);
                    for (int j = 0; j < flips; ++j)
                    {
                        const size_t at = b0 + check::lcg(s) % (b1 - b0);
                        f[at] ^= (uint8_t)(1u << (check::lcg(s) % 8));
                    }
                    ++fuzz.run;
                    Image img;
                    const Status st = pipe.decode(f.data(), f.size(), img, nullptr);
                    if (st != Status::Ok)
                    {
                        ++rejected;
                        continue;
                    }
                    const check::RefDecode ref = check::refDecode(f, false);
                    if (!ref.ok || check::firstDiff(img.pixels, ref.pixels) >= 0)
                    {
                        std::printf("FAIL mutated stream %s rst%d #%d: decoded Ok but differs from libjpeg\n",
                                    check::subName(sub), k.restart, m);
                        ++fuzz.fail;
                    }
                    else
                        ++okSame;
                }
            }
        std::printf("mutated streams: %d decoded identically to libjpeg, %d handed back (Corrupt)\n", okSame, rejected);
    }

    // ---- one camera-sized image, for the synchronisation statistics ----------------------------
    if (large)
    {
        const int w = 4032, h = 3024;
        const std::vector<uint8_t> px = check::synthImage(w, h, 3, 2024u, 1);
        EncodeCase k;
        k.quality = 92;
        Tally lenc, ldec;
        const auto t0 = std::chrono::steady_clock::now();
        checkEncode(pipe, px, w, h, 3, k, "large 4032x3024 420 q92", lenc, ldec);
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("large: encode+roundtrip %s, %.1f s on the CPU backend; %llu subsequences, %u sync rounds\n",
                    (lenc.fail || ldec.fail) ? "FAILED" : "identical",
                    std::chrono::duration<double>(t1 - t0).count(), (unsigned long long)ldec.subseqs, ldec.maxRounds);
        enc.run += lenc.run;
        enc.fail += lenc.fail;
        dec.run += ldec.run;
        dec.fail += ldec.fail;
    }

    std::printf("encode: %d cases, %d failed\n", enc.run, enc.fail);
    std::printf("decode: %d cases, %d failed; %d parallel decodes, %llu subsequences, mean %.2f / max %u sync rounds\n",
                dec.run, dec.fail, dec.decodes, (unsigned long long)dec.subseqs,
                dec.decodes ? (double)dec.rounds / dec.decodes : 0.0, dec.maxRounds);
    std::printf("mutated: %d cases, %d failed\n", fuzz.run, fuzz.fail);
    const bool ok = enc.fail == 0 && dec.fail == 0 && fuzz.fail == 0;
    std::printf("%s: %s\n", label, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv)
{
    std::string testimages;
    bool large = false;
    std::string which = "both";
    std::vector<std::string> extra;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--testimages" && i + 1 < argc)
            testimages = argv[++i];
        else if (a == "--large")
            large = true;
        else if (a == "--backend" && i + 1 < argc)
            which = argv[++i];
        else
            extra.push_back(a);
    }
    std::printf("libjpeg-turbo JPEG_LIB_VERSION %d, JSIMD_FORCENONE=%s\n", JPEG_LIB_VERSION,
                std::getenv("JSIMD_FORCENONE") ? std::getenv("JSIMD_FORCENONE") : "(unset)");

    // host: every stage as a plain loop, in order. async: the GPU backend's own code
    // (cheshiregpu/asyncBackend.hpp) over a runtime that runs queued work only when the stream is waited on.
    int rc = 0;
    if (which == "both" || which == "host")
        rc |= runChecks<cheshire::gpu::CpuBackend>("host", testimages, large, extra);
    if (which == "both" || which == "async")
        rc |= runChecks<cheshire::gpu::AsyncBackend<check::DeferredRuntime>>("async (deferred)", testimages, large, extra);
    std::printf("%s\n", rc ? "FAIL" : "PASS");
    return rc;
}
