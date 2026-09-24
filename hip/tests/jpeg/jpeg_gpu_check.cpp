// Cheshire GPU JPEG: the codec on the device, against libjpeg-turbo, with timings.
//
//   jpeg_gpu_check [--reps N] [--no-synthetic] [photo.jpg ...]
//
// Synthetic cases: every sampling layout the encoder writes, at edge-case sizes, qualities and
// restart intervals; the encoded file must equal libjpeg's byte for byte and the decode of it must
// equal libjpeg's pixel for pixel. Photos: the decode must equal libjpeg's, then decode and a
// re-encode (q90, 4:2:0) are timed against libjpeg-turbo, median of N repetitions, end to end -
// parsing, uploads, kernels and the download of the result all count.
#include "checkCommon.hpp"
#include "jpegCodec.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

using namespace cheshire::jpeg;
using check::EncodeCase;
using Clock = std::chrono::steady_clock;

namespace {

double ms(Clock::duration d)
{
    return std::chrono::duration<double, std::milli>(d).count();
}

template <class F>
double medianMs(int reps, F&& f)
{
    std::vector<double> t;
    for (int i = 0; i < reps; ++i)
    {
        const auto t0 = Clock::now();
        f();
        t.push_back(ms(Clock::now() - t0));
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

}  // namespace

int main(int argc, char** argv)
{
    int reps = 5;
    bool synthetic = true;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--reps" && i + 1 < argc)
            reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--no-synthetic")
            synthetic = false;
        else
            files.push_back(a);
    }
    if (!Codec::deviceAvailable())
    {
        std::printf("no GPU device (or CHESHIRE_GPU_JPEG=0)\n");
        return 2;
    }
    Codec codec;
    int fails = 0, cases = 0;

    if (synthetic)
    {
        const int sizes[][2] = {{1, 1}, {7, 5}, {16, 16}, {17, 9}, {33, 31}, {257, 129}, {640, 480}, {1023, 67}};
        const Subsampling subs[] = {Subsampling::S444, Subsampling::S422, Subsampling::S420, Subsampling::S440,
                                    Subsampling::Gray};
        const int qualities[] = {1, 30, 75, 90, 100};
        const int restarts[] = {0, 1, 3, 0, 7};
        int n = 0;
        for (const auto& sz : sizes)
            for (Subsampling sub : subs)
                for (int kind = 0; kind < 4; ++kind, ++n)
                {
                    const int w = sz[0], h = sz[1];
                    const int ch = (sub == Subsampling::Gray && (n & 1)) ? 1 : 3;
                    EncodeCase k;
                    k.sub = sub;
                    k.quality = qualities[n % 5];
                    k.restart = restarts[(n / 5) % 5];
                    const std::vector<uint8_t> px = check::synthImage(w, h, ch, 1234u + n, kind);
                    std::vector<uint8_t> ours, ref;
                    EncodeOptions o;
                    o.quality = k.quality;
                    o.subsampling = k.sub;
                    o.restartInterval = k.restart;
                    ++cases;
                    const Status s = codec.encode(px.data(), w, h, ch, (size_t)w * ch, o, ours);
                    check::refEncode(px.data(), w, h, ch, k, ref);
                    if (s != Status::Ok || ours != ref)
                    {
                        std::printf("FAIL encode %dx%dx%d %s q%d rst%d kind%d: %s, first difference at %ld\n", w, h, ch,
                                    check::subName(sub), k.quality, k.restart, kind, statusName(s),
                                    check::firstDiff(ours, ref));
                        ++fails;
                        continue;
                    }
                    Image img;
                    const Status d = codec.decode(ours.data(), ours.size(), img);
                    const check::RefDecode rd = check::refDecode(ours, false);
                    if (d != Status::Ok || img.pixels != rd.pixels)
                    {
                        std::printf("FAIL decode %dx%dx%d %s q%d rst%d kind%d: %s, first difference at %ld\n", w, h, ch,
                                    check::subName(sub), k.quality, k.restart, kind, statusName(d),
                                    check::firstDiff(img.pixels, rd.pixels));
                        ++fails;
                    }
                }
        std::printf("synthetic: %d encode+decode cases, %d failed\n", cases, fails);
    }

    if (!files.empty())
        std::printf("%-40s %11s %7s %6s | %9s %9s | %9s %9s\n", "file", "size", "subseq", "rounds", "dec GPU", "dec CPU",
                    "enc GPU", "enc CPU");
    for (const std::string& path : files)
    {
        const std::vector<uint8_t> file = check::readFile(path);
        Image img;
        DecodeStats st;
        ++cases;
        const Status s = codec.decode(file.data(), file.size(), img, &st);
        const check::RefDecode ref = check::refDecode(file, false);
        std::string label = path.size() > 40 ? "..." + path.substr(path.size() - 37) : path;
        if (s != Status::Ok)
        {
            const bool fine = s == Status::Unsupported || (s == Status::Corrupt && (!ref.ok || ref.warnings));
            std::printf("%-40s %s%s\n", label.c_str(), statusName(s), fine ? " (libjpeg path)" : " FAIL");
            fails += fine ? 0 : 1;
            continue;
        }
        if (img.pixels != ref.pixels)
        {
            std::printf("%-40s FAIL: differs from libjpeg at byte %ld\n", label.c_str(), check::firstDiff(img.pixels, ref.pixels));
            ++fails;
            continue;
        }
        const double gpuDec = medianMs(reps, [&] { codec.decode(file.data(), file.size(), img); });
        const double cpuDec = medianMs(reps, [&] { check::refDecode(file, false); });
        double gpuEnc = 0, cpuEnc = 0;
        if (img.channels == 3)
        {
            EncodeOptions o;
            o.quality = 90;
            EncodeCase k;
            k.quality = 90;
            std::vector<uint8_t> a, b;
            codec.encode(img.pixels.data(), img.width, img.height, 3, (size_t)img.width * 3, o, a);
            check::refEncode(img.pixels.data(), img.width, img.height, 3, k, b);
            if (a != b)
            {
                std::printf("%-40s FAIL: re-encode differs from libjpeg at byte %ld\n", label.c_str(), check::firstDiff(a, b));
                ++fails;
                continue;
            }
            gpuEnc = medianMs(reps, [&] { codec.encode(img.pixels.data(), img.width, img.height, 3, (size_t)img.width * 3, o, a); });
            cpuEnc = medianMs(reps, [&] { check::refEncode(img.pixels.data(), img.width, img.height, 3, k, b); });
        }
        char size[32];
        std::snprintf(size, sizeof(size), "%dx%d", img.width, img.height);
        std::printf("%-40s %11s %7u %6u | %7.1fms %7.1fms | %7.1fms %7.1fms\n", label.c_str(), size, st.subsequences,
                    st.syncRounds, gpuDec, cpuDec, gpuEnc, cpuEnc);
    }
    std::printf("%d cases, %d failed\n%s\n", cases, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
