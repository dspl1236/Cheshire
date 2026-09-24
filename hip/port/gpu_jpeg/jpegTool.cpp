// cheshire_jpeg: the GPU JPEG codec from the command line.
//
//   cheshire_jpeg decode in.jpg out.ppm|out.pgm
//   cheshire_jpeg encode in.ppm|in.pgm out.jpg [-q quality] [-s 444|422|420|440|gray] [-r restart]
//   cheshire_jpeg bench  in.jpg [repetitions]
//
// Exit status 0 on success, 1 on a codec error, 2 on usage or I/O errors, 3 when the file is one
// the codec hands back (progressive, arithmetic, CMYK, damaged): libjpeg decodes those.
#include "jpegCodec.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace cheshire::jpeg;

namespace {

bool readFile(const char* path, std::vector<uint8_t>& d)
{
    FILE* f = std::fopen(path, "rb");
    if (!f)
        return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    d.resize(n > 0 ? (size_t)n : 0);
    const bool ok = n > 0 && std::fread(d.data(), 1, d.size(), f) == d.size();
    std::fclose(f);
    return ok;
}

bool writeFile(const char* path, const uint8_t* p, size_t n, const char* header = nullptr)
{
    FILE* f = std::fopen(path, "wb");
    if (!f)
        return false;
    bool ok = true;
    if (header)
        ok = std::fputs(header, f) >= 0;
    ok = ok && std::fwrite(p, 1, n, f) == n;
    return std::fclose(f) == 0 && ok;
}

// Binary PPM (P6) or PGM (P5), maxval 255.
bool readPnm(const char* path, Image& img)
{
    std::vector<uint8_t> d;
    if (!readFile(path, d) || d.size() < 3 || d[0] != 'P' || (d[1] != '5' && d[1] != '6'))
        return false;
    img.channels = d[1] == '6' ? 3 : 1;
    size_t i = 2;
    int fields[3] = {0, 0, 0};
    for (int k = 0; k < 3; ++k)
    {
        for (;;)
        {
            while (i < d.size() && std::isspace(d[i]))
                ++i;
            if (i < d.size() && d[i] == '#')
                while (i < d.size() && d[i] != '\n')
                    ++i;
            else
                break;
        }
        while (i < d.size() && d[i] >= '0' && d[i] <= '9')
            fields[k] = fields[k] * 10 + (d[i++] - '0');
    }
    ++i;  // the single whitespace byte before the raster
    img.width = fields[0];
    img.height = fields[1];
    const size_t n = (size_t)img.width * img.height * img.channels;
    if (fields[2] != 255 || img.width < 1 || img.height < 1 || i + n > d.size())
        return false;
    img.pixels.assign(d.begin() + (long)i, d.begin() + (long)(i + n));
    return true;
}

int usage()
{
    std::fprintf(stderr,
                 "usage: cheshire_jpeg decode in.jpg out.ppm|out.pgm\n"
                 "       cheshire_jpeg encode in.ppm|in.pgm out.jpg [-q quality] [-s 444|422|420|440|gray] [-r restart]\n"
                 "       cheshire_jpeg bench  in.jpg [repetitions]\n");
    return 2;
}

int fail(Status s)
{
    std::fprintf(stderr, "cheshire_jpeg: %s\n", statusName(s));
    return s == Status::Unsupported || s == Status::Corrupt ? 3 : 1;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
        return usage();
    const std::string cmd = argv[1];
    if (!Codec::deviceAvailable())
    {
        std::fprintf(stderr, "cheshire_jpeg: no GPU device (or CHESHIRE_GPU_JPEG=0)\n");
        return 1;
    }
    Codec codec;

    if (cmd == "decode" && argc == 4)
    {
        std::vector<uint8_t> file;
        if (!readFile(argv[2], file))
            return usage();
        Image img;
        const Status s = codec.decode(file.data(), file.size(), img);
        if (s != Status::Ok)
            return fail(s);
        char header[64];
        std::snprintf(header, sizeof(header), "P%c\n%d %d\n255\n", img.channels == 3 ? '6' : '5', img.width, img.height);
        return writeFile(argv[3], img.pixels.data(), img.pixels.size(), header) ? 0 : 2;
    }
    if (cmd == "encode" && argc >= 4)
    {
        Image img;
        if (!readPnm(argv[2], img))
            return usage();
        EncodeOptions o;
        for (int i = 4; i + 1 < argc; i += 2)
        {
            const std::string k = argv[i], v = argv[i + 1];
            if (k == "-q")
                o.quality = std::atoi(v.c_str());
            else if (k == "-r")
                o.restartInterval = std::atoi(v.c_str());
            else if (k == "-s")
                o.subsampling = v == "444" ? Subsampling::S444
                              : v == "422" ? Subsampling::S422
                              : v == "440" ? Subsampling::S440
                              : v == "gray" ? Subsampling::Gray
                                            : Subsampling::S420;
            else
                return usage();
        }
        std::vector<uint8_t> out;
        const Status s = codec.encode(img.pixels.data(), img.width, img.height, img.channels,
                                      (size_t)img.width * img.channels, o, out);
        if (s != Status::Ok)
            return fail(s);
        return writeFile(argv[3], out.data(), out.size()) ? 0 : 2;
    }
    if (cmd == "bench")
    {
        std::vector<uint8_t> file;
        if (!readFile(argv[2], file))
            return usage();
        const int reps = argc > 3 ? std::max(1, std::atoi(argv[3])) : 10;
        Image img;
        DecodeStats st;
        Status s = codec.decode(file.data(), file.size(), img, &st);  // warm-up, allocates
        if (s != Status::Ok)
            return fail(s);
        std::vector<double> t;
        for (int i = 0; i < reps; ++i)
        {
            const auto t0 = std::chrono::steady_clock::now();
            s = codec.decode(file.data(), file.size(), img);
            t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(t.begin(), t.end());
        std::printf("%s: %dx%dx%d, %u subsequences, %u restart segments, %u sync rounds; decode median %.2f ms "
                    "(min %.2f) over %d\n",
                    argv[2], img.width, img.height, img.channels, st.subsequences, st.restartSegments, st.syncRounds,
                    t[t.size() / 2], t[0], reps);
        return 0;
    }
    return usage();
}
