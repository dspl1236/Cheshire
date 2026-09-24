// PrepareDenseScene's read phase, step by step (image::readImage for a JPEG into RGBA float, working
// colour space linear), on one thread per step as the node's per-image threads run it:
//   decode8  : the JPEG to 8-bit (libjpeg-turbo through OIIO)            - what CheshireJPG replaces
//   read     : what readImage calls, the JPEG straight to float          - decode + the float conversion
//   metadata : readImageMetadata's second open of the file
//   config   : oiio::ColorConfig(configPath), built per read in readImage
//   convert  : ImageBufAlgo::colorconvert(sRGB -> linear, unpremult) with that config
//   alpha    : ImageBufAlgo::channels to RGBA
//   copy     : get_pixels into the caller's float buffer
// Then two questions about the result. Is each converted value a function of its channel's 8-bit value
// alone (could a 256-entry table replace the conversion)? With AliceVision's config it is not: sRGB to
// linear goes through two near-inverse 3x3 matrices, so the channels mix. And does the direct path of
// generator step 6n - RGBA float straight from the 8-bit pixels, the same processor applied in place -
// give the node's buffer bit for bit, and in how long? Results: docs/04-validation.md, 0.3.4 "the JPEG
// read". Build against OpenImageIO only (Windows: cl /O2 /EHsc /MD /std:c++17 with the vcpkg OIIO).
// usage: pdsreadbench <config.ocio> photo.jpg ...
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/color.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: pdsreadbench <config.ocio> photo.jpg ...\n");
        return 2;
    }
    OIIO::attribute("threads", 1);   // one thread per step: thread-time, as the node's image threads see it
    const std::string cfgPath = argv[1];
    double tot[7] = {0, 0, 0, 0, 0, 0, 0};
    const char* names[7] = {"decode8", "read (float)", "metadata", "config", "convert", "alpha", "copy"};
    size_t lutDiff = 0, lutTotal = 0;
    double tProc = 0, tDirect = 0, tDirectFill = 0, tDirectApply = 0;
    int directSame = 0;
    int n = 0;
    for (int a = 2; a < argc; ++a)
    {
        const std::string path = argv[a];
        auto t0 = Clock::now();
        OIIO::ImageBuf b8(path);
        if (!b8.read(0, 0, true, OIIO::TypeDesc::UINT8)) { std::printf("%s: %s\n", path.c_str(), b8.geterror().c_str()); continue; }
        auto t1 = Clock::now();
        OIIO::ImageSpec configSpec;
        OIIO::ImageBuf inBuf(path, 0, 0, nullptr, &configSpec);
        inBuf.read(0, 0, true, OIIO::TypeDesc::FLOAT);
        auto t2 = Clock::now();
        {
            auto in = OIIO::ImageInput::open(path);
            if (in) { OIIO::ImageSpec s = in->spec(); (void)s; in->close(); }
        }
        auto t3 = Clock::now();
        OIIO::ColorConfig colorConfig(cfgPath);
        auto t4a = Clock::now();
        OIIO::ColorProcessorHandle proc = colorConfig.createColorProcessor("sRGB", "linear");
        auto t4 = Clock::now();
        tProc += ms(t4a, t4);
        OIIO::ImageBuf lin;
        OIIO::ImageBufAlgo::colorconvert(lin, inBuf, "sRGB", "linear", true, "", "", &colorConfig);
        auto t5 = Clock::now();
        const int w = lin.spec().width, h = lin.spec().height;
        OIIO::ImageSpec rgbaSpec(w, h, 3, OIIO::TypeDesc::FLOAT);
        OIIO::ImageBuf rgba(rgbaSpec);
        int order[] = {0, 1, 2, -1};
        float values[] = {0, 0, 0, 1.0f};
        OIIO::ImageBufAlgo::channels(rgba, lin, 4, order, values);
        auto t6 = Clock::now();
        std::vector<float> out(size_t(w) * h * 4);
        OIIO::ROI roi = rgba.roi();
        roi.chbegin = 0;
        roi.chend = 4;
        rgba.get_pixels(roi, OIIO::TypeDesc::FLOAT, out.data());
        auto t7 = Clock::now();
        const double t[7] = {ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4a), ms(t4, t5), ms(t5, t6), ms(t6, t7)};
        for (int i = 0; i < 7; ++i) tot[i] += t[i];
        ++n;
        std::printf("%-50s %dx%d  decode8 %6.1f  read %6.1f  meta %5.1f  config %6.1f  convert %6.1f  alpha %6.1f  copy %5.1f ms\n",
                    path.substr(path.size() > 50 ? path.size() - 50 : 0).c_str(), w, h, t[0], t[1], t[2], t[3], t[4], t[5], t[6]);

        // Is the result a function of (channel, 8-bit value) alone? The first output seen for each
        // (channel, value) is the table; every other pixel must match it bit for bit. Then the fused
        // loop a direct read would run: table lookup and alpha straight into the output buffer.
        {
            const unsigned char* p8 = static_cast<const unsigned char*>(b8.localpixels());
            const int c8 = b8.spec().nchannels;
            float lut[3][256];
            bool seen[3][256] = {};
            size_t diff = 0;
            for (size_t i = 0; i < size_t(w) * h; ++i)
                for (int c = 0; c < 3; ++c)
                {
                    const int v = p8[i * c8 + c];
                    const float got = out[i * 4 + c];
                    if (!seen[c][v]) { seen[c][v] = true; lut[c][v] = got; }
                    else if (std::memcmp(&lut[c][v], &got, 4) != 0) ++diff;
                }
            lutDiff += diff;
            lutTotal += size_t(w) * h * 3;
            // the 8-bit to float step of the float read, as a table of OIIO's own values
            const float* pf = static_cast<const float*>(inBuf.localpixels());
            float ftab[256];
            bool fseen[256] = {};
            size_t fdiff = 0;
            for (size_t i = 0; i < size_t(w) * h * 3; ++i)
            {
                const int v = p8[(i / 3) * c8 + i % 3];
                if (!fseen[v]) { fseen[v] = true; ftab[v] = pf[i]; }
                else if (std::memcmp(&ftab[v], &pf[i], 4) != 0) ++fdiff;
            }
            for (int v = 0; v < 256; ++v) if (!fseen[v]) ftab[v] = v * (1.0f / 255.0f);
            // direct: RGBA float straight from the 8-bit pixels, then the processor in place
            std::vector<float> direct(size_t(w) * h * 4);
            auto d0 = Clock::now();
            for (size_t i = 0; i < size_t(w) * h; ++i)
            {
                float* o = &direct[i * 4];
                const unsigned char* s = &p8[i * c8];
                o[0] = ftab[s[0]];
                o[1] = ftab[s[1]];
                o[2] = ftab[s[2]];
                o[3] = 1.0f;
            }
            auto d1 = Clock::now();
            proc->apply(direct.data(), w, h, 4, sizeof(float), 4 * sizeof(float), OIIO::stride_t(w) * 4 * sizeof(float));
            auto d2 = Clock::now();
            const bool same = std::memcmp(direct.data(), out.data(), direct.size() * 4) == 0;
            size_t ddiff = 0;
            for (size_t i = 0; i < direct.size(); ++i) if (std::memcmp(&direct[i], &out[i], 4) != 0) ++ddiff;
            tDirectFill += ms(d0, d1);
            tDirectApply += ms(d1, d2);
            tDirect += ms(d0, d2);
            directSame += same;
            std::printf("   per-channel table: %zu of %zu values not a function of their 8-bit value (8-bit to float: %zu)\n"
                        "   direct RGBA: fill %.1f ms + apply %.1f ms, %s the node's buffer (%zu values differ)\n",
                        diff, size_t(w) * h * 3, fdiff, ms(d0, d1), ms(d1, d2), same ? "IDENTICAL to" : "DIFFERENT from", ddiff);
        }
    }
    if (!n) return 1;
    double all = 0;
    for (int i = 1; i < 7; ++i) all += tot[i];
    std::printf("\nper image, one thread, mean of %d (the node's read = read + metadata + config + convert + alpha + copy = %.1f ms):\n", n, all / n);
    for (int i = 0; i < 7; ++i)
        std::printf("  %-14s %7.1f ms  %5.1f %%\n", names[i], tot[i] / n, i == 0 ? 100.0 * tot[0] / all : 100.0 * tot[i] / all);
    std::printf("  (decode8 is part of read; the float conversion is read - decode8 = %.1f ms)\n", (tot[1] - tot[0]) / n);
    std::printf("processor creation (inside the node's convert, per read): %.1f ms\n", tProc / n);
    std::printf("direct RGBA path: fill %.1f + apply %.1f = %.1f ms per image, identical on %d of %d images\n",
                tDirectFill / n, tDirectApply / n, tDirect / n, directSame, n);
    return 0;
}
