// CheshireEXR checks: OpenEXR as the reference, plus test-file generation. Shared by
// cheshireexr_cpu_check (host and deferred-async backends) and cheshireexr_gpu_check (the device).
#pragma once

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfStringAttribute.h>
#include <Imath/ImathBox.h>
#include <Imath/half.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace exrcheck {

inline uint32_t lcg(uint32_t& s)
{
    s = s * 1664525u + 1013904223u;
    return s >> 8;
}

struct ChannelSpec
{
    std::string name;
    bool half = true;
};

enum Kind
{
    kNoise,       // random bits: NaNs, infinities and denormals included, mostly incompressible
    kSmooth,      // gradients with a little noise, compresses like an image
    kConstant,    // long runs, RLE's best case
    kExhaustive,  // every half bit pattern, in order
};

// Writes an EXR with OpenEXR. Values are generated per channel and pixel from (kind, seed).
inline bool writeExr(const std::string& path,
                     int w,
                     int h,
                     const std::vector<ChannelSpec>& chans,
                     Imf::Compression comp,
                     Imf::LineOrder order,
                     Kind kind,
                     uint32_t seed,
                     const char* colorSpace = nullptr)
{
    try
    {
        Imf::Header header(w, h);
        header.compression() = comp;
        header.lineOrder() = order;
        if (colorSpace)
            header.insert("AliceVision:ColorSpace", Imf::StringAttribute(colorSpace));
        for (const auto& c : chans)
            header.channels().insert(c.name.c_str(), Imf::Channel(c.half ? Imf::HALF : Imf::FLOAT));
        std::vector<std::vector<uint32_t>> planes(chans.size(), std::vector<uint32_t>((size_t)w * h));
        std::vector<std::vector<uint16_t>> halves(chans.size());
        Imf::FrameBuffer fb;
        uint32_t s = seed;
        for (size_t c = 0; c < chans.size(); ++c)
        {
            std::vector<uint32_t>& p = planes[c];
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                {
                    const size_t i = (size_t)y * w + x;
                    uint32_t v;
                    switch (kind)
                    {
                        case kNoise: v = lcg(s) ^ (lcg(s) << 16); break;
                        case kConstant: v = (x / 97 + y / 13 + (int)c) % 3 == 0 ? 0 : 0x3c003f80u; break;
                        case kExhaustive: v = (uint32_t)((i + c * 4099) & 0xffff); break;
                        default:
                        {
                            const float f = (float)(x * (c + 1)) / (float)(w > 1 ? w : 1) + (float)y / (float)(h > 1 ? h : 1) * 0.5f +
                                            (float)(lcg(s) % 100) * 1e-4f;
                            if (chans[c].half)
                                v = half(f).bits();
                            else
                                std::memcpy(&v, &f, 4);
                        }
                    }
                    p[i] = v;
                }
            if (chans[c].half)
            {
                halves[c].resize(p.size());
                for (size_t i = 0; i < p.size(); ++i)
                    halves[c][i] = (uint16_t)p[i];
                fb.insert(chans[c].name.c_str(), Imf::Slice(Imf::HALF, (char*)halves[c].data(), 2, 2 * (size_t)w));
            }
            else
                fb.insert(chans[c].name.c_str(), Imf::Slice(Imf::FLOAT, (char*)p.data(), 4, 4 * (size_t)w));
        }
        Imf::OutputFile file(path.c_str(), header, 1);
        file.setFrameBuffer(fb);
        file.writePixels(h);
        return true;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "writeExr %s: %s\n", path.c_str(), e.what());
        return false;
    }
}

struct RefRead
{
    bool ok = false;
    std::string error;
    int width = 0, height = 0;
    std::vector<float> pixels;
};

// What Cheshire's direct reader does (image/cheshireExr.cpp): FLOAT slices for R, G, B[, A] or the
// file's only channel, interleaved.
inline RefRead refRead(const std::string& path, int nchannels, int threads = 0)
{
    RefRead r;
    try
    {
        Imf::InputFile file(path.c_str(), threads);
        const Imath::Box2i dw = file.header().dataWindow();
        r.width = dw.max.x - dw.min.x + 1;
        r.height = dw.max.y - dw.min.y + 1;
        r.pixels.assign((size_t)r.width * r.height * nchannels, 0.0f);
        const char* names[4] = {"R", "G", "B", "A"};
        if (nchannels == 1)
            names[0] = file.header().channels().begin().name();
        const size_t xs = sizeof(float) * nchannels, ys = xs * r.width;
        Imf::FrameBuffer fb;
        for (int c = 0; c < nchannels; ++c)
            fb.insert(names[c], Imf::Slice(Imf::FLOAT, (char*)(r.pixels.data() + c), xs, ys));
        file.setFrameBuffer(fb);
        file.readPixels(dw.min.y, dw.max.y);
        r.ok = true;
    }
    catch (const std::exception& e)
    {
        r.error = e.what();
    }
    return r;
}

inline std::vector<uint8_t> readFile(const std::string& path)
{
    std::vector<uint8_t> d;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        return d;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    d.resize(n > 0 ? (size_t)n : 0);
    if (n > 0 && std::fread(d.data(), 1, d.size(), f) != d.size())
        d.clear();
    std::fclose(f);
    return d;
}

inline bool writeBytes(const std::string& path, const std::vector<uint8_t>& d)
{
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f)
        return false;
    const bool ok = std::fwrite(d.data(), 1, d.size(), f) == d.size();
    return std::fclose(f) == 0 && ok;
}

inline const char* compName(Imf::Compression c)
{
    switch (c)
    {
        case Imf::NO_COMPRESSION: return "none";
        case Imf::RLE_COMPRESSION: return "rle";
        case Imf::ZIPS_COMPRESSION: return "zips";
        case Imf::ZIP_COMPRESSION: return "zip";
        case Imf::PIZ_COMPRESSION: return "piz";
        default: return "other";
    }
}

}  // namespace exrcheck
