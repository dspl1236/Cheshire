// CheshireJPG checks: libjpeg-turbo as the reference, plus synthetic test images.
// Shared by cheshirejpg_cpu_check (the pipeline on the CPU backend) and cheshirejpg_gpu_check (the device).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csetjmp>
#include <string>
#include <vector>

#include <jpeglib.h>

#include "jpegCodec.hpp"

namespace check {

struct ErrorMgr
{
    jpeg_error_mgr pub;
    jmp_buf jump;
    int warnings = 0;
    char message[JMSG_LENGTH_MAX] = {};
};

inline void onError(j_common_ptr cinfo)
{
    ErrorMgr* e = reinterpret_cast<ErrorMgr*>(cinfo->err);
    (*cinfo->err->format_message)(cinfo, e->message);
    longjmp(e->jump, 1);
}

inline void onMessage(j_common_ptr cinfo, int level)
{
    if (level < 0)
        ++reinterpret_cast<ErrorMgr*>(cinfo->err)->warnings;
}

struct RefDecode
{
    bool ok = false;
    int warnings = 0;
    std::string error;
    int width = 0, height = 0, channels = 0;
    std::vector<uint8_t> pixels;
    int ncomp = 0;
    int widthInBlocks[4] = {}, heightInBlocks[4] = {};
    std::vector<int16_t> coefs[4];  // per component, row-major blocks, natural order
};

// libjpeg's defaults: JDCT_ISLOW, fancy upsampling, RGB (or grayscale) out.
inline RefDecode refDecode(const std::vector<uint8_t>& file, bool withCoefs)
{
    RefDecode r;
    {
        jpeg_decompress_struct d;
        ErrorMgr e;
        d.err = jpeg_std_error(&e.pub);
        e.pub.error_exit = onError;
        e.pub.emit_message = onMessage;
        if (setjmp(e.jump))
        {
            r.error = e.message;
            jpeg_destroy_decompress(&d);
            return r;
        }
        jpeg_create_decompress(&d);
        jpeg_mem_src(&d, file.data(), (unsigned long)file.size());
        jpeg_read_header(&d, TRUE);
        jpeg_start_decompress(&d);
        r.width = (int)d.output_width;
        r.height = (int)d.output_height;
        r.channels = d.output_components;
        r.pixels.resize((size_t)r.width * r.height * r.channels);
        while (d.output_scanline < d.output_height)
        {
            JSAMPROW row = r.pixels.data() + (size_t)d.output_scanline * r.width * r.channels;
            jpeg_read_scanlines(&d, &row, 1);
        }
        jpeg_finish_decompress(&d);
        r.warnings = e.warnings;
        jpeg_destroy_decompress(&d);
    }
    if (withCoefs)
    {
        jpeg_decompress_struct d;
        ErrorMgr e;
        d.err = jpeg_std_error(&e.pub);
        e.pub.error_exit = onError;
        e.pub.emit_message = onMessage;
        if (setjmp(e.jump))
        {
            r.error = e.message;
            jpeg_destroy_decompress(&d);
            return r;
        }
        jpeg_create_decompress(&d);
        jpeg_mem_src(&d, file.data(), (unsigned long)file.size());
        jpeg_read_header(&d, TRUE);
        jvirt_barray_ptr* arrays = jpeg_read_coefficients(&d);
        r.ncomp = d.num_components;
        for (int c = 0; c < d.num_components; ++c)
        {
            const jpeg_component_info& ci = d.comp_info[c];
            r.widthInBlocks[c] = (int)ci.width_in_blocks;
            r.heightInBlocks[c] = (int)ci.height_in_blocks;
            r.coefs[c].resize((size_t)ci.width_in_blocks * ci.height_in_blocks * 64);
            for (JDIMENSION by = 0; by < ci.height_in_blocks; ++by)
            {
                JBLOCKARRAY rows = (*d.mem->access_virt_barray)((j_common_ptr)&d, arrays[c], by, 1, FALSE);
                for (JDIMENSION bx = 0; bx < ci.width_in_blocks; ++bx)
                    std::memcpy(&r.coefs[c][((size_t)by * ci.width_in_blocks + bx) * 64], rows[0][bx], 64 * sizeof(int16_t));
            }
        }
        jpeg_finish_decompress(&d);
        jpeg_destroy_decompress(&d);
    }
    r.ok = true;
    return r;
}

struct EncodeCase
{
    int quality = 75;
    cheshire::jpeg::Subsampling sub = cheshire::jpeg::Subsampling::S420;
    int restart = 0;
    // reference-only knobs for producing decoder inputs the encoder does not make
    bool optimize = false;         // per-image Huffman tables
    bool rgbColorspace = false;    // JCS_RGB (Adobe marker, transform 0)
    int customSampling[3][2] = {}; // non-zero: explicit h,v per component
};

// libjpeg with jpeg_set_defaults + jpeg_set_quality(q, TRUE); the sampling, restart interval and
// the extra knobs above set on top.
inline bool refEncode(const uint8_t* px, int w, int h, int ch, const EncodeCase& k, std::vector<uint8_t>& out, std::string* err = nullptr)
{
    jpeg_compress_struct c;
    ErrorMgr e;
    c.err = jpeg_std_error(&e.pub);
    e.pub.error_exit = onError;
    e.pub.emit_message = onMessage;
    unsigned char* mem = nullptr;
    unsigned long memSize = 0;
    if (setjmp(e.jump))
    {
        if (err)
            *err = e.message;
        jpeg_destroy_compress(&c);
        std::free(mem);
        return false;
    }
    jpeg_create_compress(&c);
    jpeg_mem_dest(&c, &mem, &memSize);
    c.image_width = (JDIMENSION)w;
    c.image_height = (JDIMENSION)h;
    c.input_components = ch;
    c.in_color_space = ch == 1 ? JCS_GRAYSCALE : JCS_RGB;
    jpeg_set_defaults(&c);
    using cheshire::jpeg::Subsampling;
    if (k.rgbColorspace && ch == 3)
        jpeg_set_colorspace(&c, JCS_RGB);
    else if (k.sub == Subsampling::Gray || ch == 1)
        jpeg_set_colorspace(&c, JCS_GRAYSCALE);
    jpeg_set_quality(&c, k.quality, TRUE);
    if (c.num_components == 3)
    {
        int hv[3][2] = {{1, 1}, {1, 1}, {1, 1}};
        switch (k.sub)
        {
            case Subsampling::S422: hv[0][0] = 2; break;
            case Subsampling::S420: hv[0][0] = hv[0][1] = 2; break;
            case Subsampling::S440: hv[0][1] = 2; break;
            default: break;
        }
        if (k.customSampling[0][0])
            std::memcpy(hv, k.customSampling, sizeof(hv));
        for (int i = 0; i < 3; ++i)
        {
            c.comp_info[i].h_samp_factor = hv[i][0];
            c.comp_info[i].v_samp_factor = hv[i][1];
        }
    }
    c.restart_interval = (unsigned)k.restart;
    c.optimize_coding = k.optimize ? TRUE : FALSE;
    jpeg_start_compress(&c, TRUE);
    while (c.next_scanline < c.image_height)
    {
        JSAMPROW row = const_cast<JSAMPROW>(px + (size_t)c.next_scanline * w * ch);
        jpeg_write_scanlines(&c, &row, 1);
    }
    jpeg_finish_compress(&c);
    out.assign(mem, mem + memSize);
    jpeg_destroy_compress(&c);
    std::free(mem);
    return true;
}

// Deterministic test pictures: smooth gradients, hard edges, saturated patches and noise, so that
// the range limits, large coefficients and long zero runs are all exercised.
inline uint32_t lcg(uint32_t& s)
{
    s = s * 1664525u + 1013904223u;
    return s >> 8;
}

inline std::vector<uint8_t> synthImage(int w, int h, int ch, uint32_t seed, int kind)
{
    std::vector<uint8_t> px((size_t)w * h * ch);
    uint32_t s = seed;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < ch; ++c)
            {
                int v;
                switch (kind)
                {
                    case 0:  // noise
                        v = (int)(lcg(s) & 255);
                        break;
                    case 1:  // gradients with a little noise
                        v = (x * 255 / (w > 1 ? w - 1 : 1) * (c + 1) + y * 255 / (h > 1 ? h - 1 : 1) * (3 - c)) / 4 +
                            (int)(lcg(s) % 9) - 4;
                        break;
                    case 2:  // checkerboard of saturated colours
                        v = (((x / 5) + (y / 3) + c) & 1) ? 255 : 0;
                        break;
                    default:  // circles
                    {
                        const int dx = x - w / 2, dy = y - h / 2;
                        v = ((dx * dx + dy * dy) / (7 + 5 * c)) & 255;
                        break;
                    }
                }
                px[((size_t)y * w + x) * ch + c] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
            }
    return px;
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

inline const char* subName(cheshire::jpeg::Subsampling s)
{
    using cheshire::jpeg::Subsampling;
    switch (s)
    {
        case Subsampling::S444: return "444";
        case Subsampling::S422: return "422";
        case Subsampling::S420: return "420";
        case Subsampling::S440: return "440";
        case Subsampling::Gray: return "gray";
    }
    return "?";
}

// First differing byte, or -1.
inline long firstDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b)
{
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; ++i)
        if (a[i] != b[i])
            return (long)i;
    return a.size() == b.size() ? -1 : (long)n;
}

}  // namespace check
