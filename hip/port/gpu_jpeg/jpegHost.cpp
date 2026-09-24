// Cheshire GPU JPEG: see jpegHost.hpp.
//
// This software is based in part on the work of the Independent JPEG Group. Parts of this file
// reproduce the arithmetic of IJG / libjpeg-turbo source files; ATTRIBUTION.md lists which, with
// their copyright notices and what was changed, and README.ijg carries the IJG License.
#include "jpegHost.hpp"

#include <cstring>

namespace cheshire {
namespace jpeg {

namespace {

const uint8_t kNatural[64] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
                              12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
                              35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
                              58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

const unsigned kStdLumQuant[64] = {16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,
                                   14, 13, 16, 24, 40,  57,  69,  56,  14, 17, 22, 29, 51,  87,  80,  62,
                                   18, 22, 37, 56, 68,  109, 103, 77,  24, 35, 55, 64, 81,  104, 113, 92,
                                   49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};

const unsigned kStdChromQuant[64] = {17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
                                     24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
                                     99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
                                     99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};

const uint8_t kBitsDcLum[17] = {0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
const uint8_t kValDcLum[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const uint8_t kBitsDcChrom[17] = {0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
const uint8_t kValDcChrom[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const uint8_t kBitsAcLum[17] = {0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d};
const uint8_t kValAcLum[162] = {
  0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14,
  0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09,
  0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a,
  0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65,
  0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
  0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9,
  0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca,
  0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
  0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};
const uint8_t kBitsAcChrom[17] = {0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77};
const uint8_t kValAcChrom[162] = {
  0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32,
  0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16,
  0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39,
  0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64,
  0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86,
  0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
  0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8,
  0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
  0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};

uint32_t divRoundUp(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

struct Reader
{
    const uint8_t* d;
    size_t n;
    size_t i;
    bool has(size_t k) const { return i + k <= n; }
    int u8() { return d[i++]; }
    int u16()
    {
        const int v = (d[i] << 8) | d[i + 1];
        i += 2;
        return v;
    }
};

}  // namespace

const char* statusName(Status s)
{
    switch (s)
    {
        case Status::Ok: return "ok";
        case Status::NotJpeg: return "not a JPEG";
        case Status::Unsupported: return "unsupported JPEG";
        case Status::Corrupt: return "corrupt JPEG";
        case Status::NoDevice: return "no GPU device";
        case Status::DeviceError: return "GPU runtime error";
        case Status::InvalidArgument: return "invalid argument";
    }
    return "?";
}

void buildZigzag(Zigzag& z)
{
    for (int i = 0; i < 64; ++i)
        z.natural[i] = kNatural[i];
    for (int i = 64; i < 80; ++i)
        z.natural[i] = 63;
}

const uint8_t* stdBits(bool ac, int index)
{
    return ac ? (index ? kBitsAcChrom : kBitsAcLum) : (index ? kBitsDcChrom : kBitsDcLum);
}

const uint8_t* stdVals(bool ac, int index)
{
    return ac ? (index ? kValAcChrom : kValAcLum) : (index ? kValDcChrom : kValDcLum);
}

Status parseJpeg(const uint8_t* data, size_t size, ParsedJpeg& p)
{
    p = ParsedJpeg();
    if (size < 4 || data[0] != 0xFF || data[1] != 0xD8)
        return Status::NotJpeg;
    Reader r{data, size, 2};
    bool sawFrame = false;
    for (;;)
    {
        // next marker: skip garbage up to 0xFF, then any fill bytes
        while (r.has(1) && data[r.i] != 0xFF)
            ++r.i;
        while (r.has(1) && data[r.i] == 0xFF)
            ++r.i;
        if (!r.has(1))
            return Status::NotJpeg;
        const int m = r.u8();
        if (m == 0xD8 || (m >= 0xD0 && m <= 0xD7) || m == 0x01)
            continue;  // no length
        if (m == 0xD9)
            return Status::NotJpeg;  // EOI before any scan
        if (!r.has(2))
            return Status::NotJpeg;
        const size_t segStart = r.i;
        const int len = r.u16();
        if (len < 2 || !r.has((size_t)len - 2))
            return Status::NotJpeg;
        const size_t segEnd = segStart + (size_t)len;

        if (m == 0xC0 || m == 0xC1)
        {
            if (sawFrame || len < 8)
                return Status::Corrupt;
            const int precision = r.u8();
            p.height = r.u16();
            p.width = r.u16();
            p.ncomp = r.u8();
            if (precision != 8 || p.height == 0 || p.width == 0 || (p.ncomp != 1 && p.ncomp != 3))
                return Status::Unsupported;
            if (len < 8 + 3 * p.ncomp)
                return Status::Corrupt;
            for (int c = 0; c < p.ncomp; ++c)
            {
                p.comp[c].id = r.u8();
                const int hv = r.u8();
                p.comp[c].h = hv >> 4;
                p.comp[c].v = hv & 15;
                p.comp[c].tq = r.u8();
                if (p.comp[c].h < 1 || p.comp[c].h > 4 || p.comp[c].v < 1 || p.comp[c].v > 4 || p.comp[c].tq > 3)
                    return Status::Corrupt;
            }
            sawFrame = true;
        }
        else if ((m >= 0xC2 && m <= 0xCF) && m != 0xC4 && m != 0xC8 && m != 0xCC)
        {
            return Status::Unsupported;  // progressive, lossless, hierarchical, arithmetic
        }
        else if (m == 0xCC)
        {
            return Status::Unsupported;  // arithmetic conditioning
        }
        else if (m == 0xC4)
        {
            while (r.i < segEnd)
            {
                if (!r.has(17))
                    return Status::Corrupt;
                const int tcth = r.u8();
                const int tc = tcth >> 4;
                const int th = tcth & 15;
                if (tc > 1 || th > 3)
                    return Status::Corrupt;
                HuffSpec& h = tc ? p.ac[th] : p.dc[th];
                h = HuffSpec();
                int count = 0;
                for (int l = 1; l <= 16; ++l)
                {
                    h.bits[l] = (uint8_t)r.u8();
                    count += h.bits[l];
                }
                if (count > 256 || r.i + (size_t)count > segEnd)
                    return Status::Corrupt;
                for (int k = 0; k < count; ++k)
                    h.vals[k] = (uint8_t)r.u8();
                h.count = count;
                h.present = true;
            }
        }
        else if (m == 0xDB)
        {
            while (r.i < segEnd)
            {
                const int pqtq = r.u8();
                const int pq = pqtq >> 4;
                const int tq = pqtq & 15;
                if (pq > 1 || tq > 3 || r.i + (pq ? 128u : 64u) > segEnd)
                    return Status::Corrupt;
                for (int k = 0; k < 64; ++k)
                    p.quant[tq][kNatural[k]] = (uint16_t)(pq ? r.u16() : r.u8());
                p.quantPresent[tq] = true;
            }
        }
        else if (m == 0xDD)
        {
            if (len != 4)
                return Status::Corrupt;
            p.restartInterval = (uint32_t)r.u16();
        }
        else if (m == 0xE0)
        {
            const uint8_t* a = data + r.i;
            if (len - 2 >= 14 && a[0] == 'J' && a[1] == 'F' && a[2] == 'I' && a[3] == 'F' && a[4] == 0)
                p.sawJfif = true;
        }
        else if (m == 0xEE)
        {
            const uint8_t* a = data + r.i;
            if (len - 2 >= 12 && a[0] == 'A' && a[1] == 'd' && a[2] == 'o' && a[3] == 'b' && a[4] == 'e')
            {
                p.sawAdobe = true;
                p.adobeTransform = a[11];
            }
        }
        else if (m == 0xDA)
        {
            if (!sawFrame)
                return Status::Corrupt;
            const int ns = r.u8();
            if (ns < 1 || ns > 4 || len != 6 + 2 * ns)
                return Status::Corrupt;
            for (int k = 0; k < ns; ++k)
            {
                const int cs = r.u8();
                const int tdta = r.u8();
                int idx = -1;
                for (int c = 0; c < p.ncomp; ++c)
                    if (p.comp[c].id == cs)
                        idx = c;
                if (idx < 0)
                    return Status::Corrupt;
                for (int j = 0; j < k; ++j)
                    if (p.scanOrder[j] == idx)
                        return Status::Corrupt;
                p.comp[idx].td = tdta >> 4;
                p.comp[idx].ta = tdta & 15;
                if (p.comp[idx].td > 3 || p.comp[idx].ta > 3)
                    return Status::Corrupt;
                p.scanOrder[k] = idx;
            }
            p.scanNcomp = ns;
            const int ss = r.u8(), se = r.u8(), ahal = r.u8();
            if (ss != 0 || se != 63 || ahal != 0)
                return Status::Corrupt;
            if (ns != p.ncomp)
                return Status::Unsupported;  // one scan per component (non-interleaved sequential)
            for (int c = 0; c < p.ncomp; ++c)
            {
                if (!p.quantPresent[p.comp[c].tq] || !p.dc[p.comp[c].td].present || !p.ac[p.comp[c].ta].present)
                    return Status::Corrupt;
            }
            p.entropyBegin = segEnd;
            return Status::Ok;
        }
        r.i = segEnd;
    }
}

Status unstuff(const uint8_t* data, size_t size, size_t begin, Unstuffed& u)
{
    std::vector<uint8_t> bytes;
    bytes.reserve(size - begin);
    u.segBegin.assign(1, 0);
    u.segEnd.clear();
    size_t i = begin;
    int nextRst = 0;
    while (i < size)
    {
        const void* hit = std::memchr(data + i, 0xFF, size - i);
        if (!hit)
        {
            bytes.insert(bytes.end(), data + i, data + size);
            break;  // no EOI; libjpeg would warn and pad, we decode what is there
        }
        const size_t f = (size_t)((const uint8_t*)hit - data);
        bytes.insert(bytes.end(), data + i, data + f);
        size_t j = f + 1;
        while (j < size && data[j] == 0xFF)
            ++j;
        if (j >= size)
            break;
        const uint8_t m = data[j];
        if (m == 0x00)
        {
            bytes.push_back(0xFF);
            i = j + 1;
        }
        else if (m >= 0xD0 && m <= 0xD7)
        {
            if (m != 0xD0 + nextRst)
                return Status::Corrupt;
            nextRst = (nextRst + 1) & 7;
            const uint64_t bit = (uint64_t)bytes.size() * 8;
            u.segEnd.push_back((uint32_t)bit);
            u.segBegin.push_back((uint32_t)bit);
            i = j + 1;
        }
        else
            break;  // the marker after the scan
        if (bytes.size() >= (1u << 28))
            return Status::Unsupported;  // bit offsets are 32-bit
    }
    if (bytes.size() >= (1u << 28))
        return Status::Unsupported;
    u.segEnd.push_back((uint32_t)(bytes.size() * 8));

    const size_t nw = (bytes.size() + 3) / 4;
    u.words.assign(nw + 2, 0);
    bytes.resize(nw * 4, 0);
    for (size_t w = 0; w < nw; ++w)
        u.words[w] = ((uint32_t)bytes[4 * w] << 24) | ((uint32_t)bytes[4 * w + 1] << 16) |
                     ((uint32_t)bytes[4 * w + 2] << 8) | (uint32_t)bytes[4 * w + 3];
    return Status::Ok;
}

void buildHuffDecode(const HuffSpec& spec, HuffDecode& t)
{
    std::memset(&t, 0, sizeof(t));
    int p = 0;
    int32_t code = 0;
    for (int l = 1; l <= 16; ++l)
    {
        const int n = spec.bits[l];
        if (n)
        {
            t.valoffset[l] = p - code;
            for (int k = 0; k < n; ++k, ++p, ++code)
            {
                t.huffval[p] = spec.vals[p];
                if (l <= 9)
                {
                    const int first = code << (9 - l);
                    const int count = 1 << (9 - l);
                    for (int e = 0; e < count && first + e < 512; ++e)
                        t.lookup[first + e] = (uint16_t)((l << 8) | spec.vals[p]);
                }
            }
            t.maxcode[l] = code - 1;
        }
        else
            t.maxcode[l] = -1;
        code <<= 1;
    }
    t.maxcode[17] = 0x7fffffff;
}

void buildHuffEncode(const uint8_t* bits, const uint8_t* vals, HuffEncode& t)
{
    std::memset(&t, 0, sizeof(t));
    int p = 0;
    uint32_t code = 0;
    for (int l = 1; l <= 16; ++l)
    {
        for (int k = 0; k < bits[l]; ++k, ++p, ++code)
        {
            t.code[vals[p]] = code;
            t.size[vals[p]] = (uint8_t)l;
        }
        code <<= 1;
    }
}

void buildQuantEncode(const uint16_t* quantNatural, QuantEncode& q)
{
    for (int i = 0; i < 64; ++i)
    {
        const uint32_t divisor = (uint32_t)quantNatural[i] << 3;
        if (divisor == 1)
        {
            q.recip[i] = 1;
            q.corr[i] = 0;
            q.shift[i] = 0;
            continue;
        }
        int b = 0;
        while ((divisor >> (b + 1)) != 0)
            ++b;  // flss(divisor) - 1
        int r = 16 + b;
        uint32_t fq = (1u << r) / divisor;
        const uint32_t fr = (1u << r) % divisor;
        uint32_t c = divisor / 2;
        if (fr == 0)
        {
            fq >>= 1;
            --r;
        }
        else if (fr <= divisor / 2u)
            ++c;
        else
            ++fq;
        q.recip[i] = (uint16_t)fq;
        q.corr[i] = (uint16_t)c;
        q.shift[i] = (uint16_t)r;
    }
}

void qualityTables(int quality, uint16_t lum[64], uint16_t chrom[64])
{
    if (quality <= 0)
        quality = 1;
    if (quality > 100)
        quality = 100;
    const long scale = quality < 50 ? 5000 / quality : 200 - quality * 2;
    for (int i = 0; i < 64; ++i)
    {
        long a = ((long)kStdLumQuant[i] * scale + 50L) / 100L;
        long b = ((long)kStdChromQuant[i] * scale + 50L) / 100L;
        a = a <= 0 ? 1 : (a > 255 ? 255 : a);  // force_baseline
        b = b <= 0 ? 1 : (b > 255 ? 255 : b);
        lum[i] = (uint16_t)a;
        chrom[i] = (uint16_t)b;
    }
}

Status makeDecodeGeom(const ParsedJpeg& p, DecodeGeom& g)
{
    std::memset(&g, 0, sizeof(g));
    g.width = p.width;
    g.height = p.height;
    g.ncomp = p.ncomp;
    g.maxH = g.maxV = 1;
    for (int c = 0; c < p.ncomp; ++c)
    {
        g.maxH = p.comp[c].h > g.maxH ? p.comp[c].h : g.maxH;
        g.maxV = p.comp[c].v > g.maxV ? p.comp[c].v : g.maxV;
    }
    for (int c = 0; c < p.ncomp; ++c)
    {
        if (g.maxH % p.comp[c].h || g.maxV % p.comp[c].v)
            return Status::Unsupported;  // libjpeg: fractional sampling not implemented
        g.h[c] = p.comp[c].h;
        g.v[c] = p.comp[c].v;
        g.tq[c] = p.comp[c].tq;
        g.dsW[c] = (int)divRoundUp((uint32_t)p.width * (uint32_t)g.h[c], (uint32_t)g.maxH);
        g.dsH[c] = (int)divRoundUp((uint32_t)p.height * (uint32_t)g.v[c], (uint32_t)g.maxV);
    }
    g.interleaved = p.scanNcomp > 1;
    if (g.interleaved)
    {
        g.mcusX = divRoundUp((uint32_t)p.width, 8u * (uint32_t)g.maxH);
        g.mcusY = divRoundUp((uint32_t)p.height, 8u * (uint32_t)g.maxV);
        int b = 0;
        for (int k = 0; k < p.scanNcomp; ++k)
        {
            const int c = p.scanOrder[k];
            for (int dy = 0; dy < g.v[c]; ++dy)
                for (int dx = 0; dx < g.h[c]; ++dx)
                {
                    if (b >= kMaxBlocksPerMcu)
                        return Status::Corrupt;
                    g.blkComp[b] = (uint8_t)c;
                    g.blkDx[b] = (uint8_t)dx;
                    g.blkDy[b] = (uint8_t)dy;
                    g.blkDcTbl[b] = (uint8_t)p.comp[c].td;
                    g.blkAcTbl[b] = (uint8_t)p.comp[c].ta;
                    ++b;
                }
        }
        g.bpm = b;
        size_t off = 0;
        for (int c = 0; c < p.ncomp; ++c)
        {
            g.planeW[c] = (int)g.mcusX * g.h[c] * 8;
            g.planeH[c] = (int)g.mcusY * g.v[c] * 8;
            g.blocksW[c] = (int)g.mcusX * g.h[c];
            g.planeOff[c] = off;
            off += (size_t)g.planeW[c] * g.planeH[c];
        }
    }
    else
    {
        const int c = p.scanOrder[0];
        g.blocksW[0] = (int)divRoundUp((uint32_t)g.dsW[c], 8);
        const int blocksH = (int)divRoundUp((uint32_t)g.dsH[c], 8);
        g.mcusX = (uint32_t)g.blocksW[0];
        g.mcusY = (uint32_t)blocksH;
        g.bpm = 1;
        g.blkComp[0] = (uint8_t)c;
        g.blkDcTbl[0] = (uint8_t)p.comp[c].td;
        g.blkAcTbl[0] = (uint8_t)p.comp[c].ta;
        g.planeW[0] = g.blocksW[0] * 8;
        g.planeH[0] = blocksH * 8;
        g.planeOff[0] = 0;
    }
    g.totalMcus = g.mcusX * g.mcusY;
    g.totalUnits = g.totalMcus * (uint32_t)g.bpm;
    g.restartInterval = p.restartInterval;
    g.uniformTables = 1;
    for (int b = 1; b < g.bpm; ++b)
        if (g.blkDcTbl[b] != g.blkDcTbl[0] || g.blkAcTbl[b] != g.blkAcTbl[0])
            g.uniformTables = 0;

    if (p.ncomp == 1)
    {
        g.colorMode = kGray;
        g.outChannels = 1;
    }
    else
    {
        bool rgb = false;
        if (p.sawJfif)
            rgb = false;
        else if (p.sawAdobe)
            rgb = p.adobeTransform == 0;
        else
            rgb = p.comp[0].id == 82 && p.comp[1].id == 71 && p.comp[2].id == 66;
        g.colorMode = rgb ? kRgbCopy : kYccToRgb;
        g.outChannels = 3;
    }
    return Status::Ok;
}

Status makeEncodeGeom(int width, int height, int channels, size_t rowBytes, const EncodeOptions& o, EncodeGeom& g)
{
    std::memset(&g, 0, sizeof(g));
    if (width < 1 || height < 1 || width > 65535 || height > 65535 || (channels != 1 && channels != 3) ||
        rowBytes < (size_t)width * channels || o.restartInterval < 0 || o.restartInterval > 65535)
        return Status::InvalidArgument;
    g.width = width;
    g.height = height;
    g.inChannels = channels;
    g.inPitch = rowBytes;
    const bool gray = channels == 1 || o.subsampling == Subsampling::Gray;
    g.ncomp = gray ? 1 : 3;
    g.inputMode = channels == 1 ? kInGray : (gray ? kInRgbGray : kInRgbYcc);
    for (int c = 0; c < g.ncomp; ++c)
    {
        g.h[c] = g.v[c] = 1;
        g.tbl[c] = c == 0 ? 0 : 1;
    }
    if (!gray)
    {
        switch (o.subsampling)
        {
            case Subsampling::S444: break;
            case Subsampling::S422: g.h[0] = 2; break;
            case Subsampling::S420: g.h[0] = g.v[0] = 2; break;
            case Subsampling::S440: g.v[0] = 2; break;
            case Subsampling::Gray: break;
        }
    }
    g.maxH = g.h[0];
    g.maxV = g.v[0];
    g.interleaved = g.ncomp > 1;
    size_t off = 0;
    int b = 0;
    for (int c = 0; c < g.ncomp; ++c)
    {
        g.blocksW[c] = (int)divRoundUp((uint32_t)width * (uint32_t)g.h[c], (uint32_t)g.maxH * 8u);
        g.blocksH[c] = (int)divRoundUp((uint32_t)height * (uint32_t)g.v[c], (uint32_t)g.maxV * 8u);
        g.planeW[c] = g.blocksW[c] * 8;
        g.planeH[c] = g.blocksH[c] * 8;
        g.planeOff[c] = off;
        off += (size_t)g.planeW[c] * g.planeH[c];
        g.realRows[c] = (int)divRoundUp((uint32_t)height, (uint32_t)g.maxV) * g.v[c];
        g.firstBlk[c] = b;
        g.nBlk[c] = g.interleaved ? g.h[c] * g.v[c] : 1;
        for (int dy = 0; dy < (g.interleaved ? g.v[c] : 1); ++dy)
            for (int dx = 0; dx < (g.interleaved ? g.h[c] : 1); ++dx)
            {
                g.blkComp[b] = (uint8_t)c;
                g.blkDx[b] = (uint8_t)dx;
                g.blkDy[b] = (uint8_t)dy;
                ++b;
            }
    }
    g.planeTotal = off;
    g.bpm = b;
    if (g.interleaved)
    {
        g.mcusX = divRoundUp((uint32_t)width, 8u * (uint32_t)g.maxH);
        g.mcusY = divRoundUp((uint32_t)height, 8u * (uint32_t)g.maxV);
    }
    else
    {
        g.mcusX = (uint32_t)g.blocksW[0];
        g.mcusY = (uint32_t)g.blocksH[0];
    }
    g.totalMcus = g.mcusX * g.mcusY;
    g.totalUnits = g.totalMcus * (uint32_t)g.bpm;
    g.restartInterval = (uint32_t)o.restartInterval;
    g.numSegments = g.restartInterval ? divRoundUp(g.totalMcus, g.restartInterval) : 1;
    return Status::Ok;
}

namespace {
void put16(std::vector<uint8_t>& o, int v)
{
    o.push_back((uint8_t)(v >> 8));
    o.push_back((uint8_t)v);
}
void marker(std::vector<uint8_t>& o, int m)
{
    o.push_back(0xFF);
    o.push_back((uint8_t)m);
}
}  // namespace

void writeHeaders(std::vector<uint8_t>& out, const EncodeGeom& g, const uint16_t quant[2][64])
{
    marker(out, 0xD8);

    // JFIF APP0: version 1.01, aspect ratio 1:1, no thumbnail (jpeg_set_defaults)
    marker(out, 0xE0);
    put16(out, 16);
    const uint8_t jfif[] = {'J', 'F', 'I', 'F', 0, 1, 1, 0};
    out.insert(out.end(), jfif, jfif + sizeof(jfif));
    put16(out, 1);
    put16(out, 1);
    out.push_back(0);
    out.push_back(0);

    bool sentQ[2] = {false, false};
    for (int c = 0; c < g.ncomp; ++c)
    {
        const int t = g.tbl[c];
        if (sentQ[t])
            continue;
        sentQ[t] = true;
        marker(out, 0xDB);
        put16(out, 64 + 1 + 2);
        out.push_back((uint8_t)t);
        for (int i = 0; i < 64; ++i)
            out.push_back((uint8_t)quant[t][kNatural[i]]);
    }

    marker(out, 0xC0);
    put16(out, 3 * g.ncomp + 2 + 5 + 1);
    out.push_back(8);
    put16(out, g.height);
    put16(out, g.width);
    out.push_back((uint8_t)g.ncomp);
    for (int c = 0; c < g.ncomp; ++c)
    {
        out.push_back((uint8_t)(c + 1));
        out.push_back((uint8_t)((g.h[c] << 4) + g.v[c]));
        out.push_back((uint8_t)g.tbl[c]);
    }

    bool sentDc[2] = {false, false}, sentAc[2] = {false, false};
    for (int c = 0; c < g.ncomp; ++c)
    {
        for (int ac = 0; ac < 2; ++ac)
        {
            const int t = g.tbl[c];
            bool& sent = ac ? sentAc[t] : sentDc[t];
            if (sent)
                continue;
            sent = true;
            const uint8_t* bits = stdBits(ac != 0, t);
            const uint8_t* vals = stdVals(ac != 0, t);
            int count = 0;
            for (int l = 1; l <= 16; ++l)
                count += bits[l];
            marker(out, 0xC4);
            put16(out, 2 + 1 + 16 + count);
            out.push_back((uint8_t)(t + (ac ? 0x10 : 0)));
            for (int l = 1; l <= 16; ++l)
                out.push_back(bits[l]);
            out.insert(out.end(), vals, vals + count);
        }
    }

    if (g.restartInterval)
    {
        marker(out, 0xDD);
        put16(out, 4);
        put16(out, (int)g.restartInterval);
    }

    marker(out, 0xDA);
    put16(out, 2 * g.ncomp + 2 + 1 + 3);
    out.push_back((uint8_t)g.ncomp);
    for (int c = 0; c < g.ncomp; ++c)
    {
        out.push_back((uint8_t)(c + 1));
        out.push_back((uint8_t)((g.tbl[c] << 4) + g.tbl[c]));
    }
    out.push_back(0);
    out.push_back(63);
    out.push_back(0);
}

}  // namespace jpeg
}  // namespace cheshire
