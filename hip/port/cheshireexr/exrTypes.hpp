// CheshireEXR: the per-chunk and per-sample arithmetic, shared by the device kernels and the host
// check.
//
// EXR's ZIP, ZIPS and RLE compressions are lossless, so the bar is simple: the floats a decode
// produces are the ones OpenEXR produces, bit for bit, or the decode reports the file and the
// caller keeps OpenEXR. What has to be reproduced exactly:
//   - zlib's inflate (RFC 1950/1951), including every check zlib makes, so that a stream zlib
//     rejects is rejected here too rather than decoded into something OpenEXR would never return:
//     header, block types, over-subscribed and incomplete codes, invalid symbols, distances
//     reaching before the start, the stored-block length check, the adler32 trailer;
//   - OpenEXR's run-length decoding (internal_rle.c / ImfRleCompressor.cpp);
//   - the predictor and byte reordering both apply after decompression (internal_zip.c
//     reconstruct() and interleave());
//   - half to float as Imath converts it (imath_half_to_float's table, which the formula below
//     reproduces bit for bit, NaN payloads included).
//
// No device-only intrinsics beyond a count-leading-zeros: every function compiles as plain C++,
// so the host check runs the code the kernels run.
//
// Parts of this file reproduce behaviour of OpenEXR and Imath (BSD-3-Clause, Contributors to the
// OpenEXR Project) and the decoding loop of Mark Adler's puff.c (zlib licence); ATTRIBUTION.md
// lists which parts, what was changed, and the licences.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define CHESHIRE_EXR_HD __host__ __device__
#else
#define CHESHIRE_EXR_HD
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define CHESHIRE_EXR_DEVICE_PASS 1
#endif

namespace cheshire {
namespace exr {

// ---------------------------------------------------------------------------------------------
// Half to float, Imath's imath_half_to_float() without F16C: sign, then either a normal number
// rebiased, infinity/NaN with the payload kept, or a denormal normalised.

CHESHIRE_EXR_HD inline uint32_t clz32(uint32_t x)
{
#if defined(CHESHIRE_EXR_DEVICE_PASS)
    return (uint32_t)__clz((int)x);
#else
    return x ? (uint32_t)__builtin_clz(x) : 32u;
#endif
}

CHESHIRE_EXR_HD inline uint32_t halfToFloatBits(uint16_t h)
{
    const uint32_t hexpmant = ((uint32_t)h << 17) >> 4;
    uint32_t v = ((uint32_t)(h >> 15)) << 31;
    if (hexpmant >= 0x00800000u)
    {
        v |= hexpmant;
        if (hexpmant < 0x0f800000u)
            v += 0x38000000u;
        else
            v |= 0x7f800000u;
    }
    else if (hexpmant != 0)
    {
        const uint32_t lc = clz32(hexpmant) - 8;
        v |= 0x38800000u;
        v |= hexpmant << lc;
        v -= lc << 23;
    }
    return v;
}

// ---------------------------------------------------------------------------------------------
// Inflate. One call decodes one zlib stream (an EXR chunk) into dst, which must be exactly the
// chunk's unpacked size: zlib's uncompress() into OpenEXR's buffer fails when the stream is longer,
// and OpenEXR rejects the chunk when it is shorter.

enum InflateError : int
{
    kInflateOk = 0,
    kInflateHeader,        // bad zlib header, or a preset dictionary
    kInflateBlockType,     // block type 3
    kInflateStored,        // stored block LEN != ~NLEN
    kInflateTables,        // bad code-length counts or code
    kInflateCode,          // invalid symbol
    kInflateDistance,      // distance before the start of the output
    kInflateOverflow,      // more output than the chunk holds
    kInflateTruncated,     // ran past the end of the input
    kInflateShort,         // stream ended before filling the chunk
    kInflateChecksum,      // adler32 mismatch
};

constexpr int kMaxBits = 15;
constexpr int kFastBits = 9;

// A canonical Huffman code (RFC 1951 3.2.2): counts per length, symbols in code order, and a
// lookup of the first kFastBits input bits - entries (length << 9) | symbol, 0 for longer codes.
struct HuffTable
{
    uint16_t count[kMaxBits + 1];
    uint16_t symbol[320];
    uint16_t fast[1 << kFastBits];
};

// kind: 0 = code-length code, 1 = literal/length, 2 = distance. Returns false where zlib's
// inflate_table() fails: an over-subscribed set, or an incomplete one other than a single code of
// length 1 (lengths and distances only). A code with no symbols at all is accepted, as zlib does;
// decoding with it then fails.
CHESHIRE_EXR_HD inline bool buildHuff(HuffTable& h, const uint8_t* lengths, int n, int kind)
{
    for (int l = 0; l <= kMaxBits; ++l)
        h.count[l] = 0;
    for (int s = 0; s < n; ++s)
        h.count[lengths[s]]++;
    int maxLen = 0;
    for (int l = kMaxBits; l >= 1; --l)
        if (h.count[l])
        {
            maxLen = l;
            break;
        }
    for (int i = 0; i < (1 << kFastBits); ++i)
        h.fast[i] = 0;
    if (maxLen == 0)
        return kind != 0;
    int left = 1;
    for (int l = 1; l <= kMaxBits; ++l)
    {
        left <<= 1;
        left -= h.count[l];
        if (left < 0)
            return false;  // over-subscribed
    }
    if (left > 0 && (kind == 0 || maxLen != 1))
        return false;  // incomplete
    uint16_t offs[kMaxBits + 2];
    offs[1] = 0;
    for (int l = 1; l < kMaxBits; ++l)
        offs[l + 1] = offs[l] + h.count[l];
    for (int s = 0; s < n; ++s)
        if (lengths[s])
            h.symbol[offs[lengths[s]]++] = (uint16_t)s;
    // fast table: canonical codes, bit-reversed because deflate sends them MSB first
    int code = 0, idx = 0;
    for (int l = 1; l <= kFastBits; ++l)
    {
        for (int k = 0; k < h.count[l]; ++k, ++code, ++idx)
        {
            int rev = 0;
            for (int b = 0; b < l; ++b)
                rev |= ((code >> b) & 1) << (l - 1 - b);
            for (int fill = rev; fill < (1 << kFastBits); fill += 1 << l)
                h.fast[fill] = (uint16_t)((l << 9) | h.symbol[idx]);
        }
        code <<= 1;
    }
    return true;
}

struct BitReader
{
    const uint8_t* p;
    const uint8_t* end;
    uint64_t buf;
    int cnt;
    uint64_t consumed;  // bits taken from the stream
    uint64_t total;     // bits in the stream

    CHESHIRE_EXR_HD void init(const uint8_t* src, uint32_t len)
    {
        p = src;
        end = src + len;
        buf = 0;
        cnt = 0;
        consumed = 0;
        total = (uint64_t)len * 8;
    }
    // Makes n bits visible; past the end they read as zero (overruns are caught by consumed).
    CHESHIRE_EXR_HD void need(int n)
    {
        while (cnt < n)
        {
            const uint64_t byte = p < end ? *p++ : 0;
            buf |= byte << cnt;
            cnt += 8;
        }
    }
    CHESHIRE_EXR_HD uint32_t peek(int n)
    {
        need(n);
        return (uint32_t)(buf & ((1ull << n) - 1));
    }
    CHESHIRE_EXR_HD void drop(int n)
    {
        buf >>= n;
        cnt -= n;
        consumed += (uint64_t)n;
    }
    CHESHIRE_EXR_HD uint32_t bits(int n)
    {
        if (n == 0)
            return 0;
        const uint32_t v = peek(n);
        drop(n);
        return v;
    }
    CHESHIRE_EXR_HD void alignByte() { drop(cnt & 7); }
    CHESHIRE_EXR_HD bool overrun() const { return consumed > total; }
};

// Returns the symbol, or -1 for a code the table does not contain.
CHESHIRE_EXR_HD inline int huffDecode(const HuffTable& h, BitReader& br)
{
    const uint32_t look = br.peek(kMaxBits);
    const uint16_t f = h.fast[look & ((1u << kFastBits) - 1)];
    if (f)
    {
        br.drop(f >> 9);
        return f & 511;
    }
    // canonical decode, one bit at a time (puff.c)
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= kMaxBits; ++len)
    {
        code |= (int)((look >> (len - 1)) & 1);
        const int count = h.count[len];
        if (code - count < first)
        {
            br.drop(len);
            return h.symbol[index + (code - first)];
        }
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    return -1;
}

// Scratch for one inflate call: the three tables and the code lengths. About 3.8 KB; one per
// thread on the device.
struct InflateState
{
    HuffTable lit, dist, lens;
    uint8_t lengths[320];
};

CHESHIRE_EXR_HD inline void fixedTables(InflateState& s)
{
    for (int i = 0; i < 144; ++i)
        s.lengths[i] = 8;
    for (int i = 144; i < 256; ++i)
        s.lengths[i] = 9;
    for (int i = 256; i < 280; ++i)
        s.lengths[i] = 7;
    for (int i = 280; i < 288; ++i)
        s.lengths[i] = 8;
    buildHuff(s.lit, s.lengths, 288, 1);
    // all 32 five-bit distance codes, as zlib builds them; 30 and 31 are rejected when used
    for (int i = 0; i < 32; ++i)
        s.lengths[i] = 5;
    buildHuff(s.dist, s.lengths, 32, 2);
}

CHESHIRE_EXR_HD inline int dynamicTables(InflateState& s, BitReader& br)
{
    const int nlen = (int)br.bits(5) + 257;
    const int ndist = (int)br.bits(5) + 1;
    const int ncode = (int)br.bits(4) + 4;
    if (nlen > 286 || ndist > 30)
        return kInflateTables;
    const uint8_t order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
    uint8_t cl[19];
    for (int i = 0; i < 19; ++i)
        cl[i] = 0;
    for (int i = 0; i < ncode; ++i)
        cl[order[i]] = (uint8_t)br.bits(3);
    if (!buildHuff(s.lens, cl, 19, 0))
        return kInflateTables;
    int idx = 0;
    while (idx < nlen + ndist)
    {
        const int sym = huffDecode(s.lens, br);
        if (sym < 0)
            return kInflateTables;
        if (sym < 16)
        {
            s.lengths[idx++] = (uint8_t)sym;
            continue;
        }
        int len = 0, rep;
        if (sym == 16)
        {
            if (idx == 0)
                return kInflateTables;  // repeat with no previous length
            len = s.lengths[idx - 1];
            rep = 3 + (int)br.bits(2);
        }
        else if (sym == 17)
            rep = 3 + (int)br.bits(3);
        else
            rep = 11 + (int)br.bits(7);
        if (idx + rep > nlen + ndist)
            return kInflateTables;
        while (rep--)
            s.lengths[idx++] = (uint8_t)len;
    }
    if (s.lengths[256] == 0)
        return kInflateTables;  // no end-of-block code
    if (!buildHuff(s.lit, s.lengths, nlen, 1))
        return kInflateTables;
    if (!buildHuff(s.dist, s.lengths + nlen, ndist, 2))
        return kInflateTables;
    return kInflateOk;
}

CHESHIRE_EXR_HD inline uint32_t adler32(const uint8_t* d, uint32_t n)
{
    uint32_t a = 1, b = 0;
    while (n)
    {
        const uint32_t k = n < 5552 ? n : 5552;  // zlib's NMAX: no overflow before the modulo
        for (uint32_t i = 0; i < k; ++i)
        {
            a += d[i];
            b += a;
        }
        a %= 65521u;
        b %= 65521u;
        d += k;
        n -= k;
    }
    return (b << 16) | a;
}

CHESHIRE_EXR_HD inline int zlibInflate(const uint8_t* src, uint32_t srcLen, uint8_t* dst, uint32_t dstLen, InflateState& s)
{
    const uint16_t lbase[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
    const uint8_t lext[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
    const uint16_t dbase[30] = {1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,   65,    97,    129,
                                193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
    const uint8_t dext[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

    if (srcLen < 2)
        return kInflateTruncated;
    const uint32_t cmf = src[0], flg = src[1];
    if ((cmf & 15) != 8 || (cmf >> 4) > 7 || ((cmf << 8) | flg) % 31 != 0 || (flg & 0x20))
        return kInflateHeader;
    BitReader br;
    br.init(src + 2, srcLen - 2);
    uint32_t out = 0;
    int last = 0;
    do
    {
        last = (int)br.bits(1);
        const int type = (int)br.bits(2);
        if (type == 0)
        {
            br.alignByte();
            const uint32_t len = br.bits(16);
            const uint32_t nlen = br.bits(16);
            if (len != (~nlen & 0xffffu))
                return kInflateStored;
            if (br.overrun())
                return kInflateTruncated;
            if (out + len > dstLen)
                return kInflateOverflow;
            for (uint32_t i = 0; i < len; ++i)
                dst[out++] = (uint8_t)br.bits(8);
        }
        else if (type == 1 || type == 2)
        {
            if (type == 1)
                fixedTables(s);
            else
            {
                const int e = dynamicTables(s, br);
                if (e != kInflateOk)
                    return br.overrun() ? kInflateTruncated : e;
            }
            for (;;)
            {
                int sym = huffDecode(s.lit, br);
                if (sym < 0)
                    return br.overrun() ? kInflateTruncated : kInflateCode;
                if (sym < 256)
                {
                    if (out >= dstLen)
                        return kInflateOverflow;
                    dst[out++] = (uint8_t)sym;
                    continue;
                }
                if (sym == 256)
                    break;
                sym -= 257;
                if (sym >= 29)
                    return kInflateCode;  // 286, 287
                const uint32_t len = lbase[sym] + br.bits(lext[sym]);
                const int dsym = huffDecode(s.dist, br);
                if (dsym < 0)
                    return br.overrun() ? kInflateTruncated : kInflateCode;
                if (dsym >= 30)
                    return kInflateCode;  // 30, 31
                const uint32_t dist = dbase[dsym] + br.bits(dext[dsym]);
                if (dist > out)
                    return kInflateDistance;
                if (out + len > dstLen)
                    return kInflateOverflow;
                for (uint32_t i = 0; i < len; ++i, ++out)
                    dst[out] = dst[out - dist];
                if (br.overrun())
                    return kInflateTruncated;
            }
        }
        else
            return kInflateBlockType;
        if (br.overrun())
            return kInflateTruncated;
    } while (!last);
    if (out != dstLen)
        return kInflateShort;
    br.alignByte();
    uint32_t want = 0;
    for (int i = 0; i < 4; ++i)
        want = (want << 8) | br.bits(8);
    if (br.overrun())
        return kInflateTruncated;
    if (want != adler32(dst, out))
        return kInflateChecksum;
    return kInflateOk;
}

// OpenEXR's RLE: a negative count byte -n is followed by n literal bytes, a non-negative count c by
// one byte repeated c + 1 times. The output must be exactly the chunk's unpacked size.
CHESHIRE_EXR_HD inline bool rleDecode(const uint8_t* src, uint32_t srcLen, uint8_t* dst, uint32_t dstLen)
{
    uint32_t in = 0, out = 0;
    while (in < srcLen)
    {
        const int c = (int)(int8_t)src[in++];
        if (c < 0)
        {
            const uint32_t n = (uint32_t)(-c);
            if (in + n > srcLen || out + n > dstLen)
                return false;
            for (uint32_t i = 0; i < n; ++i)
                dst[out++] = src[in++];
        }
        else
        {
            const uint32_t n = (uint32_t)c + 1;
            if (in + 1 > srcLen || out + n > dstLen)
                return false;
            const uint8_t v = src[in++];
            for (uint32_t i = 0; i < n; ++i)
                dst[out++] = v;
        }
    }
    return out == dstLen;
}

// internal_zip.c reconstruct(): t[i] = t[i-1] + t[i] - 128, in place, first byte unchanged.
CHESHIRE_EXR_HD inline void unpredict(uint8_t* t, uint32_t n)
{
    for (uint32_t i = 1; i < n; ++i)
        t[i] = (uint8_t)((int)t[i - 1] + (int)t[i] - 128);
}

// internal_zip.c interleave(): output byte k of a chunk of n bytes is the (k/2)th byte of the first
// half ((n + 1) / 2 bytes) for even k, of the second half for odd k.
CHESHIRE_EXR_HD inline uint32_t interleavedSource(uint32_t k, uint32_t n)
{
    return (k & 1) ? (n + 1) / 2 + (k >> 1) : (k >> 1);
}

}  // namespace exr
}  // namespace cheshire
