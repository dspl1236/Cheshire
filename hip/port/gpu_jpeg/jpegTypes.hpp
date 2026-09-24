// Cheshire GPU JPEG: the per-element arithmetic, shared by the device kernels and the CPU check.
//
// Everything here is libjpeg-turbo 2.1's integer path transcribed, not approximated: the
// accurate-integer IDCT (jidctint.c), its post-IDCT range-limit table (jdmaster.c), fancy
// upsampling (jdsample.c), YCbCr<->RGB (jdcolor.c / jccolor.c), the accurate-integer FDCT
// (jfdctint.c), the reciprocal quantiser (jcdctmgr.c) and the sample-plane edge rules of the
// compressor (jcsample.c, jcprepct.c, jccoefct.c). Those are the paths OpenImageIO's JPEG reader
// and writer take with libjpeg's defaults, so "identical to libjpeg-turbo" is the claim, and
// hip/tests/jpeg checks it byte for byte.
//
// Arithmetic is 32-bit where libjpeg's C path uses JLONG (64-bit long on LP64). For coefficients
// a conforming 8-bit encoder can produce no intermediate exceeds 31 bits; idctIslow() checks the
// limits that guarantee it and refuses the block otherwise (see kMaxDequantized).
//
// No device-only intrinsics: every function here compiles as plain C++ so the CPU backend runs the
// same code the kernels do.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define CHESHIRE_JPEG_HD __host__ __device__
#else
#define CHESHIRE_JPEG_HD
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define CHESHIRE_JPEG_DEVICE_PASS 1
#endif

namespace cheshire {
namespace jpeg {

constexpr int kMaxComponents = 4;
constexpr int kMaxBlocksPerMcu = 10;  // JPEG B.2.3: at most 10 blocks in an interleaved MCU

// ---------------------------------------------------------------------------------------------
// Tables the kernels read. Kept in one device allocation per image (see Tables below) rather than
// __constant__ arrays, so the same functions also run on the host.

// jpeg_natural_order with libjpeg's 16 safety entries: a corrupt run length that pushes k past 63
// writes coefficient 63, exactly as jdhuff.c does.
struct Zigzag
{
    uint8_t natural[80];
};

// Decoder Huffman table, canonical form plus a 9-bit lookahead.
struct HuffDecode
{
    int32_t maxcode[18];     // largest code of each length, -1 if none
    int32_t valoffset[17];   // huffval index = code + valoffset[len]
    uint16_t lookup[512];    // (len << 8) | symbol for codes of <= 9 bits, 0 if longer
    uint8_t huffval[256];
};

// Encoder Huffman table: code and length per symbol (jchuff.c's derived table).
struct HuffEncode
{
    uint32_t code[256];
    uint8_t size[256];
};

// jcdctmgr.c compute_reciprocal() for the 8-bit (DCTELEM = short) build, per coefficient in
// natural order, for divisor = quantval << 3.
struct QuantEncode
{
    uint16_t recip[64];
    uint16_t corr[64];
    uint16_t shift[64];  // the full right shift, r (= dtbl[3] + 16)
};

// ---------------------------------------------------------------------------------------------
// Small helpers.

CHESHIRE_JPEG_HD inline int bitLength(uint32_t x)
{
#if defined(CHESHIRE_JPEG_DEVICE_PASS)
    return x ? 32 - __clz((int)x) : 0;
#else
    return x ? 32 - __builtin_clz(x) : 0;
#endif
}

CHESHIRE_JPEG_HD inline int clamp255(int x)
{
    return x < 0 ? 0 : (x > 255 ? 255 : x);
}

CHESHIRE_JPEG_HD inline int clampInt(int x, int lo, int hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

// jdmaster.c prepare_range_limit_table(), post-IDCT half: index (x & 1023) into a table that maps
// -128..127 to 0..255, clamps up to +/-512 and wraps beyond that.
CHESHIRE_JPEG_HD inline uint8_t idctRangeLimit(int x)
{
    const int i = x & 1023;
    if (i < 128)
        return (uint8_t)(i + 128);
    if (i < 512)
        return 255;
    if (i < 896)
        return 0;
    return (uint8_t)(i - 896);
}

// ---------------------------------------------------------------------------------------------
// Accurate integer IDCT, jidctint.c jpeg_idct_islow (CONST_BITS 13, PASS1_BITS 2).

namespace islow {
constexpr int CONST_BITS = 13;
constexpr int PASS1_BITS = 2;
constexpr int32_t FIX_0_298631336 = 2446;
constexpr int32_t FIX_0_390180644 = 3196;
constexpr int32_t FIX_0_541196100 = 4433;
constexpr int32_t FIX_0_765366865 = 6270;
constexpr int32_t FIX_0_899976223 = 7373;
constexpr int32_t FIX_1_175875602 = 9633;
constexpr int32_t FIX_1_501321110 = 12299;
constexpr int32_t FIX_1_847759065 = 15137;
constexpr int32_t FIX_1_961570560 = 16069;
constexpr int32_t FIX_2_053119869 = 16819;
constexpr int32_t FIX_2_562915447 = 20995;
constexpr int32_t FIX_3_072711026 = 25172;

CHESHIRE_JPEG_HD inline int32_t descale(int32_t x, int n)
{
    return (x + (1 << (n - 1))) >> n;
}
}  // namespace islow

// Inputs no 8-bit encoder produces. A real block's dequantised coefficients stay within about
// +/-1500 (DC is exactly -1024 for a black block) and its column-pass outputs well inside +/-16383.
// Within these limits the worst case over every sign pattern is 5.0e8 in the column pass and
// 1.0e9 in the row pass (scripts/jpeg_idct_bounds.py), nothing leaves int16 where libjpeg-turbo's
// SIMD code packs to 16 bits, and so this function, libjpeg's 64-bit C path and its SIMD paths
// agree exactly. Outside them those already disagree with each other; the decoder reports the
// stream as corrupt and the caller's libjpeg decodes it, which is the only answer that is right.
constexpr int32_t kMaxDequantized = 8191;
constexpr int32_t kMaxColumnPass = 16383;

// coef: 64 coefficients in natural order. quant: the component's table in natural order.
// out: 8 rows of 8 samples, row stride in bytes. Returns false if the block is out of range.
CHESHIRE_JPEG_HD inline bool idctIslow(const int16_t* coef, const uint16_t* quant, uint8_t* out, size_t stride)
{
    using namespace islow;
    int32_t ws[64];
    bool inRange = true;
    for (int i = 0; i < 64; ++i)
    {
        const int32_t v = (int32_t)coef[i] * (int32_t)quant[i];
        inRange = inRange && v <= kMaxDequantized && v >= -kMaxDequantized;
    }
    if (!inRange)
        return false;

    for (int c = 0; c < 8; ++c)
    {
        const int16_t* in = coef + c;
        const uint16_t* q = quant + c;
        int32_t* w = ws + c;
        if (in[8] == 0 && in[16] == 0 && in[24] == 0 && in[32] == 0 && in[40] == 0 && in[48] == 0 && in[56] == 0)
        {
            const int32_t dc = ((int32_t)in[0] * (int32_t)q[0]) * (1 << PASS1_BITS);
            for (int r = 0; r < 8; ++r)
                w[8 * r] = dc;
            continue;
        }
        int32_t z2 = (int32_t)in[16] * q[16];
        int32_t z3 = (int32_t)in[48] * q[48];
        int32_t z1 = (z2 + z3) * FIX_0_541196100;
        int32_t tmp2 = z1 + z3 * (-FIX_1_847759065);
        int32_t tmp3 = z1 + z2 * FIX_0_765366865;

        z2 = (int32_t)in[0] * q[0];
        z3 = (int32_t)in[32] * q[32];
        int32_t tmp0 = (z2 + z3) * (1 << CONST_BITS);
        int32_t tmp1 = (z2 - z3) * (1 << CONST_BITS);

        const int32_t tmp10 = tmp0 + tmp3;
        const int32_t tmp13 = tmp0 - tmp3;
        const int32_t tmp11 = tmp1 + tmp2;
        const int32_t tmp12 = tmp1 - tmp2;

        tmp0 = (int32_t)in[56] * q[56];
        tmp1 = (int32_t)in[40] * q[40];
        tmp2 = (int32_t)in[24] * q[24];
        tmp3 = (int32_t)in[8] * q[8];

        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        int32_t z4 = tmp1 + tmp3;
        const int32_t z5 = (z3 + z4) * FIX_1_175875602;

        tmp0 = tmp0 * FIX_0_298631336;
        tmp1 = tmp1 * FIX_2_053119869;
        tmp2 = tmp2 * FIX_3_072711026;
        tmp3 = tmp3 * FIX_1_501321110;
        z1 = z1 * (-FIX_0_899976223);
        z2 = z2 * (-FIX_2_562915447);
        z3 = z3 * (-FIX_1_961570560);
        z4 = z4 * (-FIX_0_390180644);

        z3 += z5;
        z4 += z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        w[0] = descale(tmp10 + tmp3, CONST_BITS - PASS1_BITS);
        w[56] = descale(tmp10 - tmp3, CONST_BITS - PASS1_BITS);
        w[8] = descale(tmp11 + tmp2, CONST_BITS - PASS1_BITS);
        w[48] = descale(tmp11 - tmp2, CONST_BITS - PASS1_BITS);
        w[16] = descale(tmp12 + tmp1, CONST_BITS - PASS1_BITS);
        w[40] = descale(tmp12 - tmp1, CONST_BITS - PASS1_BITS);
        w[24] = descale(tmp13 + tmp0, CONST_BITS - PASS1_BITS);
        w[32] = descale(tmp13 - tmp0, CONST_BITS - PASS1_BITS);
    }

    for (int i = 0; i < 64; ++i)
        inRange = inRange && ws[i] <= kMaxColumnPass && ws[i] >= -kMaxColumnPass;
    if (!inRange)
        return false;

    for (int r = 0; r < 8; ++r)
    {
        const int32_t* w = ws + 8 * r;
        uint8_t* o = out + (size_t)r * stride;
        if (w[1] == 0 && w[2] == 0 && w[3] == 0 && w[4] == 0 && w[5] == 0 && w[6] == 0 && w[7] == 0)
        {
            const uint8_t dc = idctRangeLimit(descale(w[0], PASS1_BITS + 3));
            for (int c = 0; c < 8; ++c)
                o[c] = dc;
            continue;
        }
        int32_t z2 = w[2];
        int32_t z3 = w[6];
        int32_t z1 = (z2 + z3) * FIX_0_541196100;
        int32_t tmp2 = z1 + z3 * (-FIX_1_847759065);
        int32_t tmp3 = z1 + z2 * FIX_0_765366865;

        int32_t tmp0 = (w[0] + w[4]) * (1 << CONST_BITS);
        int32_t tmp1 = (w[0] - w[4]) * (1 << CONST_BITS);

        const int32_t tmp10 = tmp0 + tmp3;
        const int32_t tmp13 = tmp0 - tmp3;
        const int32_t tmp11 = tmp1 + tmp2;
        const int32_t tmp12 = tmp1 - tmp2;

        tmp0 = w[7];
        tmp1 = w[5];
        tmp2 = w[3];
        tmp3 = w[1];

        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        int32_t z4 = tmp1 + tmp3;
        const int32_t z5 = (z3 + z4) * FIX_1_175875602;

        tmp0 = tmp0 * FIX_0_298631336;
        tmp1 = tmp1 * FIX_2_053119869;
        tmp2 = tmp2 * FIX_3_072711026;
        tmp3 = tmp3 * FIX_1_501321110;
        z1 = z1 * (-FIX_0_899976223);
        z2 = z2 * (-FIX_2_562915447);
        z3 = z3 * (-FIX_1_961570560);
        z4 = z4 * (-FIX_0_390180644);

        z3 += z5;
        z4 += z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        const int n = CONST_BITS + PASS1_BITS + 3;
        o[0] = idctRangeLimit(descale(tmp10 + tmp3, n));
        o[7] = idctRangeLimit(descale(tmp10 - tmp3, n));
        o[1] = idctRangeLimit(descale(tmp11 + tmp2, n));
        o[6] = idctRangeLimit(descale(tmp11 - tmp2, n));
        o[2] = idctRangeLimit(descale(tmp12 + tmp1, n));
        o[5] = idctRangeLimit(descale(tmp12 - tmp1, n));
        o[3] = idctRangeLimit(descale(tmp13 + tmp0, n));
        o[4] = idctRangeLimit(descale(tmp13 - tmp0, n));
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// Accurate integer FDCT, jfdctint.c jpeg_fdct_islow, in place on 64 DCTELEMs (short in the 8-bit
// build; the casts below are libjpeg's).

CHESHIRE_JPEG_HD inline void fdctIslow(int16_t* data)
{
    using namespace islow;
    for (int r = 0; r < 8; ++r)
    {
        int16_t* d = data + 8 * r;
        int32_t tmp0 = d[0] + d[7];
        int32_t tmp7 = d[0] - d[7];
        int32_t tmp1 = d[1] + d[6];
        int32_t tmp6 = d[1] - d[6];
        int32_t tmp2 = d[2] + d[5];
        int32_t tmp5 = d[2] - d[5];
        int32_t tmp3 = d[3] + d[4];
        int32_t tmp4 = d[3] - d[4];

        const int32_t tmp10 = tmp0 + tmp3;
        const int32_t tmp13 = tmp0 - tmp3;
        const int32_t tmp11 = tmp1 + tmp2;
        const int32_t tmp12 = tmp1 - tmp2;

        d[0] = (int16_t)((tmp10 + tmp11) * (1 << PASS1_BITS));
        d[4] = (int16_t)((tmp10 - tmp11) * (1 << PASS1_BITS));

        int32_t z1 = (tmp12 + tmp13) * FIX_0_541196100;
        d[2] = (int16_t)descale(z1 + tmp13 * FIX_0_765366865, CONST_BITS - PASS1_BITS);
        d[6] = (int16_t)descale(z1 + tmp12 * (-FIX_1_847759065), CONST_BITS - PASS1_BITS);

        z1 = tmp4 + tmp7;
        int32_t z2 = tmp5 + tmp6;
        int32_t z3 = tmp4 + tmp6;
        int32_t z4 = tmp5 + tmp7;
        const int32_t z5 = (z3 + z4) * FIX_1_175875602;

        tmp4 = tmp4 * FIX_0_298631336;
        tmp5 = tmp5 * FIX_2_053119869;
        tmp6 = tmp6 * FIX_3_072711026;
        tmp7 = tmp7 * FIX_1_501321110;
        z1 = z1 * (-FIX_0_899976223);
        z2 = z2 * (-FIX_2_562915447);
        z3 = z3 * (-FIX_1_961570560);
        z4 = z4 * (-FIX_0_390180644);

        z3 += z5;
        z4 += z5;

        d[7] = (int16_t)descale(tmp4 + z1 + z3, CONST_BITS - PASS1_BITS);
        d[5] = (int16_t)descale(tmp5 + z2 + z4, CONST_BITS - PASS1_BITS);
        d[3] = (int16_t)descale(tmp6 + z2 + z3, CONST_BITS - PASS1_BITS);
        d[1] = (int16_t)descale(tmp7 + z1 + z4, CONST_BITS - PASS1_BITS);
    }
    for (int c = 0; c < 8; ++c)
    {
        int16_t* d = data + c;
        int32_t tmp0 = d[0] + d[56];
        int32_t tmp7 = d[0] - d[56];
        int32_t tmp1 = d[8] + d[48];
        int32_t tmp6 = d[8] - d[48];
        int32_t tmp2 = d[16] + d[40];
        int32_t tmp5 = d[16] - d[40];
        int32_t tmp3 = d[24] + d[32];
        int32_t tmp4 = d[24] - d[32];

        const int32_t tmp10 = tmp0 + tmp3;
        const int32_t tmp13 = tmp0 - tmp3;
        const int32_t tmp11 = tmp1 + tmp2;
        const int32_t tmp12 = tmp1 - tmp2;

        d[0] = (int16_t)descale(tmp10 + tmp11, PASS1_BITS);
        d[32] = (int16_t)descale(tmp10 - tmp11, PASS1_BITS);

        int32_t z1 = (tmp12 + tmp13) * FIX_0_541196100;
        d[16] = (int16_t)descale(z1 + tmp13 * FIX_0_765366865, CONST_BITS + PASS1_BITS);
        d[48] = (int16_t)descale(z1 + tmp12 * (-FIX_1_847759065), CONST_BITS + PASS1_BITS);

        z1 = tmp4 + tmp7;
        int32_t z2 = tmp5 + tmp6;
        int32_t z3 = tmp4 + tmp6;
        int32_t z4 = tmp5 + tmp7;
        const int32_t z5 = (z3 + z4) * FIX_1_175875602;

        tmp4 = tmp4 * FIX_0_298631336;
        tmp5 = tmp5 * FIX_2_053119869;
        tmp6 = tmp6 * FIX_3_072711026;
        tmp7 = tmp7 * FIX_1_501321110;
        z1 = z1 * (-FIX_0_899976223);
        z2 = z2 * (-FIX_2_562915447);
        z3 = z3 * (-FIX_1_961570560);
        z4 = z4 * (-FIX_0_390180644);

        z3 += z5;
        z4 += z5;

        d[56] = (int16_t)descale(tmp4 + z1 + z3, CONST_BITS + PASS1_BITS);
        d[40] = (int16_t)descale(tmp5 + z2 + z4, CONST_BITS + PASS1_BITS);
        d[24] = (int16_t)descale(tmp6 + z2 + z3, CONST_BITS + PASS1_BITS);
        d[8] = (int16_t)descale(tmp7 + z1 + z4, CONST_BITS + PASS1_BITS);
    }
}

// jcdctmgr.c quantize(), 8-bit reciprocal form.
CHESHIRE_JPEG_HD inline int16_t quantize(int16_t v, const QuantEncode& q, int i)
{
    int32_t t = v;
    const bool neg = t < 0;
    if (neg)
        t = -t;
    uint32_t product = (uint32_t)(t + (int32_t)q.corr[i]) * (uint32_t)q.recip[i];
    product >>= q.shift[i];
    int16_t r = (int16_t)product;
    return neg ? (int16_t)-r : r;
}

// ---------------------------------------------------------------------------------------------
// Colour conversion, jdcolor.c / jccolor.c (SCALEBITS 16). FIX() is libjpeg's rounding of the
// double constant; these are the values it produces.

namespace ycc {
constexpr int SCALEBITS = 16;
constexpr int32_t ONE_HALF = (int32_t)1 << (SCALEBITS - 1);
constexpr int32_t fix(double x)
{
    return (int32_t)(x * (1L << SCALEBITS) + 0.5);
}
constexpr int32_t F_1_40200 = fix(1.40200);
constexpr int32_t F_1_77200 = fix(1.77200);
constexpr int32_t F_0_71414 = fix(0.71414);
constexpr int32_t F_0_34414 = fix(0.34414);
constexpr int32_t F_0_29900 = fix(0.29900);
constexpr int32_t F_0_58700 = fix(0.58700);
constexpr int32_t F_0_11400 = fix(0.11400);
constexpr int32_t F_0_16874 = fix(0.16874);
constexpr int32_t F_0_33126 = fix(0.33126);
constexpr int32_t F_0_50000 = fix(0.50000);
constexpr int32_t F_0_41869 = fix(0.41869);
constexpr int32_t F_0_08131 = fix(0.08131);
constexpr int32_t CBCR_OFFSET = (int32_t)128 << SCALEBITS;
}  // namespace ycc

CHESHIRE_JPEG_HD inline void yccToRgb(int y, int cb, int cr, uint8_t* rgb)
{
    using namespace ycc;
    const int32_t x_cb = cb - 128;
    const int32_t x_cr = cr - 128;
    const int crr = (F_1_40200 * x_cr + ONE_HALF) >> SCALEBITS;
    const int cbb = (F_1_77200 * x_cb + ONE_HALF) >> SCALEBITS;
    const int32_t crg = -F_0_71414 * x_cr;
    const int32_t cbg = -F_0_34414 * x_cb + ONE_HALF;
    rgb[0] = (uint8_t)clamp255(y + crr);
    rgb[1] = (uint8_t)clamp255(y + ((cbg + crg) >> SCALEBITS));
    rgb[2] = (uint8_t)clamp255(y + cbb);
}

CHESHIRE_JPEG_HD inline int rgbToY(int r, int g, int b)
{
    using namespace ycc;
    return (F_0_29900 * r + F_0_58700 * g + F_0_11400 * b + ONE_HALF) >> SCALEBITS;
}

CHESHIRE_JPEG_HD inline int rgbToCb(int r, int g, int b)
{
    using namespace ycc;
    return (-F_0_16874 * r - F_0_33126 * g + F_0_50000 * b + CBCR_OFFSET + ONE_HALF - 1) >> SCALEBITS;
}

CHESHIRE_JPEG_HD inline int rgbToCr(int r, int g, int b)
{
    using namespace ycc;
    return (F_0_50000 * r - F_0_41869 * g - F_0_08131 * b + CBCR_OFFSET + ONE_HALF - 1) >> SCALEBITS;
}

// ---------------------------------------------------------------------------------------------
// Huffman bit access. The entropy-coded segment is unstuffed on the host into 32-bit words whose
// bit 31 is the first bit of the stream, with two zero words of padding at the end.

CHESHIRE_JPEG_HD inline uint32_t peek32(const uint32_t* words, uint32_t pos)
{
    const uint32_t i = pos >> 5;
    const uint32_t sh = pos & 31;
    const uint64_t v = ((uint64_t)words[i] << 32) | words[i + 1];
    return (uint32_t)(v >> (32 - sh));
}

// Returns the symbol, sets len (0 for a code no table entry matches).
CHESHIRE_JPEG_HD inline int huffDecode(const HuffDecode& t, uint32_t bits, int& len)
{
    const uint32_t look = t.lookup[bits >> 23];
    if (look)
    {
        len = (int)(look >> 8);
        return (int)(look & 0xff);
    }
    const int32_t c16 = (int32_t)(bits >> 16);
    for (int l = 10; l <= 16; ++l)
    {
        const int32_t code = c16 >> (16 - l);
        if (code <= t.maxcode[l])
        {
            len = l;
            return t.huffval[(code + t.valoffset[l]) & 0xff];
        }
    }
    len = 0;
    return 0;
}

// jdhuff.h HUFF_EXTEND.
CHESHIRE_JPEG_HD inline int32_t huffExtend(uint32_t r, int s)
{
    return (int32_t)r < (1 << (s - 1)) ? (int32_t)r + (int32_t)((uint32_t)-1 << s) + 1 : (int32_t)r;
}

}  // namespace jpeg
}  // namespace cheshire
