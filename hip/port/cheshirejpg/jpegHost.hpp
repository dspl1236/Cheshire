// CheshireJPG: host-side work - marker parsing, unstuffing the entropy-coded data, building
// the tables the kernels read, and writing the headers of an encoded file. Plain C++.
#pragma once

#include "jpegCodec.hpp"
#include "jpegStages.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cheshire {
namespace jpeg {

struct HuffSpec
{
    bool present = false;
    uint8_t bits[17] = {};  // bits[l] = number of codes of length l, l = 1..16
    uint8_t vals[256] = {};
    int count = 0;
};

struct FrameComponent
{
    int id = 0, h = 1, v = 1, tq = 0;
    int td = 0, ta = 0;  // from the scan header
};

struct ParsedJpeg
{
    int width = 0, height = 0, ncomp = 0;
    FrameComponent comp[kMaxComponents];
    uint16_t quant[4][64] = {};  // natural order
    bool quantPresent[4] = {};
    HuffSpec dc[4], ac[4];
    uint32_t restartInterval = 0;
    bool sawJfif = false, sawAdobe = false;
    int adobeTransform = -1;
    int scanNcomp = 0;
    int scanOrder[kMaxComponents] = {};  // frame component index, in scan order
    size_t entropyBegin = 0;             // file offset of the first entropy-coded byte
};

// Reads markers up to the first scan. Unsupported for anything outside the codec's scope.
Status parseJpeg(const uint8_t* data, size_t size, ParsedJpeg& p);

// The entropy-coded data of the scan with byte stuffing and RSTn markers removed, as 32-bit words
// (bit 31 first), plus two zero words. Restart segments start on byte boundaries.
struct Unstuffed
{
    std::vector<uint32_t> words;
    std::vector<uint32_t> segBegin;  // bit offsets
    std::vector<uint32_t> segEnd;
};

Status unstuff(const uint8_t* data, size_t size, size_t begin, Unstuffed& u);

void buildHuffDecode(const HuffSpec& spec, HuffDecode& t);
void buildHuffEncode(const uint8_t* bits, const uint8_t* vals, HuffEncode& t);
void buildQuantEncode(const uint16_t* quantNatural, QuantEncode& q);
void buildZigzag(Zigzag& z);

// jcparam.c jpeg_set_quality(quality, force_baseline = TRUE): the scaled Annex K tables.
void qualityTables(int quality, uint16_t lum[64], uint16_t chrom[64]);

// Annex K.3 standard Huffman tables (jstdhuff.c), index 0 = luminance, 1 = chrominance.
const uint8_t* stdBits(bool ac, int index);
const uint8_t* stdVals(bool ac, int index);

// The decoder's view of a parsed frame. Fails for sampling layouts libjpeg itself rejects.
Status makeDecodeGeom(const ParsedJpeg& p, DecodeGeom& g);

// The encoder's geometry for the given options (fills blocks, planes, MCU layout).
Status makeEncodeGeom(int width, int height, int channels, size_t rowBytes, const EncodeOptions& o, EncodeGeom& g);

// SOI through SOS, byte for byte as libjpeg's jcmarker.c writes them for jpeg_set_defaults.
void writeHeaders(std::vector<uint8_t>& out, const EncodeGeom& g, const uint16_t quant[2][64]);

}  // namespace jpeg
}  // namespace cheshire
