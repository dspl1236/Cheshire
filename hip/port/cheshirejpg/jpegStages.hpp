// CheshireJPG: the pipeline stages as functors.
//
// Every stage is "for i in [0, n): f(i)" over plain structs of pointers, so the same code is a
// kernel on the device (jpegGPU.cu launches forEachKernel<F>) and a loop on the host (the CPU
// backend in cheshiregpu/cpuBackend.hpp). Scans are written the same way - a chunk pass, one serial pass
// over the chunk totals, and a chunk rewrite - so no stage depends on warp or workgroup primitives
// and every line a kernel runs is also run by the host check (hip/tests/cheshirejpg).
//
// Decoding the entropy-coded segment in parallel (the part rocJPEG leaves to fixed-function VCN
// hardware) follows Weissenberger & Schmidt, "Massively Parallel Huffman Decoding on GPUs"
// (ICPP 2018): the unstuffed bitstream is cut into subsequences of kSubseqBits; each thread
// decodes its subsequence from a guessed state; then every subsequence whose entry state differs
// from its left neighbour's exit is decoded again from that exit (walking ahead, see SyncWalk),
// until no entry changes. Huffman codes resynchronise within a few symbols, so this settles in a
// few rounds (2 to 12 on 88 camera JPEGs); it is exact regardless, because the first subsequence
// of every restart segment starts from a known state and the fixed point of
// "entry(i) = exit(i-1)" is the sequential decode.
//
// The state carried across a subsequence boundary is (bit position, block within the MCU,
// coefficient index) - without the block index when every block of the MCU shares tables
// (DecodeGeom::uniformTables). DC prediction and the output block index are not part of it: each
// subsequence records how many blocks it started and its DC differences per block slot, and a
// segmented scan over those gives every subsequence its block index and DC predictors before the
// writing pass. docs/19-cheshirejpg.md has the measurements.
//
// This software is based in part on the work of the Independent JPEG Group. Parts of this file
// reproduce the arithmetic of IJG / libjpeg-turbo source files; ATTRIBUTION.md lists which, with
// their copyright notices and what was changed, and README.ijg carries the IJG License.
#pragma once

#include "jpegTypes.hpp"

namespace cheshire {
namespace jpeg {

constexpr uint32_t kSubseqBits = 1024;  // bits per decoding thread
constexpr uint32_t kScanChunk = 1024;   // elements per thread in the chunked scans
constexpr uint32_t kStuffChunk = 256;   // bytes per thread when byte-stuffing the output

struct Tables
{
    Zigzag zz;
    uint16_t quant[4][64];  // decoder: per table, natural order
    HuffDecode dc[4];
    HuffDecode ac[4];
    HuffEncode edc[2];
    HuffEncode eac[2];
    QuantEncode eq[2];
};

enum ColorMode : int
{
    kGray = 0,      // one component, written as is
    kYccToRgb = 1,  // three components, YCbCr -> RGB
    kRgbCopy = 2,   // three components already RGB (Adobe transform 0, or 'R','G','B' ids)
};

// ---------------------------------------------------------------------------------------------
// Decoder geometry.

struct DecodeGeom
{
    int width, height;
    int ncomp;
    int maxH, maxV;
    int interleaved;
    int bpm;  // blocks per MCU
    uint32_t mcusX, mcusY, totalMcus, totalUnits;
    uint32_t restartInterval;  // in MCUs, 0 = one segment
    uint8_t blkComp[kMaxBlocksPerMcu];
    uint8_t blkDx[kMaxBlocksPerMcu];
    uint8_t blkDy[kMaxBlocksPerMcu];
    uint8_t blkDcTbl[kMaxBlocksPerMcu];
    uint8_t blkAcTbl[kMaxBlocksPerMcu];
    int h[kMaxComponents], v[kMaxComponents], tq[kMaxComponents];
    int planeW[kMaxComponents], planeH[kMaxComponents];  // samples, padded to whole blocks/MCUs
    size_t planeOff[kMaxComponents];
    int dsW[kMaxComponents], dsH[kMaxComponents];  // libjpeg's downsampled_width/height
    int blocksW[kMaxComponents];
    int colorMode;
    int outChannels;
    // Every block of the MCU uses the same DC and the same AC table. The bit parse then does not
    // depend on the block index at all, so synchronisation ignores it (see SubseqRun::dc).
    int uniformTables;
};

struct Subseq
{
    uint32_t begin, end, segEnd;
    uint32_t head;  // 1 if first subsequence of its restart segment
};

struct SyncState
{
    uint32_t pos;
    uint8_t blk, k, pad0, pad1;
};

// Equal as decoder states. With uniform tables the block index is not part of the state: entries
// are then always recorded with blk 0 and exits carry the index relative to the entry.
CHESHIRE_JPEG_HD inline bool sameState(const DecodeGeom& g, const SyncState& a, const SyncState& b)
{
    return a.pos == b.pos && a.k == b.k && (g.uniformTables || a.blk == b.blk);
}

struct SubseqRun
{
    SyncState entry, exit;
    uint32_t blocks;  // blocks whose DC was decoded here
    uint32_t err;     // invalid Huffman code met from this entry
    // DC differences summed per block slot, slot 0 being the block in progress at the entry (or
    // the first one started). The component of slot s is blkComp[(entry.blk + s) % bpm], known
    // once entry.blk is - which with uniform tables is only after the block counts are scanned.
    int32_t dc[kMaxBlocksPerMcu];
};

CHESHIRE_JPEG_HD inline int32_t runCompSum(const DecodeGeom& g, const SubseqRun& r, int c)
{
    int32_t sum = 0;
    for (int s = 0; s < g.bpm; ++s)
        if (g.blkComp[(r.entry.blk + s) % g.bpm] == c)
            sum += r.dc[s];
    return sum;
}

struct SubseqPrefix
{
    uint32_t unitStart;  // blocks started before this subsequence, image-wide
    int32_t pred[kMaxComponents];
};

// Decode from st until the next symbol would start at or after `end`, or would run past
// `segEnd`. Counting mode (WRITE false): dc[] accumulates DC differences per block slot relative
// to the entry. Writing mode: dc[] holds the per-component predictors and coefficients go to
// coefs[unit * 64 + ...].
template <bool WRITE>
CHESHIRE_JPEG_HD inline SyncState decodeRun(const DecodeGeom& g,
                                             const Tables* T,
                                             const uint32_t* words,
                                             SyncState st,
                                             uint32_t end,
                                             uint32_t segEnd,
                                             uint32_t& blocks,
                                             uint32_t& err,
                                             int32_t* dc,
                                             uint32_t unit,
                                             int16_t* coefs)
{
    int slot = 0;
    while (st.pos < end)
    {
        const int blk = st.blk;
        const uint32_t bits = peek32(words, st.pos);
        int len = 0;
        if (st.k == 0)
        {
            const int s = huffDecode(T->dc[g.blkDcTbl[blk]], bits, len);
            if (len == 0 && st.pos + 16 > segEnd)
                break;  // the fill bits before a marker, followed by the next segment's data
            if (len == 0 || s > 15)
            {
                err = 1;
                st.pos = end;
                break;
            }
            if (st.pos + (uint32_t)(len + s) > segEnd)
                break;
            int32_t diff = 0;
            if (s)
                diff = huffExtend(peek32(words, st.pos + len) >> (32 - s), s);
            st.pos += (uint32_t)(len + s);
            ++blocks;
            if (WRITE)
            {
                const int comp = g.blkComp[blk];
                dc[comp] += diff;
                coefs[(size_t)unit * 64] = (int16_t)dc[comp];
            }
            else
                dc[slot] += diff;
            st.k = 1;
        }
        else
        {
            const int rs = huffDecode(T->ac[g.blkAcTbl[blk]], bits, len);
            if (len == 0 && st.pos + 16 > segEnd)
                break;
            if (len == 0)
            {
                err = 1;
                st.pos = end;
                break;
            }
            const int r = rs >> 4;
            const int s = rs & 15;
            if (st.pos + (uint32_t)(len + s) > segEnd)
                break;
            if (s)
            {
                const int k = st.k + r;  // at most 78; natural[] maps past-the-end to 63
                if (WRITE)
                    coefs[(size_t)unit * 64 + T->zz.natural[k]] =
                      (int16_t)huffExtend(peek32(words, st.pos + len) >> (32 - s), s);
                st.k = (uint8_t)(k + 1);
            }
            else if (r == 15)
                st.k = (uint8_t)(st.k + 16);
            else
                st.k = 64;
            st.pos += (uint32_t)(len + s);
        }
        if (st.k >= 64)
        {
            st.k = 0;
            ++unit;
            if (++st.blk == g.bpm)
                st.blk = 0;
            if (++slot == g.bpm)
                slot = 0;
        }
    }
    return st;
}

CHESHIRE_JPEG_HD inline void runFrom(const DecodeGeom& g,
                                     const Tables* T,
                                     const uint32_t* words,
                                     const Subseq& ss,
                                     SyncState entry,
                                     SubseqRun& run)
{
    if (g.uniformTables)
        entry.blk = 0;
    run.entry = entry;
    run.blocks = 0;
    run.err = 0;
    for (int s = 0; s < kMaxBlocksPerMcu; ++s)
        run.dc[s] = 0;
    run.exit = decodeRun<false>(g, T, words, entry, ss.end, ss.segEnd, run.blocks, run.err, run.dc, 0, nullptr);
}

struct DecodeInitial
{
    DecodeGeom g;
    const Tables* T;
    const uint32_t* words;
    const Subseq* subseqs;
    SubseqRun* runs;

    CHESHIRE_JPEG_HD void operator()(size_t i) const
    {
        SyncState s;
        s.pos = subseqs[i].begin;
        s.blk = 0;
        s.k = 0;
        s.pad0 = s.pad1 = 0;
        SubseqRun r;
        runFrom(g, T, words, subseqs[i], s, r);
        runs[i] = r;
    }
};

// Synchronisation rounds ping-pong between two run arrays, so a round reads only the previous
// round's results. SyncPrepare copies every run and flags the subsequences whose entry no longer
// matches their left neighbour's exit; SyncWalk re-decodes each flagged subsequence from that exit
// and keeps going into the following ones - which it alone writes, since it stops at the next
// flagged index - until its state meets an entry recorded in the previous round (resynchronised)
// or kMaxWalk subsequences have been decoded. Without the walk, a chain of unsynchronised
// subsequences shortens by one per round; with it, by up to kMaxWalk.
constexpr uint32_t kMaxWalk = 32;

struct SyncPrepare
{
    DecodeGeom g;
    const Subseq* subseqs;
    const SubseqRun* in;
    SubseqRun* out;
    uint8_t* flags;

    CHESHIRE_JPEG_HD void operator()(size_t i) const
    {
        out[i] = in[i];
        flags[i] = !subseqs[i].head && !sameState(g, in[i - 1].exit, in[i].entry);
    }
};

struct SyncWalk
{
    DecodeGeom g;
    const Tables* T;
    const uint32_t* words;
    const Subseq* subseqs;
    const SubseqRun* in;
    const uint8_t* flags;
    SubseqRun* out;
    uint32_t* changed;
    size_t n;

    CHESHIRE_JPEG_HD void operator()(size_t i) const
    {
        if (!flags[i])
            return;
        SyncState st = in[i - 1].exit;
        size_t j = i;
        for (uint32_t steps = 0;;)
        {
            SubseqRun r;
            runFrom(g, T, words, subseqs[j], st, r);
            out[j] = r;
            st = r.exit;
            ++j;
            ++steps;
            if (j >= n || subseqs[j].head || flags[j] || steps >= kMaxWalk || sameState(g, st, in[j].entry))
                break;
        }
        *changed = 1;
    }
};

// Segmented scan over the runs: block starts image-wide, DC sums reset at each segment head.
struct RunChunkAgg
{
    uint32_t blocks;
    uint32_t hasHead;
    int32_t dcTail[kMaxComponents];  // DC sum after the chunk's last head (or from its start)
};

struct RunScanChunk
{
    DecodeGeom g;
    const Subseq* subseqs;
    const SubseqRun* runs;
    RunChunkAgg* agg;
    size_t n;

    CHESHIRE_JPEG_HD void operator()(size_t j) const
    {
        RunChunkAgg a;
        a.blocks = 0;
        a.hasHead = 0;
        for (int c = 0; c < kMaxComponents; ++c)
            a.dcTail[c] = 0;
        const size_t e = (j + 1) * kScanChunk < n ? (j + 1) * kScanChunk : n;
        for (size_t i = j * kScanChunk; i < e; ++i)
        {
            if (subseqs[i].head)
            {
                a.hasHead = 1;
                for (int c = 0; c < kMaxComponents; ++c)
                    a.dcTail[c] = 0;
            }
            a.blocks += runs[i].blocks;
            for (int c = 0; c < g.ncomp; ++c)
                a.dcTail[c] += runCompSum(g, runs[i], c);
        }
        agg[j] = a;
    }
};

struct RunScanSerial
{
    RunChunkAgg* agg;  // rewritten in place: blocks = base, dcTail = carry-in
    size_t nChunks;

    CHESHIRE_JPEG_HD void operator()(size_t) const
    {
        uint32_t base = 0;
        int32_t carry[kMaxComponents] = {0, 0, 0, 0};
        for (size_t j = 0; j < nChunks; ++j)
        {
            const RunChunkAgg a = agg[j];
            agg[j].blocks = base;
            for (int c = 0; c < kMaxComponents; ++c)
                agg[j].dcTail[c] = carry[c];
            base += a.blocks;
            for (int c = 0; c < kMaxComponents; ++c)
                carry[c] = a.hasHead ? a.dcTail[c] : carry[c] + a.dcTail[c];
        }
    }
};

struct RunScanApply
{
    DecodeGeom g;
    const Subseq* subseqs;
    const SubseqRun* runs;
    const RunChunkAgg* agg;
    SubseqPrefix* prefix;
    size_t n;

    CHESHIRE_JPEG_HD void operator()(size_t j) const
    {
        uint32_t unit = agg[j].blocks;
        int32_t pred[kMaxComponents];
        for (int c = 0; c < kMaxComponents; ++c)
            pred[c] = agg[j].dcTail[c];
        const size_t e = (j + 1) * kScanChunk < n ? (j + 1) * kScanChunk : n;
        for (size_t i = j * kScanChunk; i < e; ++i)
        {
            if (subseqs[i].head)
                for (int c = 0; c < kMaxComponents; ++c)
                    pred[c] = 0;
            SubseqPrefix p;
            p.unitStart = unit;
            for (int c = 0; c < kMaxComponents; ++c)
                p.pred[c] = pred[c];
            prefix[i] = p;
            unit += runs[i].blocks;
            for (int c = 0; c < g.ncomp; ++c)
                pred[c] += runCompSum(g, runs[i], c);
        }
    }
};

// Uniform tables: the entries were synchronised without a block index; the scanned block counts
// give it. After this the runs read like any others and the scan is repeated for the DC sums.
struct ResolveBlockIndex
{
    DecodeGeom g;
    const SubseqPrefix* prefix;
    SubseqRun* runs;

    CHESHIRE_JPEG_HD void operator()(size_t i) const
    {
        uint32_t u = prefix[i].unitStart;
        if (runs[i].entry.k && u)
            --u;
        runs[i].entry.blk = (uint8_t)(u % (uint32_t)g.bpm);
    }
};

// Per restart segment: the synchronised decode must have produced exactly the segment's blocks,
// ended on an MCU boundary and met no invalid code. Anything else is a damaged or unsupported
// stream and the caller falls back to libjpeg.
struct DecodeVerify
{
    DecodeGeom g;
    const uint32_t* segFirst;  // nSeg + 1 subsequence indices
    const SubseqRun* runs;
    const SubseqPrefix* prefix;
    uint32_t* error;

    CHESHIRE_JPEG_HD void operator()(size_t s) const
    {
        const uint32_t first = segFirst[s];
        const uint32_t last = segFirst[s + 1] - 1;
        const uint32_t mcusPerSeg = g.restartInterval ? g.restartInterval : g.totalMcus;
        const uint32_t mcu0 = (uint32_t)s * mcusPerSeg;
        const uint32_t mcus = g.totalMcus - mcu0 < mcusPerSeg ? g.totalMcus - mcu0 : mcusPerSeg;
        const uint32_t expectStart = mcu0 * (uint32_t)g.bpm;
        const uint32_t got = prefix[last].unitStart + runs[last].blocks - prefix[first].unitStart;
        bool bad = prefix[first].unitStart != expectStart || got != mcus * (uint32_t)g.bpm ||
                   runs[last].exit.k != 0 || (!g.uniformTables && runs[last].exit.blk != 0);
        for (uint32_t i = first; i <= last && !bad; ++i)
            bad = runs[i].err != 0;
        if (bad)
            *error = 1;
    }
};

struct DecodeWrite
{
    DecodeGeom g;
    const Tables* T;
    const uint32_t* words;
    const Subseq* subseqs;
    const SubseqRun* runs;
    const SubseqPrefix* prefix;
    int16_t* coefs;

    CHESHIRE_JPEG_HD void operator()(size_t i) const
    {
        const SubseqRun& r = runs[i];
        int32_t pred[kMaxComponents];
        for (int c = 0; c < kMaxComponents; ++c)
            pred[c] = prefix[i].pred[c];
        // mid-block entry: the block was started (and counted) by an earlier subsequence
        const uint32_t unit = prefix[i].unitStart - (r.entry.k ? 1u : 0u);
        uint32_t blocks = 0, err = 0;
        decodeRun<true>(g, T, words, r.entry, subseqs[i].end, subseqs[i].segEnd, blocks, err, pred, unit, coefs);
    }
};

CHESHIRE_JPEG_HD inline void unitToBlock(const DecodeGeom& g, uint32_t u, int& c, int& bx, int& by)
{
    if (g.interleaved)
    {
        const uint32_t mcu = u / (uint32_t)g.bpm;
        const int blk = (int)(u - mcu * (uint32_t)g.bpm);
        c = g.blkComp[blk];
        const uint32_t mx = mcu % g.mcusX;
        const uint32_t my = mcu / g.mcusX;
        bx = (int)mx * g.h[c] + g.blkDx[blk];
        by = (int)my * g.v[c] + g.blkDy[blk];
    }
    else
    {
        c = 0;
        bx = (int)(u % (uint32_t)g.blocksW[0]);
        by = (int)(u / (uint32_t)g.blocksW[0]);
    }
}

struct DecodeIdct
{
    DecodeGeom g;
    const Tables* T;
    const int16_t* coefs;
    uint8_t* planes;
    uint32_t* error;  // set if a block is outside what an 8-bit encoder produces

    CHESHIRE_JPEG_HD void operator()(size_t u) const
    {
        int c, bx, by;
        unitToBlock(g, (uint32_t)u, c, bx, by);
        uint8_t* out = planes + g.planeOff[c] + (size_t)by * 8 * g.planeW[c] + (size_t)bx * 8;
        if (!idctIslow(coefs + u * 64, T->quant[g.tq[c]], out, (size_t)g.planeW[c]))
            *error = 1;
    }
};

// jdsample.c for one output sample of component c at full-resolution (x, y). Edge rows are
// replicated as jdmainct.c's context pointers do; edge columns as the fancy upsamplers' special
// cases do (those are the same as clamping).
CHESHIRE_JPEG_HD inline int upsampled(const DecodeGeom& g, const uint8_t* planes, int c, int x, int y)
{
    const uint8_t* P = planes + g.planeOff[c];
    const int W = g.planeW[c];
    const int hx = g.maxH / g.h[c];
    const int vy = g.maxV / g.v[c];
    if (hx == 1 && vy == 1)
        return P[(size_t)y * W + x];
    const int dsW = g.dsW[c];
    const int dsH = g.dsH[c];
    if (hx == 2 && vy == 1 && dsW > 2)
    {
        const uint8_t* row = P + (size_t)y * W;
        const int col = x >> 1;
        if ((x & 1) == 0)
            return (row[col] * 3 + row[col > 0 ? col - 1 : 0] + 1) >> 2;
        return (row[col] * 3 + row[col < dsW - 1 ? col + 1 : dsW - 1] + 2) >> 2;
    }
    if (hx == 1 && vy == 2)
    {
        const int r = y >> 1;
        const int r1 = (y & 1) ? clampInt(r + 1, 0, dsH - 1) : clampInt(r - 1, 0, dsH - 1);
        const int bias = (y & 1) ? 2 : 1;
        return (P[(size_t)r * W + x] * 3 + P[(size_t)r1 * W + x] + bias) >> 2;
    }
    if (hx == 2 && vy == 2 && dsW > 2)
    {
        const int r = y >> 1;
        const int r1 = (y & 1) ? clampInt(r + 1, 0, dsH - 1) : clampInt(r - 1, 0, dsH - 1);
        const uint8_t* in0 = P + (size_t)r * W;
        const uint8_t* in1 = P + (size_t)r1 * W;
        const int col = x >> 1;
        const int thiscol = in0[col] * 3 + in1[col];
        if ((x & 1) == 0)
        {
            const int lc = col > 0 ? col - 1 : 0;
            return (thiscol * 3 + in0[lc] * 3 + in1[lc] + 8) >> 4;
        }
        const int nc = col < dsW - 1 ? col + 1 : dsW - 1;
        return (thiscol * 3 + in0[nc] * 3 + in1[nc] + 7) >> 4;
    }
    // int_upsample, and h2v1/h2v2 on components too narrow for the fancy path: replication
    return P[(size_t)(y / vy) * W + x / hx];
}

struct DecodeColor
{
    DecodeGeom g;
    const uint8_t* planes;
    uint8_t* out;
    size_t outPitch;

    CHESHIRE_JPEG_HD void operator()(size_t i) const
    {
        const int x = (int)(i % (size_t)g.width);
        const int y = (int)(i / (size_t)g.width);
        uint8_t* o = out + (size_t)y * outPitch + (size_t)x * g.outChannels;
        if (g.colorMode == kGray)
        {
            o[0] = (uint8_t)upsampled(g, planes, 0, x, y);
            return;
        }
        const int a = upsampled(g, planes, 0, x, y);
        const int b = upsampled(g, planes, 1, x, y);
        const int c = upsampled(g, planes, 2, x, y);
        if (g.colorMode == kYccToRgb)
            yccToRgb(a, b, c, o);
        else
        {
            o[0] = (uint8_t)a;
            o[1] = (uint8_t)b;
            o[2] = (uint8_t)c;
        }
    }
};

// ---------------------------------------------------------------------------------------------
// Encoder geometry.

enum InputMode : int
{
    kInGray = 0,     // one channel in, one component out
    kInRgbYcc = 1,   // RGB in, YCbCr out
    kInRgbGray = 2,  // RGB in, luminance only
};

struct EncodeGeom
{
    int width, height;
    int inChannels;
    size_t inPitch;
    int inputMode;
    int ncomp;
    int maxH, maxV;
    int interleaved;
    int bpm;
    uint32_t mcusX, mcusY, totalMcus, totalUnits;
    uint32_t restartInterval;  // MCUs per segment, 0 = one segment
    uint32_t numSegments;
    uint8_t blkComp[kMaxBlocksPerMcu];
    uint8_t blkDx[kMaxBlocksPerMcu];
    uint8_t blkDy[kMaxBlocksPerMcu];
    int h[kMaxComponents], v[kMaxComponents];
    int blocksW[kMaxComponents], blocksH[kMaxComponents];  // width/height_in_blocks
    int planeW[kMaxComponents], planeH[kMaxComponents];
    size_t planeOff[kMaxComponents];
    size_t planeTotal;
    int realRows[kMaxComponents];  // downsampled rows before jcprepct's bottom padding
    int tbl[kMaxComponents];       // quant and Huffman table index (0 luma, 1 chroma)
    int firstBlk[kMaxComponents], nBlk[kMaxComponents];
};

CHESHIRE_JPEG_HD inline int inputSample(const EncodeGeom& g, const uint8_t* in, int c, int x, int y)
{
    const uint8_t* p = in + (size_t)y * g.inPitch + (size_t)x * g.inChannels;
    if (g.inputMode == kInGray)
        return p[0];
    if (g.inputMode == kInRgbGray)
        return rgbToY(p[0], p[1], p[2]);
    if (c == 0)
        return rgbToY(p[0], p[1], p[2]);
    if (c == 1)
        return rgbToCb(p[0], p[1], p[2]);
    return rgbToCr(p[0], p[1], p[2]);
}

// One sample of a component plane, as jccolor.c + jcsample.c + jcprepct.c produce it: columns past
// the image replicate the last input column (expand_right_edge before downsampling), rows past the
// last input row replicate it, and downsampled rows past the last real row group replicate the
// last downsampled row.
struct EncodePlanes
{
    EncodeGeom g;
    const uint8_t* in;
    uint8_t* planes;

    CHESHIRE_JPEG_HD void operator()(size_t i) const
    {
        int c = 0;
        while (c + 1 < g.ncomp && i >= g.planeOff[c + 1])
            ++c;
        const size_t local = i - g.planeOff[c];
        const int x = (int)(local % (size_t)g.planeW[c]);
        const int y = (int)(local / (size_t)g.planeW[c]);
        const int hx = g.maxH / g.h[c];
        const int vy = g.maxV / g.v[c];
        const int W1 = g.width - 1;
        const int H1 = g.height - 1;
        int value;
        if (hx == 1 && vy == 1)
            value = inputSample(g, in, c, x < W1 ? x : W1, y < H1 ? y : H1);
        else
        {
            const int r = y < g.realRows[c] - 1 ? y : g.realRows[c] - 1;
            int sum = 0;
            for (int dy = 0; dy < vy; ++dy)
            {
                int yy = r * vy + dy;
                yy = yy < H1 ? yy : H1;
                for (int dx = 0; dx < hx; ++dx)
                {
                    int xx = x * hx + dx;
                    xx = xx < W1 ? xx : W1;
                    sum += inputSample(g, in, c, xx, yy);
                }
            }
            if (hx == 2 && vy == 1)
                value = (sum + (x & 1)) >> 1;  // h2v1_downsample: bias 0,1,0,1
            else if (hx == 2 && vy == 2)
                value = (sum + 1 + (x & 1)) >> 2;  // h2v2_downsample: bias 1,2,1,2
            else
                value = (sum + hx * vy / 2) / (hx * vy);  // int_downsample
        }
        planes[i] = (uint8_t)value;
    }
};

CHESHIRE_JPEG_HD inline void encUnitToBlock(const EncodeGeom& g, uint32_t u, int& c, int& bx, int& by, uint32_t& mcu, int& blk)
{
    if (g.interleaved)
    {
        mcu = u / (uint32_t)g.bpm;
        blk = (int)(u - mcu * (uint32_t)g.bpm);
        c = g.blkComp[blk];
        bx = (int)(mcu % g.mcusX) * g.h[c] + g.blkDx[blk];
        by = (int)(mcu / g.mcusX) * g.v[c] + g.blkDy[blk];
    }
    else
    {
        mcu = u;
        blk = 0;
        c = 0;
        bx = (int)(u % (uint32_t)g.blocksW[0]);
        by = (int)(u / (uint32_t)g.blocksW[0]);
    }
}

CHESHIRE_JPEG_HD inline uint32_t encBlockToUnit(const EncodeGeom& g, int c, int bx, int by)
{
    if (!g.interleaved)
        return (uint32_t)by * (uint32_t)g.blocksW[0] + (uint32_t)bx;
    const uint32_t mcu = (uint32_t)(by / g.v[c]) * g.mcusX + (uint32_t)(bx / g.h[c]);
    const int blk = g.firstBlk[c] + (by % g.v[c]) * g.h[c] + (bx % g.h[c]);
    return mcu * (uint32_t)g.bpm + (uint32_t)blk;
}

struct EncodeBlocks
{
    EncodeGeom g;
    const Tables* T;
    const uint8_t* planes;
    int16_t* coefs;

    CHESHIRE_JPEG_HD void operator()(size_t u) const
    {
        int c, bx, by, blk;
        uint32_t mcu;
        encUnitToBlock(g, (uint32_t)u, c, bx, by, mcu, blk);
        if (bx >= g.blocksW[c] || by >= g.blocksH[c])
            return;  // dummy block, EncodeDummies fills it
        int16_t ws[64];
        const uint8_t* src = planes + g.planeOff[c] + (size_t)by * 8 * g.planeW[c] + (size_t)bx * 8;
        for (int r = 0; r < 8; ++r)
            for (int k = 0; k < 8; ++k)
                ws[r * 8 + k] = (int16_t)((int)src[(size_t)r * g.planeW[c] + k] - 128);
        fdctIslow(ws);
        const QuantEncode& q = T->eq[g.tbl[c]];
        int16_t* out = coefs + u * 64;
        for (int k = 0; k < 64; ++k)
            out[k] = quantize(ws[k], q, k);
    }
};

// jccoefct.c compress_data(): blocks of an edge MCU that lie outside the component get zero AC
// and the DC of the block before them in the MCU buffer - the last real block of their row for
// the right edge, and the last block of the previous row (itself possibly a right-edge dummy)
// for dummy rows at the bottom.
struct EncodeDummies
{
    EncodeGeom g;
    int16_t* coefs;

    CHESHIRE_JPEG_HD void operator()(size_t u) const
    {
        int c, bx, by, blk;
        uint32_t mcu;
        encUnitToBlock(g, (uint32_t)u, c, bx, by, mcu, blk);
        if (bx < g.blocksW[c] && by < g.blocksH[c])
            return;
        int rbx, rby;
        if (by < g.blocksH[c])
        {
            rbx = g.blocksW[c] - 1;
            rby = by;
        }
        else
        {
            const int lastInMcu = (bx / g.h[c]) * g.h[c] + g.h[c] - 1;
            rbx = lastInMcu < g.blocksW[c] - 1 ? lastInMcu : g.blocksW[c] - 1;
            rby = g.blocksH[c] - 1;
        }
        const int16_t dc = coefs[(size_t)encBlockToUnit(g, c, rbx, rby) * 64];
        int16_t* out = coefs + u * 64;
        out[0] = dc;
        for (int k = 1; k < 64; ++k)
            out[k] = 0;
    }
};

CHESHIRE_JPEG_HD inline void orWord(uint32_t* p, uint32_t v)
{
#if defined(CHESHIRE_JPEG_DEVICE_PASS)
    atomicOr(p, v);
#else
    *p |= v;
#endif
}

// Stream bit `pos` is bit (31 - pos % 32) of words[pos / 32]. len + (pos % 32) <= 58.
CHESHIRE_JPEG_HD inline void putBits(uint32_t* words, uint32_t pos, uint32_t val, int len)
{
    const uint32_t w = pos >> 5;
    const int off = (int)(pos & 31);
    const uint64_t v = (uint64_t)val << (64 - len - off);
    const uint32_t hi = (uint32_t)(v >> 32);
    const uint32_t lo = (uint32_t)v;
    if (hi)
        orWord(words + w, hi);
    if (lo)
        orWord(words + w + 1, lo);
}

// jchuff.c encode_one_block(). Returns the number of bits; writes them when EMIT.
template <bool EMIT>
CHESHIRE_JPEG_HD inline uint32_t encodeBlock(const int16_t* coef,
                                             int lastDc,
                                             const HuffEncode& dct,
                                             const HuffEncode& act,
                                             const Zigzag& zz,
                                             uint32_t* words,
                                             uint32_t pos)
{
    const uint32_t start = pos;
    const int diff = (int)coef[0] - lastDc;
    int nbits = bitLength((uint32_t)(diff < 0 ? -diff : diff));
    uint32_t extra = (uint32_t)(diff < 0 ? diff - 1 : diff) & ((1u << nbits) - 1u);
    int len = dct.size[nbits] + nbits;
    if (EMIT)
        putBits(words, pos, (dct.code[nbits] << nbits) | extra, len);
    pos += (uint32_t)len;

    int run = 0;
    for (int k = 1; k < 64; ++k)
    {
        const int v = coef[zz.natural[k]];
        if (v == 0)
        {
            ++run;
            continue;
        }
        while (run > 15)
        {
            if (EMIT)
                putBits(words, pos, act.code[0xF0], act.size[0xF0]);
            pos += act.size[0xF0];
            run -= 16;
        }
        nbits = bitLength((uint32_t)(v < 0 ? -v : v));
        extra = (uint32_t)(v < 0 ? v - 1 : v) & ((1u << nbits) - 1u);
        const int sym = (run << 4) + nbits;
        len = act.size[sym] + nbits;
        if (EMIT)
            putBits(words, pos, (act.code[sym] << nbits) | extra, len);
        pos += (uint32_t)len;
        run = 0;
    }
    if (run > 0)
    {
        if (EMIT)
            putBits(words, pos, act.code[0], act.size[0]);
        pos += act.size[0];
    }
    return pos - start;
}

// DC predictor of unit u: the previous block of the same component in the same restart segment.
CHESHIRE_JPEG_HD inline int encPrevDc(const EncodeGeom& g, const int16_t* coefs, uint32_t u, int c, uint32_t mcu, int blk)
{
    if (blk > g.firstBlk[c])
        return coefs[(size_t)(u - 1) * 64];
    const bool segStart = mcu == 0 || (g.restartInterval && mcu % g.restartInterval == 0);
    if (segStart)
        return 0;
    const uint32_t prev = (mcu - 1) * (uint32_t)g.bpm + (uint32_t)(g.firstBlk[c] + g.nBlk[c] - 1);
    return coefs[(size_t)prev * 64];
}

struct EncodeBitLengths
{
    EncodeGeom g;
    const Tables* T;
    const int16_t* coefs;
    uint32_t* lengths;

    CHESHIRE_JPEG_HD void operator()(size_t u) const
    {
        int c, bx, by, blk;
        uint32_t mcu;
        encUnitToBlock(g, (uint32_t)u, c, bx, by, mcu, blk);
        const int t = g.tbl[c];
        lengths[u] = encodeBlock<false>(coefs + u * 64, encPrevDc(g, coefs, (uint32_t)u, c, mcu, blk), T->edc[t], T->eac[t], T->zz, nullptr, 0);
    }
};

CHESHIRE_JPEG_HD inline uint32_t encSegFirstUnit(const EncodeGeom& g, uint32_t s)
{
    const uint32_t mcus = g.restartInterval ? g.restartInterval : g.totalMcus;
    const uint32_t m = s * mcus;
    return (m < g.totalMcus ? m : g.totalMcus) * (uint32_t)g.bpm;
}

struct EncodeSegBytes
{
    EncodeGeom g;
    const uint32_t* bitOffsets;  // exclusive scan of lengths, total at [totalUnits]
    uint32_t* segBytes;

    CHESHIRE_JPEG_HD void operator()(size_t s) const
    {
        const uint32_t bits = bitOffsets[encSegFirstUnit(g, (uint32_t)s + 1)] - bitOffsets[encSegFirstUnit(g, (uint32_t)s)];
        segBytes[s] = (bits + 7) / 8;
    }
};

struct EncodeEmit
{
    EncodeGeom g;
    const Tables* T;
    const int16_t* coefs;
    const uint32_t* bitOffsets;
    const uint32_t* segByteStart;
    uint32_t* words;

    CHESHIRE_JPEG_HD void operator()(size_t u) const
    {
        int c, bx, by, blk;
        uint32_t mcu;
        encUnitToBlock(g, (uint32_t)u, c, bx, by, mcu, blk);
        const uint32_t s = g.restartInterval ? mcu / g.restartInterval : 0;
        const uint32_t pos = segByteStart[s] * 8 + bitOffsets[u] - bitOffsets[encSegFirstUnit(g, s)];
        const int t = g.tbl[c];
        encodeBlock<true>(coefs + u * 64, encPrevDc(g, coefs, (uint32_t)u, c, mcu, blk), T->edc[t], T->eac[t], T->zz, words, pos);
    }
};

// jchuff.c flush_bits(): the last partial byte of a segment is filled with 1 bits.
struct EncodePad
{
    EncodeGeom g;
    const uint32_t* bitOffsets;
    const uint32_t* segByteStart;
    uint32_t* words;

    CHESHIRE_JPEG_HD void operator()(size_t s) const
    {
        const uint32_t bits = bitOffsets[encSegFirstUnit(g, (uint32_t)s + 1)] - bitOffsets[encSegFirstUnit(g, (uint32_t)s)];
        const int pad = (int)((8 - (bits & 7)) & 7);
        if (pad)
            putBits(words, segByteStart[s] * 8 + bits, (1u << pad) - 1u, pad);
    }
};

CHESHIRE_JPEG_HD inline uint8_t streamByte(const uint32_t* words, uint32_t b)
{
    return (uint8_t)(words[b >> 2] >> (24 - 8 * (b & 3)));
}

struct StuffCount
{
    const uint32_t* words;
    uint32_t totalBytes;
    uint32_t* counts;

    CHESHIRE_JPEG_HD void operator()(size_t j) const
    {
        const uint32_t b0 = (uint32_t)j * kStuffChunk;
        const uint32_t e = b0 + kStuffChunk < totalBytes ? b0 + kStuffChunk : totalBytes;
        uint32_t n = 0;
        for (uint32_t b = b0; b < e; ++b)
            n += streamByte(words, b) == 0xFF;
        counts[j] = n;
    }
};

// Segment containing byte b: last s with segByteStart[s] <= b.
CHESHIRE_JPEG_HD inline uint32_t segmentOfByte(const uint32_t* segByteStart, uint32_t nSeg, uint32_t b)
{
    uint32_t lo = 0, hi = nSeg;  // invariant: segByteStart[lo] <= b, answer in [lo, hi)
    while (hi - lo > 1)
    {
        const uint32_t mid = (lo + hi) / 2;
        if (segByteStart[mid] <= b)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

// Writes the entropy-coded data with a 0x00 after every 0xFF and an RSTn marker before every
// segment after the first.
struct StuffWrite
{
    EncodeGeom g;
    const uint32_t* words;
    uint32_t totalBytes;
    const uint32_t* ffBefore;  // exclusive scan of StuffCount
    const uint32_t* segByteStart;
    uint8_t* out;

    CHESHIRE_JPEG_HD void operator()(size_t j) const
    {
        const uint32_t b0 = (uint32_t)j * kStuffChunk;
        const uint32_t e = b0 + kStuffChunk < totalBytes ? b0 + kStuffChunk : totalBytes;
        uint32_t ff = ffBefore[j];
        uint32_t s = segmentOfByte(segByteStart, g.numSegments, b0);
        for (uint32_t b = b0; b < e; ++b)
        {
            while (s + 1 < g.numSegments && segByteStart[s + 1] <= b)
                ++s;
            const uint32_t o = b + ff + 2 * s;
            if (s > 0 && b == segByteStart[s])
            {
                out[o - 2] = 0xFF;
                out[o - 1] = (uint8_t)(0xD0 + ((s - 1) & 7));
            }
            const uint8_t v = streamByte(words, b);
            out[o] = v;
            if (v == 0xFF)
            {
                out[o + 1] = 0;
                ++ff;
            }
        }
    }
};

// ---------------------------------------------------------------------------------------------
// Chunked exclusive scan of uint32, in place, total written to data[n].

struct ScanChunkSum
{
    const uint32_t* data;
    uint32_t* chunkSums;
    size_t n;

    CHESHIRE_JPEG_HD void operator()(size_t j) const
    {
        const size_t e = (j + 1) * kScanChunk < n ? (j + 1) * kScanChunk : n;
        uint32_t s = 0;
        for (size_t i = j * kScanChunk; i < e; ++i)
            s += data[i];
        chunkSums[j] = s;
    }
};

struct ScanChunkSerial
{
    uint32_t* chunkSums;
    size_t nChunks;
    uint32_t* total;
    uint32_t* overflow;  // set if the total does not fit 32 bits

    CHESHIRE_JPEG_HD void operator()(size_t) const
    {
        uint64_t run = 0;
        for (size_t j = 0; j < nChunks; ++j)
        {
            const uint32_t v = chunkSums[j];
            chunkSums[j] = (uint32_t)run;
            run += v;
        }
        *total = (uint32_t)run;
        if (run > 0xFFFFFFFFull)
            *overflow = 1;
    }
};

struct ScanChunkApply
{
    uint32_t* data;
    const uint32_t* chunkBase;
    size_t n;

    CHESHIRE_JPEG_HD void operator()(size_t j) const
    {
        const size_t e = (j + 1) * kScanChunk < n ? (j + 1) * kScanChunk : n;
        uint32_t run = chunkBase[j];
        for (size_t i = j * kScanChunk; i < e; ++i)
        {
            const uint32_t v = data[i];
            data[i] = run;
            run += v;
        }
    }
};

}  // namespace jpeg
}  // namespace cheshire
