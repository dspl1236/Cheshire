// Cheshire: the smallest program that exercises the HIP PopSIFT the way AliceVision does.
//
// feature/sift/ImageDescriber_SIFT_popSIFT.cpp builds a popsift::Config, constructs
// PopSift(config, ExtractingMode, FloatImages), calls enqueue(width, height, floatPixels) and then
// job->get(). This does exactly that on a synthetic image, so a fault can be located without
// running the whole pipeline. Build with hip/port/popsift/CMakeLists.txt's smoke target.
#include <popsift/features.h>
#include <popsift/popsift.h>
#include <popsift/sift_conf.h>
#include <popsift/common/device_prop.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
#include <algorithm>
#include <utility>
#include <functional>

// CHESHIRE_SMOKE_IMAGE names a raw greyscale float buffer written by scratchpad/to_raw.py: two
// int32, width then height, then width*height float32 in [0,1]. Real photographs are what the
// descriptor stage comes apart on, and a synthetic image does not reproduce it.
static bool loadRaw(const char* path, std::vector<float>& pixels, int& width, int& height)
{
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) { std::printf("[smoke] cannot open %s%c", path, 10); return false; }
    int hdr[2] = {0, 0};
    if (std::fread(hdr, sizeof(int), 2, f) != 2) { std::fclose(f); return false; }
    width = hdr[0];
    height = hdr[1];
    pixels.resize((size_t)width * height);
    const size_t want = pixels.size();
    const size_t got = std::fread(pixels.data(), sizeof(float), want, f);
    std::fclose(f);
    if (got != want) { std::printf("[smoke] short read: %zu of %zu%c", got, want, 10); return false; }
    double lo = pixels[0], hi = pixels[0], sum = 0.0;
    for (float v : pixels) { if (v < lo) lo = v; if (v > hi) hi = v; sum += v; }
    std::printf("[smoke] loaded %s: %d x %d, min %.4f max %.4f mean %.4f%c",
                path, width, height, lo, hi, sum / want, 10);
    return true;
}

int main(int argc, char** argv)
{
    int width = (argc > 1) ? std::atoi(argv[1]) : 4032;
    int height = (argc > 2) ? std::atoi(argv[2]) : 3024;
    std::vector<float> pixels;
    const char* imagePath = getenv("CHESHIRE_SMOKE_IMAGE");
    if (imagePath != nullptr && imagePath[0] != 0)
    {
        if (!loadRaw(imagePath, pixels, width, height)) return 1;
    }
    std::printf("[smoke] image %d x %d\n", width, height);

    std::printf("[smoke] querying the device\n");
    std::fflush(stdout);
    popsift::cuda::device_prop_t prop;
    prop.set(0, false);

    std::printf("[smoke] building the config\n");
    std::fflush(stdout);
    popsift::Config config;
    config.setOctaves(6);
    config.setLevels(3);
    config.setDownsampling(0);
    config.setThreshold(0.04f / 3.0f);
    config.setEdgeLimit(10.0f);
    config.setNormalizationMultiplier(9);
    config.setNormMode(getenv("CHESHIRE_SMOKE_ROOTSIFT") ? popsift::Config::RootSift : popsift::Config::Classic);
    config.setFilterSorting(popsift::Config::LargestScaleFirst);
    if (argc > 4) { config.setDescMode(argv[4]); std::printf("[smoke] desc mode %s%c", argv[4], 10); }
    // Capped by default, at AliceVision's "normal" preset. Leaving it uncapped on a full-resolution
    // photograph took the host machine down hard on 2026-09-18: a synthetic image at the same size
    // is smooth and yields few extrema, real photographic content does not, and nothing in the
    // describer's path ever runs uncapped. Pass 0 explicitly to ask for no cap.
    const int cap = (argc > 3) ? std::atoi(argv[3]) : 20000;
    if (cap > 0)
    {
        config.setFilterMaxExtrema(cap);
        std::printf("[smoke] filter max extrema %d%c", cap, 10);
    }
    else
        std::printf("[smoke] NO extrema cap; on a large real image this can hang the device%c", 10);
    if (getenv("CHESHIRE_POPSIFT_DUMP"))
        config.setLogMode(popsift::Config::All);  // writes the pyramid to disk

    // a synthetic image with enough structure to produce keypoints, unless one was loaded
    if (pixels.empty())
    {
        pixels.resize((size_t)width * height);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                pixels[(size_t)y * width + x] =
                  0.5f + 0.25f * std::sin(x * 0.05f) * std::cos(y * 0.05f) + 0.1f * std::sin((x + y) * 0.31f);
    }

    std::printf("[smoke] constructing PopSift\n");
    std::fflush(stdout);
    std::unique_ptr<PopSift> sift(new PopSift(config, popsift::Config::ExtractingMode, PopSift::FloatImages));

    std::printf("[smoke] enqueue\n");
    std::fflush(stdout);
    std::unique_ptr<SiftJob> job(sift->enqueue(width, height, pixels.data()));
    if (!job)
    {
        std::printf("[smoke] enqueue returned nothing\n");
        return 1;
    }

    std::printf("[smoke] waiting for features\n");
    std::fflush(stdout);
    std::unique_ptr<popsift::Features> features(job->get());
    if (!features)
    {
        std::printf("[smoke] no features returned\n");
        return 1;
    }
    std::printf("[smoke] %d features, %d descriptors\n", features->getFeatureCount(), features->getDescriptorCount());

    // what AliceVision will see: each float cast to unsigned char. A healthy SIFT descriptor has
    // few empty bins; mostly-zero descriptors match badly however many of them there are.
    {
        long long total = 0, zeros = 0, sum = 0, mx = 0;
        long long descs = 0, dead = 0, oris = 0, feats = 0;
        // how many of the 128 bins are non-empty, in buckets of 16: an evenly sparse descriptor and
        // a mix of healthy and dead ones give the same aggregate but different shapes here
        long long hist[9] = {0};
        for (const auto& feat : *features)
        {
            ++feats;
            oris += feat.num_ori;
            for (int o = 0; o < feat.num_ori; ++o)
            {
                const popsift::Descriptor* d = feat.desc[o];
                if (d == nullptr) continue;
                ++descs;
                int nonzero = 0;
                for (int k = 0; k < 128; ++k)
                {
                    const int v = (int)(unsigned char)d->features[k];
                    ++total; sum += v; if (v == 0) ++zeros; else ++nonzero; if (v > mx) mx = v;
                }
                if (nonzero == 0) ++dead;
                ++hist[nonzero / 16];
            }
        }
        // If the extrema counter over-reserves slots, the slots nothing wrote keep a zero offset
        // and every one of them aliases extremum 0. That shows up as a feature count far larger
        // than the number of distinct keypoint positions.
        {
            struct P { float x, y, s; bool operator<(const P& o) const {
                return x != o.x ? x < o.x : (y != o.y ? y < o.y : s < o.s); }
                bool operator==(const P& o) const { return x == o.x && y == o.y && s == o.s; } };
            std::vector<P> pos;
            pos.reserve(features->getFeatureCount());
            for (const auto& feat : *features) pos.push_back(P{feat.xpos, feat.ypos, feat.sigma});
            std::sort(pos.begin(), pos.end());
            const size_t total = pos.size();
            // the run lengths tell the two candidate faults apart: one position repeated tens of
            // thousands of times means unwritten slots all aliasing extremum 0, while every
            // position repeated about warp-many times means detection firing on a whole warp
            std::vector<std::pair<size_t, P>> runs;
            for (size_t i = 0; i < total;)
            {
                size_t j = i;
                while (j < total && pos[j] == pos[i]) ++j;
                runs.push_back({j - i, pos[i]});
                i = j;
            }
            std::sort(runs.begin(), runs.end(),
                      [](const std::pair<size_t, P>& a, const std::pair<size_t, P>& b) { return a.first > b.first; });
            for (size_t i = 0; i < runs.size() && i < 6; ++i)
                std::printf("[smoke]   run %zu at x %.3f y %.3f sigma %.5f%c",
                            runs[i].first, runs[i].second.x, runs[i].second.y, runs[i].second.s, 10);
            std::printf("[smoke] %zu features at %zu distinct positions (%.1f copies each)%c",
                        total, runs.size(), runs.empty() ? 0.0 : (double)total / runs.size(), 10);
            std::printf("[smoke] largest repeat counts:");
            for (size_t i = 0; i < runs.size() && i < 8; ++i) std::printf(" %zu", runs[i].first);
            std::printf(" | median %zu%c", runs.empty() ? 0 : runs[runs.size() / 2].first, 10);
        }
        // ext_desc_loop_sub returns before touching the descriptor when DESC_MAGNIFY * sigma is
        // zero, so a keypoint with no scale leaves an all-zero descriptor behind. Sigma also sizes
        // the orientation window, so a bad one shows up as too many orientations per feature.
        {
            long long zeroSigma = 0, tiny = 0, n = 0;
            double lo = 1e30, hi = -1e30, acc = 0.0;
            for (const auto& feat : *features)
            {
                const float s = feat.sigma;
                ++n;
                if (s == 0.0f) ++zeroSigma;
                if (s < 1.0e-3f) ++tiny;
                if (s < lo) lo = s;
                if (s > hi) hi = s;
                acc += s;
            }
            std::printf("[smoke] sigma: min %.5f max %.5f mean %.5f | exactly zero %lld of %lld (%.1f%%), under 1e-3 %lld%c",
                        lo, hi, acc / (n ? n : 1), zeroSigma, n, 100.0 * zeroSigma / (n ? n : 1), tiny, 10);
        }
        // The bytes are a truncating cast of these floats, so a zero byte can mean the float was
        // zero, or NaN, or 255 and over. Those have completely different causes; separate them.
        {
            long long over = 0, nan = 0, inf = 0, neg = 0, n = 0;
            double lo = 1e30, hi = -1e30, acc = 0.0;
            for (const auto& feat : *features)
                for (int o = 0; o < feat.num_ori; ++o)
                {
                    const popsift::Descriptor* d = feat.desc[o];
                    if (d == nullptr) continue;
                    for (int k = 0; k < 128; ++k)
                    {
                        const float f = d->features[k];
                        ++n;
                        if (f != f) { ++nan; continue; }
                        if (f > 3.0e38f) { ++inf; continue; }
                        if (f < 0.0f) ++neg;
                        if (f >= 255.0f) ++over;
                        if (f < lo) lo = f;
                        if (f > hi) hi = f;
                        acc += f;
                    }
                }
            std::printf("[smoke] raw floats: min %.3f max %.3f mean %.3f | >=255 %lld (%.1f%%), NaN %lld, inf %lld, negative %lld%c",
                        lo, hi, acc / (n ? n : 1), over, 100.0 * over / (n ? n : 1), nan, inf, neg, 10);
        }
        if (total > 0)
        {
            std::printf("[smoke] descriptor bytes: mean %.1f, zeros %.1f%%, max %lld%c",
                        (double)sum / total, 100.0 * zeros / total, mx, 10);
            std::printf("[smoke] %lld features, %lld orientations (%.2f per feature), %lld descriptors, %lld all-zero%c",
                        feats, oris, (double)oris / (feats ? feats : 1), descs, dead, 10);
            std::printf("[smoke] non-empty bins per descriptor, buckets of 16:");
            for (int b = 0; b < 9; ++b) std::printf(" %lld", hist[b]);
            std::printf("%c", 10);
        }
    }

    std::printf("[smoke] uninit\n");
    std::fflush(stdout);
    sift->uninit();
    std::printf("[smoke] done\n");
    return 0;
}
