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

int main(int argc, char** argv)
{
    const int width = (argc > 1) ? std::atoi(argv[1]) : 4032;
    const int height = (argc > 2) ? std::atoi(argv[2]) : 3024;
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
    const int cap = (argc > 3) ? std::atoi(argv[3]) : 0;
    if (cap > 0)
    {
        config.setFilterMaxExtrema(cap);
        std::printf("[smoke] filter max extrema %d%c", cap, 10);
    }
    if (getenv("CHESHIRE_POPSIFT_DUMP"))
        config.setLogMode(popsift::Config::All);  // writes the pyramid to disk

    // a synthetic image with enough structure to produce keypoints
    std::vector<float> pixels((size_t)width * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            pixels[(size_t)y * width + x] =
              0.5f + 0.25f * std::sin(x * 0.05f) * std::cos(y * 0.05f) + 0.1f * std::sin((x + y) * 0.31f);

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
        for (const auto& feat : *features)
            for (int o = 0; o < feat.num_ori; ++o)
            {
                const popsift::Descriptor* d = feat.desc[o];
                if (d == nullptr) continue;
                for (int k = 0; k < 128; ++k)
                {
                    const int v = (int)(unsigned char)d->features[k];
                    ++total; sum += v; if (v == 0) ++zeros; if (v > mx) mx = v;
                }
            }
        if (total > 0)
            std::printf("[smoke] descriptor bytes: mean %.1f, zeros %.1f%%, max %lld%c",
                        (double)sum / total, 100.0 * zeros / total, mx, 10);
    }

    std::printf("[smoke] uninit\n");
    std::fflush(stdout);
    sift->uninit();
    std::printf("[smoke] done\n");
    return 0;
}
