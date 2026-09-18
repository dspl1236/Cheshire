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
    config.setNormMode(popsift::Config::Classic);
    config.setFilterSorting(popsift::Config::LargestScaleFirst);

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

    std::printf("[smoke] uninit\n");
    std::fflush(stdout);
    sift->uninit();
    std::printf("[smoke] done\n");
    return 0;
}
