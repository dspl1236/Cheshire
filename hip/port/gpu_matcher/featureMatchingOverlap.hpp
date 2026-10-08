// Cheshire (step 10a): FeatureMatching's GPU search and its geometric filter at the same time.
//
// Upstream searches every pair of the chunk first ("Regions Matching"), then filters the pairs' putative matches
// geometrically on every core. With the search on the GPU, one thread drives it while the other cores wait: on the
// False Door (884 views, 45 chunks at 2048 iterations) a chunk was 2 s of loading, 2.2-3.1 s of search and 3.2-4.2 s of
// filter. Here the thread that drives the GPU searches the pairs in upstream's order, first view by first view, and
// the other cores filter each pair as soon as its putative matches exist, by upstream's own steps:
// filterMatchesByMin2DMotion on the pair, then GeometricFilter.hpp's robustModelEstimationOne, which is the body of
// robustModelEstimation's loop. The searching thread joins the filter when its searches are done.
//
// Exact where upstream is deterministic. A pair's search does not depend on the other pairs. Its filter reads the
// caller's generator only to copy it - the pinhole AC-RANSAC filters (7-point, and 10-point with distortion) take it by
// value - as upstream's loop does after a search that never touched it: the GPU matcher's Build ignores the generator,
// the kd-tree's does not, so this runs only when every view of the chunk takes the GPU matcher
// (matching::cheshireGpuSearches, next to createRegionsMatcher). The spherical and LO-RANSAC filters share the
// generator by reference across upstream's threads already, so their pairs vary run to run either way. The maps come
// out with the same keys and values as upstream's, so the logs and the saved matches are the same bytes.
// CHESHIRE_FM_OVERLAP=0: one after the other, as upstream.
#pragma once

#include <aliceVision/alicevision_omp.hpp>
#include <aliceVision/feature/RegionsPerView.hpp>
#include <aliceVision/matching/RegionsMatcher.hpp>
#include <aliceVision/matching/matchesFiltering.hpp>
#include <aliceVision/matching/matcherType.hpp>
#include <aliceVision/matchingImageCollection/GeometricFilter.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/system/ProgressDisplay.hpp>
#include <aliceVision/types.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <random>
#include <set>
#include <vector>

namespace aliceVision {
namespace matching {

/// True when createRegionsMatcher takes the GPU matcher for these regions (defined beside it, in RegionsMatcher.cpp)
bool cheshireGpuSearches(const feature::Regions& regions, EMatcherType matcherType);

}  // namespace matching

namespace matchingImageCollection {

/// CHESHIRE_FM_OVERLAP (on by default), and every view of the pairs with regions of descType takes the GPU matcher
inline bool cheshireOverlapPossible(const feature::RegionsPerView& regionsPerView,
                                    const PairSet& pairs,
                                    feature::EImageDescriberType descType,
                                    matching::EMatcherType matcherType)
{
    if (!::cheshire::env::flag("CHESHIRE_FM_OVERLAP", true))
        return false;
    std::set<IndexT> views;
    for (const Pair& p : pairs)
    {
        views.insert(p.first);
        views.insert(p.second);
    }
    for (const IndexT v : views)
    {
        const feature::Regions& regions = regionsPerView.getRegions(v, descType);
        if (regions.RegionCount() != 0 && !matching::cheshireGpuSearches(regions, matcherType))
            return false;  // upstream's search builds no matcher for a view without regions; any other view must be ours
    }
    return true;
}

/**
 * ImageCollectionMatcher_generic::Match (without cross matching) on one thread, then filterMatchesByMin2DMotion and
 * robustModelEstimation's loop on the others, pair by pair as the searches finish. out_putatives and out_geometric
 * hold what upstream's main_featureMatching has in mapPutativesMatches after filterMatchesByMin2DMotion and in
 * geometricMatches after robustModelEstimation.
 */
template<typename GeometryFunctor>
void cheshireSearchAndFilter(matching::PairwiseMatches& out_putatives,
                             matching::PairwiseMatches& out_geometric,
                             const sfmData::SfMData& sfmData,
                             const feature::RegionsPerView& regionsPerView,
                             const PairSet& pairs,
                             feature::EImageDescriberType descType,
                             matching::EMatcherType matcherType,
                             float distRatio,
                             double minRequired2DMotion,
                             const GeometryFunctor& functor,
                             std::mt19937& randomNumberGenerator,
                             bool guidedMatching,
                             double distanceRatio = 0.6)
{
    struct Slot
    {
        Pair pair;
        matching::MatchesPerDescType putatives;  // no entry: the search found nothing, and upstream adds no pair
        matching::MatchesPerDescType inliers;
        bool kept = false;
    };
    // PairSet orders the pairs by first view, then second: the order of ImageCollectionMatcher_generic's loops
    std::vector<Slot> slots;
    std::vector<std::size_t> groupEnds;  // one past each first view's last slot
    slots.reserve(pairs.size());
    for (const Pair& p : pairs)
    {
        if (!slots.empty() && slots.back().pair.first != p.first)
            groupEnds.push_back(slots.size());
        slots.push_back(Slot{p, {}, {}, false});
    }
    if (!slots.empty())
        groupEnds.push_back(slots.size());

    std::mutex mutex;
    std::condition_variable published;
    std::size_t ready = 0;  // slots [0, ready) searched, under mutex
    std::atomic<std::size_t> next{0};
    auto progressDisplay = system::createConsoleProgressDisplay(pairs.size());
    const auto t0 = std::chrono::steady_clock::now();
    double searchSeconds = 0.0;

#pragma omp parallel
    {
        if (omp_get_thread_num() == 0)
        {
            std::size_t begin = 0;
            for (const std::size_t end : groupEnds)
            {
                const feature::Regions& regionsI = regionsPerView.getRegions(slots[begin].pair.first, descType);
                if (regionsI.RegionCount() == 0)
                    progressDisplay += static_cast<unsigned long>(end - begin);
                else
                {
                    matching::RegionsDatabaseMatcher matcher(randomNumberGenerator, matcherType, regionsI);
                    for (std::size_t k = begin; k < end; ++k)
                    {
                        const feature::Regions& regionsJ = regionsPerView.getRegions(slots[k].pair.second, descType);
                        if (regionsJ.RegionCount() != 0 && regionsI.Type_id() == regionsJ.Type_id())
                        {
                            matching::IndMatches putatives;
                            matcher.Match(distRatio, regionsJ, putatives);
                            if (!putatives.empty())
                                slots[k].putatives.emplace(descType, std::move(putatives));
                        }
                        ++progressDisplay;
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    ready = end;
                }
                published.notify_all();
                begin = end;
            }
            searchSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        // every thread, the searching one once its searches are done: the pairs in order, each when it is searched
        for (;;)
        {
            const std::size_t k = next.fetch_add(1);
            if (k >= slots.size())
                break;
            {
                std::unique_lock<std::mutex> lock(mutex);
                published.wait(lock, [&] { return ready > k; });
            }
            Slot& slot = slots[k];
            if (slot.putatives.empty())
                continue;
            matching::PairwiseMatches one;  // upstream's filter, on this pair alone
            one.emplace(slot.pair, std::move(slot.putatives));
            matching::filterMatchesByMin2DMotion(one, regionsPerView, minRequired2DMotion);
            slot.putatives = std::move(one.begin()->second);
            slot.kept = robustModelEstimationOne(
              slot.inliers, sfmData, regionsPerView, functor, slot.pair, slot.putatives, randomNumberGenerator, guidedMatching, distanceRatio);
        }
    }

    out_putatives.clear();
    out_geometric.clear();
    for (Slot& slot : slots)
    {
        if (slot.putatives.empty())
            continue;
        if (slot.kept)
            out_geometric.emplace(slot.pair, std::move(slot.inliers));
        out_putatives.emplace(slot.pair, std::move(slot.putatives));
    }
    robustEstimation::cheshireAcrBoundReport();          // as robustModelEstimation reports (steps 8d, 8e)
    multiview::relativePose::cheshireEpipolarReport();
    multiview::relativePose::Fundamental7PSolver::cheshireSolve4Report();  // step 13a
    const double totalSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ALICEVISION_LOG_INFO("cheshire: GPU search and geometric filter overlapped (CHESHIRE_FM_OVERLAP=0 for one after the other): "
                         << slots.size() << " pairs, searched in " << searchSeconds << " s, filtered by " << totalSeconds << " s");
}

}  // namespace matchingImageCollection
}  // namespace aliceVision
