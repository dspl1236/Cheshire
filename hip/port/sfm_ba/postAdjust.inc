// This file is part of the Cheshire project (MPL-2.0), a patch set over AliceVision.
// Included at the end of sfm/pipeline/sequential/ReconstructionEngine_sequentialSfM.cpp by
// scripts/apply_hip_patch.py (step 5s).
//
// The passes after every bundle-adjustment iteration, proportional to what the solve touched.
// Upstream follows each adjust() with removeOutliers() - the pixel-residual test over every
// observation of every landmark, then the angle test over every landmark - and with
// eraseUnstablePosesAndObservations(), which recounts every observation per pose. On the 884-view
// False Door that is 64 passes over a scene of up to 1.35 million landmarks, for solves that under
// the local strategy moved about a hundred poses. Measured, those passes are about 40 s of the
// whole run at that size and the restricted ones are no cheaper (the frontier is still 25-60 % of
// the scene), so this is OPT-IN: CHESHIRE_SFM_LOCAL_PASSES=1. Its case is thousands of views,
// where a full pass grows with the scene and a restricted one with the frontier (docs/04).
//
// The tests are functions of (pose, intrinsics, landmark position, observation) only, so an
// observation whose inputs did not change since it was last tested keeps its verdict. What can
// have changed after a solve: the landmarks observed by a REFINED pose (that covers every REFINED
// landmark, since a landmark is refined only when a refined camera sees it; every landmark and
// observation triangulation added since the last pass, since they hang off the new, refined,
// views; and the case the local strategy ignores but whose refined observer still moved), and
// the landmarks observed through a REFINED intrinsic, since its parameters apply to every view that
// uses it, active or not. The candidates are found from those views through the tracks-per-view
// index, never by walking the landmarks. A pose can only have become
// unstable by losing observations, so only the poses of views that lost one here, and the poses
// resected in this group, are recounted - through the same index. When a pose is erased the
// upstream full pass runs for the rest of that iteration (rare, and the full pass is exact).
// Without the local strategy every pose is refined and the upstream passes run unchanged.
// CHESHIRE_SFM_PROFILE=1 prints one line per iteration.
//
// Step 7l (0.3.7): the full passes themselves - upstream's pixel-residual filter and unstable-pose count - run on
// every core with each view's pose read once; the verdicts and counts are upstream's, applied in its order.

#include <aliceVision/sfm/sfmFilters.hpp>
#include <aliceVision/track/Track.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace aliceVision {
namespace sfm {
namespace cheshire {

inline bool localPassesEnabled()
{
    static const bool on = ::cheshire::env::flag("CHESHIRE_SFM_LOCAL_PASSES");
    return on;
}

inline bool sfmProfileEnabled()
{
    static const bool on = ::cheshire::env::flag("CHESHIRE_SFM_PROFILE");
    return on;
}

// The intrinsics the solve may have moved: refined by the local strategy, not locked, with options
// that refine intrinsics at all. Under upstream's strategy an intrinsic stays refined until its focal
// length has been stable over a window of 25 posed views, so an intrinsic with fewer views than that
// is refined for the whole reconstruction (the False Door has four such, of 1 to 27 views, beside the
// 820-view one that settles at 113 poses). A refined intrinsic changes every residual through its
// views, active or not, so those views join the candidate sources - a full pass is not needed.
inline void refinedIntrinsics(const sfmData::SfMData& sfmData, BundleAdjustment::ERefineOptions refineOptions, std::set<IndexT>& out)
{
    out.clear();
    const bool refineIntrinsics = (refineOptions & BundleAdjustment::REFINE_INTRINSICS_FOCAL) ||
                                  (refineOptions & BundleAdjustment::REFINE_INTRINSICS_OPTICALOFFSET_ALWAYS) ||
                                  (refineOptions & BundleAdjustment::REFINE_INTRINSICS_OPTICALOFFSET_IF_ENOUGH_DATA) ||
                                  (refineOptions & BundleAdjustment::REFINE_INTRINSICS_DISTORTION);
    if (!refineIntrinsics)
        return;
    for (const auto& [intrinsicId, intrinsic] : sfmData.getIntrinsics())
        if (intrinsic->getState() == EEstimatorParameterState::REFINED && !intrinsic->isLocked())
            out.insert(intrinsicId);
}

struct CandidateSources
{
    std::size_t views = 0;
    std::size_t refinedPoses = 0;
    std::size_t refinedIntrinsics = 0;
};

// The landmarks observed by a refined pose or through a refined intrinsic, from the tracks of those
// views.
inline void candidateLandmarks(const sfmData::SfMData& sfmData,
                               const track::TracksPerView& tracksPerView,
                               BundleAdjustment::ERefineOptions refineOptions,
                               std::vector<IndexT>& out,
                               CandidateSources& sources)
{
    std::set<IndexT> refinedPoses;
    for (const auto& [poseId, pose] : sfmData.getPoses().valueRange())
        if (pose.getState() == EEstimatorParameterState::REFINED)
            refinedPoses.insert(poseId);
    std::set<IndexT> movingIntrinsics;
    refinedIntrinsics(sfmData, refineOptions, movingIntrinsics);
    sources = CandidateSources{};
    sources.refinedPoses = refinedPoses.size();
    sources.refinedIntrinsics = movingIntrinsics.size();
    out.clear();
    for (const auto& [viewId, view] : sfmData.getViews())
    {
        if (!sfmData.isPoseAndIntrinsicDefined(*view))
            continue;
        if (refinedPoses.count(view->getPoseId()) == 0 && movingIntrinsics.count(view->getIntrinsicId()) == 0)
            continue;
        ++sources.views;
        const auto it = tracksPerView.find(viewId);
        if (it == tracksPerView.end())
            continue;
        for (const std::size_t trackId : it->second)
            out.push_back(static_cast<IndexT>(trackId));
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    const sfmData::Landmarks& landmarks = sfmData.getLandmarks();
    out.erase(std::remove_if(out.begin(), out.end(), [&](IndexT id) { return landmarks.find(id) == landmarks.end(); }), out.end());
}

// removeOutliersWithPixelResidualError over the given landmarks (sfmFilters.cpp:19-61, the same
// test in the same order), recording the views that lost observations.
inline IndexT removeOutliersWithPixelResidualErrorOn(sfmData::SfMData& sfmData,
                                                     EFeatureConstraint featureConstraint,
                                                     const double dThresholdPixel,
                                                     const unsigned int minTrackLength,
                                                     const std::vector<IndexT>& ids,
                                                     std::set<IndexT>& viewsWithErasures)
{
    IndexT outlierCount = 0;
    for (const IndexT landmarkId : ids)
    {
        auto iterTracks = sfmData.getLandmarks().find(landmarkId);
        if (iterTracks == sfmData.getLandmarks().end())
            continue;
        sfmData::Observations& observations = iterTracks->second.getObservations();
        sfmData::Observations::iterator itObs = observations.begin();
        while (itObs != observations.end())
        {
            const sfmData::View* view = sfmData.getViews().at(itObs->first).get();
            const geometry::Pose3 pose = sfmData.getPose(*view).getTransform();
            const camera::IntrinsicBase* intrinsic = sfmData.getIntrinsics().at(view->getIntrinsicId()).get();

            Vec2 residual = intrinsic->residual(pose, iterTracks->second.getX().homogeneous(), itObs->second.getCoordinates());
            if (featureConstraint == EFeatureConstraint::SCALE && itObs->second.getScale() > 0.0)
            {
                residual /= itObs->second.getScale();
            }

            if ((pose.depth(iterTracks->second.getX()) < 0) || (residual.norm() > dThresholdPixel))
            {
                ++outlierCount;
                viewsWithErasures.insert(itObs->first);
                itObs = observations.erase(itObs);
            }
            else
                ++itObs;
        }

        if (observations.empty() || observations.size() < minTrackLength)
        {
            for (const auto& [viewId, obs] : observations)
                viewsWithErasures.insert(viewId);
            sfmData.getLandmarks().erase(iterTracks);
        }
    }
    return outlierCount;
}

// removeOutliersWithAngleError over the given landmarks (sfmFilters.cpp:63-159, the same greedy
// test), recording the views of the landmarks it erases.
inline IndexT removeOutliersWithAngleErrorOn(sfmData::SfMData& sfmData, const double dMinAcceptedAngle, const std::vector<IndexT>& ids, std::set<IndexT>& viewsWithErasures)
{
    const double dMaxAcceptedCosAngle = std::cos(degreeToRadian(dMinAcceptedAngle));
    std::vector<IndexT> toErase;
#pragma omp parallel for
    for (int landmarkIndex = 0; landmarkIndex < static_cast<int>(ids.size()); ++landmarkIndex)
    {
        const auto lmIt = sfmData.getLandmarks().find(ids[landmarkIndex]);
        if (lmIt == sfmData.getLandmarks().end())
            continue;
        const sfmData::Landmark& landmark = lmIt->second;
        if (landmark.isParallaxRobust())
        {
            continue;
        }
        const sfmData::Observations& observations = landmark.getObservations();
        Mat3X viewDirections(3, observations.size());
        Mat3X::Index i;
        sfmData::Observations::const_iterator itObs;
        double dGreedyCos = 1.1;
        Mat3X::Index greedyI = 0;
        for (itObs = observations.begin(), i = 0; itObs != observations.end(); ++itObs, ++i)
        {
            const sfmData::View* view = sfmData.getViews().at(itObs->first).get();
            const geometry::Pose3 pose = sfmData.getPose(*view).getTransform();
            const camera::IntrinsicBase* intrinsic = sfmData.getIntrinsics().at(view->getIntrinsicId()).get();
            viewDirections.col(i) = camera::applyIntrinsicExtrinsic(pose, intrinsic, itObs->second.getCoordinates());
            double dCosAngle = viewDirections.col(i).transpose() * viewDirections.col(greedyI);
            if (dCosAngle < dMaxAcceptedCosAngle)
            {
                break;
            }
            else if (dCosAngle < dGreedyCos)
            {
                dGreedyCos = dCosAngle;
                greedyI = i;
            }
        }
        if (itObs != observations.end())
        {
            continue;
        }
        for (i = viewDirections.cols() - 1; i > 0; i -= 1)
        {
            const double dMinCosAngle = (viewDirections.col(i).transpose() * viewDirections.leftCols(i)).minCoeff();
            if (dMinCosAngle < dMaxAcceptedCosAngle)
            {
                break;
            }
        }
        if (i == 0)
        {
#pragma omp critical
            toErase.push_back(ids[landmarkIndex]);
        }
    }
    std::sort(toErase.begin(), toErase.end());  // the parallel loop collected them in finishing order
    for (const IndexT key : toErase)
    {
        const auto lmIt = sfmData.getLandmarks().find(key);
        if (lmIt == sfmData.getLandmarks().end())
            continue;
        for (const auto& [viewId, obs] : lmIt->second.getObservations())
            viewsWithErasures.insert(viewId);
        sfmData.getLandmarks().erase(lmIt);
    }
    return static_cast<IndexT>(toErase.size());
}

// The number of observations that hang off a pose's views, through the tracks-per-view index -
// what eraseUnstablePoses counts by walking every landmark (sfmFilters.cpp:161-232).
inline IndexT poseObservationCount(const sfmData::SfMData& sfmData, const track::TracksPerView& tracksPerView, IndexT poseId)
{
    IndexT count = 0;
    const sfmData::Landmarks& landmarks = sfmData.getLandmarks();
    for (const auto& [viewId, view] : sfmData.getViews())
    {
        if (view->getPoseId() != poseId)
            continue;
        const auto it = tracksPerView.find(viewId);
        if (it == tracksPerView.end())
            continue;
        for (const std::size_t trackId : it->second)
        {
            const auto lm = landmarks.find(static_cast<IndexT>(trackId));
            if (lm != landmarks.end() && lm->second.getObservations().find(viewId) != lm->second.getObservations().end())
                ++count;
        }
    }
    return count;
}

// ---- step 7l (0.3.7): upstream's full passes on every core ----------------------------------------
// Without the local passes (the default) every adjust is followed by upstream's full passes. The pixel-residual filter
// and the count behind eraseUnstablePoses walk every observation on one thread, with map lookups and a pose copy per
// observation: about 5 M observations on the False Door, 64 times, about 40 s. Here each view's pose and intrinsic
// are read once, the verdicts and counts are made on every core, and the erasures are applied in the order upstream
// makes them. The same tests on the same values, so the same scene.

// a posed view's pose and intrinsic, as upstream reads them for each of its observations
struct ViewPose
{
    geometry::Pose3 pose;
    const camera::IntrinsicBase* intrinsic = nullptr;
};

inline std::unordered_map<IndexT, ViewPose> viewPoses(const sfmData::SfMData& sfmData)
{
    std::unordered_map<IndexT, ViewPose> out;
    out.reserve(sfmData.getViews().size());
    for (const auto& [viewId, view] : sfmData.getViews())
    {
        if (!sfmData.existsPose(*view))
            continue;
        const auto intrinsic = sfmData.getIntrinsics().find(view->getIntrinsicId());
        if (intrinsic == sfmData.getIntrinsics().end())
            continue;
        ViewPose v;
        v.pose = sfmData.getPose(*view).getTransform();
        v.intrinsic = intrinsic->second.get();
        out.emplace(viewId, v);
    }
    return out;
}

// removeOutliersWithPixelResidualError (sfmFilters.cpp:19-61): each observation's verdict on every core, then the
// erasures in landmark order, and the landmarks left short
inline IndexT removeOutliersWithPixelResidualErrorAll(sfmData::SfMData& sfmData,
                                                     EFeatureConstraint featureConstraint,
                                                     const double dThresholdPixel,
                                                     const unsigned int minTrackLength)
{
    const std::unordered_map<IndexT, ViewPose> views = viewPoses(sfmData);
    sfmData::Landmarks& landmarks = sfmData.getLandmarks();
    const std::size_t n = landmarks.size();
    std::vector<sfmData::Landmarks::iterator> its;
    its.reserve(n);
    std::vector<std::size_t> first(n + 1, 0);
    for (auto it = landmarks.begin(); it != landmarks.end(); ++it)
    {
        first[its.size() + 1] = first[its.size()] + it->second.getObservations().size();
        its.push_back(it);
    }
    std::vector<unsigned char> outlier(first[n], 0), touched(n, 0);
#pragma omp parallel for schedule(dynamic, 4096)
    for (std::ptrdiff_t li = 0; li < static_cast<std::ptrdiff_t>(n); ++li)
    {
        const sfmData::Landmark& landmark = its[static_cast<std::size_t>(li)]->second;
        std::size_t k = first[static_cast<std::size_t>(li)];
        for (const auto& [viewId, obs] : landmark.getObservations())
        {
            const ViewPose& v = views.at(viewId);
            Vec2 residual = v.intrinsic->residual(v.pose, landmark.getX().homogeneous(), obs.getCoordinates());
            if (featureConstraint == EFeatureConstraint::SCALE && obs.getScale() > 0.0)
            {
                // Apply the scale of the feature to get a residual value
                // relative to the feature precision.
                residual /= obs.getScale();
            }
            if ((v.pose.depth(landmark.getX()) < 0) || (residual.norm() > dThresholdPixel))
            {
                outlier[k] = 1;
                touched[static_cast<std::size_t>(li)] = 1;
            }
            ++k;
        }
    }
    IndexT outlierCount = 0;
    for (std::size_t li = 0; li < n; ++li)
    {
        sfmData::Observations& observations = its[li]->second.getObservations();
        if (touched[li])
        {
            std::size_t k = first[li];
            for (auto itObs = observations.begin(); itObs != observations.end(); ++k)
            {
                if (outlier[k])
                {
                    ++outlierCount;
                    itObs = observations.erase(itObs);
                }
                else
                    ++itObs;
            }
        }
        if (observations.empty() || observations.size() < minTrackLength)
            landmarks.erase(its[li]);
    }
    return outlierCount;
}

// eraseUnstablePoses (sfmFilters.cpp:161-232): the observations per pose counted on every core (integer counts, the
// same in any order), then the same removal
inline bool eraseUnstablePosesAll(sfmData::SfMData& sfmData, const IndexT minPointsPerPose, std::set<IndexT>* outRemovedViewsId)
{
    IndexT removedElements = 0;
    const sfmData::Landmarks& landmarks = sfmData.getLandmarks();

    // Count the observation poses occurrence
    std::map<IndexT, IndexT> posesCount;
    for (auto& [poseId, pose] : sfmData.getPoses().valueRange())
    {
        if (!pose.isRemovable())
            continue;
        posesCount[poseId] = 0;
    }

    // each view's slot in posesCount (-1: its pose is not counted)
    std::map<IndexT, int> poseSlot;
    for (const auto& [poseId, count] : posesCount)
        poseSlot.emplace(poseId, static_cast<int>(poseSlot.size()));
    std::unordered_map<IndexT, int> viewSlot;
    viewSlot.reserve(sfmData.getViews().size());
    for (const auto& [viewId, view] : sfmData.getViews())
    {
        const auto it = poseSlot.find(view->getPoseId());
        viewSlot.emplace(viewId, it == poseSlot.end() ? -1 : it->second);
    }
    std::vector<const sfmData::Landmark*> all;
    all.reserve(landmarks.size());
    for (const auto& [idLandmark, landmark] : landmarks)
        all.push_back(&landmark);
    const int S = static_cast<int>(poseSlot.size());
    std::vector<IndexT> total(static_cast<std::size_t>(S), 0);
#pragma omp parallel
    {
        std::vector<IndexT> local(static_cast<std::size_t>(S), 0);
#pragma omp for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(all.size()); ++i)
            for (const auto& [viewId, obs] : all[static_cast<std::size_t>(i)]->getObservations())
            {
                const int s = viewSlot.at(viewId);
                if (s >= 0)
                    ++local[static_cast<std::size_t>(s)];
            }
#pragma omp critical
        for (int s = 0; s < S; ++s)
            total[static_cast<std::size_t>(s)] += local[static_cast<std::size_t>(s)];
    }
    for (auto& [poseId, count] : posesCount)
        count = total[static_cast<std::size_t>(poseSlot.at(poseId))];

    // If usage count is smaller than the threshold, remove the Pose
    for (const auto& [idPose, count] : posesCount)
    {
        if (count < minPointsPerPose)
        {
            sfmData.erasePose(idPose, true);  // no throw

            for (auto& [viewId, view] : sfmData.getViews())
            {
                if (view->getPoseId() == idPose)
                {
                    if (view->isPartOfRig())
                    {
                        // the pose is now independant
                        view->setPoseId(idPose);
                        view->setIndependantPose(true);
                    }

                    // add view id to the removedViewsId set
                    if (outRemovedViewsId != NULL)
                    {
                        outRemovedViewsId->insert(viewId);
                    }
                }
            }
            ++removedElements;
        }
    }

    if (removedElements)
    {
        ALICEVISION_LOG_DEBUG("eraseUnstablePoses: " << removedElements);
    }

    return removedElements > 0;
}

// eraseUnstablePosesAndObservations (sfmFilters.cpp:293-321) with eraseUnstablePosesAll; the passes after a pose went
// (rare) are upstream's
inline bool eraseUnstablePosesAndObservationsAll(sfmData::SfMData& sfmData,
                                                 const IndexT minPointsPerPose,
                                                 const IndexT minPointsPerLandmark,
                                                 std::set<IndexT>* outRemovedViewsId)
{
    bool removedContent = false;
    bool removedPoses = false;
    bool removedObservations = false;
    do
    {
        removedContent = false;
        if (eraseUnstablePosesAll(sfmData, minPointsPerPose, outRemovedViewsId))
        {
            removedPoses = true;
            removedContent = eraseObservationsWithMissingPoses(sfmData, minPointsPerLandmark);
            if (removedContent)
            {
                removedObservations = true;
            }

            removedContent = eraseLandmarksWithMissingReferenceView(sfmData);
            // Erase some observations can make some Poses index disappear so perform the process in a loop
        }
    } while (removedContent);

    return removedPoses || removedObservations;
}

// eraseUnstablePosesAndObservations, with the count done only for the poses that can have changed.
inline void eraseUnstableAfterAdjust(sfmData::SfMData& sfmData,
                                     const track::TracksPerView& tracksPerView,
                                     const IndexT minPointsPerPose,
                                     const IndexT minPointsPerLandmark,
                                     const bool localStrategy,
                                     const std::set<IndexT>& viewsWithErasures,
                                     const std::set<IndexT>& newReconstructedViews,
                                     std::set<IndexT>* outRemovedViewsId,
                                     std::size_t& posesChecked)
{
    posesChecked = 0;
    if (!localStrategy || !localPassesEnabled())
    {
        eraseUnstablePosesAndObservationsAll(sfmData, minPointsPerPose, minPointsPerLandmark, outRemovedViewsId);
        return;
    }
    std::set<IndexT> poses;
    for (const IndexT viewId : viewsWithErasures)
        if (sfmData.getViews().find(viewId) != sfmData.getViews().end())
            poses.insert(sfmData.getView(viewId).getPoseId());
    for (const IndexT viewId : newReconstructedViews)
        if (sfmData.getViews().find(viewId) != sfmData.getViews().end())
            poses.insert(sfmData.getView(viewId).getPoseId());
    bool removedAny = false;
    for (const IndexT poseId : poses)
    {
        const auto poseIt = sfmData.getPoses().find(poseId);
        if (poseIt == sfmData.getPoses().end() || !poseIt->second->isRemovable())
            continue;
        ++posesChecked;
        if (poseObservationCount(sfmData, tracksPerView, poseId) >= minPointsPerPose)
            continue;
        // the same removal as eraseUnstablePoses
        sfmData.erasePose(poseId, true);
        for (auto& [viewId, view] : sfmData.getViews())
        {
            if (view->getPoseId() == poseId)
            {
                if (view->isPartOfRig())
                {
                    view->setPoseId(poseId);
                    view->setIndependantPose(true);
                }
                if (outRemovedViewsId != nullptr)
                    outRemovedViewsId->insert(viewId);
            }
        }
        removedAny = true;
    }
    if (removedAny)
    {
        // a pose went: its observations, the landmarks they leave short, and any pose those leave
        // short - upstream's full pass finishes the iteration exactly
        eraseObservationsWithMissingPoses(sfmData, minPointsPerLandmark);
        eraseLandmarksWithMissingReferenceView(sfmData);
        eraseUnstablePosesAndObservationsAll(sfmData, minPointsPerPose, minPointsPerLandmark, outRemovedViewsId);
    }
}

// removeOutliers, restricted to the candidates under the local strategy.
inline std::size_t removeOutliersAfterAdjust(sfmData::SfMData& sfmData,
                                             const track::TracksPerView& tracksPerView,
                                             EFeatureConstraint featureConstraint,
                                             const double maxReprojectionError,
                                             const double minAngleForLandmark,
                                             const bool localStrategy,
                                             BundleAdjustment::ERefineOptions refineOptions,
                                             std::set<IndexT>& viewsWithErasures,
                                             std::size_t& candidates,
                                             bool& full)
{
    full = !localStrategy || !localPassesEnabled();
    std::size_t nbResidual = 0, nbAngle = 0;
    if (full)
    {
        candidates = sfmData.getLandmarks().size();
        nbResidual = removeOutliersWithPixelResidualErrorAll(sfmData, featureConstraint, maxReprojectionError, 2);   // step 7l
        nbAngle = removeOutliersWithAngleError(sfmData, minAngleForLandmark);
    }
    else
    {
        std::vector<IndexT> ids;
        CandidateSources sources;
        candidateLandmarks(sfmData, tracksPerView, refineOptions, ids, sources);
        candidates = ids.size();
        if (sfmProfileEnabled())
            ALICEVISION_LOG_INFO("cheshire: post-adjust sources: " << sources.views << " views (" << sources.refinedPoses << " refined poses, "
                                                                   << sources.refinedIntrinsics << " refined intrinsics)");
        nbResidual = removeOutliersWithPixelResidualErrorOn(sfmData, featureConstraint, maxReprojectionError, 2, ids, viewsWithErasures);
        nbAngle = removeOutliersWithAngleErrorOn(sfmData, minAngleForLandmark, ids, viewsWithErasures);
    }
    ALICEVISION_LOG_INFO("Remove outliers: " << std::endl
                                             << "\t- # outliers residual error: " << nbResidual << std::endl
                                             << "\t- # outliers angular error: " << nbAngle);
    return nbResidual + nbAngle;
}

}  // namespace cheshire
}  // namespace sfm
}  // namespace aliceVision
