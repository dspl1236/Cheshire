// This file is part of the Cheshire project (MPL-2.0), a patch set over AliceVision.
// Included at the end of sfm/bundle/BundleAdjustmentCeres.cpp by scripts/apply_hip_patch.py
// (step 5r); the declarations it needs are patched into BundleAdjustmentCeres.hpp.
//
// A Problem that lives across solves. Upstream builds a ceres::Problem from scratch for every
// bundle adjustment and destroys it afterwards: a cost function and a residual block per
// observation, then the mirror image. Measured (docs/04, "what a bundle adjustment costs around
// Ceres' Solve") at 1.2 to 1.5 us to build and 0.4 us to destroy per residual block, against 4.4 us
// of solver work per block per solve on 41 views - a quarter of the budget. Here the Problem is
// kept in the BundleAdjustmentCeres object, and each adjust() only applies what changed since the
// last one: new landmarks and observations get residual blocks, gone ones lose them, values are
// copied in, states become constant/variable. The engine keeps one BundleAdjustmentCeres for the
// whole reconstruction (ReconstructionEngine_sequentialSfM::_cheshireBA), so the first solve
// builds everything and every later one is a delta.
//
// What "the same problem" means here. Upstream leaves IGNORED poses and landmarks out of the
// Problem; here their blocks stay in and their residual blocks are removed (a landmark's
// residuals follow its state; an ignored pose observes only ignored landmarks, by the local BA
// rule that a landmark with an ignored observer is ignored). Ceres drops parameter blocks without
// residual blocks and residual blocks whose blocks are all constant from the reduced program, so
// the solver sees the same problem. What differs is the order of residual blocks and of the
// landmark blocks inside elimination group 0 (creation order rather than key order), which is the
// order of floating-point sums, so the trajectory is not bit-identical to the rebuild's; it is
// bit-identical run to run, which is what the deterministic gate checks.
//
// Where it pays. With every landmark active (no local strategy, i.e. up to 100 poses) a solve's
// delta is the new landmarks and observations, and the 41-view build went from 6.5 s to 2.4 s.
// Under the local strategy the ignored region moves with the frontier every solve: on the 884-view
// False Door 23.6 M residual blocks were added and 18.4 M removed over 419 solves against 70.8 M
// active in all, and Ceres' preprocessor scans every residual block in the Problem per Solve, not
// the active ones, so persistence cost more than the rebuild there. The engine therefore sets
// CeresOptions::cheshirePersist to "local strategy off", and the Problem is dropped when the
// strategy switches on. The state-toggling code stays (it is what the checks validated) for the
// solves before that switch.
//
// Covered: poses, intrinsics with distortion, projection residuals - incremental SfM. Rigs,
// survey points, 2D and point constraints, rotation priors, temporal smoothness, depth observations
// and mesh-referenced landmarks fall back to the rebuild (cheshirePersistAllowed), as does
// CHESHIRE_BA_PERSIST=0. CHESHIRE_BA_PERSIST_CHECK=1 recomputes, after every sync, the set of
// (landmark, view) pairs that should have a residual block and compares it with the records.

#include <algorithm>
#include <cstdlib>
#include <string>

namespace aliceVision {
namespace sfm {

namespace {

bool cheshirePersistEnabled()
{
    static const bool on = [] {
        const char* v = std::getenv("CHESHIRE_BA_PERSIST");
        return v == nullptr || std::string(v) != "0";
    }();
    return on;
}

bool cheshirePersistCheckEnabled()
{
    static const bool on = std::getenv("CHESHIRE_BA_PERSIST_CHECK") != nullptr;
    return on;
}

bool cheshireLogCostEnabled()
{
    static const bool on = std::getenv("CHESHIRE_BA_LOG_COST") != nullptr;
    return on;
}

}  // namespace

bool BundleAdjustmentCeres::cheshirePersistAllowed(const sfmData::SfMData& sfmData, ERefineOptions refineOptions) const
{
    if (!cheshirePersistEnabled() || !_ceresOptions.cheshirePersist)
        return false;
    if (refineOptions & BundleAdjustment::REFINE_TEMPORAL_SMOOTHNESS_CONSTRAINT)
        return false;
    return sfmData.getRigs().empty() && sfmData.getSurveyPoints().empty() && sfmData.getConstraints2D().empty() &&
           sfmData.getConstraintsPoint().empty() && sfmData.getRotationPriors().empty();
}

void BundleAdjustmentCeres::cheshireDropPersistent()
{
    _cheshireProblem.reset();
    _cheshireLandmarkRecs.clear();
    _cheshireSlab = 0;
}

void BundleAdjustmentCeres::setOptions(const CeresOptions& options)
{
    // the residual blocks of a persistent problem hold the loss function by pointer, so it stays
    const std::shared_ptr<ceres::LossFunction> loss = _ceresOptions.lossFunction;
    _ceresOptions = options;
    _ceresOptions.lossFunction = loss;
}

bool BundleAdjustmentCeres::cheshirePersistentBuild(const sfmData::SfMData& sfmData, ERefineOptions refineOptions, std::vector<ceres::ResidualBlockId>& landmarksBlockIds)
{
    const bool refineStructure = refineOptions & REFINE_STRUCTURE;
    const std::size_t nLandmarks = sfmData.getLandmarks().size();

    // the slab: landmark blocks never move, so it must hold every landmark this problem will see
    if (_cheshireProblem && _landmarksBlocks.size() + nLandmarks > _cheshireSlab)
    {
        std::size_t fresh = 0;
        for (const auto& [landmarkId, landmark] : sfmData.getLandmarks())
            if (_landmarksBlocks.find(landmarkId) == _landmarksBlocks.end())
                ++fresh;
        if (_landmarksBlocks.size() + fresh > _cheshireSlab)
        {
            ALICEVISION_LOG_INFO("cheshire: BA persistent: the landmark slab (" << _cheshireSlab << ") is full, rebuilding the problem");
            cheshireDropPersistent();
        }
    }

    if (_cheshireProblem && cheshirePersistCheckEnabled())
        cheshirePersistCheck(sfmData, *_cheshireProblem, true);

    const bool fresh = !_cheshireProblem;
    if (fresh)
    {
        resetProblem();
        _cheshireLandmarkRecs.clear();
        _cheshireSlab = std::max<std::size_t>(4 * nLandmarks + 100000, std::size_t(2) << 20);  // 2 M slots (64 MB); the 884-view set overflowed 1 M once
        _landmarksBlocks.reserve(_cheshireSlab);
        ceres::Problem::Options options;
        options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        options.evaluation_callback = this;
        options.enable_fast_removal = true;
        options.disable_all_safety_checks = true;
        _cheshireProblem = std::make_unique<ceres::Problem>(options);
    }
    ceres::Problem& problem = *_cheshireProblem;
    _statistics = Statistics();
    _allParametersBlocks.clear();
    _cheshireOrdered.clear();
    _cheshirePoseGroup.clear();
    _cheshireIntrinsicGroup.clear();
    {
        int g = 1;
        for (const auto& [poseId, pose] : sfmData.getPoses().valueRange())
            _cheshirePoseGroup[poseId] = g++;
        g = 2000000;
        for (const auto& [intrinsicId, intrinsicPtr] : sfmData.getIntrinsics())
            _cheshireIntrinsicGroup[intrinsicId] = g++;
    }

    // 1. poses: every block back to variable and unconstrained, then upstream's own pass adds the
    //    new ones, copies the values and sets constant and manifolds as it would for a fresh problem
    for (auto& [poseId, block] : _posesBlocks)
    {
        if (problem.HasParameterBlock(block.data()))
        {
            problem.SetParameterBlockVariable(block.data());
            problem.SetManifold(block.data(), nullptr);
        }
    }
    addExtrinsicsToProblem(sfmData, refineOptions, problem);
    for (const auto& [poseId, pose] : sfmData.getPoses().valueRange())
    {
        if (pose.getState() != EEstimatorParameterState::IGNORED)
            continue;
        auto it = _posesBlocks.find(poseId);
        if (it != _posesBlocks.end() && problem.HasParameterBlock(it->second.data()))
            problem.SetParameterBlockConstant(it->second.data());
    }

    // 2. intrinsics: same, and the cost functions keep the intrinsic object they were created with,
    //    so it is updated in place rather than replaced (addIntrinsicsToProblem is patched for that)
    for (auto& [intrinsicId, block] : _intrinsicsBlocks)
        if (problem.HasParameterBlock(block.data()))
        {
            problem.SetParameterBlockVariable(block.data());
            problem.SetManifold(block.data(), nullptr);
        }
    for (auto& [intrinsicId, block] : _distortionsBlocks)
        if (problem.HasParameterBlock(block.data()))
            problem.SetParameterBlockVariable(block.data());
    addIntrinsicsToProblem(sfmData, refineOptions, problem);

    // 3. landmarks and their residual blocks: a merge walk of the scene's landmarks and the records
    struct ViewBlocks
    {
        double* pose = nullptr;
        double* intrinsic = nullptr;
        double* distortion = nullptr;
        std::shared_ptr<camera::IntrinsicBase> intrinsicObject;
        bool ok = false;
    };
    std::unordered_map<IndexT, ViewBlocks> views;
    views.reserve(sfmData.getViews().size());
    double* const fakeDistortionBlockPtr = &_fakeDistortionBlock;
    const auto viewBlocks = [&](IndexT viewId) -> const ViewBlocks& {
        auto it = views.find(viewId);
        if (it != views.end())
            return it->second;
        ViewBlocks e;
        const sfmData::View& view = sfmData.getView(viewId);
        const IndexT intrinsicId = view.getIntrinsicId();
        if (sfmData.isPoseAndIntrinsicDefined(view) && sfmData.getAbsolutePose(view.getPoseId()).getState() != EEstimatorParameterState::IGNORED)
        {
            const auto poseIt = _posesBlocks.find(view.getPoseId());
            const auto intrinsicIt = _intrinsicsBlocks.find(intrinsicId);
            if (poseIt != _posesBlocks.end() && intrinsicIt != _intrinsicsBlocks.end())
            {
                e.pose = poseIt->second.data();
                e.intrinsic = intrinsicIt->second.data();
                e.intrinsicObject = _intrinsicObjects[intrinsicId];
                e.distortion = fakeDistortionBlockPtr;
                const auto distortionIt = _distortionsBlocks.find(intrinsicId);
                if (distortionIt != _distortionsBlocks.end())
                    e.distortion = distortionIt->second.data();
                e.ok = true;
            }
        }
        return views.emplace(viewId, e).first->second;
    };
    ceres::LossFunction* lossFunction = _ceresOptions.lossFunction.get();
    const auto addResidual = [&](double* landmarkBlock, IndexT viewId, const sfmData::Observation& observation) -> ceres::ResidualBlockId {
        const ViewBlocks& v = viewBlocks(viewId);
        if (!v.ok)
            return nullptr;
        ceres::LossFunction* weightedLossFunction = lossFunction;
        if (observation.getWeight() != 1.0)
            weightedLossFunction = new ceres::ScaledLoss(lossFunction, observation.getWeight(), ceres::TAKE_OWNERSHIP);
        ceres::CostFunction* costFunction = cheshire::createProjectionCost(false, v.intrinsicObject, observation);
        double* params[4] = {v.intrinsic, v.distortion, v.pose, landmarkBlock};
        return problem.AddResidualBlock(costFunction, weightedLossFunction, params, 4);
    };

    std::size_t added = 0, removed = 0, kept = 0;
    bool unsupported = false;
    auto lm = sfmData.getLandmarks().begin();
    const auto lmEnd = sfmData.getLandmarks().end();
    auto rec = _cheshireLandmarkRecs.begin();
    while ((lm != lmEnd || rec != _cheshireLandmarkRecs.end()) && !unsupported)
    {
        const bool haveScene = lm != lmEnd;
        const bool haveRec = rec != _cheshireLandmarkRecs.end();
        if (haveScene && (!haveRec || lm->first < rec->first))
        {
            // a landmark the problem has not seen (or saw, lost, and sees again under the same id)
            const IndexT landmarkId = lm->first;
            const sfmData::Landmark& landmark = lm->second;
            if (landmark.getReferenceViewIndex() != UndefinedIndexT)
            {
                unsupported = true;
                break;
            }
            std::array<double, 3>& block = _landmarksBlocks[landmarkId];
            for (std::size_t i = 0; i < 3; ++i)
                block[i] = landmark.getX()(Eigen::Index(i));
            problem.AddParameterBlock(block.data(), 3);
            _linearSolverOrdering.AddElementToGroup(block.data(), 0);
            CheshireLandmarkRec r;
            r.block = block.data();
            r.constant = !refineStructure || landmark.getState() == EEstimatorParameterState::CONSTANT || landmark.isLocked() ||
                         landmark.getState() == EEstimatorParameterState::IGNORED;
            if (r.constant)
                problem.SetParameterBlockConstant(block.data());
            else
                problem.SetParameterBlockVariable(block.data());
            r.active = landmark.getState() != EEstimatorParameterState::IGNORED;
            if (r.active)
            {
                r.obs.reserve(landmark.getObservations().size());
                for (const auto& [viewId, observation] : landmark.getObservations())
                {
                    if (observation.getDepth() > 0.0)
                    {
                        unsupported = true;
                        break;
                    }
                    const ceres::ResidualBlockId id = addResidual(block.data(), viewId, observation);
                    if (id != nullptr)
                        r.obs.push_back({viewId, id});
                }
                added += r.obs.size();
            }
            _statistics.addState(EParameter::LANDMARK, r.active ? (r.constant ? EEstimatorParameterState::CONSTANT : EEstimatorParameterState::REFINED)
                                                                : EEstimatorParameterState::IGNORED);
            rec = _cheshireLandmarkRecs.emplace_hint(rec, landmarkId, std::move(r));
            ++rec;
            ++lm;
        }
        else if (haveRec && (!haveScene || rec->first < lm->first))
        {
            // a landmark the scene no longer has
            for (const CheshireObs& o : rec->second.obs)
                problem.RemoveResidualBlock(o.id);
            removed += rec->second.obs.size();
            double* block = rec->second.block;
            problem.RemoveParameterBlock(block);
            _linearSolverOrdering.Remove(block);
            rec = _cheshireLandmarkRecs.erase(rec);
        }
        else
        {
            // both: the value, the state, and the observations that came or went
            const IndexT landmarkId = lm->first;
            const sfmData::Landmark& landmark = lm->second;
            if (landmark.getReferenceViewIndex() != UndefinedIndexT)
            {
                unsupported = true;
                break;
            }
            CheshireLandmarkRec& r = rec->second;
            double* const blockPtr = r.block;
            const Vec3& X = landmark.getX();
            blockPtr[0] = X(0);
            blockPtr[1] = X(1);
            blockPtr[2] = X(2);
            const bool constant = !refineStructure || landmark.getState() == EEstimatorParameterState::CONSTANT || landmark.isLocked() ||
                                  landmark.getState() == EEstimatorParameterState::IGNORED;
            if (constant != r.constant)
            {
                if (constant)
                    problem.SetParameterBlockConstant(blockPtr);
                else
                    problem.SetParameterBlockVariable(blockPtr);
                r.constant = constant;
            }
            const bool wantActive = landmark.getState() != EEstimatorParameterState::IGNORED;
            if (!wantActive)
            {
                if (r.active)
                {
                    for (const CheshireObs& o : r.obs)
                        problem.RemoveResidualBlock(o.id);
                    removed += r.obs.size();
                    r.obs.clear();
                    r.active = false;
                }
            }
            else
            {
                // merge the scene's observations (sorted by view) against the record (sorted by view)
                std::vector<CheshireObs> next;
                next.reserve(landmark.getObservations().size());
                auto ob = landmark.getObservations().begin();
                const auto obEnd = landmark.getObservations().end();
                std::size_t k = 0;
                while (ob != obEnd || k < r.obs.size())
                {
                    if (ob != obEnd && (k == r.obs.size() || ob->first < r.obs[k].viewId))
                    {
                        if (ob->second.getDepth() > 0.0)
                        {
                            unsupported = true;
                            break;
                        }
                        const ceres::ResidualBlockId id = addResidual(blockPtr, ob->first, ob->second);
                        if (id != nullptr)
                        {
                            next.push_back({ob->first, id});
                            ++added;
                        }
                        ++ob;
                    }
                    else if (k < r.obs.size() && (ob == obEnd || r.obs[k].viewId < ob->first))
                    {
                        problem.RemoveResidualBlock(r.obs[k].id);
                        ++removed;
                        ++k;
                    }
                    else
                    {
                        // still observed, but is the view still a valid residual (posed, not ignored)?
                        if (viewBlocks(ob->first).ok)
                        {
                            next.push_back(r.obs[k]);
                            ++kept;
                        }
                        else
                        {
                            problem.RemoveResidualBlock(r.obs[k].id);
                            ++removed;
                        }
                        ++ob;
                        ++k;
                    }
                }
                r.obs.swap(next);
                r.active = true;
            }
            _statistics.addState(EParameter::LANDMARK, r.active ? (r.constant ? EEstimatorParameterState::CONSTANT : EEstimatorParameterState::REFINED)
                                                                : EEstimatorParameterState::IGNORED);
            ++lm;
            ++rec;
        }
    }
    if (unsupported)
    {
        ALICEVISION_LOG_INFO("cheshire: BA persistent: a mesh-referenced landmark or a depth observation, falling back to the rebuild");
        cheshireDropPersistent();
        return false;
    }

    // 4. poses the scene no longer has: their observations went with them, so no residual references them
    for (auto it = _posesBlocks.begin(); it != _posesBlocks.end();)
    {
        if (sfmData.getPoses().find(it->first) != sfmData.getPoses().end())
        {
            ++it;
            continue;
        }
        if (problem.HasParameterBlock(it->second.data()))
            problem.RemoveParameterBlock(it->second.data());
        _linearSolverOrdering.Remove(it->second.data());
        it = _posesBlocks.erase(it);
    }

    // 5. the ordering: the camera-side blocks re-entered with this solve's groups (a pose's rank
    //    moves when poses arrive), the landmarks were entered at creation
    if (_ceresOptions.useParametersOrdering)
    {
        for (auto& [poseId, block] : _posesBlocks)
            _linearSolverOrdering.AddElementToGroup(block.data(), cheshirePoseGroup(poseId));
        for (auto& [intrinsicId, block] : _intrinsicsBlocks)
            if (_cheshireIntrinsicGroup.count(intrinsicId) != 0)
                _linearSolverOrdering.AddElementToGroup(block.data(), cheshireIntrinsicGroup(intrinsicId));
        for (auto& [intrinsicId, block] : _distortionsBlocks)
            if (_cheshireIntrinsicGroup.count(intrinsicId) != 0)
                _linearSolverOrdering.AddElementToGroup(block.data(), cheshireDistortionGroup(intrinsicId, false));
        _linearSolverOrdering.AddElementToGroup(fakeDistortionBlockPtr, cheshireDistortionGroup(0, true));
    }

    if (cheshireLogCostEnabled())
        for (const auto& [landmarkId, r] : _cheshireLandmarkRecs)
            for (const CheshireObs& o : r.obs)
                landmarksBlockIds.push_back(o.id);

    static const bool profile = std::getenv("CHESHIRE_BA_PROFILE") != nullptr;
    if (profile)
        ALICEVISION_LOG_INFO("cheshire: BA persistent: " << (fresh ? "built" : "synced") << " - residual blocks added " << added << ", removed " << removed
                                                         << ", kept " << kept << "; " << problem.NumResidualBlocks() << " residual blocks, "
                                                         << problem.NumParameterBlocks() << " parameter blocks, " << _cheshireLandmarkRecs.size()
                                                         << " landmarks in the problem");

    if (cheshirePersistCheckEnabled())
        cheshirePersistCheck(sfmData, problem, false);
    return true;
}

// The invariant: the residual blocks are exactly the observations of the non-ignored landmarks
// whose view has a non-ignored pose, one each, and nothing else.
void BundleAdjustmentCeres::cheshirePersistCheck(const sfmData::SfMData& sfmData, const ceres::Problem& problem, bool preSync) const
{
    std::size_t expected = 0, missing = 0, extra = 0, recorded = 0;
    if (preSync)
    {
        // before the sync only the ids can be compared: the scene has moved on, the records have not
        std::size_t idMismatch = 0, recordedPre = 0;
        for (const auto& [landmarkId, r] : _cheshireLandmarkRecs)
        {
            recordedPre += r.obs.size();
            std::vector<ceres::ResidualBlockId> held;
            problem.GetResidualBlocksForParameterBlock(r.block, &held);
            std::vector<ceres::ResidualBlockId> mine;
            for (const CheshireObs& o : r.obs)
                mine.push_back(o.id);
            std::sort(held.begin(), held.end());
            std::sort(mine.begin(), mine.end());
            if (held != mine)
                ++idMismatch;
        }
        ALICEVISION_LOG_INFO("cheshire: BA persistent pre-sync check: " << problem.NumResidualBlocks() << " residual blocks in the problem, " << recordedPre
                                                                    << " recorded; landmarks whose Ceres residual ids differ from the record " << idMismatch
                                                                    << (idMismatch == 0 && static_cast<std::size_t>(problem.NumResidualBlocks()) == recordedPre ? " - consistent" : " - MISMATCH"));
        return;
    }
    for (const auto& [landmarkId, landmark] : sfmData.getLandmarks())
    {
        const auto recIt = _cheshireLandmarkRecs.find(landmarkId);
        const bool active = landmark.getState() != EEstimatorParameterState::IGNORED;
        std::vector<IndexT> want;
        if (active)
            for (const auto& [viewId, observation] : landmark.getObservations())
            {
                const sfmData::View& view = sfmData.getView(viewId);
                if (sfmData.isPoseAndIntrinsicDefined(view) &&
                    sfmData.getAbsolutePose(view.getPoseId()).getState() != EEstimatorParameterState::IGNORED)
                    want.push_back(viewId);
            }
        expected += want.size();
        std::vector<IndexT> have;
        if (recIt != _cheshireLandmarkRecs.end())
            for (const CheshireObs& o : recIt->second.obs)
                have.push_back(o.viewId);
        recorded += have.size();
        std::vector<IndexT> diff;
        std::set_difference(want.begin(), want.end(), have.begin(), have.end(), std::back_inserter(diff));
        missing += diff.size();
        diff.clear();
        std::set_difference(have.begin(), have.end(), want.begin(), want.end(), std::back_inserter(diff));
        extra += diff.size();
    }
    std::size_t stale = 0, idMismatch = 0;
    for (const auto& [landmarkId, r] : _cheshireLandmarkRecs)
    {
        if (sfmData.getLandmarks().find(landmarkId) == sfmData.getLandmarks().end())
            stale += r.obs.size() + 1;
        // the residual blocks Ceres holds for this landmark block must be exactly the recorded ids
        std::vector<ceres::ResidualBlockId> held;
        problem.GetResidualBlocksForParameterBlock(r.block, &held);
        std::vector<ceres::ResidualBlockId> mine;
        mine.reserve(r.obs.size());
        for (const CheshireObs& o : r.obs)
            mine.push_back(o.id);
        std::sort(held.begin(), held.end());
        std::sort(mine.begin(), mine.end());
        if (held != mine)
            ++idMismatch;
    }
    const int inProblem = problem.NumResidualBlocks();
    const bool ok = missing == 0 && extra == 0 && stale == 0 && idMismatch == 0 && static_cast<std::size_t>(inProblem) == recorded;
    ALICEVISION_LOG_INFO("cheshire: BA persistent check: " << inProblem << " residual blocks in the problem, " << recorded << " recorded, " << expected
                                                           << " expected from the scene; missing " << missing << ", extra " << extra << ", stale " << stale
                                                           << ", landmarks whose Ceres residual ids differ from the record " << idMismatch
                                                           << (ok ? " - consistent" : " - MISMATCH"));
}

}  // namespace sfm
}  // namespace aliceVision
