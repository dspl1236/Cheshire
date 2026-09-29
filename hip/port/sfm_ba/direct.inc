// This file is part of the Cheshire patch set for AliceVision (https://github.com/dspl1236/Cheshire).
// Included by scripts/apply_hip_patch.py (step 7e) at the end of BundleAdjustmentCeres.cpp.
//
// 0.3.7, step 3b of Cheshire's own bundle-adjustment solver (docs/notes/ba-own-solver.md): the solve built
// from the SfM data directly. Upstream builds a ceres::Problem for every solve - a cost function object
// and a residual block per observation, several million on a large set - only for the solver to read it
// back and for the destructor to free it again. Here the poses, intrinsics and distortions still go
// through upstream's own addExtrinsicsToProblem / addIntrinsicsToProblem, into a small Problem that holds
// only those few hundred blocks (so their constant flags, manifolds and bounds are upstream's), and the
// landmarks and observations become the solver's rows directly, evaluated by the same fused projection
// code the cost functions run (projectSimple / fusedProject), in the order addLandmarksToProblem adds
// them. The solver sees the program Ceres' preprocessor would build from createProblem's Problem, so the
// result is the same bytes; CHESHIRE_BA_DIRECT=check runs both on every solve and compares.
//
// CHESHIRE_BA_DIRECT: 1 on (the default), 0 off, check. The persistent Problem (step 5r) is not used on
// this path: there is no Problem left to keep.

namespace aliceVision {
namespace sfm {
namespace cheshire {
namespace own {

// the projection an observation row evaluates: CheshireIntrinsicsProject's Evaluate on a fused model
struct DirectInner
{
    const FusedCamera* cam;
    const sfmData::Observation* obs;
    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
    {
        return fusedProject(*cam, *obs, parameters, residuals, jacobians);
    }
};

// One observation: blocks intrinsics, distortion (or the fake block), pose, landmark
struct DirectRow
{
    double* params[4];
    const sfmData::Observation* obs;
    int camera;       // into DirectSource's cameras
    double weight;    // ScaledLoss's factor, 1 for none
};

struct DirectCamera
{
    FusedCamera fused;
    bool hasDistortion = false;
};

class DirectSource final : public Source
{
  public:
    DirectSource(const ceres::Problem& small, const ceres::LossFunction* loss)
      : _small(small), _loss(loss)
    {}

    std::vector<DirectCamera> cameras;
    std::vector<DirectRow> rows;
    // the landmark blocks in _landmarksBlocks' order (their address order), with their constant flags
    std::vector<double*> landmarks;
    std::vector<char> landmarkConstant;
    // the camera blocks the rows use, with their ordering groups
    std::vector<std::pair<int, double*>> cameraBlocks;
    int landmarkBlocks = 0;   // every landmark block the Problem would hold (for the counts)

    bool blocks(std::vector<BlockInfo>* out, int threads, std::string*) const override
    {
        out->clear();
        out->resize(landmarks.size() + cameraBlocks.size());
        for (size_t i = 0; i < landmarks.size(); ++i)
        {
            BlockInfo& b = (*out)[i];
            b.ptr = landmarks[i];
            b.eliminate = true;
            b.constant = landmarkConstant[i] != 0;
            if (!b.constant)
            {
                b.size = 3;
                b.tsize = 3;
            }
        }
        // the camera blocks by group (one block per group, as step 5n numbers them); their properties are
        // the small Problem's, i.e. upstream's
        std::vector<std::pair<int, double*>> sorted = cameraBlocks;
        std::sort(sorted.begin(), sorted.end());
        const double big = std::numeric_limits<double>::max();
        const size_t base = landmarks.size();
        for (size_t i = 0; i < sorted.size(); ++i)
        {
            double* p = sorted[i].second;
            BlockInfo& b = (*out)[base + i];
            b.ptr = p;
            if (!_small.HasParameterBlock(p) || _small.IsParameterBlockConstant(p))
            {
                b.constant = true;
                continue;
            }
            b.size = _small.ParameterBlockSize(p);
            b.tsize = _small.ParameterBlockTangentSize(p);
            b.manifold = _small.GetManifold(p);
            if (b.tsize == 0)
                continue;
            bool any = false;
            std::vector<double> lo(static_cast<size_t>(b.size)), hi(static_cast<size_t>(b.size));
            for (int k = 0; k < b.size; ++k)
            {
                lo[k] = _small.GetParameterLowerBound(p, k);
                hi[k] = _small.GetParameterUpperBound(p, k);
                any = any || lo[k] > -big || hi[k] < big;
            }
            if (any)
            {
                b.lower = std::move(lo);
                b.upper = std::move(hi);
            }
        }
        (void)threads;
        return true;
    }
    int numRows() const override { return int(rows.size()); }
    int rowNumParams(int) const override { return 4; }
    void rowParams(int i, double** out) const override { std::copy(rows[i].params, rows[i].params + 4, out); }
    int rowNumResiduals(int) const override { return 2; }
    const ceres::LossFunction* rowLoss(int) const override { return _loss; }
    double rowLossScale(int i) const override { return rows[i].weight; }
    bool evaluate(int i, double const* const* params, double* residuals, double** jacobians) const override
    {
        const DirectRow& row = rows[i];
        const DirectCamera& cam = cameras[row.camera];
        const EvalTimer cheshireEvalTimer(jacobians != nullptr);  // step A1, as the cost function times it
        const DirectInner inner{&cam.fused, row.obs};
        return projectSimple(inner, cam.hasDistortion, cam.fused.sizes[1], params, residuals, jacobians);
    }
    void counts(int* parameterBlocks, int* parameters, int* residualBlocks, int* residuals) const override
    {
        *parameterBlocks = _small.NumParameterBlocks() + landmarkBlocks;
        *parameters = _small.NumParameters() + 3 * landmarkBlocks;
        *residualBlocks = int(rows.size());
        *residuals = 2 * int(rows.size());
    }

  private:
    const ceres::Problem& _small;
    const ceres::LossFunction* _loss;
};

}  // namespace own
}  // namespace cheshire

// Whether the direct build covers this solve, from the SfM data alone (nothing is touched when not)
bool BundleAdjustmentCeres::cheshireDirectInScope(const sfmData::SfMData& sfmData, ERefineOptions refineOptions, std::string* why) const
{
    if (refineOptions & BundleAdjustment::REFINE_TEMPORAL_SMOOTHNESS_CONSTRAINT)
        *why = "temporal smoothness";
    else if (!sfmData.getRigs().empty() || !sfmData.getSurveyPoints().empty() || !sfmData.getConstraints2D().empty() ||
             !sfmData.getConstraintsPoint().empty() || !sfmData.getRotationPriors().empty())
        *why = "rigs, surveys, constraints or rotation priors";
    else if (!_ceresOptions.useParametersOrdering)
        *why = "no parameter ordering";
    else if (cheshire::baJacobiansMode() != cheshire::BaJacobians::Analytic || cheshire::baCheckEnabled())
        *why = "Jacobians other than analytic, or the Jacobian check";
    if (!why->empty())
        return false;
    for (const auto& [intrinsicId, intrinsicPtr] : sfmData.getIntrinsics())
    {
        if (!camera::IntrinsicScaleOffset::cast(intrinsicPtr))
            continue;  // never refined, never projected through (addIntrinsicsToProblem skips it)
        if (cheshire::fusedCamera(intrinsicPtr).model < 0)
        {
            *why = "a camera model the fused projection does not cover";
            return false;
        }
    }
    for (const auto& [landmarkId, landmark] : sfmData.getLandmarks())
    {
        if (landmark.getState() == EEstimatorParameterState::IGNORED)
            continue;
        if (landmark.getReferenceViewIndex() != UndefinedIndexT)
        {
            *why = "mesh-referenced landmarks";
            return false;
        }
        for (const auto& [viewId, observation] : landmark.getObservations())
            if (observation.getDepth() > 0.0)
            {
                *why = "depth observations";
                return false;
            }
    }
    return true;
}

// The solve built from the SfM data (see the top of this file). Returns false when the direct path does not
// take this solve (out of scope, or the check mode, which only records): adjust() then runs as always.
// Otherwise *ok is adjust()'s result.
bool BundleAdjustmentCeres::cheshireDirectAdjust(sfmData::SfMData& sfmData, ERefineOptions refineOptions, bool* ok)
{
    const int mode = cheshire::own::directMode();
    _cheshireDirectPending = false;
    if (mode == 0 || !cheshire::own::solverEnabled() || cheshire::own::shadowEnabled())
        return false;
    std::string why;
    if (!cheshireDirectInScope(sfmData, refineOptions, &why))
    {
        cheshire::own::noteFallback("direct build: " + why);
        return false;
    }
    ceres::Solver::Options options;
    setSolverOptions(options);
    cheshire::own::Options o;
    if (!cheshire::own::solverOptions(options, int(_ceresOptions.nbThreads), &o, &why))
    {
        cheshire::own::noteFallback(why);
        return false;
    }

    static const bool cheshireBaProfile = ::cheshire::env::flag("CHESHIRE_BA_PROFILE");
    using Clock = std::chrono::steady_clock;
    const Clock::time_point t0 = Clock::now();

    // what createProblem does before the blocks: no Problem survives this solve, persistent or not
    cheshireDropPersistent();
    resetProblem();
    _landmarksBlocks.reserve(sfmData.getLandmarks().size());
    {
        int g = 1;
        for (const auto& [poseId, pose] : sfmData.getPoses().valueRange())
            _cheshirePoseGroup[poseId] = g++;
        g = 2000000;
        for (const auto& [intrinsicId, intrinsicPtr] : sfmData.getIntrinsics())
            _cheshireIntrinsicGroup[intrinsicId] = g++;
    }

    // the camera blocks through upstream's own code, into a Problem that holds only them
    ceres::Problem::Options problemOptions;
    problemOptions.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.evaluation_callback = this;
    problemOptions.disable_all_safety_checks = true;
    ceres::Problem small(problemOptions);
    addExtrinsicsToProblem(sfmData, refineOptions, small);
    addIntrinsicsToProblem(sfmData, refineOptions, small);

    // the landmarks and the observations, as addLandmarksToProblem adds them
    cheshire::own::DirectSource src(small, _ceresOptions.lossFunction.get());
    struct ViewBlocks
    {
        double* pose = nullptr;
        double* intrinsic = nullptr;
        double* distortion = nullptr;
        int camera = -1;
        bool ok = false;
    };
    std::unordered_map<IndexT, ViewBlocks> views;
    views.reserve(sfmData.getViews().size());
    std::map<IndexT, int> cameraOf;
    std::set<const double*> ordered;   // the camera blocks entered in the ordering, once each
    const bool refineStructure = refineOptions & REFINE_STRUCTURE;
    double* fakeDistortionBlockPtr = &_fakeDistortionBlock;
    src.rows.reserve(sfmData.getLandmarks().size() * 4);
    for (const auto& [landmarkId, landmark] : sfmData.getLandmarks())
    {
        if (landmark.getState() == EEstimatorParameterState::IGNORED)
        {
            _statistics.addState(EParameter::LANDMARK, EEstimatorParameterState::IGNORED);
            continue;
        }
        std::array<double, 3>& landmarkBlock = _landmarksBlocks[landmarkId];
        for (std::size_t i = 0; i < 3; ++i)
            landmarkBlock.at(i) = landmark.getX()(Eigen::Index(i));
        double* landmarkBlockPtr = landmarkBlock.data();
        _allParametersBlocks.push_back(landmarkBlockPtr);
        const bool constant = !refineStructure || landmark.getState() == EEstimatorParameterState::CONSTANT || landmark.isLocked();
        src.landmarks.push_back(landmarkBlockPtr);
        src.landmarkConstant.push_back(char(constant));

        for (const auto& [viewId, observation] : landmark.getObservations())
        {
            auto it = views.find(viewId);
            if (it == views.end())
            {
                ViewBlocks e;
                const sfmData::View& view = sfmData.getView(viewId);
                const IndexT intrinsicId = view.getIntrinsicId();
                e.ok = sfmData.isPoseAndIntrinsicDefined(view);
                if (!e.ok)
                {
                    ALICEVISION_LOG_ERROR("We should not have an undefined pose here");
                }
                else if (sfmData.getAbsolutePose(view.getPoseId()).getState() == EEstimatorParameterState::IGNORED)
                {
                    ALICEVISION_LOG_ERROR("We should not have an ignored pose here");
                    e.ok = false;
                }
                if (e.ok)
                {
                    e.pose = _posesBlocks.at(view.getPoseId()).data();
                    e.intrinsic = _intrinsicsBlocks.at(intrinsicId).data();
                    e.distortion = fakeDistortionBlockPtr;
                    if (_distortionsBlocks.find(intrinsicId) != _distortionsBlocks.end())
                        e.distortion = _distortionsBlocks.at(intrinsicId).data();
                    auto c = cameraOf.find(intrinsicId);
                    if (c == cameraOf.end())
                    {
                        cheshire::own::DirectCamera cam;
                        const std::shared_ptr<IntrinsicBase>& object = _intrinsicObjects[intrinsicId];
                        cam.fused = cheshire::fusedCamera(object);
                        cam.hasDistortion = cheshire::intrinsicHasDistortion(object);
                        c = cameraOf.emplace(intrinsicId, int(src.cameras.size())).first;
                        src.cameras.push_back(cam);
                    }
                    e.camera = c->second;
                    const std::pair<int, double*> blocks[3] = {{cheshirePoseGroup(view.getPoseId()), e.pose},
                                                               {cheshireIntrinsicGroup(intrinsicId), e.intrinsic},
                                                               {cheshireDistortionGroup(intrinsicId, e.distortion == fakeDistortionBlockPtr), e.distortion}};
                    for (const auto& b : blocks)
                        if (ordered.insert(b.second).second)
                            src.cameraBlocks.push_back(b);
                }
                it = views.emplace(viewId, e).first;
            }
            const ViewBlocks& vb = it->second;
            if (!vb.ok)
                continue;
            cheshire::own::DirectRow row;
            row.params[0] = vb.intrinsic;
            row.params[1] = vb.distortion;
            row.params[2] = vb.pose;
            row.params[3] = landmarkBlockPtr;
            row.obs = &observation;
            row.camera = vb.camera;
            row.weight = observation.getWeight();
            src.rows.push_back(row);
            _statistics.addState(EParameter::LANDMARK, constant ? EEstimatorParameterState::CONSTANT : EEstimatorParameterState::REFINED);
        }
    }
    src.landmarkBlocks = int(src.landmarks.size());
    for (const auto& cam : src.cameras)
        if (cam.fused.model < 0)
        {
            // cheshireDirectInScope checked the scene's cameras; a clone should not differ, but if it does,
            // the Problem path takes the solve (it rebuilds everything, persistence already dropped)
            cheshire::own::noteFallback("direct build: a camera model the fused projection does not cover");
            return false;
        }
    const double buildS = std::chrono::duration<double>(Clock::now() - t0).count();

    IterationInfos iterationInfos;
    options.callbacks.push_back(&iterationInfos);
    ceres::Solver::Summary summary;
    cheshire::own::Result result;
    const Clock::time_point t2 = Clock::now();
    if (!cheshire::own::solveSource(src, options, o, [this] { PrepareForEvaluation(true, true); }, &summary, &result))
        return false;  // the blocks are as they were; the Problem path takes the solve
    const double solveS = std::chrono::duration<double>(Clock::now() - t2).count();

    if (mode == 2)
    {
        // the check: record, leave the scene alone, let the Problem path solve the same problem
        _cheshireDirectPending = true;
        _cheshireDirectDigest = result.digest;
        _cheshireDirectDigestIn = result.digestIn;
        _cheshireDirectIterations = int(result.iterations.size()) - 1;
        _cheshireDirectFinalCost = result.finalCost;
        cheshire::own::lastResult() = cheshire::own::Result();   // so a Problem path that goes to Ceres is not compared
        return false;
    }

    if (cheshireBaProfile)
        ALICEVISION_LOG_INFO("cheshire: BA profile: total " << summary.total_time_in_seconds << " s = residuals "
                             << summary.residual_evaluation_time_in_seconds << " + jacobians " << summary.jacobian_evaluation_time_in_seconds
                             << " + linear solver " << summary.linear_solver_time_in_seconds << " + other; "
                             << summary.iterations.size() << " iterations, " << summary.num_threads_used << " threads, "
                             << ceres::LinearSolverTypeToString(summary.linear_solver_type_used) << ", "
                             << summary.num_residual_blocks << " residual blocks, " << summary.num_parameter_blocks << " parameter blocks");
    if (_ceresOptions.summary)
        ALICEVISION_LOG_INFO(summary.FullReport());
    if (!summary.IsSolutionUsable())
    {
        ALICEVISION_LOG_WARNING("Bundle Adjustment failed, the solution is not usable.");
        *ok = false;
        return true;
    }
    const Clock::time_point t4 = Clock::now();
    updateFromSolution(sfmData, refineOptions);
    surveyInfos(sfmData);
    _statistics.time = summary.total_time_in_seconds;
    _statistics.nbSuccessfullIterations = summary.num_successful_steps;
    _statistics.nbUnsuccessfullIterations = summary.num_unsuccessful_steps;
    _statistics.nbResidualBlocks = summary.num_residuals;
    _statistics.RMSEinitial = std::sqrt(summary.initial_cost / summary.num_residuals);
    _statistics.RMSEfinal = std::sqrt(summary.final_cost / summary.num_residuals);
    if (cheshireBaProfile)
        ALICEVISION_LOG_INFO("cheshire: BA adjust: build " << buildS << " s, log evaluations 0 s, solve " << solveS << " s (Ceres preprocessor "
                             << summary.preprocessor_time_in_seconds << ", minimizer " << summary.minimizer_time_in_seconds << ", postprocessor "
                             << summary.postprocessor_time_in_seconds << "), update " << std::chrono::duration<double>(Clock::now() - t4).count()
                             << " s; " << summary.num_residual_blocks << " residual blocks (direct)");
    ALICEVISION_LOG_INFO("BundleAdjustmentCeres::adjust end");
    *ok = true;
    return true;
}

// CHESHIRE_BA_DIRECT=check: the Problem path's solve against the direct one recorded before it
void BundleAdjustmentCeres::cheshireDirectCompare() const
{
    if (!_cheshireDirectPending)
        return;
    const cheshire::own::Result& r = cheshire::own::lastResult();
    if (!r.ok)
    {
        ALICEVISION_LOG_INFO("cheshire: BA direct check: the Problem path did not use the solver");
        return;
    }
    const bool same = r.digest == _cheshireDirectDigest && r.digestIn == _cheshireDirectDigestIn;
    ALICEVISION_LOG_INFO("cheshire: BA direct check: " << (same ? "same" : "DIFFERENT") << ", " << r.rows << " rows, iterations "
                         << _cheshireDirectIterations << " / " << int(r.iterations.size()) - 1 << ", final cost " << _cheshireDirectFinalCost
                         << " / " << r.finalCost << ", digests in " << std::hex << _cheshireDirectDigestIn << " / " << r.digestIn << " out "
                         << _cheshireDirectDigest << " / " << r.digest << std::dec);
}

}  // namespace sfm
}  // namespace aliceVision
