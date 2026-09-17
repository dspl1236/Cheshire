// Cheshire: filterByPixSize made deterministic, over a tree of candidate points only.
// Textually included by fuseCut/PointCloud.cpp right after upstream's function (renamed
// filterByPixSizeUpstream by scripts/apply_hip_patch.py); it lives inside namespace
// aliceVision::fuseCut and sees everything that function sees.
//
// Upstream builds a nanoflann tree over every slot of the raw point array (61,154,352 on the
// 107-photo engine bay job, 20.4 s of the 42 s dense-cloud stage) and then asks, per point and in
// parallel, "is there a point within my radius whose score is smaller", writing pixSize = -1.0 into
// the very array the other threads' predicates are reading. Which of two mutual losers survives
// therefore depends on thread timing, and that race is the last source of run-to-run variation in
// Meshing's output.
//
// This version computes what upstream computes with one thread: the points are decided in index
// order, so a point loses to any smaller-scored neighbour that either comes later in the array (not
// yet processed, so still presenting its own score) or comes earlier and survived. That is not an
// approximation of upstream, it is upstream's loop at OMP_NUM_THREADS=1, and
// CHESHIRE_FILTER_CHECK=1 runs both in one process and requires them to agree on every point. The
// work is done in rounds that read only the previous round's decisions, so the answer does not
// depend on the thread count.
//
// The other half is the tree. Most slots are not points at all: the load loop leaves every block it
// discards at the value the arrays were constructed with, coordinate (0,0,0) with pixSize and
// simScore zero, and a discarded block is explicitly set to pixSize -1.0 with the same coordinate
// and score. Upstream keeps all of them in the tree, where they act as one very cheap point at the
// origin that underbids everything (score zero), and pays for the other 54 million copies of it in
// the build. Here they are replaced by a single sentinel at that coordinate carrying the smallest
// score any of them presents, which is exactly equivalent as long as they really do share one
// coordinate and their score cannot change while the pass runs. Both conditions are checked at
// runtime, and upstream's function runs unchanged if either fails.
//
// CHESHIRE_FILTER_OLD=1 keeps upstream's function. CHESHIRE_FILTER_CHECK=1 runs both in one process
// and compares every point. CHESHIRE_FILTER_LOG=1 prints the timing.
namespace {

// exactly the expression upstream's predicate compares: float simScore promoted, then two doubles
inline double cheshireFilterScore(const std::vector<float>& simScorePrepare, const std::vector<double>& pixSizePrepare, std::size_t i)
{
    return simScorePrepare[i] * pixSizePrepare[i] * pixSizePrepare[i];
}

// exactly upstream's radius expression (the margin coefficient multiplies first)
inline double cheshireFilterRadius(double pixSizeMarginCoef,
                                   const std::vector<float>& simScorePrepare,
                                   const std::vector<double>& pixSizePrepare,
                                   std::size_t i)
{
    return pixSizeMarginCoef * simScorePrepare[i] * pixSizePrepare[i] * pixSizePrepare[i];
}

const std::uint8_t kCheshireUndecided = 0;
const std::uint8_t kCheshireSurvive = 1;
const std::uint8_t kCheshireDrop = 2;

// nanoflann result set: mirrors SmallerPixSizeInRadius (fuseCut/Kdtree.hpp), with the live array
// reads replaced by a snapshot of the scores plus the previous round's decisions. A neighbour with
// a strictly smaller score drops this point when it sits later in the array (upstream would not
// have processed it yet, so it still presents its own score) or when it sits earlier and survived.
// The sentinel is given an index past every candidate, so it drops by the same rule.
class CheshireSmallerScoreInRadius
{
  public:
    using DistanceType = double;
    using IndexType = std::size_t;

    const DistanceType radius;

    CheshireSmallerScoreInRadius(DistanceType radius_,
                                 const std::vector<double>& score,
                                 const std::vector<std::uint8_t>& state,
                                 std::size_t self)
      : radius(radius_),
        _score(score),
        _state(state),
        _self(self),
        _selfScore(score[self])
    {}

    inline void init() { clear(); }
    inline void clear()
    {
        drop = false;
        pending = false;
    }
    inline std::size_t size() const { return 0; }
    // once the point is known to lose, the rest of the walk cannot change the answer
    inline bool full() const { return drop; }

    inline bool addPoint(DistanceType dist, IndexType index)
    {
        // upstream's comparison is strict, so equal scores never drop each other and a point never
        // drops itself
        if (dist < radius && _score[index] < _selfScore)
        {
            if (index > _self || _state[index] == kCheshireSurvive)
            {
                drop = true;
                return false;
            }
            if (_state[index] == kCheshireUndecided)
                pending = true;
        }
        return true;
    }

    inline DistanceType worstDist() const { return radius; }
    inline void sort() {}

    bool drop = false;
    bool pending = false;

  private:
    const std::vector<double>& _score;
    const std::vector<std::uint8_t>& _state;
    const std::size_t _self;
    const double _selfScore;
};

}  // namespace

void filterByPixSizeCheshire(const std::vector<Point3d>& verticesCoordsPrepare,
                             std::vector<double>& pixSizePrepare,
                             double pixSizeMarginCoef,
                             std::vector<float>& simScorePrepare)
{
    using Clock = std::chrono::steady_clock;
    const auto tStart = Clock::now();
    const bool logIt = std::getenv("CHESHIRE_FILTER_LOG") != nullptr || std::getenv("CHESHIRE_FILTER_CHECK") != nullptr;
    const int nbPoints = (int)verticesCoordsPrepare.size();
    const int nbThreads = std::max(1, omp_get_max_threads());

    // 1. classify every slot. A slot is a candidate when upstream would run a query for it. Every
    //    other slot still takes part as a neighbour, presenting a score that must not change while
    //    the pass runs: -1.0 slots present simScore for ever, and slots left at the constructed
    //    value present zero for ever. A slot with a positive but denormal radius would present its
    //    own score until upstream reached it and simScore afterwards, which no sentinel can model.
    std::vector<std::uint8_t> isCandidate(verticesCoordsPrepare.size(), 0);
    std::vector<double> minPresentedPerThread(nbThreads, std::numeric_limits<double>::max());
    std::vector<double> maxScorePerThread(nbThreads, -std::numeric_limits<double>::max());
    std::vector<double> minCandSimPerThread(nbThreads, std::numeric_limits<double>::max());
    std::vector<int> unmodellablePerThread(nbThreads, 0);
    std::vector<int> excludedPerThread(nbThreads, 0);
    std::vector<int> firstExcludedPerThread(nbThreads, -1);
#pragma omp parallel for
    for (int i = 0; i < nbPoints; ++i)
    {
        const int t = omp_get_thread_num();
        const bool invalid = pixSizePrepare[i] == -1.0;
        if (!invalid && !(cheshireFilterRadius(pixSizeMarginCoef, simScorePrepare, pixSizePrepare, i) < std::numeric_limits<double>::epsilon()))
        {
            isCandidate[i] = 1;
            maxScorePerThread[t] = std::max(maxScorePerThread[t], cheshireFilterScore(simScorePrepare, pixSizePrepare, i));
            minCandSimPerThread[t] = std::min(minCandSimPerThread[t], (double)simScorePrepare[i]);
            continue;
        }
        const bool constantScore = invalid || (pixSizePrepare[i] == 0.0 && simScorePrepare[i] == 0.0f);
        if (!constantScore)
            ++unmodellablePerThread[t];
        minPresentedPerThread[t] = std::min(minPresentedPerThread[t], invalid ? (double)simScorePrepare[i] : 0.0);
        ++excludedPerThread[t];
        if (firstExcludedPerThread[t] < 0)
            firstExcludedPerThread[t] = i;
    }
    double minPresented = std::numeric_limits<double>::max();
    double maxScore = -std::numeric_limits<double>::max();
    double minCandSim = std::numeric_limits<double>::max();
    int unmodellable = 0, firstExcluded = -1;
    std::size_t nbExcluded = 0;
    for (int t = 0; t < nbThreads; ++t)
    {
        minPresented = std::min(minPresented, minPresentedPerThread[t]);
        maxScore = std::max(maxScore, maxScorePerThread[t]);
        minCandSim = std::min(minCandSim, minCandSimPerThread[t]);
        unmodellable += unmodellablePerThread[t];
        nbExcluded += (std::size_t)excludedPerThread[t];
        if (firstExcludedPerThread[t] >= 0 && (firstExcluded < 0 || firstExcludedPerThread[t] < firstExcluded))
            firstExcluded = firstExcludedPerThread[t];
    }

    // every excluded slot must sit at the same coordinate for one sentinel to stand for all of them
    int scattered = 0;
    if (firstExcluded >= 0)
    {
        const Point3d ref = verticesCoordsPrepare[firstExcluded];
        std::vector<int> scatteredPerThread(nbThreads, 0);
#pragma omp parallel for
        for (int i = 0; i < nbPoints; ++i)
        {
            if (isCandidate[i] != 0)
                continue;
            const Point3d& p = verticesCoordsPrepare[i];
            if (p.x != ref.x || p.y != ref.y || p.z != ref.z)
                ++scatteredPerThread[omp_get_thread_num()];
        }
        for (int t = 0; t < nbThreads; ++t)
            scattered += scatteredPerThread[t];
    }

    std::vector<std::uint32_t> candIdx;
    candIdx.reserve(verticesCoordsPrepare.size() / 4 + 1);
    for (int i = 0; i < nbPoints; ++i)
        if (isCandidate[i] != 0)
            candIdx.push_back((std::uint32_t)i);
    const std::size_t nbCand = candIdx.size();
    const double msClassify = std::chrono::duration<double, std::milli>(Clock::now() - tStart).count();
    if (logIt)
        ALICEVISION_LOG_INFO("cheshire: filterByPixSize: " << nbCand << " candidates and " << nbExcluded << " excluded slots of "
                             << nbPoints << " (" << unmodellable << " unmodellable, " << scattered
                             << " away from the common coordinate; largest live score " << maxScore
                             << " below the smallest a dropped candidate presents " << minCandSim << "), classify "
                             << msClassify << " ms");

    // a candidate dropped during the pass presents simScore from then on, which this pass models as
    // inert; that is upstream's behaviour only while no candidate's simScore can underbid a live
    // score, so the two are measured and upstream runs if they ever meet
    const bool droppedStayInert = nbCand == 0 || maxScore < minCandSim;
    if (unmodellable > 0 || scattered > 0 || !droppedStayInert)
    {
        ALICEVISION_LOG_WARNING("cheshire: filterByPixSize falls back to upstream (" << unmodellable
                                << " slots whose presented score changes during the pass, " << scattered
                                << " excluded slots away from the common coordinate, largest live score " << maxScore
                                << " against the smallest score a dropped candidate presents " << minCandSim << ")");
        filterByPixSizeUpstream(verticesCoordsPrepare, pixSizePrepare, pixSizeMarginCoef, simScorePrepare);
        return;
    }
    if (nbCand == 0)
    {
        for (int i = 0; i < nbPoints; ++i)
            pixSizePrepare[i] = -1.0;
        ALICEVISION_LOG_INFO("Filtering done.");
        return;
    }

    // 2. the tree: the candidates, plus one sentinel for all the excluded slots
    const bool hasSentinel = nbExcluded > 0;
    const std::size_t nbTree = nbCand + (hasSentinel ? 1 : 0);
    std::vector<Point3d> treeCoords(nbTree);
    std::vector<double> score(nbTree);
    std::vector<double> radius(nbCand);
#pragma omp parallel for
    for (int k = 0; k < (int)nbCand; ++k)
    {
        const std::size_t i = candIdx[k];
        treeCoords[k] = verticesCoordsPrepare[i];
        score[k] = cheshireFilterScore(simScorePrepare, pixSizePrepare, i);
        radius[k] = cheshireFilterRadius(pixSizeMarginCoef, simScorePrepare, pixSizePrepare, i);
    }
    if (hasSentinel)
    {
        treeCoords[nbCand] = verticesCoordsPrepare[firstExcluded];
        score[nbCand] = minPresented;
    }
    ALICEVISION_LOG_INFO("Build nanoflann KdTree index.");
    PointVectorAdaptator pointCloudRef(treeCoords);
    KdTree kdTree(3 /*dim*/,
                  pointCloudRef,
                  nanoflann::KDTreeSingleIndexAdaptorParams(
                    MAX_LEAF_ELEMENTS, nanoflann::KDTreeSingleIndexAdaptorFlags::None, std::max(1u, std::thread::hardware_concurrency())));
    kdTree.buildIndex();
    ALICEVISION_LOG_INFO("KdTree created for " << nbTree << " points (cheshire: " << nbCand << " candidates of " << nbPoints << " slots).");
    const double msTree = std::chrono::duration<double, std::milli>(Clock::now() - tStart).count() - msClassify;

    // 3. decide in index order, in rounds that read only the previous round's decisions
    static const nanoflann::SearchParameters searchParams(0.f, false);  // false: dont need to sort
    std::vector<std::uint8_t> state(nbTree, kCheshireUndecided);
    if (hasSentinel)
        state[nbCand] = kCheshireSurvive;  // never a query, always presenting its score
    std::vector<std::uint8_t> next(state);
    std::vector<std::uint32_t> work(nbCand);
    for (std::size_t k = 0; k < nbCand; ++k)
        work[k] = (std::uint32_t)k;
    std::vector<std::uint32_t> stillOpen;
    int round = 0;
    std::string roundLog;
    while (!work.empty())
    {
        const int nbWork = (int)work.size();
#pragma omp parallel for
        for (int w = 0; w < nbWork; ++w)
        {
            const std::uint32_t k = work[w];
            CheshireSmallerScoreInRadius resultSet(radius[k], score, state, k);
            kdTree.findNeighbors(resultSet, treeCoords[k].m, searchParams);
            if (resultSet.drop)
                next[k] = kCheshireDrop;
            else if (!resultSet.pending)
                next[k] = kCheshireSurvive;
        }
        stillOpen.clear();
        for (int w = 0; w < nbWork; ++w)
        {
            const std::uint32_t k = work[w];
            state[k] = next[k];
            if (state[k] == kCheshireUndecided)
                stillOpen.push_back(k);
        }
        if (stillOpen.size() == work.size())
        {
            // cannot happen: the lowest undecided index always resolves
            ALICEVISION_LOG_WARNING("cheshire: filterByPixSize made no progress with " << stillOpen.size()
                                    << " points open, falling back to upstream");
            filterByPixSizeUpstream(verticesCoordsPrepare, pixSizePrepare, pixSizeMarginCoef, simScorePrepare);
            return;
        }
        if (logIt)
            roundLog += " " + std::to_string(work.size() - stillOpen.size());
        work.swap(stillOpen);
        ++round;
    }

    // 4. the decisions, written where upstream writes them
    std::size_t kept = 0;
#pragma omp parallel for
    for (int i = 0; i < nbPoints; ++i)
        if (isCandidate[i] == 0)
            pixSizePrepare[i] = -1.0;
    for (std::size_t k = 0; k < nbCand; ++k)
    {
        if (state[k] == kCheshireDrop)
            pixSizePrepare[candIdx[k]] = -1.0;
        else
            ++kept;
    }

    const double msTotal = std::chrono::duration<double, std::milli>(Clock::now() - tStart).count();
    if (logIt)
        ALICEVISION_LOG_INFO("cheshire: filterByPixSize: " << kept << " kept, " << round << " rounds (decided per round:" << roundLog
                             << "), sentinel score " << (hasSentinel ? minPresented : 0.0) << "; tree " << msTree << " ms, total "
                             << msTotal << " ms");
    ALICEVISION_LOG_INFO("Filtering done.");
}

void filterByPixSize(const std::vector<Point3d>& verticesCoordsPrepare,
                     std::vector<double>& pixSizePrepare,
                     double pixSizeMarginCoef,
                     std::vector<float>& simScorePrepare)
{
    if (std::getenv("CHESHIRE_FILTER_OLD") != nullptr)
    {
        filterByPixSizeUpstream(verticesCoordsPrepare, pixSizePrepare, pixSizeMarginCoef, simScorePrepare);
        return;
    }
    if (std::getenv("CHESHIRE_FILTER_CHECK") == nullptr)
    {
        filterByPixSizeCheshire(verticesCoordsPrepare, pixSizePrepare, pixSizeMarginCoef, simScorePrepare);
        return;
    }
    // upstream on one thread is the reference: its loop then runs in index order, which is what this
    // pass computes, so the two must agree on every point
    std::vector<double> upstreamPixSize(pixSizePrepare);
    const int savedThreads = omp_get_max_threads();
    omp_set_num_threads(1);
    filterByPixSizeUpstream(verticesCoordsPrepare, upstreamPixSize, pixSizeMarginCoef, simScorePrepare);
    omp_set_num_threads(savedThreads);
    filterByPixSizeCheshire(verticesCoordsPrepare, pixSizePrepare, pixSizeMarginCoef, simScorePrepare);
    std::size_t keptUp = 0, keptCh = 0, onlyUp = 0, onlyCh = 0;
    for (std::size_t i = 0; i < pixSizePrepare.size(); ++i)
    {
        const bool up = upstreamPixSize[i] != -1.0;
        const bool ch = pixSizePrepare[i] != -1.0;
        keptUp += up ? 1 : 0;
        keptCh += ch ? 1 : 0;
        onlyUp += (up && !ch) ? 1 : 0;
        onlyCh += (ch && !up) ? 1 : 0;
    }
    if (onlyUp == 0 && onlyCh == 0)
        ALICEVISION_LOG_INFO("cheshire: filterByPixSize check: identical to single-threaded upstream on all "
                             << pixSizePrepare.size() << " slots (" << keptCh << " kept)");
    else
        ALICEVISION_LOG_WARNING("cheshire: filterByPixSize check: upstream kept " << keptUp << ", cheshire kept " << keptCh
                                << "; only upstream " << onlyUp << ", only cheshire " << onlyCh);
}
