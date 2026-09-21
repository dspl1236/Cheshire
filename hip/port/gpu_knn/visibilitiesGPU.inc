// Cheshire: createVerticesWithVisibilities with the nearest-neighbour search on the GPU.
// Textually included by fuseCut/PointCloud.cpp right after upstream's function (renamed
// createVerticesWithVisibilitiesUpstream by scripts/apply_hip_patch.py); it lives inside
// namespace aliceVision::fuseCut and sees everything that function sees.
//
// Upstream: three cameras at a time, every valid depth-map pixel backprojected and looked up in a
// nanoflann tree over the point cloud, the nearest vertex voted for under a per-vertex lock. The
// tree reads the live coordinates while other threads move vertices (a benign race), so upstream's
// result depends on timing. Here the tree and the device see one snapshot of the coordinates, the
// device answers every query of a pass (gpu/knnGPU.hpp walks nanoflann's own tree exactly as
// nanoflann does), and the votes are applied camera by camera in pixel order, each vertex by one
// thread: the pass is deterministic. The depth and similarity maps are read three cameras ahead on
// worker threads; the device query of one camera overlaps the host's backprojection of the next
// and the votes of the previous. The similarity map is not read: upstream reads it, blurs it and
// computes a score per pixel that nothing uses.
//
// CHESHIRE_GPU_VIS=0 keeps upstream's code. CHESHIRE_GPU_VIS_CHECK=1 also answers every query on
// the host from the same snapshot tree and counts differences (there are none unless the host
// compiler contracts the metric differently; the form follows __FMA__, CHESHIRE_GPU_KNN_FMA=0|1 overrides).
// CHESHIRE_GPU_VIS_LOG=1 prints the timing.
#ifdef ALICEVISION_HAVE_GPU_FILTER
namespace {

bool cheshireGpuVisEnabled()
{
    static int v = -1;
    if (v < 0)
    {
        const char* e = std::getenv("CHESHIRE_GPU_VIS");
        v = (e != nullptr && *e == '0') ? 0 : (cheshire::knn::available() ? 1 : 0);
        // Announced on both paths, unconditionally: until 0.3.2 the only line on the GPU path was
        // behind CHESHIRE_GPU_VIS_LOG, so a default run could not show the port had run.
        if (v == 1)
            ALICEVISION_LOG_INFO("cheshire: visibility knn on the GPU (CHESHIRE_GPU_VIS=0 for nanoflann)");
        else if (e != nullptr && *e == '0')
            ALICEVISION_LOG_INFO("cheshire: visibility knn: disabled by CHESHIRE_GPU_VIS=0, nanoflann");
        else
            ALICEVISION_LOG_INFO("cheshire: visibility knn: no GPU device, nanoflann");
    }
    return v == 1;
}

double cheshireVisMs(std::chrono::steady_clock::time_point t)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

struct CheshireCamMaps
{
    image::Image<float> depthMap;
    bool ok = false;
};

CheshireCamMaps cheshireLoadCamMaps(int c, mvsUtils::MultiViewParams& mp)
{
    CheshireCamMaps m;
    mvsUtils::readMap(c, mp, mvsUtils::EFileType::depthMapFiltered, m.depthMap);
    m.ok = m.depthMap.size() > 0;
    return m;
}

// one camera's queries: pinned buffers the device reads and writes asynchronously
struct CheshireQuerySlot
{
    double* q = nullptr;           // 3 doubles per query
    std::uint32_t* nn = nullptr;   // nearest vertex
    double* nnDist = nullptr;      // its squared distance
    std::vector<double> pixSize;
    std::size_t capacity = 0;
    std::size_t nq = 0;
    int cam = -1;
    bool empty = true;
    bool onHost = false;     // this camera's buffers are the host vectors below (map larger than mp said)
    bool queued = false;     // a device query for this camera is in flight or answered
    std::vector<double> qHost;
    std::vector<std::uint32_t> nnHost;
    std::vector<double> nnDistHost;
    double* qp = nullptr;
    std::uint32_t* nnp = nullptr;
    double* nnDistp = nullptr;

    void free()
    {
        cheshire::knn::hostFree(q);
        cheshire::knn::hostFree(nn);
        cheshire::knn::hostFree(nnDist);
        q = nullptr;
        nn = nullptr;
        nnDist = nullptr;
        capacity = 0;
    }
    bool reserve(std::size_t n)
    {
        if (n <= capacity) return true;
        free();
        q = (double*)cheshire::knn::hostAlloc(n * 3 * sizeof(double));
        nn = (std::uint32_t*)cheshire::knn::hostAlloc(n * sizeof(std::uint32_t));
        nnDist = (double*)cheshire::knn::hostAlloc(n * sizeof(double));
        pixSize.resize(n);
        if (q == nullptr || nn == nullptr || nnDist == nullptr)
        {
            free();
            return false;
        }
        capacity = n;
        return true;
    }
    ~CheshireQuerySlot() { free(); }
};

}  // namespace

void createVerticesWithVisibilitiesGPU(const StaticVector<int>& cams,
                                       std::vector<Point3d>& verticesCoordsPrepare,
                                       std::vector<double>& pixSizePrepare,
                                       std::vector<float>& simScorePrepare,
                                       std::vector<GC_vertexInfo>& verticesAttrPrepare,
                                       mvsUtils::MultiViewParams& mp,
                                       float simFactor,
                                       float voteMarginFactor,
                                       float contributeMarginFactor,
                                       float simGaussianSize)
{
    using Clock = std::chrono::steady_clock;
    const auto tStart = Clock::now();
    const bool check = std::getenv("CHESHIRE_GPU_VIS_CHECK") != nullptr;
    const bool logIt = check || std::getenv("CHESHIRE_GPU_VIS_LOG") != nullptr;
    // the metric's form follows this host build: a compiler allowed to emit FMA (clang-cl under
    // /arch:AVX2, gcc with -mfma) contracts nanoflann's `result += diff * diff`; a generic x86-64
    // build (the Linux bundle) cannot. CHESHIRE_GPU_KNN_FMA=0|1 overrides.
#ifdef __FMA__
    const bool fmaDefault = true;
#else
    const bool fmaDefault = false;
#endif
    const char* fmaEnv = std::getenv("CHESHIRE_GPU_KNN_FMA");
    const bool fma = (fmaEnv != nullptr && *fmaEnv != '\0') ? (*fmaEnv != '0') : fmaDefault;

    static_assert(sizeof(Point3d) == 3 * sizeof(double), "Point3d is three contiguous doubles");
    const std::vector<Point3d> snapshot(verticesCoordsPrepare);
    PointVectorAdaptator pointCloudRef(snapshot);
    KdTree kdTree(3 /*dim*/, pointCloudRef, nanoflann::KDTreeSingleIndexAdaptorParams(MAX_LEAF_ELEMENTS, nanoflann::KDTreeSingleIndexAdaptorFlags::None, std::max(1u, std::thread::hardware_concurrency())));
    kdTree.buildIndex();
    ALICEVISION_LOG_INFO("NANOFLANN: KdTree created.");
    const double msTree = cheshireVisMs(tStart);

    // flatten nanoflann's tree: node 0 is the root, children get the next two slots
    std::vector<cheshire::knn::Node> nodes;
    nodes.reserve(2 * snapshot.size() / MAX_LEAF_ELEMENTS + 16);
    if (!snapshot.empty())
    {
        std::vector<std::pair<const KdTree::Node*, std::size_t>> stack;
        nodes.emplace_back();
        stack.emplace_back(kdTree.root_node_, 0);
        while (!stack.empty())
        {
            const KdTree::Node* n = stack.back().first;
            const std::size_t slot = stack.back().second;
            stack.pop_back();
            cheshire::knn::Node out;
            if (n->child1 == nullptr)
            {
                out.a = (std::uint32_t)n->node_type.lr.left;
                out.b = (std::uint32_t)n->node_type.lr.right;
            }
            else
            {
                out.a = (std::uint32_t)n->node_type.sub.divfeat;
                out.lo = n->node_type.sub.divlow;
                out.hi = n->node_type.sub.divhigh;
                out.child1 = (std::int32_t)nodes.size();
                nodes.emplace_back();
                out.child2 = (std::int32_t)nodes.size();
                nodes.emplace_back();
                stack.emplace_back(n->child1, (std::size_t)out.child1);
                stack.emplace_back(n->child2, (std::size_t)out.child2);
            }
            nodes[slot] = out;
        }
    }
    std::vector<std::uint32_t> perm(kdTree.vAcc_.begin(), kdTree.vAcc_.end());
    double lo[3], hi[3];
    for (int i = 0; i < 3; ++i)
    {
        lo[i] = kdTree.root_bbox_[i].low;
        hi[i] = kdTree.root_bbox_[i].high;
    }

    // the largest map decides the buffer sizes (grown later if a map is larger than mp says)
    std::size_t maxPix = 1;
    for (int c = 0; c < cams.size(); ++c)
        maxPix = std::max(maxPix, (std::size_t)mp.getWidth(c) * (std::size_t)mp.getHeight(c));

    cheshire::knn::Index index;
    CheshireQuerySlot slots[2];
    bool ready = !snapshot.empty()
                 && index.build(reinterpret_cast<const double*>(snapshot.data()), snapshot.size(), nodes.data(), nodes.size(), perm.data(), lo, hi, maxPix)
                 && slots[0].reserve(maxPix) && slots[1].reserve(maxPix);
    if (!ready)
    {
        ALICEVISION_LOG_WARNING("cheshire: GPU knn index not built, visibilities on the CPU");
        createVerticesWithVisibilitiesUpstream(cams, verticesCoordsPrepare, pixSizePrepare, simScorePrepare, verticesAttrPrepare, mp, simFactor, voteMarginFactor, contributeMarginFactor, simGaussianSize);
        return;
    }
    const double msIndex = cheshireVisMs(tStart);
    if (logIt)
        ALICEVISION_LOG_INFO("cheshire: GPU knn index: " << snapshot.size() << " points, " << nodes.size() << " nodes, tree " << msTree << " ms, index " << msIndex << " ms");

    // nanoflann's answer on the host, from the same snapshot tree
    auto hostSearch = [&](const double* q, std::uint32_t& idx, double& d) {
        nanoflann::KNNResultSet<double, std::size_t> resultSet(1);
        std::size_t i = std::numeric_limits<std::size_t>::max();
        double dd = std::numeric_limits<double>::max();
        resultSet.init(&i, &dd);
        kdTree.findNeighbors(resultSet, q, nanoflann::SearchParameters());
        idx = (std::uint32_t)i;
        d = dd;
    };

    // the maps of the next cameras are read while this one is processed
    const int ahead = 3;
    std::vector<std::future<CheshireCamMaps>> pending(cams.size());
    auto launch = [&](int c) {
        if (c < cams.size())
            pending[c] = std::async(std::launch::async, cheshireLoadCamMaps, c, std::ref(mp));
    };
    for (int c = 0; c < ahead; ++c)
        launch(c);

    double msLoad = 0, msPrep = 0, msWait = 0, msVotes = 0, msCheck = 0;
    std::size_t nbQueries = 0, nbIdxDiff = 0, nbDistDiff = 0, nbHost = 0, nbVotes = 0;
    std::vector<std::size_t> rowStart;

    // the valid pixels of camera c, in pixel order, backprojected on the host into a slot
    auto prepare = [&](int c, CheshireQuerySlot& s) {
        s.cam = c;
        s.nq = 0;
        s.empty = true;
        s.queued = false;
        auto t1 = Clock::now();
        CheshireCamMaps maps = pending[c].get();
        launch(c + ahead);
        msLoad += cheshireVisMs(t1);
        if (!maps.ok)
        {
            ALICEVISION_LOG_WARNING("Empty depth map (cam id: " << c << ")");
            return;
        }
        const image::Image<float>& depthMap = maps.depthMap;
        const int w = depthMap.width();
        const int h = depthMap.height();
        const std::size_t pixels = (std::size_t)w * (std::size_t)h;
        s.onHost = pixels > s.capacity;
        if (s.onHost)
        {
            // a map larger than mp announced: this camera is answered on the host
            s.qHost.resize(3 * pixels);
            s.nnHost.resize(pixels);
            s.nnDistHost.resize(pixels);
            s.pixSize.resize(pixels);
            s.qp = s.qHost.data();
            s.nnp = s.nnHost.data();
            s.nnDistp = s.nnDistHost.data();
        }
        else
        {
            s.qp = s.q;
            s.nnp = s.nn;
            s.nnDistp = s.nnDist;
        }
        t1 = Clock::now();
        rowStart.assign(h + 1, 0);
#pragma omp parallel for
        for (int y = 0; y < h; ++y)
        {
            std::size_t n = 0;
            for (int x = 0; x < w; ++x)
                if (depthMap(y * w + x) > 0.0f)
                    ++n;
            rowStart[y + 1] = n;
        }
        for (int y = 0; y < h; ++y)
            rowStart[y + 1] += rowStart[y];
        double* q = s.qp;
#pragma omp parallel for
        for (int y = 0; y < h; ++y)
        {
            std::size_t k = rowStart[y];
            for (int x = 0; x < w; ++x)
            {
                const std::size_t pix = y * w + x;
                const float depth = depthMap(pix);
                if (depth <= 0.0f)
                    continue;
                const Point3d p = mp.backproject(c, Point2d(x, y), depth);
                q[3 * k] = p.m[0];
                q[3 * k + 1] = p.m[1];
                q[3 * k + 2] = p.m[2];
                s.pixSize[k] = mp.getCamPixelSize(p, c);
                ++k;
            }
        }
        s.nq = rowStart[h];
        s.empty = false;
        msPrep += cheshireVisMs(t1);
    };

    // nanoflann on the host for a whole camera (the device unavailable, or a map larger than mp said)
    auto answerOnHost = [&](CheshireQuerySlot& s) {
#pragma omp parallel for
        for (long long k = 0; k < (long long)s.nq; ++k)
            hostSearch(&s.qp[3 * k], s.nnp[k], s.nnDistp[k]);
        nbHost += s.nq;
    };

    // the votes of one answered camera, in pixel order, each vertex owned by one thread;
    // pixSizeScoreV is upstream's expression evaluated once per vertex instead of once per pixel
    const std::size_t nbVertices = verticesCoordsPrepare.size();
    std::vector<float> scoreV(nbVertices);
#pragma omp parallel for
    for (long long v = 0; v < (long long)nbVertices; ++v)
        scoreV[v] = simScorePrepare[v] * pixSizePrepare[v] * pixSizePrepare[v];
    auto votes = [&](const CheshireQuerySlot& s) {
        const int c = s.cam;
        const std::size_t nq = s.nq;
        const std::uint32_t* nn = s.nnp;
        const double* nnDist = s.nnDistp;
        const double* q = s.qp;
        std::size_t total = 0;
#pragma omp parallel reduction(+ : total)
        {
            const int nt = omp_get_num_threads();
            const int tid = omp_get_thread_num();
            const std::size_t range = (nbVertices + nt - 1) / nt;
            const std::size_t vLo = (std::size_t)tid * range;
            const std::size_t vHi = std::min(nbVertices, vLo + range);
            for (std::size_t k = 0; k < nq; ++k)
            {
                const std::size_t nearestVertexIndex = nn[k];
                if (nearestVertexIndex < vLo || nearestVertexIndex >= vHi)
                    continue;
                const double dist = nnDist[k];
                const double pixSize = s.pixSize[k];
                const float pixSizeScoreI = simScorePrepare[nearestVertexIndex] * pixSize * pixSize;
                const float pixSizeScoreV = scoreV[nearestVertexIndex];
                if (dist < voteMarginFactor * std::max(pixSizeScoreI, pixSizeScoreV))
                {
                    GC_vertexInfo& va = verticesAttrPrepare[nearestVertexIndex];
                    Point3d& vc = verticesCoordsPrepare[nearestVertexIndex];
                    va.cams.push_back_distinct(c);
                    ++total;
                    if (dist < contributeMarginFactor * pixSizeScoreV)
                    {
                        const Point3d p(q[3 * k], q[3 * k + 1], q[3 * k + 2]);
                        vc = (vc * (double)va.nrc + p) / double(va.nrc + 1);
                        va.nrc += 1;
                    }
                }
            }
        }
        nbVotes += total;
    };

    // the device answers camera c while the host prepares c + 1, then votes c while it answers c + 1;
    // after a device error the remaining cameras are answered on the host from the same tree
    bool deviceDead = false;
    auto enqueue = [&](CheshireQuerySlot& s) {
        if (s.empty || s.onHost || deviceDead) return;
        if (index.queryAsync(s.qp, s.nq, s.nnp, s.nnDistp, fma))
            s.queued = true;
        else
        {
            ALICEVISION_LOG_WARNING("cheshire: GPU knn query failed at camera " << s.cam << ", the rest of the pass on the host");
            deviceDead = true;
        }
    };
    if (cams.size() > 0)
    {
        prepare(0, slots[0]);
        enqueue(slots[0]);
    }
    for (int c = 0; c < cams.size(); ++c)
    {
        ALICEVISION_LOG_INFO("Create visibilities (" << c << "/" << cams.size() << ")");
        CheshireQuerySlot& cur = slots[c % 2];
        CheshireQuerySlot& next = slots[(c + 1) % 2];
        if (c + 1 < cams.size())
            prepare(c + 1, next);
        auto t1 = Clock::now();
        bool answered = false;
        if (cur.queued)
        {
            if (index.wait())
                answered = true;
            else
            {
                ALICEVISION_LOG_WARNING("cheshire: GPU knn failed at camera " << c << ", the rest of the pass on the host");
                deviceDead = true;
            }
        }
        msWait += cheshireVisMs(t1);
        if (c + 1 < cams.size())
            enqueue(next);
        if (cur.empty)
            continue;
        if (!answered)
            answerOnHost(cur);
        else if (index.overflowed() > 0)
        {
            for (std::size_t k = 0; k < cur.nq; ++k)
                if (cur.nnp[k] == 0xFFFFFFFFu)
                {
                    hostSearch(&cur.qp[3 * k], cur.nnp[k], cur.nnDistp[k]);
                    ++nbHost;
                }
        }
        if (check)
        {
            t1 = Clock::now();
            long long idxDiff = 0, distDiff = 0;
#pragma omp parallel for reduction(+ : idxDiff, distDiff)
            for (long long k = 0; k < (long long)cur.nq; ++k)
            {
                std::uint32_t i;
                double d;
                hostSearch(&cur.qp[3 * k], i, d);
                if (i != cur.nnp[k])
                    ++idxDiff;
                else if (d != cur.nnDistp[k])
                    ++distDiff;
            }
            nbIdxDiff += idxDiff;
            nbDistDiff += distDiff;
            msCheck += cheshireVisMs(t1);
        }
        t1 = Clock::now();
        votes(cur);
        msVotes += cheshireVisMs(t1);
        nbQueries += cur.nq;
    }
    index.wait();
    const double msDevUp = index.msUpload(), msDevKernel = index.msKernel(), msDevDown = index.msDownload();
    index.release();

// compute pixSize
#pragma omp parallel for
    for (int vi = 0; vi < verticesAttrPrepare.size(); ++vi)
    {
        GC_vertexInfo& v = verticesAttrPrepare[vi];
        v.pixSize = mp.getCamsMinPixelSize(verticesCoordsPrepare[vi], v.cams);
    }

    if (logIt)
        ALICEVISION_LOG_INFO("cheshire: visibilities on the GPU: " << nbQueries << " queries, " << nbVotes << " votes, " << nbHost
                             << " answered on the host; index " << msIndex << " ms, maps " << msLoad << " ms, backproject " << msPrep
                             << " ms, device wait " << msWait << " ms (device: upload " << msDevUp << ", kernel " << msDevKernel << ", download "
                             << msDevDown << " ms), votes " << msVotes << " ms, check " << msCheck << " ms, total " << cheshireVisMs(tStart) << " ms");
    if (check)
    {
        if (nbIdxDiff + nbDistDiff == 0)
            ALICEVISION_LOG_INFO("cheshire: GPU knn check: identical to nanoflann on all " << nbQueries << " queries");
        else
            ALICEVISION_LOG_WARNING("cheshire: GPU knn check: " << nbIdxDiff << " of " << nbQueries << " queries name a different vertex, "
                                    << nbDistDiff << " a different distance");
    }
    ALICEVISION_LOG_INFO("Visibilities created.");
}
#endif

void createVerticesWithVisibilities(const StaticVector<int>& cams,
                                    std::vector<Point3d>& verticesCoordsPrepare,
                                    std::vector<double>& pixSizePrepare,
                                    std::vector<float>& simScorePrepare,
                                    std::vector<GC_vertexInfo>& verticesAttrPrepare,
                                    mvsUtils::MultiViewParams& mp,
                                    float simFactor,
                                    float voteMarginFactor,
                                    float contributeMarginFactor,
                                    float simGaussianSize)
{
#ifdef ALICEVISION_HAVE_GPU_FILTER
    if (cheshireGpuVisEnabled())
    {
        createVerticesWithVisibilitiesGPU(cams, verticesCoordsPrepare, pixSizePrepare, simScorePrepare, verticesAttrPrepare, mp, simFactor, voteMarginFactor, contributeMarginFactor, simGaussianSize);
        return;
    }
#endif
    createVerticesWithVisibilitiesUpstream(cams, verticesCoordsPrepare, pixSizePrepare, simScorePrepare, verticesAttrPrepare, mp, simFactor, voteMarginFactor, contributeMarginFactor, simGaussianSize);
}
