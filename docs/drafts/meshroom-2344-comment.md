# Draft: comment for alicevision/Meshroom issue #2344 ("map::at" crash in StructureFromMotion)

*Posted 2026-09-22: https://github.com/alicevision/Meshroom/issues/2344#issuecomment-5777656385

Original note: Review before posting; the diff is against AliceVision `develop` as vendored in
Cheshire (third_party/aliceVision, 2026 upstream) and applies to 2023.3 with the same lines.*

---

We hit this crash three times out of three on a public 884-photograph set (the British Museum's
False Door of Ptahshepses, Sony A6000, 6000x3376) with Meshroom 2023.3.0's own
`aliceVision_incrementalSfM`, always at view 830-834 of 884, always two seconds after
`Bundle adjustment start.`:

```
[19:28:42.911249][info] Bundle adjustment start.
[19:28:44.831934][fatal] invalid map<K, T> key
```

It is not memory (5 GB in use, 43 GB free on that box). The mechanism, from a `--verboseLevel debug`
run and the source:

1. `ReconstructionEngine_sequentialSfM::incrementalReconstruction` resects views in groups and runs
   a bundle adjustment every `minimalResectionedViewsForBundle` (10) resected views; a smaller group
   `continue`s and the views accumulate as "reconstructed since the last BA"
   (`newReconstructedViews` = `getValidViews()` minus `prevReconstructedViews`).
2. `findNextBestViews` returns `false` when no remaining candidate reaches its score threshold, and
   the `while` exits **with those accumulated views still pending**. Nothing after the loop
   bundle-adjusts them, and nothing hands them to `LocalBundleAdjustmentGraph::updateGraphWithNewViews`.
3. The outer `do { } while (nbValidPoses != getPoses().size())` pass then re-reads
   `prevReconstructedViews = getValidViews()`, so from then on they count as old. In the debug log,
   nine views resected at 650-656 in single-image groups never appear in the graph, and every later
   BA prints `The pose #... does not exist in the '_mapDistancePerPoseId'` for them.
4. They keep their resection pose and observe landmarks, so `getNewEdges` proposes an edge from a
   later new view to one of them, and
   `_graph.addEdge(_nodePerViewId.at(edge.first), _nodePerViewId.at(edge.second))` throws on a view
   that has no node.

It needs a set where the candidate search runs dry mid-way, which is why 107-photo sets never show
it and 884 did every time; a 1678-view drone set (Mill 19 Rubble) with better coverage went through
untouched. The 18,000-image report above is the same crash.

The fix we run (Cheshire, `scripts/apply_hip_patch.py` step 5k, commit 410258e): after the resection
loop, the views resected since the last bundle adjustment are triangulated and bundle-adjusted the
way a full group is, before the next pass; and `updateGraphWithNewViews` skips an edge whose endpoint
it does not hold (with a warning) instead of throwing. Everything the fix runs is existing code; the
change is that it runs.

```cpp
// ReconstructionEngine_sequentialSfM.cpp, after the `while (findNextBestViews(...))` loop,
// before the rig-calibration block:
{
    std::set<IndexT> pendingViews;
    const std::set<IndexT> reconstructedNow = _sfmData.getValidViews();
    std::set_difference(reconstructedNow.begin(), reconstructedNow.end(),
                        prevReconstructedViews.begin(), prevReconstructedViews.end(),
                        std::inserter(pendingViews, pendingViews.end()));
    if (!pendingViews.empty())
    {
        ALICEVISION_LOG_INFO(pendingViews.size() << " views resected since the last bundle adjustment were still pending when the resection loop ended; triangulating and bundle-adjusting them");
        triangulate(prevReconstructedViews, pendingViews);
        bundleAdjustment(pendingViews);
        prevReconstructedViews = _sfmData.getValidViews();
        registerChanges(linkedViewIds, pendingViews);
        std::set_union(potentials.begin(), potentials.end(), linkedViewIds.begin(), linkedViewIds.end(),
                       std::inserter(potentials, potentials.end()));
        ++_resectionId;
    }
}
```

```cpp
// LocalBundleAdjustmentGraph.cpp, updateGraphWithNewViews, in place of the two .at() calls:
std::size_t skipped = 0;
for (const Pair& edge : newEdges)
{
    const auto a = _nodePerViewId.find(edge.first);
    const auto b = _nodePerViewId.find(edge.second);
    if (a == _nodePerViewId.end() || b == _nodePerViewId.end()) { ++skipped; continue; }
    _graph.addEdge(a->second, b->second);
}
if (skipped != 0)
    ALICEVISION_LOG_WARNING("local BA graph: " << skipped << " edges to posed views the graph was never handed were skipped");
```

Verified: the fixed binary on the same features and matches, local BA on, completed with 833 poses
and 1,355,362 landmarks (the fix fired on two passes, 5 and 6 pending views; the guard never had to);
the same set through Meshroom end to end afterwards. Without the fix, `useLocalBA=False` is a
workaround (every BA global: 839 poses, 2436 s against 1636 s with the fix). Happy to open a PR if
that is useful.
