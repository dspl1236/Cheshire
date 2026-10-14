# Draft: upstream PR for the Meshroom #2344 crash (NOT POSTED)

*Target: alicevision/AliceVision, base `develop`, from the same fork as #2179 / #2181. The change
is generator step 5k (`scripts/apply_hip_patch.py`, search `CHESHIRE_SFM_PENDING_BA`) with the
environment gate and the `cheshire:` log prefixes removed; the code below is what the comment on
alicevision/Meshroom#2344 already shows. Every number traces to docs/04-validation.md lines
902-957 (replay and end-to-end runs) or to that comment. Needs the user's OK before anything is
pushed or opened.*

---

**Title:** [sfm] Sequential SfM: bundle-adjust the views resected since the last BA when the resection loop ends

## Description

`ReconstructionEngine_sequentialSfM::incrementalReconstruction` resects views in groups and runs a
bundle adjustment every `minimalResectionedViewsForBundle` resected views; a smaller group
`continue`s and its views accumulate as "reconstructed since the last BA". When `findNextBestViews`
finds no candidate above its score threshold the `while` loop exits with those views still
pending: never bundle-adjusted, never handed to `LocalBundleAdjustmentGraph::updateGraphWithNewViews`.
The next pass re-reads `prevReconstructedViews = getValidViews()`, so from then on they count as
old; they keep their resection pose and observe landmarks, so a later edge from a new view to one
of them reaches `_nodePerViewId.at()` on a view the graph never saw, and incremental SfM dies with
`[fatal] invalid map<K, T> key`.

Reproduced three times out of three on a public 884-photograph set (British Museum, False Door of
Ptahshepses; Sony A6000, 6000x3376), always at view 830-834, two seconds after
`Bundle adjustment start.`, with the 2023.3.0 binary; the same lines are in v3.3.0 and in `develop`
today. It needs a set where the candidate search runs dry mid-way, which is why small sets never
show it. Reported as alicevision/Meshroom#2344 (an 18,000-image set) and alicevision/Meshroom#2649.

## Features list

- [x] Fix alicevision/Meshroom#2344
- [x] Connect to alicevision/Meshroom#2649 (same error, Meshroom 2023.3.0, unresolved)

## Implementation remarks

Two changes, both in existing code paths:

1. `ReconstructionEngine_sequentialSfM.cpp`, after the `while (findNextBestViews(...))` loop and
   before the rig-calibration block: the views in `getValidViews()` that are not in
   `prevReconstructedViews` are triangulated and bundle-adjusted the way a full group is
   (`triangulate`, `bundleAdjustment`, `registerChanges`, the `potentials` union, `++_resectionId`).
   When the loop ended right after a BA the set is empty and nothing changes.
2. `LocalBundleAdjustmentGraph::updateGraphWithNewViews`: an edge whose endpoint has no node is
   skipped with a warning instead of thrown on. With change 1 this never fires; it turns any
   remaining case into a warning rather than a crash.

Verification on the 884-view set, `useLocalBA` on: the fixed binary on the same features and
matches completed with 830 poses and 1,354,686 landmarks, the pending-BA path running on three
passes (5, 7 and 9 views pending), the guard never; the same set through Meshroom end to end
completed with 833 poses and 1,355,362 landmarks (two passes, 5 and 6 pending; its matches were
byte-identical to the replay's, the difference is incremental SfM's run-to-run variation). Without
the fix, `useLocalBA=False` is a workaround (every BA global): 839 poses, 1,351,101 landmarks in
2436 s against 1636 s with the fix and local BA on. A 1678-view drone set with better coverage
never ended a pass with views pending and was unaffected.

```cpp
// ReconstructionEngine_sequentialSfM.cpp, after the while (findNextBestViews(...)) loop,
// before the rig-calibration block
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
// LocalBundleAdjustmentGraph.cpp, updateGraphWithNewViews, in place of the two .at() calls
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

🤖 Generated with [Claude Code](https://claude.com/claude-code)
