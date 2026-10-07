#!/usr/bin/env python3
"""Turn an aliceVision_depthMapEstimation log into a per-stage time breakdown. Run it with CHESHIRE_PROFILE_SGM=1,
which synchronises the stream at the end of each stage, so a stage's log span is its own GPU work. Fully synchronous
launches (AMD_SERIALIZE_KERNEL=3 / HIP_LAUNCH_BLOCKING=1 / CUDA_LAUNCH_BLOCKING=1) inflate the stages made of many small
launches: on a False Door chunk they made SGM's aggregation 56 s against 4.8 s (docs/04, step 11b).
usage: profile_log.py <log> [<log2> ...]"""
import re, sys, collections, datetime as dt
pat = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\.(\d{6})\]\[(\w+)\] (?:\((rc: \d+, tile: \d+/\d+)\) )?(.*)$")
def ts(m):
    h, mi, s, us = (int(m.group(i)) for i in range(1, 5)); return h * 3600 + mi * 60 + s + us / 1e6
for path in sys.argv[1:]:
    open_stage = {}      # (ctx, stage) -> t0
    acc = collections.Counter(); cnt = collections.Counter()
    first = last = None; prev_t = None; gaps = collections.Counter()
    for line in open(path, encoding="utf-8", errors="replace"):
        m = pat.match(line.rstrip())
        if not m: continue
        t = ts(m); ctx = m.group(6) or "global"; msg = m.group(7).strip()
        first = first if first is not None else t; last = t
        if msg.endswith(" done.") or msg.endswith(" done"):
            key = msg[: msg.rfind(" done")].rstrip(".")
            t0 = open_stage.pop((ctx, key), None)
            if t0 is not None: acc[key] += t - t0; cnt[key] += 1
        elif msg.endswith("."):
            open_stage[(ctx, msg.rstrip("."))] = t
        prev_t = t
    total = (last - first) if first is not None else 0
    print(f"== {path}: log span {total:.1f} s")
    print(f"{'stage':60s} {'total s':>9} {'calls':>6} {'avg ms':>9} {'share':>6}")
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1])[:16]:
        print(f"{k[:60]:60s} {v:9.2f} {cnt[k]:6d} {v/cnt[k]*1e3:9.1f} {v/total*100:5.1f}%")
    print(f"{'(sum of stages)':60s} {sum(acc.values()):9.2f} {'':6} {'':9} {sum(acc.values())/total*100:5.1f}%")
