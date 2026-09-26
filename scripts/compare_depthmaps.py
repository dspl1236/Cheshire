#!/usr/bin/env python3
"""Compare two AliceVision DepthMap output folders (e.g. CUDA reference vs HIP port).

usage: compare_depthmaps.py <refDir> <testDir> [--png outDir] [--mode strict|exact]

For every <viewId>_depthMap.exr / <viewId>_simMap.exr pair present in both folders:
  * valid-pixel masks (depth > 0) and their agreement
  * abs / relative depth differences on jointly-valid pixels (mean, median, p95, max)
  * fraction of pixels within 0.5 % / 1 % / 5 % relative depth
  * simMap mean abs difference
  * decoded pixels whose bits differ, over the whole map (depth and sim; NaN included)
Optionally writes side-by-side PNGs (ref | test | abs-diff) for eyeballing.
--mode strict (default): exit 0 when every view has >= 98 % of jointly-valid pixels within 1 % depth
and mask agreement >= 95 %. --mode exact: exit 0 only when every decoded depth and sim pixel of every
view is bit-identical (the table's rounded columns print 0.0000 for anything under 5e-5, so they
cannot say "identical"; the difference counts can).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import OpenEXR


def read_exr(path: Path) -> np.ndarray:
    with OpenEXR.File(str(path)) as f:
        chans = f.channels()
        name = "Y" if "Y" in chans else next(iter(chans))
        return np.asarray(chans[name].pixels, dtype=np.float32)


def to_png(arr: np.ndarray, path: Path, vmin: float, vmax: float) -> None:
    from PIL import Image
    a = np.clip((arr - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    Image.fromarray((a * 255).astype(np.uint8)).save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("test"); ap.add_argument("--png", default=None)
    ap.add_argument("--tol", type=float, default=0.01, help="relative depth tolerance for the pass criterion")
    ap.add_argument("--mode", choices=("strict", "exact"), default="strict", help="strict: the 98 %% / 1 %% bar; exact: every decoded pixel bit-identical")
    a = ap.parse_args()
    ref, test = Path(a.ref), Path(a.test)
    views = sorted(p.name.split("_")[0] for p in ref.glob("*_depthMap.exr"))
    if not views:
        print("no *_depthMap.exr in", ref); return 2
    ok_all = True
    exact_all = True
    diff_depth_total = 0
    diff_sim_total = 0
    rows = []  # for CSV / markdown export next to the test outputs
    print(f"{'view':>11} {'valid ref':>9} {'valid tst':>9} {'agree':>6} {'meanRel':>8} {'medRel':>8} {'p95Rel':>8} {'<0.5%':>6} {'<1%':>6} {'<5%':>6} {'simMAD':>7} {'depthDiff':>9} {'simDiff':>9}")
    for v in views:
        rd, td = ref / f"{v}_depthMap.exr", test / f"{v}_depthMap.exr"
        if not td.exists():
            print(f"{v:>11}  missing in test"); ok_all = False; continue
        R, T = read_exr(rd), read_exr(td)
        if R.shape != T.shape:
            print(f"{v:>11}  shape mismatch ref{R.shape} test{T.shape}"); ok_all = False; exact_all = False; continue
        # bit-level differences over every decoded pixel (the exact verdict)
        diff_depth = int(np.count_nonzero(R.view(np.uint32) != T.view(np.uint32)))
        diff_sim = 0
        rs_, ts_ = ref / f"{v}_simMap.exr", test / f"{v}_simMap.exr"
        if rs_.exists() != ts_.exists():
            diff_sim = -1  # present on one side only
        elif rs_.exists():
            S1_, S2_ = read_exr(rs_), read_exr(ts_)
            diff_sim = int(np.count_nonzero(S1_.view(np.uint32) != S2_.view(np.uint32))) if S1_.shape == S2_.shape else -1
        diff_depth_total += diff_depth
        diff_sim_total += max(diff_sim, 0)
        if diff_depth != 0 or diff_sim != 0:
            exact_all = False
        vr, vt = R > 0, T > 0
        both = vr & vt
        agree = (vr == vt).mean()
        if both.sum() == 0:
            print(f"{v:>11}  no jointly valid pixels (depthDiff {diff_depth}, simDiff {diff_sim})"); ok_all = False; continue
        rel = np.abs(T[both] - R[both]) / np.maximum(np.abs(R[both]), 1e-6)
        f05, f1, f5 = (rel < 0.005).mean(), (rel < 0.01).mean(), (rel < 0.05).mean()
        sim = ""
        rs, ts = ref / f"{v}_simMap.exr", test / f"{v}_simMap.exr"
        if rs.exists() and ts.exists():
            S1, S2 = read_exr(rs), read_exr(ts)
            sim = f"{np.abs(S1[both] - S2[both]).mean():7.4f}"
        print(f"{v:>11} {vr.mean():9.3f} {vt.mean():9.3f} {agree:6.3f} {rel.mean():8.4f} {np.median(rel):8.4f} {np.percentile(rel, 95):8.4f} {f05:6.3f} {f1:6.3f} {f5:6.3f} {sim:>7} {diff_depth:9d} {diff_sim:9d}")
        rows.append(dict(view=v, valid_ref=round(float(vr.mean()), 4), valid_test=round(float(vt.mean()), 4), mask_agree=round(float(agree), 4),
                         mean_rel=round(float(rel.mean()), 5), median_rel=round(float(np.median(rel)), 5), p95_rel=round(float(np.percentile(rel, 95)), 5),
                         within_0p5pct=round(float(f05), 4), within_1pct=round(float(f1), 4), within_5pct=round(float(f5), 4), sim_mad=sim.strip(),
                         pixels_differing_depth=diff_depth, pixels_differing_sim=diff_sim))
        if (rel < a.tol).mean() < 0.98 or agree < 0.95:
            ok_all = False
        if a.png:
            out = Path(a.png); out.mkdir(parents=True, exist_ok=True)
            vmin, vmax = np.percentile(R[vr], 2), np.percentile(R[vr], 98)
            D = np.zeros_like(R); D[both] = np.abs(T[both] - R[both])
            panel = np.concatenate([np.where(vr, R, vmin), np.where(vt, T, vmin), D * (vmax - vmin) / max(D.max(), 1e-9) + vmin], axis=1)
            to_png(panel, out / f"{v}.png", vmin, vmax)
    # export machine-readable + markdown copies of the table next to the test outputs
    import csv, json
    if rows:
        with open(test / "compare_stats.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        med = lambda k: float(np.median([r[k] for r in rows]))
        summary = dict(views=len(rows), views_mask_agree_ge_0p95=sum(r["mask_agree"] >= 0.95 for r in rows),
                       median_of_view_median_rel=med("median_rel"), median_of_view_within_1pct=med("within_1pct"),
                       min_within_1pct=min(r["within_1pct"] for r in rows), max_p95_rel=max(r["p95_rel"] for r in rows),
                       strict_pass=bool(ok_all), criterion="every view: >=98% of jointly-valid pixels within 1% rel. depth and mask agreement >=95%",
                       pixels_differing_depth=diff_depth_total, pixels_differing_sim=diff_sim_total, exact_identical=bool(exact_all and len(rows) == len(views)),
                       mode=a.mode)
        (test / "compare_summary.json").write_text(json.dumps(summary, indent=1))
        with open(test / "compare_stats.md", "w") as f:
            f.write("| view | valid ref | valid test | mask agree | mean rel | median rel | p95 rel | <0.5% | <1% | <5% | simMAD |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
            for r in rows:
                f.write(f"| {r['view']} | {r['valid_ref']:.3f} | {r['valid_test']:.3f} | {r['mask_agree']:.3f} | {r['mean_rel']:.4f} | {r['median_rel']:.4f} | {r['p95_rel']:.4f} | {r['within_0p5pct']:.3f} | {r['within_1pct']:.3f} | {r['within_5pct']:.3f} | {r['sim_mad']} |\n")
    exact_ok = exact_all and len(rows) == len(views)
    print(f"decoded pixels differing: depth {diff_depth_total}, sim {diff_sim_total} over {len(views)} views ({'bit-identical' if exact_ok else 'not identical'})")
    passed = exact_ok if a.mode == "exact" else ok_all
    print("RESULT:", "PASS" if passed else "FAIL", f"({a.mode})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
