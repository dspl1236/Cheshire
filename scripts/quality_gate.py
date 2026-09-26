#!/usr/bin/env python3
"""Reconstruction-quality gate: does a change make incremental SfM worse than its own run-to-run spread?

  sfm <set> --leg NAME[:KEY=VALUE,...] --leg NAME[:...] [--repeat N] [--prefix P] [tolerances]
        runs incrementalSfM N times per leg on scripts/sfmbench.py's fixed features-and-matches cache,
        legs interleaved (A B A B ...) so drift on the box falls on both, then reports
  report <set> --leg NAME=TAGPREFIX --leg NAME=TAGPREFIX [tolerances]
        reports on runs already in build/sfmbench/<set>/runs (tags TAGPREFIX-1, -2, ...)

  set: 41 | eb | mini6 | fd, as scripts/sfmbench.py (prepare the cache with it first)

Why: no gate measured reconstruction quality before 0.3.5 - the end-to-end harness checks outputs,
provenance, port markers and self-check verdicts. Three defaults wait on this (analytic Jacobians,
the QR nullspace, bundle adjustment on the device), and docs/17 showed how easily a real difference
hides: at n=3 per leg the QR landmark loss sat "inside the spread", at n=10 it was p 0.0008. So the
default here is n=10 per leg, and the verdict is statistical.

What is compared, the first leg being the baseline:
  poses, landmarks, RMSE   from the incrementalSfM log, per run; Welch's t (unequal variances) and a
                           two-sided p, the mean difference and its percentage of the baseline mean
  pose agreement           every pair of runs aligned by a similarity (Umeyama, on the camera centres
                           of the views both placed): the residual RMS over the spread of the centres,
                           and the median rotation difference; cross-leg pairs against the pairs within
                           the baseline, which are the noise floor. With no ground truth this says
                           "different", not "worse", so it flags only past an absolute floor as well
Verdict per metric: FAIL when the candidate is worse with p < --alpha by more than the tolerance,
WARN when it is significantly worse but within it, PASS otherwise (a significant improvement is
reported as such). Tolerances are policy, so they are flags; the defaults are this file's proposal.
A report goes to build/quality/<set>-<prefix>.md and .json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfmbench  # noqa: E402

OUT = sfmbench.B / "quality"


# ------------------------------------------------------------------------------------------ statistics
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (Numerical Recipes, betacf)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_critical(df: float, alpha: float) -> float:
    """Two-sided critical t for df degrees of freedom, by bisection on welch()'s p."""
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _betai(df / 2.0, 0.5, df / (df + mid * mid)) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def welch(a: list[float], b: list[float]) -> dict:
    """Welch's t-test, two-sided. b minus a."""
    na, nb = len(a), len(b)
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va = float(np.var(a, ddof=1)) if na > 1 else 0.0
    vb = float(np.var(b, ddof=1)) if nb > 1 else 0.0
    se2 = va / na + vb / nb
    if se2 == 0.0:
        return {"diff": mb - ma, "t": 0.0 if mb == ma else math.copysign(math.inf, mb - ma), "df": na + nb - 2, "p": 1.0 if mb == ma else 0.0, "se": 0.0}
    t = (mb - ma) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / max(na - 1, 1) + (vb / nb) ** 2 / max(nb - 1, 1))
    p = _betai(df / 2.0, 0.5, df / (df + t * t))
    return {"diff": mb - ma, "t": t, "df": df, "p": p, "se": math.sqrt(se2)}


# ------------------------------------------------------------------------------------------ geometry
def load_cameras(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """viewId -> (world-to-camera rotation, centre) from the run's cameras.sfm (AliceVision JSON)."""
    d = json.loads((run_dir / "cameras.sfm").read_text(encoding="utf-8"))
    poses = {}
    for p in d.get("poses", []):
        tr = p["pose"]["transform"]
        poses[str(p["poseId"])] = (np.array([float(v) for v in tr["rotation"]]).reshape(3, 3),
                                   np.array([float(v) for v in tr["center"]]))
    return {str(v["viewId"]): poses[str(v["poseId"])] for v in d.get("views", []) if str(v.get("poseId")) in poses}


def align(src: dict, dst: dict) -> dict | None:
    """Similarity src -> dst on the shared views' centres (Umeyama); the disagreement left over."""
    common = sorted(set(src) & set(dst))
    if len(common) < 3:
        return None
    P = np.array([src[k][1] for k in common])
    Q = np.array([dst[k][1] for k in common])
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, D, Vt = np.linalg.svd(Y.T @ X / len(common))
    S = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U) * np.linalg.det(Vt))) or 1.0])
    Ra = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (X ** 2).sum(1).mean())
    res = Y - s * (X @ Ra.T)
    spread = math.sqrt(float((Y ** 2).sum(1).mean()))
    # x_cam = R (x_world - C): under x' = s Ra x + t the rotation becomes R Ra^T
    ang = []
    for k in common:
        Rs, Rd = src[k][0] @ Ra.T, dst[k][0]
        c = (np.trace(Rd.T @ Rs) - 1.0) / 2.0
        ang.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    return {"views": len(common), "centre_rms_rel": math.sqrt(float((res ** 2).sum(1).mean())) / spread,
            "rot_median_deg": float(np.median(ang))}


# ------------------------------------------------------------------------------------------ gate
METRICS = [  # name, worse direction (-1: lower is worse), tolerance flag
    ("poses", -1, "tol_poses"),
    ("landmarks", -1, "tol_landmarks"),
    ("rmse", +1, "tol_rmse"),
]


def runs_of(s: str, prefix: str) -> list[tuple[Path, dict]]:
    rows = {}
    for line in (sfmbench.BENCH / "bench.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["set"] == s and r["tag"].startswith(prefix + "-") and r["tag"][len(prefix) + 1:].isdigit():
            rows[r["tag"]] = r
    out = [(sfmbench.BENCH / s / "runs" / tag, r) for tag, r in sorted(rows.items(), key=lambda kv: int(kv[0].rsplit("-", 1)[1]))]
    return [(d, r) for d, r in out if (d / "cameras.sfm").is_file()]


def verdict(name: str, worse: int, tol_pct: float, alpha: float, base: list[float], cand: list[float]) -> dict:
    w = welch(base, cand)
    mb = float(np.mean(base))
    pct = 100.0 * w["diff"] / mb if mb else 0.0
    is_worse = w["diff"] * worse > 0
    sig = w["p"] < alpha
    v = "PASS"
    if sig and is_worse:
        v = "FAIL" if abs(pct) > tol_pct else "WARN"
    elif sig:
        v = "PASS (better)"
    # resolution: the smallest difference this comparison could have called significant (t_crit x SE).
    # A PASS means "no difference larger than this", which on the engine bay at n=10 is about 0.08 %
    # of the landmarks - QR's -0.05 % there passed for exactly that reason (docs/04).
    res = t_critical(w["df"], alpha) * w["se"] if w["se"] > 0 else 0.0
    return {"metric": name, "resolution": res, "resolution_pct": 100.0 * res / mb if mb else 0.0,
            "base_mean": mb, "base_sd": float(np.std(base, ddof=1)) if len(base) > 1 else 0.0,
            "cand_mean": float(np.mean(cand)), "cand_sd": float(np.std(cand, ddof=1)) if len(cand) > 1 else 0.0,
            "diff": w["diff"], "pct": pct, "t": w["t"], "p": w["p"], "verdict": v, "tolerance_pct": tol_pct}


def geometry(base_runs: list[Path], cand_runs: list[Path], ratio_warn: float, ratio_fail: float, floors: dict) -> dict:
    cams = {d: load_cameras(d) for d in base_runs + cand_runs}
    within = [align(cams[a], cams[b]) for a, b in itertools.combinations(base_runs, 2)]
    cross = [align(cams[c], cams[b]) for b in base_runs for c in cand_runs]
    within = [x for x in within if x]
    cross = [x for x in cross if x]
    out = {}
    for key in ("centre_rms_rel", "rot_median_deg"):
        wv = [x[key] for x in within]
        cv = [x[key] for x in cross]
        mw, mc = float(np.median(wv)) if wv else 0.0, float(np.median(cv)) if cv else 0.0
        ratio = mc / mw if mw > 0 else (math.inf if mc > 0 else 1.0)
        # A ratio alone flags harmless differences: two ways of rounding converge on slightly different
        # solutions, and against a tight noise floor (0.013 degrees on 41 views) 0.024 degrees reads as
        # 1.8x. Without ground truth the geometry says "different", not "worse", so it flags only when the
        # difference is also large in absolute terms.
        v = "PASS"
        if mc > floors[key]:
            v = "FAIL" if ratio > ratio_fail else "WARN" if ratio > ratio_warn else "PASS"
        out[key] = {"within_base_median": mw, "cross_median": mc, "ratio": ratio, "floor": floors[key],
                    "pairs": [len(wv), len(cv)], "verdict": v}
    return out


def report(s: str, legs: list[tuple[str, str]], a: argparse.Namespace) -> int:
    (bname, bpre), rest = legs[0], legs[1:]
    base = runs_of(s, bpre)
    if len(base) < 2:
        sys.exit(f"baseline {bname} ({bpre}-N on {s}) has {len(base)} runs; need at least 2")
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {"set": s, "baseline": {"name": bname, "prefix": bpre, "runs": len(base), "env": base[0][1].get("env", {})},
           "alpha": a.alpha, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "candidates": []}
    lines = [f"# Quality gate: {s}, baseline `{bname}` ({len(base)} runs, env {json.dumps(base[0][1].get('env', {}))})", ""]
    worst = "PASS"
    for cname, cpre in rest:
        cand = runs_of(s, cpre)
        if len(cand) < 2:
            sys.exit(f"candidate {cname} ({cpre}-N) has {len(cand)} runs; need at least 2")
        rows = [verdict(m, w, getattr(a, t), a.alpha, [r[1][m] for r in base], [r[1][m] for r in cand]) for m, w, t in METRICS]
        geo = geometry([r[0] for r in base], [r[0] for r in cand], a.geo_warn, a.geo_fail,
                       {"centre_rms_rel": a.geo_floor_centre, "rot_median_deg": a.geo_floor_rot})
        doc["candidates"].append({"name": cname, "prefix": cpre, "runs": len(cand), "env": cand[0][1].get("env", {}),
                                  "metrics": rows, "geometry": geo})
        lines += [f"## `{cname}` against `{bname}` ({len(cand)} runs, env {json.dumps(cand[0][1].get('env', {}))})", "",
                  "| metric | baseline mean (sd) | candidate mean (sd) | difference | t | p | resolution | verdict |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['metric']} | {r['base_mean']:.6g} ({r['base_sd']:.3g}) | {r['cand_mean']:.6g} ({r['cand_sd']:.3g}) | "
                         f"{r['diff']:+.4g} ({r['pct']:+.3f} %) | {r['t']:+.2f} | {r['p']:.4f} | {r['resolution']:.3g} ({r['resolution_pct']:.3f} %) | {r['verdict']} (tolerance {r['tolerance_pct']} %) |")
        lines += ["", "| pose agreement | within baseline (median) | candidate vs baseline (median) | ratio | floor | pairs | verdict |", "|---|---|---|---|---|---|---|"]
        for key, label in (("centre_rms_rel", "camera centres, RMS / spread"), ("rot_median_deg", "rotation, median degrees")):
            g = geo[key]
            lines.append(f"| {label} | {g['within_base_median']:.3g} | {g['cross_median']:.3g} | {g['ratio']:.2f} | {g['floor']:.3g} | {g['pairs'][0]} / {g['pairs'][1]} | {g['verdict']} |")
        lines.append("")
        for v in [r["verdict"] for r in rows] + [g["verdict"] for g in geo.values()]:
            if v.startswith("FAIL"):
                worst = "FAIL"
            elif v.startswith("WARN") and worst == "PASS":
                worst = "WARN"
    lines.append(f"**Overall: {worst}** (alpha {a.alpha}; geometry WARN above {a.geo_warn}x the baseline's own spread and FAIL above {a.geo_fail}x, "
                 f"either only past the floor: {a.geo_floor_centre:g} of the camera spread, {a.geo_floor_rot:g} degrees)")
    doc["overall"] = worst
    name = a.name or f"{s}-{bpre}"
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / f"{name}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {OUT / (name + '.md')}")
    return 0 if worst != "FAIL" else 1


def parse_leg(text: str) -> tuple[str, dict[str, str]]:
    name, _, rest = text.partition(":")
    env = {}
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        env[k.strip()] = v.strip()
    return name, env


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("sfm", "report"):
        p = sub.add_parser(cmd)
        p.add_argument("set", choices=sorted(set(sfmbench.SETS) | set(sfmbench.KEPT)))
        p.add_argument("--leg", action="append", required=True)
        p.add_argument("--alpha", type=float, default=0.05)
        p.add_argument("--tol-poses", dest="tol_poses", type=float, default=0.0, help="percent; default: any significant loss of placed views fails")
        p.add_argument("--tol-landmarks", dest="tol_landmarks", type=float, default=0.5, help="percent (default 0.5)")
        p.add_argument("--tol-rmse", dest="tol_rmse", type=float, default=0.5, help="percent (default 0.5)")
        p.add_argument("--geo-warn", dest="geo_warn", type=float, default=1.5)
        p.add_argument("--geo-fail", dest="geo_fail", type=float, default=3.0)
        p.add_argument("--geo-floor-centre", dest="geo_floor_centre", type=float, default=1e-3,
                       help="camera-centre disagreement below this fraction of the spread never flags (default 0.001)")
        p.add_argument("--geo-floor-rot", dest="geo_floor_rot", type=float, default=0.1,
                       help="rotation disagreement below this many degrees never flags (default 0.1)")
        p.add_argument("--name", help="report file name (default <set>-<baseline prefix>)")
        if cmd == "sfm":
            p.add_argument("--repeat", type=int, default=10)
            p.add_argument("--prefix", default=time.strftime("q%m%d%H%M"))
    a = ap.parse_args(argv)
    if a.cmd == "sfm":
        legs = [parse_leg(x) for x in a.leg]
        if len(legs) < 2:
            sys.exit("give a baseline leg and at least one candidate")
        for i in range(a.repeat):
            order = legs if i % 2 == 0 else list(reversed(legs))
            for name, env in order:
                sfmbench.run(a.set, f"{a.prefix}-{name}-{i + 1}", env, 1, [])
        return report(a.set, [(n, f"{a.prefix}-{n}") for n, _ in legs], a)
    legs = []
    for x in a.leg:
        name, _, prefix = x.partition("=")
        legs.append((name, prefix or name))
    return report(a.set, legs, a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
