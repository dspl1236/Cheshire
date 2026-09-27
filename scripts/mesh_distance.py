#!/usr/bin/env python3
"""Surface distance between two meshes, both ways: how far apart are two reconstructions?

usage: mesh_distance.py A.obj B.obj [--a-cameras A.sfm --b-cameras B.sfm] [--samples N] [--seed S] [--json out.json]

Meshing's output is not reproducible run to run (geogram renumbers the tetrahedralisation, the GPU
votes and max-flow accumulate in float), so two meshes of the same job are compared by geometry, not
bytes. Each mesh's surface is sampled, area-weighted, and each sample's distance to the other mesh's
surface is measured exactly: the closest point on the nearest triangle, not the nearest vertex or
sample. A to B says where A has surface that B lacks or has moved, B to A the reverse. Reported per
direction: mean, median, p95, p99 and max, absolute and as a fraction of the bounding-box diagonal,
and the share of samples within 0.1 %, 0.5 % and 1 % of the diagonal.

Only numpy. Each triangle is binned into every cell of a voxel grid its bounding box overlaps; a
sample tests the triangles of its own cell and the 26 around it, which is exact when the nearest
surface is closer than one cell. Samples with nothing that close go to a grid four times coarser, and
so on up to the cap (--cap, default 2 % of the diagonal): holes and extra surface are measured, not
dropped, and a sample farther than the cap from the other mesh counts as the cap and is reported as
"beyond". The max is then a lower bound; the percentiles are exact as long as fewer samples than
their tail lie beyond. The (sample, triangle) pairs are evaluated in chunks of at most 2 million, so
memory stays bounded however coarse the grid gets (an unbounded first version reached 45 GB on two
meshes in different frames). Used by scripts/quality_gate.py (mesh).

With each run's cameras.sfm (StructureFromMotion's output), A is first moved into B's frame by the
similarity that best maps A's camera centres onto B's (Umeyama). Incremental SfM fixes its frame
from the reconstruction itself, so runs whose SfM went differently (another initial pair, another
build) can sit in different frames, which is no difference in quality. Where the frames already
agree it does not help: two Cheshire engine-bay runs had camera centres 0.2 % of their spread apart
(0.03 % after aligning), and their surfaces measured closer unaligned (median 0.0020) than aligned
on the cameras (0.0030), because the surfaces agreed better than the cameras did. The alignment is
reported with the result, so the two can be compared.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (n, 3) float64 and triangles (m, 3) int64 of an OBJ; polygons are fanned into triangles."""
    verts, faces = [], []
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b"v "):
                verts.append(line[2:])
            elif line.startswith(b"f "):
                faces.append(line[2:])
    V = np.array(b" ".join(verts).split(), dtype=np.float64).reshape(-1, 3)
    # the common case, every face a triangle: one vectorised parse (texture/normal indices dropped)
    tokens = re.sub(rb"/[^\s]*", b"", b" ".join(faces)).split()
    if len(tokens) == 3 * len(faces):
        F = np.array(tokens, dtype=np.int64).reshape(-1, 3)
        return V, np.where(F > 0, F - 1, len(V) + F)
    tri = []
    for fl in faces:
        idx = [int(t.split(b"/")[0]) for t in fl.split()]
        idx = [i - 1 if i > 0 else len(V) + i for i in idx]
        for k in range(1, len(idx) - 1):
            tri.append((idx[0], idx[k], idx[k + 1]))
    return V, np.array(tri, dtype=np.int64).reshape(-1, 3)


def sample_surface(V: np.ndarray, F: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """n points on the surface, area-weighted; also the total area."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = float(area.sum())
    if total <= 0:
        raise ValueError("mesh has no area")
    cdf = np.cumsum(area)
    tri = np.minimum(np.searchsorted(cdf, rng.random(n) * total, side="right"), len(F) - 1)
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    w0, w1, w2 = 1 - r1, r1 * (1 - r2), r1 * r2
    return w0[:, None] * a[tri] + w1[:, None] * b[tri] + w2[:, None] * c[tri], total


def point_triangle_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Distance from each p to the closest point of triangle (a, b, c), rows paired."""
    return np.linalg.norm(p - point_triangle_closest(p, a, b, c), axis=1)


def point_triangle_closest(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The closest point of triangle (a, b, c) to each p, rows paired (Ericson, Real-Time Collision
    Detection 5.1.5, the Voronoi regions tested in order and selected without branches)."""
    ab, ac, ap = b - a, c - a, p - a
    dot = lambda x, y: np.einsum("ij,ij->i", x, y)
    d1, d2 = dot(ab, ap), dot(ac, ap)
    bp = p - b
    d3, d4 = dot(ab, bp), dot(ac, bp)
    cp = p - c
    d5, d6 = dot(ab, cp), dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ab = np.where(d1 - d3 != 0, d1 / (d1 - d3), 0.0)
        t_ac = np.where(d2 - d6 != 0, d2 / (d2 - d6), 0.0)
        e = (d4 - d3) + (d5 - d6)
        t_bc = np.where(e != 0, (d4 - d3) / e, 0.0)
        s = va + vb + vc
        v = np.where(s != 0, vb / s, 0.0)
        w = np.where(s != 0, vc / s, 0.0)
    conds = [(d1 <= 0) & (d2 <= 0),                                  # vertex a
             (d3 >= 0) & (d4 <= d3),                                 # vertex b
             (vc <= 0) & (d1 >= 0) & (d3 <= 0),                      # edge ab
             (d6 >= 0) & (d5 <= d6),                                 # vertex c
             (vb <= 0) & (d2 >= 0) & (d6 <= 0),                      # edge ac
             (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)]        # edge bc
    choices = [a, b, a + t_ab[:, None] * ab, c, a + t_ac[:, None] * ac, b + t_bc[:, None] * (c - b)]
    q = a + v[:, None] * ab + w[:, None] * ac                         # the face
    for cond, ch in zip(reversed(conds), reversed(choices)):          # first true condition wins
        q = np.where(cond[:, None], ch, q)
    return q


def _key(cells: np.ndarray) -> np.ndarray:
    c = cells + (1 << 20)  # offset so negative cells pack; 21 bits per axis
    return (c[..., 0] << 42) | (c[..., 1] << 21) | c[..., 2]


class TriangleGrid:
    """Every triangle listed under each voxel its bounding box overlaps, sorted by voxel key."""

    def __init__(self, V: np.ndarray, F: np.ndarray, h: float, origin: np.ndarray):
        self.h, self.origin = h, origin
        self.a, self.b, self.c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        lo = np.floor((np.minimum(np.minimum(self.a, self.b), self.c) - origin) / h).astype(np.int64)
        hi = np.floor((np.maximum(np.maximum(self.a, self.b), self.c) - origin) / h).astype(np.int64)
        dims = hi - lo + 1
        n = dims.prod(axis=1)
        tri = np.repeat(np.arange(len(F)), n)
        k = np.arange(len(tri)) - np.repeat(np.cumsum(n) - n, n)   # the cell's rank within its triangle's box
        d = dims[tri]
        off = np.stack([k // (d[:, 1] * d[:, 2]), (k // d[:, 2]) % d[:, 1], k % d[:, 2]], axis=1)
        keys = _key(lo[tri] + off)
        order = np.argsort(keys, kind="stable")
        self.keys, self.tri = keys[order], tri[order]

    def nearest(self, Q: np.ndarray, batch: int = 20000, pair_cap: int = 2_000_000) -> np.ndarray:
        """Exact distance to the nearest triangle among those in the query's 27 cells; inf if none, and
        only trusted up to h (a closer triangle cannot hide outside the block). At most pair_cap
        (query, triangle) pairs are expanded at once."""
        out = np.full(len(Q), np.inf)
        offsets = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=np.int64)
        for s in range(0, len(Q), batch):
            q = Q[s:s + batch]
            base = np.floor((q - self.origin) / self.h).astype(np.int64)
            best = np.full(len(q), np.inf)
            for off in offsets:
                keys = _key(base + off)
                lo = np.searchsorted(self.keys, keys, side="left")
                cnt = np.searchsorted(self.keys, keys, side="right") - lo
                idx = np.nonzero(cnt > 0)[0]
                if len(idx) == 0:
                    continue
                # split the queries so that no chunk expands more than pair_cap pairs (one query alone
                # may exceed it; then its triangles go in slices of pair_cap)
                csum = np.cumsum(cnt[idx])
                start = 0
                while start < len(idx):
                    before = csum[start - 1] if start else 0
                    stop = max(int(np.searchsorted(csum, before + pair_cap, side="right")), start + 1)
                    sel = idx[start:stop]
                    c = cnt[sel]
                    if len(sel) == 1 and c[0] > pair_cap:
                        qn = int(sel[0])
                        for o in range(0, int(c[0]), pair_cap):
                            t = self.tri[lo[qn] + o: lo[qn] + min(int(c[0]), o + pair_cap)]
                            d = point_triangle_distance(np.repeat(q[qn][None], len(t), 0), self.a[t], self.b[t], self.c[t])
                            best[qn] = min(best[qn], float(d.min()))
                    else:
                        qi = np.repeat(sel, c)
                        within = np.arange(len(qi)) - np.repeat(np.cumsum(c) - c, c)
                        t = self.tri[np.repeat(lo[sel], c) + within]
                        d = point_triangle_distance(q[qi], self.a[t], self.b[t], self.c[t])
                        np.minimum.at(best, qi, d)
                    start = stop
            out[s:s + batch] = best
        return out


    def closest(self, Q: np.ndarray, pair_cap: int = 2_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Distance, closest point and triangle of the nearest triangle among the query's 27 cells (inf,
        nan, -1 where there is none; trusted up to h). For the modest sample counts of the ICP below."""
        best_d = np.full(len(Q), np.inf)
        best_p = np.full((len(Q), 3), np.nan)
        best_t = np.full(len(Q), -1, dtype=np.int64)
        offsets = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=np.int64)
        base = np.floor((Q - self.origin) / self.h).astype(np.int64)
        for off in offsets:
            keys = _key(base + off)
            lo = np.searchsorted(self.keys, keys, side="left")
            cnt = np.searchsorted(self.keys, keys, side="right") - lo
            idx = np.nonzero(cnt > 0)[0]
            csum = np.cumsum(cnt[idx])
            start = 0
            while start < len(idx):
                before = csum[start - 1] if start else 0
                stop = max(int(np.searchsorted(csum, before + pair_cap, side="right")), start + 1)
                sel = idx[start:stop]
                c = np.minimum(cnt[sel], pair_cap)     # a lone query past the cap keeps its first pair_cap (ICP only)
                qi = np.repeat(sel, c)
                within = np.arange(len(qi)) - np.repeat(np.cumsum(c) - c, c)
                tri = self.tri[np.repeat(lo[sel], c) + within]
                cp = point_triangle_closest(Q[qi], self.a[tri], self.b[tri], self.c[tri])
                d = np.linalg.norm(Q[qi] - cp, axis=1)
                order = np.lexsort((d, qi))                     # per query, nearest first
                first = order[np.unique(qi[order], return_index=True)[1]]
                better = d[first] < best_d[qi[first]]
                best_d[qi[first][better]] = d[first][better]
                best_p[qi[first][better]] = cp[first][better]
                best_t[qi[first][better]] = tri[first][better]
                start = stop
        return best_d, best_p, best_t


def umeyama(P: np.ndarray, Q: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimising |Q - (s R P + t)|."""
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, D, Vt = np.linalg.svd(Y.T @ X / len(P))
    S = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U) * np.linalg.det(Vt))) or 1.0])
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (X ** 2).sum(1).mean())
    return s, R, mq - s * R @ mp


def _rodrigues(w: np.ndarray) -> np.ndarray:
    a = float(np.linalg.norm(w))
    if a < 1e-15:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def refine(S: np.ndarray, V: np.ndarray, F: np.ndarray, origin: np.ndarray, h: float, iters: int = 80) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """Trimmed point-to-plane ICP with a similarity: moves the points S onto the surface (V, F). Each
    round pairs every point with its closest surface point within the grid's reach, keeps the pairs
    under 2.5x the median distance (so surface only one mesh has, and outliers, do not pull), and
    solves the linearised similarity (rotation, shift, scale) that best closes their distances along
    the triangles' normals. It starts on a coarse grid and tightens it once most points settle.
    Returns s, R, t with S -> s R S + t, and a summary of the rounds."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    s_tot, R_tot, t_tot = 1.0, np.eye(3), np.zeros(3)
    P = S.copy()
    level = h * 16.0
    hist, calm = [], 0
    for it in range(iters):
        d, Q, T = TriangleGrid(V, F, level, origin).closest(P)
        ok = d <= level
        if ok.sum() < 100:
            level *= 4.0
            continue
        med = float(np.median(d[ok]))
        keep = ok & (d <= 2.5 * med)
        N = n[T[keep]]
        c = P[keep].mean(0)
        X = P[keep] - c
        # n . (p + s' X + w x X + t - q) = 0, linear in (w, t, s'), with X taken about the centroid
        A = np.concatenate([np.cross(X, N), N, (N * X).sum(1, keepdims=True)], axis=1)
        b = -((P[keep] - Q[keep]) * N).sum(1)
        AtA = A.T @ A
        x = np.linalg.solve(AtA + 1e-9 * np.trace(AtA) / 7 * np.eye(7), A.T @ b)
        R, s = _rodrigues(x[:3]), 1.0 + float(x[6])
        tv = c - s * R @ c + x[3:6]
        P = s * P @ R.T + tv
        s_tot, R_tot, t_tot = s * s_tot, R @ R_tot, s * R @ t_tot + tv
        step = float(np.linalg.norm(x[:3])) * float(np.sqrt((X ** 2).sum(1).mean())) + float(np.linalg.norm(x[3:6])) + abs(float(x[6])) * float(np.sqrt((X ** 2).sum(1).mean()))
        hist.append(dict(cell=level, pairs=int(keep.sum()), settled=float(ok.mean()), median=med, step=step))
        if ok.mean() > 0.6 and level > h:
            level = max(h, level / 4.0)
            calm = 0
            continue
        # converged: two rounds in a row that move the points less than 1/200 of their median distance
        # (1e-9 of the cell for meshes that already coincide)
        calm = calm + 1 if step < max(5e-3 * med, 1e-9 * level) else 0
        if calm >= 2:
            break
    total_ang = float(np.degrees(np.arccos(np.clip((np.trace(R_tot) - 1.0) / 2.0, -1.0, 1.0))))
    return s_tot, R_tot, t_tot, dict(iterations=len(hist), rotation_deg=total_ang, scale_minus_1=s_tot - 1.0,
                                     shift=float(np.linalg.norm(t_tot + (s_tot * R_tot - np.eye(3)) @ S.mean(0))),
                                     median_first=hist[0]["median"] if hist else None, median_last=hist[-1]["median"] if hist else None,
                                     rounds=hist)


def directed(samples: np.ndarray, V: np.ndarray, F: np.ndarray, h: float, origin: np.ndarray, cap: float) -> np.ndarray:
    """Distance from every sample to the surface (V, F), coarsening the grid for samples not settled,
    up to cap: a sample with no surface within cap gets the value cap (counted as beyond)."""
    d = np.full(len(samples), cap)
    todo = np.arange(len(samples))
    while len(todo):
        h = min(h, cap)
        t0 = time.time()
        got = TriangleGrid(V, F, h, origin).nearest(samples[todo])
        print(f"    cell {h:.4g}: {len(todo)} samples, {int((got <= h).sum())} settled, {time.time() - t0:.1f} s",
              file=sys.stderr, flush=True)
        settled = got <= h            # nothing outside the 27-cell block can be closer than h
        d[todo[settled]] = got[settled]
        todo = todo[~settled]
        if h >= cap:
            break                     # what is left is farther than the cap
        h *= 4.0
    return d


def summarise(d: np.ndarray, diag: float, cap: float) -> dict:
    q = np.percentile(d, [50, 95, 99])
    return dict(mean=float(d.mean()), median=float(q[0]), p95=float(q[1]), p99=float(q[2]), max=float(d.max()),
                cap=cap, beyond_cap=float((d >= cap).mean()),
                mean_rel=float(d.mean() / diag), p95_rel=float(q[1] / diag), p99_rel=float(q[2] / diag), max_rel=float(d.max() / diag),
                within_0p1pct=float((d <= 0.001 * diag).mean()), within_0p5pct=float((d <= 0.005 * diag).mean()),
                within_1pct=float((d <= 0.01 * diag).mean()))


def camera_centres(path: Path) -> dict[str, np.ndarray]:
    """photo file name -> camera centre, from an AliceVision cameras.sfm / .json. Keyed by the photo's
    name, not its viewId: the viewId hashes the path, so the same photo on two machines differs."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    poses = {str(p["poseId"]): np.array([float(v) for v in p["pose"]["transform"]["center"]]) for p in d.get("poses", [])}
    key = lambda v: re.split(r"[\\/]", v["path"])[-1] if v.get("path") else str(v["viewId"])
    return {key(v): poses[str(v["poseId"])] for v in d.get("views", []) if str(v.get("poseId")) in poses}


def similarity(src: dict, dst: dict) -> dict:
    """s, R, t with dst ~ s R src + t on the shared views' centres (Umeyama), and how well it fits."""
    common = sorted(set(src) & set(dst))
    if len(common) < 3:
        raise ValueError(f"only {len(common)} cameras in common, cannot align the frames")
    P = np.array([src[k] for k in common])
    Q = np.array([dst[k] for k in common])
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, D, Vt = np.linalg.svd(Y.T @ X / len(common))
    S = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U) * np.linalg.det(Vt))) or 1.0])
    Rm = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (X ** 2).sum(1).mean())
    tv = mq - s * Rm @ mp
    spread = float(np.sqrt((Y ** 2).sum(1).mean()))
    raw = float(np.sqrt(((P - Q) ** 2).sum(1).mean()))
    res = float(np.sqrt(((Y - s * X @ Rm.T) ** 2).sum(1).mean()))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1.0) / 2.0, -1.0, 1.0))))
    return dict(s=s, R=Rm, t=tv, cameras=len(common), scale_minus_1=s - 1.0, rotation_deg=angle,
                shift_rel=float(np.linalg.norm(tv + (s * Rm - np.eye(3)) @ mp) / spread) if spread > 0 else 0.0,
                centres_raw_rel=raw / spread if spread > 0 else 0.0, centres_aligned_rel=res / spread if spread > 0 else 0.0)


def load(path: Path, samples: int, seed: int, cache: dict | None = None):
    key = (str(path), samples, seed)
    if cache is not None and key in cache:
        return cache[key]
    V, F = read_obj(path)
    S, area = sample_surface(V, F, samples, np.random.default_rng(seed))
    r = (V, F, S, area)
    if cache is not None:
        cache[key] = r
    return r


def compare(a_path: Path, b_path: Path, samples: int = 300_000, seed: int = 0, cache: dict | None = None,
            a_cameras: Path | None = None, b_cameras: Path | None = None, cap_rel: float = 0.02,
            refine_surface: bool = False, common: bool = False) -> dict:
    # different sampling seeds for the two meshes, so identical meshes measure their true zero
    Va, Fa, Sa, areaA = load(a_path, samples, seed, cache)
    Vb, Fb, Sb, areaB = load(b_path, samples, seed + 1, cache)
    alignment = None
    if a_cameras and b_cameras:
        sim = similarity(camera_centres(a_cameras), camera_centres(b_cameras))
        # area-weighted samples move with the surface under a similarity, so A's samples follow A
        Va = sim["s"] * Va @ sim["R"].T + sim["t"]
        Sa = sim["s"] * Sa @ sim["R"].T + sim["t"]
        areaA *= sim["s"] ** 2
        alignment = {k: v for k, v in sim.items() if k not in ("s", "R", "t")}
    icp = None
    if refine_surface:
        # the ICP works on its own 20,000 of A's samples; the transform then moves all of A. With
        # common, they come from the volume both meshes cover: surface only one mesh has pairs with the
        # other's edge and drags the fit (a first version left a 1 degree pair still moving 2 mm a round)
        o = np.minimum(Va.min(0), Vb.min(0)) - 1e-9
        d0 = float(np.linalg.norm(np.maximum(Va.max(0), Vb.max(0)) - o))
        e = np.linalg.norm(Vb[Fb[:, 1]] - Vb[Fb[:, 0]], axis=1)
        pool = np.arange(len(Sa))
        if common:
            c_lo, c_hi = np.maximum(Va.min(0), Vb.min(0)), np.minimum(Va.max(0), Vb.max(0))
            pool = pool[np.all((Sa >= c_lo) & (Sa <= c_hi), axis=1)]
        pick = np.random.default_rng(seed + 7).choice(pool, size=min(20000, len(pool)), replace=False)
        s, R, tv, icp = refine(Sa[pick], Vb, Fb, o, max(2.0 * float(np.median(e)), 1e-6 * d0))
        Va, Sa = s * Va @ R.T + tv, s * Sa @ R.T + tv
        areaA *= s ** 2
    lo = np.minimum(Va.min(0), Vb.min(0)) - 1e-9
    hi = np.maximum(Va.max(0), Vb.max(0))
    diag = float(np.linalg.norm(hi - lo))

    def cell(V, F):  # about two median edges, so most triangles span one or two cells per axis
        e = np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1)
        return max(2.0 * float(np.median(e)), 1e-6 * diag)

    inside = None
    if common:
        # only the volume both meshes cover: Meshing clips each mesh to a box it estimates from the SfM
        # landmarks, and that box moves run to run (upstream's own engine-bay runs: 3.66 x 1.75 x 1.77
        # and 2.73 x 0.96 x 1.73), so surface outside the other mesh's extent is left out, and counted.
        # The diagonal, the yardstick of every relative figure, is then the common box's, so pairs with
        # different extents are measured against the same length.
        c_lo, c_hi = np.maximum(Va.min(0), Vb.min(0)), np.minimum(Va.max(0), Vb.max(0))
        ina = np.all((Sa >= c_lo) & (Sa <= c_hi), axis=1)
        inb = np.all((Sb >= c_lo) & (Sb <= c_hi), axis=1)
        Sa, Sb = Sa[ina], Sb[inb]
        inside = dict(a=float(ina.mean()), b=float(inb.mean()), box=(c_hi - c_lo).tolist())
        diag = float(np.linalg.norm(c_hi - c_lo))
    cap = cap_rel * diag
    ab = directed(Sa, Vb, Fb, cell(Vb, Fb), lo, cap)
    ba = directed(Sb, Va, Fa, cell(Va, Fa), lo, cap)
    return dict(a=str(a_path), b=str(b_path), vertices=[len(Va), len(Vb)], triangles=[len(Fa), len(Fb)], area=[areaA, areaB],
                diagonal=diag, samples=samples, alignment=alignment, refinement=icp, common=inside, a_to_b=summarise(ab, diag, cap), b_to_a=summarise(ba, diag, cap))


def fmt(r: dict) -> str:
    lines = [f"A {r['a']}: {r['vertices'][0]} vertices, {r['triangles'][0]} triangles",
             f"B {r['b']}: {r['vertices'][1]} vertices, {r['triangles'][1]} triangles",
             f"diagonal {r['diagonal']:.6g}; {r['samples']} samples per mesh"]
    al = r.get("alignment")
    if al:
        lines.append(f"A moved into B's frame on {al['cameras']} cameras: scale {al['scale_minus_1']:+.3e}, rotation "
                     f"{al['rotation_deg']:.4f} degrees, shift {100 * al['shift_rel']:.3f} % of the camera spread; centres "
                     f"{100 * al['centres_raw_rel']:.3f} % apart before, {100 * al['centres_aligned_rel']:.4f} % after")
    ic = r.get("refinement")
    if ic:
        lines.append(f"then refined on the surfaces (trimmed point-to-plane ICP, {ic['iterations']} rounds): rotation {ic['rotation_deg']:.4f} degrees, "
                     f"scale {ic['scale_minus_1']:+.3e}, shift {ic['shift']:.4g}; median pair distance {ic['median_first']:.4g} -> {ic['median_last']:.4g}")
    cm = r.get("common")
    if cm:
        lines.append(f"common volume {' x '.join(f'{v:.4g}' for v in cm['box'])}: {100 * cm['a']:.2f} % of A's samples and "
                     f"{100 * cm['b']:.2f} % of B's inside it, the rest left out")
    for k, name in (("a_to_b", "A to B"), ("b_to_a", "B to A")):
        s = r[k]
        lines.append(f"{name}: mean {s['mean']:.4g}, median {s['median']:.4g}, p95 {s['p95']:.4g}, p99 {s['p99']:.4g}, max {s['max']:.4g} "
                     f"(p95 {100 * s['p95_rel']:.4f} %, max {100 * s['max_rel']:.3f} % of the diagonal); within 0.1/0.5/1 %: "
                     f"{100 * s['within_0p1pct']:.2f} / {100 * s['within_0p5pct']:.2f} / {100 * s['within_1pct']:.2f} %"
                     + (f"; {100 * s['beyond_cap']:.2f} % beyond the {100 * s['cap'] / r['diagonal']:.1f} % cap (max is a lower bound)"
                        if s["beyond_cap"] > 0 else ""))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--samples", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--a-cameras", default=None, help="A's cameras.sfm: align A's frame to B's on the cameras first")
    ap.add_argument("--b-cameras", default=None, help="B's cameras.sfm")
    ap.add_argument("--common", action="store_true", help="measure only inside the volume both meshes cover (after any alignment)")
    ap.add_argument("--refine", action="store_true", help="refine the alignment on the surfaces (trimmed similarity ICP) before measuring")
    ap.add_argument("--cap", type=float, default=0.02, help="distances measured up to this fraction of the diagonal (default 0.02)")
    a = ap.parse_args()
    if bool(a.a_cameras) != bool(a.b_cameras):
        ap.error("give both --a-cameras and --b-cameras, or neither")
    r = compare(Path(a.a), Path(a.b), a.samples, a.seed, None, a.a_cameras, a.b_cameras, a.cap, a.refine, a.common)
    print(fmt(r))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
