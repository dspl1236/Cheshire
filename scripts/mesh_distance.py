#!/usr/bin/env python3
"""Surface distance between two meshes, both ways: how far apart are two reconstructions?

usage: mesh_distance.py A.obj B.obj [--samples N] [--seed S] [--json out.json]

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
so on, so holes and extra surface are measured, not dropped. Used by scripts/quality_gate.py (mesh).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
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
    """Distance from each p to the closest point of triangle (a, b, c), rows paired (Ericson, Real-Time
    Collision Detection 5.1.5, the Voronoi regions tested in order and selected without branches)."""
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
    return np.linalg.norm(p - q, axis=1)


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

    def nearest(self, Q: np.ndarray, batch: int = 20000) -> np.ndarray:
        """Exact distance to the nearest triangle among those in the query's 27 cells; inf if none, and
        only trusted up to h (a closer triangle cannot hide outside the block)."""
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
                has = cnt > 0
                if not has.any():
                    continue
                qi = np.repeat(np.nonzero(has)[0], cnt[has])
                within = np.arange(len(qi)) - np.repeat(np.cumsum(cnt[has]) - cnt[has], cnt[has])
                t = self.tri[np.repeat(lo[has], cnt[has]) + within]
                d = point_triangle_distance(q[qi], self.a[t], self.b[t], self.c[t])
                np.minimum.at(best, qi, d)
            out[s:s + batch] = best
        return out


def directed(samples: np.ndarray, V: np.ndarray, F: np.ndarray, h: float, origin: np.ndarray) -> np.ndarray:
    """Distance from every sample to the surface (V, F), coarsening the grid for samples not settled."""
    d = np.full(len(samples), np.inf)
    todo = np.arange(len(samples))
    for _ in range(10):
        if len(todo) == 0:
            break
        got = TriangleGrid(V, F, h, origin).nearest(samples[todo])
        settled = got <= h            # nothing outside the 27-cell block can be closer than h
        d[todo[settled]] = got[settled]
        todo = todo[~settled]
        h *= 4.0
    if len(todo):                     # beyond every grid: every triangle, rare by construction
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        for i in todo:
            d[i] = float(point_triangle_distance(np.repeat(samples[i][None], len(F), 0), a, b, c).min())
    return d


def summarise(d: np.ndarray, diag: float) -> dict:
    q = np.percentile(d, [50, 95, 99])
    return dict(mean=float(d.mean()), median=float(q[0]), p95=float(q[1]), p99=float(q[2]), max=float(d.max()),
                mean_rel=float(d.mean() / diag), p95_rel=float(q[1] / diag), p99_rel=float(q[2] / diag), max_rel=float(d.max() / diag),
                within_0p1pct=float((d <= 0.001 * diag).mean()), within_0p5pct=float((d <= 0.005 * diag).mean()),
                within_1pct=float((d <= 0.01 * diag).mean()))


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


def compare(a_path: Path, b_path: Path, samples: int = 300_000, seed: int = 0, cache: dict | None = None) -> dict:
    # different sampling seeds for the two meshes, so identical meshes measure their true zero
    Va, Fa, Sa, areaA = load(a_path, samples, seed, cache)
    Vb, Fb, Sb, areaB = load(b_path, samples, seed + 1, cache)
    lo = np.minimum(Va.min(0), Vb.min(0)) - 1e-9
    hi = np.maximum(Va.max(0), Vb.max(0))
    diag = float(np.linalg.norm(hi - lo))

    def cell(V, F):  # about two median edges, so most triangles span one or two cells per axis
        e = np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1)
        return max(2.0 * float(np.median(e)), 1e-6 * diag)

    ab = directed(Sa, Vb, Fb, cell(Vb, Fb), lo)
    ba = directed(Sb, Va, Fa, cell(Va, Fa), lo)
    return dict(a=str(a_path), b=str(b_path), vertices=[len(Va), len(Vb)], triangles=[len(Fa), len(Fb)], area=[areaA, areaB],
                diagonal=diag, samples=samples, a_to_b=summarise(ab, diag), b_to_a=summarise(ba, diag))


def fmt(r: dict) -> str:
    lines = [f"A {r['a']}: {r['vertices'][0]} vertices, {r['triangles'][0]} triangles",
             f"B {r['b']}: {r['vertices'][1]} vertices, {r['triangles'][1]} triangles",
             f"diagonal {r['diagonal']:.6g}; {r['samples']} samples per mesh"]
    for k, name in (("a_to_b", "A to B"), ("b_to_a", "B to A")):
        s = r[k]
        lines.append(f"{name}: mean {s['mean']:.4g}, median {s['median']:.4g}, p95 {s['p95']:.4g}, p99 {s['p99']:.4g}, max {s['max']:.4g} "
                     f"(p95 {100 * s['p95_rel']:.4f} %, max {100 * s['max_rel']:.3f} % of the diagonal); within 0.1/0.5/1 %: "
                     f"{100 * s['within_0p1pct']:.2f} / {100 * s['within_0p5pct']:.2f} / {100 * s['within_1pct']:.2f} %")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--samples", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    r = compare(Path(a.a), Path(a.b), a.samples, a.seed)
    print(fmt(r))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
