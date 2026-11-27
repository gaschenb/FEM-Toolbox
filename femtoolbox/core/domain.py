from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from matplotlib.path import Path

from femtoolbox.core.expr import Expr, numeric
from femtoolbox.core.mesh import (ArcPiece, Boundary, Mesh2D, PolylinePiece, SegmentPiece, compress, distmesh,
                                  label_longest_edge, polygon_boundary, rectangle_boundary, structured_rectangle)


def _finish(nodes, tris, name, boundary) -> Mesh2D:
    nodes, tris = compress(nodes, tris)
    m = Mesh2D(nodes, tris, name, boundary)
    if boundary is not None and boundary.pieces:
        bn = m.boundary_nodes
        nodes = m.nodes.copy()
        nodes[bn] = boundary.project(nodes[bn])
        m = Mesh2D(nodes, m.triangles, name, boundary)
    m.triangles = label_longest_edge(m.triangles, m.nodes)
    return m


def sdf_polygon(verts: np.ndarray):
    verts = np.asarray(verts, float)
    path = Path(np.vstack((verts, verts[:1])))
    A, B = verts, np.roll(verts, -1, axis=0)

    AB = B - A
    L2 = np.maximum((AB * AB).sum(1), 1e-300)
    n = len(A)
    tree = None
    if n > 48:
        from scipy.spatial import cKDTree
        tree = cKDTree(A)

    def seg_dist(q, idx):
        a, ab, l2 = A[idx], AB[idx], L2[idx]
        s = np.clip(((q[:, None, :] - a) * ab).sum(-1) / l2, 0, 1)
        diff = q[:, None, :] - (a + s[..., None] * ab)
        return np.sqrt((diff * diff).sum(-1).min(1))

    def fd(p):
        if tree is not None:
            _, k = tree.query(p, k=4)
            idx = np.concatenate((k, (k - 1) % n), axis=1)
            d = seg_dist(p, idx)
        else:
            d = np.empty(len(p))
            for s0 in range(0, len(p), 4096):
                d[s0:s0 + 4096] = seg_dist(p[s0:s0 + 4096], np.arange(n)[None, :])
        return np.where(path.contains_points(p), -d, d)
    return fd


def sdf_circle(cx, cy, r):
    return lambda p: np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r


def _h_from(size: float, h=None, n=None, default_n=16) -> float:
    if h is not None and float(h) > 0:
        return float(h)
    return float(size) / max(2, int(n or default_n))


@dataclass
class RectangleDomain:
    width: float = 1.0
    height: float = 1.0
    x0: float = 0.0
    y0: float = 0.0

    name = "rectangle"

    def boundary(self):
        return rectangle_boundary(self.x0, self.y0, self.x0 + self.width, self.y0 + self.height)

    def mesh(self, nx: int = 16, ny: int | None = None, h: float | None = None, structured: bool = True, **kw) -> Mesh2D:
        if structured and h is None:
            ny = nx if ny is None else ny
            if type(self) is RectangleDomain and ny == nx and self.width != self.height:
                ny = max(1, int(round(nx * self.height / self.width)))
            nodes, tris = structured_rectangle(self.x0, self.y0, self.x0 + self.width, self.y0 + self.height, nx, ny)
            return _finish(nodes, tris, self.name, self.boundary())
        h = _h_from(min(self.width, self.height), h, nx)
        x1, y1 = self.x0 + self.width, self.y0 + self.height
        verts = np.array([(self.x0, self.y0), (x1, self.y0), (x1, y1), (self.x0, y1)])
        p, t = distmesh(sdf_polygon(verts), h, ((self.x0, self.y0), (x1, y1)), pfix=_edge_fix(verts, h))
        return _finish(p, t, self.name, self.boundary())


class SquareDomain(RectangleDomain):
    name = "square"

    def __init__(self, size: float = 1.0, x0: float = 0.0, y0: float = 0.0):
        super().__init__(float(size), float(size), x0, y0)
        self.size = float(size)


def _edge_fix(verts, h):
    """Corner points plus evenly spaced points on each polygon edge (keeps straight edges exact)."""
    pts = []
    n = len(verts)
    for i in range(n):
        a, b = verts[i], verts[(i + 1) % n]
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / h)))
        for j in range(k):
            pts.append(a + (j / k) * (b - a))
    return np.asarray(pts)


@dataclass
class DiskDomain:
    radius: float = 1.0
    center: tuple[float, float] = (0.0, 0.0)
    narcs: int = 8

    name = "disk"

    def boundary(self):
        k = self.narcs
        return Boundary([ArcPiece(self.center, self.radius, 2 * np.pi * i / k, 2 * np.pi * (i + 1) / k, f"arc_{i}")
                         for i in range(k)], coordinate_name="t", period=2 * np.pi)

    def mesh(self, nr: int = 8, ntheta: int | None = None, h: float | None = None, n: int | None = None, **kw) -> Mesh2D:
        if h is None:
            if ntheta:
                h = 2 * np.pi * self.radius / ntheta
            else:
                h = self.radius / max(2, int(n or nr))
        cx, cy = self.center
        R = self.radius
        m = int(np.ceil(2 * np.pi * R / h))
        th = 2 * np.pi * np.arange(m) / m
        pfix = np.column_stack((cx + R * np.cos(th), cy + R * np.sin(th)))
        p, t = distmesh(sdf_circle(cx, cy, R), h, ((cx - R, cy - R), (cx + R, cy + R)), pfix=pfix)
        return _finish(p, t, self.name, self.boundary())


@dataclass
class LShapeDomain:
    size: float = 1.0

    name = "l-shape"

    def vertices(self):
        s = float(self.size)
        return np.array([(0, 0), (s, 0), (s, s / 2), (s / 2, s / 2), (s / 2, s), (0, s)], float)

    def boundary(self):
        return polygon_boundary(self.vertices(), ["bottom", "right_lower", "reentrant_horizontal", "reentrant_vertical",
                                                  "top_left", "left"])

    def mesh(self, nx: int = 16, ny: int | None = None, h: float | None = None, structured: bool = True, **kw) -> Mesh2D:
        s = float(self.size)
        if structured and h is None:
            nx = max(2, int(nx) + int(nx) % 2)
            nodes, tris = structured_rectangle(0, 0, s, s, nx, nx)
            c = nodes[tris].mean(1)
            tris = tris[~((c[:, 0] > s / 2) & (c[:, 1] > s / 2))]
            return _finish(nodes, tris, self.name, self.boundary())
        h = _h_from(s, h, nx)
        v = self.vertices()
        p, t = distmesh(sdf_polygon(v), h, ((0, 0), (s, s)), pfix=_edge_fix(v, h))
        return _finish(p, t, self.name, self.boundary())


@dataclass
class PolygonDomain:
    vertices: Iterable[tuple[float, float]]
    names: list[str] | None = None

    name = "polygon"

    def _verts(self):
        v = np.asarray(list(self.vertices), float)
        if len(v) < 3:
            raise ValueError("polygon needs at least 3 vertices")
        if _self_intersects(v):
            c = v.mean(0)
            v = v[np.argsort(np.arctan2(v[:, 1] - c[1], v[:, 0] - c[0]))]
        area = 0.5 * np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1])
        if area < 0:
            v = v[::-1]
        return v

    def boundary(self):
        return polygon_boundary(self._verts(), self.names)

    def mesh(self, nx: int = 24, ny: int | None = None, h: float | None = None, **kw) -> Mesh2D:
        v = self._verts()
        lo, hi = v.min(0), v.max(0)
        h = _h_from(min(hi - lo), h, nx)
        p, t = distmesh(sdf_polygon(v), h, (lo, hi), pfix=_edge_fix(v, h))
        return _finish(p, t, self.name, self.boundary())


@dataclass
class PlateWithHoleDomain:
    width: float = 2.0
    height: float = 1.0
    hole_center: tuple[float, float] | None = None
    hole_radius: float = 0.2

    name = "plate-with-hole"

    def _c(self):
        return self.hole_center if self.hole_center is not None else (self.width / 2, self.height / 2)

    def boundary(self):
        b = rectangle_boundary(0, 0, self.width, self.height)
        b.pieces.append(ArcPiece(self._c(), self.hole_radius, 0.0, 2 * np.pi, "hole"))
        b.coordinate_name = "t"
        return b

    def mesh(self, nx: int = 24, h: float | None = None, grade: float = 0.4, **kw) -> Mesh2D:
        h = _h_from(min(self.width, self.height), h, nx)
        cx, cy = self._c()
        R = self.hole_radius
        rect = sdf_polygon(np.array([(0, 0), (self.width, 0), (self.width, self.height), (0, self.height)]))
        circ = sdf_circle(cx, cy, R)
        fd = lambda p: np.maximum(rect(p), -circ(p))
        hmin = max(grade, 0.05) * h if grade else h
        fh = lambda p: np.minimum(1.0, (hmin + 0.3 * np.maximum(circ(p), 0)) / h) * h / hmin if grade else np.ones(len(p))
        m = int(np.ceil(2 * np.pi * R / hmin))
        th = 2 * np.pi * np.arange(m) / m
        pfix = np.vstack((_edge_fix(np.array([(0, 0), (self.width, 0), (self.width, self.height), (0, self.height)]), h),
                          np.column_stack((cx + R * np.cos(th), cy + R * np.sin(th)))))
        inter = []
        for (x0, y0, x1, y1) in ((0, 0, self.width, 0), (self.width, 0, self.width, self.height),
                                 (self.width, self.height, 0, self.height), (0, self.height, 0, 0)):
            dx, dy = x1 - x0, y1 - y0
            a, b, c = dx * dx + dy * dy, 2 * ((x0 - cx) * dx + (y0 - cy) * dy), (x0 - cx) ** 2 + (y0 - cy) ** 2 - R * R
            disc = b * b - 4 * a * c
            if disc >= 0:
                for sgn in (-1, 1):
                    s = (-b + sgn * np.sqrt(disc)) / (2 * a)
                    if 0 <= s <= 1:
                        inter.append((x0 + s * dx, y0 + s * dy))
        pfix = np.vstack([pfix] + ([np.array(inter)] if inter else []))
        pfix = pfix[fd(pfix) < 1e-9 * h]
        if inter:
            ip = np.array(inter)
            far = np.min(np.linalg.norm(pfix[:, None] - ip[None], axis=2), axis=1) > 0.3 * hmin
            pfix = np.vstack((pfix[far], ip))
        p, t = distmesh(fd, hmin, ((0, 0), (self.width, self.height)), pfix=pfix, fh=fh)
        return _finish(p, t, self.name, self.boundary())


@dataclass
class AnnulusDomain:
    r_inner: float = 0.5
    r_outer: float = 1.0

    name = "annulus"

    def boundary(self):
        return Boundary([ArcPiece((0, 0), self.r_outer, 0, 2 * np.pi, "outer_circle"),
                         ArcPiece((0, 0), self.r_inner, 0, 2 * np.pi, "inner_circle")], "t", 2 * np.pi)

    def mesh(self, nx: int = 16, h: float | None = None, **kw) -> Mesh2D:
        h = _h_from(self.r_outer - self.r_inner, h, max(2, nx // 4))
        fd = lambda p: np.maximum(np.hypot(p[:, 0], p[:, 1]) - self.r_outer, self.r_inner - np.hypot(p[:, 0], p[:, 1]))
        fix = []
        for R in (self.r_inner, self.r_outer):
            m = int(np.ceil(2 * np.pi * R / h))
            th = 2 * np.pi * np.arange(m) / m
            fix.append(np.column_stack((R * np.cos(th), R * np.sin(th))))
        R = self.r_outer
        p, t = distmesh(fd, h, ((-R, -R), (R, R)), pfix=np.vstack(fix))
        return _finish(p, t, self.name, self.boundary())


@dataclass
class ParametricDomain:
    """Closed curve x(t), y(t) given as a key=value block (name, x, y, t0, t1, boundary_markers)."""

    definition: str

    @property
    def name(self) -> str:
        return _parse_block(self.definition).get("name", "parametric")

    def _curve(self, nsamples: int):
        d = _parse_block(self.definition)
        fx, fy = Expr(d.get("x", "cos(t)")), Expr(d.get("y", "sin(t)"))
        t0, t1 = numeric(d.get("t0", "0")), numeric(d.get("t1", "2*pi"))
        ts = np.linspace(t0, t1, nsamples + 1)
        z = np.zeros_like(ts)
        pts = np.column_stack((fx(z, z, ts), fy(z, z, ts)))
        k = max(1, int(numeric(d.get("boundary_markers", d.get("markers", "8")))))
        return pts, ts, t0, t1, k

    def boundary(self, nsamples: int = 2048):
        pts, ts, t0, t1, k = self._curve(nsamples)
        pieces = []
        idx = np.linspace(0, nsamples, k + 1).round().astype(int)
        for i in range(k):
            sl = slice(idx[i], idx[i + 1] + 1)
            pieces.append(PolylinePiece(pts[sl], ts[sl], f"curve_{i}"))
        return Boundary(pieces, "t", period=t1 - t0, t0=t0)

    def mesh(self, nx: int = 24, h: float | None = None, **kw) -> Mesh2D:
        pts, ts, *_ = self._curve(2048)
        poly = pts[:-1]
        area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
        if area < 0:
            poly = poly[::-1]
        lo, hi = poly.min(0), poly.max(0)
        h = _h_from(min(hi - lo), h, nx)
        per = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        m = max(8, int(np.ceil(per[-1] / h)))
        s = np.linspace(0, per[-1], m, endpoint=False)
        pfix = np.column_stack((np.interp(s, per, pts[:, 0]), np.interp(s, per, pts[:, 1])))
        p, t = distmesh(sdf_polygon(poly), h, (lo, hi), pfix=pfix)
        return _finish(p, t, self.name, self.boundary())


def _parse_block(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in str(text).splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        sep = "=" if "=" in line else ":" if ":" in line else None
        if sep is None:
            raise ValueError(f"expected key = value: {raw!r}")
        k, v = line.split(sep, 1)
        out[k.strip().lower()] = v.strip()
    return out


def _self_intersects(v: np.ndarray) -> bool:
    n = len(v)

    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    for i in range(n):
        a, b = v[i], v[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = v[j], v[(j + 1) % n]
            if orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0:
                return True
    return False


DOMAINS = {
    "Square": lambda: SquareDomain(1.0),
    "Rectangle": lambda: RectangleDomain(2.0, 1.0),
    "Disk": lambda: DiskDomain(1.0),
    "L-shape": lambda: LShapeDomain(1.0),
    "Annulus": lambda: AnnulusDomain(0.5, 1.0),
    "Plate with hole": lambda: PlateWithHoleDomain(2.0, 1.0, None, 0.2),
}
