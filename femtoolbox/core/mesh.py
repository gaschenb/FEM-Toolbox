from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Sequence

import numpy as np
from scipy.spatial import Delaunay

from femtoolbox.core.basis import LOCAL_EDGES


def _fmt(v: float) -> str:
    v = float(v)
    if abs(v) < 1e-13:
        v = 0.0
    return f"{v:.6g}"


# ----------------------------------------------------------------------------- boundary geometry
@dataclass
class SegmentPiece:
    a: tuple[float, float]
    b: tuple[float, float]
    marker: str
    label: str = ""

    def __post_init__(self):
        self.a = np.asarray(self.a, float)
        self.b = np.asarray(self.b, float)
        if not self.label:
            d = self.b - self.a
            if abs(d[0]) < 1e-12:
                eq = f"x = {_fmt(self.a[0])}"
            elif abs(d[1]) < 1e-12:
                eq = f"y = {_fmt(self.a[1])}"
            else:
                eq = f"({_fmt(self.a[0])},{_fmt(self.a[1])})->({_fmt(self.b[0])},{_fmt(self.b[1])})"
            self.label = f"{self.marker}: {eq}"

    def _s(self, p):
        d = self.b - self.a
        return np.clip(((p - self.a) @ d) / max(d @ d, 1e-300), 0.0, 1.0)

    def project(self, p):
        return self.a + self._s(p)[:, None] * (self.b - self.a)

    def distance(self, p):
        return np.linalg.norm(p - self.project(p), axis=1)

    def coordinate(self, p):
        return self._s(p)


@dataclass
class ArcPiece:
    center: tuple[float, float]
    radius: float
    th0: float
    th1: float
    marker: str
    label: str = ""

    def __post_init__(self):
        self.center = np.asarray(self.center, float)
        if not self.label:
            self.label = (f"{self.marker}: |x-c| = {_fmt(self.radius)}, theta in "
                          f"[{_fmt(np.degrees(self.th0))}, {_fmt(np.degrees(self.th1))}] deg")

    def _theta(self, p):
        d = p - self.center
        return np.mod(np.arctan2(d[:, 1], d[:, 0]), 2 * np.pi)

    def _inside(self, th):
        span = np.mod(self.th1 - self.th0, 2 * np.pi) or 2 * np.pi
        return np.mod(th - self.th0, 2 * np.pi) <= span + 1e-12

    def project(self, p):
        d = p - self.center
        r = np.maximum(np.linalg.norm(d, axis=1), 1e-300)
        q = self.center + self.radius * d / r[:, None]
        th = self._theta(p)
        out = ~self._inside(th)
        if np.any(out):
            ea = self.center + self.radius * np.array([np.cos(self.th0), np.sin(self.th0)])
            eb = self.center + self.radius * np.array([np.cos(self.th1), np.sin(self.th1)])
            po = p[out]
            da = np.linalg.norm(po - ea, axis=1)
            db = np.linalg.norm(po - eb, axis=1)
            q[out] = np.where((da <= db)[:, None], ea, eb)
        return q

    def distance(self, p):
        return np.linalg.norm(p - self.project(p), axis=1)

    def coordinate(self, p):
        return self._theta(p)


@dataclass
class PolylinePiece:
    points: np.ndarray
    params: np.ndarray
    marker: str
    label: str = ""

    def __post_init__(self):
        self.points = np.asarray(self.points, float)
        self.params = np.asarray(self.params, float)
        if not self.label:
            self.label = f"{self.marker}: t in [{_fmt(self.params[0])}, {_fmt(self.params[-1])}]"

    def _nearest(self, p):
        A, B = self.points[:-1], self.points[1:]
        D = B - A
        L2 = np.maximum((D * D).sum(1), 1e-300)
        best_d = np.full(len(p), np.inf)
        best_q = np.zeros_like(p)
        best_t = np.zeros(len(p))
        for s0 in range(0, len(A), 512):
            a, d, l2 = A[s0:s0 + 512], D[s0:s0 + 512], L2[s0:s0 + 512]
            s = np.clip(((p[:, None, :] - a[None]) * d[None]).sum(-1) / l2[None], 0, 1)
            q = a[None] + s[..., None] * d[None]
            dist = np.linalg.norm(p[:, None, :] - q, axis=-1)
            k = dist.argmin(1)
            idx = np.arange(len(p))
            better = dist[idx, k] < best_d
            best_d[better] = dist[idx, k][better]
            best_q[better] = q[idx, k][better]
            t0, t1 = self.params[s0 + k], self.params[s0 + k + 1]
            best_t[better] = (t0 + s[idx, k] * (t1 - t0))[better]
        return best_q, best_d, best_t

    def project(self, p):
        return self._nearest(p)[0]

    def distance(self, p):
        return self._nearest(p)[1]

    def coordinate(self, p):
        return self._nearest(p)[2]


@dataclass
class Boundary:
    pieces: list = field(default_factory=list)
    coordinate_name: str = "s"
    period: float = 0.0
    t0: float = 0.0

    def classify(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.atleast_2d(p)
        D = np.column_stack([pc.distance(p) for pc in self.pieces])
        k = D.argmin(1)
        return k, D[np.arange(len(p)), k]

    def markers(self, p):
        k, _ = self.classify(p)
        names = np.array([pc.marker for pc in self.pieces], dtype=object)
        return names[k]

    def project(self, p):
        p = np.atleast_2d(np.asarray(p, float))
        if not self.pieces:
            return p.copy()
        k, _ = self.classify(p)
        out = p.copy()
        for i, pc in enumerate(self.pieces):
            m = k == i
            if np.any(m):
                out[m] = pc.project(p[m])
        return out

    def coordinates(self, p):
        p = np.atleast_2d(p)
        k, _ = self.classify(p)
        t = np.zeros(len(p))
        for i, pc in enumerate(self.pieces):
            m = k == i
            if np.any(m):
                t[m] = pc.coordinate(p[m])
        return t

    def labels(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for pc in self.pieces:
            out.setdefault(pc.marker, pc.label)
        return out


def polygon_boundary(vertices, names=None) -> Boundary:
    v = np.asarray(vertices, float)
    n = len(v)
    names = names or [f"edge_{i}" for i in range(n)]
    return Boundary([SegmentPiece(v[i], v[(i + 1) % n], names[i]) for i in range(n)])


def rectangle_boundary(x0, y0, x1, y1) -> Boundary:
    return polygon_boundary([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], ["bottom", "right", "top", "left"])


# ----------------------------------------------------------------------------- mesh
class Mesh2D:
    """Triangle mesh. Vertex 0 of each triangle is the newest vertex (refinement edge = local edge 0)."""

    def __init__(self, nodes, triangles, name: str = "mesh", boundary: Boundary | None = None, keep_order: bool = False):
        self.nodes = np.ascontiguousarray(nodes, dtype=float)
        tris = np.ascontiguousarray(triangles, dtype=np.int64)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2 or tris.ndim != 2 or tris.shape[1] != 3:
            raise ValueError("nodes must be (n,2) and triangles (m,3)")
        P = self.nodes[tris]
        det = (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0])
        neg = det < 0
        if np.any(neg):
            tris[neg] = tris[neg][:, [0, 2, 1]]
        self.triangles = tris
        self.name = name
        self.boundary = boundary

    # --------------------------------------------------------------- sizes
    @property
    def nnodes(self) -> int:
        return len(self.nodes)

    @property
    def nelements(self) -> int:
        return len(self.triangles)

    @property
    def nedges(self) -> int:
        return len(self.edges)

    @property
    def bbox(self):
        lo, hi = self.nodes.min(0), self.nodes.max(0)
        return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])

    # --------------------------------------------------------------- topology
    @cached_property
    def _topology(self):
        M = self.nelements
        le = self.triangles[:, np.array(LOCAL_EDGES)]  # (M,3,2)
        key = np.sort(le, axis=2).reshape(-1, 2)
        edges, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        tri_edges = inv.reshape(M, 3)
        E = len(edges)
        order = np.argsort(inv, kind="stable")
        first = np.full(E, -1)
        second = np.full(E, -1)
        sorted_inv = inv[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_inv)) + 1]
        first[sorted_inv[starts]] = order[starts]
        counts = np.diff(np.r_[starts, len(order)])
        two = starts[counts >= 2]
        second[sorted_inv[two]] = order[two + 1]
        edge_cells = np.column_stack((np.where(first >= 0, first // 3, -1), np.where(second >= 0, second // 3, -1)))
        edge_local = np.column_stack((np.where(first >= 0, first % 3, -1), np.where(second >= 0, second % 3, -1)))
        return edges, tri_edges, edge_cells, edge_local

    @property
    def edges(self):
        return self._topology[0]

    @property
    def tri_edges(self):
        return self._topology[1]

    @property
    def edge_cells(self):
        return self._topology[2]

    @property
    def edge_local(self):
        return self._topology[3]

    @cached_property
    def boundary_edge_ids(self) -> np.ndarray:
        return np.flatnonzero(self.edge_cells[:, 1] < 0)

    @cached_property
    def interior_edge_ids(self) -> np.ndarray:
        return np.flatnonzero(self.edge_cells[:, 1] >= 0)

    @cached_property
    def boundary_nodes(self) -> np.ndarray:
        return np.unique(self.edges[self.boundary_edge_ids])

    @cached_property
    def edge_markers(self) -> np.ndarray:
        """Marker for each boundary edge (aligned with boundary_edge_ids)."""
        mids = self.nodes[self.edges[self.boundary_edge_ids]].mean(1)
        if self.boundary is not None and self.boundary.pieces:
            return self.boundary.markers(mids)
        xmin, xmax, ymin, ymax = self.bbox
        tol = 1e-9 * max(xmax - xmin, ymax - ymin, 1.0)
        out = np.full(len(mids), "outer", dtype=object)
        out[np.abs(mids[:, 1] - ymax) < tol] = "top"
        out[np.abs(mids[:, 1] - ymin) < tol] = "bottom"
        out[np.abs(mids[:, 0] - xmax) < tol] = "right"
        out[np.abs(mids[:, 0] - xmin) < tol] = "left"
        return out

    def boundary_marker_names(self) -> list[str]:
        seen: list[str] = []
        for m in self.edge_markers:
            if m not in seen:
                seen.append(m)
        if self.boundary is not None:
            order = [pc.marker for pc in self.boundary.pieces]
            seen.sort(key=lambda m: order.index(m) if m in order else 1e9)
        return seen

    def boundary_marker_label(self, marker: str) -> str:
        if self.boundary is not None:
            return self.boundary.labels().get(marker, marker)
        xmin, xmax, ymin, ymax = self.bbox
        return {"left": f"left: x = {_fmt(xmin)}", "right": f"right: x = {_fmt(xmax)}",
                "bottom": f"bottom: y = {_fmt(ymin)}", "top": f"top: y = {_fmt(ymax)}"}.get(marker, marker)

    def boundary_marker_descriptions(self) -> list[tuple[str, str]]:
        return [(m, self.boundary_marker_label(m)) for m in self.boundary_marker_names()]

    def boundary_coordinate(self, p) -> np.ndarray | None:
        if self.boundary is None or not self.boundary.pieces:
            return None
        return self.boundary.coordinates(np.atleast_2d(p))

    # --------------------------------------------------------------- geometry
    @cached_property
    def _geometry(self):
        P = self.nodes[self.triangles]
        J = np.stack((P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=2)  # columns
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        invJ = np.empty_like(J)
        invJ[:, 0, 0] = J[:, 1, 1] / det
        invJ[:, 1, 1] = J[:, 0, 0] / det
        invJ[:, 0, 1] = -J[:, 0, 1] / det
        invJ[:, 1, 0] = -J[:, 1, 0] / det
        return P[:, 0].copy(), J, det, invJ

    @property
    def x0(self):
        return self._geometry[0]

    @property
    def J(self):
        return self._geometry[1]

    @property
    def detJ(self):
        return self._geometry[2]

    @property
    def invJ(self):
        return self._geometry[3]

    @cached_property
    def areas(self) -> np.ndarray:
        return 0.5 * np.abs(self.detJ)

    @cached_property
    def edge_lengths(self) -> np.ndarray:
        d = self.nodes[self.edges[:, 1]] - self.nodes[self.edges[:, 0]]
        return np.hypot(d[:, 0], d[:, 1])

    @cached_property
    def diameters(self) -> np.ndarray:
        return self.edge_lengths[self.tri_edges].max(1)

    def cell_area(self, c):
        return float(self.areas[c])

    def cell_diameter(self, c):
        return float(self.diameters[c])

    def representative_h(self) -> float:
        return float(np.sqrt(np.mean(self.areas))) if self.nelements else 0.0

    def quality(self) -> np.ndarray:
        """4*sqrt(3)*area / sum(l^2): 1 for equilateral, 0 for degenerate."""
        l2 = (self.edge_lengths[self.tri_edges] ** 2).sum(1)
        return 4.0 * np.sqrt(3.0) * self.areas / np.maximum(l2, 1e-300)

    def physical(self, ref: np.ndarray, cells=None) -> np.ndarray:
        """Map reference points (q,2) to physical points (m,q,2)."""
        x0, J = (self.x0, self.J) if cells is None else (self.x0[cells], self.J[cells])
        return x0[:, None, :] + np.einsum("eab,qb->eqa", J, ref)

    def edge_normals(self, edge_ids, side: int = 0):
        """Unit outward normal of the side-cell for each edge; also the edge vertex coords oriented CCW for that cell."""
        cells = self.edge_cells[edge_ids, side]
        le = self.edge_local[edge_ids, side]
        loc = np.array(LOCAL_EDGES)[le]
        a = self.triangles[cells, loc[:, 0]]
        b = self.triangles[cells, loc[:, 1]]
        t = self.nodes[b] - self.nodes[a]
        L = np.hypot(t[:, 0], t[:, 1])
        n = np.column_stack((t[:, 1], -t[:, 0])) / L[:, None]
        return n, a, b, L

    @cached_property
    def trifinder(self):
        import matplotlib.tri as mtri
        return mtri.Triangulation(self.nodes[:, 0], self.nodes[:, 1], self.triangles).get_trifinder()

    def locate(self, x, y):
        """Return (cell, reference coords) for physical points; cell = -1 outside."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        cells = np.asarray(self.trifinder(x, y), dtype=int)
        ref = np.zeros((len(x), 2))
        ok = cells >= 0
        if np.any(ok):
            d = np.column_stack((x[ok], y[ok])) - self.x0[cells[ok]]
            ref[ok] = np.einsum("eab,eb->ea", self.invJ[cells[ok]], d)
        return cells, ref

    def copy(self) -> "Mesh2D":
        return Mesh2D(self.nodes.copy(), self.triangles.copy(), self.name, self.boundary)

    def with_nodes(self, nodes) -> "Mesh2D":
        return Mesh2D(nodes, self.triangles.copy(), self.name, self.boundary)

    def summary(self) -> str:
        q = self.quality()
        return (f"{self.name}: {self.nnodes} nodes, {self.nelements} triangles, {self.nedges} edges, "
                f"h_max={self.diameters.max():.3g}, h_min={self.diameters.min():.3g}, "
                f"quality min/mean={q.min():.2f}/{q.mean():.2f}")


# ----------------------------------------------------------------------------- generators
def label_longest_edge(tris: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Rotate each triangle so its longest edge is local edge 0 (initial newest-vertex labelling)."""
    P = nodes[tris]
    l = np.stack([np.linalg.norm(P[:, b] - P[:, a], axis=1) for a, b in LOCAL_EDGES], 1)
    k = l.argmax(1)
    idx = (np.arange(3)[None, :] + k[:, None]) % 3
    return np.take_along_axis(tris, idx, 1)


def structured_rectangle(x0, y0, x1, y1, nx, ny, pattern="alternating") -> tuple[np.ndarray, np.ndarray]:
    nx, ny = max(1, int(nx)), max(1, int(ny))
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    X, Y = np.meshgrid(xs, ys)
    nodes = np.column_stack((X.ravel(), Y.ravel()))
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    i, j = i.ravel(), j.ravel()
    n00 = j * (nx + 1) + i
    n10, n01, n11 = n00 + 1, n00 + nx + 1, n00 + nx + 2
    if pattern == "alternating":
        flip = ((i + j) % 2).astype(bool)
    else:
        flip = np.zeros_like(i, dtype=bool)
    t1 = np.where(flip[:, None], np.column_stack((n00, n10, n01)), np.column_stack((n00, n10, n11)))
    t2 = np.where(flip[:, None], np.column_stack((n10, n11, n01)), np.column_stack((n00, n11, n01)))
    return nodes, np.vstack((t1, t2))


def compress(nodes, tris):
    used = np.unique(tris)
    inv = -np.ones(len(nodes), dtype=np.int64)
    inv[used] = np.arange(len(used))
    return nodes[used], inv[tris]


def distmesh(fd: Callable, h0: float, bbox, pfix=None, fh: Callable | None = None, maxiter: int = 200,
             seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Persson-Strang DistMesh: force-equilibrium boundary-fitted triangulation of {fd < 0}."""
    rng = np.random.default_rng(seed)
    dptol, ttol, Fscale, deltat = 1e-3, 0.1, 1.2, 0.2
    geps = 1e-3 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0
    fh = fh or (lambda p: np.ones(len(p)))
    (xmin, ymin), (xmax, ymax) = bbox
    x, y = np.meshgrid(np.arange(xmin, xmax + h0 * 0.5, h0), np.arange(ymin, ymax + h0 * 0.5, h0 * np.sqrt(3) / 2))
    x[1::2] += h0 / 2
    p = np.column_stack((x.ravel(), y.ravel()))
    p = p[fd(p) < geps]
    r0 = 1.0 / fh(p) ** 2
    p = p[rng.random(len(p)) < r0 / r0.max()]
    pfix = np.zeros((0, 2)) if pfix is None or len(pfix) == 0 else np.unique(np.asarray(pfix, float), axis=0)
    if len(pfix):
        keep = np.min(np.linalg.norm(p[:, None] - pfix[None], axis=2), axis=1) > 0.5 * h0 * np.min(fh(pfix)) if len(p) else []
        p = np.vstack((pfix, p[keep]))
    nfix = len(pfix)
    pold = np.inf
    t = None
    for it in range(maxiter):
        if t is None or np.max(np.linalg.norm(p - pold, axis=1)) / h0 > ttol:
            pold = p.copy()
            t = Delaunay(p).simplices
            pmid = p[t].mean(1)
            t = t[fd(pmid) < -geps]
            bars = np.unique(np.sort(np.vstack((t[:, [0, 1]], t[:, [1, 2]], t[:, [0, 2]])), axis=1), axis=0)
        barvec = p[bars[:, 0]] - p[bars[:, 1]]
        L = np.linalg.norm(barvec, axis=1)
        hbars = fh(p[bars].mean(1))
        L0 = hbars * Fscale * np.sqrt((L ** 2).sum() / (hbars ** 2).sum())
        F = np.maximum(L0 - L, 0)
        Fvec = (F / L)[:, None] * barvec
        Ftot = np.zeros_like(p)
        np.add.at(Ftot, bars[:, 0], Fvec)
        np.add.at(Ftot, bars[:, 1], -Fvec)
        Ftot[:nfix] = 0
        p = p + deltat * Ftot
        d = fd(p)
        ix = d > 0
        if np.any(ix):
            pi = p[ix]
            dgx = (fd(pi + [deps, 0]) - d[ix]) / deps
            dgy = (fd(pi + [0, deps]) - d[ix]) / deps
            g2 = dgx ** 2 + dgy ** 2 + 1e-300
            p[ix] -= np.column_stack((d[ix] * dgx / g2, d[ix] * dgy / g2))
            p[:nfix] = pfix
        move = np.abs(deltat * Ftot[d < -geps]).max() / h0 if np.any(d < -geps) else 0
        if move < dptol:
            break
    t = Delaunay(p).simplices
    t = t[fd(p[t].mean(1)) < -geps]
    P = p[t]
    area = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
    t = t[area > 1e-12 * h0 * h0]
    return compress(p, t)


# ----------------------------------------------------------------------------- refinement
def _project_new(nodes, boundary, idx):
    if boundary is not None and boundary.pieces and len(idx):
        nodes[idx] = boundary.project(nodes[idx])


def bisect(mesh: Mesh2D, marked, project: bool = True) -> tuple[Mesh2D, np.ndarray]:
    """Conforming newest-vertex bisection of marked cells. Returns (mesh, parent index per new cell)."""
    marked = np.unique(np.asarray(marked, dtype=np.int64))
    if marked.size == 0:
        return mesh.copy(), np.arange(mesh.nelements)
    edges, tri_edges = mesh.edges, mesh.tri_edges
    E = len(edges)
    cut = np.zeros(E, dtype=bool)
    cut[tri_edges[marked, 0]] = True
    while True:
        need = cut[tri_edges].any(1) & ~cut[tri_edges[:, 0]]
        if not need.any():
            break
        cut[tri_edges[need, 0]] = True
    cut_ids = np.flatnonzero(cut)
    N = mesh.nnodes
    new_nodes = 0.5 * (mesh.nodes[edges[cut_ids, 0]] + mesh.nodes[edges[cut_ids, 1]])
    mid_of = -np.ones(E, dtype=np.int64)
    mid_of[cut_ids] = N + np.arange(len(cut_ids))
    nodes = np.vstack((mesh.nodes, new_nodes))
    if project and mesh.boundary is not None:
        on_b = mesh.edge_cells[cut_ids, 1] < 0
        _project_new(nodes, mesh.boundary, N + np.flatnonzero(on_b))
    key_scale = np.int64(N + len(cut_ids) + 1)
    cut_keys = np.sort(edges[cut_ids], axis=1)
    cut_keys = cut_keys[:, 0] * key_scale + cut_keys[:, 1]
    order = np.argsort(cut_keys)
    cut_keys_sorted = cut_keys[order]
    mids_sorted = mid_of[cut_ids][order]

    def lookup(a, b):
        k = np.minimum(a, b) * key_scale + np.maximum(a, b)
        pos = np.searchsorted(cut_keys_sorted, k)
        pos = np.minimum(pos, len(cut_keys_sorted) - 1)
        hit = cut_keys_sorted[pos] == k
        return np.where(hit, mids_sorted[pos], -1)

    tris = mesh.triangles.copy()
    parent = np.arange(len(tris))
    for _ in range(8):
        m = lookup(tris[:, 1], tris[:, 2])
        sel = m >= 0
        if not sel.any():
            break
        p1, p2, p3, p4 = tris[sel, 0], tris[sel, 1], tris[sel, 2], m[sel]
        left = np.column_stack((p4, p1, p2))
        right = np.column_stack((p4, p3, p1))
        keep = ~sel
        par = parent[sel]
        tris = np.vstack((tris[keep], left, right))
        parent = np.concatenate((parent[keep], par, par))
    out = Mesh2D(nodes, tris, mesh.name, mesh.boundary)
    return out, parent


def refine_uniform(mesh: Mesh2D, project: bool = True) -> tuple[Mesh2D, np.ndarray]:
    """Red refinement (each triangle -> 4 similar ones). Returns (mesh, parent)."""
    edges, te = mesh.edges, mesh.tri_edges
    N = mesh.nnodes
    mids = 0.5 * (mesh.nodes[edges[:, 0]] + mesh.nodes[edges[:, 1]])
    nodes = np.vstack((mesh.nodes, mids))
    if project and mesh.boundary is not None:
        _project_new(nodes, mesh.boundary, N + mesh.boundary_edge_ids)
    a, b, c = mesh.triangles.T
    m0, m1, m2 = N + te[:, 0], N + te[:, 1], N + te[:, 2]  # m0 on (b,c), m1 on (c,a), m2 on (a,b)
    tris = np.vstack((np.column_stack((a, m2, m1)), np.column_stack((m2, b, m0)),
                      np.column_stack((m1, m0, c)), np.column_stack((m0, m1, m2))))
    parent = np.tile(np.arange(mesh.nelements), 4)
    out = Mesh2D(nodes, tris, mesh.name, mesh.boundary)
    out.triangles = label_longest_edge(out.triangles, out.nodes)
    return out, parent


def uniform_refine(mesh: Mesh2D) -> Mesh2D:
    return refine_uniform(mesh)[0]


def refine_h_marked(mesh: Mesh2D, marked) -> Mesh2D:
    return bisect(mesh, marked)[0]


def smooth_laplacian(mesh: Mesh2D, iterations: int = 3) -> Mesh2D:
    """Laplacian smoothing of interior nodes, rejected per-step if any triangle inverts."""
    nodes = mesh.nodes.copy()
    E = mesh.edges
    interior = np.ones(mesh.nnodes, bool)
    interior[mesh.boundary_nodes] = False
    deg = np.bincount(E.ravel(), minlength=mesh.nnodes).astype(float)
    for _ in range(iterations):
        acc = np.zeros_like(nodes)
        np.add.at(acc, E[:, 0], nodes[E[:, 1]])
        np.add.at(acc, E[:, 1], nodes[E[:, 0]])
        trial = nodes.copy()
        trial[interior] = acc[interior] / np.maximum(deg[interior, None], 1)
        P = trial[mesh.triangles]
        det = (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0])
        if np.any(det <= 0):
            break
        nodes = trial
    return Mesh2D(nodes, mesh.triangles.copy(), mesh.name, mesh.boundary)


# ----------------------------------------------------------------------------- curved (isoparametric) geometry
class GeoEval:
    __slots__ = ("X", "J", "det", "invJ")

    def __init__(self, X, J, det, invJ):
        self.X, self.J, self.det, self.invJ = X, J, det, invJ


def _inv2(J):
    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    inv = np.empty_like(J)
    inv[..., 0, 0] = J[..., 1, 1] / det
    inv[..., 1, 1] = J[..., 0, 0] / det
    inv[..., 0, 1] = -J[..., 0, 1] / det
    inv[..., 1, 0] = -J[..., 1, 0] / det
    return det, inv


class Geometry:
    """Degree-g Lagrange geometry map; cells on curved boundary pieces get edge-bubble blended nodes (Solin)."""

    def __init__(self, mesh: "Mesh2D", order: int):
        from femtoolbox.core.basis import get_basis, lattice_nodes
        self.mesh = mesh
        self.order = g = max(1, int(order))
        self.is_curved = np.zeros(mesh.nelements, bool)
        self.slot = -np.ones(mesh.nelements, dtype=np.int64)
        self.cell_nodes = np.zeros((0, 1, 2))
        if g < 2 or mesh.boundary is None or not mesh.boundary.pieces:
            return
        curved_piece = np.array([not isinstance(pc, SegmentPiece) for pc in mesh.boundary.pieces])
        if not curved_piece.any():
            return
        be = mesh.boundary_edge_ids
        mids = mesh.nodes[mesh.edges[be]].mean(1)
        k, _ = mesh.boundary.classify(mids)
        ce = be[curved_piece[k]]
        if not len(ce):
            return
        self.basis = get_basis(g, "lagrange")
        ref = self.basis.nodes
        lam = np.column_stack((1 - ref[:, 0] - ref[:, 1], ref[:, 0], ref[:, 1]))
        cells = np.unique(mesh.edge_cells[ce, 0])
        self.is_curved[cells] = True
        self.slot[cells] = np.arange(len(cells))
        nodes = mesh.physical(ref, cells)
        disp = np.zeros_like(nodes)
        cc, le = mesh.edge_cells[ce, 0], mesh.edge_local[ce, 0]
        for e in range(3):
            sel = le == e
            if not sel.any():
                continue
            a, b = LOCAL_EDGES[e]
            tri = mesh.triangles[cc[sel]]
            xa, xb = mesh.nodes[tri[:, a]], mesh.nodes[tri[:, b]]
            la, lb = lam[:, a], lam[:, b]
            bub = la * lb
            s = np.clip(0.5 * (1.0 + lb - la), 1e-12, 1 - 1e-12)
            chord = xa[:, None, :] + s[None, :, None] * (xb - xa)[:, None, :]
            proj = mesh.boundary.project(chord.reshape(-1, 2)).reshape(chord.shape)
            fac = np.where(bub > 1e-14, bub / (s * (1 - s)), 0.0)
            d = (proj - chord) * fac[None, :, None]
            np.add.at(disp, self.slot[cc[sel]], d)
        self.cell_nodes = nodes + disp

    @property
    def any_curved(self) -> bool:
        return bool(self.is_curved.any())

    def at(self, ref: np.ndarray, cells=None) -> GeoEval:
        """Physical points, Jacobians (e,q,2,2) at reference points (q,2)."""
        m = self.mesh
        cells = np.arange(m.nelements) if cells is None else np.asarray(cells)
        nq = len(ref)
        X = m.physical(ref, cells)
        J = np.broadcast_to(m.J[cells][:, None], (len(cells), nq, 2, 2)).copy()
        if self.any_curved:
            cm = self.is_curved[cells]
            if cm.any():
                cn = self.cell_nodes[self.slot[cells[cm]]]
                X[cm] = np.einsum("qk,ekd->eqd", self.basis.eval(ref), cn)
                J[cm] = np.einsum("qkb,eka->eqab", self.basis.grad(ref), cn)
        det, inv = _inv2(J)
        return GeoEval(X, J, det, inv)

    def at_points(self, cells: np.ndarray, ref: np.ndarray) -> GeoEval:
        """One reference point per entry (n,2) in the given cells."""
        m = self.mesh
        X = m.x0[cells] + np.einsum("eab,eb->ea", m.J[cells], ref)
        J = m.J[cells].copy()
        if self.any_curved:
            cm = self.is_curved[cells]
            if cm.any():
                cn = self.cell_nodes[self.slot[cells[cm]]]
                phi = self.basis.eval(ref[cm])
                gp = self.basis.grad(ref[cm])
                X[cm] = np.einsum("ek,ekd->ed", phi, cn)
                J[cm] = np.einsum("ekb,eka->eab", gp, cn)
        det, inv = _inv2(J)
        return GeoEval(X, J, det, inv)

    def inverse_map(self, cells: np.ndarray, x: np.ndarray, ref0: np.ndarray) -> np.ndarray:
        ref = ref0.copy()
        if not self.any_curved:
            return ref
        cm = self.is_curved[cells]
        if not cm.any():
            return ref
        for _ in range(8):
            g = self.at_points(cells[cm], ref[cm])
            r = x[cm] - g.X
            ref[cm] += np.einsum("eab,eb->ea", g.invJ, r)
        return ref

    def edge(self, eids: np.ndarray, side: int, s: np.ndarray, reverse: bool = False):
        """Edge traces for side-cell: X (e,q,2), outward normal (e,q,2), ds weight (e,q), invJ (e,q,2,2)."""
        from femtoolbox.core.basis import REF_VERTS
        m = self.mesh
        cells = m.edge_cells[eids, side]
        le = m.edge_local[eids, side]
        nq = len(s)
        X = np.zeros((len(eids), nq, 2))
        nrm = np.zeros_like(X)
        ds = np.zeros((len(eids), nq))
        inv = np.zeros((len(eids), nq, 2, 2))
        ss = 1.0 - s if reverse else s
        for e in range(3):
            sel = np.flatnonzero(le == e)
            if not len(sel):
                continue
            a, b = LOCAL_EDGES[e]
            ref = REF_VERTS[a][None] + ss[:, None] * (REF_VERTS[b] - REF_VERTS[a])[None]
            g = self.at(ref, cells[sel])
            t = np.einsum("eqab,b->eqa", g.J, REF_VERTS[b] - REF_VERTS[a])
            L = np.hypot(t[..., 0], t[..., 1])
            X[sel], ds[sel], inv[sel] = g.X, L, g.invJ
            nrm[sel] = np.stack((t[..., 1], -t[..., 0]), -1) / L[..., None]
        return X, nrm, ds, inv


def _geometry(self, order: int) -> Geometry:
    cache = self.__dict__.setdefault("_geo_cache", {})
    key = max(1, int(order))
    if key not in cache:
        cache[key] = Geometry(self, key)
    return cache[key]


Mesh2D.geometry = _geometry
