from __future__ import annotations

from functools import lru_cache

import numpy as np

from femtoolbox.core.quadrature import triangle_rule

REF_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
LOCAL_EDGES = ((1, 2), (2, 0), (0, 1))  # edge e is opposite vertex e
REPRESENTATIONS = ("lagrange", "modal")
_ALIASES = {"lagrange-nodal": "lagrange", "nodal": "lagrange", "modal-legendre": "modal", "orthonormal": "modal",
            "hierarchical-lobatto": "modal", "dubiner": "modal"}


def canonical_representation(name: str) -> str:
    name = str(name or "lagrange").strip().lower()
    name = _ALIASES.get(name, name)
    if name not in REPRESENTATIONS:
        raise ValueError(f"unknown basis {name!r}; use one of {REPRESENTATIONS}")
    return name


def _exponents(p: int) -> np.ndarray:
    return np.array([(k - j, j) for k in range(p + 1) for j in range(k + 1)], dtype=int)


def lattice_nodes(p: int) -> tuple[np.ndarray, list[str], list[int]]:
    """Equispaced nodes ordered vertices, edges (each oriented as LOCAL_EDGES), interior."""
    if p == 0:
        return np.array([[1.0 / 3.0, 1.0 / 3.0]]), ["interior"], [0]
    pts, kind, ent = [], [], []
    for v in range(3):
        pts.append(REF_VERTS[v]); kind.append("vertex"); ent.append(v)
    for e, (a, b) in enumerate(LOCAL_EDGES):
        for k in range(1, p):
            pts.append(REF_VERTS[a] + (k / p) * (REF_VERTS[b] - REF_VERTS[a])); kind.append("edge"); ent.append(e)
    for j in range(1, p):
        for i in range(1, p - j):
            pts.append(np.array([i / p, j / p])); kind.append("interior"); ent.append(0)
    return np.asarray(pts, dtype=float), kind, ent


class PkBasis:
    """Degree-p polynomial basis on the reference triangle stored as monomial coefficients.

    lagrange: nodal basis on the equispaced lattice; modal: L2-orthonormal, degree-hierarchical
    (Gram-Schmidt of graded monomials, same span as Dubiner).
    """

    def __init__(self, degree: int, representation: str = "lagrange"):
        self.degree = p = int(degree)
        if p < 0 or p > 8:
            raise ValueError("degree must be in 0..8")
        self.representation = canonical_representation(representation)
        self.exps = _exponents(p)
        self.ndofs = len(self.exps)
        self.mode_degree = self.exps.sum(axis=1)
        if self.representation == "lagrange":
            nodes, kind, ent = lattice_nodes(p)
            V = self._mono(nodes)
            self.C = np.linalg.inv(V).T
            self.nodes = nodes
            self.node_kind = kind
            self.node_entity = ent
        else:
            q, w = triangle_rule(2 * p)
            M = self._mono(q)
            G = (M * w[:, None]).T @ M
            C = np.linalg.inv(np.linalg.cholesky(G))
            for _ in range(2):
                P = M @ C.T
                C = np.linalg.inv(np.linalg.cholesky((P * w[:, None]).T @ P)) @ C
            self.C = C
            self.nodes = None
            self.node_kind = None
            self.node_entity = None

    def _mono(self, pts):
        pts = np.atleast_2d(pts)
        r = pts[:, 0:1] - 1.0 / 3.0
        s = pts[:, 1:2] - 1.0 / 3.0
        return r ** self.exps[:, 0] * s ** self.exps[:, 1]

    def _mono_grad(self, pts):
        pts = np.atleast_2d(pts)
        r = pts[:, 0:1] - 1.0 / 3.0
        s = pts[:, 1:2] - 1.0 / 3.0
        a, b = self.exps[:, 0], self.exps[:, 1]
        dr = a * r ** np.maximum(a - 1, 0) * s ** b
        ds = b * r ** a * s ** np.maximum(b - 1, 0)
        return np.stack((dr, ds), axis=-1)

    def _mono_hess(self, pts):
        pts = np.atleast_2d(pts)
        r = pts[:, 0:1] - 1.0 / 3.0
        s = pts[:, 1:2] - 1.0 / 3.0
        a, b = self.exps[:, 0], self.exps[:, 1]
        rr = a * (a - 1) * r ** np.maximum(a - 2, 0) * s ** b
        ss = b * (b - 1) * r ** a * s ** np.maximum(b - 2, 0)
        rs = a * b * r ** np.maximum(a - 1, 0) * s ** np.maximum(b - 1, 0)
        H = np.empty(rr.shape + (2, 2))
        H[..., 0, 0] = rr; H[..., 1, 1] = ss; H[..., 0, 1] = H[..., 1, 0] = rs
        return H

    def eval(self, pts) -> np.ndarray:
        return self._mono(pts) @ self.C.T

    def grad(self, pts) -> np.ndarray:
        return np.einsum("qmd,im->qid", self._mono_grad(pts), self.C)

    def hess(self, pts) -> np.ndarray:
        return np.einsum("qmab,im->qiab", self._mono_hess(pts), self.C)

    def edge_dofs(self, e: int) -> list[int]:
        """Local nodal DOFs on edge e ordered from LOCAL_EDGES[e][0] to [1]."""
        a, b = LOCAL_EDGES[e]
        inner = [i for i, (k, n) in enumerate(zip(self.node_kind, self.node_entity)) if k == "edge" and n == e]
        return [a] + inner + [b]


@lru_cache(maxsize=None)
def get_basis(degree: int, representation: str = "lagrange") -> PkBasis:
    return PkBasis(int(degree), canonical_representation(representation))


def ndofs_for(p) -> np.ndarray:
    p = np.asarray(p)
    return (p + 1) * (p + 2) // 2
