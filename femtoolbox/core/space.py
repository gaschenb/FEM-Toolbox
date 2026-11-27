from __future__ import annotations

from functools import cached_property

import numpy as np

from femtoolbox.core.basis import LOCAL_EDGES, REF_VERTS, get_basis, ndofs_for
from femtoolbox.core.mesh import Mesh2D


class FunctionSpace:
    """CG: continuous Lagrange P_p (uniform p). DG: discontinuous P_{p_T}, p_T may vary per cell."""

    def __init__(self, mesh: Mesh2D, method: str = "CG", degree=1, basis_name: str = "lagrange"):
        self.mesh = mesh
        self.method = str(method).upper()
        if self.method not in ("CG", "DG"):
            raise ValueError("method must be CG or DG")
        deg = np.asarray(degree, dtype=int)
        if deg.ndim == 0:
            deg = np.full(mesh.nelements, int(deg))
        if len(deg) != mesh.nelements:
            raise ValueError("degree array must have one entry per cell")
        self.degrees = deg
        self.basis_name = basis_name
        self.geo = mesh.geometry(int(deg.max()))
        if self.method == "CG":
            if np.any(deg != deg[0]):
                raise ValueError("CG requires a uniform degree")
            if deg[0] < 1:
                raise ValueError("CG requires degree >= 1")
            self.basis_name = "lagrange"
            self._build_cg(int(deg[0]))
        else:
            self._build_dg()

    @property
    def degree(self) -> int:
        return int(self.degrees.max())

    @property
    def uniform(self) -> bool:
        return bool(np.all(self.degrees == self.degrees[0]))

    @property
    def basis(self):
        return get_basis(int(self.degrees[0]), self.basis_name)

    def groups(self):
        """[(p, cells, dofs (nc,nb))] for each distinct degree."""
        return self._groups

    # ------------------------------------------------------------------ CG
    def _build_cg(self, p: int):
        mesh = self.mesh
        N, E, M = mesh.nnodes, mesh.nedges, mesh.nelements
        b = get_basis(p, "lagrange")
        nint = (p - 1) * (p - 2) // 2
        cell_dofs = np.empty((M, b.ndofs), dtype=np.int64)
        cell_dofs[:, :3] = mesh.triangles
        tri = mesh.triangles
        for e, (la, lb) in enumerate(LOCAL_EDGES):
            if p < 2:
                break
            eid = mesh.tri_edges[:, e]
            same = tri[:, la] < tri[:, lb]
            k = np.arange(p - 1)
            idx = np.where(same[:, None], k[None, :], (p - 2 - k)[None, :])
            local = [i for i, (kind, ent) in enumerate(zip(b.node_kind, b.node_entity)) if kind == "edge" and ent == e]
            cell_dofs[:, local] = N + eid[:, None] * (p - 1) + idx
        if nint:
            local = [i for i, kind in enumerate(b.node_kind) if kind == "interior"]
            cell_dofs[:, local] = N + E * (p - 1) + np.arange(M)[:, None] * nint + np.arange(nint)[None, :]
        self.ndofs = int(N + E * (p - 1) + M * nint)
        self.cell_dofs = cell_dofs
        coords = np.empty((self.ndofs, 2))
        X = self.geo.at(b.nodes).X
        coords[cell_dofs.ravel()] = X.reshape(-1, 2)
        self.dof_coords = coords
        self._groups = [(p, np.arange(M), cell_dofs)]
        edge_local = np.zeros((E, p + 1), dtype=np.int64)
        c0, l0 = mesh.edge_cells[:, 0], mesh.edge_local[:, 0]
        for e in range(3):
            sel = l0 == e
            edge_local[sel] = cell_dofs[c0[sel]][:, b.edge_dofs(e)]
        self.edge_dofs = edge_local

    # ------------------------------------------------------------------ DG
    def _build_dg(self):
        nb = ndofs_for(self.degrees)
        self.offsets = np.r_[0, np.cumsum(nb)].astype(np.int64)
        self.ndofs = int(self.offsets[-1])
        self._groups = []
        for p in np.unique(self.degrees):
            cells = np.flatnonzero(self.degrees == p)
            nbp = int(ndofs_for(p))
            dofs = self.offsets[cells][:, None] + np.arange(nbp)[None, :]
            self._groups.append((int(p), cells, dofs))
        if self.uniform:
            self.cell_dofs = self._groups[0][2]
        else:
            self.cell_dofs = None
        self.dof_coords = self._dg_coords()

    def _dg_coords(self):
        coords = np.zeros((self.ndofs, 2))
        for p, cells, dofs in self._groups:
            b = get_basis(p, "lagrange")
            nodes = b.nodes if p > 0 else np.array([[1 / 3, 1 / 3]])
            coords[dofs.ravel()] = self.geo.at(nodes, cells).X.reshape(-1, 2)
        return coords

    def dofs_of(self, cell: int) -> np.ndarray:
        if self.method == "CG" or self.cell_dofs is not None:
            return self.cell_dofs[cell]
        return np.arange(self.offsets[cell], self.offsets[cell + 1])

    def basis_of(self, cell: int):
        return get_basis(int(self.degrees[cell]), self.basis_name)

    def boundary_dofs(self, marker: str | None = None) -> np.ndarray:
        mesh = self.mesh
        eids = mesh.boundary_edge_ids
        if marker not in (None, "all", "outer"):
            eids = eids[mesh.edge_markers == marker]
        if self.method == "CG":
            return np.unique(self.edge_dofs[eids])
        out = []
        for e in eids:
            c, le = mesh.edge_cells[e, 0], mesh.edge_local[e, 0]
            b = get_basis(int(self.degrees[c]), "lagrange")
            out.extend(self.dofs_of(c)[b.edge_dofs(le)] if b.degree > 0 else self.dofs_of(c))
        return np.unique(np.asarray(out, dtype=np.int64))

    def summary(self) -> str:
        if self.method == "CG":
            return f"CG-P{self.degree}: {self.ndofs} DOFs"
        if self.uniform:
            return f"DG-P{self.degree} ({self.basis_name}): {self.ndofs} DOFs"
        return f"hp-DG p in [{self.degrees.min()}, {self.degrees.max()}] ({self.basis_name}): {self.ndofs} DOFs"


def edge_reference_points(local_edge: int, s: np.ndarray, reverse: bool = False) -> np.ndarray:
    a, b = LOCAL_EDGES[local_edge]
    s = 1.0 - s if reverse else s
    return REF_VERTS[a][None, :] + s[:, None] * (REF_VERTS[b] - REF_VERTS[a])[None, :]
