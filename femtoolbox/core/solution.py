from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from femtoolbox.core.basis import get_basis
from femtoolbox.core.space import FunctionSpace


@dataclass
class SolveInfo:
    method: str
    degree: int
    basis: str
    ndofs: int
    nelements: int
    nnz: int
    residual_norm: float
    warnings: list = field(default_factory=list)
    solver: str = ""
    iterations: int = 0
    t_assembly: float = 0.0
    t_setup: float = 0.0
    t_solve: float = 0.0
    extra: dict = field(default_factory=dict)

    def as_text(self) -> str:
        s = (f"{self.method}-P{self.degree} [{self.basis}]  DOFs={self.ndofs}  cells={self.nelements}  nnz={self.nnz}\n"
             f"solver={self.solver}  iterations={self.iterations}  rel. residual={self.residual_norm:.2e}\n"
             f"assembly {self.t_assembly * 1e3:.0f} ms, setup {self.t_setup * 1e3:.0f} ms, solve {self.t_solve * 1e3:.0f} ms")
        for k, v in self.extra.items():
            if not isinstance(v, (list, tuple, dict)):
                s += f"\n{k}: {v}"
        for w in self.warnings:
            s += f"\nwarning: {w}"
        return s


@lru_cache(maxsize=None)
def sub_lattice(k: int):
    pts = np.array([(i / k, j / k) for j in range(k + 1) for i in range(k + 1 - j)])
    idx = {}
    c = 0
    for j in range(k + 1):
        for i in range(k + 1 - j):
            idx[(i, j)] = c
            c += 1
    tris = []
    for j in range(k):
        for i in range(k - j):
            tris.append((idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]))
            if i + j < k - 1:
                tris.append((idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]))
    return pts, np.asarray(tris)


@dataclass
class FEMSolution:
    space: FunctionSpace
    values: np.ndarray
    info: SolveInfo

    @property
    def mesh(self):
        return self.space.mesh

    def value_at_dofs(self) -> np.ndarray:
        return self.values.copy()

    def _local(self, p, cells):
        if self.space.method == "CG" or self.space.uniform:
            return self.values[self.space.cell_dofs[cells]]
        nb = (p + 1) * (p + 2) // 2
        return self.values[self.space.offsets[cells][:, None] + np.arange(nb)[None, :]]

    def evaluate_ref(self, cells: np.ndarray, ref: np.ndarray, gradient: bool = False):
        """Values (and gradients) at per-point (cell, reference coordinate)."""
        out = np.full(len(cells), np.nan)
        grad = np.full((len(cells), 2), np.nan) if gradient else None
        for p, gcells, _ in self.space.groups():
            mask = np.isin(cells, gcells) & (cells >= 0)
            if not np.any(mask):
                continue
            b = get_basis(p, self.space.basis_name)
            c = cells[mask]
            loc = self._local(p, c)
            phi = b.eval(ref[mask])
            out[mask] = np.einsum("ni,ni->n", phi, loc)
            if gradient:
                g = b.grad(ref[mask])
                gx = np.einsum("nib,nba->nia", g, self.space.geo.at_points(c, ref[mask]).invJ)
                grad[mask] = np.einsum("nia,ni->na", gx, loc)
        return (out, grad) if gradient else out

    def evaluate(self, x, y, gradient: bool = False):
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        cells, ref = self.mesh.locate(x, y)
        ok = cells >= 0
        if self.space.geo.any_curved and ok.any():
            ref[ok] = self.space.geo.inverse_map(cells[ok], np.column_stack((x[ok], y[ok])), ref[ok])
        return self.evaluate_ref(cells, ref, gradient)

    def __call__(self, x, y):
        return self.evaluate(x, y)

    def plot_data(self, subdiv: int | None = None, field: str = "u"):
        """Exact high-order sampling on a sub-triangulation: (points, triangles, values)."""
        mesh = self.mesh
        p = self.space.degree
        k = max(1, int(subdiv) if subdiv else p)
        if field == "u" and self.space.method == "CG" and p == 1 and k == 1:
            return mesh.nodes, mesh.triangles, self.values[: mesh.nnodes]
        ref, sub = sub_lattice(k)
        nloc = len(ref)
        M = mesh.nelements
        geo = self.space.geo.at(ref)
        pts = geo.X.reshape(-1, 2)
        vals = np.empty((M, nloc))
        for pg, cells, _ in self.space.groups():
            b = get_basis(pg, self.space.basis_name)
            loc = self._local(pg, cells)
            if field == "u":
                vals[cells] = loc @ b.eval(ref).T
            else:
                G = np.einsum("qib,eqba->eqia", b.grad(ref), geo.invJ[cells])
                g = np.einsum("eqia,ei->eqa", G, loc)
                vals[cells] = {"ux": g[..., 0], "uy": g[..., 1]}.get(field, np.hypot(g[..., 0], g[..., 1]))
        tris = (np.arange(M)[:, None, None] * nloc + sub[None]).reshape(-1, 3)
        return pts, tris, vals.ravel()

    def nodal_values_for_plot(self):
        return self.plot_data()

    def cell_average_values(self) -> np.ndarray:
        from femtoolbox.core.quadrature import triangle_rule
        q, w = triangle_rule(2 * self.space.degree)
        out = np.zeros(self.mesh.nelements)
        for p, cells, _ in self.space.groups():
            b = get_basis(p, self.space.basis_name)
            wd = w[None] * np.abs(self.space.geo.at(q, cells).det)
            out[cells] = ((self._local(p, cells) @ b.eval(q).T) * wd).sum(1) / wd.sum(1)
        return out

    def gradient_at_centroids(self) -> np.ndarray:
        ref = np.tile([[1 / 3, 1 / 3]], (self.mesh.nelements, 1))
        return self.evaluate_ref(np.arange(self.mesh.nelements), ref, gradient=True)[1]

    def line_profile(self, a, b, n: int = 400):
        a, b = np.asarray(a, float), np.asarray(b, float)
        s = np.linspace(0, 1, n)
        P = a[None] + s[:, None] * (b - a)[None]
        return s * np.linalg.norm(b - a), self.evaluate(P[:, 0], P[:, 1])


@dataclass
class TransientFEMSolution:
    space: FunctionSpace
    times: np.ndarray
    values_by_step: np.ndarray
    info: SolveInfo

    @property
    def mesh(self):
        return self.space.mesh

    @property
    def nsteps(self) -> int:
        return max(0, len(self.times) - 1)

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self.values_by_step[-1], float)

    def step_solution(self, step: int) -> FEMSolution:
        step = int(np.clip(step, 0, len(self.times) - 1))
        return FEMSolution(self.space, np.asarray(self.values_by_step[step], float), self.info)

    def final_solution(self) -> FEMSolution:
        return self.step_solution(len(self.times) - 1)

    def nodal_values_for_plot_at(self, step: int):
        return self.step_solution(step).plot_data()


@dataclass
class EigenSolution:
    space: FunctionSpace
    eigenvalues: np.ndarray
    modes: np.ndarray  # (k, ndofs)
    info: SolveInfo

    @property
    def mesh(self):
        return self.space.mesh

    def mode(self, i: int) -> FEMSolution:
        return FEMSolution(self.space, self.modes[int(i)], self.info)

    @property
    def values(self):
        return self.modes[0]
