from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from femtoolbox.core.assembly import _coo_to_csr, _edge_tab, phys_grad
from femtoolbox.core.basis import get_basis
from femtoolbox.core.expr import Expr
from femtoolbox.core.linalg import Factorization, reduce_dirichlet
from femtoolbox.core.mesh import Mesh2D
from femtoolbox.core.pde import BoundaryCondition, assign_boundary_conditions
from femtoolbox.core.quadrature import line_rule, triangle_rule
from femtoolbox.core.solution import SolveInfo, sub_lattice
from femtoolbox.core.space import FunctionSpace


@dataclass
class Elasticity:
    E: object = 1.0
    nu: object = 0.3
    fx: object = 0.0
    fy: object = 0.0
    plane: str = "stress"
    name: str = "Linear elasticity"

    def lame(self, x, y):
        E = Expr(self.E)(x, y)
        nu = Expr(self.nu)(x, y)
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        if self.plane.startswith("stress"):
            lam = 2 * lam * mu / (lam + 2 * mu)
        return lam, mu


def assemble_elasticity(space: FunctionSpace, prob: Elasticity, bcs: list[BoundaryCondition]):
    mesh = space.mesh
    p = space.degree
    n = space.ndofs
    b = get_basis(p, "lagrange")
    q, w = triangle_rule(2 * p + 1)
    phi, gref = b.eval(q), b.grad(q)
    geo = space.geo.at(q)
    X = geo.X
    x, y = X[..., 0], X[..., 1]
    lam, mu = prob.lame(x, y)
    wd = w[None] * np.abs(geo.det)
    G = phys_grad(gref, geo.invJ)
    gx, gy = G[..., 0], G[..., 1]
    e = lambda c, a, bb: np.einsum("eq,eqi,eqj->eij", wd * c, a, bb, optimize=True)
    Kxx = e(lam + 2 * mu, gx, gx) + e(mu, gy, gy)
    Kyy = e(lam + 2 * mu, gy, gy) + e(mu, gx, gx)
    Kxy = e(lam, gx, gy) + e(mu, gy, gx)
    d = space.cell_dofs
    nb = d.shape[1]
    R, C = np.repeat(d, nb, 1), np.tile(d, (1, nb))
    K = sp.bmat([[_coo_to_csr(R, C, Kxx, n), _coo_to_csr(R, C, Kxy, n)],
                 [_coo_to_csr(R, C, np.swapaxes(Kxy, 1, 2), n), _coo_to_csr(R, C, Kyy, n)]]).tocsr()
    F = np.zeros(2 * n)
    for comp, f in enumerate((prob.fx, prob.fy)):
        fe = Expr(f)
        if fe.is_constant and fe.const == 0.0:
            continue
        Fe = np.einsum("eq,qi->ei", wd * fe(x, y), phi)
        np.add.at(F, comp * n + d.ravel(), Fe.ravel())
    eids = mesh.boundary_edge_ids
    idx = assign_boundary_conditions(mesh, bcs)
    s, wl = line_rule(2 * p + 2)
    cells, le = mesh.edge_cells[eids, 0], mesh.edge_local[eids, 0]
    XE, nrmq, dsq, _ = space.geo.edge(eids, 0, s)
    phi_t, _ = _edge_tab(p, "lagrange", s, False)
    fixed = np.full(2 * n, np.nan)
    for i, bc in enumerate(bcs):
        m = np.flatnonzero(idx == i)
        if not len(m):
            continue
        if bc.kind in ("traction", "pressure", "neumann"):
            if bc.kind == "pressure":
                pr = bc.funcs[0](XE[m, :, 0], XE[m, :, 1])
                tx, ty = -pr * nrmq[m, :, 0], -pr * nrmq[m, :, 1]
            elif bc.kind == "neumann":
                tx = bc.funcs[0](XE[m, :, 0], XE[m, :, 1])
                ty = np.zeros_like(tx)
            else:
                tx = bc.funcs[0](XE[m, :, 0], XE[m, :, 1])
                ty = bc.funcs[1](XE[m, :, 0], XE[m, :, 1])
            ph = phi_t[le[m]]
            dofs = d[cells[m]]
            for comp, t in enumerate((tx, ty)):
                Fe = np.einsum("eq,eqi->ei", wl[None] * dsq[m] * t, ph)
                np.add.at(F, comp * n + dofs.ravel(), Fe.ravel())
        elif bc.kind in ("fixed", "fixed_x", "fixed_y", "dirichlet"):
            ed = np.unique(space.edge_dofs[eids[m]])
            xc = space.dof_coords[ed]
            if bc.kind in ("fixed", "fixed_x", "dirichlet"):
                fixed[ed] = bc.funcs[0](xc[:, 0], xc[:, 1])
            if bc.kind in ("fixed", "fixed_y", "dirichlet"):
                fixed[n + ed] = bc.funcs[1 if len(bc.funcs) > 1 else 0](xc[:, 0], xc[:, 1]) if bc.kind != "dirichlet" else 0.0
    ddofs = np.flatnonzero(~np.isnan(fixed))
    return K, F, ddofs, fixed[ddofs]


@dataclass
class ElasticitySolution:
    space: FunctionSpace
    values: np.ndarray
    info: SolveInfo
    problem: Elasticity

    @property
    def mesh(self):
        return self.space.mesh

    @property
    def ux(self):
        return self.values[: self.space.ndofs]

    @property
    def uy(self):
        return self.values[self.space.ndofs:]

    def _fields_at(self, ref):
        mesh, sp_ = self.mesh, self.space
        b = get_basis(sp_.degree, "lagrange")
        d = sp_.cell_dofs
        phi = b.eval(ref)
        geo = sp_.geo.at(ref)
        G = phys_grad(b.grad(ref), geo.invJ)
        lx, ly = self.ux[d], self.uy[d]
        ux, uy = lx @ phi.T, ly @ phi.T
        exx = np.einsum("eqi,ei->eq", G[..., 0], lx)
        eyy = np.einsum("eqi,ei->eq", G[..., 1], ly)
        exy = 0.5 * (np.einsum("eqi,ei->eq", G[..., 1], lx) + np.einsum("eqi,ei->eq", G[..., 0], ly))
        X = geo.X
        lam, mu = self.problem.lame(X[..., 0], X[..., 1])
        tr = exx + eyy
        sxx, syy, sxy = lam * tr + 2 * mu * exx, lam * tr + 2 * mu * eyy, 2 * mu * exy
        if self.problem.plane.startswith("strain"):
            nu = Expr(self.problem.nu)(X[..., 0], X[..., 1])
            szz = nu * (sxx + syy)
        else:
            szz = np.zeros_like(sxx)
        vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * sxy ** 2)
        return dict(ux=ux, uy=uy, umag=np.hypot(ux, uy), sxx=sxx, syy=syy, sxy=sxy, vm=vm, X=X)

    def plot_data(self, subdiv: int | None = None, field: str = "vm", scale: float = 0.0):
        k = max(1, int(subdiv) if subdiv else self.space.degree)
        ref, sub = sub_lattice(k)
        f = self._fields_at(ref)
        X = f["X"].copy()
        if scale:
            X[..., 0] += scale * f["ux"]
            X[..., 1] += scale * f["uy"]
        M = self.mesh.nelements
        tris = (np.arange(M)[:, None, None] * len(ref) + sub[None]).reshape(-1, 3)
        return X.reshape(-1, 2), tris, f[field].ravel()

    def max_field(self, field: str = "vm", subdiv: int = 6) -> float:
        return float(np.nanmax(self.plot_data(subdiv, field)[2]))

    def auto_scale(self) -> float:
        span = max(np.ptp(self.mesh.nodes[:, 0]), np.ptp(self.mesh.nodes[:, 1]))
        umax = max(np.abs(self.values).max(), 1e-300)
        return 0.1 * span / umax


def solve_elasticity(prob: Elasticity, mesh: Mesh2D, degree: int = 2, bcs=None, solver: str = "auto") -> ElasticitySolution:
    space = FunctionSpace(mesh, "CG", degree)
    t0 = time.perf_counter()
    K, F, dd, dv = assemble_elasticity(space, prob, bcs or [])
    t_asm = time.perf_counter() - t0
    Kff, rhs, fidx, u = reduce_dirichlet(K, F, dd, dv)
    if len(dd) == 0:
        raise ValueError("elasticity needs displacement constraints (fixed / fixed_x / fixed_y) to remove rigid motions")
    fac = Factorization(Kff, solver, True)
    u[fidx] = fac.solve(rhs)
    st = fac.stats
    info = SolveInfo("CG", degree, "lagrange", 2 * space.ndofs, mesh.nelements, int(K.nnz), st.residual,
                     list(st.warnings), st.solver, st.iterations, t_asm, st.setup_time, st.solve_time)
    return ElasticitySolution(space, u, info, prob)


def timoshenko_beam(L=10.0, D=2.0, P=1.0, E=1000.0, nu=0.3):
    """Exact plane-stress cantilever (clamped x=0, parabolic shear at x=L). Returns (ux, uy, sxx, sxy) strings."""
    I = D ** 3 / 12.0
    ux = f"-{P}*y/(6*{E}*{I})*((6*{L}-3*x)*x + (2+{nu})*(y^2-{D}^2/4))"
    uy = f"{P}/(6*{E}*{I})*(3*{nu}*y^2*({L}-x) + (4+5*{nu})*{D}^2*x/4 + (3*{L}-x)*x^2)"
    sxx = f"-{P}*({L}-x)*y/{I}"
    sxy = f"{P}/(2*{I})*({D}^2/4 - y^2)"
    return ux, uy, sxx, sxy
