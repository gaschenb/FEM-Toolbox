from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femtoolbox.core.basis import get_basis
from femtoolbox.core.expr import Expr
from femtoolbox.core.pde import PDE, BoundaryCondition, assign_boundary_conditions
from femtoolbox.core.quadrature import line_rule, triangle_rule
from femtoolbox.core.solution import FEMSolution
from femtoolbox.core.space import edge_reference_points


@dataclass
class ErrorNorms:
    l2: float
    h1: float
    h1_semi: float
    l2_cells: np.ndarray | None = None
    h1_cells: np.ndarray | None = None


def _exact_funcs(u_exact, ux=None, uy=None):
    u = Expr(u_exact)
    if ux not in (None, "") and uy not in (None, ""):
        return u, Expr(ux), Expr(uy)

    def dx(x, y, t=0.0):
        h = 1e-6 * np.maximum(1.0, np.abs(x))
        return (u(x + h, y) - u(x - h, y)) / (2 * h)

    def dy(x, y, t=0.0):
        h = 1e-6 * np.maximum(1.0, np.abs(y))
        return (u(x, y + h) - u(x, y - h)) / (2 * h)
    return u, dx, dy


def _cell_fields(sol: FEMSolution, cells, p, q, hess=False):
    b = get_basis(p, sol.space.basis_name)
    loc = sol._local(p, cells)
    geo = sol.space.geo.at(q, cells)
    invJ = geo.invJ
    uh = loc @ b.eval(q).T
    G = np.einsum("qib,eqba->eqia", b.grad(q), invJ, optimize=True)
    gh = np.einsum("eqia,ei->eqa", G, loc, optimize=True)
    if not hess:
        return uh, gh, geo
    H = np.einsum("qiab,eqac,eqbd->eqicd", b.hess(q), invJ, invJ, optimize=True)
    return uh, gh, geo, np.einsum("eqicd,ei->eqcd", H, loc, optimize=True)


def error_norms(sol: FEMSolution, u_exact, ux=None, uy=None, extra_degree: int = 4) -> ErrorNorms:
    u, dx, dy = _exact_funcs(u_exact, ux, uy)
    mesh = sol.mesh
    l2c = np.zeros(mesh.nelements)
    h1c = np.zeros(mesh.nelements)
    for p, cells, _ in sol.space.groups():
        q, w = triangle_rule(2 * p + extra_degree)
        uh, gh, geo = _cell_fields(sol, cells, p, q)
        x, y = geo.X[..., 0], geo.X[..., 1]
        wd = w[None] * np.abs(geo.det)
        l2c[cells] = (wd * (uh - u(x, y)) ** 2).sum(1)
        h1c[cells] = (wd * ((gh[..., 0] - dx(x, y)) ** 2 + (gh[..., 1] - dy(x, y)) ** 2)).sum(1)
    l2, h1s = float(np.sqrt(l2c.sum())), float(np.sqrt(h1c.sum()))
    return ErrorNorms(l2, float(np.hypot(l2, h1s)), h1s, np.sqrt(l2c), np.sqrt(h1c))


solution_error_norms = error_norms


def residual_estimator(sol: FEMSolution, pde: PDE, bcs: list[BoundaryCondition] | None = None, time: float = 0.0,
                       dg_jump_weight: float = 1.0) -> np.ndarray:
    """Cellwise residual a-posteriori estimator eta_T (hp-scaled).

    eta_T^2 = (h/p)^2 ||f + div(A grad u_h) - b.grad u_h - c u_h||_T^2
              + 1/2 sum_{interior e} (h_e/p) ||[A grad u_h . n]||_e^2 + sum_{Neumann/Robin e} (h_e/p) ||g - alpha u_h - A grad u_h . n||_e^2
              + DG only: sum_e (p^2/h_e) ||[u_h]||_e^2 (+ Dirichlet mismatch).
    """
    mesh, space = sol.mesh, sol.space
    bcs = list(bcs) if bcs else [BoundaryCondition("dirichlet", 0.0, "all")]
    eta2 = np.zeros(mesh.nelements)
    pT = np.maximum(space.degrees, 1).astype(float)
    hT = mesh.diameters
    for p, cells, _ in space.groups():
        q, w = triangle_rule(2 * p + 2)
        uh, gh, geo, Hh = _cell_fields(sol, cells, p, q, hess=True)
        x, y = geo.X[..., 0], geo.X[..., 1]
        A = pde.diffusion(x, y, time)
        divAg = np.einsum("eqab,eqab->eq", A, Hh)
        if not pde.diffusion.is_constant:
            h = 1e-6
            dAx = (pde.diffusion(x + h, y, time) - pde.diffusion(x - h, y, time)) / (2 * h)
            dAy = (pde.diffusion(x, y + h, time) - pde.diffusion(x, y - h, time)) / (2 * h)
            divAg += np.einsum("eqb,eqb->eq", dAx[..., 0, :], gh) + np.einsum("eqb,eqb->eq", dAy[..., 1, :], gh)
        R = pde.source(x, y, time) + divAg - np.einsum("eqa,eqa->eq", pde.advection(x, y, time), gh) - pde.reaction(x, y, time) * uh
        wd = w[None] * np.abs(geo.det)
        eta2[cells] += (hT[cells] / pT[cells]) ** 2 * (wd * R ** 2).sum(1)
    pmax = space.degree
    s, w = line_rule(2 * pmax + 2)

    def trace(cells, le, reverse, invJ):
        vals = np.zeros((len(cells), len(s)))
        grads = np.zeros((len(cells), len(s), 2))
        for p in np.unique(space.degrees[cells]):
            m = np.flatnonzero(space.degrees[cells] == p)
            b = get_basis(int(p), space.basis_name)
            loc = sol._local(int(p), cells[m])
            for e in range(3):
                mm = m[le[m] == e]
                if not len(mm):
                    continue
                ref = edge_reference_points(e, s, reverse)
                lm = loc[le[m] == e]
                vals[mm] = lm @ b.eval(ref).T
                G = np.einsum("qib,eqba->eqia", b.grad(ref), invJ[mm])
                grads[mm] = np.einsum("eqia,ei->eqa", G, lm)
        return vals, grads

    ie = mesh.interior_edge_ids
    if len(ie):
        cL, cR = mesh.edge_cells[ie, 0], mesh.edge_cells[ie, 1]
        X, nrm, ds, invL = space.geo.edge(ie, 0, s)
        _, _, _, invR = space.geo.edge(ie, 1, s, reverse=True)
        L = mesh.edge_lengths[ie]
        A = pde.diffusion(X[..., 0], X[..., 1], time)
        uL, gL = trace(cL, mesh.edge_local[ie, 0], False, invL)
        uR, gR = trace(cR, mesh.edge_local[ie, 1], True, invR)
        jf = np.einsum("eqa,eqab,eqb->eq", gL - gR, A, nrm)
        pe = np.maximum(np.maximum(pT[cL], pT[cR]), 1)
        flux = (L / pe) * (w[None] * ds * jf ** 2).sum(1)
        np.add.at(eta2, cL, 0.5 * flux)
        np.add.at(eta2, cR, 0.5 * flux)
        if space.method == "DG":
            jump = dg_jump_weight * (pe ** 2 / L) * (w[None] * ds * (uL - uR) ** 2).sum(1)
            np.add.at(eta2, cL, 0.5 * jump)
            np.add.at(eta2, cR, 0.5 * jump)
    be = mesh.boundary_edge_ids
    if len(be):
        idx = assign_boundary_conditions(mesh, bcs)
        c = mesh.edge_cells[be, 0]
        X, nrm, ds, invB = space.geo.edge(be, 0, s)
        L = mesh.edge_lengths[be]
        uB, gB = trace(c, mesh.edge_local[be, 0], False, invB)
        A = pde.diffusion(X[..., 0], X[..., 1], time)
        fl = np.einsum("eqa,eqab,eqb->eq", gB, A, nrm)
        for i, bc in enumerate(bcs):
            m = np.flatnonzero(idx == i)
            if not len(m):
                continue
            g = bc.funcs[0](X[m, :, 0], X[m, :, 1], time)
            if bc.kind in ("neumann", "robin"):
                r = g - fl[m] - (bc.alpha * uB[m] if bc.kind == "robin" else 0.0)
                eta2[c[m]] += (L[m] / pT[c[m]]) * (w[None] * ds[m] * r ** 2).sum(1)
            elif bc.kind == "dirichlet" and space.method == "DG":
                eta2[c[m]] += dg_jump_weight * (pT[c[m]] ** 2 / L[m]) * (w[None] * ds[m] * (uB[m] - g) ** 2).sum(1)
    return np.sqrt(eta2)


def zz_estimator(sol: FEMSolution) -> np.ndarray:
    """Zienkiewicz-Zhu: ||G(u_h) - grad u_h||_T with G the nodally averaged gradient (CG only)."""
    space, mesh = sol.space, sol.mesh
    if space.method != "CG":
        raise ValueError("ZZ estimator is implemented for CG spaces")
    p = space.degree
    b = get_basis(p, "lagrange")
    loc = sol.values[space.cell_dofs]
    Gn = np.einsum("qib,eqba->eqia", b.grad(b.nodes), space.geo.at(b.nodes).invJ)
    gn = np.einsum("eqia,ei->eqa", Gn, loc)
    acc = np.zeros((space.ndofs, 2))
    cnt = np.zeros(space.ndofs)
    wts = np.repeat(mesh.areas[:, None], b.ndofs, 1)
    np.add.at(acc, space.cell_dofs.ravel(), (gn * wts[..., None]).reshape(-1, 2))
    np.add.at(cnt, space.cell_dofs.ravel(), wts.ravel())
    rec = acc / cnt[:, None]
    q, w = triangle_rule(2 * p + 2)
    phi = b.eval(q)
    gq_geo = space.geo.at(q)
    Gq = np.einsum("qib,eqba->eqia", b.grad(q), gq_geo.invJ)
    gq = np.einsum("eqia,ei->eqa", Gq, loc)
    rq = np.einsum("qi,eia->eqa", phi, rec[space.cell_dofs])
    wd = w[None] * np.abs(gq_geo.det)
    return np.sqrt((wd * ((rq - gq) ** 2).sum(-1)).sum(1))


def smoothness_indicator(sol: FEMSolution) -> np.ndarray:
    """Decay rate sigma of orthonormal modal coefficients |a_k| ~ C exp(-sigma k) per cell (large = smooth).

    For p < 2 there is not enough spectrum; returns +inf for p >= 2 cells only.
    """
    mesh, space = sol.mesh, sol.space
    out = np.full(mesh.nelements, np.nan)
    for p, cells, _ in space.groups():
        if p < 2:
            continue
        modal = get_basis(p, "modal")
        q, w = triangle_rule(2 * p + 2)
        uh = sol._local(p, cells) @ get_basis(p, space.basis_name).eval(q).T
        coef = (uh * w[None]) @ modal.eval(q)
        k = modal.mode_degree
        Ek = np.stack([np.sqrt((coef[:, k == d] ** 2).sum(1)) for d in range(1, p + 1)], 1)
        logE = np.log(np.maximum(Ek, 1e-300))
        deg = np.arange(1, p + 1, dtype=float)
        dm = deg - deg.mean()
        slope = (logE - logE.mean(1, keepdims=True)) @ dm / (dm @ dm)
        scale = np.sqrt((coef ** 2).sum(1))
        sigma = -slope
        sigma[scale < 1e-14] = np.inf
        out[cells] = sigma
    return out


def estimate_error(solution: FEMSolution, pde: PDE | None = None, bcs=None) -> np.ndarray:
    if pde is not None and pde.is_nonlinear:
        pde = None
    if pde is None:
        return zz_estimator(solution) if solution.space.method == "CG" else np.zeros(solution.mesh.nelements)
    return residual_estimator(solution, pde, bcs)
