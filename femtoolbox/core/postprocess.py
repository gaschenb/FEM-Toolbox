"""Derived quantities: domain integrals, boundary fluxes, energy, conservation balance."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from femtoolbox.core.assembly import phys_grad
from femtoolbox.core.basis import get_basis
from femtoolbox.core.expr import Expr, clear_field, set_field
from femtoolbox.core.pde import PDE
from femtoolbox.core.quadrature import line_rule, triangle_rule
from femtoolbox.core.space import edge_reference_points


def _volume_points(sol, extra_deg: int = 2):
    """Yield (X (e,q,2), wdet (e,q), u (e,q), grad u (e,q,2)) per degree group."""
    space = sol.space
    for p, cells, dofs in space.groups():
        b = get_basis(p, space.basis_name)
        q, w = triangle_rule(2 * p + extra_deg)
        geo = space.geo.at(q, cells)
        ue = sol.values[dofs]
        uq = np.einsum("qi,ei->eq", b.eval(q), ue, optimize=True)
        gq = np.einsum("eqia,ei->eqa", phys_grad(b.grad(q), geo.invJ), ue, optimize=True)
        yield geo.X, w[None] * np.abs(geo.det), uq, gq


def _boundary_points(sol, eids):
    """Yield (sel, X, normal, w*ds, u, grad u) on boundary edges eids, grouped by cell degree."""
    space, mesh = sol.space, sol.space.mesh
    s, w = line_rule(2 * space.degree + 2)
    cells = mesh.edge_cells[eids, 0]
    le = mesh.edge_local[eids, 0]
    X, nrm, ds, inv = space.geo.edge(eids, 0, s)
    U = np.zeros(ds.shape)
    G = np.zeros(ds.shape + (2,))
    for p, gcells, gdofs in space.groups():
        row = np.full(mesh.nelements, -1)
        row[gcells] = np.arange(len(gcells))
        b = get_basis(p, space.basis_name)
        for e in range(3):
            sel = np.flatnonzero((le == e) & (row[cells] >= 0))
            if not len(sel):
                continue
            ref = edge_reference_points(e, s)
            ue = sol.values[gdofs[row[cells[sel]]]]
            U[sel] = np.einsum("qi,ei->eq", b.eval(ref), ue, optimize=True)
            G[sel] = np.einsum("eqia,ei->eqa", phys_grad(b.grad(ref), inv[sel]), ue, optimize=True)
    return X, nrm, w[None] * ds, U, G


def integrate(sol, expression="u", t: float = 0.0) -> float:
    """Integral over the domain of an expression in x, y, u, ux, uy."""
    f = Expr(expression)
    total = 0.0
    try:
        for X, wd, uq, gq in _volume_points(sol):
            set_field(u=uq, ux=gq[..., 0], uy=gq[..., 1])
            total += float((wd * f(X[..., 0], X[..., 1], t)).sum())
    finally:
        clear_field()
    return total


def boundary_integral(sol, expression="u", marker: str | None = None, t: float = 0.0) -> float:
    """Integral over boundary edges (one marker or all) of an expression in x, y, u, ux, uy, nx, ny."""
    mesh = sol.space.mesh
    eids = mesh.boundary_edge_ids
    if marker not in (None, "all"):
        eids = eids[mesh.edge_markers == marker]
    if not len(eids):
        return 0.0
    X, nrm, wds, U, G = _boundary_points(sol, eids)
    f = Expr(expression) if not isinstance(expression, Expr) else expression
    try:
        set_field(u=U, ux=G[..., 0], uy=G[..., 1], nx=nrm[..., 0], ny=nrm[..., 1])
        val = f(X[..., 0], X[..., 1], t)
    finally:
        clear_field()
    return float((wds * val).sum())


def _residual_vector(sol, pde: PDE, t: float = 0.0) -> np.ndarray:
    from femtoolbox.core.assembly import assemble_volume
    cache = sol.__dict__.setdefault("_flux_cache", {})
    key = (id(pde), t)
    if key not in cache:
        K, _, F = assemble_volume(sol.space, pde, t, want_M=False, u_state=sol.values if pde.is_nonlinear else None)
        cache[key] = K @ sol.values - F
    return cache[key]


def boundary_flux(sol, pde: PDE, marker: str | None = None, t: float = 0.0, consistent: bool = True) -> float:
    """Outward flux  int_G (-A grad u) . n ds  (plus the advective part b.n u for CG-consistent fluxes).

    CG: variational (residual) flux  -(K u - F) . chi_G, which balances the source exactly and converges at the
    rate of the energy functional; dofs shared by two markers count half. Otherwise the gradient trace is used."""
    space = sol.space
    if consistent and space.method == "CG" and pde.advection.is_zero:
        R = _residual_vector(sol, pde, t)
        mesh = space.mesh
        eids = mesh.boundary_edge_ids
        markers = mesh.edge_markers
        own = {}
        for m in mesh.boundary_marker_names():
            d = np.unique(space.edge_dofs[eids[markers == m]].ravel()) if hasattr(space, "edge_dofs") else None
            if d is None:
                break
            own[m] = d
        if own:
            count = np.zeros(space.ndofs)
            for d in own.values():
                count[d] += 1
            if marker in (None, "all"):
                d = np.unique(np.concatenate(list(own.values())))
                return float(-R[d].sum())
            d = own.get(marker)
            if d is None:
                return 0.0
            return float(-(R[d] / count[d]).sum())
    mesh = space.mesh
    eids = mesh.boundary_edge_ids
    if marker not in (None, "all"):
        eids = eids[mesh.edge_markers == marker]
    if not len(eids):
        return 0.0
    X, nrm, wds, U, G = _boundary_points(sol, eids)
    try:
        set_field(u=U, ux=G[..., 0], uy=G[..., 1])
        A = pde.diffusion(X[..., 0], X[..., 1], t)
    finally:
        clear_field()
    q = -np.einsum("eqab,eqb->eqa", A, G)
    return float((wds * np.einsum("eqa,eqa->eq", q, nrm)).sum())


@dataclass
class Quantities:
    rows: list = field(default_factory=list)
    flux_rows: list = field(default_factory=list)

    def as_text(self) -> str:
        w = max(len(r[0]) for r in self.rows + self.flux_rows) if self.rows else 10
        out = [f"{n:<{w}}  {v: .8g}" for n, v in self.rows]
        if self.flux_rows:
            out += ["", "boundary fluxes  (-A grad u . n, outward)"] + [f"{n:<{w}}  {v: .8g}" for n, v in self.flux_rows]
        return "\n".join(out)


def quantities(sol, pde: PDE | None = None, t: float = 0.0) -> Quantities:
    area = l2 = h1 = iu = 0.0
    energy = src = rea = 0.0
    umin, umax = np.inf, -np.inf
    xmin = xmax = (np.nan, np.nan)
    try:
        for X, wd, uq, gq in _volume_points(sol):
            x, y = X[..., 0], X[..., 1]
            area += wd.sum()
            iu += (wd * uq).sum()
            l2 += (wd * uq * uq).sum()
            h1 += (wd * (gq * gq).sum(-1)).sum()
            k, j = np.unravel_index(np.argmin(uq), uq.shape)
            if uq[k, j] < umin:
                umin, xmin = float(uq[k, j]), (float(x[k, j]), float(y[k, j]))
            k, j = np.unravel_index(np.argmax(uq), uq.shape)
            if uq[k, j] > umax:
                umax, xmax = float(uq[k, j]), (float(x[k, j]), float(y[k, j]))
            if pde is not None:
                set_field(u=uq, ux=gq[..., 0], uy=gq[..., 1])
                A = pde.diffusion(x, y, t)
                energy += (wd * np.einsum("eqa,eqab,eqb->eq", gq, A, gq)).sum()
                src += (wd * pde.source(x, y, t)).sum()
                rea += (wd * pde.reaction(x, y, t) * uq).sum()
                if not pde.advection.is_zero:
                    rea += (wd * np.einsum("eqa,eqa->eq", pde.advection(x, y, t), gq)).sum()
                clear_field()
    finally:
        clear_field()
    q = Quantities()
    q.rows = [("area", area), ("integral u", iu), ("mean u", iu / area), ("||u||_L2", np.sqrt(l2)),
              ("|u|_H1", np.sqrt(h1)), (f"min u  at ({xmin[0]:.4g}, {xmin[1]:.4g})", umin),
              (f"max u  at ({xmax[0]:.4g}, {xmax[1]:.4g})", umax)]
    if pde is not None:
        q.rows += [("energy  int grad u . A grad u", energy), ("integral f", src)]
        total = 0.0
        for m in sol.space.mesh.boundary_marker_names():
            v = boundary_flux(sol, pde, m, t)
            total += v
            q.flux_rows.append((m, v))
        q.flux_rows.append(("total", total))
        if pde.advection.is_zero:
            q.rows.append(("balance  int f - int c u - flux", src - rea - total))
    return q


def integration_weights(space) -> np.ndarray:
    """w with  int_Omega u_h = w . values  for every u_h in the space."""
    out = np.zeros(space.ndofs)
    for p, cells, dofs in space.groups():
        b = get_basis(p, space.basis_name)
        q, w = triangle_rule(p + 1)
        wd = w[None] * np.abs(space.geo.at(q, cells).det)
        np.add.at(out, dofs.ravel(), (wd @ b.eval(q)).ravel())
    return out


def point_series(sol, x: float, y: float):
    """u(x, y, t_k) for a transient solution (nan outside the mesh)."""
    mesh = sol.space.mesh
    cells, ref = mesh.locate(np.array([x]), np.array([y]))
    if cells[0] < 0:
        return np.full(len(sol.times), np.nan)
    if sol.space.geo.any_curved:
        ref = sol.space.geo.inverse_map(cells, np.array([[x, y]]), ref)
    from femtoolbox.core.solution import FEMSolution
    return np.array([FEMSolution(sol.space, np.asarray(v, float), sol.info).evaluate_ref(cells, ref)[0]
                     for v in sol.values_by_step])
