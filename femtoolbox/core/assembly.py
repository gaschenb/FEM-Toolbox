from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from femtoolbox.core.basis import get_basis
from femtoolbox.core.expr import clear_field, set_field
from femtoolbox.core.pde import PDE, BoundaryCondition, assign_boundary_conditions
from femtoolbox.core.quadrature import line_rule, triangle_rule
from femtoolbox.core.space import FunctionSpace, edge_reference_points

CHUNK_ENTRIES = 4_000_000


@dataclass
class AssemblyResult:
    K: sp.csr_matrix
    F: np.ndarray
    M: sp.csr_matrix
    space: FunctionSpace
    dirichlet_dofs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    dirichlet_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    timings: dict = field(default_factory=dict)


def phys_grad(gref: np.ndarray, invJ: np.ndarray) -> np.ndarray:
    """gref (q,i,2), invJ (e,2,2) or (e,q,2,2) -> (e,q,i,2)."""
    if invJ.ndim == 3:
        return np.einsum("qib,eba->eqia", gref, invJ, optimize=True)
    return np.einsum("qib,eqba->eqia", gref, invJ, optimize=True)


def _qdeg(p: int, pde: PDE | None) -> int:
    extra = 2
    if pde is not None and not (pde.diffusion.is_constant and pde.advection.is_constant and pde.reaction.is_constant):
        extra = 3
    return max(2 * p + extra, 2)


def _coo_to_csr(rows, cols, vals, n):
    return sp.csr_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(n, n))


def supg_tau(mesh, cells, pde: PDE) -> np.ndarray:
    c = mesh.nodes[mesh.triangles[cells]].mean(1)
    b = pde.advection(c[:, 0], c[:, 1])
    bn = np.linalg.norm(b, axis=1)
    k = np.maximum(pde.diffusion.max_eig(c[:, 0], c[:, 1]), 1e-14)
    h = mesh.diameters[cells]
    pe = bn * h / (2 * k)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where(pe > 1e-8, 1.0 / np.tanh(pe) - 1.0 / pe, pe / 3.0)
        tau = np.where(bn > 1e-14, h / (2 * bn) * xi, 0.0)
    return tau


def assemble_volume(space: FunctionSpace, pde: PDE, time: float = 0.0, want_K=True, want_M=True, want_F=True,
                    supg: bool = False, u_state: np.ndarray | None = None, newton: bool = False):
    """u_state: current iterate for u-dependent coefficients (Picard); newton adds the exact linearization of
    u-dependent source and reaction terms."""
    if pde.is_nonlinear and u_state is None:
        u_state = np.zeros(space.ndofs)
    if supg and (pde.diffusion.uses_u or pde.advection.uses_u):
        raise ValueError("SUPG with u-dependent diffusion or advection is not supported")
    try:
        return _assemble_volume(space, pde, time, want_K, want_M, want_F, supg, u_state, newton)
    finally:
        clear_field()


def _newton_delta(uq):
    return 1e-7 * (1.0 + np.abs(uq))


def _assemble_volume(space, pde, time, want_K, want_M, want_F, supg, u_state, newton):
    mesh = space.mesh
    n = space.ndofs
    K = sp.csr_matrix((n, n))
    M = sp.csr_matrix((n, n))
    F = np.zeros(n)
    has_adv = not pde.advection.is_zero
    has_react = not (pde.reaction.is_constant and pde.reaction.const == 0.0)
    for p, cells_all, dofs_all in space.groups():
        basis = get_basis(p, space.basis_name)
        q, w = triangle_rule(_qdeg(p, pde))
        phi = basis.eval(q)
        gref = basis.grad(q)
        href = basis.hess(q) if supg and p >= 2 else None
        nb = basis.ndofs
        chunk = max(1, CHUNK_ENTRIES // max(1, len(w) * nb * nb))
        for s0 in range(0, len(cells_all), chunk):
            cells = cells_all[s0:s0 + chunk]
            dofs = dofs_all[s0:s0 + chunk]
            geo = space.geo.at(q, cells)
            X = geo.X
            x, y = X[..., 0], X[..., 1]
            wdet = w[None, :] * np.abs(geo.det)
            rows = np.repeat(dofs, nb, axis=1)
            cols = np.tile(dofs, (1, nb))
            uq = dfq = dcq = None
            if u_state is not None:
                ue = u_state[dofs]
                uq = np.einsum("qi,ei->eq", phi, ue, optimize=True)
                gq = np.einsum("eqia,ei->eqa", phys_grad(gref, geo.invJ), ue, optimize=True)
                field = dict(u=uq, ux=gq[..., 0], uy=gq[..., 1])
                if newton:
                    d = _newton_delta(uq)
                    if pde.source.uses_u:
                        set_field(**dict(field, u=uq + d))
                        f1 = pde.source(x, y, time)
                        set_field(**field)
                        dfq = (f1 - pde.source(x, y, time)) / d
                    if pde.reaction.uses_u:
                        set_field(**dict(field, u=uq + d))
                        c1 = pde.reaction(x, y, time)
                        set_field(**field)
                        dcq = (c1 - pde.reaction(x, y, time)) / d
                set_field(**field)
            if want_K:
                G = phys_grad(gref, geo.invJ)
                A = pde.diffusion(x, y, time)
                AG = np.einsum("eqab,eqjb->eqja", A, G, optimize=True)
                Ke = np.einsum("eq,eqia,eqja->eij", wdet, G, AG, optimize=True)
                bv = pde.advection(x, y, time) if has_adv else None
                if has_adv:
                    bG = np.einsum("eqa,eqja->eqj", bv, G, optimize=True)
                    Ke += np.einsum("eq,qi,eqj->eij", wdet, phi, bG, optimize=True)
                if has_react:
                    cq = pde.reaction(x, y, time)
                    Ke += np.einsum("eq,qi,qj->eij", wdet * cq, phi, phi, optimize=True)
                if dcq is not None:
                    Ke += np.einsum("eq,qi,qj->eij", wdet * dcq * uq, phi, phi, optimize=True)
                if dfq is not None:
                    Ke -= np.einsum("eq,qi,qj->eij", wdet * dfq, phi, phi, optimize=True)
                if supg and has_adv:
                    tau = supg_tau(mesh, cells, pde)
                    Lu = bG.copy()
                    if has_react:
                        Lu += cq[..., None] * phi[None]
                    if href is not None:
                        Hx = np.einsum("qiab,eqac,eqbd->eqicd", href, geo.invJ, geo.invJ, optimize=True)
                        Lu -= np.einsum("eqab,eqjab->eqj", A, Hx, optimize=True)
                    Ke += np.einsum("e,eq,eqi,eqj->eij", tau, wdet, bG, Lu, optimize=True)
                K = K + _coo_to_csr(rows, cols, Ke, n)
            if want_M:
                mq = pde.mass(x, y, time)
                Me = np.einsum("eq,qi,qj->eij", wdet * mq, phi, phi, optimize=True)
                M = M + _coo_to_csr(rows, cols, Me, n)
            if want_F:
                fq = pde.source(x, y, time)
                if dfq is not None:
                    fq = fq - dfq * uq
                if dcq is not None:
                    fq = fq + dcq * uq * uq
                Fe = np.einsum("eq,qi->ei", wdet * fq, phi, optimize=True)
                if supg and has_adv:
                    if not want_K:
                        G = phys_grad(gref, geo.invJ)
                        bG = np.einsum("eqa,eqja->eqj", pde.advection(x, y, time), G, optimize=True)
                        tau = supg_tau(mesh, cells, pde)
                    Fe += np.einsum("e,eq,eqi->ei", tau, wdet * fq, bG, optimize=True)
                np.add.at(F, dofs.ravel(), Fe.ravel())
    return K, M, F


def _boundary_data(space: FunctionSpace, bcs: list[BoundaryCondition]):
    mesh = space.mesh
    idx = assign_boundary_conditions(mesh, bcs) if bcs else np.full(len(mesh.boundary_edge_ids), -1)
    return mesh.boundary_edge_ids, idx


def _edge_tab(p: int, basis_name: str, s: np.ndarray, reverse: bool):
    b = get_basis(p, basis_name)
    phis, grads = [], []
    for e in range(3):
        ref = edge_reference_points(e, s, reverse)
        phis.append(b.eval(ref))
        grads.append(b.grad(ref))
    return np.stack(phis), np.stack(grads)


def _cell_group_lookup(space: FunctionSpace):
    """For every cell: (degree, row in its group's dofs array)."""
    pos = np.empty(space.mesh.nelements, dtype=np.int64)
    for p, cells, dofs in space.groups():
        pos[cells] = np.arange(len(cells))
    return pos


def _gather_dofs(space: FunctionSpace, cells: np.ndarray, p: int):
    if space.method == "CG" or space.uniform:
        return space.cell_dofs[cells]
    return space.offsets[cells][:, None] + np.arange((p + 1) * (p + 2) // 2)[None, :]


def assemble_boundary_natural(space: FunctionSpace, pde: PDE, bcs, time: float = 0.0, want_K=True, robin_only_load=False):
    """Neumann and Robin terms (CG and DG). Returns (K_b, F_b)."""
    mesh = space.mesh
    n = space.ndofs
    K = sp.csr_matrix((n, n))
    F = np.zeros(n)
    eids, idx = _boundary_data(space, bcs)
    if len(eids) == 0:
        return K, F
    s, w = line_rule(2 * space.degree + 2)
    cells = mesh.edge_cells[eids, 0]
    le = mesh.edge_local[eids, 0]
    X, nrm, ds, _ = space.geo.edge(eids, 0, s)
    for i, bc in enumerate(bcs):
        if bc.kind not in ("neumann", "robin"):
            continue
        sel = np.flatnonzero(idx == i)
        if not len(sel):
            continue
        g = bc.funcs[0](X[sel, :, 0], X[sel, :, 1], time)
        for p in np.unique(space.degrees[cells[sel]]):
            sp_ = sel[space.degrees[cells[sel]] == p]
            phi_t, _ = _edge_tab(int(p), space.basis_name, s, False)
            phi = phi_t[le[sp_]]
            dofs = _gather_dofs(space, cells[sp_], int(p))
            wl = w[None, :] * ds[sp_]
            gsel = g[np.searchsorted(sel, sp_)]
            Fe = np.einsum("eq,eqi->ei", wl * gsel, phi)
            np.add.at(F, dofs.ravel(), Fe.ravel())
            if bc.kind == "robin" and want_K:
                Ke = bc.alpha * np.einsum("eq,eqi,eqj->eij", wl, phi, phi)
                nb = dofs.shape[1]
                K = K + _coo_to_csr(np.repeat(dofs, nb, 1), np.tile(dofs, (1, nb)), Ke, n)
    return K, F


def cg_dirichlet(space: FunctionSpace, bcs, time: float = 0.0):
    mesh = space.mesh
    eids, idx = _boundary_data(space, bcs)
    vals = np.full(space.ndofs, np.nan)
    for i, bc in enumerate(bcs):
        if bc.kind != "dirichlet":
            continue
        sel = eids[idx == i]
        if not len(sel):
            continue
        d = np.unique(space.edge_dofs[sel])
        x = space.dof_coords[d]
        vals[d] = bc.funcs[0](x[:, 0], x[:, 1], time)
    dofs = np.flatnonzero(~np.isnan(vals))
    return dofs, vals[dofs]


def assemble_dg_facets(space: FunctionSpace, pde: PDE, bcs, penalty: float = 10.0, theta: float = 1.0, time: float = 0.0,
                       want_K=True):
    """SIPG (theta=1) / NIPG (-1) / IIPG (0) diffusion + upwind advection on all facets."""
    mesh = space.mesh
    n = space.ndofs
    K = sp.csr_matrix((n, n))
    F = np.zeros(n)
    pmax = space.degree
    s, w = line_rule(2 * pmax + 2)
    has_adv = not pde.advection.is_zero
    tabs = {}

    def tab(p, rev):
        key = (p, rev)
        if key not in tabs:
            tabs[key] = _edge_tab(p, space.basis_name, s, rev)
        return tabs[key]

    def block(dofs_a, dofs_b, Ke):
        na, nb_ = dofs_a.shape[1], dofs_b.shape[1]
        return _coo_to_csr(np.repeat(dofs_a, nb_, 1), np.tile(dofs_b, (1, na)), Ke, n)

    # interior
    ie = mesh.interior_edge_ids
    if len(ie) and want_K:
        cL, cR = mesh.edge_cells[ie, 0], mesh.edge_cells[ie, 1]
        eL, eR = mesh.edge_local[ie, 0], mesh.edge_local[ie, 1]
        X, nrmq, ds, invL = space.geo.edge(ie, 0, s)
        _, _, _, invR = space.geo.edge(ie, 1, s, reverse=True)
        nrm = nrmq[:, 0]
        L = mesh.edge_lengths[ie]
        A = pde.diffusion(X[..., 0], X[..., 1], time)
        kmax = np.maximum(pde.diffusion.max_eig(X[..., 0], X[..., 1]).max(1), 0.0)
        hF = np.minimum(mesh.areas[cL], mesh.areas[cR]) / L
        pL, pR = space.degrees[cL], space.degrees[cR]
        sig = penalty * (np.maximum(pL, pR) + 1) ** 2 * kmax / hF
        bn = np.einsum("eqa,eqa->eq", pde.advection(X[..., 0], X[..., 1], time), nrmq) if has_adv else None
        for p1 in np.unique(pL):
            for p2 in np.unique(pR):
                g = np.flatnonzero((pL == p1) & (pR == p2))
                if not len(g):
                    continue
                phiL_t, gL_t = tab(int(p1), False)
                phiR_t, gR_t = tab(int(p2), True)
                phiL, phiR = phiL_t[eL[g]], phiR_t[eR[g]]
                GL = np.einsum("eqib,eqba->eqia", gL_t[eL[g]], invL[g])
                GR = np.einsum("eqib,eqba->eqia", gR_t[eR[g]], invR[g])
                An = np.einsum("eqab,eqb->eqa", A[g], nrmq[g])
                fL = np.einsum("eqia,eqa->eqi", GL, An)
                fR = np.einsum("eqia,eqa->eqi", GR, An)
                wl = w[None, :] * ds[g]
                sides = ((phiL, fL, 1.0, _gather_dofs(space, cL[g], int(p1))),
                         (phiR, fR, -1.0, _gather_dofs(space, cR[g], int(p2))))
                for ta, (pha, fa, sa, da) in enumerate(sides):
                    for tb, (phb, fb, sb, db) in enumerate(sides):
                        Ke = (-0.5 * sa * np.einsum("eq,eqi,eqj->eij", wl, pha, fb)
                              - 0.5 * theta * sb * np.einsum("eq,eqi,eqj->eij", wl, fa, phb)
                              + sa * sb * np.einsum("eq,eqi,eqj->eij", wl * sig[g, None], pha, phb))
                        if has_adv:
                            up = np.maximum(-bn[g], 0) if ta == 0 else -np.maximum(bn[g], 0)
                            Ke += sb * np.einsum("eq,eqi,eqj->eij", wl * up, pha, phb)
                        K = K + block(da, db, Ke)
    # boundary
    eids, idx = _boundary_data(space, bcs)
    if len(eids):
        c = mesh.edge_cells[eids, 0]
        le = mesh.edge_local[eids, 0]
        X, nrmq, ds, invB = space.geo.edge(eids, 0, s)
        L = mesh.edge_lengths[eids]
        A = pde.diffusion(X[..., 0], X[..., 1], time)
        kmax = np.maximum(pde.diffusion.max_eig(X[..., 0], X[..., 1]).max(1), 0.0)
        sig = penalty * (space.degrees[c] + 1) ** 2 * kmax / (mesh.areas[c] / L)
        bn = np.einsum("eqa,eqa->eq", pde.advection(X[..., 0], X[..., 1], time), nrmq) if has_adv else np.zeros(X.shape[:2])
        kinds = np.array([bcs[i].kind if i >= 0 else "dirichlet" for i in idx], dtype=object)
        g = np.zeros(X.shape[:2])
        for i, bc in enumerate(bcs):
            m = idx == i
            if np.any(m):
                g[m] = bc.funcs[0](X[m, :, 0], X[m, :, 1], time)
        for p in np.unique(space.degrees[c]):
            sel = np.flatnonzero((space.degrees[c] == p) & (kinds == "dirichlet"))
            if not len(sel):
                continue
            phi_t, g_t = tab(int(p), False)
            phi = phi_t[le[sel]]
            G = np.einsum("eqib,eqba->eqia", g_t[le[sel]], invB[sel])
            fl = np.einsum("eqia,eqab,eqb->eqi", G, A[sel], nrmq[sel])
            wl = w[None, :] * ds[sel]
            inflow = np.maximum(-bn[sel], 0)
            dofs = _gather_dofs(space, c[sel], int(p))
            if want_K:
                Ke = (-np.einsum("eq,eqi,eqj->eij", wl, phi, fl) - theta * np.einsum("eq,eqi,eqj->eij", wl, fl, phi)
                      + np.einsum("eq,eqi,eqj->eij", wl * (sig[sel, None] + inflow), phi, phi))
                K = K + block(dofs, dofs, Ke)
            gs = g[sel]
            Fe = (-theta * np.einsum("eq,eqi->ei", wl * gs, fl)
                  + np.einsum("eq,eqi->ei", wl * gs * (sig[sel, None] + inflow), phi))
            np.add.at(F, dofs.ravel(), Fe.ravel())
    return K, F


def assemble(space: FunctionSpace, pde: PDE, bcs: list[BoundaryCondition] | None = None, dg_penalty: float = 10.0,
             apply_dirichlet: bool = True, time: float = 0.0, supg: bool = False, dg_theta: float = 1.0,
             u_state: np.ndarray | None = None, newton: bool = False) -> AssemblyResult:
    import time as _t
    bcs = list(bcs) if bcs else [BoundaryCondition("dirichlet", 0.0, "all")]
    t0 = _t.perf_counter()
    if space.method == "DG" and (pde.diffusion.uses_u or pde.advection.uses_u):
        raise ValueError("u-dependent diffusion or advection needs CG; DG supports u in c and f")
    K, M, F = assemble_volume(space, pde, time, supg=supg and space.method == "CG", u_state=u_state, newton=newton)
    Kb, Fb = assemble_boundary_natural(space, pde, bcs, time)
    K, F = K + Kb, F + Fb
    ddofs, dvals = np.zeros(0, dtype=np.int64), np.zeros(0)
    if space.method == "DG":
        Kf, Ff = assemble_dg_facets(space, pde, bcs, dg_penalty, dg_theta, time)
        K, F = K + Kf, F + Ff
    elif apply_dirichlet:
        ddofs, dvals = cg_dirichlet(space, bcs, time)
    res = AssemblyResult(K.tocsr(), F, M.tocsr(), space, ddofs, dvals)
    res.timings["assembly"] = _t.perf_counter() - t0
    return res


def assemble_load_vector(space: FunctionSpace, pde: PDE, bcs=None, time: float = 0.0, dg_penalty: float = 10.0,
                         supg: bool = False, dg_theta: float = 1.0, u_state: np.ndarray | None = None) -> np.ndarray:
    bcs = list(bcs) if bcs else [BoundaryCondition("dirichlet", 0.0, "all")]
    _, _, F = assemble_volume(space, pde, time, want_K=False, want_M=False, supg=supg and space.method == "CG",
                              u_state=u_state)
    F = F + assemble_boundary_natural(space, pde, bcs, time, want_K=False)[1]
    if space.method == "DG":
        F = F + assemble_dg_facets(space, pde, bcs, dg_penalty, dg_theta, time, want_K=False)[1]
    return F
