"""Discretization advisor: quantitative a-priori diagnostics, post-solve checks and concrete, applicable actions."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from femtoolbox.core.basis import LOCAL_EDGES
from femtoolbox.core.mesh import Mesh2D
from femtoolbox.core.pde import PDE

SEVERITIES = ("critical", "warning", "suggest", "info")
_DISCONTINUOUS = ("where", "heaviside", "sign", "abs", "floor", "ceil", "<", ">", "mod")


@dataclass
class Finding:
    severity: str
    topic: str
    title: str
    detail: str = ""
    action: dict = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return SEVERITIES.index(self.severity)


@dataclass
class AdvisorReport:
    method: str
    degree: int
    basis: str
    confidence: float
    reasons: list[str]
    warnings: list[str]
    cg_score: float
    dg_score: float
    solver: str = "auto"
    stabilization: str = "none"
    adaptivity: str = "none"
    metrics: dict = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    @property
    def recommended(self) -> dict:
        """Merged config changes of all findings that carry an action (higher severity wins)."""
        out: dict = {}
        for f in sorted(self.findings, key=lambda f: -f.rank):
            out.update(f.action)
        return out

    def summary(self) -> str:
        adapt = "" if self.adaptivity == "none" else f", {self.adaptivity} adaptivity"
        stab = "" if self.stabilization == "none" else f", {self.stabilization}"
        return f"{self.method}-P{self.degree} ({self.basis}), solver {self.solver}{stab}{adapt}"

    def as_text(self) -> str:
        m = self.metrics
        lines = [f"Recommended: {self.summary()}",
                 f"confidence {self.confidence:.2f}   (CG score {self.cg_score:.1f}, DG score {self.dg_score:.1f})",
                 f"h={m.get('h', 0):.3g}  Pe_h={m.get('peclet', 0):.3g}  anisotropy={m.get('anisotropy', 1):.3g}  "
                 f"kh={m.get('kh', 0):.3g}  min quality={m.get('quality', 1):.2f}  est. DOFs={m.get('ndofs', 0):,}"]
        for sev in SEVERITIES:
            group = [f for f in self.findings if f.severity == sev]
            if group:
                lines += ["", sev.upper()]
                for f in group:
                    lines.append(f"  [{f.topic}] {f.title}")
                    if f.detail:
                        lines.append(f"      {f.detail}")
                    if f.action:
                        lines.append("      -> " + ", ".join(f"{k} = {v}" for k, v in f.action.items()))
        return "\n".join(lines)


# ----------------------------------------------------------------------------- metrics
def _sample_points(mesh: Mesh2D, nmax: int = 600):
    c = mesh.nodes[mesh.triangles].mean(1)
    idx = np.arange(len(c)) if len(c) <= nmax else np.linspace(0, len(c) - 1, nmax).astype(int)
    return idx, c[idx]


def _metrics(pde: PDE, mesh: Mesh2D) -> dict:
    idx, c = _sample_points(mesh)
    x, y = c[:, 0], c[:, 1]
    q = mesh.quality()
    out = dict(h=float(mesh.diameters.mean()), h_max=float(mesh.diameters.max()), h_min=float(mesh.diameters.min()),
               quality=float(q.min()), quality_low=float((q < 0.3).mean()),
               wavenumber=abs(pde.wavenumber), kh=abs(pde.wavenumber) * float(mesh.diameters.mean()),
               smooth_coeffs=pde.diffusion.is_constant and pde.advection.is_constant, nonlinear=pde.is_nonlinear)
    try:
        from femtoolbox.core.expr import field_state
        z = np.zeros_like(x)
        with field_state(u=z, ux=z, uy=z):
            A = pde.diffusion(x, y)
            b = pde.advection(x, y)
            react = pde.reaction(x, y)
        lam = np.linalg.eigvalsh(0.5 * (A + np.swapaxes(A, -1, -2)))
        lmin = np.maximum(lam[:, 0], 1e-14)
        bn = np.linalg.norm(b, axis=-1)
        hc = mesh.diameters[idx]
        pe = bn * hc / (2 * lmin)
        out.update(lam_min=float(lmin.min()), lam_max=float(lam[:, 1].max()), adv=float(bn.max()),
                   reaction_min=float(react.min()), reaction_max=float(react.max()),
                   peclet=float(pe.max()), peclet_frac=float((pe > 1).mean()),
                   anisotropy=float((lam[:, 1] / lmin).max()),
                   damkohler=float((np.abs(react) * hc ** 2 / lmin).max()),
                   layer=float(lmin.min() / max(bn.max(), 1e-300)) if bn.max() > 1e-12 else np.inf)
    except Exception:
        out.update(lam_min=1.0, lam_max=1.0, adv=0.0, reaction_min=0.0, reaction_max=0.0, peclet=0.0, peclet_frac=0.0,
                   anisotropy=1.0, damkohler=0.0, layer=np.inf)
    return out


def boundary_corners(mesh: Mesh2D, bcs=None, min_angle_deg: float = 190.0):
    """Re-entrant corners and Dirichlet/Neumann junctions: [(x, y, angle_deg, alpha, kind)] with u ~ r^alpha."""
    eids = mesh.boundary_edge_ids
    cells, le = mesh.edge_cells[eids, 0], mesh.edge_local[eids, 0]
    loc = np.asarray(LOCAL_EDGES)[le]
    tri = mesh.triangles[cells]
    r = np.arange(len(eids))
    a, b = tri[r, loc[:, 0]], tri[r, loc[:, 1]]
    kinds = np.full(len(eids), "dirichlet", dtype=object)
    if bcs:
        from femtoolbox.core.pde import assign_boundary_conditions
        try:
            idx = assign_boundary_conditions(mesh, bcs)
            kinds = np.array(["dirichlet" if i < 0 or bcs[i].kind in ("dirichlet", "fixed") else "natural"
                              for i in idx], dtype=object)
        except Exception:
            pass
    out_edge = {int(u): k for k, u in enumerate(a)}
    in_edge = {int(v): k for k, v in enumerate(b)}
    P = mesh.nodes
    res = []
    for v, ko in out_edge.items():
        ki = in_edge.get(v)
        if ki is None:
            continue
        d1, d2 = P[b[ko]] - P[v], P[a[ki]] - P[v]
        omega = float(np.arctan2(d1[0] * d2[1] - d1[1] * d2[0], d1 @ d2) % (2 * np.pi))
        if omega < 1e-9:
            continue
        mixed = kinds[ko] != kinds[ki]
        alpha = np.pi / (2 * omega) if mixed else np.pi / omega
        if (not mixed and np.degrees(omega) >= min_angle_deg) or (mixed and alpha < 0.95):
            res.append((float(P[v, 0]), float(P[v, 1]), float(np.degrees(omega)), float(alpha),
                        "Dirichlet/Neumann junction" if mixed else "re-entrant corner"))
    res.sort(key=lambda t: t[3])
    return res


def _ndofs(mesh: Mesh2D, method: str, p: int, components: int = 1) -> int:
    p = max(int(p), 0 if method == "DG" else 1)
    if method == "DG":
        n = mesh.nelements * (p + 1) * (p + 2) // 2
    else:
        n = mesh.nnodes + mesh.nedges * (p - 1) + mesh.nelements * max(p - 1, 0) * max(p - 2, 0) // 2
    return int(n * components)


def _resolution(expr, mesh: Mesh2D) -> float:
    """Largest within-cell variation of a coefficient relative to its range (0 = resolved, ~1 = not)."""
    if getattr(expr, "is_constant", True) or getattr(expr, "uses_u", False):
        return 0.0
    idx, c = _sample_points(mesh, 2000)
    V = mesh.nodes[mesh.triangles[idx]]
    try:
        fc = expr(c[:, 0], c[:, 1])
        fv = expr(V[..., 0], V[..., 1])
    except Exception:
        return 0.0
    rng = float(np.ptp(np.r_[fc, fv.ravel()]))
    if not np.isfinite(rng) or rng < 1e-14:
        return 0.0
    return float(np.max(np.abs(fv - fc[:, None])) / rng)


def _pollution_degree(k: float, h: float, target: float = 0.1, pmax: int = 8) -> int:
    for p in range(1, pmax + 1):
        if k * (k * h / (2 * p)) ** (2 * p) < target and 2 * np.pi * p / max(k * h, 1e-12) >= 6:
            return p
    return pmax


# ----------------------------------------------------------------------------- main entry
def recommend(pde: PDE, mesh: Mesh2D, degree: int = 1, basis: str = "lagrange", need_local_conservation: bool = False,
              hp_objective: bool = False, *, method: str | None = None, analysis: str = "Steady", bcs=None,
              solver: str = "auto", supg: bool = False, penalty: float = 10.0, dt: float | None = None,
              nsteps: int | None = None, scheme: str | None = None, u0=None, n_modes: int | None = None,
              elasticity=None, solution=None, eta=None, history=None) -> AdvisorReport:
    f = _metrics(pde, mesh)
    F: list[Finding] = []
    cg = dg = 0.0
    stab, adapt = "none", "none"
    deg = int(degree)
    cur = (method or "CG").upper()
    elastic = analysis == "Elasticity" or elasticity is not None

    # advection / reaction / anisotropy
    if not elastic:
        if f["adv"] <= 1e-12:
            cg += 3
            F.append(Finding("info", "Method", "Diffusion/reaction dominated",
                             "CG is the cheapest stable choice for H1-regular solutions."))
        elif f["peclet"] > 10:
            dg += 5
            stab = "upwind (DG) or SUPG (CG)"
            bad = cur == "CG" and not supg
            F.append(Finding("critical" if bad else "info", "Stability",
                             f"Advection dominated: mesh Peclet up to {f['peclet']:.3g} ({100 * f['peclet_frac']:.0f}% of cells > 1)",
                             f"Layer width ~ eps/|b| = {f['layer']:.2e} vs h = {f['h']:.2e}. Galerkin CG oscillates; "
                             "use SUPG (CG) or upwind DG, and refine towards outflow boundaries.",
                             {"supg": True} if bad else {}))
        elif f["peclet"] > 1:
            cg += 1
            dg += 2
            stab = "SUPG"
            if cur == "CG" and not supg:
                F.append(Finding("warning", "Stability", f"Moderate mesh Peclet {f['peclet']:.3g}",
                                 "Plain Galerkin may show wiggles near layers.", {"supg": True}))
        else:
            cg += 2
            F.append(Finding("info", "Stability", f"Mesh Peclet {f['peclet']:.2g} < 1", "Galerkin CG is stable."))
        if f["damkohler"] > 10 and f["reaction_min"] > 0:
            F.append(Finding("warning", "Accuracy", f"Reaction dominated (Da_h = {f['damkohler']:.3g})",
                             "Expect boundary layers of width sqrt(A/c); refine near boundaries or use adaptivity.",
                             {"strategy": "h"}))
        if f["anisotropy"] > 100:
            dg += 2
            F.append(Finding("warning", "Accuracy", f"Strong diffusion anisotropy {f['anisotropy']:.3g}",
                             "Align the mesh with the principal directions or refine adaptively."))
        elif f["anisotropy"] > 10:
            cg += 0.5
            dg += 1
    if need_local_conservation:
        dg += 3
        F.append(Finding("suggest", "Method", "Local conservation requested", "DG fluxes balance cell by cell.",
                         {"method": "DG"}))
    if hp_objective:
        dg += 2
        adapt = "hp (DG)"

    # singularities from geometry and boundary conditions
    corners = boundary_corners(mesh, bcs)
    if corners and analysis != "Eigenmodes":
        x, y, ang, alpha, kind = corners[0]
        rate = min(max(deg, 1), alpha)
        adapt = adapt if adapt != "none" else "h"
        act = {"strategy": "hp", "method": "DG", "basis": "modal"} if (hp_objective or cur == "DG") else {"strategy": "h"}
        F.append(Finding("warning", "Accuracy", f"Singularity: {kind} at ({x:.3g}, {y:.3g}), angle {ang:.0f} deg",
                         f"u ~ r^{alpha:.3f}: uniform refinement converges like h^{rate:.2f} in H1 for any p. "
                         "Run the adaptive loop (h, or hp with DG) to recover N^(-p/2)."
                         + (f" {len(corners) - 1} more corner(s)." if len(corners) > 1 else ""), act))
        f["corner_alpha"] = alpha

    # data resolution and discontinuities
    srcs = [("f", pde.source)]
    if pde.diffusion.entries:
        srcs += [("A", e) for row in pde.diffusion.entries for e in row]
    worst = max(((n, _resolution(e, mesh)) for n, e in srcs), key=lambda t: t[1], default=("", 0.0))
    if worst[1] > 0.35:
        F.append(Finding("warning", "Accuracy",
                         f"Data under-resolved: {worst[0]} varies by {100 * worst[1]:.0f}% of its range within one cell",
                         "Refine the mesh (or run the adaptive loop) until features of the data span several cells."))
    texts = " ".join(str(getattr(e, "source", "")) for _, e in srcs + [("c", pde.reaction)])
    if any(tok in texts for tok in _DISCONTINUOUS):
        F.append(Finding("suggest", "Accuracy", "Possibly discontinuous coefficients or data",
                         "Rates drop to h^(1/2) across interfaces unless the mesh is aligned with them; adaptivity helps.",
                         {"strategy": "h"}))

    # Helmholtz / indefinite
    if f["wavenumber"] > 0:
        k, hmax = f["wavenumber"], f["h_max"]
        pd = max(deg, 1)
        ppw = 2 * np.pi * pd / max(k * hmax, 1e-12)
        need = _pollution_degree(k, hmax)
        sev = "critical" if ppw < 6 else ("warning" if need > deg else "info")
        F.append(Finding(sev, "Accuracy", f"Helmholtz k = {k:.3g}: {ppw:.1f} points per wavelength at P{pd}",
                         f"Pollution error ~ k (kh/2p)^(2p) = {k * (k * hmax / (2 * pd)) ** (2 * pd):.2e}. "
                         f"P{need} keeps it below 0.1 on this mesh.", {"degree": need} if need > deg else {}))
        deg = max(deg, need)
        F.append(Finding("info" if solver in ("auto", "direct") else "critical", "Solver", "Indefinite operator",
                         "Use sparse direct LU; CG/AMG do not apply.",
                         {} if solver in ("auto", "direct") else {"solver": "direct"}))
    elif f["reaction_min"] < 0:
        F.append(Finding("warning", "Solver", "Negative reaction: the operator may be indefinite", "Use a direct solver.",
                         {} if solver in ("auto", "direct") else {"solver": "direct"}))

    # mesh
    if f["quality"] < 0.2:
        F.append(Finding("warning", "Mesh", f"Poor element quality (min {f['quality']:.2f}, {100 * f['quality_low']:.0f}% < 0.3)",
                         "Smooth the mesh or use an unstructured domain; bad angles hurt conditioning."))
    grade = f["h_max"] / max(f["h_min"], 1e-300)
    if grade > 50:
        F.append(Finding("info", "Mesh", f"Strongly graded mesh (h_max/h_min = {grade:.0f})",
                         "Fine for adaptivity; explicit time stepping would be limited by h_min."))

    # method / basis consistency
    if cur == "DG" and deg == 0 and not elastic:
        F.append(Finding("critical", "Method", "DG with p = 0 is inconsistent for diffusion",
                         "SIPG at p = 0 does not converge. Use p >= 1.", {"degree": 1}))
        deg = 1
    if cur == "DG" and basis == "lagrange" and deg >= 3:
        F.append(Finding("suggest", "Method", "Modal basis for high-order DG",
                         "The orthonormal modal basis gives a diagonal mass matrix and better conditioning.",
                         {"basis": "modal"}))
    if cur == "DG" and penalty < 2:
        F.append(Finding("warning", "Method", f"Low interior-penalty constant ({penalty:g})",
                         "SIPG needs C_ip large enough for coercivity (default 10).", {"penalty": 10.0}))
    if (f["smooth_coeffs"] and f["adv"] <= 1e-12 and not corners and deg < 2 and not elastic
            and f["wavenumber"] == 0):
        F.append(Finding("suggest", "Accuracy", "Smooth problem: raise the degree",
                         "P2+ reaches a given accuracy with far fewer DOFs than P1.", {"degree": 2}))
        deg = 2

    # elasticity
    if elastic and elasticity is not None:
        try:
            nu = float(np.max(elasticity.nu(*_sample_points(mesh)[1].T))) if callable(elasticity.nu) else float(elasticity.nu)
        except Exception:
            nu = 0.3
        if nu > 0.45:
            F.append(Finding("warning" if deg < 2 else "info", "Accuracy", f"Nearly incompressible (nu = {nu:.3f})",
                             "Low-order elements lock volumetrically; use p >= 2"
                             + (" (plane strain is worse than plane stress)." if getattr(elasticity, "plane", "") == "strain" else "."),
                             {"degree": 2} if deg < 2 else {}))
        xmin, xmax, ymin, ymax = mesh.bbox
        aspect = max(xmax - xmin, ymax - ymin) / max(min(xmax - xmin, ymax - ymin), 1e-300)
        if deg == 1 and aspect > 3:
            F.append(Finding("warning", "Accuracy", f"Slender body (aspect {aspect:.1f}) with P1: shear locking",
                             "Linear triangles are far too stiff in bending; P2 or P3 is accurate on the same mesh.",
                             {"degree": 3}))

    # nonlinear
    if f["nonlinear"]:
        picard = pde.diffusion.uses_u or pde.advection.uses_u
        F.append(Finding("info", "Solver", "Nonlinear problem",
                         ("A or b depend on u: Picard iteration (linear convergence). If it stalls, use relaxation 0.5-0.8."
                          if picard else "u enters only c and f: exact Newton linearization (quadratic convergence).")
                         + " For strong nonlinearities use continuation in the parameter."))
        if cur == "DG" and picard:
            F.append(Finding("critical", "Method", "u-dependent A or b needs CG", "", {"method": "CG"}))

    best = "DG" if dg > cg else "CG"
    if f["nonlinear"] and (pde.diffusion.uses_u or pde.advection.uses_u):
        best = "CG"
    if best == "DG" and deg >= 2:
        basis = "modal"

    # cost and solver
    comps = 2 if elastic else 1
    N = _ndofs(mesh, best, deg, comps)
    nb = (deg + 1) * (deg + 2) // 2 * comps
    nnz = int(mesh.nelements * nb * nb * (4.0 if best == "DG" else 0.6))
    t_est = 0.07 * (N / 4225) ** 1.1 * (1 + 0.1 * deg)
    sym = (pde.is_symmetric or elastic) and not supg
    indefinite = f["wavenumber"] > 0 or f["reaction_min"] < 0
    if indefinite:
        sol_rec = "direct"
    elif N <= 250_000:
        sol_rec = "direct" if sym or N <= 150_000 else "gmres-ilu"
    else:
        sol_rec = "cg-amg" if sym else "gmres-ilu"
    if solver.startswith("cg") and not sym:
        F.append(Finding("critical", "Solver", f"Solver {solver} requires a symmetric operator",
                         "Advection, SUPG or nonsymmetric DG make the matrix nonsymmetric.", {"solver": sol_rec}))
    elif solver not in ("auto", sol_rec) and N > 50_000:
        F.append(Finding("suggest", "Solver", f"{sol_rec} is expected to be fastest for N = {N:,}", "",
                         {"solver": sol_rec}))
    F.append(Finding("info", "Cost", f"~{N:,} DOFs, ~{nnz / 1e6:.2g}M nonzeros, ~{t_est:.2g} s per solve",
                     f"Recommended solver: {sol_rec}."))

    # transient
    if analysis == "Transient" and dt:
        sch = (scheme or "").lower()
        h = f["h"]
        if sch.startswith("backward") and dt > 10 * h ** (deg + 1):
            F.append(Finding("suggest", "Time stepping", f"Backward Euler with dt = {dt:g} dominates the error",
                             f"Spatial error ~ h^(p+1) = {h ** (deg + 1):.1e} but BE error ~ dt. BDF2 or CN are second order.",
                             {"scheme": "bdf2"}))
        lam_max = f["lam_max"] * (deg + 1) ** 4 / max(f["h_min"], 1e-300) ** 2
        if sch == "crank-nicolson" and dt * lam_max > 10:
            compat = True
            if bcs and u0 is not None:
                try:
                    from femtoolbox.core.expr import Expr
                    from femtoolbox.core.pde import assign_boundary_conditions
                    mid = mesh.nodes[mesh.edges[mesh.boundary_edge_ids]].mean(1)
                    idx = assign_boundary_conditions(mesh, bcs)
                    g = np.zeros(len(mid))
                    dmask = np.zeros(len(mid), bool)
                    for i, bc in enumerate(bcs):
                        sel = idx == i
                        if bc.kind == "dirichlet" and np.any(sel):
                            g[sel] = bc.funcs[0](mid[sel, 0], mid[sel, 1], 0.0)
                            dmask |= sel
                    u0v = Expr(u0)(mid[:, 0], mid[:, 1], 0.0)
                    compat = not np.any(np.abs(u0v - g)[dmask] > 1e-8 * (1 + np.abs(g).max(initial=0)))
                except Exception:
                    pass
            F.append(Finding("info" if compat else "warning", "Time stepping",
                             "Stiff modes with Crank-Nicolson" + ("" if compat else ": u0 does not match the Dirichlet data"),
                             f"dt * lambda_max ~ {dt * lam_max:.1e}: CN damps high-frequency errors poorly"
                             + ("." if compat else "; the incompatible start will ring. Rannacher start-up fixes it."),
                             {} if compat else {"scheme": "rannacher"}))
        if nsteps:
            T = dt * nsteps
            L = max(mesh.bbox[1] - mesh.bbox[0], mesh.bbox[3] - mesh.bbox[2])
            tau = L ** 2 / (np.pi ** 2 * 2 * max(f["lam_min"], 1e-300))
            F.append(Finding("info", "Time stepping", f"Final time {T:g}, diffusive time scale ~ {tau:.3g}",
                             "Steady state is reached after a few time scales; a steady solve is cheaper."
                             if T > 5 * tau else ""))

    # eigenproblems
    if analysis == "Eigenmodes" and n_modes:
        area = float(mesh.areas.sum())
        kk = np.sqrt(4 * np.pi * n_modes / area)
        ppw = 2 * np.pi / kk / f["h_max"] * max(deg, 1)
        if ppw < 8:
            F.append(Finding("warning", "Accuracy", f"Mode {n_modes} is under-resolved (~{ppw:.1f} points per wavelength)",
                             "Higher eigenvalues converge like h^(2p) once the mode is resolved (~8+ points per wavelength).",
                             {"degree": min(deg + 1, 8)}))
        else:
            F.append(Finding("info", "Accuracy", f"Highest requested mode has ~{ppw:.0f} points per wavelength",
                             f"Eigenvalues converge like h^{2 * max(deg, 1)} (twice the energy-norm rate)."))

    # post-solve checks
    if solution is not None:
        info = getattr(solution, "info", None)
        if info is not None and np.isfinite(info.residual_norm) and info.residual_norm > 1e-6:
            F.append(Finding("warning", "Solver", f"Linear residual {info.residual_norm:.1e}",
                             "The iterative solver stopped early; tighten the tolerance or use direct.",
                             {"solver": "direct"} if N < 250_000 else {}))
        if info is not None and any("did not converge" in w for w in info.warnings):
            F.append(Finding("critical", "Solver", "Nonlinear iteration did not converge",
                             "Lower the relaxation, give u0 as initial guess or continue in the parameter.",
                             {"nl_relax": 0.6}))
        if eta is not None and len(eta) > 20 and np.all(np.isfinite(eta)) and (eta ** 2).sum() > 0:
            e2 = np.sort(eta ** 2)[::-1]
            frac = (int(np.searchsorted(np.cumsum(e2), 0.5 * e2.sum())) + 1) / len(eta)
            if frac < 0.05:
                F.append(Finding("warning", "Accuracy",
                                 f"Error is concentrated: {100 * frac:.1f}% of cells carry half of eta^2",
                                 "A local feature or singularity dominates; adaptive refinement beats uniform refinement.",
                                 {"strategy": "h"} if cur == "CG" else {"strategy": "hp"}))
            elif frac > 0.2:
                F.append(Finding("info", "Accuracy", f"Error is spread out ({100 * frac:.0f}% of cells carry half of eta^2)",
                                 "Uniform p-enrichment is efficient for smooth solutions." if deg < 4 else ""))
            else:
                F.append(Finding("info", "Accuracy", f"Error is moderately local ({100 * frac:.0f}% of cells carry half of eta^2)",
                                 "Adaptive refinement gives a modest gain over uniform refinement."))
        try:
            vals = np.asarray(solution.values)
            src_ok = pde.source.is_constant and (pde.source.const or 0) >= 0
            if not elastic and f["adv"] > 1e-12 and src_ok and f["reaction_min"] >= 0 and vals.size:
                umin, umax = float(vals.min()), float(vals.max())
                if umin < -1e-3 * max(abs(umax), 1e-300):
                    F.append(Finding("critical" if (cur == "CG" and not supg) else "warning", "Stability",
                                     f"Spurious undershoot: min u = {umin:.3g} with non-negative data",
                                     "The discrete maximum principle is violated by unresolved layers.",
                                     {"supg": True} if cur == "CG" and not supg else {}))
        except Exception:
            pass
    if history:
        pts = [(n, e) for n, e in history if e and e > 0]
        if len(pts) >= 3:
            (n0, e0), (n1, e1) = pts[-3], pts[-1]
            r = -2 * np.log(e1 / e0) / np.log(n1 / n0)
            exp_rate = min(max(deg, 1), f.get("corner_alpha", max(deg, 1)))
            F.append(Finding("info" if r > 0.8 * exp_rate else "warning", "Accuracy",
                             f"Observed H1 rate ~ h^{r:.2f} (expected h^{exp_rate:.2f} under uniform refinement)",
                             "" if r > 0.8 * exp_rate else "Pre-asymptotic or limited by regularity: try adaptivity."))

    F.sort(key=lambda x: x.rank)
    deg = max([deg] + [int(x.action["degree"]) for x in F if "degree" in x.action])
    f.update(ndofs=N, nnz=nnz, t_est=t_est)
    conf = min(0.98, 0.5 + abs(dg - cg) / (2 * (abs(dg - cg) + 1)))
    reasons = [x.title + (": " + x.detail if x.detail else "") for x in F if x.severity in ("info", "suggest")]
    warnings = [x.title + (": " + x.detail if x.detail else "") for x in F if x.severity in ("warning", "critical")]
    return AdvisorReport(best, deg, basis, conf, reasons, warnings, cg, dg, sol_rec, stab, adapt, f, F)
