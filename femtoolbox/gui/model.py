"""GUI-independent configuration, problem construction and job execution."""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from femtoolbox.core.adapt import LSHAPE_EXACT, LSHAPE_UX, LSHAPE_UY, adaptive_solve, convergence_study, mark_dorfler
from femtoolbox.core.advisor import recommend
from femtoolbox.core.domain import (AnnulusDomain, DiskDomain, LShapeDomain, ParametricDomain, PlateWithHoleDomain,
                                    PolygonDomain, RectangleDomain, SquareDomain)
from femtoolbox.core.elasticity import Elasticity, solve_elasticity, timoshenko_beam
from femtoolbox.core.estimate import error_norms, residual_estimator, smoothness_indicator, zz_estimator
from femtoolbox.core.mesh import bisect, refine_uniform, smooth_laplacian
from femtoolbox.core.pde import PDE, BoundaryCondition, parse_boundary_conditions
from femtoolbox.core.solver import solve, solve_eigen, solve_transient

DOMAINS = ["Square", "Rectangle", "Disk", "L-shape", "Annulus", "Plate with hole", "Polygon", "Parametric", "Gmsh file"]
ANALYSES = ["Steady", "Transient", "Eigenmodes", "Elasticity"]
PDE_PRESETS = ["Poisson", "Diffusion", "Reaction-diffusion", "Advection-diffusion", "Helmholtz", "Heat", "Custom PDE"]
SCALAR_BC = ["dirichlet", "neumann", "robin"]
VECTOR_BC = ["free", "fixed", "fixed_x", "fixed_y", "traction", "pressure"]
SCHEMES = ["crank-nicolson", "backward-euler", "bdf2", "rannacher"]
SOLVERS = ["auto", "direct", "cg-amg", "cg-ilu", "cg-jacobi", "gmres-ilu", "gmres-amg", "bicgstab-ilu"]

DEFAULT_CUSTOM = """name = Rotating advection-diffusion
Axx = 0.01
Axy = 0
Ayy = 0.01
bx = -(y-0.5)
by = x-0.5
c = 0
f = exp(-40*((x-0.7)^2+(y-0.5)^2))
latex = -\\nabla\\cdot(A\\nabla u)+\\mathbf{b}\\cdot\\nabla u+c u=f
"""
DEFAULT_PARAMETRIC = """name = flower
x = (1+0.25*cos(5*t))*cos(t)
y = (1+0.25*cos(5*t))*sin(t)
t0 = 0
t1 = 2*pi
boundary_markers = 5
"""


@dataclass
class Config:
    analysis: str = "Steady"
    domain: str = "Square"
    width: float = 1.0
    height: float = 1.0
    radius: float = 1.0
    inner_radius: float = 0.5
    hole_radius: float = 0.2
    hole_x: str = ""
    hole_y: str = ""
    polygon: str = "0,0; 1,0; 1,0.5; 0.4,1; 0,0.7"
    parametric: str = DEFAULT_PARAMETRIC
    mesh_file: str = ""
    structured: bool = True
    n: int = 16
    h: float = 0.0
    pde: str = "Poisson"
    Axx: str = "1"
    Axy: str = "0"
    Ayy: str = "1"
    bx: str = "0"
    by: str = "0"
    c: str = "0"
    f: str = "1"
    m: str = "1"
    k: float = 10.0
    custom: str = DEFAULT_CUSTOM
    u_exact: str = ""
    ux: str = ""
    uy: str = ""
    bc_rows: list = field(default_factory=list)  # [marker, kind, value, alpha]
    bc_default: str = "dirichlet"
    bc_extra: str = ""
    method: str = "CG"
    degree: int = 2
    basis: str = "lagrange"
    supg: bool = False
    penalty: float = 10.0
    solver: str = "auto"
    tol: float = 1e-10
    strategy: str = "h"
    estimator: str = "residual"
    theta_mark: float = 0.5
    max_dofs: int = 20000
    max_levels: int = 25
    smooth_threshold: float = 1.5
    p_max: int = 8
    dt: float = 0.01
    nsteps: int = 100
    scheme: str = "crank-nicolson"
    u0: str = "0"
    n_modes: int = 8
    E: str = "1000"
    nu: str = "0.3"
    plane: str = "stress"
    fx: str = "0"
    fy: str = "0"
    nl_tol: float = 1e-8
    nl_maxit: int = 40
    nl_relax: float = 1.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def _num(s, default=0.0):
    from femtoolbox.core.expr import numeric
    try:
        return float(numeric(s)) if str(s).strip() else default
    except Exception:
        return default


def build_domain(cfg: Config):
    d = cfg.domain
    if d == "Square":
        return SquareDomain(cfg.width)
    if d == "Rectangle":
        return RectangleDomain(cfg.width, cfg.height)
    if d == "Disk":
        return DiskDomain(cfg.radius)
    if d == "L-shape":
        return LShapeDomain(cfg.width)
    if d == "Annulus":
        return AnnulusDomain(cfg.inner_radius, cfg.radius)
    if d == "Plate with hole":
        hc = None
        if str(cfg.hole_x).strip() or str(cfg.hole_y).strip():
            hc = (_num(cfg.hole_x, cfg.width / 2), _num(cfg.hole_y, cfg.height / 2))
        return PlateWithHoleDomain(cfg.width, cfg.height, hc, cfg.hole_radius)
    if d == "Polygon":
        pts = []
        for chunk in cfg.polygon.replace("\n", ";").split(";"):
            if chunk.strip():
                x, y = chunk.replace("(", "").replace(")", "").split(",")[:2]
                pts.append((_num(x), _num(y)))
        return PolygonDomain(pts)
    if d == "Parametric":
        return ParametricDomain(cfg.parametric)
    raise ValueError(f"unknown domain {d!r}")


def build_mesh(cfg: Config):
    if cfg.domain == "Gmsh file":
        from femtoolbox.core.meshio import read_msh
        if not str(cfg.mesh_file).strip():
            raise ValueError("choose a .msh file (Geometry section or File > Import Gmsh mesh)")
        return read_msh(cfg.mesh_file)
    dom = build_domain(cfg)
    h = cfg.h if cfg.h and cfg.h > 0 else None
    if cfg.domain in ("Square", "Rectangle", "L-shape"):
        return dom.mesh(cfg.n, h=h, structured=cfg.structured and h is None)
    if cfg.domain == "Disk":
        return dom.mesh(n=cfg.n, h=h)
    return dom.mesh(cfg.n, h=h)


def build_pde(cfg: Config) -> PDE:
    ex = dict(exact=cfg.u_exact or None, exact_grad=(cfg.ux, cfg.uy))
    p = cfg.pde
    A = [[cfg.Axx, cfg.Axy], [cfg.Axy, cfg.Ayy]]
    if p == "Custom PDE":
        pde = PDE.custom_from_definition(cfg.custom)
        if cfg.u_exact and pde.exact is None:
            pde = PDE.linear_second_order(pde.diffusion, pde.advection, pde.reaction, pde.source, pde.mass, pde.name,
                                          latex=pde.latex, **ex)
        return pde
    if p == "Poisson":
        return PDE.poisson(cfg.Axx, cfg.f, **ex)
    if p == "Diffusion":
        return PDE.diffusion_pde(A, cfg.f, **ex)
    if p == "Reaction-diffusion":
        return PDE.reaction_diffusion(A, cfg.c, cfg.f, **ex)
    if p == "Advection-diffusion":
        return PDE.advection_diffusion(A, (cfg.bx, cfg.by), cfg.c, cfg.f, **ex)
    if p == "Helmholtz":
        return PDE.helmholtz(cfg.k, cfg.f, **ex)
    if p == "Heat":
        return PDE.linear_second_order(A, (cfg.bx, cfg.by), cfg.c, cfg.f, cfg.m, name="Heat",
                                       latex=r"m\,\partial_t u-\nabla\cdot(A\nabla u)+\mathbf{b}\cdot\nabla u+c u=f", **ex)
    raise ValueError(p)


def build_bcs(cfg: Config) -> list[BoundaryCondition]:
    elastic = cfg.analysis == "Elasticity"
    bcs = []
    for row in cfg.bc_rows:
        marker, kind, value, alpha = (list(row) + ["", "", "0", "1"])[:4]
        if elastic and kind == "free":
            kind, value = "traction", "0, 0"
        bcs.append(BoundaryCondition(kind, value if str(value).strip() else "0", marker, _num(alpha, 1.0)))
    if not elastic and not any(b.is_global_fallback for b in bcs):
        bcs.insert(0, BoundaryCondition(cfg.bc_default, 0.0, "all"))
    if cfg.bc_extra.strip():
        bcs.extend(parse_boundary_conditions(cfg.bc_extra))
    return bcs


def advisor_choice(cfg: Config, mesh, pde):
    rep = recommend(pde, mesh, cfg.degree, cfg.basis, hp_objective=cfg.strategy == "hp")
    return rep


@dataclass
class JobResult:
    kind: str
    solution: object = None
    mesh: object = None
    indicators: np.ndarray | None = None
    steps: list = field(default_factory=list)
    text: str = ""
    elapsed: float = 0.0
    pde: object = None
    bcs: list = field(default_factory=list)
    errors: object = None


def _method(cfg, mesh, pde):
    if cfg.method == "Auto":
        rep = advisor_choice(cfg, mesh, pde)
        return rep.method, rep.degree, rep.basis, f"advisor: {rep.method}-P{rep.degree} {rep.basis}\n"
    return cfg.method, cfg.degree, cfg.basis, ""


def run_job(kind: str, cfg: Config, mesh, progress=None) -> JobResult:
    t0 = time.perf_counter()
    bcs = build_bcs(cfg)
    if cfg.analysis == "Elasticity" or kind == "elasticity":
        prob = Elasticity(cfg.E, cfg.nu, cfg.fx, cfg.fy, cfg.plane)
        sol = solve_elasticity(prob, mesh, max(1, cfg.degree), bcs, cfg.solver)
        text = sol.info.as_text() + f"\nmax |u| = {np.abs(sol.values).max():.4g}, max von Mises = {sol.max_field('vm'):.4g}"
        return JobResult("elasticity", sol, mesh, text=text, elapsed=time.perf_counter() - t0, bcs=bcs)
    pde = build_pde(cfg)
    method, degree, basis, note = _method(cfg, mesh, pde)
    if method == "CG":
        degree = max(1, degree)
    if kind == "solve" and cfg.analysis == "Transient":
        sol = solve_transient(pde, mesh, cfg.u0, cfg.dt, cfg.nsteps, cfg.scheme, method, degree, basis, bcs, cfg.solver,
                              cfg.penalty, supg=cfg.supg, store_every=max(1, cfg.nsteps // 200))
        text = note + sol.info.as_text() + f"\n{sol.nsteps} stored steps, t_end = {sol.times[-1]:.4g}"
        return JobResult("transient", sol, mesh, text=text, elapsed=time.perf_counter() - t0, pde=pde, bcs=bcs)
    if kind == "solve" and cfg.analysis == "Eigenmodes":
        sol = solve_eigen(pde, mesh, cfg.n_modes, method, degree, basis, bcs, dg_penalty=cfg.penalty)
        text = note + sol.info.as_text()
        return JobResult("eigen", sol, mesh, text=text, elapsed=time.perf_counter() - t0, pde=pde, bcs=bcs)
    if kind == "adapt":
        steps_live = []

        def cb(row, sol, eta):
            steps_live.append(row)
            if progress:
                progress(row, sol, eta)
        strategy = cfg.strategy if method == "DG" else "h"
        res = adaptive_solve(pde, mesh, method, degree, basis, bcs, strategy, cfg.theta_mark, cfg.max_levels,
                             cfg.max_dofs, cfg.estimator if method == "CG" else "residual", pde.exact.source if pde.exact else None,
                             pde.exact_grad[0].source if pde.exact_grad else None,
                             pde.exact_grad[1].source if pde.exact_grad else None, cfg.p_max, cfg.smooth_threshold,
                             cfg.solver, cfg.supg, cb)
        text = note + res.table() + "\n" + res.solution.info.as_text()
        return JobResult("adapt", res.solution, res.solution.mesh, res.indicators, res.steps, text,
                         time.perf_counter() - t0, pde, bcs)
    if kind == "study":
        if pde.exact is None:
            raise ValueError("a uniform refinement study needs an exact solution (u_exact)")
        rows = convergence_study(pde, mesh, pde.exact.source, pde.exact_grad[0].source if pde.exact_grad else None,
                                 pde.exact_grad[1].source if pde.exact_grad else None, 5 if mesh.nelements < 2000 else 4,
                                 method, degree, basis, bcs, cfg.solver, cfg.supg)
        text = f"{'lvl':>3} {'h':>9} {'DOFs':>8} {'L2':>10} {'rate':>5} {'H1':>10} {'rate':>5} {'eta':>10} {'time':>7}\n"
        for r in rows:
            text += (f"{r.level:>3} {r.h:>9.3e} {r.ndofs:>8} {r.l2:>10.3e} {'' if r.l2_order is None else f'{r.l2_order:5.2f}':>5} "
                     f"{r.h1:>10.3e} {'' if r.h1_order is None else f'{r.h1_order:5.2f}':>5} {r.eta:>10.3e} {r.time:>6.2f}s\n")
        return JobResult("study", None, mesh, steps=rows, text=note + text, elapsed=time.perf_counter() - t0, pde=pde, bcs=bcs)
    sol = solve(pde, mesh, method, degree, basis, bcs, cfg.solver, cfg.penalty, cfg.supg, tol=cfg.tol,
                nonlinear_tol=cfg.nl_tol, max_iter=cfg.nl_maxit, relax=cfg.nl_relax, u_init=cfg.u0 if pde.is_nonlinear else None)
    if pde.is_nonlinear:
        eta = zz_estimator(sol) if method == "CG" else np.zeros(mesh.nelements)
    else:
        eta = zz_estimator(sol) if (cfg.estimator == "zz" and method == "CG") else residual_estimator(sol, pde, bcs)
    text = note + sol.info.as_text() + f"\nestimator eta = {np.sqrt((eta ** 2).sum()):.4e}"
    errs = None
    if pde.exact is not None:
        errs = error_norms(sol, pde.exact.source, pde.exact_grad[0].source if pde.exact_grad else None,
                           pde.exact_grad[1].source if pde.exact_grad else None)
        text += f"\n|u-u_h|_L2 = {errs.l2:.4e}   |u-u_h|_H1 = {errs.h1_semi:.4e}   effectivity = {np.sqrt((eta ** 2).sum()) / max(errs.h1_semi, 1e-300):.2f}"
    return JobResult("solve", sol, mesh, eta, text=text, elapsed=time.perf_counter() - t0, pde=pde, bcs=bcs, errors=errs)


def refine_marked_once(cfg: Config, sol, pde, bcs):
    from femtoolbox.core.estimate import estimate_error
    eta = estimate_error(sol, pde, bcs)
    marked = mark_dorfler(eta, cfg.theta_mark)
    return bisect(sol.mesh, marked)[0], marked


# ----------------------------------------------------------------------------- presets
_LBC = [["all", "dirichlet", LSHAPE_EXACT, "1"]]
_TB = [re.sub(r"\by\b", "(y-1)", e) for e in timoshenko_beam(10.0, 2.0, 1.0, 1000.0, 0.3)]
_KIRSCH_SXX = "1 - 0.25/(x^2+y^2)*(1.5*cos(2*theta)+cos(4*theta)) + 1.5*0.0625/(x^2+y^2)^2*cos(4*theta)"
_KIRSCH_SYY = "-0.25/(x^2+y^2)*(0.5*cos(2*theta)-cos(4*theta)) - 1.5*0.0625/(x^2+y^2)^2*cos(4*theta)"
_KIRSCH_SXY = "-0.25/(x^2+y^2)*(0.5*sin(2*theta)+sin(4*theta)) + 1.5*0.0625/(x^2+y^2)^2*sin(4*theta)"

PRESETS: dict[str, dict] = {
    "Poisson, manufactured solution (P3)": dict(
        analysis="Steady", domain="Square", width=1.0, n=8, structured=True, pde="Poisson", Axx="1",
        f="2*pi^2*sin(pi*x)*sin(pi*y)", u_exact="sin(pi*x)*sin(pi*y)", ux="pi*cos(pi*x)*sin(pi*y)",
        uy="pi*sin(pi*x)*cos(pi*y)", bc_rows=[["all", "dirichlet", "0", "1"]], method="CG", degree=3),
    "L-shape corner singularity, h-adaptive CG-P2": dict(
        analysis="Steady", domain="L-shape", width=1.0, n=4, structured=True, pde="Poisson", Axx="1", f="0",
        u_exact=LSHAPE_EXACT, ux=LSHAPE_UX, uy=LSHAPE_UY, bc_rows=_LBC, method="CG", degree=2, strategy="h",
        max_dofs=20000, max_levels=30, theta_mark=0.5),
    "L-shape corner singularity, hp-adaptive DG": dict(
        analysis="Steady", domain="L-shape", width=1.0, n=4, structured=True, pde="Poisson", Axx="1", f="0",
        u_exact=LSHAPE_EXACT, ux=LSHAPE_UX, uy=LSHAPE_UY, bc_rows=_LBC, method="DG", degree=2, basis="modal",
        strategy="hp", max_dofs=8000, max_levels=60, smooth_threshold=1.5, theta_mark=0.5),
    "Boundary layer: advection-diffusion with SUPG": dict(
        analysis="Steady", domain="Square", width=1.0, n=32, structured=True, pde="Advection-diffusion", Axx="0.002",
        Axy="0", Ayy="0.002", bx="1", by="0.5", c="0", f="1", u_exact="", ux="", uy="",
        bc_rows=[["all", "dirichlet", "0", "1"]], method="CG", degree=1, supg=True),
    "Rotating plume (DG upwind, custom PDE)": dict(
        analysis="Steady", domain="Disk", radius=1.0, n=14, pde="Custom PDE", custom="""name = Rotating plume
Axx = 0.002
Axy = 0
Ayy = 0.002
bx = -y
by = x
c = 0.05
f = exp(-60*((x-0.5)^2+y^2))
""", u_exact="", ux="", uy="", bc_rows=[["all", "dirichlet", "0", "1"]], method="DG", degree=2, basis="modal"),
    "Helmholtz scattering in a disk (P4)": dict(
        analysis="Steady", domain="Disk", radius=1.0, n=10, pde="Helmholtz", k=16.0,
        f="200*exp(-200*((x-0.3)^2+(y-0.2)^2))", u_exact="", ux="", uy="",
        bc_rows=[["all", "robin", "0", "0"]], method="CG", degree=4, solver="direct"),
    "Heat: orbiting source (Crank-Nicolson)": dict(
        analysis="Transient", domain="Square", width=1.0, n=24, structured=True, pde="Heat", Axx="0.01", Ayy="0.01",
        Axy="0", bx="0", by="0", c="0", m="1", f="20*exp(-80*((x-0.5-0.3*cos(2*pi*t))^2+(y-0.5-0.3*sin(2*pi*t))^2))",
        u0="0", u_exact="", ux="", uy="", bc_rows=[["all", "dirichlet", "0", "1"]], method="CG", degree=2, dt=0.01,
        nsteps=200, scheme="crank-nicolson"),
    "Drum modes of the L-shape (P3)": dict(
        analysis="Eigenmodes", domain="L-shape", width=1.0, n=16, structured=True, pde="Poisson", Axx="1", c="0",
        m="1", u_exact="", ux="", uy="", bc_rows=[["all", "dirichlet", "0", "1"]], method="CG", degree=3, n_modes=8),
    "Membrane modes of a flower (P3)": dict(
        analysis="Eigenmodes", domain="Parametric", parametric=DEFAULT_PARAMETRIC, n=20, pde="Poisson", Axx="1", c="0",
        m="1", u_exact="", ux="", uy="", bc_rows=[["all", "dirichlet", "0", "1"]], method="CG", degree=3, n_modes=8),
    "Cantilever beam vs Timoshenko (P3)": dict(
        analysis="Elasticity", domain="Rectangle", width=10.0, height=2.0, n=20, structured=True, E="1000", nu="0.3",
        plane="stress", degree=3,
        bc_rows=[["left", "fixed", f"{_TB[0]}, {_TB[1]}", "1"], ["right", "traction", f"0, {_TB[3]}", "1"],
                 ["top", "free", "0, 0", "1"], ["bottom", "free", "0, 0", "1"]]),
    "Plate with hole vs Kirsch (P3)": dict(
        analysis="Elasticity", domain="Plate with hole", width=4.0, height=4.0, hole_radius=0.5, hole_x="0", hole_y="0",
        n=20, E="1", nu="0.3", plane="stress", degree=3,
        bc_rows=[["left", "fixed_x", "0", "1"], ["bottom", "fixed_y", "0", "1"],
                 ["right", "traction", f"{_KIRSCH_SXX}, {_KIRSCH_SXY}", "1"],
                 ["top", "traction", f"{_KIRSCH_SXY}, {_KIRSCH_SYY}", "1"], ["hole", "free", "0, 0", "1"]]),
    "Bratu problem: -Δu = 6 e^u (Newton)": dict(
        analysis="Steady", domain="Square", width=1.0, n=16, structured=True, pde="Poisson", Axx="1", f="6*exp(u)",
        u_exact="", ux="", uy="", u0="0", bc_rows=[["all", "dirichlet", "0", "1"]], method="CG", degree=3),
    "Nonlinear diffusion κ(u) = 1 + 10u² (Picard)": dict(
        analysis="Steady", domain="Disk", radius=1.0, n=12, pde="Poisson", Axx="1+10*u^2",
        f="20*exp(-30*((x-0.3)^2+y^2))", u_exact="", ux="", uy="", u0="0",
        bc_rows=[["all", "dirichlet", "0", "1"]], method="CG", degree=2),
    "Nonlinear heat with u-dependent conductivity": dict(
        analysis="Transient", domain="Square", width=1.0, n=20, structured=True, pde="Heat", Axx="0.002*(1+30*u^2)",
        Ayy="0.002*(1+30*u^2)", Axy="0", bx="0", by="0", c="0", m="1",
        f="30*exp(-200*((x-0.5)^2+(y-0.5)^2))*(t<0.5)", u0="0", u_exact="", ux="", uy="",
        bc_rows=[["all", "dirichlet", "0", "1"]], method="CG", degree=2, dt=0.01, nsteps=150, scheme="backward-euler"),
    "Anisotropic diffusion on an annulus (P2)": dict(
        analysis="Steady", domain="Annulus", radius=1.0, inner_radius=0.4, n=16, pde="Diffusion", Axx="1", Axy="0.9",
        Ayy="1", f="0", u_exact="", ux="", uy="",
        bc_rows=[["inner_circle", "dirichlet", "1", "1"], ["outer_circle", "dirichlet", "0", "1"]], method="CG",
        degree=2),
}
KIRSCH = dict(sxx=_KIRSCH_SXX, syy=_KIRSCH_SYY, sxy=_KIRSCH_SXY)


def apply_preset(cfg: Config, name: str) -> Config:
    base = Config()
    d = base.to_dict()
    d.update(PRESETS[name])
    return Config.from_dict(d)
