from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from femtoolbox.core.estimate import error_norms, residual_estimator, smoothness_indicator, zz_estimator
from femtoolbox.core.mesh import Mesh2D, bisect, refine_uniform
from femtoolbox.core.pde import PDE, BoundaryCondition
from femtoolbox.core.solution import FEMSolution
from femtoolbox.core.solver import solve
from femtoolbox.core.space import FunctionSpace

_PHI = "mod(atan2(y-0.5,x-0.5)-pi/2, 2*pi)"
_TH = "atan2(y-0.5,x-0.5)"
_R13 = "((x-0.5)^2+(y-0.5)^2)^(-1/6)"
LSHAPE_EXACT = f"((x-0.5)^2+(y-0.5)^2)^(1/3)*sin(2/3*{_PHI})"
LSHAPE_UX = f"2/3*{_R13}*(sin(2/3*{_PHI})*cos({_TH}) - cos(2/3*{_PHI})*sin({_TH}))"
LSHAPE_UY = f"2/3*{_R13}*(sin(2/3*{_PHI})*sin({_TH}) + cos(2/3*{_PHI})*cos({_TH}))"


def mark_dorfler(eta: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """Minimal set M with sum_M eta^2 >= theta * sum eta^2."""
    e2 = np.asarray(eta, float) ** 2
    if e2.size == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(e2)[::-1]
    c = np.cumsum(e2[order])
    if c[-1] <= 0:
        return order[:1]
    k = int(np.searchsorted(c, theta * c[-1])) + 1
    return np.sort(order[:k])


def mark_maximum(eta: np.ndarray, theta: float = 0.5) -> np.ndarray:
    return np.flatnonzero(eta >= theta * eta.max())


@dataclass
class AdaptiveStep:
    level: int
    ndofs: int
    nelements: int
    eta: float
    error_l2: float | None = None
    error_h1: float | None = None
    p_min: int = 1
    p_max: int = 1
    n_h: int = 0
    n_p: int = 0
    time: float = 0.0


@dataclass
class AdaptiveResult:
    steps: list[AdaptiveStep]
    solution: FEMSolution
    indicators: np.ndarray
    meshes: list[Mesh2D] = field(default_factory=list)

    def table(self) -> str:
        s = f"{'lvl':>3} {'DOFs':>8} {'cells':>7} {'eta':>10} {'|e|_L2':>10} {'|e|_H1':>10} {'p':>5} {'h/p marks':>10}\n"
        for r in self.steps:
            l2 = f"{r.error_l2:.3e}" if r.error_l2 is not None else "-"
            h1 = f"{r.error_h1:.3e}" if r.error_h1 is not None else "-"
            s += f"{r.level:>3} {r.ndofs:>8} {r.nelements:>7} {r.eta:>10.3e} {l2:>10} {h1:>10} {r.p_min}-{r.p_max:<3} {r.n_h:>4}/{r.n_p:<4}\n"
        return s


def adaptive_solve(pde: PDE, mesh: Mesh2D, method: str = "CG", degree: int = 1, basis: str = "lagrange",
                   bcs: list[BoundaryCondition] | None = None, strategy: str = "h", theta: float = 0.5,
                   max_levels: int = 12, max_dofs: int = 50_000, estimator: str = "residual", u_exact=None,
                   ux=None, uy=None, p_max: int = 8, smooth_threshold: float = 1.0, solver: str = "auto",
                   supg: bool = False, callback=None) -> AdaptiveResult:
    """SOLVE -> ESTIMATE -> MARK (Dorfler) -> REFINE. strategy 'h' (NVB) or 'hp' (DG only: smooth cells get p+1)."""
    strategy = strategy.lower()
    if strategy == "hp" and method.upper() != "DG":
        raise ValueError("hp adaptivity requires DG (CG uses a uniform polynomial degree)")
    if pde.is_nonlinear:
        if method.upper() != "CG":
            raise ValueError("adaptivity for u-dependent coefficients uses the ZZ estimator and needs CG")
        estimator = "zz"
    degrees = np.full(mesh.nelements, int(degree))
    steps, meshes = [], []
    sol = eta = None
    for level in range(max_levels):
        t0 = time.perf_counter()
        space = FunctionSpace(mesh, method, degrees if method.upper() == "DG" else int(degree), basis)
        sol = solve(pde, mesh, space=space, bcs=bcs, solver=solver, supg=supg)
        eta = zz_estimator(sol) if estimator == "zz" else residual_estimator(sol, pde, bcs)
        row = AdaptiveStep(level, space.ndofs, mesh.nelements, float(np.sqrt((eta ** 2).sum())),
                           p_min=int(space.degrees.min()), p_max=int(space.degrees.max()))
        if u_exact not in (None, ""):
            e = error_norms(sol, u_exact, ux, uy)
            row.error_l2, row.error_h1 = e.l2, e.h1_semi
        meshes.append(mesh)
        steps.append(row)
        if callback is not None:
            callback(row, sol, eta)
        if level == max_levels - 1 or space.ndofs >= max_dofs:
            row.time = time.perf_counter() - t0
            break
        marked = mark_dorfler(eta, theta)
        if strategy == "hp":
            sig = smoothness_indicator(sol)
            smooth = (degrees[marked] >= 2) & (sig[marked] > smooth_threshold) & (degrees[marked] < p_max)
            low = degrees[marked] < 2
            p_up = marked[smooth | low]
            h_ref = marked[~(smooth | low)]
            degrees = degrees.copy()
            degrees[p_up] += 1
            row.n_p, row.n_h = len(p_up), len(h_ref)
            if len(h_ref):
                mesh, parent = bisect(mesh, h_ref)
                degrees = degrees[parent]
        else:
            mesh, parent = bisect(mesh, marked)
            row.n_h = len(marked)
            degrees = degrees[parent]
        row.time = time.perf_counter() - t0
    return AdaptiveResult(steps, sol, eta, meshes)


@dataclass
class ConvergenceRow:
    level: int
    h: float
    ndofs: int
    nelements: int
    l2: float
    h1: float
    l2_ratio: float | None
    h1_ratio: float | None
    l2_order: float | None
    h1_order: float | None
    eta: float | None = None
    time: float = 0.0


def convergence_study(pde: PDE, mesh: Mesh2D, u_exact, ux_exact=None, uy_exact=None, levels: int = 4,
                      method: str = "CG", degree: int = 1, basis: str = "lagrange", bcs=None, solver_name: str = "auto",
                      supg: bool = False) -> list[ConvergenceRow]:
    rows, prev, cur = [], None, mesh
    for level in range(max(1, int(levels))):
        t0 = time.perf_counter()
        sol = solve(pde, cur, method, degree, basis, bcs, solver_name, supg=supg)
        dt = time.perf_counter() - t0
        e = error_norms(sol, u_exact, ux_exact, uy_exact)
        eta = float(np.sqrt((residual_estimator(sol, pde, bcs) ** 2).sum()))
        h = float(cur.diameters.max())
        l2r = h1r = l2o = h1o = None
        if prev is not None:
            l2r, h1r = prev.l2 / max(e.l2, 1e-300), prev.h1 / max(e.h1_semi, 1e-300)
            hr = prev.h / h
            l2o, h1o = float(np.log(l2r) / np.log(hr)), float(np.log(h1r) / np.log(hr))
        row = ConvergenceRow(level, h, sol.info.ndofs, cur.nelements, e.l2, e.h1_semi, l2r, h1r, l2o, h1o, eta, dt)
        rows.append(row)
        prev = row
        if level < levels - 1:
            cur = refine_uniform(cur)[0]
    return rows


def refine_h_uniform(mesh: Mesh2D) -> Mesh2D:
    return refine_uniform(mesh)[0]


def refine_h_selected(mesh: Mesh2D, marked) -> Mesh2D:
    return bisect(mesh, marked)[0]


def estimate_error(solution: FEMSolution, pde: PDE | None = None, bcs=None) -> np.ndarray:
    if pde is not None:
        return residual_estimator(solution, pde, bcs)
    if solution.space.method == "CG":
        return zz_estimator(solution)
    from femtoolbox.core.pde import PDE as _P
    return residual_estimator(solution, _P.poisson(source=0.0), bcs)


def hp_decisions(solution: FEMSolution, indicators=None, threshold: float = 1.0) -> list[str]:
    sig = smoothness_indicator(solution)
    return ["p" if (np.isfinite(s) and s > threshold) or np.isnan(s) and solution.space.degrees[i] < 2 else "h"
            for i, s in enumerate(sig)]
