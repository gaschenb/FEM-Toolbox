from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from femtoolbox.core.assembly import assemble, assemble_load_vector, cg_dirichlet
from femtoolbox.core.expr import Expr
from femtoolbox.core.linalg import Factorization, reduce_dirichlet
from femtoolbox.core.mesh import Mesh2D
from femtoolbox.core.pde import PDE, BoundaryCondition
from femtoolbox.core.solution import EigenSolution, FEMSolution, SolveInfo, TransientFEMSolution
from femtoolbox.core.space import FunctionSpace


def _space(mesh, method, degree, basis):
    return degree if isinstance(degree, FunctionSpace) else FunctionSpace(mesh, method, degree, basis)


def _info(space, K, stats, t_asm, extra_warn=()):
    return SolveInfo(space.method, space.degree, space.basis_name, space.ndofs, space.mesh.nelements, int(K.nnz),
                     stats.residual, list(stats.warnings) + list(extra_warn), stats.solver, stats.iterations, t_asm,
                     stats.setup_time, stats.solve_time)


def solve(pde: PDE, mesh: Mesh2D, method: str = "CG", degree=1, basis: str = "lagrange",
          bcs: list[BoundaryCondition] | None = None, solver: str = "auto", dg_penalty: float = 10.0,
          supg: bool = False, dg_theta: float = 1.0, space: FunctionSpace | None = None, tol: float = 1e-10,
          nonlinear_tol: float = 1e-8, max_iter: int = 40, relax: float = 1.0, u_init=None, callback=None,
          **_) -> FEMSolution:
    space = space or FunctionSpace(mesh, method, degree, basis)
    if pde.is_nonlinear:
        return solve_nonlinear(pde, mesh, space=space, bcs=bcs, solver=solver, dg_penalty=dg_penalty, supg=supg,
                               dg_theta=dg_theta, tol=nonlinear_tol, max_iter=max_iter, relax=relax, u_init=u_init,
                               lin_tol=tol, callback=callback)
    res = assemble(space, pde, bcs, dg_penalty=dg_penalty, supg=supg, dg_theta=dg_theta)
    sym = pde.is_symmetric and not supg and (space.method == "CG" or dg_theta == 1.0)
    Kff, rhs, fidx, u = reduce_dirichlet(res.K, res.F, res.dirichlet_dofs, res.dirichlet_values)
    fac = Factorization(Kff, solver, sym, tol)
    u[fidx] = fac.solve(rhs)
    warn = []
    if not pde.advection.is_zero and space.method == "CG" and not supg:
        warn.append("advection with plain Galerkin CG; enable SUPG or use DG if the solution oscillates")
    return FEMSolution(space, u, _info(space, res.K, fac.stats, res.timings["assembly"], warn))


def solve_nonlinear(pde: PDE, mesh: Mesh2D, method: str = "CG", degree=1, basis: str = "lagrange", bcs=None,
                    solver: str = "auto", dg_penalty: float = 10.0, supg: bool = False, dg_theta: float = 1.0,
                    space: FunctionSpace | None = None, tol: float = 1e-8, max_iter: int = 40, relax: float = 1.0,
                    u_init=None, lin_tol: float = 1e-10, callback=None) -> FEMSolution:
    """Coefficients may depend on u, ux, uy. Each iteration freezes A and b at u_k (Picard) and linearizes the
    u-dependence of c and f exactly (Newton), so source-driven problems such as Bratu converge quadratically."""
    space = space or FunctionSpace(mesh, method, degree, basis)
    u = np.zeros(space.ndofs)
    if u_init is not None:
        u = np.asarray(u_init, float).copy() if isinstance(u_init, np.ndarray) else interpolate_initial_condition(space, u_init)
    newton = not (pde.diffusion.uses_u or pde.advection.uses_u)
    hist, t_asm, conv = [], 0.0, False
    fac = None
    for it in range(1, int(max_iter) + 1):
        res = assemble(space, pde, bcs, dg_penalty=dg_penalty, supg=supg, dg_theta=dg_theta, u_state=u, newton=True)
        t_asm += res.timings["assembly"]
        Kff, rhs, fidx, un = reduce_dirichlet(res.K, res.F, res.dirichlet_dofs, res.dirichlet_values)
        r0 = float(np.linalg.norm(res.K[fidx] @ u - res.F[fidx])) if it > 1 else float("nan")
        sym = pde.is_symmetric and not supg and (space.method == "CG" or dg_theta == 1.0)
        fac = Factorization(Kff, solver, sym, lin_tol)
        un[fidx] = fac.solve(rhs)
        step = un - u
        u = u + relax * step
        rel = float(np.linalg.norm(step) / max(np.linalg.norm(u), 1e-300))
        hist.append((it, rel, r0))
        if callback is not None:
            callback(it, rel, r0, u)
        if not np.all(np.isfinite(u)):
            break
        if rel < tol:
            conv = True
            break
    warn = [] if conv else [f"nonlinear iteration did not converge in {it} iterations (last step {rel:.2e}); "
                            "try relaxation < 1, a better initial guess or continuation in the parameter"]
    info = _info(space, res.K, fac.stats, t_asm, warn)
    info.extra["nonlinear"] = (f"{'Newton' if newton else 'Picard'}: {len(hist)} iterations, "
                               f"last step {hist[-1][1]:.2e}{'' if conv else ' (not converged)'}")
    info.extra["nonlinear_history"] = hist
    return FEMSolution(space, u, info)


def interpolate_initial_condition(space: FunctionSpace, expression=0.0) -> np.ndarray:
    """Nodal interpolation for Lagrange spaces, L2 projection for modal DG."""
    f = Expr(expression)
    if space.basis_name == "lagrange":
        x = space.dof_coords
        return f(x[:, 0], x[:, 1], 0.0)
    from femtoolbox.core.assembly import assemble_volume
    tmp = PDE.linear_second_order(source=f.source, mass=1.0)
    _, M, F = assemble_volume(space, tmp, 0.0, want_K=False)
    return spla.spsolve(M.tocsc(), F)


def solve_transient(pde: PDE, mesh: Mesh2D, u0="0", dt: float = 0.01, nsteps: int = 50, scheme: str = "crank-nicolson",
                    method: str = "CG", degree=1, basis: str = "lagrange", bcs=None, solver: str = "auto",
                    dg_penalty: float = 10.0, theta: float | None = None, supg: bool = False, store_every: int = 1,
                    space: FunctionSpace | None = None) -> TransientFEMSolution:
    """m u_t + L u = f. Schemes: backward-euler, crank-nicolson, theta, rannacher (2 BE half steps + CN), bdf2.

    The system matrix is factorized once (constant dt); f(t) and Dirichlet g(t) are re-evaluated each step.
    """
    dt, nsteps = float(dt), int(nsteps)
    if dt <= 0 or nsteps < 1:
        raise ValueError("dt > 0 and nsteps >= 1 required")
    scheme = scheme.lower().replace("_", "-").replace(" ", "-")
    th = {"backward-euler": 1.0, "crank-nicolson": 0.5, "rannacher": 0.5, "bdf2": 1.0}.get(scheme, theta if theta is not None else 0.5)
    space = space or FunctionSpace(mesh, method, degree, basis)
    if pde.is_nonlinear:
        return _transient_lagged(pde, space, u0, dt, nsteps, scheme, th, bcs, solver, dg_penalty, supg, store_every)
    t0 = time.perf_counter()
    res = assemble(space, pde, bcs, dg_penalty=dg_penalty, supg=supg)
    K, M = res.K, res.M
    t_asm = time.perf_counter() - t0
    bcs_l = list(bcs) if bcs else [BoundaryCondition("dirichlet", 0.0, "all")]
    time_dep = pde.source.uses_t or any(f.uses_t for bc in bcs_l for f in bc.funcs)
    F_static = res.F

    def load(t):
        return assemble_load_vector(space, pde, bcs_l, t, dg_penalty, supg) if time_dep else F_static

    def dirichlet(t):
        if space.method != "CG":
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        return cg_dirichlet(space, bcs_l, t) if time_dep else (res.dirichlet_dofs, res.dirichlet_values)

    u = np.asarray(u0, float).copy() if isinstance(u0, np.ndarray) else interpolate_initial_condition(space, u0)
    ddofs, _ = dirichlet(0.0)
    free = np.ones(space.ndofs, bool)
    free[ddofs] = False
    fidx = np.flatnonzero(free)
    d0, g0 = dirichlet(0.0)
    u[d0] = g0
    facs: dict = {}
    sym = pde.is_symmetric and not supg

    def fac_for(a_m, a_k):
        key = (round(a_m, 14), round(a_k, 14))
        if key not in facs:
            A = (a_m * M + a_k * K).tocsr()
            facs[key] = (A, Factorization(A[fidx][:, fidx], solver, sym))
        return facs[key]

    def step(rhs_full, A, fac, t_new):
        dd, gg = dirichlet(t_new)
        un = np.zeros(space.ndofs)
        un[dd] = gg
        r = rhs_full - A @ un
        un[fidx] = fac.solve(r[fidx], x0=None)
        return un

    times, hist = [0.0], [u.copy()]
    warnings = []
    t = 0.0
    u_prev = None
    F_old = load(0.0)
    k = 0
    while k < nsteps:
        if scheme == "rannacher" and k == 0:
            for _ in range(2):
                A, fac = fac_for(1.0, 0.5 * dt)
                F_new = load(t + 0.5 * dt)
                u = step(M @ u + 0.5 * dt * F_new, A, fac, t + 0.5 * dt)
                t += 0.5 * dt
                F_old = F_new
            warnings.append("Rannacher start-up: two backward-Euler half steps")
        elif scheme == "bdf2" and u_prev is not None:
            A, fac = fac_for(1.5, dt)
            F_new = load(t + dt)
            un = step(M @ (2.0 * u - 0.5 * u_prev) + dt * F_new, A, fac, t + dt)
            u_prev, u = u, un
            t += dt
            F_old = F_new
        else:
            A, fac = fac_for(1.0, th * dt)
            F_new = load(t + dt)
            rhs = M @ u - (1 - th) * dt * (K @ u) + dt * ((1 - th) * F_old + th * F_new)
            un = step(rhs, A, fac, t + dt)
            u_prev, u = u, un
            t += dt
            F_old = F_new
        k += 1
        if k % max(1, store_every) == 0 or k == nsteps:
            times.append(t)
            hist.append(u.copy())
    stats = list(facs.values())[-1][1].stats
    info = _info(space, K, stats, t_asm, warnings)
    info.extra["scheme"] = scheme
    info.extra["factorizations"] = len(facs)
    return TransientFEMSolution(space, np.asarray(times), np.asarray(hist), info)


def _transient_lagged(pde, space, u0, dt, nsteps, scheme, th, bcs, solver, dg_penalty, supg, store_every):
    """u-dependent coefficients: linearly implicit theta scheme with coefficients frozen at u_n (one solve per step)."""
    bcs_l = list(bcs) if bcs else [BoundaryCondition("dirichlet", 0.0, "all")]
    u = np.asarray(u0, float).copy() if isinstance(u0, np.ndarray) else interpolate_initial_condition(space, u0)
    if scheme in ("bdf2", "rannacher"):
        th = 1.0 if scheme == "bdf2" else 0.5
    t, times, hist, t_asm = 0.0, [0.0], [u.copy()], 0.0
    res = None
    for k in range(1, nsteps + 1):
        t0 = time.perf_counter()
        res = assemble(space, pde, bcs_l, dg_penalty=dg_penalty, supg=supg, time=t + dt, u_state=u)
        F_old = assemble_load_vector(space, pde, bcs_l, t, dg_penalty, supg, u_state=u) if th < 1 else 0.0
        t_asm += time.perf_counter() - t0
        A = (res.M + th * dt * res.K).tocsr()
        rhs = res.M @ u - (1 - th) * dt * (res.K @ u) + dt * ((1 - th) * F_old + th * res.F)
        Aff, r, fidx, un = reduce_dirichlet(A, rhs, res.dirichlet_dofs, res.dirichlet_values)
        fac = Factorization(Aff, solver, pde.is_symmetric and not supg)
        un[fidx] = fac.solve(r)
        u, t = un, t + dt
        if k % max(1, store_every) == 0 or k == nsteps:
            times.append(t)
            hist.append(u.copy())
    info = _info(space, res.K, fac.stats, t_asm, ["u-dependent coefficients lagged one step (linearly implicit)"])
    info.extra["scheme"] = f"{scheme} (lagged coefficients)"
    return TransientFEMSolution(space, np.asarray(times), np.asarray(hist), info)


def solve_heat_theta(pde, mesh, u0, dt, nsteps, theta=0.5, method="CG", degree=1, basis="lagrange", bcs=None,
                     solver="auto", stabilization="theta", dg_penalty=10.0, return_history=False):
    scheme = {"backward-euler": "backward-euler", "crank-nicolson": "crank-nicolson", "rannacher": "rannacher",
              "bdf2": "bdf2"}.get(str(stabilization).lower(), "theta")
    sol = solve_transient(pde, mesh, u0, dt, nsteps, scheme, method, degree, basis, bcs, solver, dg_penalty, theta=theta)
    return sol if return_history else sol.final_solution()


def solve_eigen(pde: PDE, mesh: Mesh2D, k: int = 6, method: str = "CG", degree=1, basis: str = "lagrange", bcs=None,
                sigma: float = 0.0, dg_penalty: float = 10.0) -> EigenSolution:
    """Smallest eigenpairs of L u = lambda m u (shift-invert Lanczos about sigma)."""
    space = FunctionSpace(mesh, method, degree, basis)
    t0 = time.perf_counter()
    zero = PDE.linear_second_order(pde.diffusion, None, pde.reaction, 0.0, pde.mass)
    res = assemble(space, zero, bcs, dg_penalty=dg_penalty)
    t_asm = time.perf_counter() - t0
    Kff, _, fidx, _ = reduce_dirichlet(res.K, res.F, res.dirichlet_dofs, np.zeros(len(res.dirichlet_dofs)))
    Mff = res.M.tocsc()[fidx][:, fidx]
    Kff = 0.5 * (Kff + Kff.T)
    k = int(min(k, len(fidx) - 2))
    t1 = time.perf_counter()
    vals, vecs = spla.eigsh(Kff.tocsc(), k=k, M=Mff.tocsc(), sigma=sigma, which="LM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    modes = np.zeros((k, space.ndofs))
    modes[:, fidx] = vecs.T
    for i in range(k):
        j = np.argmax(np.abs(modes[i]))
        modes[i] *= np.sign(modes[i, j]) / max(np.abs(modes[i, j]), 1e-300)
    from femtoolbox.core.linalg import LinearStats
    st = LinearStats("eigsh shift-invert", 0, 0.0, time.perf_counter() - t1, 0.0)
    info = _info(space, res.K, st, t_asm)
    info.extra["eigenvalues"] = ", ".join(f"{v:.6g}" for v in vals)
    return EigenSolution(space, vals, modes, info)
