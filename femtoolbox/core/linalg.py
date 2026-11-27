from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import pyamg
except Exception:  # pragma: no cover
    pyamg = None

SOLVERS = ("auto", "direct", "cg-amg", "cg-ilu", "cg-jacobi", "gmres-ilu", "gmres-amg", "bicgstab-ilu")
_ALIASES = {"cg": "cg-amg", "gmres": "gmres-ilu", "bicgstab": "bicgstab-ilu", "spsolve": "direct", "lu": "direct"}
AUTO_DIRECT_LIMIT = 250_000


@dataclass
class LinearStats:
    solver: str = ""
    iterations: int = 0
    setup_time: float = 0.0
    solve_time: float = 0.0
    residual: float = 0.0
    converged: bool = True
    warnings: list = field(default_factory=list)


def canonical_solver(name: str) -> str:
    name = str(name or "auto").strip().lower()
    name = _ALIASES.get(name, name)
    if name not in SOLVERS:
        raise ValueError(f"unknown solver {name!r}; choose from {SOLVERS}")
    if name.endswith("amg") and pyamg is None:
        return "cg-ilu" if name.startswith("cg") else "gmres-ilu"
    return name


def is_symmetric(A: sp.spmatrix, tol: float = 1e-10) -> bool:
    d = A - A.T
    return d.nnz == 0 or abs(d).max() <= tol * max(abs(A).max(), 1e-300)


def pick_solver(A, name: str, symmetric: bool | None = None) -> str:
    name = canonical_solver(name)
    if name != "auto":
        return name
    if A.shape[0] <= AUTO_DIRECT_LIMIT:
        return "direct"
    sym = is_symmetric(A) if symmetric is None else symmetric
    if sym and pyamg is not None and np.all(A.diagonal() > 0):
        return "cg-amg"
    return "gmres-ilu"


class Factorization:
    """Reusable solver for a fixed matrix (direct LU or preconditioned Krylov)."""

    def __init__(self, A, solver: str = "auto", symmetric: bool | None = None, tol: float = 1e-10, maxiter: int = 2000):
        self.A = sp.csc_matrix(A) if not sp.isspmatrix_csr(A) else A
        self.name = pick_solver(A, solver, symmetric)
        self.tol, self.maxiter = tol, maxiter
        self.stats = LinearStats(solver=self.name)
        t0 = time.perf_counter()
        self.lu = self.M = None
        if self.name == "direct":
            sym = is_symmetric(A) if symmetric is None else symmetric
            Ac = sp.csc_matrix(A)
            self.lu = None
            if sym:
                try:
                    lu = spla.splu(Ac, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
                    z = np.random.default_rng(0).standard_normal(Ac.shape[0])
                    if np.linalg.norm(Ac @ lu.solve(z) - z) <= 1e-8 * np.linalg.norm(z) * max(1.0, abs(Ac).max()):
                        self.lu = lu
                except RuntimeError:
                    pass
            if self.lu is None:
                self.lu = spla.splu(Ac, permc_spec="MMD_AT_PLUS_A" if sym else "COLAMD")
        else:
            Acsr = sp.csr_matrix(A)
            if self.name.endswith("amg"):
                ml = pyamg.smoothed_aggregation_solver(Acsr, symmetry="symmetric" if self.name.startswith("cg") else "nonsymmetric")
                self.M = ml.aspreconditioner(cycle="V")
            elif self.name.endswith("ilu"):
                try:
                    ilu = spla.spilu(sp.csc_matrix(A), drop_tol=1e-4, fill_factor=10)
                    self.M = spla.LinearOperator(A.shape, ilu.solve)
                except RuntimeError:
                    self.stats.warnings.append("ILU breakdown; using Jacobi")
                    self.M = self._jacobi(A)
            else:
                self.M = self._jacobi(A)
            self.A = Acsr
        self.stats.setup_time = time.perf_counter() - t0

    @staticmethod
    def _jacobi(A):
        d = A.diagonal().copy()
        d[d == 0] = 1.0
        return sp.diags(1.0 / d)

    def solve(self, b: np.ndarray, x0=None) -> np.ndarray:
        t0 = time.perf_counter()
        it = [0]

        def cb(*_):
            it[0] += 1
        if self.lu is not None:
            x = self.lu.solve(np.asarray(b, float))
        else:
            kind = self.name.split("-")[0]
            bn = np.linalg.norm(b)
            if bn == 0:
                return np.zeros_like(b)
            if kind == "cg":
                x, code = spla.cg(self.A, b, x0=x0, rtol=self.tol, maxiter=self.maxiter, M=self.M, callback=cb)
            elif kind == "gmres":
                x, code = spla.gmres(self.A, b, x0=x0, rtol=self.tol, restart=100, maxiter=self.maxiter, M=self.M,
                                     callback=cb, callback_type="pr_norm")
            else:
                x, code = spla.bicgstab(self.A, b, x0=x0, rtol=self.tol, maxiter=self.maxiter, M=self.M, callback=cb)
            if code != 0:
                self.stats.converged = False
                self.stats.warnings.append(f"{self.name} did not converge (code {code}) after {it[0]} iterations")
        self.stats.iterations = it[0]
        self.stats.solve_time += time.perf_counter() - t0
        r = self.A @ x - b
        self.stats.residual = float(np.linalg.norm(r) / max(np.linalg.norm(b), 1e-300))
        return np.asarray(x, float)


def reduce_dirichlet(K: sp.csr_matrix, F: np.ndarray, ddofs: np.ndarray, dvals: np.ndarray):
    n = K.shape[0]
    free = np.ones(n, bool)
    free[ddofs] = False
    fidx = np.flatnonzero(free)
    g = np.zeros(n)
    g[ddofs] = dvals
    Kc = K.tocsc()
    rhs = F[fidx] - (Kc[:, ddofs] @ dvals)[fidx] if len(ddofs) else F[fidx]
    Kff = Kc[fidx][:, fidx].tocsr() if len(ddofs) else K.tocsr()
    return Kff, rhs, fidx, g


def solve_linear(K, F, ddofs=None, dvals=None, solver: str = "auto", symmetric: bool | None = None, tol: float = 1e-10):
    ddofs = np.zeros(0, dtype=np.int64) if ddofs is None else ddofs
    dvals = np.zeros(0) if dvals is None else dvals
    Kff, rhs, fidx, u = reduce_dirichlet(K, F, ddofs, dvals)
    fac = Factorization(Kff, solver, symmetric, tol)
    u[fidx] = fac.solve(rhs)
    return u, fac.stats
