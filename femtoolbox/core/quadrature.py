from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import roots_jacobi, roots_legendre


@lru_cache(maxsize=None)
def triangle_rule(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Collapsed Gauss-Jacobi (Stroud) rule on the reference triangle, exact to `degree`. Weights sum to 1/2."""
    n = max(1, (int(degree) + 2) // 2)
    gu, wu = roots_legendre(n)
    gv, wv = roots_jacobi(n, 1.0, 0.0)
    u, wu = 0.5 * (gu + 1.0), 0.5 * wu
    v, wv = 0.5 * (gv + 1.0), 0.25 * wv
    U, V = np.meshgrid(u, v, indexing="ij")
    W = np.outer(wu, wv)
    pts = np.column_stack(((U * (1.0 - V)).ravel(), V.ravel()))
    return pts, W.ravel()


@lru_cache(maxsize=None)
def line_rule(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre on [0, 1]."""
    n = max(1, (int(degree) + 2) // 2)
    g, w = roots_legendre(n)
    return 0.5 * (g + 1.0), 0.5 * w
