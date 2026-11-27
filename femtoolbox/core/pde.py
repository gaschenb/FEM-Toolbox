from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from femtoolbox.core.expr import Expr, MatrixExpr, VectorExpr, numeric

SCALAR_KINDS = ("dirichlet", "neumann", "robin")
VECTOR_KINDS = ("fixed", "fixed_x", "fixed_y", "traction", "pressure", "free")


def split_top_level(text: str, sep: str = ",") -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in str(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [s.strip() for s in out]


@dataclass
class BoundaryCondition:
    """Boundary data on a mesh marker or on a geometric selector.

    Scalar kinds: dirichlet (u=g), neumann (A grad u . n = g), robin (A grad u . n + alpha u = g).
    Elasticity kinds: fixed (u=(gx,gy)), fixed_x, fixed_y, traction (t=(tx,ty)), pressure (t=-p n), free.
    Selectors: ``left``, ``marker:left``, ``segment:(x0,y0)->(x1,y1)``, ``arc:45deg->135deg[;cx,cy,r]``,
    ``t:0->pi/2`` (intrinsic boundary coordinate), ``where:x < 0 and y > 0``.
    Priority: geometric selector (last wins) > exact marker > all/outer fallback.
    """

    kind: str = "dirichlet"
    value: Any = 0.0
    marker: str = "outer"
    alpha: float = 1.0
    label: str | None = None
    selector: str = field(init=False, default="marker")
    params: tuple = field(init=False, default=())
    where_expr: Expr | None = field(init=False, default=None)

    def __post_init__(self):
        self.kind = str(self.kind).strip().lower().replace("-", "_").replace(" ", "_")
        if self.kind not in SCALAR_KINDS + VECTOR_KINDS:
            raise ValueError(f"unknown boundary condition kind {self.kind!r}")
        self.alpha = float(numeric(self.alpha)) if isinstance(self.alpha, str) else float(self.alpha)
        if self.kind in ("fixed", "traction"):
            v = self.value
            parts = list(v) if isinstance(v, (tuple, list)) else split_top_level(v) if isinstance(v, str) else [v, v]
            if len(parts) == 1:
                parts = [parts[0], parts[0]] if self.kind == "fixed" else [parts[0], 0.0]
            self.funcs = [Expr(parts[0]), Expr(parts[1])]
        else:
            self.funcs = [Expr(self.value)]
        self._parse_selector("all" if self.marker is None else str(self.marker).strip())

    @property
    def value_function(self):
        return self.funcs[0]

    def value_at(self, x, y, t: float = 0.0):
        return self.funcs[0](x, y, t)

    def _parse_selector(self, text: str):
        low = text.lower()
        self.selector = "marker"
        if low.startswith("marker:"):
            self.marker = low.split(":", 1)[1].strip() or "all"
        elif low.startswith("segment:"):
            self.selector, self.marker = "segment", low
            nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.split(":", 1)[1])]
            if len(nums) < 4:
                raise ValueError(f"segment selector needs four numbers: {text!r}")
            self.params = tuple(nums[:4])
        elif low.startswith("arc:"):
            self.selector, self.marker = "arc", low
            self.params = _parse_arc(text.split(":", 1)[1])
        elif low.split(":", 1)[0] in ("t", "param", "parameter") and ":" in low:
            self.selector, self.marker = "t", low
            self.params = _parse_range(text.split(":", 1)[1])
        elif low.startswith("where:") or low.startswith("expr:"):
            self.selector, self.marker = "where", low
            expr = text.split(":", 1)[1].strip()
            if not expr:
                raise ValueError("where: selector needs a boolean expression")
            self.where_expr = Expr(expr)
        else:
            self.marker = low or "all"

    @property
    def is_global_fallback(self) -> bool:
        return self.selector == "marker" and self.marker in ("all", "outer")

    @property
    def is_geometric_selector(self) -> bool:
        return self.selector != "marker"

    def matches(self, markers: np.ndarray, pa: np.ndarray, pb: np.ndarray, coord: np.ndarray | None, period=0.0, t0=0.0):
        """Vectorized facet test; pa, pb: (n,2) facet endpoints."""
        n = len(markers)
        if self.selector == "marker":
            if self.is_global_fallback:
                return np.ones(n, bool)
            return np.asarray(markers == self.marker, bool)
        mid = 0.5 * (pa + pb)
        flen = np.maximum(np.linalg.norm(pb - pa, axis=1), 1e-14)
        if self.selector == "segment":
            x0, y0, x1, y1 = self.params
            s0, s1 = np.array([x0, y0]), np.array([x1, y1])
            d = s1 - s0
            L = max(np.linalg.norm(d), 1e-14)
            s = ((mid - s0) @ d) / (L * L)
            q = s0 + np.clip(s, 0, 1)[:, None] * d
            dist = np.linalg.norm(mid - q, axis=1)
            tol = np.maximum(1e-9, 0.05 * flen)
            return (dist <= tol) & (s >= -1e-9) & (s <= 1 + 1e-9)
        if self.selector == "arc":
            th0, th1, cx, cy, R = self.params
            dx, dy = mid[:, 0] - cx, mid[:, 1] - cy
            th = np.degrees(np.arctan2(dy, dx)) % 360.0
            ok = _in_interval(th, th0, th1, 360.0, 0.0)
            if R > 0:
                ok &= np.abs(np.hypot(dx, dy) - R) <= np.maximum(1e-8, 0.75 * flen)
            return ok
        if self.selector == "t":
            if coord is None:
                return np.zeros(n, bool)
            return _in_interval(coord, self.params[0], self.params[1], period, t0)
        if self.selector == "where":
            th = np.degrees(np.arctan2(mid[:, 1], mid[:, 0])) % 360.0
            tt = coord if coord is not None else th
            v = self.where_expr(mid[:, 0], mid[:, 1], 0.0,
                                extra=dict(theta=th, theta_deg=th, t=tt, param_t=tt))
            return np.asarray(v, bool)
        return np.zeros(n, bool)

    def matches_facet(self, marker: str, pa, pb, coordinate_data: dict | None = None) -> bool:
        coord = None
        if coordinate_data and "t" in coordinate_data:
            coord = np.array([float(coordinate_data["t"])])
        return bool(self.matches(np.array([str(marker).lower()], dtype=object), np.atleast_2d(pa), np.atleast_2d(pb),
                                 coord, float((coordinate_data or {}).get("period", 0.0)),
                                 float((coordinate_data or {}).get("t0", 0.0)))[0])

    def target_description(self) -> str:
        if self.label:
            return self.label
        if self.selector == "segment":
            return "segment ({}, {}) -> ({}, {})".format(*[f"{v:g}" for v in self.params])
        if self.selector == "arc":
            th0, th1, cx, cy, R = self.params
            return f"arc theta in [{th0:g}, {th1:g}] deg about ({cx:g}, {cy:g})" + (f", r = {R:g}" if R > 0 else "")
        if self.selector == "t":
            return f"boundary parameter t in [{self.params[0]:g}, {self.params[1]:g}]"
        if self.selector == "where":
            return f"where {self.where_expr.source}"
        return self.marker


def _in_interval(v, a, b, period, origin):
    v = np.asarray(v, float)
    if period and period > 0:
        norm = lambda z: np.mod(np.asarray(z, float) - origin, period) + origin
        vv, lo, hi = norm(v), float(norm(a)), float(norm(b))
        if abs((hi - lo) % period) < 1e-12 and abs(b - a) > 1e-12:
            return np.ones(v.shape, bool)
        return (vv >= lo) & (vv <= hi) if lo <= hi else (vv >= lo) | (vv <= hi)
    return (v >= min(a, b)) & (v <= max(a, b))


def _split_range(text: str):
    main = str(text).split(";", 1)[0].strip()
    if "->" in main:
        return main.split("->", 1)
    if "," in main:
        return main.split(",", 1)
    parts = main.split()
    if len(parts) < 2:
        raise ValueError(f"range needs two endpoints: {text!r}")
    return parts[:2]


def _parse_range(text: str):
    a, b = _split_range(text)
    return numeric(a), numeric(b)


def _angle_deg(text: str) -> float:
    s = text.strip().lower().replace("°", "deg")
    deg = "deg" in s
    rad = "rad" in s or "pi" in s
    s = re.sub(r"degrees|degree|deg|radians|radian|rad", "", s).strip()
    v = numeric(s)
    return float(np.degrees(v)) if rad and not deg else float(v)


def _parse_arc(text: str):
    a, b = _split_range(text)
    rest = text.split(";", 1)[1] if ";" in text else ""
    nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", rest)]
    cx, cy = (nums[0], nums[1]) if len(nums) >= 2 else (0.0, 0.0)
    R = nums[2] if len(nums) >= 3 else -1.0
    return _angle_deg(a) % 360.0, _angle_deg(b) % 360.0, cx, cy, R


def parse_boundary_conditions(text: str) -> list[BoundaryCondition]:
    """Lines ``name | kind | value | selector | alpha`` or ``kind=..; value=..; selector=..; alpha=..``."""
    bcs = []
    for raw in str(text).splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4:
            name, kind, value, sel = parts[:4]
            alpha = parts[4] if len(parts) >= 5 and parts[4] else 1.0
            bcs.append(BoundaryCondition(kind, value, sel, alpha, name or None))
            continue
        data = {}
        for chunk in line.split(";"):
            if chunk.strip():
                if "=" not in chunk:
                    raise ValueError(f"cannot parse boundary condition line {raw!r}")
                k, v = chunk.split("=", 1)
                data[k.strip().lower()] = v.strip()
        if data:
            bcs.append(BoundaryCondition(data.get("kind", "dirichlet"), data.get("value", data.get("g", "0")),
                                         data.get("selector", data.get("marker", "all")), data.get("alpha", 1.0),
                                         data.get("name")))
    return bcs


def assign_boundary_conditions(mesh, bcs: list[BoundaryCondition]) -> np.ndarray:
    """Index into ``bcs`` for each boundary edge (aligned with mesh.boundary_edge_ids); -1 if none."""
    eids = mesh.boundary_edge_ids
    markers = mesh.edge_markers
    pa = mesh.nodes[mesh.edges[eids, 0]]
    pb = mesh.nodes[mesh.edges[eids, 1]]
    coord = None
    period, t0 = 0.0, 0.0
    if any(bc.selector in ("t", "where") for bc in bcs):
        coord = mesh.boundary_coordinate(0.5 * (pa + pb))
        if mesh.boundary is not None:
            period, t0 = mesh.boundary.period, mesh.boundary.t0
    n = len(eids)
    geo = np.full(n, -1)
    exact = np.full(n, -1)
    fallback = np.full(n, -1)
    for i, bc in enumerate(bcs):
        m = bc.matches(markers, pa, pb, coord, period, t0)
        if bc.is_geometric_selector:
            geo[m] = i
        elif bc.is_global_fallback:
            fallback[m & (fallback < 0)] = i
        else:
            exact[m] = i
    return np.where(geo >= 0, geo, np.where(exact >= 0, exact, fallback))


# ----------------------------------------------------------------------------- PDE
@dataclass
class PDE:
    """-div(A grad u) + b.grad u + c u = f  (with m u_t for transient problems)."""

    name: str
    diffusion: MatrixExpr
    advection: VectorExpr
    reaction: Expr
    source: Expr
    mass: Expr
    wavenumber: float = 0.0
    conservative: bool = False
    latex: str = r"-\nabla\cdot(A\nabla u)+\mathbf{b}\cdot\nabla u+c u=f"
    description: str = "Linear scalar second-order PDE."
    exact: Expr | None = None
    exact_grad: tuple[Expr, Expr] | None = None

    def source_value(self, x, y, t: float = 0.0):
        return self.source(x, y, t)

    @property
    def is_symmetric(self) -> bool:
        if not self.advection.is_zero:
            return False
        if self.diffusion.is_constant:
            A = self.diffusion(0.0, 0.0)
            return bool(np.allclose(A, A.T))
        e = self.diffusion.entries
        if e is not None:
            a, b = e[0][1], e[1][0]
            if a.source == b.source:
                return True
            if not (a.uses_u or b.uses_u):
                rng = np.random.default_rng(0)
                x, y = rng.uniform(-3, 3, (2, 64))
                return bool(np.allclose(a(x, y), b(x, y)))
        return False

    @property
    def is_nonlinear(self) -> bool:
        return any(c.uses_u for c in (self.diffusion, self.advection, self.reaction, self.source, self.mass))

    @classmethod
    def linear_second_order(cls, diffusion=None, advection=None, reaction=0.0, source=1.0, mass=1.0,
                            name="Linear second-order PDE", conservative=False, latex=None, description="",
                            exact=None, exact_grad=None) -> "PDE":
        ex = Expr(exact) if exact not in (None, "") else None
        eg = (Expr(exact_grad[0]), Expr(exact_grad[1])) if exact_grad and all(g not in (None, "") for g in exact_grad) else None
        return cls(name, MatrixExpr(diffusion), VectorExpr(advection), Expr(reaction), Expr(source, 1.0), Expr(mass, 1.0),
                   conservative=conservative, latex=latex or r"-\nabla\cdot(A\nabla u)+\mathbf{b}\cdot\nabla u+c u=f",
                   description=description or "Linear scalar second-order PDE.", exact=ex, exact_grad=eg)

    @classmethod
    def poisson(cls, kappa=1.0, source=1.0, **kw) -> "PDE":
        return cls.linear_second_order([[kappa, 0], [0, kappa]], source=source, name="Poisson",
                                       latex=r"-\nabla\cdot(\kappa\nabla u)=f", description="Diffusion-only elliptic model.", **kw)

    @classmethod
    def diffusion_pde(cls, A=None, source=1.0, **kw) -> "PDE":
        return cls.linear_second_order(A, source=source, name="Diffusion", latex=r"-\nabla\cdot(A\nabla u)=f",
                                       description="Anisotropic diffusion with tensor A(x,y).", **kw)

    @classmethod
    def reaction_diffusion(cls, A=None, c=1.0, source=1.0, **kw) -> "PDE":
        return cls.linear_second_order(A, reaction=c, source=source, name="Reaction-diffusion",
                                       latex=r"-\nabla\cdot(A\nabla u)+c u=f", **kw)

    @classmethod
    def advection_diffusion(cls, A=None, b=(1.0, 0.0), c=0.0, source=1.0, **kw) -> "PDE":
        return cls.linear_second_order(A if A is not None else [[1e-2, 0], [0, 1e-2]], b, c, source,
                                       name="Advection-diffusion", conservative=True,
                                       latex=r"-\nabla\cdot(A\nabla u)+\mathbf{b}\cdot\nabla u+c u=f",
                                       description="Transport-diffusion; use DG upwind or CG+SUPG at high Peclet.", **kw)

    @classmethod
    def helmholtz(cls, k: float = 8.0, source=1.0, **kw) -> "PDE":
        p = cls.linear_second_order(None, reaction=-(float(k) ** 2), source=source, name="Helmholtz",
                                    latex=r"-\Delta u-k^2 u=f", description="Indefinite time-harmonic wave model.", **kw)
        p.wavenumber = float(k)
        return p

    @classmethod
    def heat(cls, kappa=1.0, source=1.0, mass=1.0, reaction=0.0, **kw) -> "PDE":
        return cls.linear_second_order([[kappa, 0], [0, kappa]], None, reaction, source, mass, name="Heat",
                                       latex=r"m\,\partial_t u-\nabla\cdot(\kappa\nabla u)+c u=f", **kw)

    heat_static_step = heat

    @classmethod
    def custom_from_definition(cls, text: str) -> "PDE":
        """key = value block: name, Axx, Axy, Ayx, Ayy, bx, by, c, f, mass, u_exact, ux, uy, latex, conservative."""
        d = _parse_block(text)
        A = [[d.get("axx", "1"), d.get("axy", "0")], [d.get("ayx", d.get("axy", "0")), d.get("ayy", "1")]]
        return cls.linear_second_order(
            A, (d.get("bx", "0"), d.get("by", "0")), d.get("c", d.get("reaction", "0")), d.get("f", d.get("source", "1")),
            d.get("mass", "1"), name=d.get("name", "Custom PDE"),
            conservative=d.get("conservative", "false").lower() in ("1", "true", "yes"),
            latex=d.get("latex"), description="User-defined coefficient-form PDE.",
            exact=d.get("u_exact", d.get("exact")), exact_grad=(d.get("ux"), d.get("uy")))


def _parse_block(text: str) -> dict[str, str]:
    out = {}
    for raw in str(text).splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        sep = "=" if "=" in line else ":" if ":" in line else None
        if sep is None:
            raise ValueError(f"expected key = value: {raw!r}")
        k, v = line.split(sep, 1)
        out[k.strip().lower()] = v.strip()
    return out


PDE_FORMULAS = {
    "Poisson": (r"-\nabla\cdot(\kappa\nabla u)=f", "kappa scalar, f(x,y)."),
    "Diffusion": (r"-\nabla\cdot(A\nabla u)=f", "Tensor A from Axx, Axy, Ayy."),
    "Reaction-diffusion": (r"-\nabla\cdot(A\nabla u)+c u=f", "Adds reaction c(x,y)."),
    "Advection-diffusion": (r"-\nabla\cdot(A\nabla u)+\mathbf{b}\cdot\nabla u+c u=f", "Adds transport b=(bx,by)."),
    "Helmholtz": (r"-\Delta u-k^2 u=f", "Indefinite; direct solver recommended."),
    "Heat": (r"m\,\partial_t u-\nabla\cdot(\kappa\nabla u)+c u=f", "Transient: theta or BDF2 stepping; f may depend on t."),
    "Custom PDE": (r"-\nabla\cdot(A\nabla u)+\mathbf{b}\cdot\nabla u+c u=f", "Coefficient-form block; optional u_exact."),
    "Eigenproblem": (r"-\nabla\cdot(A\nabla u)+c u=\lambda\, m u", "Smallest eigenpairs by shift-invert Lanczos."),
    "Linear elasticity": (r"-\nabla\cdot\sigma(\mathbf{u})=\mathbf{f},\ \sigma=\lambda\,\mathrm{tr}\,\varepsilon\,I+2\mu\varepsilon", "Plane stress or plane strain."),
}
