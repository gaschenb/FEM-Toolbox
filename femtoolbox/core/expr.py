from __future__ import annotations

import ast
from typing import Any, Callable

import numpy as np

_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "atan2": np.arctan2, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "exp": np.exp, "log": np.log,
    "log10": np.log10, "sqrt": np.sqrt, "abs": np.abs, "floor": np.floor, "ceil": np.ceil, "sign": np.sign,
    "min": np.minimum, "max": np.maximum, "where": np.where, "mod": np.mod, "pow": np.power,
    "heaviside": lambda z: np.heaviside(z, 0.5), "erf": None, "logical_not": np.logical_not,
}
try:
    from scipy.special import erf as _erf
    _FUNCS["erf"] = _erf
except Exception:
    _FUNCS.pop("erf")
_CONSTS = {"pi": np.pi, "e": np.e}
_VARS = ("x", "y", "t", "r", "theta", "theta_deg", "param_t", "u", "ux", "uy", "nx", "ny")
_FIELD_VARS = {"u", "ux", "uy", "nx", "ny"}
_FIELD: dict = {}


def set_field(**arrays):
    _FIELD.clear()
    _FIELD.update(arrays)


def clear_field():
    _FIELD.clear()


class field_state:
    """Make the current iterate (u, ux, uy at the evaluation points) visible to u-dependent coefficients."""

    def __init__(self, **arrays):
        self.arrays = arrays

    def __enter__(self):
        self._old = dict(_FIELD)
        _FIELD.clear()
        _FIELD.update(self.arrays)
        return self

    def __exit__(self, *exc):
        _FIELD.clear()
        _FIELD.update(self._old)
        return False
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.IfExp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd,
    ast.And, ast.Or, ast.Not, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.BitAnd, ast.BitOr,
)


class ExprError(ValueError):
    pass


class _Vectorize(ast.NodeTransformer):
    """Rewrite and/or/not/if-else into elementwise numpy operations."""

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        op = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        out = node.values[0]
        for v in node.values[1:]:
            out = ast.BinOp(left=out, op=op, right=v)
        return out

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.Call(func=ast.Name(id="logical_not", ctx=ast.Load()), args=[node.operand], keywords=[])
        return node

    def visit_IfExp(self, node):
        self.generic_visit(node)
        return ast.Call(func=ast.Name(id="where", ctx=ast.Load()), args=[node.test, node.body, node.orelse], keywords=[])


def _check(tree: ast.AST, text: str):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExprError(f"unsupported syntax {type(node).__name__} in {text!r}")
        if isinstance(node, ast.Name) and node.id not in _FUNCS and node.id not in _CONSTS and node.id not in _VARS:
            raise ExprError(f"unknown name {node.id!r} in {text!r}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
            raise ExprError(f"unknown function in {text!r}")


class Expr:
    """Vectorized f(x, y, t) from a number, a callable or a safe expression string."""

    def __init__(self, value: Any, default: float = 0.0):
        if isinstance(value, Expr):
            self.__dict__.update(value.__dict__)
            return
        if value is None or (isinstance(value, str) and not value.strip()):
            value = default
        self.source = value
        self.const: float | None = None
        self._fn: Callable | None = None
        self._code = None
        self.uses_t = False
        self.uses_xy = False
        self.uses_u = False
        if callable(value):
            self._fn = value
            self.uses_t = self.uses_xy = True
        elif isinstance(value, str):
            text = value.strip().replace("^", "**")
            try:
                tree = ast.parse(text, mode="eval")
            except SyntaxError as exc:
                raise ExprError(f"cannot parse {value!r}: {exc.msg}") from None
            _check(tree, value)
            tree = ast.fix_missing_locations(_Vectorize().visit(tree))
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            self.uses_t = "t" in names
            self.uses_xy = bool(names & {"x", "y", "r", "theta"})
            self.uses_u = bool(names & _FIELD_VARS)
            self._code = compile(tree, "<expr>", "eval")
            if not self.uses_t and not self.uses_xy and not self.uses_u:
                self.const = float(eval(self._code, {"__builtins__": {}}, dict(_FUNCS, **_CONSTS)))
        else:
            self.const = float(value)

    @property
    def is_constant(self) -> bool:
        return self.const is not None

    def __call__(self, x, y, t: float = 0.0, extra: dict | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        tt = np.asarray(t, dtype=float)
        shape = np.broadcast(x, y, tt).shape
        if self.const is not None:
            return np.full(shape, self.const)
        if self._fn is not None:
            try:
                out = self._fn(x, y, t)
            except TypeError:
                out = self._fn(x, y)
            out = np.asarray(out, dtype=float)
            if out.shape != shape:
                try:
                    out = np.broadcast_to(out, shape).copy()
                except ValueError:
                    out = np.vectorize(lambda a, b: float(self._fn(a, b)))(x, y)
            return out
        scope = dict(_FUNCS, **_CONSTS)
        scope.update(x=x, y=y, t=tt if tt.ndim else float(tt), r=np.hypot(x, y), theta=np.arctan2(y, x))
        if self.uses_u:
            if not _FIELD:
                raise ExprError(f"{self.source!r} uses u, ux or uy, which are only available in PDE coefficients")
            scope.update(_FIELD)
            shape = np.broadcast_shapes(shape, *(np.shape(v) for v in _FIELD.values()))
        if extra:
            scope.update(extra)
        with np.errstate(all="ignore"):
            out = eval(self._code, {"__builtins__": {}}, scope)
        return np.broadcast_to(np.asarray(out, dtype=float), shape).copy()

    def __repr__(self):
        return f"Expr({self.source!r})"


class MatrixExpr:
    def __init__(self, value: Any = None):
        if isinstance(value, MatrixExpr):
            self.__dict__.update(value.__dict__)
            return
        self._fn = None
        self.entries = None
        if callable(value):
            self._fn = value
            self.entries = None
            self.is_constant = False
            return
        if value is None:
            value = [[1.0, 0.0], [0.0, 1.0]]
        if np.isscalar(value) or isinstance(value, str):
            value = [[value, 0.0], [0.0, value]]
        arr = np.asarray(value, dtype=object).reshape(2, 2)
        self.entries = [[Expr(arr[i, j]) for j in range(2)] for i in range(2)]
        self.is_constant = all(e.is_constant for row in self.entries for e in row)

    @property
    def uses_u(self) -> bool:
        return bool(self.entries) and any(e.uses_u for row in self.entries for e in row)

    def __call__(self, x, y, t: float = 0.0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self._fn is not None:
            out = np.asarray(self._fn(x, y), dtype=float)
            if out.shape == (2, 2):
                return np.broadcast_to(out, x.shape + (2, 2)).copy()
            if out.shape[:2] == (2, 2):
                return np.moveaxis(out, (0, 1), (-2, -1))
            return out
        out = np.empty(np.broadcast(x, y).shape + (2, 2))
        for i in range(2):
            for j in range(2):
                out[..., i, j] = self.entries[i][j](x, y, t)
        return out

    def max_eig(self, x, y) -> np.ndarray:
        A = self(x, y)
        S = 0.5 * (A + np.swapaxes(A, -1, -2))
        return np.linalg.eigvalsh(S)[..., -1]


class VectorExpr:
    def __init__(self, value: Any = (0.0, 0.0)):
        if isinstance(value, VectorExpr):
            self.__dict__.update(value.__dict__)
            return
        self._fn = None
        self.entries = None
        if callable(value):
            self._fn = value
            self.is_constant = False
            self.is_zero = False
            return
        if value is None:
            value = (0.0, 0.0)
        self.entries = [Expr(v) for v in list(value)[:2]]
        self.is_constant = all(e.is_constant for e in self.entries)
        self.is_zero = self.is_constant and all(e.const == 0.0 for e in self.entries)

    @property
    def uses_u(self) -> bool:
        return bool(getattr(self, "entries", None)) and any(e.uses_u for e in self.entries)

    def __call__(self, x, y, t: float = 0.0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self._fn is not None:
            out = np.asarray(self._fn(x, y), dtype=float)
            if out.shape == (2,):
                return np.broadcast_to(out, x.shape + (2,)).copy()
            if out.shape[0] == 2 and out.shape[1:] == x.shape:
                return np.moveaxis(out, 0, -1)
            return out
        return np.stack([e(x, y, t) for e in self.entries], axis=-1)


def numeric(text: Any) -> float:
    return float(Expr(text)(0.0, 0.0))
