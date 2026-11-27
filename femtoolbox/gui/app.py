from __future__ import annotations

import json
import sys
import traceback

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.tri as mtri
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from femtoolbox.core.advisor import SEVERITIES, recommend
from femtoolbox.core.elasticity import Elasticity
from femtoolbox.core.meshio import read_msh, write_msh
from femtoolbox.core.postprocess import (boundary_integral, integrate, integration_weights, point_series,
                                         quantities)
from femtoolbox.core.estimate import estimate_error, residual_estimator, smoothness_indicator
from femtoolbox.core.export import export_all_csv, export_vtu
from femtoolbox.core.mesh import refine_uniform, smooth_laplacian
from femtoolbox.core.pde import PDE_FORMULAS
from femtoolbox.core.solution import EigenSolution, FEMSolution, TransientFEMSolution
from femtoolbox.gui import model as M

SCALAR_FIELDS = ["u", "|grad u|", "du/dx", "du/dy", "error indicator", "|u - u_exact|", "polynomial degree",
                 "smoothness (hp)", "mesh quality"]
ELASTIC_FIELDS = ["von Mises", "|u|", "u_x", "u_y", "sigma_xx", "sigma_yy", "sigma_xy"]
_EL_KEY = {"von Mises": "vm", "|u|": "umag", "u_x": "ux", "u_y": "uy", "sigma_xx": "sxx", "sigma_yy": "syy",
           "sigma_xy": "sxy"}
CMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "coolwarm", "RdBu_r", "Spectral_r", "gray"]


# ----------------------------------------------------------------------------- theme
THEME = {"dark": False}
_LIGHT_RC = {"figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff", "axes.edgecolor": "#333333",
             "axes.labelcolor": "#191919", "text.color": "#191919", "xtick.color": "#333333", "ytick.color": "#333333",
             "grid.color": "#b0b0b0", "legend.facecolor": "#ffffff", "legend.edgecolor": "#cccccc"}
_DARK_RC = {"figure.facecolor": "#202124", "axes.facecolor": "#202124", "axes.edgecolor": "#9aa0a6",
            "axes.labelcolor": "#e8eaed", "text.color": "#e8eaed", "xtick.color": "#bdc1c6", "ytick.color": "#bdc1c6",
            "grid.color": "#5f6368", "legend.facecolor": "#2b2c2f", "legend.edgecolor": "#5f6368"}


def ink() -> str:
    return "#e8eaed" if THEME["dark"] else "k"


def apply_theme(app: QtWidgets.QApplication, dark: bool):
    THEME["dark"] = bool(dark)
    matplotlib.rcParams.update(_DARK_RC if dark else _LIGHT_RC)
    app.setStyle("Fusion")
    if not dark:
        app.setPalette(app.style().standardPalette())
        app.setStyleSheet("")
        return
    pal = QtGui.QPalette()
    c = QtGui.QColor
    for role, col in ((QtGui.QPalette.Window, c(37, 38, 41)), (QtGui.QPalette.WindowText, c(232, 234, 237)),
                      (QtGui.QPalette.Base, c(28, 29, 31)), (QtGui.QPalette.AlternateBase, c(44, 45, 48)),
                      (QtGui.QPalette.ToolTipBase, c(48, 49, 52)), (QtGui.QPalette.ToolTipText, c(232, 234, 237)),
                      (QtGui.QPalette.Text, c(232, 234, 237)), (QtGui.QPalette.Button, c(48, 49, 52)),
                      (QtGui.QPalette.ButtonText, c(232, 234, 237)), (QtGui.QPalette.BrightText, c(255, 120, 120)),
                      (QtGui.QPalette.Highlight, c(37, 125, 224)), (QtGui.QPalette.HighlightedText, c(255, 255, 255)),
                      (QtGui.QPalette.Link, c(120, 170, 255))):
        pal.setColor(role, col)
    for role in (QtGui.QPalette.Text, QtGui.QPalette.ButtonText, QtGui.QPalette.WindowText):
        pal.setColor(QtGui.QPalette.Disabled, role, c(120, 122, 126))
    app.setPalette(pal)
    app.setStyleSheet(_DARK_QSS)


_DARK_QSS = """
QCheckBox::indicator, QGroupBox::indicator { width: 12px; height: 12px; border: 1px solid #9aa0a6;
    border-radius: 2px; background: #1c1d1f; }
QCheckBox::indicator:checked, QGroupBox::indicator:checked { background: #3d8bfd; border-color: #3d8bfd; }
QCheckBox::indicator:disabled { border-color: #55585c; }
"""


# ----------------------------------------------------------------------------- widgets
class Canvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, three_d=False):
        self.fig = Figure(figsize=(6, 5), dpi=100, layout="constrained")
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.cax = None

    def clear(self):
        self.fig.clear()
        self.fig.set_facecolor(matplotlib.rcParams["figure.facecolor"])

    def draw(self):
        super().draw()
        for ax in self.fig.axes:
            if getattr(ax, "_home", None) is None and ax.get_navigate() and ax.name != "3d":
                ax._home = (ax.get_xlim(), ax.get_ylim())
                ax.callbacks.connect("xlim_changed", self._clamp)
                ax.callbacks.connect("ylim_changed", self._clamp)

    @staticmethod
    def _clamp(ax):
        """Keep the view inside the plotted extent: no zooming or panning out past the data."""
        if getattr(ax, "_clamping", False):
            return
        ax._clamping = True
        try:
            for axis, lim, setter in ((ax.xaxis, ax._home[0], ax.set_xlim), (ax.yaxis, ax._home[1], ax.set_ylim)):
                tr = axis.get_transform()
                h0, h1 = tr.transform(np.asarray(lim, float))
                cur = axis.get_view_interval()
                c0, c1 = tr.transform(np.asarray(cur, float))
                flip, (h0, h1), (c0, c1) = h0 > h1, sorted((h0, h1)), sorted((c0, c1))
                w = min(c1 - c0, h1 - h0)
                c0 = min(max(c0, h0), h1 - w)
                v = tr.inverted().transform(np.array([c0, c0 + w]))
                v = v[::-1] if flip else v
                if not np.allclose(v, sorted(cur, reverse=flip)):
                    setter(*v)
        finally:
            ax._clamping = False

    def reset(self, projection=None, colorbar=True):
        self.clear()
        if projection == "3d":
            self.ax = self.fig.add_subplot(111, projection="3d")
            self.cax = None
        else:
            gs = self.fig.add_gridspec(1, 2, width_ratios=[30, 1])
            self.ax = self.fig.add_subplot(gs[0, 0])
            self.cax = self.fig.add_subplot(gs[0, 1]) if colorbar else None
        return self.ax


class Stopped(Exception):
    pass


class Worker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(object)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.finished_ok.emit(self.fn(self.progress.emit))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=6)}")


def _line(text="", width=None):
    w = QtWidgets.QLineEdit(str(text))
    if width:
        w.setMaximumWidth(width)
    return w


def _dspin(v, lo, hi, step=0.1, dec=4):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setValue(v)
    return w


def _ispin(v, lo, hi):
    w = QtWidgets.QSpinBox()
    w.setRange(lo, hi)
    w.setValue(v)
    return w


def _combo(items, current=None):
    w = QtWidgets.QComboBox()
    w.addItems(items)
    if current is not None:
        w.setCurrentText(current)
    return w


_PATH_CODES = {1: "move", 2: "line", 3: "quad", 4: "cubic", 79: "close"}


def _tex_fix(tex: str) -> str:
    import re
    tex = re.sub(r"\\(mathbf|mathrm|mathit|boldsymbol)\s+([A-Za-z])", r"\\\1{\2}", tex.strip().strip("$"))
    return tex


def tex_path(tex: str) -> QtGui.QPainterPath | None:
    """Vector outline of a mathtext string (y up flipped to Qt's y down), or None if it does not parse."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    try:
        with matplotlib.rc_context({"mathtext.fontset": "cm"}):
            tp = TextPath((0, 0), f"${_tex_fix(tex)}$", size=20, prop=FontProperties(family="DejaVu Sans"))
            verts, codes = tp.vertices, tp.codes
    except Exception:
        return None
    path = QtGui.QPainterPath()
    i = 0
    pt = lambda k: QtCore.QPointF(verts[k, 0], -verts[k, 1])
    while i < len(codes):
        c = codes[i]
        if c == 1:
            path.moveTo(pt(i)); i += 1
        elif c == 2:
            path.lineTo(pt(i)); i += 1
        elif c == 3:
            path.quadTo(pt(i), pt(i + 1)); i += 2
        elif c == 4:
            path.cubicTo(pt(i), pt(i + 1), pt(i + 2)); i += 3
        else:
            path.closeSubpath(); i += 1
    path.setFillRule(Qt.WindingFill)
    return path


class EquationView(QtWidgets.QWidget):
    """Equations drawn as vector glyph outlines: sharp at any DPI scale, coloured from the palette."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lines: list[tuple[QtGui.QPainterPath | None, str]] = []
        self._note = ""
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_equations(self, lines, note: str = ""):
        if isinstance(lines, str):
            lines = [lines]
        self._lines = [(tex_path(t), t) for t in lines if t and t.strip()]
        self._note = note
        self.setMinimumHeight(self.heightForWidth(max(self.width(), 240)))
        self.updateGeometry()
        self.update()

    def hasHeightForWidth(self):
        return True

    def _layout(self, width):
        fm = QtGui.QFontMetricsF(self.font())
        W = max(width - 12, 40)
        paths = [(p, t, p.boundingRect() if p is not None else QtCore.QRectF()) for p, t in self._lines]
        widest = max([b.width() for p, _, b in paths if p is not None and not p.isEmpty()] or [1.0])
        k = min(fm.height() * 1.3 / 20.0, W / widest)
        gap = fm.height() * 0.45
        rows, y = [], 6.0
        for p, t, b in paths:
            h = b.height() * k if p is not None and not p.isEmpty() else fm.height()
            rows.append((p, t, b, y, h))
            y += h + gap
        note_h = 0.0
        if self._note:
            note_h = fm.boundingRect(QtCore.QRectF(0, 0, W, 1e4), Qt.TextWordWrap, self._note).height()
        return rows, k, y, note_h, W

    def heightForWidth(self, width):
        _, _, y, note_h, _ = self._layout(width)
        return int(y + note_h + 6)

    def sizeHint(self):
        return QtCore.QSize(240, self.heightForWidth(max(self.width(), 240)))

    def minimumSizeHint(self):
        return QtCore.QSize(80, self.heightForWidth(max(self.width(), 80)))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.setMinimumHeight(self.heightForWidth(self.width()))

    def paintEvent(self, _):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        ink = self.palette().color(QtGui.QPalette.WindowText)
        rows, k, y, _, W = self._layout(self.width())
        for path, tex, br, y0, h in rows:
            if path is None or path.isEmpty():
                qp.setPen(ink)
                qp.drawText(QtCore.QRectF(6, y0, W, h), Qt.AlignVCenter | Qt.AlignLeft, tex)
                continue
            qp.save()
            qp.translate(6 - br.left() * k, y0 - br.top() * k)
            qp.scale(k, k)
            qp.fillPath(path, ink)
            qp.restore()
        if self._note:
            col = QtGui.QColor(ink)
            col.setAlpha(150)
            qp.setPen(col)
            qp.drawText(QtCore.QRectF(6, y, W, 1e4), Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap, self._note)
        qp.end()


EQUATIONS = {
    "Poisson": [r"-\nabla\cdot(\kappa\,\nabla u)=f\quad\mathrm{in}\ \Omega"],
    "Diffusion": [r"-\nabla\cdot(A\,\nabla u)=f", r"A=\left[\genfrac{}{}{0}{}{A_{xx}}{A_{xy}}\;\;\genfrac{}{}{0}{}{A_{xy}}{A_{yy}}\right]"],
    "Reaction-diffusion": [r"-\nabla\cdot(A\,\nabla u)+c\,u=f"],
    "Advection-diffusion": [r"-\nabla\cdot(A\,\nabla u)+\mathbf{b}\cdot\nabla u+c\,u=f"],
    "Helmholtz": [r"-\Delta u-k^2u=f"],
    "Heat": [r"m\,\partial_t u-\nabla\cdot(A\,\nabla u)+\mathbf{b}\cdot\nabla u+c\,u=f(\mathbf{x},t)"],
    "Eigenproblem": [r"-\nabla\cdot(A\,\nabla u)+c\,u=\lambda\,m\,u"],
    "Linear elasticity": [r"-\nabla\cdot\sigma(\mathbf{u})=\mathbf{f}",
                          r"\sigma=\lambda\,(\nabla\cdot\mathbf{u})\,I+2\mu\,\varepsilon(\mathbf{u})",
                          r"\varepsilon(\mathbf{u})=\frac{1}{2}(\nabla\mathbf{u}+\nabla\mathbf{u}^{T})"],
}


class AdvisorDialog(QtWidgets.QDialog):
    COLORS = {"critical": "#ec4647", "warning": "#f2a600", "suggest": "#257de0", "info": "#8a8f98"}

    def __init__(self, rep, win):
        super().__init__(win)
        self.win, self.rep = win, rep
        self.setWindowTitle("Advisor")
        self.resize(820, 620)
        v = QtWidgets.QVBoxLayout(self)
        m = rep.metrics
        head = QtWidgets.QLabel(
            f"<b style='font-size:13pt'>Recommended: {rep.summary()}</b><br>"
            f"<span style='color:#8a8f98'>h = {m.get('h', 0):.3g} &nbsp; Pe<sub>h</sub> = {m.get('peclet', 0):.3g} &nbsp; "
            f"anisotropy = {m.get('anisotropy', 1):.3g} &nbsp; kh = {m.get('kh', 0):.3g} &nbsp; "
            f"min quality = {m.get('quality', 1):.2f} &nbsp; est. DOFs = {m.get('ndofs', 0):,}</span>")
        head.setTextFormat(Qt.RichText)
        head.setWordWrap(True)
        v.addWidget(head)
        counts = {s: sum(f.severity == s for f in rep.findings) for s in SEVERITIES}
        v.addWidget(QtWidgets.QLabel("   ".join(f"{n} {s}" for s, n in counts.items() if n)))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(body)
        bl.setSpacing(6)
        self.checks = []
        for f in rep.findings:
            card = QtWidgets.QFrame()
            card.setFrameShape(QtWidgets.QFrame.StyledPanel)
            col = self.COLORS[f.severity]
            card.setStyleSheet(f"QFrame {{ border-left: 4px solid {col}; }} QLabel {{ border: none; }}")
            cl = QtWidgets.QVBoxLayout(card)
            cl.setContentsMargins(10, 6, 8, 6)
            t = QtWidgets.QLabel(f"<span style='color:{col}; font-weight:600'>{f.severity.upper()}</span>"
                                 f" &nbsp;<span style='color:#8a8f98'>{f.topic}</span><br><b>{f.title}</b>")
            t.setTextFormat(Qt.RichText)
            t.setWordWrap(True)
            cl.addWidget(t)
            if f.detail:
                d = QtWidgets.QLabel(f.detail)
                d.setWordWrap(True)
                d.setTextInteractionFlags(Qt.TextSelectableByMouse)
                cl.addWidget(d)
            if f.action:
                cb = QtWidgets.QCheckBox("apply: " + ", ".join(f"{k} = {v}" for k, v in f.action.items()))
                cb.setChecked(f.severity in ("critical", "warning"))
                cb.setStyleSheet("QCheckBox { border: none; }")
                cl.addWidget(cb)
                self.checks.append((cb, f))
            bl.addWidget(card)
        bl.addStretch(1)
        scroll.setWidget(body)
        v.addWidget(scroll, 1)
        bb = QtWidgets.QHBoxLayout()
        self.chk_rerun = QtWidgets.QCheckBox("re-run after applying")
        bb.addWidget(self.chk_rerun)
        bb.addStretch(1)
        b = QtWidgets.QPushButton("Copy report")
        b.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(rep.as_text()))
        bb.addWidget(b)
        b = QtWidgets.QPushButton("Apply checked")
        b.setDefault(True)
        b.clicked.connect(self.apply)
        bb.addWidget(b)
        b = QtWidgets.QPushButton("Close")
        b.clicked.connect(self.reject)
        bb.addWidget(b)
        v.addLayout(bb)

    def checked_actions(self) -> dict:
        act = {}
        for cb, f in sorted(self.checks, key=lambda t: -t[1].rank):
            if cb.isChecked():
                act.update(f.action)
        return act

    def apply(self):
        act = self.checked_actions()
        self.win.apply_actions(act)
        self.accept()
        if self.chk_rerun.isChecked():
            if act.get("n"):
                self.win.generate_mesh()
            (self.win.run_adaptive if "strategy" in act and self.win.cb_analysis.currentText() == "Steady"
             else self.win.solve)()


# ----------------------------------------------------------------------------- main window
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FEM Toolbox")
        self.resize(1480, 900)
        self.cfg = M.Config()
        self.mesh = None
        self.result: M.JobResult | None = None
        self.worker: Worker | None = None
        self.history = []
        self._plot_tri = None
        self._plot_vals = None
        self._interp = None
        self._anim = QtCore.QTimer(self)
        self._anim.timeout.connect(self._anim_step)
        self._build()
        self.apply_config(self.cfg)
        self.generate_mesh()

    # ------------------------------------------------------------------ layout
    def _build(self):
        tb = self.addToolBar("main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.cb_preset = _combo(["Examples..."] + list(M.PRESETS))
        self.cb_preset.setMinimumWidth(320)
        self.cb_preset.activated.connect(self.load_preset)
        tb.addWidget(self.cb_preset)
        tb.addSeparator()
        self.act_mesh = tb.addAction("Mesh", self.generate_mesh)
        self.act_mesh.setShortcut("Ctrl+M")
        self.act_solve = tb.addAction("Solve", self.solve)
        self.act_solve.setShortcut("F5")
        self.act_adapt = tb.addAction("Adaptive loop", self.run_adaptive)
        self.act_adapt.setShortcut("F6")
        self.act_study = tb.addAction("Uniform study", self.run_study)
        self.act_study.setShortcut("F7")
        self.act_stop = tb.addAction("Stop", self.stop_worker)
        self.act_stop.setShortcut("Esc")
        tb.addSeparator()
        self.act_advisor = tb.addAction("Advisor", self.run_advisor)
        self.act_advisor.setShortcut("F8")
        for a in (self.act_mesh, self.act_solve, self.act_adapt, self.act_study, self.act_stop, self.act_advisor):
            a.setToolTip(f"{a.text()}  ({a.shortcut().toString()})")
        self._build_menus()

        split = QtWidgets.QSplitter(Qt.Horizontal)
        self.setCentralWidget(split)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(420)
        panel = QtWidgets.QWidget()
        scroll.setWidget(panel)
        lay = QtWidgets.QVBoxLayout(panel)
        self.toolbox = QtWidgets.QToolBox()
        lay.addWidget(self.toolbox)
        self.toolbox.addItem(self._page_problem(), "1  Problem")
        self.toolbox.addItem(self._page_geometry(), "2  Geometry && mesh")
        self.toolbox.addItem(self._page_bcs(), "3  Boundary conditions")
        self.toolbox.addItem(self._page_method(), "4  Discretization && solver")
        self.toolbox.addItem(self._page_adapt(), "5  Adaptivity")
        self.toolbox.addItem(self._page_time(), "6  Time stepping / eigenmodes")
        split.addWidget(scroll)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(2, 2, 2, 2)
        self.tabs = QtWidgets.QTabWidget()
        rl.addWidget(self.tabs)
        self.tabs.addTab(self._tab_solution(), "Solution")
        self.tabs.addTab(self._tab_mesh(), "Mesh")
        self.tabs.addTab(self._tab_convergence(), "Convergence")
        self.tabs.addTab(self._tab_profile(), "Line profile")
        self.tab_quant = self._tab_quantities()
        self.tabs.addTab(self.tab_quant, "Quantities")
        self.tab_hist = self._tab_history()
        self.tabs.addTab(self.tab_hist, "History")
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.tabs.addTab(self.log, "Log")
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([440, 1040])
        self.status = self.statusBar()
        self.lbl_probe = QtWidgets.QLabel("")
        self.status.addPermanentWidget(self.lbl_probe)
        self.lbl_run = QtWidgets.QLabel("")
        self.lbl_run.setContentsMargins(8, 0, 8, 0)
        self.status.addPermanentWidget(self.lbl_run)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(160)
        self.progress.setRange(0, 0)
        self.progress.hide()
        self.status.addPermanentWidget(self.progress)
        self.act_stop.setEnabled(False)

    def _build_menus(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        for text, fn, key in (("Load setup...", self.load_setup, "Ctrl+O"), ("Save setup...", self.save_setup, "Ctrl+S"),
                              (None, None, None), ("Import Gmsh mesh (.msh)...", self.import_msh, "Ctrl+I"),
                              ("Export mesh as Gmsh (.msh)...", self.export_msh, ""), (None, None, None),
                              ("Export CSV...", self.export_csv, ""), ("Export VTU (ParaView)...", self.export_vtu, ""),
                              ("Save plot as PNG...", self.save_plot, "Ctrl+P"), (None, None, None),
                              ("Quit", self.close, "Ctrl+Q")):
            if text is None:
                fm.addSeparator()
                continue
            a = fm.addAction(text, fn)
            if key:
                a.setShortcut(key)
        self.recent_menu = fm.addMenu("Recent setups")
        self._fill_recent()
        rm = mb.addMenu("&Run")
        for a in (self.act_mesh, self.act_solve, self.act_adapt, self.act_study, self.act_stop):
            rm.addAction(a)
        rm.addSeparator()
        rm.addAction("Refine mesh x4", self.refine_mesh)
        rm.addAction("Smooth mesh", self.smooth_mesh)
        rm.addAction("Refine marked cells once", self.refine_marked)
        am = mb.addMenu("&Analyze")
        am.addAction(self.act_advisor)
        a = am.addAction("Quantities and fluxes", lambda: self.tabs.setCurrentWidget(self.tab_quant))
        a.setShortcut("Ctrl+1")
        a = am.addAction("History / spectrum / iterations", lambda: self.tabs.setCurrentWidget(self.tab_hist))
        a.setShortcut("Ctrl+2")
        vm = mb.addMenu("&View")
        self.act_dark = vm.addAction("Dark theme")
        self.act_dark.setCheckable(True)
        self.act_dark.setShortcut("Ctrl+D")
        self.act_dark.toggled.connect(self.set_dark)
        a = vm.addAction("Toggle mesh edges", lambda: self.chk_edges.toggle())
        a.setShortcut("E")
        a = vm.addAction("Next step / mode", lambda: self.sl_step.setValue(self.sl_step.value() + 1))
        a.setShortcut("Right")
        a = vm.addAction("Previous step / mode", lambda: self.sl_step.setValue(self.sl_step.value() - 1))
        a.setShortcut("Left")
        hm = mb.addMenu("&Help")
        hm.addAction("Keyboard shortcuts and mouse", self.show_help)
        hm.addAction("About", lambda: QtWidgets.QMessageBox.about(
            self, "FEM Toolbox", "FEM Toolbox 2.1\nHigh-order CG / hp-DG finite elements in 2D."))

    def _settings(self):
        return QtCore.QSettings("femtoolbox", "FEM Toolbox")

    def _fill_recent(self):
        self.recent_menu.clear()
        files = self._settings().value("recent", []) or []
        if isinstance(files, str):
            files = [files]
        for f in files[:8]:
            self.recent_menu.addAction(f, lambda f=f: self.load_setup(f))
        self.recent_menu.setEnabled(bool(files))

    def _add_recent(self, path):
        st = self._settings()
        files = st.value("recent", []) or []
        if isinstance(files, str):
            files = [files]
        files = [path] + [f for f in files if f != path]
        st.setValue("recent", files[:8])
        self._fill_recent()

    def set_dark(self, on):
        apply_theme(QtWidgets.QApplication.instance(), on)
        self._settings().setValue("dark", bool(on))
        self.lbl_formula.update()
        self.redraw()
        self.draw_mesh()
        self.draw_convergence()
        self.draw_history()

    def show_help(self):
        QtWidgets.QMessageBox.information(self, "Shortcuts", "\n".join([
            "F5  solve        F6  adaptive loop      F7  uniform study      Esc  stop",
            "F8  advisor      Ctrl+M  mesh           Ctrl+D  dark theme     E  mesh edges",
            "Left / Right  previous / next time step or eigenmode",
            "Ctrl+O / Ctrl+S  load / save setup      Ctrl+I  import Gmsh mesh   Ctrl+P  save plot",
            "Ctrl+1  quantities    Ctrl+2  history",
            "",
            "Mouse on the solution plot:",
            "  move: probe value in the status bar",
            "  click: set the probe point for time series (transient)",
            "  shift+click: set line-profile start, then end"]))

    def _page_problem(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.cb_analysis = _combo(M.ANALYSES)
        self.cb_analysis.currentTextChanged.connect(self._on_analysis)
        f.addRow("Analysis", self.cb_analysis)
        self.cb_pde = _combo(M.PDE_PRESETS)
        self.cb_pde.currentTextChanged.connect(self._on_pde)
        f.addRow("PDE", self.cb_pde)
        self._problem_form = f
        self.lbl_formula = EquationView()
        f.addRow(self.lbl_formula)
        self.ed = {}
        self.grp_coef = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(self.grp_coef)
        g.setContentsMargins(0, 0, 0, 0)
        for i, (key, lab) in enumerate([("Axx", "A_xx / kappa"), ("Axy", "A_xy"), ("Ayy", "A_yy"), ("bx", "b_x"),
                                        ("by", "b_y"), ("c", "c"), ("m", "m (mass)"), ("f", "f(x,y,t)")]):
            self.ed[key] = _line()
            g.addWidget(QtWidgets.QLabel(lab), i, 0)
            g.addWidget(self.ed[key], i, 1)
        self.sp_k = _dspin(10, 0, 1e4, 1, 3)
        g.addWidget(QtWidgets.QLabel("k (Helmholtz)"), 8, 0)
        g.addWidget(self.sp_k, 8, 1)
        f.addRow(self.grp_coef)
        self.txt_custom = QtWidgets.QPlainTextEdit()
        self.txt_custom.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.txt_custom.setMaximumHeight(150)
        f.addRow("Custom PDE", self.txt_custom)
        self.grp_elastic = QtWidgets.QWidget()
        ge = QtWidgets.QFormLayout(self.grp_elastic)
        ge.setContentsMargins(0, 0, 0, 0)
        self.ed["E"], self.ed["nu"], self.ed["fx"], self.ed["fy"] = _line(), _line(), _line(), _line()
        self.cb_plane = _combo(["stress", "strain"])
        ge.addRow("Young's modulus E", self.ed["E"])
        ge.addRow("Poisson ratio nu", self.ed["nu"])
        ge.addRow("Plane", self.cb_plane)
        ge.addRow("body force f_x", self.ed["fx"])
        ge.addRow("body force f_y", self.ed["fy"])
        f.addRow(self.grp_elastic)
        box = QtWidgets.QGroupBox("Exact solution (optional: errors, rates)")
        bf = QtWidgets.QFormLayout(box)
        for key in ("u_exact", "ux", "uy"):
            self.ed[key] = _line()
            bf.addRow({"u_exact": "u", "ux": "du/dx", "uy": "du/dy"}[key], self.ed[key])
        f.addRow(box)
        return w

    def _page_geometry(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.cb_domain = _combo(M.DOMAINS)
        self.cb_domain.currentTextChanged.connect(self._on_domain)
        f.addRow("Domain", self.cb_domain)
        self.sp_w, self.sp_h = _dspin(1, 1e-3, 1e4), _dspin(1, 1e-3, 1e4)
        self.sp_r, self.sp_ri, self.sp_hr = _dspin(1, 1e-3, 1e4), _dspin(0.5, 1e-3, 1e4), _dspin(0.2, 1e-3, 1e4)
        self.ed["hole_x"], self.ed["hole_y"] = _line(width=80), _line(width=80)
        hole = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(hole)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.ed["hole_x"])
        hl.addWidget(self.ed["hole_y"])
        self.geo_rows = {}
        for key, lab, wid in [("w", "Width / size", self.sp_w), ("h", "Height", self.sp_h), ("r", "Radius", self.sp_r),
                              ("ri", "Inner radius", self.sp_ri), ("hr", "Hole radius", self.sp_hr),
                              ("hc", "Hole centre x, y", hole)]:
            f.addRow(lab, wid)
            self.geo_rows[key] = (f.labelForField(wid), wid)
        self.txt_poly = QtWidgets.QPlainTextEdit()
        self.txt_poly.setMaximumHeight(60)
        f.addRow("Polygon x,y; ...", self.txt_poly)
        self.geo_rows["poly"] = (f.labelForField(self.txt_poly), self.txt_poly)
        self.txt_param = QtWidgets.QPlainTextEdit()
        self.txt_param.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.txt_param.setMaximumHeight(120)
        f.addRow("Parametric curve", self.txt_param)
        self.geo_rows["param"] = (f.labelForField(self.txt_param), self.txt_param)
        mshrow = QtWidgets.QWidget()
        ml = QtWidgets.QHBoxLayout(mshrow)
        ml.setContentsMargins(0, 0, 0, 0)
        self.ed_msh = _line()
        self.ed_msh.setPlaceholderText("path to .msh (Gmsh 2.2 / 4.1 ASCII)")
        b = QtWidgets.QPushButton("Browse...")
        b.clicked.connect(self.import_msh)
        ml.addWidget(self.ed_msh, 1)
        ml.addWidget(b)
        f.addRow("Mesh file", mshrow)
        self.geo_rows["msh"] = (f.labelForField(mshrow), mshrow)
        self.chk_struct = QtWidgets.QCheckBox("structured grid (rectangles, L-shape)")
        f.addRow(self.chk_struct)
        self.sp_n = _ispin(16, 2, 2000)
        f.addRow("Cells per side n", self.sp_n)
        self.sp_hmesh = _dspin(0.0, 0.0, 100, 0.01, 4)
        self.sp_hmesh.setToolTip("Target edge length; 0 = use n")
        f.addRow("Target h (0 = from n)", self.sp_hmesh)
        row = QtWidgets.QWidget()
        rl = QtWidgets.QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        for txt, fn in (("Generate", self.generate_mesh), ("Refine x4", self.refine_mesh), ("Smooth", self.smooth_mesh)):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            rl.addWidget(b)
        f.addRow(row)
        self.lbl_mesh = QtWidgets.QLabel()
        self.lbl_mesh.setWordWrap(True)
        f.addRow(self.lbl_mesh)
        return w

    def _page_bcs(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.bc_table = QtWidgets.QTableWidget(0, 5)
        self.bc_table.setHorizontalHeaderLabels(["marker", "geometry", "type", "value  g  (or gx, gy)", "alpha"])
        self.bc_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.bc_table.verticalHeader().hide()
        self.bc_table.setMinimumHeight(190)
        v.addWidget(self.bc_table)
        h = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Rows from mesh markers")
        b.clicked.connect(lambda: self.refresh_bc_rows(keep=False))
        h.addWidget(b)
        h.addStretch(1)
        v.addLayout(h)
        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(QtWidgets.QLabel("Edges not covered by a row:"))
        self.cb_bc_default = _combo(M.SCALAR_BC)
        h2.addWidget(self.cb_bc_default)
        h2.addWidget(QtWidgets.QLabel("with g = 0"))
        h2.addStretch(1)
        v.addLayout(h2)
        hint = QtWidgets.QLabel("Extra rows (override markers):  name | kind | value | selector | alpha\n"
                                "selector: segment:(x0,y0)->(x1,y1), arc:0deg->90deg, t:0->pi/2, where:x<0.5")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.txt_bc_extra = QtWidgets.QPlainTextEdit()
        self.txt_bc_extra.setMaximumHeight(80)
        self.txt_bc_extra.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        v.addWidget(self.txt_bc_extra)
        return w

    def _page_method(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.cb_method = _combo(["CG", "DG", "Auto"])
        f.addRow("Method", self.cb_method)
        self.sp_deg = _ispin(2, 0, 8)
        f.addRow("Polynomial degree p", self.sp_deg)
        self.cb_basis = _combo(["lagrange", "modal"])
        self.cb_basis.setToolTip("modal = L2-orthonormal hierarchical (DG only)")
        f.addRow("DG basis", self.cb_basis)
        self.chk_supg = QtWidgets.QCheckBox("SUPG stabilization (CG, advection)")
        f.addRow(self.chk_supg)
        self.sp_pen = _dspin(10, 0.1, 1e4, 1, 2)
        f.addRow("DG penalty C_ip", self.sp_pen)
        self.cb_solver = _combo(M.SOLVERS)
        f.addRow("Linear solver", self.cb_solver)
        self.sp_tol = _dspin(1e-10, 1e-15, 1e-2, 1e-10, 15)
        f.addRow("Krylov tolerance", self.sp_tol)
        nl = QtWidgets.QGroupBox("Nonlinear (coefficients using u, ux, uy)")
        nf = QtWidgets.QFormLayout(nl)
        self.sp_nltol = _dspin(1e-8, 1e-14, 1e-1, 1e-8, 14)
        self.sp_nlit = _ispin(40, 1, 1000)
        self.sp_relax = _dspin(1.0, 0.05, 1.0, 0.05, 2)
        nf.addRow("step tolerance", self.sp_nltol)
        nf.addRow("max iterations", self.sp_nlit)
        nf.addRow("relaxation", self.sp_relax)
        nl.setToolTip("Newton for u in c and f, Picard for u in A or b. Initial guess: u0 (Time section).")
        f.addRow(nl)
        return w

    def _page_adapt(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.cb_strategy = _combo(["h", "hp"])
        self.cb_strategy.setToolTip("hp needs DG: smooth cells (modal decay) get p+1, others are bisected")
        f.addRow("Strategy", self.cb_strategy)
        self.cb_est = _combo(["residual", "zz"])
        f.addRow("Estimator", self.cb_est)
        self.sp_theta = _dspin(0.5, 0.05, 1.0, 0.05, 2)
        f.addRow("Dorfler theta", self.sp_theta)
        self.sp_maxdofs = _ispin(20000, 100, 5_000_000)
        f.addRow("Max DOFs", self.sp_maxdofs)
        self.sp_levels = _ispin(25, 1, 200)
        f.addRow("Max levels", self.sp_levels)
        self.sp_smooth = _dspin(1.5, 0.1, 10, 0.1, 2)
        f.addRow("hp smoothness threshold", self.sp_smooth)
        self.sp_pmax = _ispin(8, 1, 8)
        f.addRow("hp max degree", self.sp_pmax)
        b = QtWidgets.QPushButton("Refine marked cells once")
        b.clicked.connect(self.refine_marked)
        f.addRow(b)
        return w

    def _page_time(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.ed["u0"] = _line()
        f.addRow("u0(x,y)  (initial value; nonlinear guess)", self.ed["u0"])
        self.sp_dt = _dspin(0.01, 1e-8, 1e4, 0.001, 6)
        f.addRow("dt", self.sp_dt)
        self.sp_steps = _ispin(100, 1, 1_000_000)
        f.addRow("steps", self.sp_steps)
        self.cb_scheme = _combo(M.SCHEMES)
        f.addRow("scheme", self.cb_scheme)
        self.sp_modes = _ispin(8, 1, 200)
        f.addRow("eigenmodes", self.sp_modes)
        return w

    def _tab_solution(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        h = QtWidgets.QHBoxLayout()
        self.cb_field = _combo(SCALAR_FIELDS)
        self.cb_style = _combo(["filled", "filled + contours", "contours", "filled + flux arrows", "3D surface"])
        self.cb_cmap = _combo(CMAPS, "viridis")
        self.chk_edges = QtWidgets.QCheckBox("mesh")
        self.sp_sub = _ispin(0, 0, 12)
        self.sp_sub.setToolTip("Sub-sampling per cell (0 = p)")
        self.sp_scale = _dspin(0.0, 0.0, 1e9, 1.0, 3)
        self.sp_scale.setToolTip("Deformation scale (elasticity), 0 = auto")
        self.chk_deform = QtWidgets.QCheckBox("deformed")
        for lab, wd in (("field", self.cb_field), ("style", self.cb_style), ("cmap", self.cb_cmap), ("sub", self.sp_sub),
                        ("deform x", self.sp_scale)):
            h.addWidget(QtWidgets.QLabel(lab))
            h.addWidget(wd)
        h.addWidget(self.chk_deform)
        h.addWidget(self.chk_edges)
        h.addStretch(1)
        for wd in (self.cb_field, self.cb_style, self.cb_cmap):
            wd.currentTextChanged.connect(self.redraw)
        self.sp_sub.valueChanged.connect(self.redraw)
        self.sp_scale.valueChanged.connect(self.redraw)
        self.chk_edges.toggled.connect(self.redraw)
        self.chk_deform.toggled.connect(self.redraw)
        v.addLayout(h)
        self.canvas = Canvas(w)
        v.addWidget(NavigationToolbar2QT(self.canvas, w))
        v.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("motion_notify_event", self._probe)
        self.canvas.mpl_connect("button_press_event", self._click)
        self._nav = v.itemAt(1).widget()
        hs = QtWidgets.QHBoxLayout()
        self.lbl_step = QtWidgets.QLabel("")
        self.sl_step = QtWidgets.QSlider(Qt.Horizontal)
        self.sl_step.valueChanged.connect(self.redraw)
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        hs.addWidget(self.lbl_step)
        hs.addWidget(self.sl_step, 1)
        hs.addWidget(self.btn_play)
        self.step_bar = QtWidgets.QWidget()
        self.step_bar.setLayout(hs)
        self.step_bar.hide()
        v.addWidget(self.step_bar)
        return w

    def _tab_mesh(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.mesh_canvas = Canvas(w)
        v.addWidget(NavigationToolbar2QT(self.mesh_canvas, w))
        v.addWidget(self.mesh_canvas, 1)
        return w

    def _tab_convergence(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.conv_canvas = Canvas(w)
        v.addWidget(NavigationToolbar2QT(self.conv_canvas, w))
        v.addWidget(self.conv_canvas, 1)
        h = QtWidgets.QHBoxLayout()
        self.chk_keep = QtWidgets.QCheckBox("keep previous runs for comparison")
        self.chk_keep.setChecked(True)
        b = QtWidgets.QPushButton("Clear")
        b.clicked.connect(self._clear_history)
        h.addWidget(self.chk_keep)
        h.addStretch(1)
        h.addWidget(b)
        v.addLayout(h)
        return w

    def _tab_profile(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        h = QtWidgets.QHBoxLayout()
        self.ed_a, self.ed_b = _line("0, 0.5"), _line("1, 0.5")
        h.addWidget(QtWidgets.QLabel("from (x,y)"))
        h.addWidget(self.ed_a)
        h.addWidget(QtWidgets.QLabel("to"))
        h.addWidget(self.ed_b)
        b = QtWidgets.QPushButton("Plot")
        b.clicked.connect(self.plot_profile)
        h.addWidget(b)
        v.addLayout(h)
        self.prof_canvas = Canvas(w)
        v.addWidget(NavigationToolbar2QT(self.prof_canvas, w))
        v.addWidget(self.prof_canvas, 1)
        return w

    def _tab_quantities(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.tbl_quant = QtWidgets.QTableWidget(0, 2)
        self.tbl_quant.setHorizontalHeaderLabels(["quantity", "value"])
        self.tbl_quant.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tbl_quant.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tbl_quant.verticalHeader().hide()
        self.tbl_quant.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_quant.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        v.addWidget(self.tbl_quant, 1)
        box = QtWidgets.QGroupBox("Functionals  (expressions in x, y, u, ux, uy; boundary also nx, ny)")
        g = QtWidgets.QGridLayout(box)
        self.ed_int = _line("u^2")
        self.ed_bint = _line("-(ux*nx + uy*ny)")
        self.cb_bmark = _combo(["all"])
        self.lbl_int, self.lbl_bint = QtWidgets.QLabel("-"), QtWidgets.QLabel("-")
        for lab in (self.lbl_int, self.lbl_bint):
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lab.setMinimumWidth(140)
        g.addWidget(QtWidgets.QLabel("∫_Ω"), 0, 0)
        g.addWidget(self.ed_int, 0, 1, 1, 2)
        g.addWidget(QtWidgets.QLabel("dx  ="), 0, 3)
        g.addWidget(self.lbl_int, 0, 4)
        g.addWidget(QtWidgets.QLabel("∫_Γ"), 1, 0)
        g.addWidget(self.ed_bint, 1, 1)
        g.addWidget(self.cb_bmark, 1, 2)
        g.addWidget(QtWidgets.QLabel("ds  ="), 1, 3)
        g.addWidget(self.lbl_bint, 1, 4)
        b = QtWidgets.QPushButton("Evaluate")
        b.clicked.connect(self.eval_functionals)
        self.ed_int.returnPressed.connect(self.eval_functionals)
        self.ed_bint.returnPressed.connect(self.eval_functionals)
        g.addWidget(b, 0, 5, 2, 1)
        v.addWidget(box)
        h = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Recompute for shown step / mode")
        b.clicked.connect(self.update_quantities)
        h.addWidget(b)
        b = QtWidgets.QPushButton("Copy table")
        b.clicked.connect(self.copy_quantities)
        h.addWidget(b)
        h.addStretch(1)
        v.addLayout(h)
        return w

    def _tab_history(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("probe point x, y"))
        self.ed_probe = _line("0.5, 0.5", 140)
        self.ed_probe.returnPressed.connect(self.draw_history)
        h.addWidget(self.ed_probe)
        b = QtWidgets.QPushButton("Update")
        b.clicked.connect(self.draw_history)
        h.addWidget(b)
        self.lbl_hist = QtWidgets.QLabel("transient: time series  |  eigen: spectrum  |  nonlinear: iterations")
        h.addWidget(self.lbl_hist, 1)
        v.addLayout(h)
        self.hist_canvas = Canvas(w)
        v.addWidget(NavigationToolbar2QT(self.hist_canvas, w))
        v.addWidget(self.hist_canvas, 1)
        return w

    # ------------------------------------------------------------------ config <-> widgets
    def read_config(self) -> M.Config:
        c = M.Config()
        c.analysis = self.cb_analysis.currentText()
        c.pde = self.cb_pde.currentText()
        for key in ("Axx", "Axy", "Ayy", "bx", "by", "c", "m", "f", "E", "nu", "fx", "fy", "u_exact", "ux", "uy", "u0",
                    "hole_x", "hole_y"):
            setattr(c, key, self.ed[key].text().strip())
        c.k = self.sp_k.value()
        c.custom = self.txt_custom.toPlainText()
        c.plane = self.cb_plane.currentText()
        c.domain = self.cb_domain.currentText()
        c.width, c.height, c.radius = self.sp_w.value(), self.sp_h.value(), self.sp_r.value()
        c.inner_radius, c.hole_radius = self.sp_ri.value(), self.sp_hr.value()
        c.polygon, c.parametric = self.txt_poly.toPlainText(), self.txt_param.toPlainText()
        c.structured, c.n, c.h = self.chk_struct.isChecked(), self.sp_n.value(), self.sp_hmesh.value()
        rows = []
        for r in range(self.bc_table.rowCount()):
            marker = self.bc_table.item(r, 0).text()
            kind = self.bc_table.cellWidget(r, 2).currentText()
            val = self.bc_table.item(r, 3).text() if self.bc_table.item(r, 3) else "0"
            alpha = self.bc_table.item(r, 4).text() if self.bc_table.item(r, 4) else "1"
            rows.append([marker, kind, val, alpha])
        c.bc_rows, c.bc_default, c.bc_extra = rows, self.cb_bc_default.currentText(), self.txt_bc_extra.toPlainText()
        c.method, c.degree, c.basis = self.cb_method.currentText(), self.sp_deg.value(), self.cb_basis.currentText()
        c.supg, c.penalty, c.solver, c.tol = self.chk_supg.isChecked(), self.sp_pen.value(), self.cb_solver.currentText(), self.sp_tol.value()
        c.strategy, c.estimator, c.theta_mark = self.cb_strategy.currentText(), self.cb_est.currentText(), self.sp_theta.value()
        c.max_dofs, c.max_levels = self.sp_maxdofs.value(), self.sp_levels.value()
        c.smooth_threshold, c.p_max = self.sp_smooth.value(), self.sp_pmax.value()
        c.dt, c.nsteps, c.scheme, c.n_modes = self.sp_dt.value(), self.sp_steps.value(), self.cb_scheme.currentText(), self.sp_modes.value()
        c.mesh_file = self.ed_msh.text().strip()
        c.nl_tol, c.nl_maxit, c.nl_relax = self.sp_nltol.value(), self.sp_nlit.value(), self.sp_relax.value()
        self.cfg = c
        return c

    def apply_config(self, c: M.Config):
        blockers = [QtCore.QSignalBlocker(w) for w in (self.cb_analysis, self.cb_pde, self.cb_domain)]
        self.cb_analysis.setCurrentText(c.analysis)
        self.cb_pde.setCurrentText(c.pde)
        for key in ("Axx", "Axy", "Ayy", "bx", "by", "c", "m", "f", "E", "nu", "fx", "fy", "u_exact", "ux", "uy", "u0",
                    "hole_x", "hole_y"):
            self.ed[key].setText(str(getattr(c, key)))
        self.sp_k.setValue(c.k)
        self.txt_custom.setPlainText(c.custom)
        self.cb_plane.setCurrentText(c.plane)
        self.cb_domain.setCurrentText(c.domain)
        self.sp_w.setValue(c.width)
        self.sp_h.setValue(c.height)
        self.sp_r.setValue(c.radius)
        self.sp_ri.setValue(c.inner_radius)
        self.sp_hr.setValue(c.hole_radius)
        self.txt_poly.setPlainText(c.polygon)
        self.txt_param.setPlainText(c.parametric)
        self.chk_struct.setChecked(c.structured)
        self.sp_n.setValue(c.n)
        self.sp_hmesh.setValue(c.h)
        self.cb_bc_default.setCurrentText(c.bc_default)
        self.txt_bc_extra.setPlainText(c.bc_extra)
        self.cb_method.setCurrentText(c.method)
        self.sp_deg.setValue(c.degree)
        self.cb_basis.setCurrentText(c.basis)
        self.chk_supg.setChecked(c.supg)
        self.sp_pen.setValue(c.penalty)
        self.cb_solver.setCurrentText(c.solver)
        self.sp_tol.setValue(c.tol)
        self.cb_strategy.setCurrentText(c.strategy)
        self.cb_est.setCurrentText(c.estimator)
        self.sp_theta.setValue(c.theta_mark)
        self.sp_maxdofs.setValue(c.max_dofs)
        self.sp_levels.setValue(c.max_levels)
        self.sp_smooth.setValue(c.smooth_threshold)
        self.sp_pmax.setValue(c.p_max)
        self.sp_dt.setValue(c.dt)
        self.sp_steps.setValue(c.nsteps)
        self.cb_scheme.setCurrentText(c.scheme)
        self.sp_modes.setValue(c.n_modes)
        self.ed_msh.setText(c.mesh_file)
        self.sp_nltol.setValue(c.nl_tol)
        self.sp_nlit.setValue(c.nl_maxit)
        self.sp_relax.setValue(c.nl_relax)
        del blockers
        self._pending_bc_rows = c.bc_rows
        self._on_analysis(c.analysis, refresh=False)
        self._on_pde(c.pde)
        self._on_domain(c.domain)

    # ------------------------------------------------------------------ reactions
    def _on_analysis(self, name, refresh=True):
        el = name == "Elasticity"
        self.grp_elastic.setVisible(el)
        for wdg in (self.grp_coef, self.txt_custom, self.cb_pde):
            wdg.setVisible(not el)
            lab = self._problem_form.labelForField(wdg)
            if lab is not None:
                lab.setVisible(not el)
        self.cb_field.blockSignals(True)
        self.cb_field.clear()
        self.cb_field.addItems(ELASTIC_FIELDS if el else SCALAR_FIELDS)
        self.cb_field.blockSignals(False)
        if el:
            self.lbl_formula.set_equations(EQUATIONS["Linear elasticity"], PDE_FORMULAS["Linear elasticity"][1])
        if refresh:
            self.refresh_bc_rows(keep=False)
        self._on_pde(self.cb_pde.currentText())

    def _on_pde(self, name):
        if self.cb_analysis.currentText() == "Elasticity":
            return
        mode = self.cb_analysis.currentText()
        key = "Eigenproblem" if mode == "Eigenmodes" else name
        tex = PDE_FORMULAS.get(key, PDE_FORMULAS["Custom PDE"])[0]
        if name == "Custom PDE":
            for line in self.txt_custom.toPlainText().splitlines():
                if line.strip().lower().startswith("latex"):
                    tex = line.split("=", 1)[1].strip()
        lines = list(EQUATIONS.get(key, [tex])) if name != "Custom PDE" else [tex]
        if mode == "Transient" and key != "Heat":
            lines[0] = r"m\,\partial_t u" + ("" if lines[0].lstrip().startswith("-") else "+") + lines[0]
        if mode == "Transient":
            lines.append(r"u(\mathbf{x},0)=u_0(\mathbf{x})")
        self.lbl_formula.set_equations(lines, PDE_FORMULAS.get(key, ("", ""))[1])
        enabled = {"Poisson": {"Axx", "f"}, "Diffusion": {"Axx", "Axy", "Ayy", "f"},
                   "Reaction-diffusion": {"Axx", "Axy", "Ayy", "c", "f"},
                   "Advection-diffusion": {"Axx", "Axy", "Ayy", "bx", "by", "c", "f"}, "Helmholtz": {"f"},
                   "Heat": {"Axx", "Axy", "Ayy", "bx", "by", "c", "m", "f"}, "Custom PDE": set()}.get(name, set())
        for k in ("Axx", "Axy", "Ayy", "bx", "by", "c", "m", "f"):
            self.ed[k].setEnabled(k in enabled)
        self.sp_k.setEnabled(name == "Helmholtz")
        self.txt_custom.setEnabled(name == "Custom PDE")

    def _on_domain(self, name):
        show = {"Square": {"w"}, "Rectangle": {"w", "h"}, "Disk": {"r"}, "L-shape": {"w"}, "Annulus": {"r", "ri"},
                "Plate with hole": {"w", "h", "hr", "hc"}, "Polygon": {"poly"}, "Parametric": {"param"},
                "Gmsh file": {"msh"}}[name]
        for key, (lab, wid) in self.geo_rows.items():
            lab.setVisible(key in show)
            wid.setVisible(key in show)
        self.chk_struct.setEnabled(name in ("Square", "Rectangle", "L-shape"))

    def load_preset(self, idx):
        name = self.cb_preset.itemText(idx)
        if name not in M.PRESETS:
            return
        self.apply_config(M.apply_preset(self.cfg, name))
        self.generate_mesh()
        self.log_msg(f"loaded example: {name}")

    # ------------------------------------------------------------------ BC table
    def refresh_bc_rows(self, keep=True, rows=None):
        if self.mesh is None:
            return
        el = self.cb_analysis.currentText() == "Elasticity"
        kinds = M.VECTOR_BC if el else M.SCALAR_BC
        old = {}
        if keep:
            for r in range(self.bc_table.rowCount()):
                old[self.bc_table.item(r, 0).text()] = (self.bc_table.cellWidget(r, 2).currentText(),
                                                       self.bc_table.item(r, 3).text(), self.bc_table.item(r, 4).text())
        given = {}
        if rows:
            for mk, kind, val, alpha in rows:
                given[mk] = (kind, val, alpha)
        markers = self.mesh.boundary_marker_names()
        if "all" in given and not any(m in given for m in markers):
            markers = ["all"]
        extra = [m for m in given if m not in markers and m != "all"]
        self.bc_table.setRowCount(0)
        for mk in markers + extra:
            kind, val, alpha = given.get(mk, old.get(mk, ("fixed" if el else "dirichlet", "0, 0" if el else "0", "1")))
            if kind not in kinds:
                kind = kinds[0]
            r = self.bc_table.rowCount()
            self.bc_table.insertRow(r)
            it = QtWidgets.QTableWidgetItem(mk)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.bc_table.setItem(r, 0, it)
            gl = QtWidgets.QTableWidgetItem(self.mesh.boundary_marker_label(mk) if mk != "all" else "all boundary edges")
            gl.setFlags(gl.flags() & ~Qt.ItemIsEditable)
            gl.setToolTip(gl.text())
            self.bc_table.setItem(r, 1, gl)
            self.bc_table.setCellWidget(r, 2, _combo(kinds, kind))
            self.bc_table.setItem(r, 3, QtWidgets.QTableWidgetItem(str(val)))
            self.bc_table.setItem(r, 4, QtWidgets.QTableWidgetItem(str(alpha)))
        self.bc_table.resizeColumnToContents(0)
        self.bc_table.setColumnWidth(1, 120)

    # ------------------------------------------------------------------ mesh actions
    def generate_mesh(self):
        cfg = self.read_config()
        try:
            self.mesh = M.build_mesh(cfg)
        except Exception as exc:  # noqa: BLE001
            return self.fail(exc)
        self.result = None
        rows = getattr(self, "_pending_bc_rows", None)
        self.refresh_bc_rows(keep=rows is None, rows=rows)
        self._pending_bc_rows = None
        self._mesh_changed()

    def refine_mesh(self):
        if self.mesh is None:
            return
        self.mesh = refine_uniform(self.mesh)[0]
        self.result = None
        self._mesh_changed()

    def smooth_mesh(self):
        if self.mesh is None:
            return
        self.mesh = smooth_laplacian(self.mesh, 5)
        self.result = None
        self._mesh_changed()

    def refine_marked(self):
        if self.result is None or not isinstance(self.result.solution, FEMSolution) or self.result.pde is None:
            return self.status.showMessage("solve a steady scalar problem first", 4000)
        self.mesh, marked = M.refine_marked_once(self.read_config(), self.result.solution, self.result.pde, self.result.bcs)
        self.log_msg(f"refined {len(marked)} marked cells (Dorfler theta={self.cfg.theta_mark})")
        self.result = None
        self._mesh_changed()

    def _mesh_changed(self):
        self.lbl_mesh.setText(self.mesh.summary())
        self.draw_mesh()
        self.redraw()

    # ------------------------------------------------------------------ jobs
    def _start(self, kind):
        if self.worker is not None and self.worker.isRunning():
            return
        if self.mesh is None:
            self.generate_mesh()
        cfg = self.read_config()
        mesh = self.mesh
        self._stop_flag = False

        def job(progress):
            def cb(row, sol, eta):
                if self._stop_flag:
                    raise Stopped("stopped by user")
                progress((row, sol, eta))
            return M.run_job(kind, cfg, mesh, cb)

        self.worker = Worker(job)
        self.worker.finished_ok.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.progress.connect(self._progress)
        self._busy(True, kind)
        self._live_steps = []
        self.worker.start()

    def solve(self):
        self._start("solve")

    def run_adaptive(self):
        if self.cb_analysis.currentText() != "Steady":
            return self.status.showMessage("adaptivity applies to steady scalar problems", 4000)
        self._start("adapt")

    def run_study(self):
        if self.cb_analysis.currentText() != "Steady":
            return self.status.showMessage("uniform study applies to steady scalar problems", 4000)
        self._start("study")

    def stop_worker(self):
        self._stop_flag = True

    def _busy(self, on, kind=""):
        for a in (self.act_solve, self.act_adapt, self.act_study, self.act_mesh):
            a.setEnabled(not on)
        self.act_stop.setEnabled(on)
        self.progress.setVisible(on)
        if on:
            self.status.showMessage(f"running {kind} ...")

    def _progress(self, payload):
        row, sol, eta = payload
        self._live_steps.append(row)
        self.status.showMessage(f"level {row.level}: {row.ndofs} DOFs, eta={row.eta:.3e}")
        now = QtCore.QDateTime.currentMSecsSinceEpoch()
        if now - getattr(self, "_last_live", 0) < 700:
            return
        self._last_live = now
        self.result = M.JobResult("adapt", sol, sol.mesh, eta, list(self._live_steps))
        self.mesh = sol.mesh
        self.lbl_mesh.setText(self.mesh.summary())
        self.redraw()
        self.draw_convergence(live=True)

    def _done(self, res: M.JobResult):
        self._busy(False)
        self.result = res
        self.mesh = res.mesh if res.mesh is not None else self.mesh
        self.lbl_mesh.setText(self.mesh.summary())
        self.log_msg(f"[{res.kind}] {res.elapsed:.2f} s\n{res.text}")
        self.status.showMessage(f"{res.kind} finished in {res.elapsed:.2f} s", 6000)
        n = 0
        if isinstance(res.solution, TransientFEMSolution):
            n = len(res.solution.times) - 1
        elif isinstance(res.solution, EigenSolution):
            n = len(res.solution.eigenvalues) - 1
        self.step_bar.setVisible(n > 0)
        self.sl_step.blockSignals(True)
        self.sl_step.setRange(0, n)
        self.sl_step.setValue(n if isinstance(res.solution, TransientFEMSolution) else 0)
        self.sl_step.blockSignals(False)
        if res.kind in ("adapt", "study"):
            self._push_history(res)
            self.draw_convergence()
        self.draw_mesh()
        self.redraw()
        self.update_quantities()
        self.draw_history()
        info = getattr(res.solution, "info", None)
        if info is not None:
            self.lbl_run.setText(f"{info.method}-P{info.degree}  {info.ndofs:,} DOFs  {res.elapsed:.2f} s"
                                 + (f"  |e|_H1={res.errors.h1_semi:.2e}" if res.errors is not None else ""))
        if info is not None and info.warnings:
            self.status.showMessage("finished with warnings: " + "; ".join(info.warnings)[:200], 10000)

    def _failed(self, msg):
        self._busy(False)
        if "stopped by user" in msg:
            self.status.showMessage("stopped", 4000)
            if self._live_steps:
                self._push_history(self.result)
            return
        self.log_msg("ERROR " + msg)
        QtWidgets.QMessageBox.warning(self, "FEM Toolbox", msg.split("\n\n")[0])

    def fail(self, exc):
        self.log_msg("ERROR " + "".join(traceback.format_exception_only(type(exc), exc)))
        QtWidgets.QMessageBox.warning(self, "FEM Toolbox", str(exc))

    def log_msg(self, text):
        self.log.appendPlainText(text.rstrip() + "\n")

    def run_advisor(self):
        try:
            cfg = self.read_config()
            res = self.result
            sol = res.solution if res is not None else None
            eta = res.indicators if res is not None else None
            history = None
            if self.history:
                steps = self.history[-1][2]
                history = [(r.ndofs, getattr(r, "error_h1", None) or getattr(r, "h1", None) or r.eta) for r in steps]
            elastic = cfg.analysis == "Elasticity"
            pde = M.build_pde(cfg) if not elastic else M.PDE.poisson()
            rep = recommend(pde, self.mesh, cfg.degree, cfg.basis, hp_objective=cfg.strategy == "hp",
                            method=cfg.method if cfg.method != "Auto" else None, analysis=cfg.analysis,
                            bcs=M.build_bcs(cfg), solver=cfg.solver, supg=cfg.supg, penalty=cfg.penalty, dt=cfg.dt,
                            nsteps=cfg.nsteps, scheme=cfg.scheme, u0=cfg.u0, n_modes=cfg.n_modes,
                            elasticity=Elasticity(cfg.E, cfg.nu, cfg.fx, cfg.fy, cfg.plane) if elastic else None,
                            solution=sol if isinstance(sol, FEMSolution) else None,
                            eta=eta if isinstance(sol, FEMSolution) else None, history=history)
            self.log_msg("ADVISOR\n" + rep.as_text())
            if getattr(self, "advisor_dialog", None) is not None:
                self.advisor_dialog.close()
            self.advisor_dialog = AdvisorDialog(rep, self)
            self.advisor_dialog.show()
            return rep
        except Exception as exc:  # noqa: BLE001
            self.fail(exc)

    def apply_actions(self, act: dict):
        setters = {"method": lambda v: self.cb_method.setCurrentText(v), "degree": lambda v: self.sp_deg.setValue(int(v)),
                   "basis": lambda v: self.cb_basis.setCurrentText(v), "supg": lambda v: self.chk_supg.setChecked(bool(v)),
                   "solver": lambda v: self.cb_solver.setCurrentText(v), "penalty": lambda v: self.sp_pen.setValue(float(v)),
                   "strategy": lambda v: self.cb_strategy.setCurrentText(v), "scheme": lambda v: self.cb_scheme.setCurrentText(v),
                   "nl_relax": lambda v: self.sp_relax.setValue(float(v)), "n": lambda v: self.sp_n.setValue(int(v)),
                   "dt": lambda v: self.sp_dt.setValue(float(v))}
        done = []
        for k, v in act.items():
            if k in setters:
                setters[k](v)
                done.append(f"{k} = {v}")
        if done:
            self.log_msg("advisor applied: " + ", ".join(done))
            self.status.showMessage("applied: " + ", ".join(done), 6000)
        return done

    # ------------------------------------------------------------------ plotting
    def _current_solution(self):
        res = self.result
        if res is None or res.solution is None:
            return None, ""
        s = res.solution
        k = self.sl_step.value()
        if isinstance(s, TransientFEMSolution):
            self.lbl_step.setText(f"t = {s.times[min(k, len(s.times) - 1)]:.4g}")
            return s.step_solution(k), f"t = {s.times[min(k, len(s.times) - 1)]:.4g}"
        if isinstance(s, EigenSolution):
            k = min(k, len(s.eigenvalues) - 1)
            self.lbl_step.setText(f"mode {k + 1}: lambda = {s.eigenvalues[k]:.8g}")
            return s.mode(k), f"mode {k + 1}, $\\lambda$ = {s.eigenvalues[k]:.8g}"
        return s, ""

    def _field_data(self, sol, field, sub):
        mesh = sol.mesh
        if hasattr(sol, "problem"):
            scale = (self.sp_scale.value() or sol.auto_scale()) if self.chk_deform.isChecked() else 0.0
            return sol.plot_data(sub, _EL_KEY[field], scale), "dg"
        if field == "u":
            return sol.plot_data(sub), "fe"
        if field in ("|grad u|", "du/dx", "du/dy"):
            return sol.plot_data(sub, {"|grad u|": "grad", "du/dx": "ux", "du/dy": "uy"}[field]), "fe"
        if field == "|u - u_exact|":
            pde = self.result.pde if self.result else None
            if pde is None or pde.exact is None:
                raise ValueError("set an exact solution in the Problem section")
            pts, tris, v = sol.plot_data(sub)
            return (pts, tris, np.abs(v - pde.exact(pts[:, 0], pts[:, 1]))), "fe"
        if field == "error indicator":
            eta = self.result.indicators if self.result is not None and self.result.indicators is not None else None
            if eta is None or len(eta) != mesh.nelements:
                eta = residual_estimator(sol, self.result.pde, self.result.bcs)
            return (mesh.nodes, mesh.triangles, eta), "cell"
        if field == "polynomial degree":
            return (mesh.nodes, mesh.triangles, sol.space.degrees.astype(float)), "cell"
        if field == "smoothness (hp)":
            return (mesh.nodes, mesh.triangles, smoothness_indicator(sol)), "cell"
        if field == "mesh quality":
            return (mesh.nodes, mesh.triangles, mesh.quality()), "cell"
        raise ValueError(field)

    def redraw(self, *_):
        cv = self.canvas
        sol, subtitle = self._current_solution()
        style = self.cb_style.currentText()
        cmap = self.cb_cmap.currentText()
        field = self.cb_field.currentText()
        if sol is None:
            ax = cv.reset(colorbar=False)
            if self.mesh is not None:
                self._draw_edges(ax, self.mesh, lw=0.4, color="0.35")
                ax.set_title(f"{self.mesh.name}: {self.mesh.nelements} cells (press Solve)")
            ax.set_aspect("equal")
            cv.draw_idle()
            self._interp = None
            return
        try:
            (pts, tris, vals), kind = self._field_data(sol, field, self.sp_sub.value() or None)
        except Exception as exc:  # noqa: BLE001
            ax = cv.reset(colorbar=False)
            ax.text(0.5, 0.5, str(exc), ha="center", transform=ax.transAxes)
            cv.draw_idle()
            return
        tri = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
        finite = np.isfinite(vals)
        vv = np.where(finite, vals, np.nanmin(vals[finite]) if finite.any() else 0.0)
        if style == "3D surface" and kind != "cell":
            ax = cv.reset("3d")
            step = max(1, len(tris) // 40000)
            t3 = mtri.Triangulation(pts[:, 0], pts[:, 1], tris[::step])
            ax.plot_trisurf(t3, vv, cmap=cmap, linewidth=0, antialiased=False)
            ax.set_box_aspect((np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 0.5 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))))
        else:
            ax = cv.reset()
            if kind == "cell":
                m = ax.tripcolor(tri, facecolors=vv, cmap=cmap, edgecolors="face" if not self.chk_edges.isChecked() else "k",
                                 linewidth=0.15)
            else:
                if style in ("filled", "filled + contours", "filled + flux arrows"):
                    m = ax.tripcolor(tri, vv, shading="gouraud", cmap=cmap)
                if style == "filled + flux arrows":
                    self._draw_arrows(ax, sol)
                if style in ("contours", "filled + contours"):
                    lv = np.linspace(vv.min(), vv.max(), 16)[1:-1] if np.ptp(vv) > 0 else 8
                    cs = ax.tricontour(tri, vv, levels=lv, colors=ink() if style != "contours" else None,
                                       cmap=None if style != "contours" else cmap, linewidths=0.6)
                    if style == "contours":
                        m = cs
            if self.chk_edges.isChecked() and kind != "cell":
                self._draw_edges(ax, sol.mesh, lw=0.25, color=ink(), alpha=0.5)
            ax.set_aspect("equal")
            if cv.cax is not None:
                cv.fig.colorbar(m, cax=cv.cax)
            self._plot_tri, self._plot_vals, self._interp = tri, vv, None
            self._plot_kind = kind
        info = sol.info
        sp = getattr(sol, "space", None)
        disc = (f"hp-DG p={sp.degrees.min()}..{sp.degrees.max()}" if sp is not None and not sp.uniform
                else f"{info.method}-P{info.degree}")
        ax.set_title(f"{field}   {disc}, {info.ndofs} DOFs   {subtitle}", fontsize=10)
        cv.draw_idle()

    def _draw_arrows(self, ax, sol, nmax=700):
        mesh = sol.mesh
        c = mesh.nodes[mesh.triangles].mean(1)
        idx = np.arange(len(c)) if len(c) <= nmax else np.random.default_rng(0).choice(len(c), nmax, replace=False)
        if hasattr(sol, "problem"):
            pts, _, ux = sol.plot_data(1, "ux", 0.0)
            uy = sol.plot_data(1, "uy", 0.0)[2]
            idx = np.arange(len(pts)) if len(pts) <= nmax else np.random.default_rng(0).choice(len(pts), nmax, replace=False)
            c, vec, lab = pts, np.column_stack((ux, uy))[idx], "displacement"
            if self.chk_deform.isChecked():
                c = c + (self.sp_scale.value() or sol.auto_scale()) * np.column_stack((ux, uy))
        else:
            g = sol.gradient_at_centroids()[idx]
            pde = self.result.pde if self.result is not None else None
            if pde is not None and not pde.is_nonlinear:
                A = pde.diffusion(c[idx, 0], c[idx, 1])
                vec = -np.einsum("eab,eb->ea", A, g)
            else:
                vec = -g
            lab = "flux -A grad u"
        ax.quiver(c[idx, 0], c[idx, 1], vec[:, 0], vec[:, 1], color=ink(), alpha=0.75, width=0.0025,
                  pivot="mid", label=lab)

    def _draw_edges(self, ax, mesh, lw=0.4, color="k", alpha=1.0):
        if color == "k":
            color = ink()
        seg = mesh.nodes[mesh.edges[mesh.interior_edge_ids]]
        ax.add_collection(LineCollection(seg, linewidths=lw, colors=color, alpha=alpha))
        geo = mesh.geometry(max(2, self.sp_deg.value())) if mesh.boundary is not None else None
        be = mesh.boundary_edge_ids
        if geo is not None and geo.any_curved:
            X, _, _, _ = geo.edge(be, 0, np.linspace(0, 1, 9))
            ax.add_collection(LineCollection(X, linewidths=lw * 2, colors=color, alpha=alpha))
        else:
            ax.add_collection(LineCollection(mesh.nodes[mesh.edges[be]], linewidths=lw * 2, colors=color, alpha=alpha))
        ax.autoscale_view()

    def draw_mesh(self):
        if self.mesh is None:
            return
        cv = self.mesh_canvas
        cv.clear()
        gs = cv.fig.add_gridspec(1, 2, width_ratios=[3, 1])
        ax = cv.fig.add_subplot(gs[0, 0])
        mesh = self.mesh
        self._draw_edges(ax, mesh, lw=0.35, color="0.45")
        names = mesh.boundary_marker_names()
        cmap = matplotlib.colormaps["tab10"]
        be = mesh.boundary_edge_ids
        for i, mk in enumerate(names):
            sel = be[mesh.edge_markers == mk]
            ax.add_collection(LineCollection(mesh.nodes[mesh.edges[sel]], linewidths=3, colors=[cmap(i % 10)], label=mk))
        if self.result is not None and self.result.solution is not None and hasattr(self.result.solution, "space"):
            sp = self.result.solution.space
            if sp.method == "DG" and not sp.uniform:
                tri = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.triangles)
                pc = ax.tripcolor(tri, facecolors=sp.degrees.astype(float), cmap="viridis", alpha=0.6)
                cv.fig.colorbar(pc, ax=ax, label="p", shrink=0.6)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, title="markers")
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_title(mesh.summary(), fontsize=9)
        ax2 = cv.fig.add_subplot(gs[0, 1])
        ax2.hist(mesh.quality(), bins=30, range=(0, 1), color="0.4")
        ax2.set_xlabel("quality 4√3A/Σl²")
        ax2.set_yticks([])
        cv.draw_idle()

    def _push_history(self, res):
        if res is None or not res.steps:
            return
        if not self.chk_keep.isChecked():
            self.history.clear()
        cfg = self.cfg
        lab = f"{cfg.method}-P{cfg.degree} {'uniform' if res.kind == 'study' else cfg.strategy + '-adaptive'}"
        self.history.append((lab, res.kind, list(res.steps)))

    def _clear_history(self):
        self.history.clear()
        self.draw_convergence()

    def draw_convergence(self, live=False):
        cv = self.conv_canvas
        cv.clear()
        ax = cv.fig.add_subplot(111)
        runs = list(self.history)
        if live and self._live_steps:
            runs = runs + [("current run", "adapt", self._live_steps)]
        for i, (lab, kind, steps) in enumerate(runs):
            col = f"C{i % 10}"
            n = np.array([s.ndofs for s in steps], float)
            if kind == "study":
                ax.loglog(n, [s.l2 for s in steps], "o-", color=col, label=f"{lab}: L2")
                ax.loglog(n, [s.h1 for s in steps], "s--", color=col, label=f"{lab}: H1")
                ax.loglog(n, [s.eta for s in steps], ":", color=col, label=f"{lab}: eta")
            else:
                ax.loglog(n, [s.eta for s in steps], "-", color=col, label=f"{lab}: eta")
                if steps[0].error_h1 is not None:
                    ax.loglog(n, [s.error_h1 for s in steps], "s--", color=col, ms=3, label=f"{lab}: H1 error")
                    ax.loglog(n, [s.error_l2 for s in steps], "o:", color=col, ms=3, label=f"{lab}: L2 error")
        ax.set_xlabel("DOFs")
        ax.set_ylabel("error / estimator")
        ax.grid(True, which="both", alpha=0.3)
        if runs:
            ax.legend(fontsize=8)
        cv.draw_idle()

    def plot_profile(self):
        sol, _ = self._current_solution()
        if sol is None or not isinstance(sol, FEMSolution):
            return self.status.showMessage("solve a scalar problem first", 4000)
        try:
            a = [float(v) for v in self.ed_a.text().split(",")]
            b = [float(v) for v in self.ed_b.text().split(",")]
        except ValueError:
            return self.status.showMessage("endpoints must be x, y", 4000)
        s, u = sol.line_profile(a, b, 600)
        cv = self.prof_canvas
        cv.clear()
        ax = cv.fig.add_subplot(111)
        ax.plot(s, u, label="u_h")
        pde = self.result.pde if self.result else None
        if pde is not None and pde.exact is not None:
            P = np.asarray(a)[None] + (s / max(s[-1], 1e-300))[:, None] * (np.asarray(b) - np.asarray(a))[None]
            ax.plot(s, pde.exact(P[:, 0], P[:, 1]), "--", label="exact")
        ax.set_xlabel("arc length")
        ax.grid(alpha=0.3)
        ax.legend()
        cv.draw_idle()
        self.tabs.setCurrentIndex(3)

    def update_quantities(self):
        tbl = self.tbl_quant
        tbl.setRowCount(0)
        res = self.result
        sol, sub = self._current_solution()
        if sol is None or res is None:
            return
        rows = []
        try:
            if hasattr(sol, "problem"):
                for k, lab in (("umag", "max |u|"), ("vm", "max von Mises"), ("sxx", "max sigma_xx"), ("syy", "max sigma_yy")):
                    rows.append((lab, sol.max_field(k)))
            else:
                t = 0.0
                if isinstance(res.solution, TransientFEMSolution):
                    t = float(res.solution.times[min(self.sl_step.value(), len(res.solution.times) - 1)])
                q = quantities(sol, res.pde if not isinstance(res.solution, EigenSolution) else None, t)
                rows = list(q.rows)
                if q.flux_rows:
                    rows.append(("boundary flux  -A grad u . n  (outward)", None))
                    rows += [(f"   {m}", v) for m, v in q.flux_rows]
                if res.errors is not None:
                    rows += [("|u - u_exact|_L2", res.errors.l2), ("|u - u_exact|_H1", res.errors.h1_semi)]
                if res.indicators is not None and len(res.indicators) == sol.mesh.nelements:
                    rows.append(("estimator eta", float(np.sqrt((res.indicators ** 2).sum()))))
            info = sol.info
            rows += [("DOFs", info.ndofs), ("cells", info.nelements), ("nonzeros", info.nnz)]
        except Exception as exc:  # noqa: BLE001
            rows.append(("error", str(exc)))
        for name, val in rows:
            r = tbl.rowCount()
            tbl.insertRow(r)
            it = QtWidgets.QTableWidgetItem(name + (f"   [{sub}]" if r == 0 and sub else ""))
            if val is None:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            tbl.setItem(r, 0, it)
            txt = "" if val is None else (f"{val:,}" if isinstance(val, (int, np.integer)) else
                                          (f"{val:.10g}" if isinstance(val, (float, np.floating)) else str(val)))
            vi = QtWidgets.QTableWidgetItem(txt)
            vi.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            tbl.setItem(r, 1, vi)
        markers = ["all"] + (sol.mesh.boundary_marker_names() if sol is not None else [])
        cur = self.cb_bmark.currentText()
        self.cb_bmark.clear()
        self.cb_bmark.addItems(markers)
        if cur in markers:
            self.cb_bmark.setCurrentText(cur)

    def eval_functionals(self):
        sol, _ = self._current_solution()
        if sol is None or hasattr(sol, "problem"):
            return self.status.showMessage("solve a scalar problem first", 4000)
        for ed, lab, fn in ((self.ed_int, self.lbl_int, lambda e: integrate(sol, e)),
                            (self.ed_bint, self.lbl_bint, lambda e: boundary_integral(sol, e, self.cb_bmark.currentText()))):
            try:
                lab.setText(f"{fn(ed.text().strip() or '0'):.10g}")
            except Exception as exc:  # noqa: BLE001
                lab.setText(f"error: {exc}")

    def copy_quantities(self):
        lines = []
        for r in range(self.tbl_quant.rowCount()):
            a, b = self.tbl_quant.item(r, 0), self.tbl_quant.item(r, 1)
            lines.append(f"{a.text() if a else ''}\t{b.text() if b else ''}")
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        self.status.showMessage("quantities copied", 3000)

    def draw_history(self):
        cv = self.hist_canvas
        cv.clear()
        ax = cv.fig.add_subplot(111)
        res = self.result
        sol = res.solution if res is not None else None
        if isinstance(sol, TransientFEMSolution):
            try:
                x, y = (float(v) for v in self.ed_probe.text().split(",")[:2])
            except ValueError:
                x, y = 0.5, 0.5
            u = point_series(sol, x, y)
            W = integration_weights(sol.space)
            area = float(sol.mesh.areas.sum())
            mean = np.asarray(sol.values_by_step) @ W / area
            ax.plot(sol.times, u, "-", color="#257de0", label=f"u({x:.3g}, {y:.3g}, t)")
            ax.plot(sol.times, mean, "-", color="#f47230", label="mean u over the domain")
            if sol.space.basis_name == "lagrange":
                ax.plot(sol.times, np.asarray(sol.values_by_step).max(1), "--", color="#00b894", lw=1.2,
                        label="max nodal u")
            k = min(self.sl_step.value(), len(sol.times) - 1)
            ax.axvline(sol.times[k], color="0.5", lw=0.8)
            ax.set(xlabel="t", title="Time series")
            ax.legend(fontsize=8)
            self.lbl_hist.setText("click the solution plot to move the probe point")
        elif isinstance(sol, EigenSolution):
            lam = np.asarray(sol.eigenvalues)
            k = np.arange(1, len(lam) + 1)
            area = float(sol.mesh.areas.sum())
            ax.plot(k, lam, "o-", color="#257de0", label="computed")
            ax.plot(k, 4 * np.pi * k / area, "--", color="#f47230", label="Weyl 4πk/|Ω| (A = I)")
            ax.set(xlabel="mode k", ylabel="λ_k", title="Spectrum")
            ax.legend(fontsize=8)
            self.lbl_hist.setText("eigenvalues and Weyl asymptotics")
        elif sol is not None and getattr(sol, "info", None) is not None and "nonlinear_history" in sol.info.extra:
            hist = sol.info.extra["nonlinear_history"]
            it = [h[0] for h in hist]
            ax.semilogy(it, [h[1] for h in hist], "o-", color="#257de0", label="relative step |du|/|u|")
            r = [(h[0], h[2]) for h in hist if np.isfinite(h[2])]
            if r:
                ax.semilogy(*zip(*r), "s--", color="#f47230", label="nonlinear residual")
            ax.set(xlabel="iteration", title=sol.info.extra.get("nonlinear", "nonlinear iterations"))
            ax.legend(fontsize=8)
            self.lbl_hist.setText("nonlinear iteration history")
        else:
            ax.text(0.5, 0.5, "Run a transient, eigenmode or nonlinear problem", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
        ax.grid(alpha=0.3)
        cv.draw_idle()

    def _click(self, ev):
        if ev.inaxes is None or ev.xdata is None or self._nav.mode or ev.button != 1:
            return
        mods = QtWidgets.QApplication.keyboardModifiers()
        if mods & Qt.ShiftModifier:
            target = self.ed_b if getattr(self, "_profile_next_end", False) else self.ed_a
            target.setText(f"{ev.xdata:.4g}, {ev.ydata:.4g}")
            self._profile_next_end = not getattr(self, "_profile_next_end", False)
            if not self._profile_next_end:
                self.plot_profile()
            else:
                self.status.showMessage("profile start set; shift+click the end point", 4000)
            return
        self.ed_probe.setText(f"{ev.xdata:.4g}, {ev.ydata:.4g}")
        if self.result is not None and isinstance(self.result.solution, TransientFEMSolution):
            self.draw_history()
            self.status.showMessage("probe point moved (History tab)", 3000)

    def import_msh(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import Gmsh mesh", "", "Gmsh (*.msh)")
        if not path:
            return
        self.ed_msh.setText(path)
        self.cb_domain.setCurrentText("Gmsh file")
        self.generate_mesh()
        if self.mesh is not None:
            self.log_msg(f"imported {path}: {self.mesh.summary()}\nmarkers: {', '.join(self.mesh.boundary_marker_names())}")

    def export_msh(self):
        if self.mesh is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export mesh", "mesh.msh", "Gmsh (*.msh)")
        if path:
            self.log_msg(f"wrote {write_msh(self.mesh, path)}")

    def save_plot(self):
        cv = {0: self.canvas, 1: self.mesh_canvas, 2: self.conv_canvas, 3: self.prof_canvas}.get(
            self.tabs.currentIndex(), self.hist_canvas if self.tabs.currentWidget() is self.tab_hist else self.canvas)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save plot", "fem_plot.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if path:
            cv.fig.savefig(path, dpi=200)
            self.status.showMessage(f"saved {path}", 4000)

    def _probe(self, ev):
        if ev.inaxes is None or self._plot_tri is None or ev.xdata is None or self.cb_style.currentText() == "3D surface":
            return
        try:
            if self._interp is None:
                self._finder = self._plot_tri.get_trifinder()
                self._interp = True
            t = int(self._finder(ev.xdata, ev.ydata))
            if t < 0:
                self.lbl_probe.setText(f"x={ev.xdata:.4g}  y={ev.ydata:.4g}")
                return
            if self._plot_kind == "cell":
                v = self._plot_vals[t]
            else:
                tri = self._plot_tri.triangles[t]
                P = np.column_stack((self._plot_tri.x[tri], self._plot_tri.y[tri]))
                T = np.column_stack((P[1] - P[0], P[2] - P[0]))
                l = np.linalg.solve(T, np.array([ev.xdata, ev.ydata]) - P[0])
                v = (1 - l.sum()) * self._plot_vals[tri[0]] + l[0] * self._plot_vals[tri[1]] + l[1] * self._plot_vals[tri[2]]
            self.lbl_probe.setText(f"x={ev.xdata:.4g}  y={ev.ydata:.4g}  {self.cb_field.currentText()}={v:.6g}")
        except Exception:  # noqa: BLE001
            pass

    def toggle_play(self):
        if self._anim.isActive():
            self._anim.stop()
            self.btn_play.setText("Play")
        else:
            self._anim.start(60)
            self.btn_play.setText("Pause")

    def _anim_step(self):
        v = self.sl_step.value() + 1
        self.sl_step.setValue(0 if v > self.sl_step.maximum() else v)

    # ------------------------------------------------------------------ export / setup
    def export_csv(self):
        if self.mesh is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV stem", "fem_export.csv", "CSV (*.csv)")
        if not path:
            return
        sol = self.result.solution if self.result and not hasattr(self.result.solution, "problem") else None
        paths = export_all_csv(self.mesh, sol, path)
        self.log_msg("exported:\n" + "\n".join(str(p) for p in paths.values()))

    def export_vtu(self):
        sol, _ = self._current_solution()
        if sol is None:
            return self.status.showMessage("nothing to export", 4000)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export VTU", "fem_solution.vtu", "VTK (*.vtu)")
        if not path:
            return
        if hasattr(sol, "problem"):
            data = {k: sol.plot_data(None, k)[2] for k in ("ux", "uy", "sxx", "syy", "sxy", "vm")}
            pts, tris, _ = sol.plot_data(None, "vm")
        else:
            pts, tris, u = sol.plot_data()
            data = {"u": u, "grad_mag": sol.plot_data(None, "grad")[2]}
        self.log_msg(f"exported {export_vtu(path, pts, tris, data)}")

    def save_setup(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save setup", "fem_setup.json", "JSON (*.json)")
        if path:
            with open(path, "w") as fh:
                json.dump(self.read_config().to_dict(), fh, indent=1)
            self._add_recent(path)

    def load_setup(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load setup", "", "JSON (*.json)")
        if path:
            try:
                with open(path) as fh:
                    self.apply_config(M.Config.from_dict(json.load(fh)))
            except Exception as exc:  # noqa: BLE001
                return self.fail(exc)
            self._add_recent(path)
            self.generate_mesh()


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    dark = str(QtCore.QSettings("femtoolbox", "FEM Toolbox").value("dark", False)).lower() in ("true", "1")
    apply_theme(app, dark)
    win = MainWindow()
    win.act_dark.blockSignals(True)
    win.act_dark.setChecked(dark)
    win.act_dark.blockSignals(False)
    win.show()
    if not hasattr(sys, "ps1"):
        sys.exit(app.exec_())
    return win


if __name__ == "__main__":
    main()
