"""Regenerate the README figures: python -m femtoolbox.examples.gallery [outdir] [figure names...]"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection

from femtoolbox import (PDE, BoundaryCondition as BC, DiskDomain, Elasticity, LShapeDomain, ParametricDomain,
                        PlateWithHoleDomain, RectangleDomain, SquareDomain, adaptive_solve, error_norms, refine_uniform,
                        solve, solve_eigen, solve_elasticity, solve_transient)
from femtoolbox.core.adapt import LSHAPE_EXACT, LSHAPE_UX, LSHAPE_UY
from femtoolbox.core.elasticity import timoshenko_beam
from femtoolbox.gui.model import DEFAULT_PARAMETRIC, KIRSCH

C = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "lines.linewidth": 2, "lines.markersize": 5, "font.size": 9, "axes.titlesize": 10,
                     "legend.frameon": False})
D0 = [BC("dirichlet", 0.0, "all")]
BENCH = Path(__file__).resolve().parents[2] / "docs" / "benchmarks"


def _field(ax, sol, field="u", cmap="viridis", edges=False, vlim=None, **kw):
    pts, tris, v = sol.plot_data(None, field, **kw) if kw else sol.plot_data(None, field)
    m = ax.tripcolor(mtri.Triangulation(pts[:, 0], pts[:, 1], tris), v, shading="gouraud", cmap=cmap,
                     **(dict(vmin=vlim[0], vmax=vlim[1]) if vlim else {}))
    if edges:
        mesh = sol.mesh
        ax.add_collection(LineCollection(mesh.nodes[mesh.edges], lw=0.2, colors="k", alpha=0.4))
    _bare(ax)
    return m


def _bare(ax):
    ax.set_aspect("equal")
    ax.set_axis_off()


def convergence(out: Path):
    src, ue, ux, uy = "2*pi^2*sin(pi*x)*sin(pi*y)", "sin(pi*x)*sin(pi*y)", "pi*cos(pi*x)*sin(pi*y)", "pi*sin(pi*x)*cos(pi*y)"
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for i, p in enumerate([1, 2, 3, 4, 5]):
        n, e = [], []
        for k in (4, 8, 16, 32):
            s = solve(PDE.poisson(source=src), SquareDomain(1).mesh(k), "CG", p, bcs=D0)
            n.append(s.info.ndofs)
            e.append(error_norms(s, ue, ux, uy).l2)
        axs[0].loglog(n, e, "o-", color=C[i], label=f"P{p}  (slope {np.polyfit(np.log(n[-2:]), np.log(e[-2:]), 1)[0]:.2f})")
    axs[0].set(title="CG on the unit square, L2 error (slope vs DOFs = -(p+1)/2)", xlabel="DOFs")
    axs[0].legend(fontsize=8)
    for i, p in enumerate([1, 2, 3]):
        n, e = [], []
        for k in (4, 8, 16, 32):
            s = solve(PDE.poisson(source=src), SquareDomain(1).mesh(k), "DG", p, "modal", D0)
            n.append(s.info.ndofs)
            e.append(error_norms(s, ue, ux, uy).l2)
        axs[1].loglog(n, e, "s-", color=C[i], label=f"SIPG P{p}")
    axs[1].set(title="DG (SIPG, modal basis), L2 error", xlabel="DOFs")
    axs[1].legend(fontsize=8)
    ue = "cos(pi*(x^2+y^2)/2)"
    f = "2*pi*sin(pi*(x^2+y^2)/2) + pi^2*(x^2+y^2)*cos(pi*(x^2+y^2)/2)"
    for i, p in enumerate([1, 2, 3, 4]):
        m = DiskDomain(1).mesh(h=0.5)
        n, e = [], []
        for _ in range(4):
            s = solve(PDE.poisson(source=f), m, "CG", p, bcs=D0)
            n.append(s.info.ndofs)
            e.append(error_norms(s, ue).l2)
            m = refine_uniform(m)[0]
        axs[2].loglog(n, e, "o-", color=C[i], label=f"P{p} isoparametric (slope {np.polyfit(np.log(n[-2:]), np.log(e[-2:]), 1)[0]:.2f})")
    axs[2].set(title="Curved boundary (disk), L2 error", xlabel="DOFs")
    axs[2].legend(fontsize=8)
    fig.savefig(out / "convergence.png", dpi=140)
    plt.close(fig)


def adaptivity(out: Path):
    pde = PDE.poisson(source=0.0)
    bcs = [BC("dirichlet", LSHAPE_EXACT, "all")]
    kw = dict(bcs=bcs, u_exact=LSHAPE_EXACT, ux=LSHAPE_UX, uy=LSHAPE_UY)
    m0 = LShapeDomain(1).mesh(4)
    fig = plt.figure(figsize=(13, 4.2), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1])
    ax = fig.add_subplot(gs[0])
    runs = []
    for i, p in enumerate([1, 2]):
        n, e, m = [], [], m0
        for _ in range(5 if p == 1 else 4):
            s = solve(pde, m, "CG", p, bcs=bcs)
            n.append(s.info.ndofs)
            e.append(error_norms(s, LSHAPE_EXACT, LSHAPE_UX, LSHAPE_UY).h1_semi)
            m = refine_uniform(m)[0]
        ax.loglog(n, e, ":", marker="o", color=C[i], label=f"uniform CG-P{p}")
    for i, p in enumerate([1, 2]):
        r = adaptive_solve(pde, m0, "CG", p, max_dofs=60000, max_levels=60, **kw)
        ax.loglog([s.ndofs for s in r.steps], [s.error_h1 for s in r.steps], "-", color=C[i], label=f"h-adaptive CG-P{p}")
        runs.append(r)
    hp = adaptive_solve(pde, m0, "DG", 2, "modal", strategy="hp", max_dofs=12000, max_levels=80, smooth_threshold=1.5, **kw)
    ax.loglog([s.ndofs for s in hp.steps], [s.error_h1 for s in hp.steps], "-", color=C[2], label="hp-adaptive DG")
    ax.loglog([s.ndofs for s in hp.steps], [s.eta for s in hp.steps], "--", color=C[2], lw=1, label="hp-DG estimator")
    ax.set(title="L-shape, u = r^(2/3) sin(2φ/3): H1 error", xlabel="DOFs", ylabel="|u - u_h|_H1")
    ax.legend(fontsize=8)
    ax2 = fig.add_subplot(gs[1])
    k = max(i for i, m in enumerate(runs[1].meshes) if m.nelements < 2500)
    mesh = runs[1].meshes[k]
    ax2.add_collection(LineCollection(mesh.nodes[mesh.edges], lw=0.3, colors="k"))
    ax2.set(title=f"h-adaptive CG-P2 mesh, step {k} ({mesh.nelements} cells)", xlim=(0, 1), ylim=(0, 1))
    _bare(ax2)
    ax3 = fig.add_subplot(gs[2])
    mesh = hp.solution.mesh
    pc = ax3.tripcolor(mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.triangles),
                       facecolors=hp.solution.space.degrees.astype(float), cmap="viridis", edgecolors="w", lw=0.3)
    fig.colorbar(pc, ax=ax3, label="p", shrink=0.8)
    ax3.set(title=f"hp-DG: {mesh.nelements} cells, polynomial degree")
    _bare(ax3)
    fig.savefig(out / "adaptivity.png", dpi=140)
    plt.close(fig)
    return runs, hp


def supg(out: Path):
    pde = PDE.advection_diffusion(A=[[2e-3, 0], [0, 2e-3]], b=(1, 0.5), source=1.0)
    m = SquareDomain(1).mesh(32)
    sols = [("CG-P1 Galerkin", solve(pde, m, "CG", 1, bcs=D0)), ("CG-P1 + SUPG", solve(pde, m, "CG", 1, bcs=D0, supg=True))]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    for ax, (lab, s) in zip(axs, sols):
        mm = _field(ax, s, vlim=(0, 1.4))
        ax.set_title(f"{lab}: max {s.values.max():.2f}")
    fig.colorbar(mm, ax=axs[:2], shrink=0.8)
    for i, (lab, s) in enumerate(sols):
        x, u = s.line_profile((0, 0.75), (1, 0.75), 800)
        axs[2].plot(x, u, color=C[i], label=lab)
    axs[2].set(title="profile y = 0.75 (eps = 2e-3)", xlabel="x")
    axs[2].legend(fontsize=8)
    fig.savefig(out / "supg.png", dpi=140)
    plt.close(fig)


def gallery(out: Path):
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), layout="constrained")
    m = PlateWithHoleDomain(4, 4, (0, 0), 0.5).mesh(20)
    bcs = [BC("fixed_x", 0, "left"), BC("fixed_y", 0, "bottom"), BC("traction", f"{KIRSCH['sxx']}, {KIRSCH['sxy']}", "right"),
           BC("traction", f"{KIRSCH['sxy']}, {KIRSCH['syy']}", "top"), BC("traction", "0,0", "hole")]
    s = solve_elasticity(Elasticity(1, 0.3), m, 3, bcs)
    fig.colorbar(_field(axs[0, 0], s, "sxx", "magma", scale=0.0), ax=axs[0, 0], shrink=0.8)
    axs[0, 0].set_title(f"Kirsch plate, sigma_xx (max {s.max_field('sxx'):.4f}, exact 3)")
    ux, uy, _, sxy = timoshenko_beam(10, 2, 1, 1000, 0.3)
    s = solve_elasticity(Elasticity(1000, 0.3), RectangleDomain(10, 2, 0, -1).mesh(20, 4), 3,
                         [BC("fixed", f"{ux}, {uy}", "left"), BC("traction", f"0, {sxy}", "right"),
                          BC("traction", "0,0", "top"), BC("traction", "0,0", "bottom")])
    fig.colorbar(_field(axs[0, 1], s, "vm", "magma", scale=2.0), ax=axs[0, 1], shrink=0.5)
    axs[0, 1].set_title("Cantilever (x2 deformation), von Mises")
    e = solve_eigen(PDE.poisson(), LShapeDomain(1).mesh(16), 4, "CG", 3, bcs=D0)
    _field(axs[0, 2], e.mode(0), cmap="RdBu_r")
    axs[0, 2].set_title(f"L-shape mode 1, lambda = {e.eigenvalues[0]:.5f}")
    e = solve_eigen(PDE.poisson(), ParametricDomain(DEFAULT_PARAMETRIC).mesh(24), 8, "CG", 3, bcs=D0)
    _field(axs[0, 3], e.mode(5), cmap="RdBu_r")
    axs[0, 3].set_title(f"Flower membrane mode 6, lambda = {e.eigenvalues[5]:.4f}")
    s = solve(PDE.helmholtz(16, "200*exp(-200*((x-0.3)^2+(y-0.2)^2))"), DiskDomain(1).mesh(n=10), "CG", 4,
              bcs=[BC("robin", 0, "all", 0.0)])
    _field(axs[1, 0], s, cmap="RdBu_r")
    axs[1, 0].set_title("Helmholtz k = 16, CG-P4 curved")
    pde = PDE.heat(0.01, "20*exp(-80*((x-0.5-0.3*cos(2*pi*t))^2+(y-0.5-0.3*sin(2*pi*t))^2))")
    tr = solve_transient(pde, SquareDomain(1).mesh(24), "0", 0.01, 125, "crank-nicolson", "CG", 2, bcs=D0)
    fig.colorbar(_field(axs[1, 1], tr.final_solution(), cmap="inferno"), ax=axs[1, 1], shrink=0.8)
    axs[1, 1].set_title("Heat, orbiting source, t = 1.25 (CN)")
    plume = PDE.custom_from_definition("Axx=0.002\nAyy=0.002\nbx=-y\nby=x\nc=0.05\nf=exp(-60*((x-0.5)^2+y^2))")
    s = solve(plume, DiskDomain(1).mesh(n=14), "DG", 2, "modal", D0)
    fig.colorbar(_field(axs[1, 2], s), ax=axs[1, 2], shrink=0.8)
    axs[1, 2].set_title("Rotating plume, DG-P2 upwind")
    s = solve(PDE.diffusion_pde([[1, 0.9], [0.9, 1]], 0), ParametricDomain(DEFAULT_PARAMETRIC).mesh(20), "CG", 2,
              bcs=[BC("dirichlet", "x", "all")])
    fig.colorbar(_field(axs[1, 3], s, cmap="cividis"), ax=axs[1, 3], shrink=0.8)
    axs[1, 3].set_title("Anisotropic diffusion, flower domain")
    fig.savefig(out / "gallery.png", dpi=130)
    plt.close(fig)


def solvers(out: Path):
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    res = {k: ([], []) for k in ("direct", "cg-amg")}
    for n in (64, 128, 256, 512, 768):
        m = SquareDomain(1).mesh(n)
        for k in res:
            s = solve(PDE.poisson(source=1.0), m, "CG", 1, bcs=D0, solver=k, tol=1e-8)
            res[k][0].append(s.info.ndofs)
            res[k][1].append(s.info.t_setup + s.info.t_solve)
    for i, (k, (n, t)) in enumerate(res.items()):
        ax.loglog(n, t, "o-", color=C[i], label=k)
    ax.set(title="Poisson CG-P1: linear solve time", xlabel="DOFs", ylabel="seconds")
    ax.legend()
    fig.savefig(out / "solvers.png", dpi=140)
    plt.close(fig)


def speed(out: Path):
    import json
    data = json.loads((BENCH / "poisson.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for i, (meth, deg) in enumerate([("CG", 1), ("CG", 2), ("DG", 1), ("DG", 2)]):
        r = [d for d in data if d["method"] == meth and d["deg"] == deg]
        for ax, key in zip(axs, ("time", "l2")):
            ax.loglog([d["ndofs"] for d in r], [d[key] for d in r], "-", marker="o", color=C[i], label=f"{meth}-P{deg}")
    axs[0].set(title="Poisson solve, wall time (assembly + solve)", xlabel="DOFs", ylabel="seconds")
    axs[1].set(title="L2 error vs DOFs", xlabel="DOFs")
    axs[0].legend(fontsize=8)
    fig.savefig(out / "speed.png", dpi=140)
    plt.close(fig)


def main(argv=None):
    args = list(argv if argv is not None else sys.argv[1:])
    out = Path(args.pop(0) if args else "docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    for fn in (speed, convergence, adaptivity, supg, gallery, solvers):
        if args and fn.__name__ not in args:
            continue
        t = time.perf_counter()
        fn(out)
        print(f"{fn.__name__}: {time.perf_counter() - t:.1f} s")


if __name__ == "__main__":
    main()
