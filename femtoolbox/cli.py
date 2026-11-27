"""Command line: run a saved GUI setup or a built-in example without the window."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _config(args):
    from femtoolbox.gui import model as M
    if args.preset:
        names = [n for n in M.PRESETS if args.preset.lower() in n.lower()]
        if not names:
            sys.exit(f"no example matches '{args.preset}' (see: fem-toolbox list)")
        return M.apply_preset(M.Config(), names[0]), names[0]
    if not args.setup:
        sys.exit("give a setup .json (File > Save setup in the GUI) or --preset NAME")
    return M.Config.from_dict(json.loads(Path(args.setup).read_text())), args.setup


def cmd_list(_args):
    from femtoolbox.gui import model as M
    for i, name in enumerate(M.PRESETS, 1):
        print(f"{i:2d}. {name}")


def cmd_run(args):
    from femtoolbox.core.advisor import recommend
    from femtoolbox.core.export import export_all_csv, export_solution_vtu
    from femtoolbox.core.meshio import write_msh
    from femtoolbox.core.postprocess import quantities
    from femtoolbox.gui import model as M
    cfg, label = _config(args)
    for kv in args.set or []:
        key, _, val = kv.partition("=")
        if not hasattr(cfg, key):
            sys.exit(f"unknown setting '{key}'")
        cur = getattr(cfg, key)
        setattr(cfg, key, type(cur)(val) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else
                (val.lower() in ("1", "true", "yes") if isinstance(cur, bool) else val))
    mesh = M.build_mesh(cfg)
    kind = {"solve": "solve", "adapt": "adapt", "study": "study"}[args.kind]
    if cfg.analysis == "Elasticity":
        kind = "elasticity"
    print(f"{label}: {mesh.nelements} cells")
    res = M.run_job(kind, cfg, mesh)
    print(res.text)
    print(f"elapsed {res.elapsed:.2f} s")
    sol = res.solution
    if sol is not None and res.pde is not None and res.kind == "solve":
        print()
        print(quantities(sol, res.pde).as_text())
    if args.advise and res.pde is not None:
        rep = recommend(res.pde, res.mesh, cfg.degree, cfg.basis, method=cfg.method if cfg.method != "Auto" else None,
                        analysis=cfg.analysis, bcs=res.bcs, solver=cfg.solver, supg=cfg.supg, penalty=cfg.penalty,
                        solution=sol if res.kind == "solve" else None,
                        eta=res.indicators if res.kind == "solve" else None)
        print()
        print(rep.as_text())
    if args.vtu and sol is not None and hasattr(sol, "plot_data"):
        print("wrote", export_solution_vtu(sol, args.vtu))
    if args.csv and sol is not None:
        for p in export_all_csv(res.mesh, sol, args.csv).values():
            print("wrote", p)
    if args.msh:
        print("wrote", write_msh(res.mesh, args.msh))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "gui":
        from femtoolbox.gui.app import main as gui_main
        return gui_main()
    ap = argparse.ArgumentParser(prog="fem-toolbox", description="FEM Toolbox. With no arguments the GUI opens.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gui", help="open the GUI")
    sub.add_parser("list", help="list the built-in examples").set_defaults(fn=cmd_list)
    r = sub.add_parser("run", help="solve a saved setup or a built-in example")
    r.add_argument("setup", nargs="?", help="setup .json saved from the GUI")
    r.add_argument("--preset", help="built-in example (any unique part of its name)")
    r.add_argument("--kind", choices=["solve", "adapt", "study"], default="solve")
    r.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a setting, e.g. --set degree=3")
    r.add_argument("--advise", action="store_true", help="print the advisor report")
    r.add_argument("--vtu", help="write the solution as .vtu")
    r.add_argument("--csv", help="write mesh and solution CSVs with this base path")
    r.add_argument("--msh", help="write the final mesh as Gmsh .msh")
    r.set_defaults(fn=cmd_run)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
