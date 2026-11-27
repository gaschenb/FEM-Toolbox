from __future__ import annotations

from pathlib import Path

import numpy as np

from femtoolbox.core.mesh import Mesh2D


def _stem(base_path) -> Path:
    p = Path(base_path)
    stem = p.with_suffix("") if p.suffix.lower() in (".csv", ".vtu") else p
    stem.parent.mkdir(parents=True, exist_ok=True)
    return stem


def _csv(path: Path, header: list[str], cols: list[np.ndarray], fmts: list[str]) -> Path:
    data = np.column_stack([np.asarray(c, dtype=object) for c in cols]) if cols else np.zeros((0, 0))
    with path.open("w", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in data:
            f.write(",".join(fm % v if fm != "%s" else str(v) for fm, v in zip(fmts, row)) + "\n")
    return path


def export_mesh_csv(mesh: Mesh2D, base_path) -> dict[str, Path]:
    stem = _stem(base_path)
    n, m, E = mesh.nnodes, mesh.nelements, mesh.nedges
    marker = np.full(E, "interior", dtype=object)
    geom = np.full(E, "interior", dtype=object)
    marker[mesh.boundary_edge_ids] = mesh.edge_markers
    labels = {mk: mesh.boundary_marker_label(mk) for mk in mesh.boundary_marker_names()}
    geom[mesh.boundary_edge_ids] = [labels[mk] for mk in mesh.edge_markers]
    ec, el = mesh.edge_cells, mesh.edge_local
    right = np.where(ec[:, 1] >= 0, ec[:, 1].astype(object), "")
    right_e = np.where(ec[:, 1] >= 0, el[:, 1].astype(object), "")
    return {
        "mesh_nodes": _csv(stem.with_name(stem.name + "_mesh_nodes.csv"), ["node_id", "x", "y"],
                           [np.arange(n), mesh.nodes[:, 0], mesh.nodes[:, 1]], ["%d", "%.17g", "%.17g"]),
        "mesh_elements": _csv(stem.with_name(stem.name + "_mesh_elements.csv"), ["element_id", "node0", "node1", "node2"],
                              [np.arange(m), *mesh.triangles.T], ["%d"] * 4),
        "mesh_facets": _csv(stem.with_name(stem.name + "_mesh_facets.csv"),
                            ["facet_id", "node0", "node1", "left_cell", "left_edge", "right_cell", "right_edge", "marker",
                             "boundary_geometry", "is_boundary"],
                            [np.arange(E), mesh.edges[:, 0], mesh.edges[:, 1], ec[:, 0], el[:, 0], right, right_e, marker,
                             geom, (ec[:, 1] < 0).astype(int)],
                            ["%d", "%d", "%d", "%d", "%d", "%s", "%s", "%s", "%s", "%d"]),
    }


def _solution_parts(solution):
    from femtoolbox.core.solution import EigenSolution, TransientFEMSolution
    if isinstance(solution, TransientFEMSolution):
        return [(i, float(t), solution.step_solution(i)) for i, t in enumerate(solution.times)]
    if isinstance(solution, EigenSolution):
        return [(i, float(lam), solution.mode(i)) for i, lam in enumerate(solution.eigenvalues)]
    return [(None, None, solution)]


def export_solution_csv(solution, base_path) -> dict[str, Path]:
    from femtoolbox.core.solution import EigenSolution, TransientFEMSolution
    stem = _stem(base_path)
    parts = _solution_parts(solution)
    tag = "_time" if isinstance(solution, TransientFEMSolution) else "_modes" if isinstance(solution, EigenSolution) else ""
    pre = ["step", "time"] if tag == "_time" else ["mode", "eigenvalue"] if tag == "_modes" else []
    node_cols, dof_cols = [], []
    for i, t, s in parts:
        pts, _, v = s.plot_data()
        x = s.space.dof_coords
        pfx_n = [np.full(len(v), i), np.full(len(v), t)] if pre else []
        pfx_d = [np.full(len(x), i), np.full(len(x), t)] if pre else []
        node_cols.append(pfx_n + [np.arange(len(v)), pts[:, 0], pts[:, 1], v])
        dof_cols.append(pfx_d + [np.arange(len(x)), x[:, 0], x[:, 1], s.values])
    cat = lambda cols: [np.concatenate([c[k] for c in cols]) for k in range(len(cols[0]))]
    f_pre = ["%d", "%.17g"] if pre else []
    out = {
        "solution_nodes": _csv(stem.with_name(stem.name + f"_solution{tag}_nodes.csv"), pre + ["node_id", "x", "y", "u"],
                               cat(node_cols), f_pre + ["%d", "%.17g", "%.17g", "%.17g"]),
        "solution_dofs": _csv(stem.with_name(stem.name + f"_solution{tag}_dofs.csv"), pre + ["dof_id", "x", "y", "u"],
                              cat(dof_cols), f_pre + ["%d", "%.17g", "%.17g", "%.17g"]),
    }
    return out


def export_all_csv(mesh: Mesh2D, solution, base_path) -> dict[str, Path]:
    paths = export_mesh_csv(mesh, base_path)
    if solution is not None:
        paths.update(export_solution_csv(solution, base_path))
    return paths


def export_vtu(path, points: np.ndarray, triangles: np.ndarray, point_data: dict[str, np.ndarray] | None = None,
               cell_data: dict[str, np.ndarray] | None = None) -> Path:
    """ASCII VTK UnstructuredGrid (ParaView). Use FEMSolution.plot_data() to export high-order fields exactly."""
    path = Path(path).with_suffix(".vtu")
    path.parent.mkdir(parents=True, exist_ok=True)
    npnt, ncell = len(points), len(triangles)
    P = np.column_stack((points, np.zeros(npnt)))

    def arr(name, a, ncomp=1):
        a = np.asarray(a, float).reshape(-1)
        return (f'<DataArray type="Float64" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">\n'
                + " ".join(f"{v:.9g}" for v in a) + "\n</DataArray>\n")
    s = ['<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n<UnstructuredGrid>\n',
         f'<Piece NumberOfPoints="{npnt}" NumberOfCells="{ncell}">\n<Points>\n', arr("Points", P, 3), "</Points>\n<Cells>\n",
         '<DataArray type="Int64" Name="connectivity" format="ascii">\n' + " ".join(map(str, np.asarray(triangles).ravel())) + "\n</DataArray>\n",
         '<DataArray type="Int64" Name="offsets" format="ascii">\n' + " ".join(map(str, 3 * np.arange(1, ncell + 1))) + "\n</DataArray>\n",
         '<DataArray type="UInt8" Name="types" format="ascii">\n' + " ".join(["5"] * ncell) + "\n</DataArray>\n</Cells>\n"]
    if point_data:
        s.append("<PointData>\n" + "".join(arr(k, v) for k, v in point_data.items()) + "</PointData>\n")
    if cell_data:
        s.append("<CellData>\n" + "".join(arr(k, v) for k, v in cell_data.items()) + "</CellData>\n")
    s.append("</Piece>\n</UnstructuredGrid>\n</VTKFile>\n")
    path.write_text("".join(s))
    return path


def export_solution_vtu(solution, path) -> Path:
    pts, tris, v = solution.plot_data()
    return export_vtu(path, pts, tris, {"u": v})
