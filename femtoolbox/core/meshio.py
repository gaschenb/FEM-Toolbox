"""Gmsh .msh import/export (ASCII 2.2 and 4.1). Physical line groups become boundary markers."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from femtoolbox.core.mesh import Boundary, Mesh2D, PolylinePiece

_TRI = {2: 3, 9: 6, 21: 10}
_LINE = {1: 2, 8: 3, 26: 4}
_NNODES = {1: 2, 2: 3, 3: 4, 4: 4, 8: 3, 9: 6, 15: 1, 21: 10, 26: 4}


def _sections(text: str) -> dict[str, list[str]]:
    out, cur, buf = {}, None, []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("$End"):
            out[cur] = buf
            cur, buf = None, []
        elif s.startswith("$"):
            cur, buf = s[1:], []
        elif cur is not None:
            buf.append(s)
    return out


def _parse_v2(sec):
    n = int(sec["Nodes"][0])
    ids, xy = [], []
    for line in sec["Nodes"][1:n + 1]:
        p = line.split()
        ids.append(int(p[0]))
        xy.append((float(p[1]), float(p[2])))
    tris, lines = [], []
    for line in sec["Elements"][1:]:
        p = [int(v) for v in line.split()]
        etype, ntags = p[1], p[2]
        phys = p[3] if ntags else 0
        conn = p[3 + ntags:]
        if etype in _TRI:
            tris.append(conn[:3])
        elif etype in _LINE:
            lines.append((phys, conn[:2]))
    return np.array(ids), np.array(xy), tris, lines


def _parse_v4(sec):
    head = sec["Nodes"][0].split()
    nblocks = int(head[0])
    ids, xy, i = [], [], 1
    for _ in range(nblocks):
        _, _, param, nn = (int(v) for v in sec["Nodes"][i].split())
        i += 1
        tags = [int(sec["Nodes"][i + k]) for k in range(nn)]
        i += nn
        for k in range(nn):
            c = sec["Nodes"][i + k].split()
            xy.append((float(c[0]), float(c[1])))
        i += nn
        ids += tags
    ent_phys = {}
    if "Entities" in sec:
        e = sec["Entities"]
        npts, ncurves = (int(v) for v in e[0].split()[:2])
        for line in e[1 + npts:1 + npts + ncurves]:
            p = line.split()
            tag, nphys = int(p[0]), int(p[7])
            ent_phys[tag] = int(p[8]) if nphys else 0
    head = sec["Elements"][0].split()
    tris, lines, i = [], [], 1
    for _ in range(int(head[0])):
        dim, etag, etype, ne = (int(v) for v in sec["Elements"][i].split())
        i += 1
        for k in range(ne):
            p = [int(v) for v in sec["Elements"][i + k].split()]
            if etype in _TRI:
                tris.append(p[1:4])
            elif etype in _LINE and dim == 1:
                lines.append((ent_phys.get(etag, etag), p[1:3]))
        i += ne
    return np.array(ids), np.array(xy), tris, lines


def _chains(segs: np.ndarray) -> list[list[int]]:
    """Split a set of edges (node pairs) into polylines."""
    from collections import defaultdict
    adj = defaultdict(list)
    for k, (a, b) in enumerate(segs):
        adj[a].append((b, k))
        adj[b].append((a, k))
    used = np.zeros(len(segs), bool)
    out = []
    starts = [v for v, nb in adj.items() if len(nb) != 2] + list(adj)
    for s0 in starts:
        for nxt, k in adj[s0]:
            if used[k]:
                continue
            chain, cur = [s0], s0
            while True:
                step = next(((v, j) for v, j in adj[cur] if not used[j]), None)
                if step is None:
                    break
                used[step[1]] = True
                cur = step[0]
                chain.append(cur)
                if cur == s0:
                    break
            out.append(chain)
    return out


def read_msh(path, name: str | None = None) -> Mesh2D:
    text = Path(path).read_text(errors="replace")
    sec = _sections(text)
    version = float(sec.get("MeshFormat", ["2.2"])[0].split()[0])
    ids, xy, tris, lines = (_parse_v4 if version >= 4 else _parse_v2)(sec)
    if not tris:
        raise ValueError("no triangles in the mesh file")
    names = {}
    for line in sec.get("PhysicalNames", [])[1:]:
        p = line.split(maxsplit=2)
        names[int(p[1])] = p[2].strip().strip('"')
    lookup = np.full(int(ids.max()) + 1, -1)
    lookup[ids] = np.arange(len(ids))
    T = lookup[np.asarray(tris)]
    used = np.unique(T)
    remap = np.full(len(xy), -1)
    remap[used] = np.arange(len(used))
    nodes = xy[used]
    T = remap[T]
    pieces = []
    if lines:
        groups: dict = {}
        for phys, (a, b) in lines:
            groups.setdefault(phys, []).append((remap[lookup[a]], remap[lookup[b]]))
        for phys, segs in groups.items():
            marker = names.get(phys, f"line_{phys}" if phys else "boundary")
            for chain in _chains(np.asarray(segs)):
                P = nodes[chain]
                t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
                pieces.append(PolylinePiece(P, t, marker, f"{marker} (gmsh)"))
    from femtoolbox.core.mesh import label_longest_edge
    mesh = Mesh2D(nodes, label_longest_edge(T, nodes), name or Path(path).stem)
    if pieces:
        tagged = {tuple(sorted((remap[lookup[a]], remap[lookup[b]]))) for _, (a, b) in lines}
        rest = [tuple(e) for e in mesh.edges[mesh.boundary_edge_ids] if tuple(sorted(e)) not in tagged]
        for chain in _chains(np.asarray(rest)) if rest else []:
            P = nodes[chain]
            t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            pieces.append(PolylinePiece(P, t, "untagged", "untagged boundary (gmsh)"))
        mesh.boundary = Boundary(pieces)
    return mesh


def write_msh(mesh: Mesh2D, path) -> Path:
    """ASCII 2.2 with boundary edges as physical lines named after their markers."""
    path = Path(path)
    markers = mesh.boundary_marker_names()
    tag = {m: i + 1 for i, m in enumerate(markers)}
    surf = len(markers) + 1
    be = mesh.edges[mesh.boundary_edge_ids]
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(markers) + 1)]
    lines += [f'1 {tag[m]} "{m}"' for m in markers] + [f'2 {surf} "domain"', "$EndPhysicalNames", "$Nodes",
                                                        str(mesh.nnodes)]
    lines += [f"{i + 1} {x:.17g} {y:.17g} 0" for i, (x, y) in enumerate(mesh.nodes)]
    lines += ["$EndNodes", "$Elements", str(len(be) + mesh.nelements)]
    k = 1
    for (a, b), m in zip(be, mesh.edge_markers):
        lines.append(f"{k} 1 2 {tag[m]} {tag[m]} {a + 1} {b + 1}")
        k += 1
    for t in mesh.triangles:
        lines.append(f"{k} 2 2 {surf} {surf} {t[0] + 1} {t[1] + 1} {t[2] + 1}")
        k += 1
    lines.append("$EndElements")
    path.write_text("\n".join(lines) + "\n")
    return path
