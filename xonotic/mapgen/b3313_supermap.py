#!/usr/bin/env python3
"""B3313-Inspired Non-Euclidean Real-Map Variant Supermap Stitcher.

Tiles M real existing Xonotic stock maps + parametric architectural variants & motifs
(from `map_variants.py`) plus optional B3313 transitional nexus chambers into a single
leak-free `.bsp` supermap connected by velocity-preserving portal-membrane joints
(`trigger_teleport` with `TELEPORT_KEEP_SPEED` + verified walkable `info_teleport_destination`
exits + native Xonotic `waypoint_spawnforteleporter` bot pathing).

Design properties:
  - Supports K <= 24 teams, strictly J > K cartpaths (up to 256), N <= 255 players.
  - Arbitrary non-Euclidean expander topology (d >= 3 per submap for M >= 4), just like B3313.
  - Parametric variant & motif genome per submap: rotation, mirror (exact winding flip),
    scale, material/lighting palette, monolith baffles, skybridge overpasses, jumppad shafts,
    bunker archways, and pickup economies.
  - Disjoint 3D spatial atlas packing keeps coordinates cleanly inside DarkPlaces bounds.
  - Emits `.map`, `.bsp`, `.waypoints`, `.waypoints.cache`, `.mapinfo`, and `.joints.json`
    for direct live kinetic-gas membrane thermometry (`gas_thermometry_audit.py`).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
import zipfile
from pathlib import Path

from gas_world import sample_gas_world_params
from map_variants import (
    ALL_MOTIFS,
    PALETTES,
    PICKUP_ECONOMIES,
    STOCK_MAP_CATALOG,
    ParsedEntity,
    StockMapAsset,
    SubmapVariantSpec,
    _box_block,
    instantiate_stock_variant,
    load_stock_map,
)
from spiralgen import fmt, quad_prism, vlen, vsub


def build_expander_edges(M: int, rng: random.Random, min_deg: int = 3) -> list[tuple[int, int]]:
    """Build a connected non-Euclidean expander graph over M submaps with degree >= min_deg."""
    if M <= 1:
        return []
    if M == 2:
        return [(0, 1)]
    if M == 3:
        return [(0, 1), (1, 2), (0, 2)]
    target_d = min(M - 1, max(3, min_deg))
    edges: set[tuple[int, int]] = set()
    # Start with a Hamiltonian ring for guaranteed connectivity + short cycle
    for i in range(M):
        a, b = i, (i + 1) % M
        edges.add((min(a, b), max(a, b)))
    # Add opposite/chordal expander edges until every node has deg >= target_d
    deg = [2] * M
    # Prefer long-range non-planar chords (stride ~ M//2 and M//3) for true B3313 expansion!
    candidates = []
    for i in range(M):
        for j in range(i + 2, M):
            if i == 0 and j == M - 1:
                continue
            dist_ring = min(j - i, M - (j - i))
            candidates.append((dist_ring, rng.random(), i, j))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    for _, _, i, j in candidates:
        if deg[i] < target_d or deg[j] < target_d:
            if deg[i] < target_d + 1 and deg[j] < target_d + 1:
                edges.add((i, j))
                deg[i] += 1
                deg[j] += 1
        if all(d >= target_d for d in deg):
            break
    # Ensure every vertex reaches target_d even if tight
    for i in range(M):
        if deg[i] < target_d:
            for j in range(M):
                if j != i and (min(i, j), max(i, j)) not in edges:
                    edges.add((min(i, j), max(i, j)))
                    deg[i] += 1
                    deg[j] += 1
                    if deg[i] >= target_d:
                        break
    return sorted(edges)


def build_nexus_tile(tile_id: int, dst_offset: tuple[float, float, float], rng: random.Random) -> dict:
    """Create a B3313-style Transitional Castle Foyer / Nexus Chamber tile."""
    S = 1536.0
    H = 448.0
    t = 64.0
    ox, oy, oz = dst_offset[0] - S * 0.5, dst_offset[1] - S * 0.5, dst_offset[2]
    x0, y0, z0 = ox, oy, oz
    x1, y1, z1 = ox + S, oy + S, oz + H

    tex_f = "exx/base-crete01"
    tex_w = "exx/wall-bigrib02"
    tex_t = "exx/trim-01"
    brushes = [
        _box_block(x0 - t, y0 - t, z0 - t, x1 + t, y1 + t, z0, tex_f, tex_w),
        _box_block(x0 - t, y0 - t, z1, x1 + t, y1 + t, z1 + t, tex_w, tex_w),
        _box_block(x0 - t, y0 - t, z0, x0, y1 + t, z1, tex_w, tex_w),
        _box_block(x1, y0 - t, z0, x1 + t, y1 + t, z1, tex_w, tex_w),
        _box_block(x0, y0 - t, z0, x1, y0, z1, tex_w, tex_w),
        _box_block(x0, y1, z0, x1, y1 + t, z1, tex_w, tex_w),
    ]
    # Iconic B3313 4-column central peristyle
    mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    for sx in (-256.0, 256.0):
        for sy in (-256.0, 256.0):
            brushes.append(_box_block(mx + sx - 32, my + sy - 32, z0 - 8, mx + sx + 32, my + sy + 32, z1 + 8, tex_t, tex_w))

    ents = [
        ParsedEntity(kv={"classname": "light", "origin": f"{mx:.1f} {my:.1f} {z1 - 64:.1f}", "light": "950"}, brushes=[]),
        ParsedEntity(kv={"classname": "item_health_mega", "origin": f"{mx:.1f} {my:.1f} {z0 + 28:.1f}"}, brushes=[]),
        ParsedEntity(kv={"classname": "weapon_vortex", "origin": f"{mx + 140:.1f} {my:.1f} {z0 + 28:.1f}"}, brushes=[]),
        ParsedEntity(kv={"classname": "weapon_devastator", "origin": f"{mx - 140:.1f} {my:.1f} {z0 + 28:.1f}"}, brushes=[]),
    ]

    # 5x5 walkable waypoint grid
    wps, wp_centers, wp_edges = [], [], []
    mat = []
    margin = 160.0
    for iy in range(5):
        row = []
        py = y0 + margin + (S - 2 * margin) * iy / 4.0
        for ix in range(5):
            px = x0 + margin + (S - 2 * margin) * ix / 4.0
            p = (px, py, z0 + 28.0)
            idx = len(wps)
            wps.append((p, p, 0))
            wp_centers.append(p)
            row.append(idx)
        mat.append(row)
    for iy in range(5):
        for ix in range(5):
            if ix + 1 < 5:
                wp_edges.append((mat[iy][ix], mat[iy][ix + 1]))
            if iy + 1 < 5:
                wp_edges.append((mat[iy][ix], mat[iy + 1][ix]))
            if ix + 1 < 5 and iy + 1 < 5:
                wp_edges.append((mat[iy][ix], mat[iy + 1][ix + 1]))

    stock_spawns = [
        ((x0 + 200, y0 + 200, z0 + 32), 45.0),
        ((x1 - 200, y1 - 200, z0 + 32), 225.0),
        ((x0 + 200, y1 - 200, z0 + 32), 315.0),
        ((x1 - 200, y0 + 200, z0 + 32), 135.0),
    ]
    return {
        "tile_id": tile_id,
        "base_map": "b3313_nexus",
        "variant_label": "b3313_nexus[peristyle_foyer]",
        "spec": {"base_map": "b3313_nexus", "motifs": ["peristyle_foyer"], "palette": "obsidian_ruin"},
        "brushes": brushes,
        "entities": ents,
        "waypoints": wps,
        "wp_centers": wp_centers,
        "wp_edges": wp_edges,
        "walk_ids": list(range(len(wps))),
        "stock_spawns": stock_spawns,
        "aabb_min": (x0 - t, y0 - t, z0 - t),
        "aabb_max": (x1 + t, y1 + t, z1 + t),
    }


def _build_local_walk_adj(tile: dict) -> dict[int, list[int]]:
    """Build horizontal walkable adjacency graph for a submap tile."""
    walk_set = set(tile["walk_ids"])
    adj: dict[int, set[int]] = {i: set() for i in tile["walk_ids"]}
    pts = tile["wp_centers"]
    for a, b in tile["wp_edges"]:
        if a in walk_set and b in walk_set:
            pa, pb = pts[a], pts[b]
            if abs(pa[2] - pb[2]) <= 64.0 and vlen(vsub(pa, pb)) <= 480.0:
                adj[a].add(b)
                adj[b].add(a)
    # Augment sparse/uncached stock maps with conservative same-level k-NN links
    for i in tile["walk_ids"]:
        if len(adj[i]) >= 3:
            continue
        pi = pts[i]
        cands = []
        for j in tile["walk_ids"]:
            if j == i or j in adj[i]:
                continue
            pj = pts[j]
            dh = math.hypot(pj[0] - pi[0], pj[1] - pi[1])
            dz = abs(pj[2] - pi[2])
            if 48.0 <= dh <= 320.0 and dz <= 36.0:
                cands.append((dh, j))
        cands.sort()
        for _, j in cands[:3]:
            adj[i].add(j)
            adj[j].add(i)
    return {k: sorted(v) for k, v in adj.items()}


def _pick_portal_anchor_and_exit(
    tile: dict,
    adj: dict[int, list[int]],
    used_anchors: list[tuple[float, float, float]],
    rng: random.Random,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float], int, int]:
    """Select a well-spaced walkable portal anchor P and a verified walkable exit Q
    110-260u away on the same floor so exiting players never re-trigger the return portal."""
    pts = tile["wp_centers"]
    # Score candidates by separation from already-placed portals and having a clean exit neighbor
    best = None
    cands = list(tile["walk_ids"])
    rng.shuffle(cands)
    for wi in cands:
        p = pts[wi]
        min_sep = min((vlen(vsub(p, u)) for u in used_anchors), default=9999.0)
        if used_anchors and min_sep < 340.0:
            continue
        # Find a verified same-level walkable exit neighbor 115..280u away
        exits = []
        for wj in adj.get(wi, []):
            q = pts[wj]
            dh = math.hypot(q[0] - p[0], q[1] - p[1])
            dz = abs(q[2] - p[2])
            if 115.0 <= dh <= 280.0 and dz <= 28.0:
                exits.append((abs(dh - 160.0), wj, q))
        if not exits:
            # Fallback to any same-level walkable node in 115..260u
            for wj in tile["walk_ids"]:
                if wj == wi:
                    continue
                q = pts[wj]
                dh = math.hypot(q[0] - p[0], q[1] - p[1])
                dz = abs(q[2] - p[2])
                if 115.0 <= dh <= 260.0 and dz <= 24.0:
                    exits.append((abs(dh - 160.0), wj, q))
        if not exits:
            continue
        exits.sort()
        _, wq, q = exits[0]
        score = min(min_sep, 1200.0) + 40.0 * min(len(adj.get(wi, [])), 6)
        if best is None or score > best[0]:
            dx, dy = q[0] - p[0], q[1] - p[1]
            L = math.hypot(dx, dy) or 1.0
            best = (score, p, q, (dx / L, dy / L), wi, wq)
    if best is None:
        # Ultimate fallback: pick farthest unused walkable node
        wi = max(tile["walk_ids"], key=lambda i: min((vlen(vsub(pts[i], u)) for u in used_anchors), default=9999.0))
        p = pts[wi]
        wj = min((j for j in tile["walk_ids"] if j != wi), key=lambda j: abs(vlen(vsub(pts[j], p)) - 150.0))
        q = pts[wj]
        dx, dy = q[0] - p[0], q[1] - p[1]
        L = math.hypot(dx, dy) or 1.0
        return p, q, (dx / L, dy / L), wi, wj
    return best[1], best[2], best[3], best[4], best[5]


def _make_portal_trigger_brush(
    center: tuple[float, float, float],
    normal: tuple[float, float],
    width: float,
    depth: float = 48.0,
    height: float = 128.0,
) -> str:
    """Build an oriented `common/trigger` prism of exact aperture width `W_e`."""
    nx, ny = normal
    tx, ty = -ny, nx
    hw, hd = width * 0.5, depth * 0.5
    z0 = center[2] - 24.0
    p0 = (center[0] - tx * hw - nx * hd, center[1] - ty * hw - ny * hd, z0)
    p1 = (center[0] + tx * hw - nx * hd, center[1] + ty * hw - ny * hd, z0)
    p2 = (center[0] + tx * hw + nx * hd, center[1] + ty * hw + ny * hd, z0)
    p3 = (center[0] - tx * hw + nx * hd, center[1] - ty * hw + ny * hd, z0)
    br = quad_prism(p0, p1, p2, p3, (0.0, 0.0, height), "common/trigger", "common/trigger")
    lines = br.to_map(0.5).strip().splitlines()
    return "\n".join(lines[1:-1])


def _make_portal_arch_illusionary(
    center: tuple[float, float, float],
    normal: tuple[float, float],
    width: float,
    height: float = 128.0,
) -> list[str]:
    """Non-blocking visual B3313 portal arch (floor threshold + high overhead lintel)."""
    nx, ny = normal
    tx, ty = -ny, nx
    hw = width * 0.5
    z0 = center[2] - 24.0
    # Floor threshold strip (4u thin, visual beacon)
    out = []
    for za, dz, hd, tex in ((z0, 4.0, 18.0, "exx/trim-01"), (z0 + height, 16.0, 14.0, "exx/trim-01")):
        p0 = (center[0] - tx * hw - nx * hd, center[1] - ty * hw - ny * hd, za)
        p1 = (center[0] + tx * hw - nx * hd, center[1] + ty * hw - ny * hd, za)
        p2 = (center[0] + tx * hw + nx * hd, center[1] + ty * hw + ny * hd, za)
        p3 = (center[0] - tx * hw + nx * hd, center[1] - ty * hw + ny * hd, za)
        br = quad_prism(p0, p1, p2, p3, (0.0, 0.0, dz), tex, tex)
        lines = br.to_map(0.5).strip().splitlines()
        out.append("\n".join(lines[1:-1]))
    return out


def _trace_cart_tracks_on_tile(
    tile: dict,
    adj: dict[int, list[int]],
    n_carts: int,
    rng: random.Random,
    avoid_pts: list[tuple[float, float, float]],
    target_len: float = 820.0,
) -> list[list[tuple[float, float, float]]]:
    """Trace `n_carts` clean walkable cartpaths over a real stock submap's waypoint graph."""
    if n_carts <= 0:
        return []
    pts = tile["wp_centers"]
    walk = list(tile["walk_ids"])
    used_starts: list[tuple[float, float, float]] = []
    tracks: list[list[tuple[float, float, float]]] = []

    for ci in range(n_carts):
        # Pick start farthest from previously chosen cart starts & portal anchors
        cands = [w for w in walk if len(adj.get(w, [])) >= 1]
        if not cands:
            cands = walk
        if used_starts:
            start = max(cands, key=lambda w: min(vlen(vsub(pts[w], u)) for u in used_starts) + rng.uniform(0, 60.0))
        else:
            start = rng.choice(cands)
        used_starts.append(pts[start])

        # Try up to 4 candidate starts to guarantee a clean walkable cartpath >= 500u
        best_poly = None
        for _attempt in range(4):
            path_ids = [start]
            visited = {start}
            acc_len = 0.0
            cur = start
            while acc_len < target_len and len(path_ids) < 10:
                nbs = [nb for nb in adj.get(cur, []) if nb not in visited]
                if not nbs:
                    break
                p_cur = pts[cur]
                p_prev = pts[path_ids[-2]] if len(path_ids) >= 2 else None
                def _score(nb):
                    pn = pts[nb]
                    step = vlen(vsub(pn, p_cur))
                    dz_pen = 3.0 * abs(pn[2] - p_cur[2])
                    fwd = 0.0
                    if p_prev is not None:
                        v1 = (p_cur[0] - p_prev[0], p_cur[1] - p_prev[1])
                        v2 = (pn[0] - p_cur[0], pn[1] - p_cur[1])
                        l1 = math.hypot(*v1) or 1.0
                        l2 = math.hypot(*v2) or 1.0
                        fwd = 120.0 * ((v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2))
                    return vlen(vsub(pn, pts[start])) + fwd - dz_pen - 0.1 * abs(step - 220.0)
                nxt = max(nbs, key=_score)
                step = vlen(vsub(pts[nxt], p_cur))
                if step < 32.0:
                    visited.add(nxt)
                    continue
                path_ids.append(nxt)
                visited.add(nxt)
                acc_len += step
                cur = nxt
            if len(path_ids) >= 2 and (best_poly is None or acc_len > best_poly[0]):
                best_poly = (acc_len, [pts[i] for i in path_ids])
            if acc_len >= 560.0:
                break
            start = rng.choice(cands)

        if best_poly is None:
            p0 = pts[start]
            other = min((w for w in walk if w != start), key=lambda w: abs(vlen(vsub(pts[w], p0)) - 600.0))
            poly = [p0, pts[other]]
        else:
            poly = best_poly[1]

        # Trim tail to [560, target_len] so all carts across heterogeneous stock maps stay length-symmetric!
        seg_lens = [vlen(vsub(poly[i + 1], poly[i])) for i in range(len(poly) - 1)]
        tot = min(sum(seg_lens) or 1.0, target_len)
        sampled = []
        for k in range(5):
            target_d = tot * (k / 4.0)
            acc = 0.0
            pt = poly[-1]
            for i, sl in enumerate(seg_lens):
                if acc + sl >= target_d or i == len(seg_lens) - 1:
                    f = 0.0 if sl < 1e-6 else max(0.0, min(1.0, (target_d - acc) / sl))
                    a, b = poly[i], poly[i + 1]
                    pt = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f)
                    break
                acc += sl
            sampled.append(pt)
        tracks.append(sampled)
    return tracks


def sample_variant_specs(
    M: int,
    seed: int,
    base_pool: list[str] | None = None,
    include_nexus: bool = False,
    allow_repeats: bool = True,
) -> list[SubmapVariantSpec]:
    """Generate M diverse B3313-style stock-map variant & motif specifications."""
    rng = random.Random(seed)
    pool = list(base_pool or STOCK_MAP_CATALOG)
    pal_names = list(PALETTES.keys())
    econ_names = list(PICKUP_ECONOMIES.keys())

    specs: list[SubmapVariantSpec] = []
    # Keep first tile close to iconic stock feel, then progressively mutate subsequent tiles & repeats!
    for m in range(M):
        if include_nexus and m == 0 and M >= 4:
            specs.append(SubmapVariantSpec(base_map="b3313_nexus", seed=seed + m))
            continue
        base = pool[m % len(pool)] if not allow_repeats else pool[m % len(pool)]
        is_repeat = any(s.base_map == base for s in specs)
        if m == 0 and not include_nexus:
            # Canonical anchor room (unrotated stock feel + 1 light motif)
            specs.append(SubmapVariantSpec(
                base_map=base, rot_deg=0, mirror_axis="none", xy_scale=1.0, z_scale=1.0,
                palette="stock", motifs=["monolith_baffles"], pickup_economy="stock", seed=seed + m,
            ))
            continue
        rot = rng.choice([0, 90, 180, 270]) if (is_repeat or rng.random() < 0.75) else 0
        mirror = rng.choice(["x", "y"]) if (is_repeat or rng.random() < 0.35) else "none"
        xy_s = round(rng.uniform(1.0, 1.16), 2) if (is_repeat or rng.random() < 0.5) else 1.0
        z_s = round(rng.uniform(1.0, 1.14), 2) if rng.random() < 0.4 else 1.0
        pal = rng.choice(pal_names[1:]) if (is_repeat or rng.random() < 0.65) else "stock"
        n_motifs = rng.choice([1, 2, 3])
        motifs = rng.sample(list(ALL_MOTIFS), n_motifs)
        econ = rng.choice(econ_names[1:]) if "pickup_remix" in motifs else "stock"
        specs.append(SubmapVariantSpec(
            base_map=base, rot_deg=rot, mirror_axis=mirror, xy_scale=xy_s, z_scale=z_s,
            palette=pal, motifs=motifs, pickup_economy=econ, seed=seed + 37 * m,
        ))
    return specs


def build_b3313_supermap(
    name: str,
    specs: list[SubmapVariantSpec],
    *,
    teams: int,
    carts: int,
    joint_width_u: float,
    xonotic_dir: Path,
    out_dir: Path,
    seed: int = 101,
    compile_bsp: bool = True,
    fast_meta_only: bool = False,
    pk3_out: Path | None = None,
) -> dict:
    """Stitch M real stock Xonotic map variants into a single B3313 non-Euclidean supermap."""
    assert 2 <= teams <= 24, f"teams must be in [2, 24], got {teams}"
    assert carts > teams and carts <= 256, f"Need J > K and J <= 256, got K={teams}, J={carts}"
    M = len(specs)
    rng = random.Random(seed)

    # 1. Load unique stock assets (cached)
    assets: dict[str, StockMapAsset] = {}
    for sp in specs:
        if sp.base_map != "b3313_nexus" and sp.base_map not in assets:
            assets[sp.base_map] = load_stock_map(sp.base_map, xonotic_dir)

    # 2. Compute disjoint 3D atlas slots (keep well inside DarkPlaces +-24000 world bounds!)
    max_span_xy = 2200.0
    max_span_z = 1600.0
    for sp in specs:
        if sp.base_map == "b3313_nexus":
            continue
        a = assets[sp.base_map]
        sx = (a.bbox_max[0] - a.bbox_min[0]) * sp.xy_scale
        sy = (a.bbox_max[1] - a.bbox_min[1]) * sp.xy_scale
        sz = (a.bbox_max[2] - a.bbox_min[2]) * sp.z_scale
        max_span_xy = max(max_span_xy, sx, sy)
        max_span_z = max(max_span_z, sz)

    cell_xy = math.ceil((max_span_xy + 768.0) / 256.0) * 256.0
    cell_z = math.ceil((max_span_z + 768.0) / 256.0) * 256.0
    # Use 2D grid up to 4x4 (16 tiles), stack Z layers beyond 16 so coordinates stay compact!
    per_layer = min(M, 16)
    cols = max(1, math.ceil(math.sqrt(per_layer)))
    rows = max(1, math.ceil(per_layer / cols))

    tiles: list[dict] = []
    for m, sp in enumerate(specs):
        layer = m // 16
        in_l = m % 16
        cx, cy = in_l % cols, in_l // cols
        dx = (cx - 0.5 * (cols - 1)) * cell_xy
        dy = (cy - 0.5 * (rows - 1)) * cell_xy
        dz = layer * cell_z
        # Snap to 64u grid
        dst_off = (round(dx / 64.0) * 64.0, round(dy / 64.0) * 64.0, round(dz / 64.0) * 64.0)
        if sp.base_map == "b3313_nexus":
            tiles.append(build_nexus_tile(m, dst_off, rng))
        else:
            tiles.append(instantiate_stock_variant(assets[sp.base_map], sp, m, dst_off))

    # 3. Build per-tile walkable adjacency & non-Euclidean B3313 expander graph
    adjs = [_build_local_walk_adj(t) for t in tiles]
    exp_edges = build_expander_edges(M, rng, min_deg=3)

    # 4. Install B3313 Velocity-Preserving Portal-Membrane Joints along every expander edge!
    used_portals: list[list[tuple[float, float, float]]] = [[] for _ in range(M)]
    portal_entities: list[ParsedEntity] = []
    joints_meta: list[dict] = []

    for jid, (ta, tb) in enumerate(exp_edges):
        pa, qa, na, wpa, wqa = _pick_portal_anchor_and_exit(tiles[ta], adjs[ta], used_portals[ta], rng)
        used_portals[ta].append(pa)
        pb, qb, nb, wpb, wqb = _pick_portal_anchor_and_exit(tiles[tb], adjs[tb], used_portals[tb], rng)
        used_portals[tb].append(pb)

        dst_in_b = f"b3313_j{jid}_to_b"
        dst_in_a = f"b3313_j{jid}_to_a"
        yaw_a = math.degrees(math.atan2(na[1], na[0])) % 360.0
        yaw_b = math.degrees(math.atan2(nb[1], nb[0])) % 360.0

        # A -> B trigger_teleport (spawnflags 2 = TELEPORT_KEEP_SPEED) + visual arch + multi-pad anti-telefrag exits in B
        trig_a = _make_portal_trigger_brush(pa, na, joint_width_u)
        portal_entities.append(ParsedEntity(
            kv={"classname": "trigger_teleport", "target": dst_in_b, "spawnflags": "2"},
            brushes=[trig_a],
        ))
        portal_entities.append(ParsedEntity(
            kv={"classname": "func_illusionary"},
            brushes=_make_portal_arch_illusionary(pa, na, joint_width_u),
        ))
        # Emit up to 3 verified walkable exit pads near qb (leveraging native TELEPORT_TELEFRAG_AVOID)
        b_exits = [qb]
        for cand_w in adjs[tb].get(wqb, []):
            cq = tiles[tb]["wp_centers"][cand_w]
            if vlen(vsub(cq, pb)) >= 115.0 and all(vlen(vsub(cq, ex)) >= 64.0 for ex in b_exits):
                b_exits.append(cq)
                if len(b_exits) >= 3:
                    break
        for ex_p in b_exits:
            portal_entities.append(ParsedEntity(
                kv={"classname": "info_teleport_destination", "targetname": dst_in_b,
                    "origin": f"{ex_p[0]:.1f} {ex_p[1]:.1f} {ex_p[2] + 16.0:.1f}", "angle": f"{yaw_b:.1f}"},
                brushes=[],
            ))

        # B -> A trigger_teleport (TELEPORT_KEEP_SPEED) + visual arch + multi-pad anti-telefrag exits in A
        trig_b = _make_portal_trigger_brush(pb, nb, joint_width_u)
        portal_entities.append(ParsedEntity(
            kv={"classname": "trigger_teleport", "target": dst_in_a, "spawnflags": "2"},
            brushes=[trig_b],
        ))
        portal_entities.append(ParsedEntity(
            kv={"classname": "func_illusionary"},
            brushes=_make_portal_arch_illusionary(pb, nb, joint_width_u),
        ))
        a_exits = [qa]
        for cand_w in adjs[ta].get(wqa, []):
            cq = tiles[ta]["wp_centers"][cand_w]
            if vlen(vsub(cq, pa)) >= 115.0 and all(vlen(vsub(cq, ex)) >= 64.0 for ex in a_exits):
                a_exits.append(cq)
                if len(a_exits) >= 3:
                    break
        for ex_p in a_exits:
            portal_entities.append(ParsedEntity(
                kv={"classname": "info_teleport_destination", "targetname": dst_in_a,
                    "origin": f"{ex_p[0]:.1f} {ex_p[1]:.1f} {ex_p[2] + 16.0:.1f}", "angle": f"{yaw_a:.1f}"},
                brushes=[],
            ))

        joints_meta.append({
            "id": jid,
            "type": "portal",
            "tile_a": ta, "tile_b": tb,
            "origin": [round(pa[0], 1), round(pa[1], 1), round(pa[2], 1)],
            "normal": [round(na[0], 4), round(na[1], 4), 0.0],
            "origin_b": [round(pb[0], 1), round(pb[1], 1), round(pb[2], 1)],
            "normal_b": [round(nb[0], 4), round(nb[1], 4), 0.0],
            "width": float(joint_width_u),
            "length": 48.0,
            "height": 128.0,
        })

    # 5. Allocate J > K balanced cartpaths across the M real stock submaps
    base_j, extra_j = divmod(carts, M)
    all_tracks: list[list[tuple[float, float, float]]] = []
    for m in range(M):
        jm = base_j + (1 if m < extra_j else 0)
        all_tracks.extend(_trace_cart_tracks_on_tile(tiles[m], adjs[m], jm, rng, used_portals[m]))

    payload_entities: list[ParsedEntity] = []
    # Anchor point entities (plc_team) inside tile 0's sealed walkable interior so q3map2 never sees (0,0,0) in void
    t0_anchor = tiles[0]["wp_centers"][tiles[0]["walk_ids"][0]]
    t0_org = f"{t0_anchor[0]:.1f} {t0_anchor[1]:.1f} {t0_anchor[2]:.1f}"
    for t_idx in range(teams):
        payload_entities.append(ParsedEntity(kv={"classname": "plc_team", "cnt": str(t_idx), "origin": t0_org}, brushes=[]))

    for ci, nodes in enumerate(all_tracks):
        names = [f"plc{ci}n{ni}" for ni in range(len(nodes))]
        for ni, p in enumerate(nodes):
            kv = {"classname": "plc_path", "targetname": names[ni], "origin": f"{p[0]:.1f} {p[1]:.1f} {p[2]:.1f}"}
            if ni + 1 < len(nodes):
                kv["target"] = names[ni + 1]
            if ni > 0:
                kv["spawnflags"] = "1"
            payload_entities.append(ParsedEntity(kv=kv, brushes=[]))
        payload_entities.append(ParsedEntity(kv={
            "classname": "func_plc_cart",
            "origin": f"{nodes[0][0]:.1f} {nodes[0][1]:.1f} {nodes[0][2]:.1f}",
            "target": names[0], "radius": "160", "height": "96", "plc_speed_scale": "1",
        }, brushes=[]))

    # 6. Allocate K spatially disjoint team spawn bases using verified stock spawns + walkable nodes!
    spawn_records: list[tuple[int, tuple[float, float, float], float]] = []
    team_anchors: list[tuple[float, float, float]] = []
    for team_idx in range(teams):
        tid = team_idx % M
        tile = tiles[tid]
        # Candidate pool: verified stock spawns first, then walkable floor waypoints
        cand_pts = [(p, ang) for p, ang in tile["stock_spawns"]]
        for wi in tile["walk_ids"]:
            p = tile["wp_centers"][wi]
            cand_pts.append(((p[0], p[1], p[2] + 8.0), 0.0))
        if team_anchors:
            anchor_p, anchor_ang = max(
                cand_pts,
                key=lambda item: min(vlen(vsub(item[0], u)) for u in team_anchors),
            )
        else:
            anchor_p, anchor_ang = cand_pts[0]
        team_anchors.append(anchor_p)
        # Pick up to 4 verified walkable spots near this team's anchor (>= 320u from enemy anchors)
        near = sorted(
            cand_pts,
            key=lambda item: vlen(vsub(item[0], anchor_p)),
        )
        chosen = []
        for p, ang in near:
            if len(chosen) >= 4:
                break
            if any(vlen(vsub(p, c[0])) < 64.0 for c in chosen):
                continue
            if any(vlen(vsub(p, ea)) < 320.0 for ea in team_anchors[:-1]):
                continue
            chosen.append((p, ang))
        if not chosen:
            chosen = [(anchor_p, anchor_ang)]
        for p, ang in chosen:
            spawn_records.append((team_idx + 1, p, ang))
            payload_entities.append(ParsedEntity(kv={
                "classname": "info_player_team",
                "origin": f"{p[0]:.1f} {p[1]:.1f} {p[2]:.1f}",
                "angle": f"{ang:.1f}",
                "team": str(team_idx + 1),
            }, brushes=[]))

    # 7. Merge all world brushes, entities, waypoints, and waypoint cache edges
    out_dir.mkdir(parents=True, exist_ok=True)
    map_lines = [
        f"// B3313 Real-Map Variant Supermap: {name} K={teams} J={carts} M={M} W={joint_width_u} seed={seed}",
        "{",
        '"classname" "worldspawn"',
        '"_minlight" "28"',
        '"_ambient" "24"',
        f'"message" "{name}"',
    ]
    total_brushes = 0
    for tile in tiles:
        for br in tile["brushes"]:
            map_lines.append("{")
            map_lines.append(br)
            map_lines.append("}")
            total_brushes += 1
        # 6-brush hermetic outer containment hull (96u outside tile AABB) guarantees 0 leaks under arbitrary affine scales
        mn, mx = tile["aabb_min"], tile["aabb_max"]
        pad, th = 96.0, 64.0
        hx0, hy0, hz0 = mn[0] - pad, mn[1] - pad, mn[2] - pad
        hx1, hy1, hz1 = mx[0] + pad, mx[1] + pad, mx[2] + pad
        for hbr in (
            _box_block(hx0 - th, hy0 - th, hz0 - th, hx1 + th, hy1 + th, hz0, "exx/base-metal03"),
            _box_block(hx0 - th, hy0 - th, hz1, hx1 + th, hy1 + th, hz1 + th, "exx/base-metal03"),
            _box_block(hx0 - th, hy0 - th, hz0, hx0, hy1 + th, hz1, "exx/base-metal03"),
            _box_block(hx1, hy0 - th, hz0, hx1 + th, hy1 + th, hz1, "exx/base-metal03"),
            _box_block(hx0, hy0 - th, hz0, hx1, hy0, hz1, "exx/base-metal03"),
            _box_block(hx0, hy1, hz0, hx1, hy1 + th, hz1, "exx/base-metal03"),
        ):
            map_lines.append("{")
            map_lines.append(hbr)
            map_lines.append("}")
            total_brushes += 1
    map_lines.append("}")

    all_ents: list[ParsedEntity] = []
    for tile in tiles:
        all_ents.extend(tile["entities"])
    all_ents.extend(portal_entities)
    all_ents.extend(payload_entities)

    for ent in all_ents:
        map_lines.append("{")
        for k, v in ent.kv.items():
            map_lines.append(f'"{k}" "{v}"')
        for br in ent.brushes:
            map_lines.append("{")
            map_lines.append(br)
            map_lines.append("}")
        map_lines.append("}")

    map_path = out_dir / f"{name}.map"
    map_path.write_text("\n".join(map_lines) + "\n")

    (out_dir / f"{name}.mapinfo").write_text(
        f"title {name}\n"
        f"description B3313 Supermap ({M} stock-map variants, K={teams}, J={carts}, W={joint_width_u}).\n"
        f"author b3313_supermap\nhas weapons\ngametype plc\ngametype dm\ngametype tdm\n"
    )

    # Merged transformed waypoints + cache
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    wl = ["//WAYPOINT_VERSION 1.04", "//WAYPOINT_SYMMETRY 0", f"//WAYPOINT_TIME {ts}"]
    cl = ["//WAYPOINT_VERSION 1.04", f"//WAYPOINT_TIME {ts}"]
    tot_wps = 0
    tot_edges = 0
    for tile in tiles:
        for mn, mx, fl in tile["waypoints"]:
            wl.append(f"'{mn[0]:.1f} {mn[1]:.1f} {mn[2]:.1f}'")
            wl.append(f"'{mx[0]:.1f} {mx[1]:.1f} {mx[2]:.1f}'")
            wl.append(str(int(fl)))
            tot_wps += 1
        pts = tile["wp_centers"]
        edges_use = set(tile["wp_edges"])
        for a, nbs in adjs[tile["tile_id"]].items():
            for b in nbs:
                edges_use.add((min(a, b), max(a, b)))
        for a, b in sorted(edges_use):
            pa, pb = pts[a], pts[b]
            sa = f"'{pa[0]:.1f} {pa[1]:.1f} {pa[2]:.1f}'"
            sb = f"'{pb[0]:.1f} {pb[1]:.1f} {pb[2]:.1f}'"
            cl.append(f"{sa}*{sb}")
            cl.append(f"{sb}*{sa}")
            tot_edges += 1

    (out_dir / f"{name}.waypoints").write_text("\n".join(wl) + "\n")
    (out_dir / f"{name}.waypoints.cache").write_text("\n".join(cl) + "\n")

    # Estimate walkable area per tile from waypoint convex hull / bounding span (or ~3.5 Mqu^2 stock median)
    tiles_meta = []
    deg_map = {m: 0 for m in range(M)}
    for jt in joints_meta:
        deg_map[jt["tile_a"]] += 1
        deg_map[jt["tile_b"]] += 1
    for tile in tiles:
        mn, mx = tile["aabb_min"], tile["aabb_max"]
        # Estimate walkable footprint from walkable waypoints span (clamped to realistic stock floor range)
        wpts = [tile["wp_centers"][i] for i in tile["walk_ids"]]
        if wpts:
            wx = max(p[0] for p in wpts) - min(p[0] for p in wpts) + 256.0
            wy = max(p[1] for p in wpts) - min(p[1] for p in wpts) + 256.0
            area_mqu2 = round(max(1.5, min(8.0, (wx * wy * 0.55) / 1e6)), 3)
        else:
            area_mqu2 = 3.5
        tiles_meta.append({
            "id": tile["tile_id"],
            "base_map": tile["base_map"],
            "variant_label": tile["variant_label"],
            "spec": tile["spec"],
            "x0": round(mn[0], 1), "y0": round(mn[1], 1), "z0": round(mn[2], 1),
            "x1": round(mx[0], 1), "y1": round(mx[1], 1), "z1": round(mx[2], 1),
            "center": [round(0.5 * (mn[0] + mx[0]), 1), round(0.5 * (mn[1] + mx[1]), 1), round(0.5 * (mn[2] + mx[2]), 1)],
            "area_mqu2": area_mqu2,
            "degree": deg_map[tile["tile_id"]],
        })

    joints_doc = {
        "name": name, "seed": seed, "kind": "b3313_supermap",
        "teams": teams, "carts": carts, "tiles_count": M,
        "joint_width_u": float(joint_width_u),
        "total_walkable_area_mqu2": round(sum(t["area_mqu2"] for t in tiles_meta), 3),
        "tile_degrees": deg_map,
        "tiles": tiles_meta,
        "joints": joints_meta,
    }
    (out_dir / f"{name}.joints.json").write_text(json.dumps(joints_doc, indent=2))

    lens = [round(sum(vlen(vsub(tr[i + 1], tr[i])) for i in range(len(tr) - 1)), 1) for tr in all_tracks]
    cross_d = [
        vlen(vsub(p1, p2))
        for i, (t1, p1, _) in enumerate(spawn_records)
        for t2, p2, _ in spawn_records[i + 1:]
        if t1 != t2
    ]
    meta = {
        "name": name, "seed": seed, "teams": teams, "carts": carts, "tiles": M,
        "variants": [t["variant_label"] for t in tiles_meta],
        "joints": len(joints_meta), "joint_width_u": float(joint_width_u),
        "min_tile_degree": min(deg_map.values()) if deg_map else 0,
        "max_tile_degree": max(deg_map.values()) if deg_map else 0,
        "total_brushes": total_brushes, "total_entities": len(all_ents),
        "waypoints": tot_wps, "waypoint_edges": tot_edges,
        "cart_len_min": min(lens) if lens else 0.0,
        "cart_len_max": max(lens) if lens else 0.0,
        "path_length_ratio": round(max(lens) / max(min(lens), 1.0), 3) if lens else 1.0,
        "min_cross_team_spawn_dist": round(min(cross_d), 1) if cross_d else None,
    }

    if compile_bsp:
        q3 = Path(__file__).resolve().parents[2] / "bin/mesh-q3map2"
        fs_home = map_path.parent.parent.parent
        base = [str(q3), "-game", "xonotic", "-fs_basepath", str(xonotic_dir), "-fs_homepath", str(fs_home)]
        t0 = time.monotonic()
        r_meta = subprocess.run(base + ["-meta", str(map_path)], capture_output=True, text=True)
        if r_meta.returncode != 0 or "LEAKED" in (r_meta.stdout + r_meta.stderr).upper():
            raise RuntimeError(f"q3map2 -meta failed/leaked on {name}:\n{(r_meta.stdout + r_meta.stderr)[-800:]}")
        if not fast_meta_only:
            r_vis = subprocess.run(base + ["-vis", "-fast", str(map_path)], capture_output=True, text=True)
            if r_vis.returncode != 0:
                raise RuntimeError(f"q3map2 -vis failed on {name}:\n{(r_vis.stdout + r_vis.stderr)[-400:]}")
        bsp = map_path.with_suffix(".bsp")
        meta["compile_s"] = round(time.monotonic() - t0, 2)
        meta["bsp_kb"] = round(bsp.stat().st_size / 1024.0, 1)
        if pk3_out is not None:
            pk3_out.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(pk3_out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
                for ext in (".bsp", ".mapinfo", ".waypoints", ".waypoints.cache"):
                    f = map_path.with_suffix(ext)
                    if f.exists():
                        zf.write(f, f"maps/{f.name}")

    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser(description="B3313 Real-Map Variant Non-Euclidean Supermap Stitcher")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="/tmp/b3313_build/data/maps")
    ap.add_argument("--xonotic-dir", default="/Users/mdot/dox/xonotic/Xonotic")
    ap.add_argument("--teams", type=int, default=6)
    ap.add_argument("--carts", type=int, default=24)
    ap.add_argument("--tiles", type=int, default=4)
    ap.add_argument("--maps", default="", help="Comma-separated stock base maps (optional)")
    ap.add_argument("--include-nexus", action="store_true")
    ap.add_argument("--joint-width", type=float, default=320.0)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--sample", action="store_true", help="Sample (K, J, N, M) from gas-law distribution")
    ap.add_argument("--fast-meta-only", action="store_true", help="Skip -vis for ultra-fast dedicated server builds")
    ap.add_argument("--pk3-out", default="")
    args = ap.parse_args()

    K, J, M = args.teams, args.carts, args.tiles
    if args.sample:
        d = sample_gas_world_params(args.seed)
        K, J, M = d["teams"], d["carts"], d["tiles"]

    pool = [s.strip() for s in args.maps.split(",") if s.strip()] if args.maps else None
    specs = sample_variant_specs(M, seed=args.seed, base_pool=pool, include_nexus=args.include_nexus)
    meta = build_b3313_supermap(
        args.name, specs,
        teams=K, carts=J, joint_width_u=args.joint_width,
        xonotic_dir=Path(args.xonotic_dir), out_dir=Path(args.out_dir),
        seed=args.seed, compile_bsp=True, fast_meta_only=args.fast_meta_only,
        pk3_out=Path(args.pk3_out) if args.pk3_out else None,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
