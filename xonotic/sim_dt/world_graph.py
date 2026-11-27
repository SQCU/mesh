from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class WorldGraphIR:
    name: str
    teams: int
    carts: int
    players_default: int
    wp_pos: torch.Tensor          # [V, 3] float32
    wp_dist: torch.Tensor         # [V, V] float32
    wp_next_hop: torch.Tensor     # [V, V] int64
    wp_hop_len: torch.Tensor      # [V, V] float32
    cart_nodes: torch.Tensor      # [J, S_max, 3] float32
    cart_node_s: torch.Tensor     # [J, S_max] float32
    cart_seg_count: torch.Tensor  # [J] int64
    cart_length: torch.Tensor     # [J] float32
    cart_cp_dists: torch.Tensor   # [J, CP_max] float32 (padded with +inf)
    cart_cp_count: torch.Tensor   # [J] int64
    cart_radius: torch.Tensor     # [J] float32
    cart_height: torch.Tensor     # [J] float32
    cart_speed_scale: torch.Tensor  # [J] float32
    cart_sample_wp: torch.Tensor  # [J, M_s] int64 (nearest wp along track)
    spawn_pos: torch.Tensor       # [K, S_sp, 3] float32
    spawn_wp: torch.Tensor        # [K, S_sp] int64
    map_center: torch.Tensor      # [3] float32
    map_radius: torch.Tensor      # [] float32

    @property
    def num_waypoints(self) -> int:
        return int(self.wp_pos.shape[0])

    @property
    def max_checkpoints(self) -> int:
        return int(self.cart_cp_dists.shape[1])

    @property
    def device(self) -> torch.device:
        return self.wp_pos.device

    def to(self, device: torch.device | str) -> "WorldGraphIR":
        dev = torch.device(device)
        return WorldGraphIR(
            name=self.name,
            teams=self.teams,
            carts=self.carts,
            players_default=self.players_default,
            wp_pos=self.wp_pos.to(dev),
            wp_dist=self.wp_dist.to(dev),
            wp_next_hop=self.wp_next_hop.to(dev),
            wp_hop_len=self.wp_hop_len.to(dev),
            cart_nodes=self.cart_nodes.to(dev),
            cart_node_s=self.cart_node_s.to(dev),
            cart_seg_count=self.cart_seg_count.to(dev),
            cart_length=self.cart_length.to(dev),
            cart_cp_dists=self.cart_cp_dists.to(dev),
            cart_cp_count=self.cart_cp_count.to(dev),
            cart_radius=self.cart_radius.to(dev),
            cart_height=self.cart_height.to(dev),
            cart_speed_scale=self.cart_speed_scale.to(dev),
            cart_sample_wp=self.cart_sample_wp.to(dev),
            spawn_pos=self.spawn_pos.to(dev),
            spawn_wp=self.spawn_wp.to(dev),
            map_center=self.map_center.to(dev),
            map_radius=self.map_radius.to(dev),
        )


def _grid_dims(n_tiles: int) -> tuple[int, int]:
    cols = max(1, int(math.ceil(math.sqrt(max(1, n_tiles)))))
    rows = max(1, int(math.ceil(max(1, n_tiles) / float(cols))))
    return cols, rows


def _compute_apsp(wp_pos_np: np.ndarray, edges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = len(wp_pos_np)
    dist = np.full((v, v), 1e9, dtype=np.float32)
    nxt = np.broadcast_to(np.arange(v, dtype=np.int64)[None, :], (v, v)).copy()
    np.fill_diagonal(dist, 0.0)
    for a, b in edges:
        if 0 <= a < v and 0 <= b < v and a != b:
            d = float(np.linalg.norm(wp_pos_np[b] - wp_pos_np[a]))
            if d < dist[a, b]:
                dist[a, b] = d
                nxt[a, b] = b
            if d < dist[b, a]:
                dist[b, a] = d
                nxt[b, a] = a
    # Ensure full connectivity by connecting disconnected components via nearest bridge
    if v > 1:
        for k in range(v):
            dk = dist[:, k:k + 1] + dist[k:k + 1, :]
            better = dk < dist
            if np.any(better):
                dist = np.where(better, dk, dist)
                nk = np.broadcast_to(nxt[:, k:k + 1], (v, v))
                nxt = np.where(better, nk, nxt)
        # Fallback direct hop for any unreachable pair
        unreach = dist >= 1e8
        if np.any(unreach):
            euclid = np.linalg.norm(wp_pos_np[:, None, :] - wp_pos_np[None, :, :], axis=-1).astype(np.float32)
            dist = np.where(unreach, euclid, dist)
            dst_idx = np.broadcast_to(np.arange(v, dtype=np.int64)[None, :], (v, v))
            nxt = np.where(unreach, dst_idx, nxt)
    hop_pos = wp_pos_np[nxt]
    hop_len = np.linalg.norm(hop_pos - wp_pos_np[:, None, :], axis=-1).astype(np.float32)
    np.fill_diagonal(hop_len, 0.0)
    return dist, nxt, hop_len


def _compile_world_graph(
    name: str,
    teams: int,
    players_default: int,
    wp_pts: list[tuple[float, float, float]],
    edges: list[tuple[int, int]],
    cart_tracks: list[dict[str, Any]],
    team_spawns: list[list[tuple[float, float, float]]],
    *,
    sample_bins: int = 16,
    device: torch.device | str = "cpu",
) -> WorldGraphIR:
    wp_pos_np = np.asarray(wp_pts, dtype=np.float32)
    if wp_pos_np.ndim != 2 or wp_pos_np.shape[0] == 0:
        wp_pos_np = np.zeros((1, 3), dtype=np.float32)
    v = wp_pos_np.shape[0]
    dist_np, nxt_np, hop_len_np = _compute_apsp(wp_pos_np, edges)

    J = len(cart_tracks)
    s_max = max(2, max((len(ct["nodes"]) for ct in cart_tracks), default=2))
    cp_max = max(1, max((len(ct["checkpoints"]) for ct in cart_tracks), default=1))

    cart_nodes_np = np.zeros((J, s_max, 3), dtype=np.float32)
    cart_node_s_np = np.full((J, s_max), 1e9, dtype=np.float32)
    cart_seg_count_np = np.zeros((J,), dtype=np.int64)
    cart_length_np = np.zeros((J,), dtype=np.float32)
    cart_cp_dists_np = np.full((J, cp_max), 1e9, dtype=np.float32)
    cart_cp_count_np = np.zeros((J,), dtype=np.int64)
    cart_radius_np = np.full((J,), 160.0, dtype=np.float32)
    cart_height_np = np.full((J,), 96.0, dtype=np.float32)
    cart_speed_scale_np = np.ones((J,), dtype=np.float32)
    cart_sample_wp_np = np.zeros((J, sample_bins), dtype=np.int64)

    for ci, ct in enumerate(cart_tracks):
        nodes = np.asarray(ct["nodes"], dtype=np.float32)
        n_nodes = len(nodes)
        cum_s = np.zeros((n_nodes,), dtype=np.float32)
        for ni in range(1, n_nodes):
            cum_s[ni] = cum_s[ni - 1] + float(np.linalg.norm(nodes[ni] - nodes[ni - 1]))
        cart_nodes_np[ci, :n_nodes] = nodes
        cart_nodes_np[ci, n_nodes:] = nodes[-1]
        cart_node_s_np[ci, :n_nodes] = cum_s
        cart_node_s_np[ci, n_nodes:] = cum_s[-1]
        cart_seg_count_np[ci] = max(1, n_nodes - 1)
        cart_length_np[ci] = max(1.0, float(cum_s[-1]))
        cps = sorted(float(x) for x in ct["checkpoints"])
        cart_cp_count_np[ci] = len(cps)
        if cps:
            cart_cp_dists_np[ci, :len(cps)] = np.asarray(cps, dtype=np.float32)
        cart_radius_np[ci] = float(ct.get("radius", 160.0))
        cart_height_np[ci] = float(ct.get("height", 96.0))
        cart_speed_scale_np[ci] = float(ct.get("speed_scale", 1.0))

        # Precompute nearest waypoint at uniform arc-length bins along track
        for bi in range(sample_bins):
            target_s = cart_length_np[ci] * (bi / max(1, sample_bins - 1))
            seg_i = int(np.searchsorted(cum_s, target_s, side="right")) - 1
            seg_i = max(0, min(n_nodes - 2, seg_i))
            seg_len = max(1e-6, float(cum_s[seg_i + 1] - cum_s[seg_i]))
            frac = max(0.0, min(1.0, (target_s - float(cum_s[seg_i])) / seg_len))
            pt = nodes[seg_i] + frac * (nodes[seg_i + 1] - nodes[seg_i])
            d2 = np.sum((wp_pos_np - pt[None, :]) ** 2, axis=-1)
            cart_sample_wp_np[ci, bi] = int(np.argmin(d2))

    K = int(teams)
    sp_max = max(1, max((len(sp) for sp in team_spawns), default=1))
    spawn_pos_np = np.zeros((K, sp_max, 3), dtype=np.float32)
    spawn_wp_np = np.zeros((K, sp_max), dtype=np.int64)
    for ti in range(K):
        pts = team_spawns[ti] if ti < len(team_spawns) and team_spawns[ti] else [tuple(wp_pos_np[ti % v])]
        for si in range(sp_max):
            p = np.asarray(pts[si % len(pts)], dtype=np.float32)
            spawn_pos_np[ti, si] = p
            d2 = np.sum((wp_pos_np - p[None, :]) ** 2, axis=-1)
            spawn_wp_np[ti, si] = int(np.argmin(d2))

    all_pts = np.concatenate(
        [wp_pos_np, cart_nodes_np.reshape(-1, 3), spawn_pos_np.reshape(-1, 3)],
        axis=0,
    )
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    center = 0.5 * (mn + mx)
    radius = max(256.0, float(0.5 * np.linalg.norm(mx - mn)))

    dev = torch.device(device)
    return WorldGraphIR(
        name=name,
        teams=K,
        carts=J,
        players_default=int(players_default),
        wp_pos=torch.from_numpy(wp_pos_np).to(dev),
        wp_dist=torch.from_numpy(dist_np).to(dev),
        wp_next_hop=torch.from_numpy(nxt_np).to(dev),
        wp_hop_len=torch.from_numpy(hop_len_np).to(dev),
        cart_nodes=torch.from_numpy(cart_nodes_np).to(dev),
        cart_node_s=torch.from_numpy(cart_node_s_np).to(dev),
        cart_seg_count=torch.from_numpy(cart_seg_count_np).to(dev),
        cart_length=torch.from_numpy(cart_length_np).to(dev),
        cart_cp_dists=torch.from_numpy(cart_cp_dists_np).to(dev),
        cart_cp_count=torch.from_numpy(cart_cp_count_np).to(dev),
        cart_radius=torch.from_numpy(cart_radius_np).to(dev),
        cart_height=torch.from_numpy(cart_height_np).to(dev),
        cart_speed_scale=torch.from_numpy(cart_speed_scale_np).to(dev),
        cart_sample_wp=torch.from_numpy(cart_sample_wp_np).to(dev),
        spawn_pos=torch.from_numpy(spawn_pos_np).to(dev),
        spawn_wp=torch.from_numpy(spawn_wp_np).to(dev),
        map_center=torch.from_numpy(center.astype(np.float32)).to(dev),
        map_radius=torch.tensor(radius, dtype=torch.float32, device=dev),
    )


def build_gas_world_graph(
    *,
    map_name: str = "gas_dt",
    seed: int = 1,
    teams: int = 4,
    carts: int = 12,
    players: int = 32,
    tiles: int = 4,
    tile_size: float = 1792.0,
    joint_width: float = 320.0,
    joint_length: float = 96.0,
    checkpoints_per_cart: int = 4,
    device: torch.device | str = "cpu",
) -> WorldGraphIR:
    """Exact topological twin of bsp_stream.build_gas_world_bsp_in_memory without IBSP compilation."""
    rng = random.Random(seed)
    S = float(tile_size)
    wall_t = float(joint_length)
    K = max(2, min(64, int(teams)))
    J = max(K + 1, min(256, int(carts)))
    M_req = max(1, min(24, int(tiles)))
    N = max(max(2 * K, 12), int(players))
    cols, rows = _grid_dims(M_req)
    M = cols * rows
    pitch = S + wall_t

    tiles_meta = []
    for cy in range(rows):
        for cx in range(cols):
            x0, y0 = cx * pitch, cy * pitch
            tiles_meta.append({"id": cy * cols + cx, "x0": x0, "y0": y0, "center": (x0 + 0.5 * S, y0 + 0.5 * S, 0.0)})

    joints_meta = []
    jid = 0
    for cy in range(rows):
        for cx in range(cols - 1):
            a_id, b_id = cy * cols + cx, cy * cols + cx + 1
            y0, xw0, y1 = cy * pitch, cx * pitch + S, cy * pitch + S
            xw1 = xw0 + wall_t
            dual = (rows == 1 and cols >= 2) or (cy in (0, rows - 1) and cx in (0, cols - 2) and M >= 4)
            centers = [y0 + S * 0.30, y0 + S * 0.70] if dual else [0.5 * (y0 + y1)]
            for myc in centers:
                joints_meta.append({"id": jid, "tile_a": a_id, "tile_b": b_id, "origin": (0.5 * (xw0 + xw1), myc, 0.0), "normal": (1.0, 0.0, 0.0)})
                jid += 1

    for cy in range(rows - 1):
        for cx in range(cols):
            a_id, b_id = cy * cols + cx, (cy + 1) * cols + cx
            x0, x1, yw0 = cx * pitch, cx * pitch + S, cy * pitch + S
            yw1 = yw0 + wall_t
            dual = cols == 1 and rows >= 2
            centers = [x0 + S * 0.30, x0 + S * 0.70] if dual else [0.5 * (x0 + x1)]
            for mxc in centers:
                joints_meta.append({"id": jid, "tile_a": a_id, "tile_b": b_id, "origin": (mxc, 0.5 * (yw0 + yw1), 0.0), "normal": (0.0, 1.0, 0.0)})
                jid += 1

    wp_pts: list[tuple[float, float, float]] = []
    edges: list[tuple[int, int]] = []
    tile_wps: dict[int, list[int]] = {}

    def _add_wp(x: float, y: float, z: float = 28.0) -> int:
        wp_pts.append((round(x, 1), round(y, 1), round(z, 1)))
        return len(wp_pts) - 1

    for t in tiles_meta:
        mat = []
        margin = 160.0
        for iy in range(5):
            row = []
            py = t["y0"] + margin + (S - 2.0 * margin) * iy / 4.0
            for ix in range(5):
                px = t["x0"] + margin + (S - 2.0 * margin) * ix / 4.0
                if ix == 2 and iy == 2:
                    px += 96.0
                row.append(_add_wp(px, py, 28.0))
            mat.append(row)
        for iy in range(5):
            for ix in range(5):
                if ix + 1 < 5:
                    edges.append((mat[iy][ix], mat[iy][ix + 1]))
                if iy + 1 < 5:
                    edges.append((mat[iy][ix], mat[iy + 1][ix]))
                if ix + 1 < 5 and iy + 1 < 5:
                    edges.append((mat[iy][ix], mat[iy + 1][ix + 1]))
        tile_wps[t["id"]] = [w for r in mat for w in r]

    for jt in joints_meta:
        ox, oy, _ = jt["origin"]
        nx, ny, _ = jt["normal"]
        off = 0.5 * wall_t + 120.0
        wa = _add_wp(ox - nx * off, oy - ny * off, 28.0)
        wc = _add_wp(ox, oy, 28.0)
        wb = _add_wp(ox + nx * off, oy + ny * off, 28.0)
        edges.append((wa, wc))
        edges.append((wc, wb))
        for wp_end, tid in ((wa, jt["tile_a"]), (wb, jt["tile_b"])):
            cands = sorted(tile_wps[tid], key=lambda w: math.hypot(wp_pts[w][0] - wp_pts[wp_end][0], wp_pts[w][1] - wp_pts[wp_end][1]))
            for nb in cands[:2]:
                edges.append((wp_end, nb))

    corners = [(220.0, 220.0), (S - 220.0, S - 220.0), (220.0, S - 220.0), (S - 220.0, 220.0)]
    team_spawns: list[list[tuple[float, float, float]]] = []
    for t_idx in range(K):
        tm = tiles_meta[t_idx % M]
        ax, ay = corners[(t_idx // M) % len(corners)]
        bx, by = tm["x0"] + ax, tm["y0"] + ay
        team_spawns.append([
            (bx + dx, by + dy, 32.0)
            for dx, dy in ((-48.0, -48.0), (48.0, -48.0), (-48.0, 48.0), (48.0, 48.0))
        ])

    tracks_endpoints: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    n_cross = min(len(joints_meta), int(J * 0.35)) if M > 1 else 0
    j_ord = list(joints_meta)
    rng.shuffle(j_ord)
    for jt in j_ord[:n_cross]:
        ox, oy, _ = jt["origin"]
        nx, ny, _ = jt["normal"]
        sgn = 1.0 if (jt["id"] % 2 == 0) else -1.0
        tracks_endpoints.append(((ox - sgn * nx * 420.0, oy - sgn * ny * 420.0, 24.0), (ox + sgn * nx * 420.0, oy + sgn * ny * 420.0, 24.0)))
    lane_tpls = [
        ((240, 320), (1040, 320)), ((1552, 1472), (752, 1472)), ((320, 1552), (320, 752)), ((1472, 240), (1472, 1040)),
        ((540, 620), (1252, 620)), ((1252, 1172), (540, 1172)), ((620, 1252), (620, 540)), ((1172, 540), (1172, 1252)),
    ]
    for k in range(J - len(tracks_endpoints)):
        tm = tiles_meta[k % M]
        slot = k // M
        (ax, ay), (bx, by) = lane_tpls[slot % len(lane_tpls)]
        sh = (slot // len(lane_tpls)) * 48.0
        tracks_endpoints.append(((tm["x0"] + ax + sh, tm["y0"] + ay, 24.0), (tm["x0"] + bx + sh, tm["y0"] + by, 24.0)))

    ncp = max(1, int(checkpoints_per_cart))
    cart_tracks = []
    for p0, p1 in tracks_endpoints:
        nodes = [
            (p0[0] + (p1[0] - p0[0]) * k / float(ncp), p0[1] + (p1[1] - p0[1]) * k / float(ncp), 24.0)
            for k in range(ncp + 1)
        ]
        total_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        cps = [total_len * k / float(ncp) for k in range(1, ncp + 1)]
        cart_tracks.append({"nodes": nodes, "checkpoints": cps, "radius": 160.0, "height": 96.0, "speed_scale": 1.0})

    return _compile_world_graph(
        map_name,
        K,
        N,
        wp_pts,
        edges,
        cart_tracks,
        team_spawns,
        device=device,
    )


def world_graph_from_vfs_bundle(
    spec: dict[str, Any],
    vfs_files: dict[str, bytes],
    *,
    device: torch.device | str = "cpu",
) -> WorldGraphIR:
    """Compile any StreamedWorldBundle (gas_world_bsp or b3313_binary_stitch) into a WorldGraphIR."""
    map_name = str(spec["name"])
    ent_text = vfs_files[f"maps/{map_name}.ent"].decode("utf-8", errors="ignore")
    wp_text = vfs_files.get(f"maps/{map_name}.waypoints", b"").decode("utf-8", errors="ignore")

    ents: list[dict[str, str]] = []
    for blk in re.findall(r"\{[^{}]*\}", ent_text):
        kv = dict(re.findall(r'"([^"]+)"\s+"([^"]*)"', blk))
        if kv:
            ents.append(kv)

    def _p3(s: str) -> tuple[float, float, float]:
        ps = [float(x) for x in s.split()]
        ps = (ps + [0.0, 0.0, 0.0])[:3]
        return (ps[0], ps[1], ps[2])

    wp_pts: list[tuple[float, float, float]] = []
    lines = [ln.strip() for ln in wp_text.splitlines() if ln.strip()]
    i = 0
    while i + 2 < len(lines):
        try:
            a = _p3(lines[i])
            b = _p3(lines[i + 1])
            wp_pts.append((0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])))
            i += 3
        except ValueError:
            i += 1

    # Subsample waypoints if very dense (e.g., B3313 supermaps) to keep APSP compact
    if len(wp_pts) > 384:
        stride = max(1, len(wp_pts) // 320)
        wp_pts = wp_pts[::stride]

    # k-NN graph over waypoints
    edges: list[tuple[int, int]] = []
    if len(wp_pts) > 1:
        pts_arr = np.asarray(wp_pts, dtype=np.float32)
        dmat = np.linalg.norm(pts_arr[:, None, :] - pts_arr[None, :, :], axis=-1)
        np.fill_diagonal(dmat, 1e9)
        k_nb = min(6, len(wp_pts) - 1)
        nbrs = np.argpartition(dmat, k_nb, axis=1)[:, :k_nb]
        for u in range(len(wp_pts)):
            for nb in nbrs[u]:
                if dmat[u, nb] < 1200.0:
                    edges.append((u, int(nb)))

    by_targetname = {e["targetname"]: e for e in ents if "targetname" in e}
    cart_tracks: list[dict[str, Any]] = []
    for e in ents:
        if e.get("classname") != "func_plc_cart":
            continue
        cur = by_targetname.get(e.get("target", ""))
        nodes = []
        cps = []
        cum = 0.0
        seen = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            p = _p3(cur.get("origin", "0 0 0"))
            if nodes:
                cum += math.dist(nodes[-1], p)
            nodes.append(p)
            if int(float(cur.get("spawnflags", "0"))) & 1:
                cps.append(cum)
            cur = by_targetname.get(cur.get("target", ""))
        if len(nodes) >= 2:
            cart_tracks.append({
                "nodes": nodes,
                "checkpoints": cps,
                "radius": float(e.get("radius", 160.0)),
                "height": float(e.get("height", 96.0)),
                "speed_scale": float(e.get("plc_speed_scale", 1.0)),
            })

    K = int(spec["teams"])
    team_spawns: list[list[tuple[float, float, float]]] = [[] for _ in range(K)]
    for e in ents:
        if e.get("classname") == "info_player_team":
            ti = int(float(e.get("team", "1"))) - 1
            if 0 <= ti < K:
                team_spawns[ti].append(_p3(e.get("origin", "0 0 0")))

    return _compile_world_graph(
        map_name,
        K,
        int(spec["players"]),
        wp_pts,
        edges,
        cart_tracks,
        team_spawns,
        device=device,
    )
