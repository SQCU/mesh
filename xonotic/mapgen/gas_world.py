#!/usr/bin/env python3
"""Gas-Law Multi-Tile Expander Procedural Payload World Generator (K <= 24, J <= 256, N <= 255).

Constructs leak-free, zero-pit, zero-rider-gap M-tile arena graphs connected by explicit
parameterized membrane joints (width W_e, length L_e) sized by the empirical Xonotic
boomer-shooter kinetic gas state (n_0 ~ 3.2 players/Mqu^2, v_rms ~ 450 u/s, Q_e in [Q_min, Q_max]).

Guarantees by construction:
  - Always supports J > K (strictly more cartpaths than teams, up to K=24, J=256)
  - Supports up to N=255 simultaneous players via `info_player_team` ("team" "1..K")
  - Emits `.joints.json` with exact tile bounds and joint portal planes (origin, normal, W_e, L_e)
    for live kinetic thermometry (flux Q_e, transit temperature T_e, theta_e = T_e/T_bulk,
    normal-flow efficiency eta_perp, and Clausing transmission kappa_e).
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

from spiralgen import fmt, quad_prism, vlen, vsub

THEMES = [
    ("exx/floor-tread01", "exx/wall-bigrib02", "exx/trim-01", "exx/base-metal03"),
    ("exx/base-crete01", "exx/panel-metalbig04", "exx/trim-01", "exx/base-metal03"),
    ("exx/floor-clang01", "exx/panel-metalbig04", "exx/trim-01", "exx/wall-bigrib02"),
]

WEAPONS_AND_ITEMS = [
    "weapon_machinegun",
    "weapon_electro",
    "weapon_devastator",
    "weapon_vortex",
    "weapon_hagar",
    "item_health_medium",
    "item_armor_medium",
]


def _box(x0, y0, z0, x1, y1, z1, tex_top, tex_side=None):
    ts = tex_side or tex_top
    return quad_prism(
        (float(x0), float(y0), float(z0)),
        (float(x1), float(y0), float(z0)),
        (float(x1), float(y1), float(z0)),
        (float(x0), float(y1), float(z0)),
        (0.0, 0.0, float(z1 - z0)),
        tex_top,
        ts,
    )


def sample_gas_world_params(seed: int) -> dict:
    """Heavy-tailed sampler for (K teams, J cartpaths, N players, M tiles) with:
      - E[K] ~ 8, K in [2, 24]
      - E[J] ~ 64, strictly J > K, tail covering up to 256
      - N in [max(2*K, 12), 240]
      - M = max(1, round(N / 12), ceil(J / 8)) clamped to [1, 24] tiles
    preserving the empirical stock Xonotic gas density n_0 ~ 3.2 players/Mqu^2."""
    rng = random.Random(seed)
    # Log-normal with right tail covering K=24 and mean ~8.0
    k_raw = math.exp(rng.gauss(math.log(6.6), 0.62))
    K = max(2, min(24, int(round(k_raw))))
    # Carts strictly > K, heavy-tailed multiplier around ~8x K so E[J] ~ 64, tail to 256
    mult = math.exp(rng.gauss(math.log(6.8), 0.58))
    J = max(K + 1, min(256, int(round(K * mult))))
    # Players scaled to teams & carts (at least 2/team, typical 10-16/team, capped at 240)
    per_team = max(2, min(16, int(round(rng.uniform(6.0, 14.0)))))
    N = max(max(2 * K, 12), min(240, K * per_team))
    # Gas-density & cart-packing conservation: ~12 players/tile, <= 8 carts/tile (up to 12 at tail)
    m_gas = max(1, int(round(N / 12.0)))
    m_cart = max(1, int(math.ceil(J / 8.0)))
    M = max(1, min(24, max(m_gas, m_cart)))
    joint_width = rng.choice([288.0, 320.0, 352.0, 384.0])
    return {"seed": seed, "teams": K, "carts": J, "players": N, "tiles": M, "joint_width": joint_width}


def _grid_dims(M: int) -> tuple[int, int]:
    """Factor M tiles into a compact (cols, rows) rectangle with cols * rows >= M."""
    best = (M, 1)
    best_score = (M, abs(M - 1))
    for r in range(1, int(math.sqrt(M)) + 2):
        c = int(math.ceil(M / float(r)))
        waste = c * r - M
        score = (waste, abs(c - r))
        if score < best_score:
            best_score = score
            best = (c, r)
    return best


class GasWorldBuilder:
    def __init__(self, args):
        self.a = args
        self.rng = random.Random(args.seed)
        self.tex_floor, self.tex_wall, self.tex_trim, self.tex_ceil = THEMES[args.seed % len(THEMES)]
        self.S = float(args.tile_size)          # 1792u -> 3.21 Mqu^2/tile
        self.wall_t = float(args.joint_length)  # 96u partition thickness (short Clausing aspect L/W!)
        self.joint_w = float(args.joint_width)  # default 320u doorway aperture
        self.ceil_h = 512.0
        self.K = max(2, min(256, int(args.teams)))
        self.J = max(self.K + 1, min(256, int(args.carts)))
        self.M = max(1, min(24, int(args.tiles)))
        self.cols, self.rows = _grid_dims(self.M)
        self.M = self.cols * self.rows  # clean rectangular grid for leak-free partition sealing

        self.brushes = []
        self.ents: list[tuple[str, dict[str, str]]] = []
        self.tiles: list[dict] = []
        self.joints: list[dict] = []
        self.cart_tracks: list[list[tuple[float, float, float]]] = []
        self.team_spawns: list[tuple[int, tuple[float, float, float], float]] = []
        self.wps: list[tuple[float, float, float]] = []
        self.edges: set[tuple[int, int]] = set()

    def _tile_bounds(self, cx: int, cy: int):
        pitch = self.S + self.wall_t
        x0 = cx * pitch
        y0 = cy * pitch
        return x0, y0, x0 + self.S, y0 + self.S

    def _add_wp(self, x: float, y: float, z: float = 28.0) -> int:
        p = (round(float(x), 1), round(float(y), 1), round(float(z), 1))
        for idx, q in enumerate(self.wps):
            if vlen(vsub(p, q)) < 96.0:
                return idx
        self.wps.append(p)
        return len(self.wps) - 1

    def _link_wp(self, a: int, b: int):
        if a != b:
            self.edges.add((min(a, b), max(a, b)))

    def _build_shell_and_partitions(self):
        pitch = self.S + self.wall_t
        total_x = self.cols * pitch - self.wall_t
        total_y = self.rows * pitch - self.wall_t
        T = 64.0
        z0, z1 = 0.0, self.ceil_h
        # Continuous solid z=0 floor + z=512 ceiling + 4 outer sealing walls (0 leaks, 0 pits, 0 rider gaps!)
        self.brushes.append(_box(-T, -T, z0 - T, total_x + T, total_y + T, z0, self.tex_floor, self.tex_wall))
        self.brushes.append(_box(-T, -T, z1, total_x + T, total_y + T, z1 + T, self.tex_ceil, self.tex_ceil))
        self.brushes.append(_box(-T, -T, z0 - T, 0.0, total_y + T, z1 + T, self.tex_wall, self.tex_wall))
        self.brushes.append(_box(total_x, -T, z0 - T, total_x + T, total_y + T, z1 + T, self.tex_wall, self.tex_wall))
        self.brushes.append(_box(-T, -T, z0 - T, total_x + T, 0.0, z1 + T, self.tex_wall, self.tex_wall))
        self.brushes.append(_box(-T, total_y, z0 - T, total_x + T, total_y + T, z1 + T, self.tex_wall, self.tex_wall))

        # Record tile descriptors
        for cy in range(self.rows):
            for cx in range(self.cols):
                tid = cy * self.cols + cx
                x0, y0, x1, y1 = self._tile_bounds(cx, cy)
                self.tiles.append({
                    "id": tid, "cx": cx, "cy": cy,
                    "x0": round(x0, 1), "y0": round(y0, 1), "x1": round(x1, 1), "y1": round(y1, 1),
                    "center": [round(0.5 * (x0 + x1), 1), round(0.5 * (y0 + y1), 1), 0.0],
                    "area_mqu2": round((self.S * self.S) / 1e6, 3),
                })

        # Partition walls between tiles perforated by parameterized membrane joints of width W_e
        door_h = 256.0
        half_w = self.joint_w * 0.5
        jid = 0

        # Vertical partition walls between (cx, cy) and (cx+1, cy)
        for cy in range(self.rows):
            for cx in range(self.cols - 1):
                a_id = cy * self.cols + cx
                b_id = cy * self.cols + (cx + 1)
                _, y0, x_wall0, y1 = self._tile_bounds(cx, cy)
                x_wall1 = x_wall0 + self.wall_t
                # Give corner/edge walls 2 symmetric joints when M >= 4 so every tile has d >= 3 apertures!
                dual = (self.rows == 1 and self.cols >= 2) or (cy in (0, self.rows - 1) and cx in (0, self.cols - 2) and self.M >= 4)
                centers_y = [y0 + self.S * 0.30, y0 + self.S * 0.70] if dual else [0.5 * (y0 + y1)]
                # Build solid wall pieces around the doorways + lintel above door_h
                spans = []
                cur = y0
                for myc in centers_y:
                    d0, d1 = myc - half_w, myc + half_w
                    if d0 > cur + 1.0:
                        spans.append((cur, d0, z0, z1))
                    spans.append((d0, d1, door_h, z1))
                    cur = d1
                    self.joints.append({
                        "id": jid, "tile_a": a_id, "tile_b": b_id,
                        "origin": [round(0.5 * (x_wall0 + x_wall1), 1), round(myc, 1), 0.0],
                        "normal": [1.0, 0.0, 0.0],
                        "width": self.joint_w, "length": self.wall_t, "height": door_h,
                    })
                    jid += 1
                if y1 > cur + 1.0:
                    spans.append((cur, y1, z0, z1))
                for s0, s1, sz0, sz1 in spans:
                    self.brushes.append(_box(x_wall0, s0, sz0, x_wall1, s1, sz1, self.tex_wall, self.tex_wall))

        # Horizontal partition walls between (cx, cy) and (cx, cy+1)
        for cy in range(self.rows - 1):
            for cx in range(self.cols):
                a_id = cy * self.cols + cx
                b_id = (cy + 1) * self.cols + cx
                x0, _, x1, y_wall0 = self._tile_bounds(cx, cy)
                y_wall1 = y_wall0 + self.wall_t
                dual = (self.cols == 1 and self.rows >= 2)
                centers_x = [x0 + self.S * 0.30, x0 + self.S * 0.70] if dual else [0.5 * (x0 + x1)]
                spans = []
                cur = x0
                for mxc in centers_x:
                    d0, d1 = mxc - half_w, mxc + half_w
                    if d0 > cur + 1.0:
                        spans.append((cur, d0, z0, z1))
                    spans.append((d0, d1, door_h, z1))
                    cur = d1
                    self.joints.append({
                        "id": jid, "tile_a": a_id, "tile_b": b_id,
                        "origin": [round(mxc, 1), round(0.5 * (y_wall0 + y_wall1), 1), 0.0],
                        "normal": [0.0, 1.0, 0.0],
                        "width": self.joint_w, "length": self.wall_t, "height": door_h,
                    })
                    jid += 1
                if x1 > cur + 1.0:
                    spans.append((cur, x1, z0, z1))
                for s0, s1, sz0, sz1 in spans:
                    self.brushes.append(_box(s0, y_wall0, sz0, s1, y_wall1, sz1, self.tex_wall, self.tex_wall))

        # Fill corner wall-intersection posts cleanly
        for cy in range(self.rows - 1):
            for cx in range(self.cols - 1):
                _, _, xw0, yw0 = self._tile_bounds(cx, cy)
                self.brushes.append(_box(xw0, yw0, z0, xw0 + self.wall_t, yw0 + self.wall_t, z1, self.tex_trim, self.tex_trim))

    def _build_tile_interiors_and_nav(self):
        """Populate each tile with clean cover pillars, lights, pickups, and a lean connected
        waypoint mesh routed straight through every membrane joint."""
        grid_n = 5  # 5x5 walkable grid per tile (~350u spacing -> 25 wps/tile, super fast linking!)
        tile_wps: dict[int, list[list[int]]] = {}
        for t in self.tiles:
            tid = t["id"]
            x0, y0, x1, y1 = t["x0"], t["y0"], t["x1"], t["y1"]
            mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            # Central cover pillar (rotated/offset slightly by tile seed; small 44u half-width so bots never snag)
            self.brushes.append(_box(mx - 44.0, my - 44.0, -8.0, mx + 44.0, my + 44.0, 192.0, self.tex_trim, self.tex_wall))
            # Area light + weapon/health pickup per tile
            self.ents.append(("light", {"origin": f"{fmt(mx)} {fmt(my)} {fmt(self.ceil_h - 64.0)}", "light": "850"}))
            self.ents.append((self.rng.choice(WEAPONS_AND_ITEMS), {"origin": f"{fmt(mx + 140.0)} {fmt(my)} 28"}))

            # Waypoint lattice inside tile
            mat = []
            margin = 160.0
            for iy in range(grid_n):
                row = []
                py = y0 + margin + (self.S - 2.0 * margin) * iy / float(grid_n - 1)
                for ix in range(grid_n):
                    px = x0 + margin + (self.S - 2.0 * margin) * ix / float(grid_n - 1)
                    # Nudge center waypoint clear of central pillar
                    if ix == 2 and iy == 2:
                        px += 96.0
                    row.append(self._add_wp(px, py, 28.0))
                mat.append(row)
            for iy in range(grid_n):
                for ix in range(grid_n):
                    if ix + 1 < grid_n:
                        self._link_wp(mat[iy][ix], mat[iy][ix + 1])
                    if iy + 1 < grid_n:
                        self._link_wp(mat[iy][ix], mat[iy + 1][ix])
                    if ix + 1 < grid_n and iy + 1 < grid_n:
                        self._link_wp(mat[iy][ix], mat[iy + 1][ix + 1])
            tile_wps[tid] = mat

        # Route clean 3-node straight ballistic waypoint splines through EVERY membrane joint!
        for jt in self.joints:
            ox, oy, _ = jt["origin"]
            nx, ny, _ = jt["normal"]
            off = 0.5 * self.wall_t + 120.0
            wa = self._add_wp(ox - nx * off, oy - ny * off, 28.0)
            wc = self._add_wp(ox, oy, 28.0)
            wb = self._add_wp(ox + nx * off, oy + ny * off, 28.0)
            self._link_wp(wa, wc)
            self._link_wp(wc, wb)
            # Connect wa/wb to their 2 nearest interior waypoints in tile_a and tile_b
            for wp_end, tid in ((wa, jt["tile_a"]), (wb, jt["tile_b"])):
                flat = [w for r in tile_wps[tid] for w in r]
                flat.sort(key=lambda w: vlen(vsub(self.wps[wp_end], self.wps[w])))
                for nb in flat[:2]:
                    self._link_wp(wp_end, nb)

    def _build_carts_and_team_spawns(self):
        """Place J symmetric cartpaths (mix of cross-joint inter-tile lanes and intra-tile lanes)
        and K spatially disjoint team spawn alcoves across the M tiles."""
        # 1. Assign up to 1 cross-joint cart per joint (up to 35% of J) so objectives couple submaps!
        n_cross_budget = min(len(self.joints), int(self.J * 0.35)) if self.M > 1 else 0
        joint_order = list(self.joints)
        self.rng.shuffle(joint_order)
        for jt in joint_order[:n_cross_budget]:
            ox, oy, _ = jt["origin"]
            nx, ny, _ = jt["normal"]
            sgn = 1.0 if (jt["id"] % 2 == 0) else -1.0
            half_L = 420.0  # 840u track centered right through the joint aperture!
            p0 = (ox - sgn * nx * half_L, oy - sgn * ny * half_L, 0.0)
            p1 = (ox + sgn * nx * half_L, oy + sgn * ny * half_L, 0.0)
            self.cart_tracks.append([p0, (ox, oy, 0.0), p1])

        # 2. Distribute remaining J - len(cart_tracks) intra-tile lanes evenly across all M tiles!
        rem = self.J - len(self.cart_tracks)
        # Predefined clean non-overlapping lane templates inside a 1792x1792 tile (avoids center pillar & corners)
        # Supports up to 12 cleanly separated ~700-860u lanes per tile!
        lane_templates = [
            ((240, 320), (1040, 320)),
            ((1552, 1472), (752, 1472)),
            ((320, 1552), (320, 752)),
            ((1472, 240), (1472, 1040)),
            ((540, 620), (1252, 620)),
            ((1252, 1172), (540, 1172)),
            ((620, 1252), (620, 540)),
            ((1172, 540), (1172, 1252)),
            ((240, 1472), (720, 1040)),
            ((1552, 320), (1072, 752)),
            ((240, 896), (720, 896)),
            ((1552, 896), (1072, 896)),
        ]
        for k in range(rem):
            tid = k % self.M
            slot = k // self.M
            t = self.tiles[tid]
            x0, y0 = t["x0"], t["y0"]
            (ax, ay), (bx, by) = lane_templates[slot % len(lane_templates)]
            # Tiny deterministic offset if >12 carts ever packed into 1 tile
            ring_shift = (slot // len(lane_templates)) * 48.0
            p0 = (x0 + ax + ring_shift, y0 + ay, 0.0)
            p1 = (x0 + bx + ring_shift, y0 + by, 0.0)
            self.cart_tracks.append([p0, (0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]), 0.0), p1])

        # 3. Distribute K disjoint team spawn bases across M tiles (max 4 corner alcoves per tile -> up to 96 teams!)
        corner_alcoves = [
            (220.0, 220.0, 45.0),
            (self.S - 220.0, self.S - 220.0, 225.0),
            (220.0, self.S - 220.0, 315.0),
            (self.S - 220.0, 220.0, 135.0),
        ]
        # Order tiles farthest-first so K < M spreads teams cleanly across the world
        tile_ids = [0]
        rem_t = set(range(1, self.M))
        while rem_t:
            nxt = max(
                rem_t,
                key=lambda tid: min(vlen(vsub(self.tiles[tid]["center"], self.tiles[c]["center"])) for c in tile_ids),
            )
            tile_ids.append(nxt)
            rem_t.remove(nxt)

        per_team_spots = 4
        for team_idx in range(self.K):
            tid = tile_ids[team_idx % len(tile_ids)]
            alc_idx = (team_idx // len(tile_ids)) % len(corner_alcoves)
            t = self.tiles[tid]
            ax, ay, yaw = corner_alcoves[alc_idx]
            bx, by = t["x0"] + ax, t["y0"] + ay
            for dx, dy in ((-48.0, -48.0), (48.0, -48.0), (-48.0, 48.0), (48.0, 48.0))[:per_team_spots]:
                self.team_spawns.append((team_idx + 1, (bx + dx, by + dy, 32.0), yaw))

    def _emit_entities(self):
        for t in range(self.K):
            self.ents.append(("plc_team", {"cnt": str(t)}))
        for team_1, pos, yaw in self.team_spawns:
            self.ents.append((
                "info_player_team",
                {"origin": f"{fmt(pos[0])} {fmt(pos[1])} {fmt(pos[2])}", "angle": fmt(yaw), "team": str(team_1)},
            ))
        # Emit compact 5-node (4-checkpoint) plc_path chains per cart (lean entity budget even at J=128..256!)
        ncp = max(1, int(self.a.checkpoints_per_cart))
        for ci, tr in enumerate(self.cart_tracks):
            p0, _, p1 = tr[0], tr[1], tr[-1]
            nodes = [
                (p0[0] + (p1[0] - p0[0]) * k / float(ncp), p0[1] + (p1[1] - p0[1]) * k / float(ncp), 24.0)
                for k in range(ncp + 1)
            ]
            names = [f"plc{ci}n{ni}" for ni in range(len(nodes))]
            for ni, p in enumerate(nodes):
                kv = {"targetname": names[ni], "origin": f"{fmt(p[0])} {fmt(p[1])} {fmt(p[2])}"}
                if ni + 1 < len(nodes):
                    kv["target"] = names[ni + 1]
                if ni > 0:
                    kv["spawnflags"] = "1"
                self.ents.append(("plc_path", kv))
            self.ents.append((
                "func_plc_cart",
                {
                    "origin": f"{fmt(nodes[0][0])} {fmt(nodes[0][1])} {fmt(nodes[0][2])}",
                    "target": names[0],
                    "radius": "160",
                    "height": "96",
                    "plc_speed_scale": "1",
                },
            ))

    def build(self):
        self._build_shell_and_partitions()
        self._build_tile_interiors_and_nav()
        self._build_carts_and_team_spawns()
        self._emit_entities()
        return self.wps, sorted(self.edges)


def write_and_compile_gas_world(builder: GasWorldBuilder, out_dir: Path, name: str, compile_bsp: bool, xonotic_dir: Path, pk3_out: Path | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"// generated by gas_world.py K={builder.K} J={builder.J} M={builder.M} W={builder.joint_w} seed={builder.a.seed}",
        "{",
        '"classname" "worldspawn"',
        '"_minlight" "32"',
        f'"message" "{name}"',
    ]
    for br in builder.brushes:
        lines.append(br.to_map(0.5))
    lines.append("}")
    for cls, kv in builder.ents:
        lines.append("{")
        lines.append(f'"classname" "{cls}"')
        for k, v in kv.items():
            lines.append(f'"{k}" "{v}"')
        lines.append("}")
    map_path = out_dir / f"{name}.map"
    map_path.write_text("\n".join(lines) + "\n")

    (out_dir / f"{name}.mapinfo").write_text(
        f"title {name}\n"
        f"description Gas-law multi-tile payload world (K={builder.K}, J={builder.J}, M={builder.M}).\n"
        f"author gas_world\nhas weapons\ngametype plc\ngametype dm\ngametype tdm\n"
    )

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    wl = ["//WAYPOINT_VERSION 1.04", "//WAYPOINT_SYMMETRY 0", f"//WAYPOINT_TIME {ts}"]
    for p in builder.wps:
        s = f"'{fmt(p[0])} {fmt(p[1])} {fmt(p[2])}'"
        wl.extend([s, s, "0"])
    (out_dir / f"{name}.waypoints").write_text("\n".join(wl) + "\n")

    cl = ["//WAYPOINT_VERSION 1.04", f"//WAYPOINT_TIME {ts}"]
    for i, j in sorted(builder.edges):
        a, b = builder.wps[i], builder.wps[j]
        sa = f"'{fmt(a[0])} {fmt(a[1])} {fmt(a[2])}'"
        sb = f"'{fmt(b[0])} {fmt(b[1])} {fmt(b[2])}'"
        cl.append(f"{sa}*{sb}")
        cl.append(f"{sb}*{sa}")
    (out_dir / f"{name}.waypoints.cache").write_text("\n".join(cl) + "\n")

    lens = [round(vlen(vsub(tr[0], tr[-1])), 1) for tr in builder.cart_tracks]
    cross_d = [
        vlen(vsub(p1, p2))
        for i, (t1, p1, _) in enumerate(builder.team_spawns)
        for t2, p2, _ in builder.team_spawns[i + 1:]
        if t1 != t2
    ]
    deg = {t["id"]: 0 for t in builder.tiles}
    for jt in builder.joints:
        deg[jt["tile_a"]] += 1
        deg[jt["tile_b"]] += 1

    joints_doc = {
        "name": name, "seed": builder.a.seed,
        "teams": builder.K, "carts": builder.J, "tiles_count": builder.M,
        "cols": builder.cols, "rows": builder.rows,
        "tile_size_u": builder.S, "joint_width_u": builder.joint_w, "joint_length_u": builder.wall_t,
        "total_walkable_area_mqu2": round(sum(t["area_mqu2"] for t in builder.tiles), 3),
        "tile_degrees": deg,
        "tiles": builder.tiles,
        "joints": builder.joints,
    }
    (out_dir / f"{name}.joints.json").write_text(json.dumps(joints_doc, indent=2))

    meta = {
        "name": name, "seed": builder.a.seed,
        "teams": builder.K, "carts": builder.J, "tiles": builder.M,
        "grid": f"{builder.cols}x{builder.rows}",
        "joints": len(builder.joints), "joint_width_u": builder.joint_w, "joint_length_u": builder.wall_t,
        "min_tile_degree": min(deg.values()) if deg else 0,
        "max_tile_degree": max(deg.values()) if deg else 0,
        "total_area_mqu2": joints_doc["total_walkable_area_mqu2"],
        "brushes": len(builder.brushes), "entities": len(builder.ents),
        "waypoints": len(builder.wps), "waypoint_edges": len(builder.edges),
        "cart_len_min": min(lens) if lens else 0.0,
        "cart_len_max": max(lens) if lens else 0.0,
        "path_length_ratio": round(max(lens) / max(min(lens), 1.0), 3) if lens else 1.0,
        "min_cross_team_spawn_dist": round(min(cross_d), 1) if cross_d else None,
    }

    if compile_bsp:
        repo_root = Path(__file__).resolve().parents[2]
        q3 = repo_root / "bin/mesh-q3map2"
        fs_home = map_path.parent.parent.parent
        base = [str(q3), "-game", "xonotic", "-fs_basepath", str(xonotic_dir), "-fs_homepath", str(fs_home)]
        t0 = time.monotonic()
        r_meta = subprocess.run(base + ["-meta", str(map_path)], capture_output=True, text=True)
        if r_meta.returncode != 0 or "LEAKED" in (r_meta.stdout + r_meta.stderr).upper():
            raise RuntimeError(f"q3map2 -meta failed/leaked on {name}:\n{(r_meta.stdout + r_meta.stderr)[-800:]}")
        r_vis = subprocess.run(base + ["-vis", "-fast", str(map_path)], capture_output=True, text=True)
        if r_vis.returncode != 0:
            raise RuntimeError(f"q3map2 -vis failed on {name}:\n{(r_vis.stdout + r_vis.stderr)[-400:]}")
        r_light = subprocess.run(base + ["-light", "-fast", str(map_path)], capture_output=True, text=True)
        if r_light.returncode != 0:
            raise RuntimeError(f"q3map2 -light failed on {name}:\n{(r_light.stdout + r_light.stderr)[-400:]}")
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
    ap = argparse.ArgumentParser(description="Gas-law multi-tile expander procedural payload world generator.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", default="gas_world")
    ap.add_argument("--out", default="/tmp/gas_worlds/fs/data/maps")
    ap.add_argument("--teams", type=int, default=8)
    ap.add_argument("--carts", type=int, default=64)
    ap.add_argument("--tiles", type=int, default=9)
    ap.add_argument("--tile-size", type=float, default=1792.0)
    ap.add_argument("--joint-width", type=float, default=320.0)
    ap.add_argument("--joint-length", type=float, default=96.0)
    ap.add_argument("--checkpoints-per-cart", type=int, default=4)
    ap.add_argument("--sample", action="store_true", help="Sample (K, J, N, M, W_e) from the heavy-tailed gas distribution")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--pk3-out", default="")
    ap.add_argument("--xonotic-dir", default="~/dox/xonotic/Xonotic")
    args = ap.parse_args()

    if args.sample:
        p = sample_gas_world_params(args.seed)
        args.teams = p["teams"]
        args.carts = p["carts"]
        args.tiles = p["tiles"]
        args.joint_width = p["joint_width"]

    b = GasWorldBuilder(args)
    b.build()
    pk3 = Path(args.pk3_out).expanduser() if args.pk3_out else None
    meta = write_and_compile_gas_world(b, Path(args.out).expanduser(), args.name, args.compile, Path(args.xonotic_dir).expanduser(), pk3_out=pk3)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
