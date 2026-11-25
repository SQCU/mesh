from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from payload.tools.strategy_io_schema import CS, EVT, EVT_KIND, OBS, TS
from .neighborhood import NavigationRows


MAJOR_ITEM_CLASS = {
    "item_strength": 4.0,
    "item_shield": 4.0,
    "item_invincible": 4.0,
    "item_health_mega": 3.0,
    "item_armor_big": 3.0,
    "item_armor_mega": 3.0,
    "weapon_devastator": 2.0,
    "weapon_vortex": 2.0,
    "weapon_electro": 2.0,
    "weapon_mortar": 2.0,
    "weapon_hagar": 2.0,
    "weapon_crylink": 2.0,
    "weapon_machinegun": 2.0,
    "weapon_arc": 2.0,
}

PUSH_RADIUS = 160.0
PUSH_HEIGHT = 96.0
PUSH_CAP = 3
BASE_PUSH_SPEED = 30.0
CONTEST_MAX_REV = 24.0
REVERSE_SPEED = 14.0
MAX_CART_SPEED = 200.0


def _parse_ent_blocks(ent_text: str) -> list[dict[str, str]]:
    out = []
    for raw in re.findall(r"\{[^{}]*\}", ent_text):
        d = {}
        for k, v in re.findall(r'"([^"]+)"\s+"([^"]*)"', raw):
            d[k] = v
        if d:
            out.append(d)
    return out


def _parse_vec3(s: str) -> np.ndarray:
    parts = [float(x) for x in s.split()]
    if len(parts) < 3:
        parts = (parts + [0.0, 0.0, 0.0])[:3]
    return np.asarray(parts[:3], dtype=np.float32)


@dataclass
class CartSpline:
    cart_id: int
    points: np.ndarray          # [P, 3]
    arc_s: np.ndarray           # [P]
    tangents: np.ndarray        # [P, 3]
    length: float
    checkpoints_s: np.ndarray   # [M]
    checkpoints_pos: np.ndarray # [M, 3]
    sample_s: np.ndarray        # [S] fixed 5 spline anchors
    sample_pos: np.ndarray      # [S, 3]
    sample_tan: np.ndarray      # [S, 3]
    sample_is_cp: np.ndarray    # [S]

    def eval_at(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        sc = float(np.clip(s, 0.0, max(self.length, 1e-6)))
        idx = int(np.searchsorted(self.arc_s, sc, side="right")) - 1
        idx = max(0, min(idx, len(self.arc_s) - 2))
        s0, s1 = float(self.arc_s[idx]), float(self.arc_s[idx + 1])
        t = (sc - s0) / max(s1 - s0, 1e-6)
        pos = self.points[idx] * (1.0 - t) + self.points[idx + 1] * t
        tan = self.tangents[idx]
        return pos.astype(np.float32), tan.astype(np.float32)

    def checkpoint_margins(self, s: float) -> tuple[float, float, float]:
        if len(self.checkpoints_s) == 0:
            return max(0.0, self.length - s), s, s
        ahead = self.checkpoints_s[self.checkpoints_s > s + 1e-3]
        behind = self.checkpoints_s[self.checkpoints_s <= s + 1e-3]
        d_next = float(ahead[0] - s) if len(ahead) else 0.0
        d_prev = float(s - behind[-1]) if len(behind) else float(s)
        return d_next, d_prev, float(s)


def _cart_speed_ode(w_ctrl: float, w_opp_total: float, w_opp_max: float, has_ctrl: bool) -> float:
    if has_ctrl and w_ctrl > 0.0:
        v = BASE_PUSH_SPEED * (w_ctrl - w_opp_total) / (1.0 + w_opp_total * w_opp_total)
        return float(np.clip(v, -CONTEST_MAX_REV, MAX_CART_SPEED))
    if has_ctrl and w_opp_max > 0.0:
        return float(np.clip(-REVERSE_SPEED * w_opp_max, -MAX_CART_SPEED, 0.0))
    return 0.0


class MapManifold:
    """Closed-form map geometry coupling real Voronoi cells (VCellMap + NavigationRows),
    literal cart Bézier/polyline splines with checkpoint arc-lengths, and typed major pickups."""

    def __init__(
        self,
        nav: NavigationRows,
        splines: list[CartSpline],
        item_pos: np.ndarray,
        item_class: np.ndarray,
        item_names: list[str],
        cell_centroids: np.ndarray,
        cell_areas: np.ndarray,
    ):
        self.nav = nav
        self.splines = splines
        self.item_pos = item_pos.astype(np.float32)       # [I, 3]
        self.item_class = item_class.astype(np.float32)   # [I]
        self.item_names = item_names
        self.cell_centroids = cell_centroids.astype(np.float32)  # [C, 3]
        self.cell_areas = cell_areas.astype(np.float32)          # [C]
        self.n_cells = len(self.cell_centroids)
        self.n_items = len(self.item_pos)
        # Map center & spatial scale for coordinate-invariant EDM2 input prenormalization
        all_pts = [p for p in [self.cell_centroids, self.item_pos] if len(p) > 0]
        for sp in splines:
            if len(sp.points) > 0:
                all_pts.append(sp.points)
        if all_pts:
            cat = np.concatenate(all_pts, axis=0)
            self.map_center = cat.mean(axis=0).astype(np.float32)
            self.map_radius = float(max(256.0, np.percentile(np.linalg.norm(cat - self.map_center[None, :], axis=-1), 95)))
        else:
            self.map_center = np.zeros((3,), dtype=np.float32)
            self.map_radius = 1000.0
        self.item_cells = (
            np.asarray([nav.map.assign_cell(p) for p in self.item_pos], dtype=np.int64)
            if (nav.map is not None and self.n_items > 0)
            else np.zeros((self.n_items,), dtype=np.int64)
        )
        # Precompute per-cell minimum distance to each cart spline
        self.cell_track_dist = np.zeros((self.n_cells, max(1, len(splines))), dtype=np.float32)
        self.cell_track_s = np.zeros((self.n_cells, max(1, len(splines))), dtype=np.float32)
        for j, sp in enumerate(splines):
            for c in range(self.n_cells):
                d = np.linalg.norm(sp.points - self.cell_centroids[c][None, :], axis=-1)
                k = int(np.argmin(d))
                self.cell_track_dist[c, j] = float(d[k])
                self.cell_track_s[c, j] = float(sp.arc_s[k]) / max(sp.length, 1.0)

    def summary(self) -> dict:
        return {
            "nav_report": dict(self.nav.report),
            "n_cells": self.n_cells,
            "n_carts": len(self.splines),
            "cart_lengths": [round(s.length, 1) for s in self.splines],
            "n_major_items": self.n_items,
            "major_items": self.item_names,
        }

    def build_features(
        self,
        obs_rows: np.ndarray,
        cart_rows: np.ndarray,
        team_rows: np.ndarray,
        event_rows: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute literal closed-form geometric features:
        - bot_geo [N, 12]: Voronoi cell coords, nearest-cart Frenet (sigma, nu, cyl_margin, dz),
          marginal speed gain/loss (delta_sdot_join, delta_sdot_leave), Voronoi-foveated local item/rival density
        - cart_geo [J, 10]: tangent_xy, norm_d_next_cp, norm_d_prev_cp, norm_d_origin, cp_eta,
          ctrl_push_count, opp_push_count, marginal_ctrl_gain, marginal_opp_stall
        - track_tokens [J*5, 10]: literal spline anchor points along every cart path
        - cell_tokens [C, 10]: Voronoi cell sites with live cart-ahead/behind relation & local bot/item density
        - item_tokens [I, 9]: major pickups with class, live availability, respawn_eta, nearest cart dist
        - fovea_weights [N, C]: real geodesic Gaussian Voronoi fovea kernel from VCellMap
        """
        n = len(obs_rows)
        j = len(cart_rows)
        obs_pos = obs_rows[:, OBS["POS_X"] : OBS["POS_Z"] + 1].astype(np.float32)
        obs_team = obs_rows[:, OBS["TEAM"]].astype(np.int64)
        obs_alive = obs_rows[:, OBS["ALIVE"]] > 0.5
        cart_pos = cart_rows[:, CS["POS_X"] : CS["POS_Z"] + 1].astype(np.float32)
        cart_s = cart_rows[:, CS["PATH_POSITION"]].astype(np.float32)
        cart_len = np.maximum(cart_rows[:, CS["PATH_LENGTH"]].astype(np.float32), 1.0)
        cart_ctrl = cart_rows[:, CS["CONTROL_TEAM"]].astype(np.int64)
        cart_spd = cart_rows[:, CS["SPEED"]].astype(np.float32)

        # 1. Cart occupancy inside exact push cylinder (R=160, H=96)
        n_teams = max(len(team_rows), int(obs_team.max(initial=1)))
        occ = np.zeros((j, n_teams + 1), dtype=np.int32)
        in_cart_id = np.full((n,), -1, dtype=np.int64)
        for ci in range(j):
            diff = obs_pos - cart_pos[ci][None, :]
            dxy = np.hypot(diff[:, 0], diff[:, 1])
            dz = np.abs(diff[:, 2])
            inside = obs_alive & (dxy <= PUSH_RADIUS) & (dz <= PUSH_HEIGHT)
            for bi in np.flatnonzero(inside):
                t = int(obs_team[bi])
                if 1 <= t <= n_teams:
                    occ[ci, t] += 1
                    in_cart_id[bi] = ci

        # 2. Cart geometric + exact ODE marginal features [J, 10]
        cart_geo = np.zeros((j, 10), dtype=np.float32)
        cart_tans = np.zeros((j, 3), dtype=np.float32)
        marg_join = np.zeros((j, n_teams + 1), dtype=np.float32)
        marg_leave = np.zeros((j, n_teams + 1), dtype=np.float32)
        for ci in range(j):
            sp = self.splines[ci] if ci < len(self.splines) else None
            if sp is not None:
                _, tan = sp.eval_at(float(cart_s[ci]))
                d_next, d_prev, d_orig = sp.checkpoint_margins(float(cart_s[ci]))
                L = max(sp.length, 1.0)
            else:
                tan = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
                L = float(cart_len[ci])
                d_next, d_prev, d_orig = max(0.0, L - float(cart_s[ci])), float(cart_s[ci]), float(cart_s[ci])
            cart_tans[ci] = tan
            ctrl = int(cart_ctrl[ci])
            w_t = np.minimum(occ[ci], PUSH_CAP).astype(np.float32)
            w_ctrl = float(w_t[ctrl]) if 1 <= ctrl <= n_teams else 0.0
            w_opp_tot = float(w_t[1:].sum() - w_ctrl)
            w_opp_max = float(max([w_t[t] for t in range(1, n_teams + 1) if t != ctrl], default=0.0))
            v_now = _cart_speed_ode(w_ctrl, w_opp_tot, w_opp_max, ctrl > 0)
            for t in range(1, n_teams + 1):
                w_add = w_t.copy()
                w_add[t] = min(PUSH_CAP, w_add[t] + 1.0)
                wc_a = float(w_add[ctrl]) if 1 <= ctrl <= n_teams else 0.0
                wo_a = float(w_add[1:].sum() - wc_a)
                wm_a = float(max([w_add[u] for u in range(1, n_teams + 1) if u != ctrl], default=0.0))
                v_add = _cart_speed_ode(wc_a, wo_a, wm_a, ctrl > 0)
                if ctrl == 0 and w_add[t] > 0 and wo_a == 0:
                    v_add = BASE_PUSH_SPEED
                marg_join[ci, t] = (v_add - v_now) / 30.0
                w_sub = w_t.copy()
                w_sub[t] = max(0.0, w_sub[t] - 1.0)
                wc_s = float(w_sub[ctrl]) if 1 <= ctrl <= n_teams else 0.0
                wo_s = float(w_sub[1:].sum() - wc_s)
                wm_s = float(max([w_sub[u] for u in range(1, n_teams + 1) if u != ctrl], default=0.0))
                v_sub = _cart_speed_ode(wc_s, wo_s, wm_s, ctrl > 0)
                marg_leave[ci, t] = (v_now - v_sub) / 30.0

            spd_abs = max(abs(float(cart_spd[ci])), 5.0)
            cp_eta = float(np.clip(d_next / spd_abs, 0.0, 30.0)) / 30.0
            m_ctrl_g = marg_join[ci, ctrl] if 1 <= ctrl <= n_teams else 1.0
            m_opp_s = min([marg_join[ci, t] for t in range(1, n_teams + 1) if t != ctrl], default=0.0)
            cart_geo[ci] = np.asarray([
                tan[0], tan[1],
                d_next / L, d_prev / L, d_orig / L,
                cp_eta,
                w_ctrl / 3.0, w_opp_tot / 3.0,
                m_ctrl_g, m_opp_s,
            ], dtype=np.float32)

        # 3. Literal cart-spline anchor tokens [J*5, 10]
        track_list = []
        track_teams = []
        for ci in range(j):
            sp = self.splines[ci] if ci < len(self.splines) else None
            if sp is None:
                continue
            L = max(sp.length, 1.0)
            s_cur = float(cart_s[ci])
            for r in range(len(sp.sample_s)):
                sr = float(sp.sample_s[r])
                p = sp.sample_pos[r]
                t = sp.sample_tan[r]
                track_list.append([
                    (p[0] - self.map_center[0]) / self.map_radius,
                    (p[1] - self.map_center[1]) / self.map_radius,
                    (p[2] - self.map_center[2]) / self.map_radius,
                    t[0], t[1],
                    sr / L,
                    (sr - s_cur) / L,
                    1.0 if sr > s_cur else -1.0,
                    float(sp.sample_is_cp[r]),
                    float(ci) / max(1.0, j - 1.0),
                ])
                track_teams.append(float(cart_ctrl[ci]))
        track_tokens = (
            np.asarray(track_list, dtype=np.float32)
            if track_list
            else np.zeros((0, 10), dtype=np.float32)
        )
        track_team_arr = np.asarray(track_teams, dtype=np.float32)

        # 4. Live major item state updated from perception events [I, 9]
        now_t = float(obs_rows[0, OBS["ENGINE_TIME"]]) if n > 0 else 0.0
        item_avail = np.ones((self.n_items,), dtype=np.float32)
        item_eta = np.zeros((self.n_items,), dtype=np.float32)
        if event_rows is not None and len(event_rows) > 0 and self.n_items > 0:
            ev_kind = event_rows[:, EVT["KIND"]].astype(np.int64)
            ev_pos = event_rows[:, EVT["POS_X"] : EVT["POS_Z"] + 1].astype(np.float32)
            ev_resp = event_rows[:, EVT["RESPAWN_TIME"]].astype(np.float32)
            mask_item = (ev_kind == EVT_KIND["ITEM_HERE"]) | (ev_kind == EVT_KIND["ITEM_GONE"])
            for idx in np.flatnonzero(mask_item):
                d = np.linalg.norm(self.item_pos - ev_pos[idx][None, :], axis=-1)
                mi = int(np.argmin(d))
                if d[mi] < 96.0:
                    if ev_kind[idx] == EVT_KIND["ITEM_HERE"]:
                        item_avail[mi] = 1.0
                        item_eta[mi] = 0.0
                    else:
                        item_avail[mi] = 0.0
                        item_eta[mi] = float(np.clip(ev_resp[idx] - now_t, 0.0, 60.0)) / 60.0

        item_tokens = np.zeros((self.n_items, 9), dtype=np.float32)
        for mi in range(self.n_items):
            p = self.item_pos[mi]
            d_cart = float(np.min(np.linalg.norm(cart_pos - p[None, :], axis=-1))) if j > 0 else 1000.0
            item_tokens[mi] = [
                (p[0] - self.map_center[0]) / self.map_radius,
                (p[1] - self.map_center[1]) / self.map_radius,
                (p[2] - self.map_center[2]) / self.map_radius,
                self.item_class[mi] / 4.0,
                1.0 if self.item_class[mi] >= 3.0 else 0.0,
                item_avail[mi],
                item_eta[mi],
                min(d_cart / self.map_radius, 3.0),
                float(self.item_cells[mi]) / max(1.0, self.n_cells),
            ]

        # 5. Voronoi fovea kernel [N, C] + per-bot Frenet/ODE/fovea features [N, 12]
        bot_cells = np.zeros((n,), dtype=np.int64)
        fovea_w = np.ones((n, max(1, self.n_cells)), dtype=np.float32) / max(1, self.n_cells)
        if self.nav.map is not None and self.n_cells > 0:
            for bi in range(n):
                c = self.nav.map.assign_cell(obs_pos[bi])
                bot_cells[bi] = c
                sm = self.nav.map.spatial_mask(c).astype(np.float32)
                s_sum = float(sm.sum())
                fovea_w[bi] = sm / s_sum if s_sum > 1e-6 else np.full((self.n_cells,), 1.0 / self.n_cells, dtype=np.float32)

        # Voronoi cell tokens [C, 10]
        cell_tokens = np.zeros((self.n_cells, 10), dtype=np.float32)
        if self.n_cells > 0:
            area_norm = self.cell_areas / max(float(self.cell_areas.sum()), 1.0)
            for c in range(self.n_cells):
                p = self.cell_centroids[c]
                bots_in_c = float(np.sum((bot_cells == c) & obs_alive))
                major_in_c = float(np.sum((self.item_cells == c) & (self.item_class >= 3.0) & (item_avail > 0.5))) if self.n_items else 0.0
                d_carts = np.linalg.norm(cart_pos - p[None, :], axis=-1) if j > 0 else np.asarray([1000.0])
                near_j = int(np.argmin(d_carts))
                min_tr = float(self.cell_track_dist[c, near_j]) if near_j < self.cell_track_dist.shape[1] else 1000.0
                s_rel = float(self.cell_track_s[c, near_j] - (cart_s[near_j] / cart_len[near_j])) if j > 0 and near_j < self.cell_track_s.shape[1] else 0.0
                cell_tokens[c] = [
                    (p[0] - self.map_center[0]) / self.map_radius,
                    (p[1] - self.map_center[1]) / self.map_radius,
                    (p[2] - self.map_center[2]) / self.map_radius,
                    float(area_norm[c]),
                    min(float(d_carts[near_j]) / self.map_radius, 3.0),
                    min(min_tr / self.map_radius, 3.0),
                    s_rel,
                    bots_in_c / 4.0,
                    major_in_c,
                    float(near_j) / max(1.0, j - 1.0),
                ]

        bot_geo = np.zeros((n, 12), dtype=np.float32)
        for bi in range(n):
            p = obs_pos[bi]
            t = int(obs_team[bi])
            c = int(bot_cells[bi])
            if j > 0:
                dists = np.linalg.norm(cart_pos - p[None, :], axis=-1)
                cj = int(np.argmin(dists))
                rel = p - cart_pos[cj]
                tan = cart_tans[cj]
                nrm = np.asarray([-tan[1], tan[0], 0.0], dtype=np.float32)
                sigma = float(np.dot(rel, tan)) / 500.0
                nu = float(np.dot(rel, nrm)) / 500.0
                cyl_m = (float(np.hypot(rel[0], rel[1])) - PUSH_RADIUS) / 500.0
                dz_m = (abs(float(rel[2])) - PUSH_HEIGHT) / 200.0
                mj = float(marg_join[cj, t]) if 1 <= t <= n_teams else 0.0
                ml = float(marg_leave[cj, t]) if 1 <= t <= n_teams else 0.0
                in_c = 1.0 if in_cart_id[bi] >= 0 else 0.0
            else:
                sigma = nu = cyl_m = dz_m = mj = ml = in_c = 0.0
            # Local Voronoi-foveated major item & rival signal
            fov_major = float(np.dot(fovea_w[bi], cell_tokens[:, 8])) if self.n_cells else 0.0
            fov_bots = float(np.dot(fovea_w[bi], cell_tokens[:, 7])) if self.n_cells else 0.0
            c_xy = self.cell_centroids[c] if self.n_cells else p
            bot_geo[bi] = [
                sigma, nu, cyl_m, dz_m,
                in_c, mj, ml,
                fov_major, fov_bots,
                (p[0] - c_xy[0]) / 500.0,
                (p[1] - c_xy[1]) / 500.0,
                float(c) / max(1.0, self.n_cells),
            ]

        return {
            "bot_geo": bot_geo,
            "cart_geo": cart_geo,
            "track_tokens": track_tokens,
            "track_teams": track_team_arr,
            "cell_tokens": cell_tokens,
            "item_tokens": item_tokens,
            "fovea_weights": fovea_w,
            "bot_cells": bot_cells,
            "map_center": self.map_center.copy(),
            "map_radius": np.asarray([self.map_radius], dtype=np.float32),
        }


def load_map_manifold(ent_path: str | Path, measurements_path: str | Path | None = None) -> MapManifold:
    ent_path = Path(ent_path)
    if measurements_path is None:
        measurements_path = Path(str(ent_path) + ".measurements.json")
    else:
        measurements_path = Path(measurements_path)

    meas = json.loads(measurements_path.read_text()) if measurements_path.exists() else {}
    nav = NavigationRows(meas)
    cp_dists_all = meas.get("checkpoint_distances", [])

    blocks = _parse_ent_blocks(ent_path.read_text())
    path_by_name = {b["targetname"]: b for b in blocks if b.get("classname") == "plc_path" and "targetname" in b}
    carts = [b for b in blocks if b.get("classname") == "func_plc_cart"]

    splines: list[CartSpline] = []
    for cid, cb in enumerate(carts):
        pts, arc_s, is_cp = [], [], []
        cur = path_by_name.get(cb.get("target", ""))
        seen = set()
        acc = 0.0
        prev_p = None
        while cur is not None:
            tname = cur.get("targetname", "")
            if tname in seen:
                break
            seen.add(tname)
            p = _parse_vec3(cur.get("origin", "0 0 0"))
            if prev_p is not None:
                acc += float(np.linalg.norm(p - prev_p))
            pts.append(p)
            arc_s.append(acc)
            sf = int(float(cur.get("spawnflags", "0") or 0))
            is_cp.append(1.0 if (sf & 1) else 0.0)
            prev_p = p
            cur = path_by_name.get(cur.get("target", ""))
        if len(pts) < 2:
            p0 = pts[0] if pts else _parse_vec3(cb.get("origin", "0 0 0"))
            pts = [p0, p0 + np.asarray([1.0, 0.0, 0.0], dtype=np.float32)]
            arc_s = [0.0, 1.0]
            is_cp = [0.0, 1.0]
        pts_arr = np.asarray(pts, dtype=np.float32)
        arc_arr = np.asarray(arc_s, dtype=np.float32)
        length = float(arc_arr[-1])
        tans = np.zeros_like(pts_arr)
        for i in range(len(pts_arr) - 1):
            d = pts_arr[i + 1] - pts_arr[i]
            nrm = float(np.linalg.norm(d))
            tans[i] = d / max(nrm, 1e-6)
        tans[-1] = tans[-2]

        cp_s = np.asarray([arc_arr[i] for i, f in enumerate(is_cp) if f > 0.5], dtype=np.float32)
        if len(cp_s) == 0 and cid < len(cp_dists_all):
            cp_s = np.asarray(cp_dists_all[cid], dtype=np.float32)
        cp_pos = np.zeros((len(cp_s), 3), dtype=np.float32)

        sp = CartSpline(
            cart_id=cid,
            points=pts_arr,
            arc_s=arc_arr,
            tangents=tans,
            length=length,
            checkpoints_s=cp_s,
            checkpoints_pos=cp_pos,
            sample_s=np.linspace(0.0, length, 5, dtype=np.float32),
            sample_pos=np.zeros((5, 3), dtype=np.float32),
            sample_tan=np.zeros((5, 3), dtype=np.float32),
            sample_is_cp=np.zeros((5,), dtype=np.float32),
        )
        for mi, s_val in enumerate(cp_s):
            sp.checkpoints_pos[mi], _ = sp.eval_at(float(s_val))
        for si, s_val in enumerate(sp.sample_s):
            sp.sample_pos[si], sp.sample_tan[si] = sp.eval_at(float(s_val))
            sp.sample_is_cp[si] = 1.0 if (len(cp_s) > 0 and float(np.min(np.abs(cp_s - s_val))) < 40.0) else 0.0
        splines.append(sp)

    item_pos, item_cls, item_names = [], [], []
    for b in blocks:
        cname = b.get("classname", "")
        if cname in MAJOR_ITEM_CLASS and "origin" in b:
            item_pos.append(_parse_vec3(b["origin"]))
            item_cls.append(MAJOR_ITEM_CLASS[cname])
            item_names.append(cname)
    item_pos_arr = np.asarray(item_pos, dtype=np.float32).reshape(-1, 3)
    item_cls_arr = np.asarray(item_cls, dtype=np.float32)

    if nav.map is not None and nav.map.n_cells > 0:
        cell_centroids = nav.map.centroids.astype(np.float32)
        cell_areas = nav.map.areas.astype(np.float32)
    else:
        targets = [b for b in blocks if b.get("classname") == "plc_nav_target" and "origin" in b]
        cell_centroids = np.asarray([_parse_vec3(b["origin"]) for b in targets], dtype=np.float32).reshape(-1, 3)
        cell_areas = np.ones((len(cell_centroids),), dtype=np.float32)

    return MapManifold(nav, splines, item_pos_arr, item_cls_arr, item_names, cell_centroids, cell_areas)
