#!/usr/bin/env python3
"""Cart-Free 5-Layer Phase-Space & Markov-Flow Behavioral Fingerprint.

Computes a structural, architecture-agnostic, cart-free behavioral signature directly from
live 6D `OBS` (`POS_X..Z`, `VEL_X..Z`, `CELL_X/Y`, `HEALTH`, `ARMOR`, `WEAPONS`, `ALIVE`)
and `EVT` (`DAMAGE`, `KILL`, `PICKUP`) streams — working identically across:
  - Pure 1v1 / 2v2 / 3v3 Deathmatch & Team Deathmatch (zero cartlanes)
  - Stock payload & B3313 real-map supermaps
  - Stock HavocBot vs every RL policy checkpoint throughout training

Five structural layers:
  L1. Spatial & Kinetic Field: stationary 256u nav-cell occupancy pi(c), bulk T/v_rms,
      speed quantiles, trajectory straightness (tortuosity), and >=1s stuck rate.
  L2. Directed Cell/Zone Markov Flux & Broken Detailed Balance: empirical transition flux
      F_{u->v}, net probability current J_{uv} = F_{u->v} - F_{v->u}, flux coherence
      ||J||_1 / ||F||_1, and entropy production rate sigma (separating aimless diffusion
      and 2-cell oscillation from genuine FPS rotation / item-timing cycles).
  L3. 2-Body Pair Correlation g(r, dz): ally spacing distribution (clumping / deathball),
      enemy engagement distance distribution, and vertical high-ground offset dz.
  L4. Combat-Conditioned Thermometry: distributions at EVT_DAMAGE and EVT_KILL of
      (r_eng, dz_highground, v_attacker, v_victim, stack_advantage_hp_ap).
  L5. Resource & Weapon Spectra: pickup rate, mean stack (HP+AP), weapon bitset entropy.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from payload.tools.strategy_io_schema import EVT, EVT_KIND, OBS

DIST_BINS = np.array([0.0, 160.0, 320.0, 512.0, 768.0, 1024.0, 1536.0, 2560.0, 1e9], dtype=np.float64)
DZ_BINS = np.array([-1e9, -96.0, -24.0, 24.0, 96.0, 1e9], dtype=np.float64)
SPEED_BINS = np.array([0.0, 60.0, 180.0, 300.0, 420.0, 560.0, 1e9], dtype=np.float64)


def _jsd_counts(ca: dict[Any, float], cb: dict[Any, float]) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1]) between two discrete count maps."""
    keys = set(ca.keys()) | set(cb.keys())
    if not keys:
        return 0.0
    sa = float(sum(ca.values()))
    sb = float(sum(cb.values()))
    if sa <= 0.0 or sb <= 0.0:
        return 1.0 if (sa > 0.0 or sb > 0.0) else 0.0
    jsd = 0.0
    for k in keys:
        pa = float(ca.get(k, 0.0)) / sa
        pb = float(cb.get(k, 0.0)) / sb
        m = 0.5 * (pa + pb)
        if pa > 0.0:
            jsd += 0.5 * pa * math.log2(pa / m)
        if pb > 0.0:
            jsd += 0.5 * pb * math.log2(pb / m)
    return float(max(0.0, min(1.0, jsd)))


def _l1_hist(ha: list[float], hb: list[float]) -> float:
    a = np.asarray(ha, dtype=np.float64)
    b = np.asarray(hb, dtype=np.float64)
    if a.sum() > 0:
        a = a / a.sum()
    if b.sum() > 0:
        b = b / b.sum()
    return float(0.5 * np.abs(a - b).sum())


class PhaseSpaceAccumulator:
    """Streaming accumulator for a 5-layer cart-free Phase-Space & Markov-Flow Fingerprint."""

    def __init__(self, label: str = "unnamed", cell_size: float = 256.0):
        self.label = label
        self.cell_size = float(cell_size)
        self.ticks = 0
        self.alive_samples = 0

        # L1: Spatial occupancy & kinetic field
        self.cell_occ: Counter[tuple[int, int]] = Counter()
        self.speed_hist = np.zeros((len(SPEED_BINS) - 1,), dtype=np.int64)
        self.v_sum = 0.0
        self.v2_sum = 0.0
        self.vz2_sum = 0.0
        self.straight_ratios: list[float] = []
        self.stuck_n = 0
        self.stuck_d = 0

        # Per-entity tracking for Markov flux & 1s trajectory straightness
        self._prev_cell: dict[int, tuple[int, int]] = {}
        self._prev_pos: dict[int, np.ndarray] = {}
        self._prev_alive: dict[int, bool] = {}
        self._win_start: dict[int, tuple[float, np.ndarray, float]] = {}  # (t0, p0, path_len)

        # L2: Directed cell transition flux F_{u -> v}
        self.flux: Counter[tuple[tuple[int, int], tuple[int, int]]] = Counter()
        self.cell_transitions = 0

        # L3: 2-body pair correlation histograms
        self.ally_r_hist = np.zeros((len(DIST_BINS) - 1,), dtype=np.int64)
        self.enemy_r_hist = np.zeros((len(DIST_BINS) - 1,), dtype=np.int64)
        self.enemy_dz_hist = np.zeros((len(DZ_BINS) - 1,), dtype=np.int64)

        # L4: Combat-conditioned thermometry
        self.dmg_events = 0
        self.kill_events = 0
        self.combat_r: list[float] = []
        self.combat_dz: list[float] = []
        self.combat_v_atk: list[float] = []
        self.combat_v_vic: list[float] = []
        self.combat_stack_adv: list[float] = []

        # L5: Resource & weapon spectra
        self.pickup_events = 0
        self.stack_sum = 0.0
        self.weapon_bits: Counter[int] = Counter()
        self._prev_wpn_sig: dict[int, int] = {}
        self.weapon_switches = 0

        self.t_first: float | None = None
        self.t_last: float | None = None

    def update(
        self,
        obs_raw: np.ndarray,
        evt_raw: np.ndarray | None = None,
        *,
        game_time: float | None = None,
        team_filter: set[int] | None = None,
    ) -> None:
        """Ingest one live `OBS` (+ optional `EVT`) tick from xonwire."""
        if obs_raw is None or len(obs_raw) == 0:
            return
        obs = obs_raw[obs_raw[:, OBS["PRESENT"]] > 0.5]
        if len(obs) == 0:
            return
        self.ticks += 1
        t_now = float(obs[0, OBS["ENGINE_TIME"]]) if game_time is None else float(game_time)
        if self.t_first is None:
            self.t_first = t_now
        self.t_last = t_now

        # Snapshot all alive entities by edict id for pair correlations & combat lookups
        alive_state: dict[int, tuple[int, np.ndarray, float, float]] = {}  # eid -> (team, pos, v_xy, stack)
        for row in obs:
            eid = int(row[OBS["ID"]])
            tm = int(row[OBS["TEAM"]])
            alive = row[OBS["ALIVE"]] > 0.5
            if not alive:
                self._prev_alive[eid] = False
                self._prev_cell.pop(eid, None)
                self._prev_pos.pop(eid, None)
                self._win_start.pop(eid, None)
                continue
            p = np.array([row[OBS["POS_X"]], row[OBS["POS_Y"]], row[OBS["POS_Z"]]], dtype=np.float64)
            v_xy2 = float(row[OBS["VEL_X"]] ** 2 + row[OBS["VEL_Y"]] ** 2)
            v_xy = math.sqrt(v_xy2)
            stack = float(max(0.0, row[OBS["HEALTH"]]) + max(0.0, row[OBS["ARMOR"]]))
            alive_state[eid] = (tm, p, v_xy, stack)

            if team_filter is not None and tm not in team_filter:
                continue

            self.alive_samples += 1
            self.v_sum += v_xy
            self.v2_sum += v_xy2
            self.vz2_sum += float(row[OBS["VEL_Z"]] ** 2)
            self.stack_sum += stack
            sbin = int(np.searchsorted(SPEED_BINS, v_xy, side="right") - 1)
            sbin = max(0, min(len(self.speed_hist) - 1, sbin))
            self.speed_hist[sbin] += 1

            # Discrete nav cell (use OBS CELL_X/Y if populated, else quantize POS)
            cx = int(math.floor(p[0] / self.cell_size))
            cy = int(math.floor(p[1] / self.cell_size))
            cell = (cx, cy)
            self.cell_occ[cell] += 1

            # L2: Directed Markov flux on cell transitions (ignore teleport jumps > 1200u in 1 tick? Keep portal jumps!)
            if self._prev_alive.get(eid, False) and eid in self._prev_cell:
                prev_c = self._prev_cell[eid]
                if prev_c != cell:
                    self.flux[(prev_c, cell)] += 1
                    self.cell_transitions += 1

            # L1: 1.0s trajectory straightness & stuck check
            p_old = self._prev_pos.get(eid)
            if not self._prev_alive.get(eid, False) or eid not in self._win_start:
                self._win_start[eid] = (t_now, p, 0.0)
            else:
                wt0, wp0, wlen = self._win_start[eid]
                step_d = float(np.linalg.norm(p - p_old)) if p_old is not None else 0.0
                if step_d < 800.0:  # ignore instantaneous portal jumps in path length
                    wlen += step_d
                dt = t_now - wt0
                if dt >= 1.0:
                    disp = float(np.linalg.norm(p - wp0))
                    self.stuck_d += 1
                    if disp < 24.0:
                        self.stuck_n += 1
                    if wlen >= 48.0:
                        self.straight_ratios.append(min(1.0, disp / max(wlen, 1e-6)))
                    self._win_start[eid] = (t_now, p, 0.0)
                else:
                    self._win_start[eid] = (wt0, wp0, wlen)

            # L5: Weapon bitset spectrum
            wx = int(round(float(row[OBS["WEAPONS_X"]])))
            wy = int(round(float(row[OBS["WEAPONS_Y"]])))
            wsig = (wx & 0xFFFFFF) ^ ((wy & 0xFFFFFF) << 24)
            for b in range(24):
                if wx & (1 << b):
                    self.weapon_bits[b] += 1
            if eid in self._prev_wpn_sig and self._prev_wpn_sig[eid] != wsig:
                self.weapon_switches += 1
            self._prev_wpn_sig[eid] = wsig

            self._prev_cell[eid] = cell
            self._prev_pos[eid] = p
            self._prev_alive[eid] = True

        # L3: 2-body pair correlations (subsample if N > 32 so O(N^2) stays instant)
        eids = list(alive_state.keys())
        if len(eids) > 32:
            eids = eids[:32]
        for i in range(len(eids)):
            ei = eids[i]
            ti, pi, _, _ = alive_state[ei]
            if team_filter is not None and ti not in team_filter:
                continue
            for j in range(i + 1, len(eids)):
                ej = eids[j]
                tj, pj, _, _ = alive_state[ej]
                r = float(np.linalg.norm(pi[:2] - pj[:2]))
                rbin = int(np.searchsorted(DIST_BINS, r, side="right") - 1)
                rbin = max(0, min(len(self.ally_r_hist) - 1, rbin))
                # In FFA DM (team == 0), everyone is an enemy
                same_team = (ti > 0 and ti == tj)
                if same_team:
                    self.ally_r_hist[rbin] += 1
                else:
                    self.enemy_r_hist[rbin] += 1
                    dz = float(pi[2] - pj[2])
                    zbin = int(np.searchsorted(DZ_BINS, dz, side="right") - 1)
                    zbin = max(0, min(len(self.enemy_dz_hist) - 1, zbin))
                    self.enemy_dz_hist[zbin] += 1

        # L4 & L5: Realized combat & pickup events
        if evt_raw is not None and len(evt_raw) > 0:
            for ev in evt_raw:
                kind = int(round(float(ev[EVT["KIND"]])))
                obs_id = int(round(float(ev[EVT["OBSERVER"]])))
                subj_id = int(round(float(ev[EVT["SUBJECT"]])))
                ev_tm = int(round(float(ev[EVT["TEAM"]])))
                if team_filter is not None and ev_tm not in team_filter:
                    continue
                if kind == EVT_KIND["PICKUP"]:
                    self.pickup_events += 1
                elif kind in (EVT_KIND["DAMAGE"], EVT_KIND["KILL"]):
                    if kind == EVT_KIND["DAMAGE"]:
                        self.dmg_events += 1
                    else:
                        self.kill_events += 1
                    sa = alive_state.get(obs_id)
                    sv = alive_state.get(subj_id)
                    if sa is not None and sv is not None:
                        self.combat_r.append(float(np.linalg.norm(sa[1][:2] - sv[1][:2])))
                        self.combat_dz.append(float(sa[1][2] - sv[1][2]))
                        self.combat_v_atk.append(sa[2])
                        self.combat_v_vic.append(sv[2])
                        self.combat_stack_adv.append(sa[3] - sv[3])

    def finalize(self) -> dict[str, Any]:
        """Return JSON-serializable 5-layer Phase-Space & Markov-Flow Fingerprint."""
        n = max(1, self.alive_samples)
        v_mean = self.v_sum / n
        v_rms = math.sqrt(self.v2_sum / n)
        vz_rms = math.sqrt(self.vz2_sum / n)
        mean_stack = self.stack_sum / n
        dur_s = max(1.0, (self.t_last - self.t_first) if (self.t_first is not None and self.t_last is not None) else 1.0)
        mean_players = self.alive_samples / max(1, self.ticks)

        # L1: Cell occupancy entropy
        tot_occ = float(sum(self.cell_occ.values())) or 1.0
        occ_ent = -sum((c / tot_occ) * math.log2(c / tot_occ) for c in self.cell_occ.values() if c > 0)

        # L2: Broken detailed balance / entropy production rate sigma & flux coherence
        undirected_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for u, v in self.flux.keys():
            undirected_pairs.add((u, v) if u < v else (v, u))
        tot_flux = float(sum(self.flux.values())) or 1.0
        net_j_l1 = 0.0
        sigma_ep = 0.0
        eps = 0.5  # Laplace smoothing for finite-sample KL
        for u, v in undirected_pairs:
            f_uv = float(self.flux.get((u, v), 0))
            f_vu = float(self.flux.get((v, u), 0))
            j_uv = f_uv - f_vu
            net_j_l1 += abs(j_uv)
            sigma_ep += (j_uv / tot_flux) * math.log((f_uv + eps) / (f_vu + eps))
        flux_coherence = net_j_l1 / tot_flux

        # L3: Normalized pair correlation histograms
        ally_tot = int(self.ally_r_hist.sum())
        enemy_tot = int(self.enemy_r_hist.sum())
        ally_clump_pct = round(100.0 * float(self.ally_r_hist[0]) / max(1, ally_tot), 2) if ally_tot > 0 else 0.0
        enemy_close_pct = round(100.0 * float(self.enemy_r_hist[:2].sum()) / max(1, enemy_tot), 2) if enemy_tot > 0 else 0.0

        # L5: Weapon entropy
        w_tot = float(sum(self.weapon_bits.values())) or 1.0
        wpn_ent = -sum((c / w_tot) * math.log2(c / w_tot) for c in self.weapon_bits.values() if c > 0)

        # Keep compact string-keyed top occupancy & flux maps for cross-policy divergence
        occ_map = {f"{c[0]},{c[1]}": int(v) for c, v in self.cell_occ.items()}
        flux_map = {f"{u[0]},{u[1]}->{v[0]},{v[1]}": int(cnt) for (u, v), cnt in self.flux.items()}

        return {
            "label": self.label,
            "ticks": self.ticks,
            "alive_samples": self.alive_samples,
            "duration_game_s": round(dur_s, 1),
            "mean_alive_players": round(mean_players, 2),
            "L1_kinetic_spatial": {
                "active_cells": len(self.cell_occ),
                "occupancy_entropy_bits": round(occ_ent, 3),
                "v_mean_u_s": round(v_mean, 1),
                "v_rms_u_s": round(v_rms, 1),
                "vz_rms_u_s": round(vz_rms, 1),
                "straightness_mean": round(float(np.mean(self.straight_ratios)), 3) if self.straight_ratios else None,
                "stuck_pct": round(100.0 * self.stuck_n / max(1, self.stuck_d), 2),
                "speed_hist": [int(x) for x in self.speed_hist.tolist()],
            },
            "L2_markov_flux": {
                "cell_transitions": self.cell_transitions,
                "transitions_per_player_s": round(self.cell_transitions / max(1.0, mean_players * dur_s), 3),
                "unique_directed_edges": len(self.flux),
                "flux_coherence_J_over_F": round(flux_coherence, 3),
                "entropy_production_sigma": round(sigma_ep, 4),
            },
            "L3_pair_correlation": {
                "ally_pairs": ally_tot,
                "enemy_pairs": enemy_tot,
                "ally_clump_160u_pct": ally_clump_pct,
                "enemy_close_320u_pct": enemy_close_pct,
                "ally_r_hist": [int(x) for x in self.ally_r_hist.tolist()],
                "enemy_r_hist": [int(x) for x in self.enemy_r_hist.tolist()],
                "enemy_dz_hist": [int(x) for x in self.enemy_dz_hist.tolist()],
            },
            "L4_combat_thermometry": {
                "damage_events": self.dmg_events,
                "kill_events": self.kill_events,
                "engage_r_mean": round(float(np.mean(self.combat_r)), 1) if self.combat_r else None,
                "engage_dz_mean": round(float(np.mean(self.combat_dz)), 1) if self.combat_dz else None,
                "v_attacker_mean": round(float(np.mean(self.combat_v_atk)), 1) if self.combat_v_atk else None,
                "v_victim_mean": round(float(np.mean(self.combat_v_vic)), 1) if self.combat_v_vic else None,
                "stack_adv_mean": round(float(np.mean(self.combat_stack_adv)), 1) if self.combat_stack_adv else None,
            },
            "L5_resource_weapon": {
                "pickup_events": self.pickup_events,
                "pickups_per_player_min": round(60.0 * self.pickup_events / max(1.0, mean_players * dur_s), 2),
                "mean_stack_hp_ap": round(mean_stack, 1),
                "weapon_entropy_bits": round(wpn_ent, 3),
                "weapon_switches_per_player_min": round(60.0 * self.weapon_switches / max(1.0, mean_players * dur_s), 2),
            },
            "_occ_map": occ_map,
            "_flux_map": flux_map,
        }


def compare_fingerprints(fp_cur: dict[str, Any], fp_ref: dict[str, Any]) -> dict[str, Any]:
    """Compute structural divergence & weirdness diagnostics between two PhaseSpaceFingerprints."""
    occ_jsd = _jsd_counts(fp_cur.get("_occ_map") or {}, fp_ref.get("_occ_map") or {})
    flux_jsd = _jsd_counts(fp_cur.get("_flux_map") or {}, fp_ref.get("_flux_map") or {})

    l1c, l1r = fp_cur["L1_kinetic_spatial"], fp_ref["L1_kinetic_spatial"]
    l2c, l2r = fp_cur["L2_markov_flux"], fp_ref["L2_markov_flux"]
    l3c, l3r = fp_cur["L3_pair_correlation"], fp_ref["L3_pair_correlation"]
    l4c, l4r = fp_cur["L4_combat_thermometry"], fp_ref["L4_combat_thermometry"]

    speed_l1 = _l1_hist(l1c["speed_hist"], l1r["speed_hist"])
    ally_l1 = _l1_hist(l3c["ally_r_hist"], l3r["ally_r_hist"]) if (l3c["ally_pairs"] > 0 and l3r["ally_pairs"] > 0) else 0.0
    enemy_l1 = _l1_hist(l3c["enemy_r_hist"], l3r["enemy_r_hist"]) if (l3c["enemy_pairs"] > 0 and l3r["enemy_pairs"] > 0) else 0.0
    dz_l1 = _l1_hist(l3c["enemy_dz_hist"], l3r["enemy_dz_hist"]) if (l3c["enemy_pairs"] > 0 and l3r["enemy_pairs"] > 0) else 0.0

    v_ratio = l1c["v_rms_u_s"] / max(1.0, l1r["v_rms_u_s"])
    st_c, st_r = l1c.get("straightness_mean"), l1r.get("straightness_mean")
    straight_ratio = (st_c / max(0.05, st_r)) if (st_c is not None and st_r is not None) else 1.0
    sigma_ratio = (l2c["entropy_production_sigma"] + 0.01) / (l2r["entropy_production_sigma"] + 0.01)
    stack_shift = (
        round(l4c["stack_adv_mean"] - l4r["stack_adv_mean"], 1)
        if (l4c.get("stack_adv_mean") is not None and l4r.get("stack_adv_mean") is not None)
        else None
    )

    # Composite structural distance & weirdness flags
    flags: list[str] = []
    if v_ratio < 0.78:
        flags.append("SPEED_COLLAPSE")
    if l1c["stuck_pct"] > max(5.0, l1r["stuck_pct"] + 4.0):
        flags.append("STUCK_SPIKE")
    if straight_ratio < 0.72:
        flags.append("TORTUOUS_ORBITING")
    if l3c["ally_pairs"] > 20 and l3c["ally_clump_160u_pct"] > max(20.0, l3r["ally_clump_160u_pct"] + 12.0):
        flags.append("ALLY_CLUMPING")
    if l1c["occupancy_entropy_bits"] < 0.75 * max(0.5, l1r["occupancy_entropy_bits"]):
        flags.append("MAP_CAMPING_COLLAPSE")
    if occ_jsd > 0.35:
        flags.append("HIGH_OCCUPANCY_SHIFT")

    weirdness = round(
        0.30 * occ_jsd
        + 0.25 * flux_jsd
        + 0.15 * speed_l1
        + 0.15 * enemy_l1
        + 0.15 * ally_l1
        + 0.20 * max(0.0, 1.0 - straight_ratio),
        4,
    )
    return {
        "cur": fp_cur["label"],
        "ref": fp_ref["label"],
        "occ_jsd": round(occ_jsd, 4),
        "flux_jsd": round(flux_jsd, 4),
        "speed_l1": round(speed_l1, 4),
        "ally_r_l1": round(ally_l1, 4),
        "enemy_r_l1": round(enemy_l1, 4),
        "enemy_dz_l1": round(dz_l1, 4),
        "v_rms_ratio": round(v_ratio, 3),
        "straightness_ratio": round(straight_ratio, 3),
        "sigma_ratio": round(sigma_ratio, 3),
        "combat_stack_adv_shift": stack_shift,
        "weirdness_index": weirdness,
        "flags": flags,
    }


def strip_raw_maps(fp: dict[str, Any]) -> dict[str, Any]:
    """Return compact summary copy without large raw cell/edge dicts for concise JSON logging."""
    return {k: v for k, v in fp.items() if not k.startswith("_")}
