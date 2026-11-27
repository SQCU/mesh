from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from payload.tools.strategy_io_schema import CART_WIDTH, CS, OBS, OBS_WIDTH, TEAM_WIDTH, TS

from .world_graph import WorldGraphIR


@dataclass(frozen=True)
class SimRulesConfig:
    """Exact rule constants transcribed from gamemodes-payload.cfg & sv_payload.qc."""
    point_limit_base: float = 1200.0
    checkpoint_rate: float = 1.0
    cart_base_speed: float = 30.0
    cart_max_speed: float = 200.0
    push_cap: int = 3
    push_falloff: float = 1.0
    contest_speed: float = 24.0
    reverse_speed: float = 14.0
    score_rate: float = 0.01
    block_rate: float = 1.0
    stall_eps: float = 1.0
    plc_tick_ref: float = 0.05

    # Agent kinematics & HavocBot parameters
    move_speed: float = 360.0
    replan_interval: float = 2.5
    plan_replan_hysteresis: float = 2.0
    respawn_delay: float = 3.0
    max_health: float = 100.0
    start_armor: float = 25.0
    armor_absorb: float = 0.66
    combat_range: float = 950.0
    base_dps: float = 42.0
    skill_dps_scale: float = 0.06
    dps_stochastic_cv: float = 0.25


@dataclass
class SimSnapshot:
    """Flat in-memory savestate for O(1) restore and GRPO counterfactual branching."""
    sim_time: torch.Tensor
    step_count: torch.Tensor
    seed: torch.Tensor
    p_pos: torch.Tensor
    p_vel: torch.Tensor
    p_team: torch.Tensor
    p_hp: torch.Tensor
    p_armor: torch.Tensor
    p_alive: torch.Tensor
    p_respawn_t: torch.Tensor
    p_spawn_t: torch.Tensor
    p_skill: torch.Tensor
    p_wp_cur: torch.Tensor
    p_wp_next: torch.Tensor
    p_wp_goal: torch.Tensor
    p_edge_rem: torch.Tensor
    p_goal_cart: torch.Tensor
    p_goal_timer: torch.Tensor
    p_replan_timer: torch.Tensor
    p_last_pick: torch.Tensor
    p_plan: torch.Tensor
    p_has_plan: torch.Tensor
    p_dmg_dealt: torch.Tensor
    p_dmg_taken: torch.Tensor
    p_kills: torch.Tensor
    p_deaths: torch.Tensor
    p_pickups: torch.Tensor
    p_cart_push: torch.Tensor
    p_cart_contest: torch.Tensor
    c_s: torch.Tensor
    c_pos: torch.Tensor
    c_ctrl: torch.Tensor
    c_speed: torch.Tensor
    c_idle: torch.Tensor
    c_lead: torch.Tensor
    c_second: torch.Tensor
    c_home: torch.Tensor
    c_cp_depth: torch.Tensor
    c_cp_owner: torch.Tensor
    t_score: torch.Tensor
    t_cp_held: torch.Tensor
    t_finished: torch.Tensor
    t_winner: torch.Tensor


def _philox_u01(seed: torch.Tensor, counter: torch.Tensor, salt: int, shape: tuple[int, ...]) -> torch.Tensor:
    """Stateless counter-based deterministic PRNG in [0, 1) — identical across CPU and MPS."""
    dev = seed.device
    n_elem = 1
    for s in shape[1:]:
        n_elem *= s
    idx = torch.arange(n_elem, device=dev, dtype=torch.int64).view(1, *shape[1:])
    # 32-bit Weyl / splitmix-like hash using int64 arithmetic (supported natively on MPS & CPU)
    x = (seed.view(-1, *([1] * (len(shape) - 1))) * 0x9E3779B1 + counter.view(-1, *([1] * (len(shape) - 1))) * 0x85EBCA77 + idx * 0xC2B2AE3D + int(salt)) & 0x7FFFFFFF
    x = ((x ^ (x >> 15)) * 0x2C1B3C6D) & 0x7FFFFFFF
    x = ((x ^ (x >> 12)) * 0x297A2D39) & 0x7FFFFFFF
    x = (x ^ (x >> 15)) & 0x00FFFFFF
    return x.to(torch.float32) * (1.0 / 16777216.0)


class DeltaTimePayloadEngine:
    """
    SIMD / MPS-batched variable-dt Payload engine.

    - Pure Structure-of-Arrays tensors over [E, N, ...], [E, J, ...], [E, K, ...] on unified memory.
    - Exact dt-parameterized integration rules identical to sv_payload.qc at dt = PLC_TICK = 0.05s,
      with analytical score-out & checkpoint crossing root integrals for arbitrary macro dt.
    - Zero-copy observation tensors matching strategy_io_schema.py (OBS_WIDTH=84, CART_WIDTH=18, TEAM_WIDTH=7).
    - O(1) snapshot / restore / fork_group for mid-match GRPO counterfactual rollout groups.
    """

    def __init__(
        self,
        world: WorldGraphIR,
        num_envs: int,
        num_players: int | None = None,
        *,
        rules: SimRulesConfig = SimRulesConfig(),
        device: torch.device | str | None = None,
        base_seed: int = 1,
    ):
        self.device = torch.device(device) if device is not None else world.device
        self.world = world.to(self.device) if world.device != self.device else world
        self.rules = rules
        self.E = int(num_envs)
        self.K = int(self.world.teams)
        self.J = int(self.world.carts)
        self.N = int(num_players if num_players is not None else self.world.players_default)
        self.CP = int(self.world.max_checkpoints)
        self.M_s = int(self.world.cart_sample_wp.shape[1])

        per_team_carts = self.J / float(max(1, self.K))
        self.point_limit = float(self.rules.point_limit_base * max(1.0, per_team_carts))

        # Precompute constant indices
        self._env_idx = torch.arange(self.E, device=self.device, dtype=torch.int64)
        self._p_idx = torch.arange(self.N, device=self.device, dtype=torch.int64)
        self._j_idx = torch.arange(self.J, device=self.device, dtype=torch.int64)
        self._k_ids = torch.arange(1, self.K + 1, device=self.device, dtype=torch.int64)
        # Archetype and default pref_cart matching sv_payload.qc:809-810 (etof = 1..N)
        etof = torch.arange(1, self.N + 1, device=self.device, dtype=torch.int64)
        self._etof = etof
        self._archetype = etof % 4
        self._pref_cart = etof % max(1, self.J)

        # Precompute exact geometric push weight LUT for counts 0..push_cap (sv_payload.qc:146-153)
        cap = max(1, int(self.rules.push_cap))
        lut = [0.0]
        acc, term = 0.0, 1.0
        for _ in range(cap):
            acc += term
            term *= float(self.rules.push_falloff)
            lut.append(acc)
        self._push_weight_lut = torch.tensor(lut, device=self.device, dtype=torch.float32)

        # Persistent zero-alloc observation scratch buffers
        self._obs_buf = torch.zeros((self.E, self.N, OBS_WIDTH), device=self.device, dtype=torch.float32)
        self._cart_buf = torch.zeros((self.E, self.J, CART_WIDTH), device=self.device, dtype=torch.float32)
        self._team_buf = torch.zeros((self.E, self.K, TEAM_WIDTH), device=self.device, dtype=torch.float32)

        self.reset(base_seed=base_seed)

    def _eval_cart_pos(self, c_s: torch.Tensor) -> torch.Tensor:
        """Vectorized piecewise-linear spline lookup P_c(s) for [E, J] -> [E, J, 3]."""
        s_clamped = c_s.clamp(min=0.0).minimum(self.world.cart_length[None, :])
        # cart_node_s: [J, S_max]
        node_s = self.world.cart_node_s[None, :, :]  # [1, J, S_max]
        # Segment index = count of nodes with node_s <= s, minus 1, clamped to [0, seg_count - 1]
        seg = (node_s <= s_clamped.unsqueeze(-1)).sum(dim=-1) - 1
        max_seg = (self.world.cart_seg_count - 1).clamp_min(0)[None, :]
        seg = seg.clamp(min=0).minimum(max_seg)  # [E, J]
        j_bc = self._j_idx[None, :].expand(self.E, self.J)
        s0 = self.world.cart_node_s[j_bc, seg]
        s1 = self.world.cart_node_s[j_bc, seg + 1]
        frac = ((s_clamped - s0) / (s1 - s0).clamp_min(1e-6)).clamp(0.0, 1.0)
        p0 = self.world.cart_nodes[j_bc, seg]
        p1 = self.world.cart_nodes[j_bc, seg + 1]
        return p0 + frac.unsqueeze(-1) * (p1 - p0)

    def _cart_nearest_wp(self, c_s: torch.Tensor) -> torch.Tensor:
        """O(1) table lookup for nearest waypoint to each cart's arc-length [E, J] -> [E, J] int64."""
        frac = (c_s / self.world.cart_length[None, :].clamp_min(1.0)).clamp(0.0, 1.0)
        bi = torch.round(frac * float(self.M_s - 1)).to(torch.int64).clamp(0, self.M_s - 1)
        j_bc = self._j_idx[None, :].expand(self.E, self.J)
        return self.world.cart_sample_wp[j_bc, bi]

    @torch.no_grad()
    def reset(
        self,
        env_mask: torch.Tensor | None = None,
        *,
        base_seed: int | None = None,
        skill_mean: float = 4.0,
        skill_std: float = 0.0,
    ) -> None:
        dev = self.device
        E, N, J, K, CP = self.E, self.N, self.J, self.K, self.CP
        full = env_mask is None or not hasattr(self, "sim_time")

        if full:
            self.sim_time = torch.zeros((E,), device=dev, dtype=torch.float32)
            self.step_count = torch.zeros((E,), device=dev, dtype=torch.int64)
            s0 = 1 if base_seed is None else int(base_seed)
            self.seed = torch.arange(s0, s0 + E, device=dev, dtype=torch.int64)

            # Balanced team assignment 1..K across players 0..N-1
            teams_1d = (torch.arange(N, device=dev, dtype=torch.int64) % K) + 1
            self.p_team = teams_1d[None, :].expand(E, N).clone()

            # Assign spawn slot per player
            n_sp = int(self.world.spawn_pos.shape[1])
            sp_slot = (torch.arange(N, device=dev, dtype=torch.int64) // K) % n_sp
            t0 = self.p_team - 1
            self.p_pos = self.world.spawn_pos[t0, sp_slot[None, :].expand(E, N)].clone()
            self.p_vel = torch.zeros((E, N, 3), device=dev, dtype=torch.float32)
            self.p_hp = torch.full((E, N), self.rules.max_health, device=dev, dtype=torch.float32)
            self.p_armor = torch.full((E, N), self.rules.start_armor, device=dev, dtype=torch.float32)
            self.p_alive = torch.ones((E, N), device=dev, dtype=torch.bool)
            self.p_respawn_t = torch.zeros((E, N), device=dev, dtype=torch.float32)
            self.p_spawn_t = torch.zeros((E, N), device=dev, dtype=torch.float32)
            if skill_std > 0.0:
                u = _philox_u01(self.seed, self.step_count, 101, (E, N))
                self.p_skill = (skill_mean + skill_std * (2.0 * u - 1.0)).clamp(0.0, 20.0)
            else:
                self.p_skill = torch.full((E, N), float(skill_mean), device=dev, dtype=torch.float32)

            init_wp = self.world.spawn_wp[t0, sp_slot[None, :].expand(E, N)].clone()
            self.p_wp_cur = init_wp
            self.p_wp_next = init_wp.clone()
            self.p_wp_goal = init_wp.clone()
            self.p_edge_rem = torch.zeros((E, N), device=dev, dtype=torch.float32)
            self.p_goal_cart = self._pref_cart[None, :].expand(E, N).clone()
            self.p_goal_timer = torch.zeros((E, N), device=dev, dtype=torch.float32)
            self.p_replan_timer = torch.zeros((E, N), device=dev, dtype=torch.float32)
            self.p_last_pick = torch.full((E, N), -1, device=dev, dtype=torch.int64)
            self.p_plan = torch.zeros((E, N, 16), device=dev, dtype=torch.float32)
            self.p_has_plan = torch.zeros((E, N), device=dev, dtype=torch.bool)

            self.p_dmg_dealt = torch.zeros((E, N), device=dev, dtype=torch.float32)
            self.p_dmg_taken = torch.zeros((E, N), device=dev, dtype=torch.float32)
            self.p_kills = torch.zeros((E, N), device=dev, dtype=torch.float32)
            self.p_deaths = torch.zeros((E, N), device=dev, dtype=torch.float32)
            self.p_pickups = torch.zeros((E, N), device=dev, dtype=torch.float32)
            self.p_cart_push = torch.zeros((E, N), device=dev, dtype=torch.float32)
            self.p_cart_contest = torch.zeros((E, N), device=dev, dtype=torch.float32)

            self.c_s = torch.zeros((E, J), device=dev, dtype=torch.float32)
            self.c_pos = self._eval_cart_pos(self.c_s)
            self.c_ctrl = torch.zeros((E, J), device=dev, dtype=torch.int64)
            self.c_speed = torch.zeros((E, J), device=dev, dtype=torch.float32)
            self.c_idle = torch.zeros((E, J), device=dev, dtype=torch.float32)
            self.c_lead = torch.zeros((E, J), device=dev, dtype=torch.int64)
            self.c_second = torch.zeros((E, J), device=dev, dtype=torch.int64)
            self.c_home = torch.zeros((E, J), device=dev, dtype=torch.int64)
            self.c_cp_depth = torch.zeros((E, J), device=dev, dtype=torch.int64)
            self.c_cp_owner = torch.zeros((E, J, CP), device=dev, dtype=torch.int64)

            self.t_score = torch.zeros((E, K), device=dev, dtype=torch.float32)
            self.t_cp_held = torch.zeros((E, K), device=dev, dtype=torch.int64)
            self.t_finished = torch.zeros((E,), device=dev, dtype=torch.bool)
            self.t_winner = torch.zeros((E,), device=dev, dtype=torch.int64)
            return

        # Partial masked reset of finished/selected environments (zero reallocation!)
        m = env_mask.to(device=dev, dtype=torch.bool)
        if not bool(m.any().item()):
            return
        m1 = m
        m2 = m[:, None]
        m3 = m[:, None, None]
        self.sim_time = torch.where(m1, torch.zeros_like(self.sim_time), self.sim_time)
        self.step_count = torch.where(m1, torch.zeros_like(self.step_count), self.step_count)
        if base_seed is not None:
            self.seed = torch.where(m1, self.seed + int(base_seed), self.seed)
        else:
            self.seed = torch.where(m1, self.seed + 1000003, self.seed)

        n_sp = int(self.world.spawn_pos.shape[1])
        sp_slot = (torch.arange(N, device=dev, dtype=torch.int64) // K) % n_sp
        t0 = self.p_team - 1
        sp_pos = self.world.spawn_pos[t0, sp_slot[None, :].expand(E, N)]
        sp_wp = self.world.spawn_wp[t0, sp_slot[None, :].expand(E, N)]
        self.p_pos = torch.where(m3, sp_pos, self.p_pos)
        self.p_vel = torch.where(m3, torch.zeros_like(self.p_vel), self.p_vel)
        self.p_hp = torch.where(m2, torch.full_like(self.p_hp, self.rules.max_health), self.p_hp)
        self.p_armor = torch.where(m2, torch.full_like(self.p_armor, self.rules.start_armor), self.p_armor)
        self.p_alive = torch.where(m2, torch.ones_like(self.p_alive), self.p_alive)
        self.p_respawn_t = torch.where(m2, torch.zeros_like(self.p_respawn_t), self.p_respawn_t)
        self.p_spawn_t = torch.where(m2, torch.zeros_like(self.p_spawn_t), self.p_spawn_t)
        self.p_wp_cur = torch.where(m2, sp_wp, self.p_wp_cur)
        self.p_wp_next = torch.where(m2, sp_wp, self.p_wp_next)
        self.p_wp_goal = torch.where(m2, sp_wp, self.p_wp_goal)
        self.p_edge_rem = torch.where(m2, torch.zeros_like(self.p_edge_rem), self.p_edge_rem)
        self.p_goal_cart = torch.where(m2, self._pref_cart[None, :].expand(E, N), self.p_goal_cart)
        self.p_goal_timer = torch.where(m2, torch.zeros_like(self.p_goal_timer), self.p_goal_timer)
        self.p_replan_timer = torch.where(m2, torch.zeros_like(self.p_replan_timer), self.p_replan_timer)
        self.p_last_pick = torch.where(m2, torch.full_like(self.p_last_pick, -1), self.p_last_pick)
        self.p_has_plan = torch.where(m2, torch.zeros_like(self.p_has_plan), self.p_has_plan)

        self.c_s = torch.where(m2, torch.zeros_like(self.c_s), self.c_s)
        self.c_pos = self._eval_cart_pos(self.c_s)
        self.c_ctrl = torch.where(m2, torch.zeros_like(self.c_ctrl), self.c_ctrl)
        self.c_speed = torch.where(m2, torch.zeros_like(self.c_speed), self.c_speed)
        self.c_idle = torch.where(m2, torch.zeros_like(self.c_idle), self.c_idle)
        self.c_lead = torch.where(m2, torch.zeros_like(self.c_lead), self.c_lead)
        self.c_second = torch.where(m2, torch.zeros_like(self.c_second), self.c_second)
        self.c_home = torch.where(m2, torch.zeros_like(self.c_home), self.c_home)
        self.c_cp_depth = torch.where(m2, torch.zeros_like(self.c_cp_depth), self.c_cp_depth)
        self.c_cp_owner = torch.where(m3, torch.zeros_like(self.c_cp_owner), self.c_cp_owner)

        self.t_score = torch.where(m2, torch.zeros_like(self.t_score), self.t_score)
        self.t_cp_held = torch.where(m2, torch.zeros_like(self.t_cp_held), self.t_cp_held)
        self.t_finished = torch.where(m1, torch.zeros_like(self.t_finished), self.t_finished)
        self.t_winner = torch.where(m1, torch.zeros_like(self.t_winner), self.t_winner)

    @torch.no_grad()
    def snapshot(self, env_ids: torch.Tensor | None = None) -> SimSnapshot:
        """O(1) flat GPU/CPU tensor snapshot of full match state."""
        sl = slice(None) if env_ids is None else env_ids.to(self.device)
        return SimSnapshot(
            sim_time=self.sim_time[sl].clone(),
            step_count=self.step_count[sl].clone(),
            seed=self.seed[sl].clone(),
            p_pos=self.p_pos[sl].clone(),
            p_vel=self.p_vel[sl].clone(),
            p_team=self.p_team[sl].clone(),
            p_hp=self.p_hp[sl].clone(),
            p_armor=self.p_armor[sl].clone(),
            p_alive=self.p_alive[sl].clone(),
            p_respawn_t=self.p_respawn_t[sl].clone(),
            p_spawn_t=self.p_spawn_t[sl].clone(),
            p_skill=self.p_skill[sl].clone(),
            p_wp_cur=self.p_wp_cur[sl].clone(),
            p_wp_next=self.p_wp_next[sl].clone(),
            p_wp_goal=self.p_wp_goal[sl].clone(),
            p_edge_rem=self.p_edge_rem[sl].clone(),
            p_goal_cart=self.p_goal_cart[sl].clone(),
            p_goal_timer=self.p_goal_timer[sl].clone(),
            p_replan_timer=self.p_replan_timer[sl].clone(),
            p_last_pick=self.p_last_pick[sl].clone(),
            p_plan=self.p_plan[sl].clone(),
            p_has_plan=self.p_has_plan[sl].clone(),
            p_dmg_dealt=self.p_dmg_dealt[sl].clone(),
            p_dmg_taken=self.p_dmg_taken[sl].clone(),
            p_kills=self.p_kills[sl].clone(),
            p_deaths=self.p_deaths[sl].clone(),
            p_pickups=self.p_pickups[sl].clone(),
            p_cart_push=self.p_cart_push[sl].clone(),
            p_cart_contest=self.p_cart_contest[sl].clone(),
            c_s=self.c_s[sl].clone(),
            c_pos=self.c_pos[sl].clone(),
            c_ctrl=self.c_ctrl[sl].clone(),
            c_speed=self.c_speed[sl].clone(),
            c_idle=self.c_idle[sl].clone(),
            c_lead=self.c_lead[sl].clone(),
            c_second=self.c_second[sl].clone(),
            c_home=self.c_home[sl].clone(),
            c_cp_depth=self.c_cp_depth[sl].clone(),
            c_cp_owner=self.c_cp_owner[sl].clone(),
            t_score=self.t_score[sl].clone(),
            t_cp_held=self.t_cp_held[sl].clone(),
            t_finished=self.t_finished[sl].clone(),
            t_winner=self.t_winner[sl].clone(),
        )

    @torch.no_grad()
    def restore(self, snap: SimSnapshot) -> None:
        assert int(snap.sim_time.shape[0]) == self.E, f"snapshot batch {snap.sim_time.shape[0]} != engine E={self.E}"
        for field in SimSnapshot.__dataclass_fields__:
            setattr(self, field, getattr(snap, field).to(self.device).clone())

    @classmethod
    @torch.no_grad()
    def fork_group(
        cls,
        world: WorldGraphIR,
        snap: SimSnapshot,
        group_size: int,
        *,
        rules: SimRulesConfig = SimRulesConfig(),
        common_random_numbers: bool = False,
        seed_stride: int = 10007,
    ) -> "DeltaTimePayloadEngine":
        """Fork B snapshots into B*G parallel counterfactual branches for GRPO/tree rollouts."""
        B = int(snap.sim_time.shape[0])
        G = int(group_size)
        N = int(snap.p_pos.shape[1])
        eng = cls(world, num_envs=B * G, num_players=N, rules=rules, device=snap.sim_time.device)
        for field in SimSnapshot.__dataclass_fields__:
            val = getattr(snap, field).to(eng.device)
            # Repeat each base state G times consecutively: [B, ...] -> [B*G, ...]
            rep = val.repeat_interleave(G, dim=0).clone()
            setattr(eng, field, rep)
        if not common_random_numbers:
            g_off = torch.arange(G, device=eng.device, dtype=torch.int64).repeat(B) * int(seed_stride)
            eng.seed = eng.seed + g_off
        return eng

    @torch.no_grad()
    def apply_plan(self, plan: torch.Tensor, active_mask: torch.Tensor | None = None) -> None:
        """Apply [E, N, 16] PLAN tensor directly on device (matching sv_payload_strategy_io.qc:638-658)."""
        plan_d = plan.to(device=self.device, dtype=torch.float32)
        if active_mask is None:
            valid = plan_d[:, :, 0] > 0.0
        else:
            valid = active_mask.to(device=self.device, dtype=torch.bool)
        self.p_plan = torch.where(valid.unsqueeze(-1), plan_d, self.p_plan)
        self.p_has_plan = valid

        # Replan hysteresis on pick change (sv_payload_strategy_io.qc:644-650)
        pick_tag = torch.floor(plan_d[:, :, 1]).to(torch.int64) + 1
        changed = valid & (self.p_last_pick != pick_tag) & (self.p_replan_timer <= 0.0)
        self.p_last_pick = torch.where(changed, pick_tag, self.p_last_pick)
        self.p_replan_timer = torch.where(changed, torch.full_like(self.p_replan_timer, self.rules.plan_replan_hysteresis), self.p_replan_timer)
        self.p_goal_timer = torch.where(changed, self.p_goal_timer.clamp(max=0.35), self.p_goal_timer)

    @torch.no_grad()
    def _integrate_score(self, dt_vec: torch.Tensor) -> torch.Tensor:
        """Exact transcription of sv_payload.qc:188-211 & 272-291 (analytical score-out root)."""
        rates = self.t_cp_held.to(torch.float32) * float(self.rules.checkpoint_rate)  # [E, K]
        rem_pts = (self.point_limit - self.t_score).clamp_min(0.0)
        can_win = (rates > 0.0) | (self.t_score >= self.point_limit)
        tau = torch.where(
            self.t_score >= self.point_limit,
            torch.zeros_like(rem_pts),
            torch.where(rates > 0.0, rem_pts / rates.clamp_min(1e-9), torch.full_like(rem_pts, 1e12)),
        )
        min_tau, argmin_k = tau.min(dim=-1)  # [E]
        any_proj = can_win.any(dim=-1)
        ties = (can_win & (tau <= min_tau.unsqueeze(-1) + 1e-6)).sum(dim=-1)
        hit_win = (~self.t_finished) & any_proj & (min_tau <= dt_vec)

        dt_eff = torch.where(self.t_finished, torch.zeros_like(dt_vec), torch.where(hit_win, min_tau, dt_vec))
        self.t_score = self.t_score + dt_eff.unsqueeze(-1) * rates
        winner_id = torch.where(ties == 1, argmin_k + 1, torch.zeros_like(argmin_k))
        self.t_winner = torch.where(hit_win, winner_id, self.t_winner)
        self.t_finished = self.t_finished | hit_win
        return dt_eff

    @torch.no_grad()
    def _step_carts_and_checkpoints(
        self,
        dt_vec: torch.Tensor,
        *,
        exact_crossing_score: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact transcription of sv_payload.qc:118-155 & 333-447 with analytical checkpoint root integration."""
        E, N, J, K, CP = self.E, self.N, self.J, self.K, self.CP
        # Cylindrical occupancy check: [E, N, J]
        diff = self.p_pos[:, :, None, :] - self.c_pos[:, None, :, :]
        in_z = diff[..., 2].abs() <= self.world.cart_height[None, None, :]
        in_xy = (diff[..., 0].square() + diff[..., 1].square()) <= self.world.cart_radius[None, None, :].square()
        in_cart = self.p_alive[:, :, None] & in_z & in_xy  # [E, N, J]

        # Per-team player counts at each cart: [E, J, K]
        team_oh = (self.p_team[:, :, None] == self._k_ids[None, None, :]).to(torch.float32)  # [E, N, K]
        n_jk = torch.einsum("enj,enk->ejk", in_cart.to(torch.float32), team_oh)  # [E, J, K]
        n_jk_int = n_jk.round().to(torch.int64).clamp(0, self.rules.push_cap)
        w_jk = self._push_weight_lut[n_jk_int]  # [E, J, K]

        total_w = w_jk.sum(dim=-1)  # [E, J]
        if K >= 2:
            top2_w, top2_idx = torch.topk(w_jk, k=2, dim=-1)
            best_w = top2_w[..., 0]
            second_w = top2_w[..., 1]
            lead_team = torch.where(best_w > 0.0, top2_idx[..., 0] + 1, torch.zeros_like(top2_idx[..., 0]))
            second_team = torch.where(second_w > 0.0, top2_idx[..., 1] + 1, torch.zeros_like(top2_idx[..., 1]))
            tie = (best_w > 0.0) & (second_w == best_w)
        else:
            best_w = w_jk[..., 0]
            second_w = torch.zeros_like(best_w)
            lead_team = (best_w > 0.0).to(torch.int64)
            second_team = torch.zeros_like(lead_team)
            tie = torch.zeros_like(lead_team, dtype=torch.bool)

        ctrl = torch.where(tie, torch.zeros_like(lead_team), lead_team)
        color = torch.where((self.c_ctrl == 0) & (ctrl > 0), ctrl, self.c_ctrl)
        self.c_ctrl = color

        # wA = weight of controlling team, wopp = total - wA, opp = strongest non-controlling team
        color_mask = (color.unsqueeze(-1) == self._k_ids[None, None, :])  # [E, J, K]
        wA = (w_jk * color_mask.to(torch.float32)).sum(dim=-1)
        wopp = total_w - wA
        opp_w_masked = w_jk.masked_fill(color_mask, -1.0)
        oppw_raw, opp_idx = opp_w_masked.max(dim=-1)
        oppw = oppw_raw.clamp_min(0.0)
        opp_team = torch.where(oppw > 0.0, opp_idx + 1, torch.zeros_like(opp_idx))

        self.c_lead = lead_team
        self.c_second = second_team
        self.c_home = opp_team
        self.c_idle = torch.where(total_w <= 0.0, self.c_idle + dt_vec.unsqueeze(-1), torch.zeros_like(self.c_idle))

        max_v = float(self.rules.cart_max_speed) * self.world.cart_speed_scale[None, :].clamp_min(1.0)
        v_fwd = (float(self.rules.cart_base_speed) * (wA - wopp) / (1.0 + wopp * wopp)).clamp_min(-float(self.rules.contest_speed)).minimum(max_v)
        v_rev = (-float(self.rules.reverse_speed) * oppw).clamp(-max_v, torch.zeros_like(max_v))
        v = torch.where(
            (color > 0) & (wA > 0.0),
            v_fwd,
            torch.where((color > 0) & (oppw > 0.0), v_rev, torch.zeros_like(wA)),
        )
        live = (~self.t_finished).unsqueeze(-1)
        v = torch.where(live, v, torch.zeros_like(v))

        s_prev = self.c_s
        c_len = self.world.cart_length[None, :]
        dt_ej = dt_vec.unsqueeze(-1)
        s_new = (s_prev + v * dt_ej).clamp(min=0.0).minimum(c_len)
        self.c_s = s_new
        self.c_speed = (s_new - s_prev) / dt_ej.clamp_min(1e-6)
        self.c_pos = self._eval_cart_pos(s_new)

        # Analytical checkpoint crossing & score-integral correction over [t, t + dt]
        cp_d = self.world.cart_cp_dists[None, :, :]  # [1, J, CP]
        cp_valid = cp_d < 1e8
        prev_owner = self.c_cp_owner  # [E, J, CP]
        new_owner = torch.where(cp_valid & (cp_d <= s_new.unsqueeze(-1)), color.unsqueeze(-1), torch.zeros_like(prev_owner))

        # Origin regression clears progress & resets cart to neutral (sv_payload.qc:421-427)
        regressed = (s_new <= 0.0) & (s_prev > 0.0)
        new_owner = torch.where(regressed.unsqueeze(-1), torch.zeros_like(new_owner), new_owner)
        self.c_ctrl = torch.where(regressed, torch.zeros_like(self.c_ctrl), self.c_ctrl)

        if exact_crossing_score:
            # Branchless exact analytical sub-interval score integral for newly crossed/lost checkpoints
            v_nz = torch.where(v.abs() > 1e-6, v, torch.ones_like(v)).unsqueeze(-1)
            tau_cross = ((cp_d - s_prev.unsqueeze(-1)) / v_nz).clamp(0.0, 1e9).minimum(dt_ej.unsqueeze(-1))
            rem_t = (dt_ej.unsqueeze(-1) - tau_cross).clamp_min(0.0) * live.unsqueeze(-1).to(torch.float32)
            gained = cp_valid & (new_owner > 0) & (prev_owner == 0)
            lost = cp_valid & (new_owner == 0) & (prev_owner > 0)
            cp_rate = float(self.rules.checkpoint_rate)
            g_oh = (new_owner.unsqueeze(-1) == self._k_ids[None, None, None, :]) & gained.unsqueeze(-1)
            l_oh = (prev_owner.unsqueeze(-1) == self._k_ids[None, None, None, :]) & lost.unsqueeze(-1)
            delta_pts = ((g_oh.to(torch.float32) - l_oh.to(torch.float32)) * rem_t.unsqueeze(-1) * cp_rate).sum(dim=(1, 2))
            self.t_score = (self.t_score + delta_pts).clamp_min(0.0)
            hit_post = (~self.t_finished) & (self.t_score >= self.point_limit).any(dim=-1)
            best_k = self.t_score.argmax(dim=-1) + 1
            self.t_winner = torch.where(hit_post, best_k, self.t_winner)
            self.t_finished = self.t_finished | hit_post

        self.c_cp_owner = new_owner
        self.c_cp_depth = (new_owner > 0).sum(dim=-1)
        # Recompute checkpoints held per team [E, K]
        own_k = (new_owner.unsqueeze(-1) == self._k_ids[None, None, None, :]).sum(dim=(1, 2))
        self.t_cp_held = own_k

        # Per-player push & contest outcome accumulators (sv_payload.qc:429-447)
        ds_pos = (s_new - s_prev).clamp_min(0.0)  # [E, J]
        # Team count of controlling team on each cart
        n_ctrl = (n_jk * color_mask.to(torch.float32)).sum(dim=-1).clamp_min(1.0)
        push_share = torch.where((ds_pos > 0.0) & (color > 0), ds_pos / n_ctrl, torch.zeros_like(ds_pos))  # [E, J]
        p_is_ctrl = (self.p_team[:, :, None] == color[:, None, :]) & in_cart
        self.p_cart_push = self.p_cart_push + (p_is_ctrl.to(torch.float32) * push_share[:, None, :]).sum(dim=-1)

        # Contest: player inside cart where rival weight > 0 (total_w > w_own)
        p_w_own = torch.einsum("ejk,enk->enj", w_jk, team_oh)
        contested = in_cart & (total_w[:, None, :] > p_w_own + 1e-6)
        self.p_cart_contest = self.p_cart_contest + contested.any(dim=-1).to(torch.float32) * dt_vec.unsqueeze(-1)

        return in_cart, (total_w > 0.0) & (wopp > 0.0)

    @torch.no_grad()
    def _select_goals(self, dt_vec: torch.Tensor) -> None:
        """Branchless vectorized HavocBot cart goal rating (sv_payload.qc:806-943) — zero GPU-CPU syncs."""
        self.p_goal_timer = (self.p_goal_timer - dt_vec.unsqueeze(-1)).clamp_min(0.0)
        self.p_replan_timer = (self.p_replan_timer - dt_vec.unsqueeze(-1)).clamp_min(0.0)
        due = self.p_alive & (self.p_goal_timer <= 0.0)

        E, N, J = self.E, self.N, self.J
        arch = self._archetype[None, :, None]  # [1, N, 1]
        c_ctrl = self.c_ctrl[:, None, :]       # [E, 1, J]
        p_team = self.p_team[:, :, None]       # [E, N, 1]
        c_s = self.c_s[:, None, :]
        c_len = self.world.cart_length[None, None, :]

        # Base archetype weights (sv_payload.qc:822-828)
        is_own = (c_ctrl == p_team)
        is_enemy = (c_ctrl > 0) & (~is_own)
        own_w = torch.where(c_s < c_len * 0.98, torch.where(arch == 0, 2.0, 1.3), torch.full_like(c_s, 0.4))
        enemy_w = torch.where(arch == 1, 2.4, 1.7)
        neut_w = torch.where((arch == 0) | (arch == 3), 1.8, 1.4)
        w = torch.where(is_own, own_w, torch.where(is_enemy, enemy_w, neut_w))

        # Apply PLAN modulation (sv_payload.qc:829-847)
        has_plan = self.p_has_plan[:, :, None]
        plan_pick = torch.floor(self.p_plan[:, :, 1]).to(torch.int64).clamp(-1, J - 1)[:, :, None]  # [E, N, 1]
        w0 = self.p_plan[:, :, 2:3]
        w1 = self.p_plan[:, :, 3:4]
        w2 = self.p_plan[:, :, 4:5]
        j_ids = self._j_idx[None, None, :]
        if J <= 3:
            u = torch.where(j_ids == 0, w0, torch.where(j_ids == 1, w1, w2))
        else:
            u = torch.where((plan_pick >= 0) & (j_ids == plan_pick), w0, w1)
        plan_mult = torch.exp(u.clamp(-2.0, 2.0)) * torch.where((plan_pick >= 0) & (j_ids == plan_pick), 2.2, 1.0)
        pref_mult = torch.where(j_ids == self._pref_cart[None, :, None], 1.35, 1.0)
        w = w * torch.where(has_plan, plan_mult, pref_mult)

        # Graph distance rating (HavocBot routerating ~ weight / (dist + 200))
        cart_wps = self._cart_nearest_wp(self.c_s)  # [E, J]
        # wp_dist[p_wp_cur, cart_wps]: [E, N, J]
        p_wp = self.p_wp_cur[:, :, None].expand(E, N, J)
        c_wp = cart_wps[:, None, :].expand(E, N, J)
        d_graph = self.world.wp_dist[p_wp, c_wp]
        rating = w / (d_graph + 250.0)
        best_cart = rating.argmax(dim=-1)  # [E, N]

        self.p_goal_cart = torch.where(due, best_cart, self.p_goal_cart)
        # Target waypoint = cart nearest wp (offset by plan_ds if set)
        plan_ds = self.p_plan[:, :, 10].clamp(-600.0, 600.0)
        chosen_s = self.c_s.gather(1, self.p_goal_cart)
        chosen_len = self.world.cart_length[self.p_goal_cart]
        target_s = torch.where(self.p_has_plan & (plan_ds.abs() > 120.0), (chosen_s + plan_ds).clamp(0.0).minimum(chosen_len), chosen_s)
        frac = (target_s / chosen_len.clamp_min(1.0)).clamp(0.0, 1.0)
        bi = torch.round(frac * float(self.M_s - 1)).to(torch.int64).clamp(0, self.M_s - 1)
        target_wp = self.world.cart_sample_wp[self.p_goal_cart, bi]

        self.p_wp_goal = torch.where(due, target_wp, self.p_wp_goal)
        self.p_goal_timer = torch.where(due, torch.full_like(self.p_goal_timer, self.rules.replan_interval), self.p_goal_timer)

    @torch.no_grad()
    def _step_kinematics(self, dt_vec: torch.Tensor, max_dt_hint: float) -> None:
        """Branchless continuous-time multi-hop graph transit + direct local cart steering — zero GPU-CPU syncs."""
        E, N = self.E, self.N
        speed = float(self.rules.move_speed)
        dt_en = dt_vec.unsqueeze(-1)
        rem_dist = torch.where(self.p_alive, speed * dt_en, torch.zeros_like(dt_en))

        gc_idx = self.p_goal_cart.unsqueeze(-1).expand(E, N, 3)
        cart_target_pos = self.c_pos.gather(1, gc_idx)
        to_cart = cart_target_pos - self.p_pos
        d_cart = to_cart.norm(dim=-1)
        local_steer = self.p_alive & (d_cart <= self.world.cart_radius[self.p_goal_cart] * 1.6)

        max_hops = max(1, min(8, int(math.ceil(speed * max_dt_hint / 240.0))))
        pos = self.p_pos
        vel = torch.zeros_like(self.p_vel)
        graph_rem = torch.where(self.p_alive & (~local_steer), rem_dist, torch.zeros_like(rem_dist))

        for _ in range(max_hops):
            nxt = self.world.wp_next_hop[self.p_wp_cur, self.p_wp_goal]
            nxt_pos = self.world.wp_pos[nxt]
            delta = nxt_pos - pos
            d_seg = delta.norm(dim=-1)
            at_goal = (self.p_wp_cur == self.p_wp_goal) & (d_seg <= 1e-3)
            move_mask = (graph_rem > 1e-3) & (~at_goal)
            step = torch.minimum(graph_rem, d_seg)
            dir_v = delta / d_seg.clamp_min(1e-6).unsqueeze(-1)
            pos = torch.where(move_mask.unsqueeze(-1), pos + dir_v * step.unsqueeze(-1), pos)
            vel = torch.where(move_mask.unsqueeze(-1), dir_v * speed, vel)
            reached = move_mask & (d_seg <= graph_rem + 1e-3)
            self.p_wp_cur = torch.where(reached, nxt, self.p_wp_cur)
            self.p_wp_next = torch.where(move_mask, nxt, self.p_wp_next)
            graph_rem = torch.where(move_mask, (graph_rem - step).clamp_min(0.0), torch.zeros_like(graph_rem))

        stop_r = self.world.cart_radius[self.p_goal_cart] * 0.45
        need_move = local_steer & (d_cart > stop_r)
        l_step = torch.minimum(rem_dist, (d_cart - stop_r).clamp_min(0.0))
        l_dir = to_cart / d_cart.clamp_min(1e-6).unsqueeze(-1)
        pos = torch.where(need_move.unsqueeze(-1), pos + l_dir * l_step.unsqueeze(-1), pos)
        vel = torch.where(need_move.unsqueeze(-1), l_dir * speed, vel)

        self.p_pos = pos
        self.p_vel = vel

    @torch.no_grad()
    def _step_combat_and_respawn(self, dt_vec: torch.Tensor) -> torch.Tensor:
        """Branchless continuous-time combat DPS/hazard integration + respawn timer — zero GPU-CPU syncs."""
        E, N = self.E, self.N
        dt_en = dt_vec.unsqueeze(-1)

        # 1. Branchless Respawns
        dead = ~self.p_alive
        self.p_respawn_t = torch.where(dead, (self.p_respawn_t - dt_en).clamp_min(0.0), self.p_respawn_t)
        respawn_now = dead & (self.p_respawn_t <= 0.0)
        n_sp = int(self.world.spawn_pos.shape[1])
        u_sp = _philox_u01(self.seed, self.step_count, 307, (E, N))
        sp_slot = (u_sp * n_sp).to(torch.int64).clamp(0, n_sp - 1)
        t0 = self.p_team - 1
        sp_pos = self.world.spawn_pos[t0, sp_slot]
        sp_wp = self.world.spawn_wp[t0, sp_slot]
        self.p_pos = torch.where(respawn_now.unsqueeze(-1), sp_pos, self.p_pos)
        self.p_vel = torch.where(respawn_now.unsqueeze(-1), torch.zeros_like(self.p_vel), self.p_vel)
        self.p_wp_cur = torch.where(respawn_now, sp_wp, self.p_wp_cur)
        self.p_wp_next = torch.where(respawn_now, sp_wp, self.p_wp_next)
        self.p_hp = torch.where(respawn_now, torch.full_like(self.p_hp, self.rules.max_health), self.p_hp)
        self.p_armor = torch.where(respawn_now, torch.full_like(self.p_armor, self.rules.start_armor), self.p_armor)
        self.p_alive = self.p_alive | respawn_now
        self.p_spawn_t = torch.where(respawn_now, self.sim_time.unsqueeze(-1), self.p_spawn_t)
        self.p_goal_timer = torch.where(respawn_now, torch.zeros_like(self.p_goal_timer), self.p_goal_timer)

        # 2. Vectorized pairwise combat engagement [E, N, N] (branchless)
        if self.rules.combat_range <= 0.0:
            return torch.zeros((E, N), device=self.device, dtype=torch.bool)
        diff = self.p_pos[:, :, None, :] - self.p_pos[:, None, :, :]
        dist2 = (diff * diff).sum(dim=-1)
        r2 = float(self.rules.combat_range) ** 2
        rival_mask = (
            self.p_alive[:, :, None]
            & self.p_alive[:, None, :]
            & (self.p_team[:, :, None] != self.p_team[:, None, :])
            & (dist2 <= r2)
        )
        in_combat = rival_mask.any(dim=-1)

        # Focus-team preference from PLAN column 7 (sv_payload.qc:923)
        focus_t = torch.round(self.p_plan[:, :, 7]).to(torch.int64)  # [E, N]
        is_focus = self.p_has_plan[:, :, None] & (focus_t[:, :, None] > 0) & (self.p_team[:, None, :] == focus_t[:, :, None])
        eff_dist2 = torch.where(rival_mask, dist2 * torch.where(is_focus, 0.55, 1.0), torch.full_like(dist2, 1e18))
        best_d2, tgt_idx = eff_dist2.min(dim=-1)  # [E, N]
        has_tgt = in_combat & (best_d2 < 1e17)

        # Continuous-time DPS intensity: scales with proximity, skill, and posture (PLAN col 15)
        dist = best_d2.clamp_min(1.0).sqrt()
        prox = (1.0 - dist / float(self.rules.combat_range)).clamp(0.15, 1.0)
        posture = torch.where(self.p_has_plan, self.p_plan[:, :, 15].clamp(-1.0, 1.0), torch.zeros_like(self.p_hp))
        skill_mult = (1.0 + float(self.rules.skill_dps_scale) * self.p_skill).clamp(0.4, 3.0)
        dps_rate = float(self.rules.base_dps) * prox * skill_mult * (1.0 + 0.15 * posture)

        # Deterministic stochastic variation scaled as sqrt(dt) (proper continuous-time diffusion scaling!)
        u_hit = _philox_u01(self.seed, self.step_count, 509, (E, N))
        noise = 1.0 + float(self.rules.dps_stochastic_cv) * (2.0 * u_hit - 1.0) / dt_en.clamp_min(0.05).sqrt()
        dmg_out = torch.where(has_tgt & (~self.t_finished).unsqueeze(-1), (dps_rate * dt_en * noise.clamp(0.25, 2.2)).clamp_min(0.0), torch.zeros_like(self.p_hp))

        # Scatter damage onto targets
        dmg_in = torch.zeros_like(self.p_hp)
        dmg_in.scatter_add_(1, tgt_idx, dmg_out)

        # Xonotic armor absorption split (66% armor, remainder health)
        armor_take = torch.minimum(self.p_armor, dmg_in * float(self.rules.armor_absorb))
        hp_take = dmg_in - armor_take
        self.p_armor = (self.p_armor - armor_take).clamp_min(0.0)
        prev_alive = self.p_alive
        self.p_hp = self.p_hp - hp_take
        died = prev_alive & (self.p_hp <= 0.0)
        self.p_hp = torch.where(died, torch.zeros_like(self.p_hp), self.p_hp)
        self.p_alive = self.p_alive & (~died)
        self.p_respawn_t = torch.where(died, torch.full_like(self.p_respawn_t, self.rules.respawn_delay), self.p_respawn_t)

        # Attribute kills & outcome counters
        self.p_dmg_dealt = self.p_dmg_dealt + dmg_out
        self.p_dmg_taken = self.p_dmg_taken + dmg_in
        self.p_deaths = self.p_deaths + died.to(torch.float32)
        tgt_died = died.gather(1, tgt_idx) & has_tgt & (dmg_out > 0.0)
        self.p_kills = self.p_kills + tgt_died.to(torch.float32)
        return in_combat

    @torch.no_grad()
    def _single_substep(self, dt_vec: torch.Tensor, max_dt_hint: float, *, exact_crossing_score: bool) -> tuple[torch.Tensor, torch.Tensor]:
        dt_eff = self._integrate_score(dt_vec)
        _, contested_carts = self._step_carts_and_checkpoints(dt_eff, exact_crossing_score=exact_crossing_score)
        self._select_goals(dt_eff)
        self._step_kinematics(dt_eff, max_dt_hint)
        in_combat = self._step_combat_and_respawn(dt_eff)
        self.sim_time = self.sim_time + dt_eff
        self.step_count = self.step_count + 1
        return contested_carts, in_combat

    @torch.no_grad()
    def step(
        self,
        dt: float | torch.Tensor = 0.4,
        *,
        mode: str = "analytic",
        dt_ref: float | None = None,
        reset_outcomes: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Advance all E environments by delta-time dt (zero GPU-CPU syncs on 'analytic' & 'substep' paths!).
        """
        if reset_outcomes:
            self.p_dmg_dealt.zero_()
            self.p_dmg_taken.zero_()
            self.p_kills.zero_()
            self.p_deaths.zero_()
            self.p_pickups.zero_()
            self.p_cart_push.zero_()
            self.p_cart_contest.zero_()

        if isinstance(dt, torch.Tensor):
            dt_vec = dt.to(device=self.device, dtype=torch.float32)
            max_dt = float(dt_vec.max().item()) if self.E > 0 else 0.0
        else:
            max_dt = float(dt)
            dt_vec = torch.full((self.E,), max_dt, device=self.device, dtype=torch.float32)

        dref = float(self.rules.plc_tick_ref if dt_ref is None else dt_ref)

        if mode == "substep":
            n_sub = max(1, int(math.ceil(max_dt / max(dref, 1e-4))))
            sub_dt = dt_vec / float(n_sub)
            sub_hint = max_dt / float(n_sub)
            for _ in range(n_sub):
                self._single_substep(sub_dt, sub_hint, exact_crossing_score=False)
            return {"finished": self.t_finished, "winner": self.t_winner, "substeps": torch.full((self.E,), n_sub, device=self.device, dtype=torch.int64)}

        if mode == "adaptive":
            n_fine = max(1, int(math.ceil(max_dt / max(dref, 1e-4))))
            if n_fine <= 1:
                self._single_substep(dt_vec, max_dt, exact_crossing_score=True)
                return {"finished": self.t_finished, "winner": self.t_winner, "substeps": torch.ones((self.E,), device=self.device, dtype=torch.int64)}

            diff_p = self.p_pos[:, :, None, :] - self.p_pos[:, None, :, :]
            d2_p = (diff_p * diff_p).sum(dim=-1)
            hot_combat = (
                self.p_alive[:, :, None]
                & self.p_alive[:, None, :]
                & (self.p_team[:, :, None] != self.p_team[:, None, :])
                & (d2_p <= float(self.rules.combat_range) ** 2)
            ).any(dim=(1, 2))
            calm = ~hot_combat
            self._single_substep(torch.where(calm, dt_vec, torch.zeros_like(dt_vec)), max_dt, exact_crossing_score=True)
            fine_dt = torch.where(hot_combat, dt_vec / float(n_fine), torch.zeros_like(dt_vec))
            sub_hint = max_dt / float(n_fine)
            for _ in range(n_fine):
                self._single_substep(fine_dt, sub_hint, exact_crossing_score=False)
            sub_counts = torch.where(hot_combat, torch.full((self.E,), n_fine, device=self.device, dtype=torch.int64), torch.ones((self.E,), device=self.device, dtype=torch.int64))
            return {"finished": self.t_finished, "winner": self.t_winner, "substeps": sub_counts}

        self._single_substep(dt_vec, max_dt, exact_crossing_score=True)
        return {"finished": self.t_finished, "winner": self.t_winner, "substeps": torch.ones((self.E,), device=self.device, dtype=torch.int64)}

    @torch.no_grad()
    def observe_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Zero-copy on-device observation tensors matching strategy_io_schema.py & PlanWally."""
        obs = self._obs_buf
        obs.zero_()
        obs[:, :, OBS["ID"]] = self._etof[None, :].to(torch.float32)
        obs[:, :, OBS["TEAM"]] = self.p_team.to(torch.float32)
        obs[:, :, OBS["HEALTH"]] = self.p_hp
        obs[:, :, OBS["ARMOR"]] = self.p_armor
        obs[:, :, OBS["POS_X"]:OBS["POS_Z"] + 1] = self.p_pos
        obs[:, :, OBS["VEL_X"]:OBS["VEL_Z"] + 1] = self.p_vel
        obs[:, :, OBS["SPAWN_TIME"]] = self.p_spawn_t
        obs[:, :, OBS["ENGINE_TIME"]] = self.sim_time.unsqueeze(-1)
        obs[:, :, OBS["ALIVE"]] = self.p_alive.to(torch.float32)
        obs[:, :, OBS["CONTROL"]] = 1.0
        obs[:, :, OBS["PRESENT"]] = 1.0
        obs[:, :, OBS["BOT_GLOBAL_SKILL"]] = self.p_skill
        obs[:, :, OBS["ENEMY_DAMAGE_DEALT"]] = self.p_dmg_dealt
        obs[:, :, OBS["ENEMY_DAMAGE_TAKEN"]] = self.p_dmg_taken
        obs[:, :, OBS["ENEMY_KILLS"]] = self.p_kills
        obs[:, :, OBS["DEATHS"]] = self.p_deaths
        obs[:, :, OBS["PICKUPS"]] = self.p_pickups
        obs[:, :, OBS["CART_PUSH"]] = self.p_cart_push
        obs[:, :, OBS["CART_CONTEST"]] = self.p_cart_contest

        carts = self._cart_buf
        carts.zero_()
        carts[:, :, CS["ID"]] = self._j_idx[None, :].to(torch.float32)
        carts[:, :, CS["PATH_POSITION"]] = self.c_s
        carts[:, :, CS["PATH_LENGTH"]] = self.world.cart_length[None, :]
        carts[:, :, CS["CONTROL_TEAM"]] = self.c_ctrl.to(torch.float32)
        carts[:, :, CS["SPEED"]] = self.c_speed
        carts[:, :, CS["IDLE_TIME"]] = self.c_idle
        carts[:, :, CS["LEAD_TEAM"]] = self.c_lead.to(torch.float32)
        carts[:, :, CS["SECOND_TEAM"]] = self.c_second.to(torch.float32)
        carts[:, :, CS["HOME_TEAM"]] = self.c_home.to(torch.float32)
        carts[:, :, CS["POS_X"]:CS["POS_Z"] + 1] = self.c_pos
        carts[:, :, CS["SUPPORTS_PLAYER"]] = 1.0
        carts[:, :, CS["TEAM_COUNT"]] = float(self.K)
        carts[:, :, CS["CHECKPOINT_DEPTH"]] = self.c_cp_depth.to(torch.float32)
        carts[:, :, CS["CHECKPOINT_COUNT"]] = self.world.cart_cp_count[None, :].to(torch.float32)

        teams = self._team_buf
        teams.zero_()
        teams[:, :, TS["ID"]] = self._k_ids[None, :].to(torch.float32)
        teams[:, :, TS["SCORE"]] = self.t_score
        teams[:, :, TS["CHECKPOINTS_HELD"]] = self.t_cp_held.to(torch.float32)
        teams[:, :, TS["CHECKPOINT_RATE"]] = float(self.rules.checkpoint_rate)
        teams[:, :, TS["SCORE_LIMIT"]] = self.point_limit
        teams[:, :, TS["EPISODE"]] = 1.0
        teams[:, :, TS["FINISHED"]] = self.t_finished.to(torch.float32).unsqueeze(-1)

        geo = {
            "map_center": self.world.map_center,
            "map_radius": self.world.map_radius,
        }
        return obs, carts, teams, geo
