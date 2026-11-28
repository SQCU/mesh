"""Single locus of responsibility for dataset sampling, persistent match pools,
optional counterfactual group forking, and multi-variant policy scatter-gather.

Zero training / PPO / optimizer logic lives here. Emits flat atemporal transition
batches (AtemporalBatch) sampled continuously across all match progressions
(early, mid, late, terminal auto-reset) with zero throwaway warmup.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from mapgen.bsp_stream import sample_world_spec
from payload.tools.strategy_io_schema import OBS
from solver.strat.league_curriculum import SCRIPTED_ANCHORS
from solver.strat.plan_policy import PLAN_WIDTH, PlanWally
from solver.strat.train_config import ActionPackConfig, TrainingConfig

from .engine import SimRulesConfig
from .metal_runtime import (
    CompiledPlanWally,
    MetalCGTRewardComputer,
    MetalDeltaTimePayloadEngine,
)
from .world_graph import WorldGraphIR, build_gas_world_graph


PHYSICS_DR_PROFILES: dict[str, dict[str, float]] = {
    "baseline": {},
    "fast": {"cart_base_speed": 45.0, "cart_max_speed": 260.0, "push_falloff": 1.0},
    "slow": {"cart_base_speed": 20.0, "cart_max_speed": 120.0, "push_falloff": 0.5},
    "volatile": {"contest_speed": 36.0, "reverse_speed": 24.0, "push_falloff": 0.75},
}


def make_rules_for_profile(point_limit_base: float, profile: str = "baseline") -> SimRulesConfig:
    kw: dict[str, Any] = {"point_limit_base": float(point_limit_base)}
    kw.update(PHYSICS_DR_PROFILES.get(profile, {}))
    return SimRulesConfig(**kw)


def sample_roster_skills(
    rng: np.random.Generator,
    E: int,
    N: int,
    K: int,
    mode: str = "mixed_dr",
) -> np.ndarray:
    """Sample per-bot deathmatch mechanics skill [E, N] float32 (sk_m = clamp(1 + 0.06*skill, 0.4, 3.0)).

    Modes:
      - uniform_balanced: homogeneous lobby skill ~ U[2, 8]
      - gaussian_lobby: lobby mean ~ U[0, 12], per-bot N(mean, 5.0)
      - team_stacked_weak: 1-2 teams stacked with weak-mechanics bots (N(-7.5, 2.0), ~0.40-0.65x DPS)
                           against strong/ace teams (N(+19, 2.5), ~2.0-2.4x DPS)
      - within_team_skewed: 25% ace carries (N(+20, 2.5)) + 75% weak-mechanics teammates (N(-7.5, 2.0))
      - mixed_dr: per-env mixture (20% balanced, 20% gaussian, 35% team_stacked_weak, 25% within_team_skewed)
      - stacked_main_weak: deterministic eval stress — team 1 stacked weak (-8), rivals strong (+20)
      - stacked_main_strong: deterministic eval stress — team 1 stacked strong (+20), rivals weak (-8)
      - within_team_carry_weak: deterministic eval stress — 25% carries (+22) + 75% weak (-8) on every team
    """
    out = np.empty((E, N), dtype=np.float32)
    teams_1d = (np.arange(N, dtype=np.int32) % max(1, K))  # 0..K-1
    slot_in_team = np.arange(N, dtype=np.int32) // max(1, K)

    if mode == "stacked_main_weak":
        out[:] = np.where(teams_1d[None, :] == 0, -8.0, 20.0).astype(np.float32)
        return out
    if mode == "stacked_main_strong":
        out[:] = np.where(teams_1d[None, :] == 0, 20.0, -8.0).astype(np.float32)
        return out
    if mode == "mixed_enemies_main_weak":
        # Team 1 weak (-8), Team 2 ace (+20), Team 3 strong (+14), middle teams (+4), last team weak (-8)
        t_sk = np.full((max(1, K),), 4.0, dtype=np.float32)
        t_sk[0] = -8.0
        if K >= 2:
            t_sk[1] = 20.0
        if K >= 3:
            t_sk[2] = 14.0
        if K >= 4:
            t_sk[K - 1] = -8.0
        out[:] = t_sk[teams_1d][None, :]
        return out
    if mode == "within_team_carry_weak":
        is_carry = (slot_in_team % 4) == 0
        out[:] = np.where(is_carry[None, :], 22.0, -8.0).astype(np.float32)
        return out
    if mode == "balanced_fixed":
        out.fill(4.0)
        return out

    for e in range(E):
        if mode == "mixed_dr":
            u = float(rng.random())
            m = "uniform_balanced" if u < 0.20 else ("gaussian_lobby" if u < 0.40 else ("team_stacked_weak" if u < 0.75 else "within_team_skewed"))
        else:
            m = mode

        if m == "uniform_balanced":
            out[e] = rng.uniform(2.0, 8.0, size=N).astype(np.float32)
        elif m == "gaussian_lobby":
            mu = float(rng.uniform(0.0, 12.0))
            out[e] = np.clip(rng.normal(mu, 5.0, size=N), -10.0, 28.0).astype(np.float32)
        elif m == "team_stacked_weak":
            # Randomly stack 1..max(1, K//3) teams weak and 1..max(1, K//3) teams ace!
            perm = rng.permutation(K)
            n_weak = max(1, K // 3)
            n_ace = max(1, K // 3)
            weak_t = set(int(x) for x in perm[:n_weak])
            ace_t = set(int(x) for x in perm[n_weak : n_weak + n_ace])
            t_mu = np.array([-7.5 if k in weak_t else (19.0 if k in ace_t else 4.0) for k in range(K)], dtype=np.float32)
            t_sd = np.array([2.0 if k in weak_t else (2.5 if k in ace_t else 2.5) for k in range(K)], dtype=np.float32)
            out[e] = np.clip(rng.normal(t_mu[teams_1d], t_sd[teams_1d]), -10.0, 28.0).astype(np.float32)
        else:  # within_team_skewed
            is_carry = (slot_in_team % 4) == 0
            mu = np.where(is_carry, 20.0, -7.5).astype(np.float32)
            sd = np.where(is_carry, 2.5, 2.0).astype(np.float32)
            out[e] = np.clip(rng.normal(mu, sd), -10.0, 28.0).astype(np.float32)
    return out


@dataclass
class AtemporalBatch:
    """Flat atemporal transition dataset emitted by DatasetSampler for any PPO/value learner."""

    obs: torch.Tensor            # [M, N, OBS_WIDTH] bf16
    carts: torch.Tensor          # [M, J, CART_WIDTH] bf16
    teams: torch.Tensor          # [M, K, TEAM_WIDTH] bf16
    obs_next: torch.Tensor | None  # [M, N, OBS_WIDTH] bf16 (None if obs_drho precomputed)
    carts_next: torch.Tensor | None
    teams_next: torch.Tensor | None
    cont: torch.Tensor           # [M, N, 11] bf16
    cart_pick: torch.Tensor      # [M, N] int16
    rival_pick: torch.Tensor     # [M, N] int16
    beh_logp: torch.Tensor       # [M, N] fp32
    adv: torch.Tensor            # [M, N] fp32
    gae_ret: torch.Tensor        # [M, N] fp32
    wm_in: torch.Tensor          # [M, N] bool
    actor_mask: torch.Tensor     # [M, N] bool
    geo: dict[str, torch.Tensor]
    policy_version: int
    world_meta: dict[str, int]
    opp_wr_obs: dict[str, float]
    stats: dict[str, Any]
    obs_drho: torch.Tensor | None = None  # optional precomputed [M, N] fp32 dynamics target


@torch.no_grad()
def _scripted_plan_tensor(
    arm_name: str,
    obs: torch.Tensor,
    carts: torch.Tensor,
    teams: torch.Tensor,
) -> torch.Tensor:
    from payload.tools.strategy_io_schema import CS, TS

    e, n, _ = obs.shape
    j = carts.shape[1]
    out = torch.zeros((e, n, PLAN_WIDTH), dtype=torch.float32)
    if arm_name not in SCRIPTED_ANCHORS or n == 0 or j == 0:
        return out
    p_pos = obs[:, :, OBS["POS_X"] : OBS["POS_Z"] + 1].float()
    c_pos = carts[:, :, CS["POS_X"] : CS["POS_Z"] + 1].float()
    dists = (p_pos[:, :, None, :] - c_pos[:, None, :, :]).square().sum(dim=-1).clamp_min(1e-6).sqrt()
    nearest_cart = dists.argmin(dim=-1)
    my_team = torch.round(obs[:, :, OBS["TEAM"]].float()).to(torch.int64)
    cart_ctrl = torch.round(carts[:, :, CS["CONTROL_TEAM"]].float()).to(torch.int64)
    team_ids = torch.round(teams[:, :, TS["ID"]].float()).to(torch.int64)
    lead_idx = teams[:, :, TS["SCORE"]].float().argmax(dim=-1)
    leader_team = team_ids.gather(1, lead_idx[:, None])
    own_mask = cart_ctrl[:, None, :] == my_team[:, :, None]
    own_min_d, own_pick = torch.where(own_mask, dists, torch.full_like(dists, 1e18)).min(dim=-1)
    own_cart = torch.where(own_min_d < 1e17, own_pick, nearest_cart)
    lead_mask = (cart_ctrl[:, None, :] == leader_team[:, :, None]) & (my_team[:, :, None] != leader_team[:, :, None])
    lead_min_d, lead_pick = torch.where(lead_mask, dists, torch.full_like(dists, 1e18)).min(dim=-1)
    lead_cart = torch.where(lead_min_d < 1e17, lead_pick, nearest_cart)

    out[:, :, 0] = obs[:, :, OBS["ID"]].float()
    if arm_name == "script_cart_rush":
        cpick = own_cart
        out[:, :, 1] = cpick.float()
        out.scatter_(2, (cpick.clamp(0, 2) + 2).unsqueeze(-1), torch.full((e, n, 1), 1.6))
        out[:, :, 5] = 1.4; out[:, :, 6] = -0.6; out[:, :, 14] = 0.4; out[:, :, 15] = -0.3
    elif arm_name == "script_leader_gank":
        cpick = lead_cart
        out[:, :, 1] = cpick.float()
        out.scatter_(2, (cpick.clamp(0, 2) + 2).unsqueeze(-1), torch.full((e, n, 1), 1.2))
        out[:, :, 5] = 0.6; out[:, :, 6] = 1.4
        out[:, :, 7] = torch.where(my_team != leader_team, leader_team.expand(e, n).float(), torch.zeros((e, n)))
        out[:, :, 14] = 1.2; out[:, :, 15] = 0.7
    elif arm_name == "script_powerup_roam":
        out[:, :, 1] = nearest_cart.float()
        out[:, :, 5] = 0.3; out[:, :, 6] = 0.5; out[:, :, 8] = 1.4; out[:, :, 9] = 1.8; out[:, :, 15] = 0.4
    elif arm_name == "script_vanguard_ambush":
        cpick = own_cart
        out[:, :, 1] = cpick.float()
        out.scatter_(2, (cpick.clamp(0, 2) + 2).unsqueeze(-1), torch.full((e, n, 1), 1.2))
        out[:, :, 5] = 1.1; out[:, :, 6] = 0.8
        even = (torch.arange(n) % 2 == 0).view(1, n)
        out[:, :, 10] = torch.where(even, torch.full((e, n), 240.0), torch.zeros((e, n)))
        out[:, :, 14] = 0.8; out[:, :, 15] = 0.5
    return out


class PolicyVariantDispatcher:
    """Compiled Metal 4 sparse scatter-gather dispatcher for k policy variants.

    When k=1 (all neural seats use the active policy), executes 1 single Metal 4 forward_sample
    call across all E envs (1x compute). When k>1, partitions envs by opponent variant so each
    variant executes ONLY on the envs where it is seated (zero k*E waste).
    """

    def __init__(
        self,
        variants: dict[str, PlanWally | None],
        E: int,
        N: int,
        J: int,
        K: int,
        *,
        pack_cfg: ActionPackConfig,
        seed_base: int = 1,
        train_arm: str = "main",
    ):
        self.E = int(E)
        self.N = int(N)
        self.J = int(J)
        self.K = int(K)
        self.pack_cfg = pack_cfg
        self.seed_base = int(seed_base)
        self.train_arm = train_arm
        self.compiled: dict[tuple[str, int], CompiledPlanWally] = {}
        self._variants = variants

    def close(self) -> None:
        for c in self.compiled.values():
            c.close()
        self.compiled.clear()

    def _get(self, arm: str, n_envs: int, idx_seed: int) -> CompiledPlanWally | None:
        mdl = self._variants.get(arm)
        if mdl is None:
            return None
        key = (arm, n_envs)
        c = self.compiled.get(key)
        if c is None:
            c = CompiledPlanWally(
                mdl, E=n_envs, N=self.N, J=self.J, K=self.K,
                pack_cfg=self.pack_cfg, do_pack=True, stochastic=True, write_sample=True,
                sample_seed=self.seed_base + 97 * idx_seed + 13,
            )
            self.compiled[key] = c
        return c

    def sync_variant_(self, arm: str, model: PlanWally) -> None:
        self._variants[arm] = model
        for (a, _), c in self.compiled.items():
            if a == arm:
                c.sync_weights_(model)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        carts: torch.Tensor,
        teams: torch.Tensor,
        geo: dict[str, torch.Tensor],
        seat_arms: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        e, n, _ = obs.shape
        plan = torch.zeros((e, n, PLAN_WIDTH), dtype=torch.float32)
        has_plan = torch.zeros((e, n), dtype=torch.bool)
        p_team = torch.round(obs[:, :, OBS["TEAM"]].float()).to(torch.int64)

        arm_to_teams: dict[str, list[int]] = defaultdict(list)
        for ti, arm in enumerate(seat_arms):
            arm_to_teams[arm].append(ti + 1)

        # Fast path: single neural arm controls ALL teams (pure self-play, 1x GPU pass, 0 masks!)
        if len(arm_to_teams) == 1 and self.train_arm in arm_to_teams:
            cpol = self._get(self.train_arm, e, 0)
            assert cpol is not None
            res = cpol.forward_sample(obs, carts, teams, geo=geo)
            all_mask = torch.ones((e, n), dtype=torch.bool)
            return res["plan"], all_mask, {
                "cont": res["cont"],
                "cart_pick": res["cart_pick"],
                "rival_pick": res["rival_pick"],
                "logp": res["logp"],
                "value_winnie": res["value_winnie"],
                "value_lou": res["value_lou"],
                "actor_mask": all_mask,
            }

        train_data: dict[str, Any] | None = None
        for idx, (arm, t_list) in enumerate(arm_to_teams.items()):
            mask = torch.zeros((e, n), dtype=torch.bool)
            for tid in t_list:
                mask = mask | (p_team == tid)
            if arm == "default":
                continue
            if arm in SCRIPTED_ANCHORS:
                sp = _scripted_plan_tensor(arm, obs, carts, teams)
                plan = torch.where(mask.unsqueeze(-1), sp, plan)
                has_plan = has_plan | mask
                continue
            cpol = self._get(arm, e, idx)
            if cpol is None:
                continue
            res = cpol.forward_sample(obs, carts, teams, geo=geo)
            plan = torch.where(mask.unsqueeze(-1), res["plan"], plan)
            has_plan = has_plan | mask
            if arm == self.train_arm:
                train_data = {
                    "cont": res["cont"],
                    "cart_pick": res["cart_pick"],
                    "rival_pick": res["rival_pick"],
                    "logp": res["logp"],
                    "value_winnie": res["value_winnie"],
                    "value_lou": res["value_lou"],
                    "actor_mask": mask,
                }
        assert train_data is not None, f"train_arm={self.train_arm} not present in seat_arms={seat_arms}"
        return plan, has_plan, train_data


class _WorldSlot:
    """One persistent world slot owning a live MetalDeltaTimePayloadEngine + compiled policies + reward GPU."""

    def __init__(
        self,
        world: WorldGraphIR,
        spec: dict[str, Any],
        cfg: TrainingConfig,
        variants: dict[str, PlanWally | None],
        *,
        seed_base: int,
        group_size: int,
        physics_profile: str = "baseline",
        roster_mode: str = "mixed_dr",
    ):
        sc = cfg.sim_dt
        self.world = world
        self.spec = dict(spec)
        self.K = int(world.teams)
        self.J = int(world.carts)
        self.N = int(world.players_default)
        self.B = int(sc.base_envs)
        self.G = max(1, int(group_size))
        self.E = self.B * self.G
        self.physics_profile = str(physics_profile)
        self.roster_mode = str(roster_mode)
        self.dr_rng = np.random.default_rng(seed_base + 4242)
        self.rules = make_rules_for_profile(float(sc.point_limit_base), self.physics_profile)
        self.geo = {"map_center": world.map_center.clone(), "map_radius": world.map_radius.clone()}

        # Persistent live engine (auto-resets finished envs -> continuous coverage of all match phases!)
        n_live = self.E if self.G == 1 else self.B
        self.live_eng = MetalDeltaTimePayloadEngine(
            world, num_envs=n_live,
            num_players=self.N, rules=self.rules, base_seed=seed_base,
        )
        self.live_eng.set_skills(sample_roster_skills(self.dr_rng, n_live, self.N, self.K, self.roster_mode))
        # Optional counterfactual fork engine (allocated once and reused via O(1) memcpy!)
        self.fork_eng: MetalDeltaTimePayloadEngine | None = (
            MetalDeltaTimePayloadEngine(
                world, num_envs=self.E, num_players=self.N, rules=self.rules, base_seed=seed_base + 50000,
            )
            if self.G > 1
            else None
        )
        self.rew_gpu = MetalCGTRewardComputer(E=self.E, N=self.N, J=self.J, K=self.K)
        self.rew_gpu.set_config(cfg.losses.reward, terminal_win_bonus=float(sc.terminal_win_bonus))
        self.dispatcher = PolicyVariantDispatcher(
            variants, E=self.E, N=self.N, J=self.J, K=self.K,
            pack_cfg=cfg.action_pack, seed_base=seed_base, train_arm="main",
        )
        self.steps_elapsed = 0

    def close(self) -> None:
        self.dispatcher.close()
        self.rew_gpu.close()
        self.live_eng.close()
        if self.fork_eng is not None:
            self.fork_eng.close()

    def sync_variant_(self, arm: str, model: PlanWally) -> None:
        self.dispatcher.sync_variant_(arm, model)


class PersistentMatchPool:
    """Persistent multi-world pool sampling continuous atemporal transition batches.

    Zero throwaway warmup: matches persist across iterations, auto-resetting on score-out,
    so batches continuously cover early, mid, late, and terminal phases.
    Supports both:
      - G = 1: pure continuous streaming (all E envs step directly, zero fork overhead)
      - G > 1: O(1) savestate fork of B live persistent envs -> B*G counterfactual branches
               (advancing the B live envs from branch 0 after each rollout so progression continues!)
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        variants: dict[str, PlanWally | None],
        *,
        num_worlds: int = 4,
        group_size: int | None = None,
        seed_base: int = 30000,
        self_play_only: bool = False,
        roster_mode: str = "mixed_dr",
        randomize_physics: bool = True,
    ):
        self.cfg = cfg
        self.sc = cfg.sim_dt
        self.variants = variants
        self.G = int(self.sc.group_size if group_size is None else group_size)
        self.self_play_only = bool(self_play_only)
        self.roster_mode = str(roster_mode)
        self.rng = np.random.default_rng(seed_base + 777)
        self.slots: list[_WorldSlot] = []
        self._cursor = 0
        phys_cycle = ("baseline", "fast", "slow", "volatile") if randomize_physics else ("baseline",)

        for wi in range(max(1, int(num_worlds))):
            spec = sample_world_spec(seed_base + (wi + 1) * 101, cfg.world)
            map_name = f"pool_w{wi}_k{spec['teams']}_j{spec['carts']}_n{spec['players']}"
            world = build_gas_world_graph(
                map_name=map_name,
                teams=int(spec["teams"]),
                carts=int(spec["carts"]),
                players=int(spec["players"]),
                tiles=int(spec["tiles"]),
                seed=int(spec["seed"]),
                device=torch.device("cpu"),
            )
            self.slots.append(
                _WorldSlot(
                    world, spec, cfg, variants,
                    seed_base=seed_base + wi * 10009,
                    group_size=self.G,
                    physics_profile=phys_cycle[wi % len(phys_cycle)],
                    roster_mode=self.roster_mode,
                )
            )

        # Initialize natural staggered match progression once at pool creation (0 per-iter warmup!)
        # Slot i starts at offset i * (H // 2) ticks using 1-step analytic physics so slot phases differ immediately.
        for wi, slot in enumerate(self.slots):
            stagger = (wi * max(4, int(self.sc.rollout_horizon) // 3)) % 45
            if stagger > 0:
                slot.live_eng.step(self.sc.dt, mode=self.sc.mode, n_steps=stagger)
                fin = slot.live_eng.t_finished
                if bool(fin.any().item()):
                    slot.live_eng.reset(fin)

    def close(self) -> None:
        for s in self.slots:
            s.close()
        self.slots.clear()

    def sync_variant_(self, arm: str, model: PlanWally) -> None:
        self.variants[arm] = model
        for s in self.slots:
            s.sync_variant_(arm, model)

    def _choose_seats(self, K: int, pfsp_pool: list[str], pfsp_wr: dict[str, float]) -> list[str]:
        if self.self_play_only or K <= 1:
            return ["main"] * K
        lg = self.cfg.league
        seats = ["main"]
        cands = sorted(set(pfsp_pool))
        w = np.asarray(
            [max(lg.pfsp_min_weight, 1.0 - float(pfsp_wr.get(a, lg.pfsp_default_winrate))) ** lg.pfsp_power for a in cands],
            dtype=np.float64,
        )
        w /= w.sum()
        seats.append("default" if self.rng.random() < 0.5 else str(self.rng.choice(SCRIPTED_ANCHORS)))
        for _ in range(K - 2):
            seats.append(str(self.rng.choice(cands, p=w)))
        return seats

    @torch.no_grad()
    def sample_batch(
        self,
        *,
        policy_version: int,
        pfsp_pool: list[str] | None = None,
        pfsp_wr: dict[str, float] | None = None,
        seat_arms: list[str] | None = None,
        slot_index: int | None = None,
    ) -> AtemporalBatch:
        t0 = time.perf_counter()
        idx = self._cursor % len(self.slots) if slot_index is None else (int(slot_index) % len(self.slots))
        self._cursor += 1
        slot = self.slots[idx]
        sc = self.sc
        B, G, E, H = slot.B, slot.G, slot.E, int(sc.rollout_horizon)
        N, J, K = slot.N, slot.J, slot.K

        if seat_arms is None:
            seat_arms = self._choose_seats(
                K,
                pfsp_pool or ["default", *SCRIPTED_ANCHORS, "main"],
                pfsp_wr or {},
            )

        # Choose active rollout engine (zero warmup!):
        if G > 1:
            assert slot.fork_eng is not None
            # Auto-reset any finished base envs + resample fresh DR roster skills before counterfactual fork
            fin0 = slot.live_eng.t_finished
            if bool(fin0.any().item()):
                slot.live_eng.reset(fin0)
                fids = np.nonzero(fin0.numpy())[0]
                slot.live_eng.set_skills(
                    sample_roster_skills(slot.dr_rng, len(fids), N, K, slot.roster_mode), env_ids=fids,
                )
            snap = slot.live_eng.snapshot()
            src = np.frombuffer(snap.data, dtype=np.uint8).reshape(B, snap.env_stride)
            slot.fork_eng._state_u8[:] = np.repeat(src, G, axis=0)
            if not bool(sc.common_random_numbers):
                sm = snap.soff_meta
                mu32 = slot.fork_eng._state_u8[:, sm : sm + 32].view(np.uint32)
                seeds = mu32[:, 3].astype(np.uint64) | (mu32[:, 4].astype(np.uint64) << np.uint64(32))
                seeds = seeds + np.tile(np.arange(G, dtype=np.uint64), B) * np.uint64(10007)
                mu32[:, 3] = (seeds & np.uint64(0xFFFFFFFF)).astype(np.uint32)
                mu32[:, 4] = (seeds >> np.uint64(32)).astype(np.uint32)
            eng = slot.fork_eng
        else:
            eng = slot.live_eng

        act_dtype = torch.bfloat16
        buf_obs = torch.empty((H, E, N, 84), dtype=act_dtype)
        buf_carts = torch.empty((H, E, J, 18), dtype=act_dtype)
        buf_teams = torch.empty((H, E, K, 7), dtype=act_dtype)
        buf_obs_next = torch.empty((H, E, N, 84), dtype=act_dtype)
        buf_carts_next = torch.empty((H, E, J, 18), dtype=act_dtype)
        buf_teams_next = torch.empty((H, E, K, 7), dtype=act_dtype)
        buf_cont = torch.empty((H, E, N, 11), dtype=act_dtype)
        buf_cart_pick = torch.empty((H, E, N), dtype=torch.int16)
        buf_rival_pick = torch.empty((H, E, N), dtype=torch.int16)
        buf_beh_logp = torch.empty((H, E, N), dtype=torch.float32)
        buf_reward = torch.empty((H, E, N), dtype=torch.float32)
        buf_wm_in = torch.empty((H, E, N), dtype=torch.bool)
        buf_wm_out = torch.empty((H, E, N), dtype=torch.bool)
        buf_val_w = torch.empty((H, E, N), dtype=torch.float32)
        buf_val_l = torch.empty((H, E, N), dtype=torch.float32)
        buf_not_done = torch.empty((H, E), dtype=torch.bool)
        buf_actor_mask = torch.empty((H, E, N), dtype=torch.bool)

        start_scores = eng.t_score.clone()
        score_outs = 0

        for t in range(H):
            if G == 1:
                fin_pre = eng.t_finished
                if bool(fin_pre.any().item()):
                    eng.reset(fin_pre)
                    fids = np.nonzero(fin_pre.numpy())[0]
                    eng.set_skills(
                        sample_roster_skills(slot.dr_rng, len(fids), N, K, slot.roster_mode), env_ids=fids,
                    )
                was_finished = torch.zeros((E,), dtype=torch.bool)
            else:
                was_finished = eng.t_finished.clone()

            obs_in, carts_in, teams_in, _ = eng.observe_tensors()
            obs_in_c = obs_in.clone()
            carts_in_c = carts_in.clone()
            teams_in_c = teams_in.clone()

            plan, hp, tdata = slot.dispatcher.act(obs_in_c, carts_in_c, teams_in_c, slot.geo, seat_arms)
            eng.apply_plan(plan, hp)
            eng.step(sc.dt, mode=sc.mode)
            obs_out, carts_out, teams_out, _ = eng.observe_tensors()
            now_finished = eng.t_finished
            score_outs += int(((~was_finished) & now_finished).sum().item())

            rew, wm_in, wm_out = slot.rew_gpu.compute(
                obs_in_c, carts_in_c, teams_in_c,
                obs_out, carts_out, teams_out,
                was_finished, now_finished, eng.t_winner,
            )
            active_env = ~was_finished

            buf_obs[t] = obs_in_c
            buf_carts[t] = carts_in_c
            buf_teams[t] = teams_in_c
            buf_obs_next[t] = obs_out
            buf_carts_next[t] = carts_out
            buf_teams_next[t] = teams_out
            buf_cont[t] = tdata["cont"]
            buf_cart_pick[t] = tdata["cart_pick"]
            buf_rival_pick[t] = tdata["rival_pick"]
            buf_beh_logp[t] = tdata["logp"]
            buf_reward[t] = rew
            buf_wm_in[t] = wm_in
            buf_wm_out[t] = wm_out
            buf_val_w[t] = tdata["value_winnie"]
            buf_val_l[t] = tdata["value_lou"]
            buf_not_done[t] = ~now_finished
            buf_actor_mask[t] = tdata["actor_mask"] & active_env[:, None]

        # Bootstrap value at t = H
        obs_f, carts_f, teams_f, _ = eng.observe_tensors()
        c_main = slot.dispatcher._get("main", E, 0)
        assert c_main is not None
        out_f = c_main.forward(obs_f, carts_f, teams_f, geo=slot.geo)
        end_finished = eng.t_finished.clone()
        end_scores = eng.t_score.clone()

        # Advance persistent base pool directly from branch 0 of the fork so matches progress continuously!
        if G > 1:
            slot.live_eng._state_u8[:] = slot.fork_eng._state_u8[0::G]
        slot.steps_elapsed += H

        # Winnie/Lou GAE backward pass
        gamma = float(self.cfg.losses.gamma)
        lam = float(self.cfg.losses.gae_lambda)
        val_in = torch.where(buf_wm_in, buf_val_w, buf_val_l)
        v_boot_last = torch.where(buf_wm_out[H - 1], out_f.value_winnie.float(), out_f.value_lou.float())
        v_boot_last = torch.where((~end_finished)[:, None], v_boot_last, torch.zeros_like(v_boot_last))

        gae_adv = torch.zeros((H, E, N), dtype=torch.float32)
        next_gae = torch.zeros((E, N), dtype=torch.float32)
        for t in range(H - 1, -1, -1):
            still_going = buf_not_done[t][:, None]
            same_role = buf_wm_in[t] == buf_wm_out[t]
            v_nxt = val_in[t + 1] if t + 1 < H else v_boot_last
            v_nxt = torch.where(still_going & same_role, v_nxt, torch.zeros_like(v_nxt))
            g_nxt = torch.where(still_going & same_role, next_gae, torch.zeros_like(next_gae))
            delta = buf_reward[t] + gamma * v_nxt - val_in[t]
            cur_gae = delta + gamma * lam * g_nxt
            gae_adv[t] = cur_gae
            next_gae = cur_gae
        gae_ret = gae_adv + val_in

        # Hybrid counterfactual GRPO (when G > 1) + normalized GAE
        am_f = buf_actor_mask.float()
        am_sum = am_f.sum().clamp_min(1.0)
        gae_mean = (gae_adv * am_f).sum() / am_sum
        gae_var = (((gae_adv - gae_mean) ** 2) * am_f).sum() / am_sum
        gae_norm = (gae_adv - gae_mean) / gae_var.clamp_min(1e-6).sqrt()

        if G > 1 and float(sc.grpo_weight) > 0.0:
            disc_pows = (gamma ** torch.arange(H, dtype=torch.float32)).view(H, 1, 1)
            main_mask_0 = buf_actor_mask[0].float()
            main_cnt = main_mask_0.sum(dim=-1).clamp_min(1.0)
            ep_ret_agent = (buf_reward * disc_pows).sum(dim=0) + (gamma ** H) * v_boot_last
            ep_ret_team = (ep_ret_agent * main_mask_0).sum(dim=-1) / main_cnt
            grp_ret = ep_ret_team.view(B, G)
            grp_std = grp_ret.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-3)
            grpo_adv = ((grp_ret - grp_ret.mean(dim=1, keepdim=True)) / grp_std).view(1, E, 1).expand(H, E, N)
            adv = float(sc.grpo_weight) * grpo_adv + float(sc.gae_weight) * gae_norm
            grpo_ret_mean = float(grp_ret.mean().item())
            grpo_within_std = float(grp_ret.std(dim=1, unbiased=False).mean().item())
        else:
            adv = gae_norm
            grpo_ret_mean = float(gae_ret.mean().item())
            grpo_within_std = 0.0

        # Pairwise win-rates vs opponents
        d_scores = end_scores - start_scores
        opp_wr_obs: dict[str, list[float]] = defaultdict(list)
        d_sc_np = d_scores.numpy()
        for ti in range(1, K):
            opp_arm = seat_arms[ti]
            wins = np.where(d_sc_np[:, 0] > d_sc_np[:, ti] + 1e-6, 1.0, np.where(d_sc_np[:, ti] > d_sc_np[:, 0] + 1e-6, 0.0, 0.5))
            opp_wr_obs[opp_arm].append(float(wins.mean()))

        wall_s = time.perf_counter() - t0
        sim_s = E * H * float(sc.dt)
        M = H * E
        return AtemporalBatch(
            obs=buf_obs.reshape(M, N, 84),
            carts=buf_carts.reshape(M, J, 18),
            teams=buf_teams.reshape(M, K, 7),
            obs_next=buf_obs_next.reshape(M, N, 84),
            carts_next=buf_carts_next.reshape(M, J, 18),
            teams_next=buf_teams_next.reshape(M, K, 7),
            cont=buf_cont.reshape(M, N, 11),
            cart_pick=buf_cart_pick.reshape(M, N),
            rival_pick=buf_rival_pick.reshape(M, N),
            beh_logp=buf_beh_logp.reshape(M, N),
            adv=adv.reshape(M, N),
            gae_ret=gae_ret.reshape(M, N),
            wm_in=buf_wm_in.reshape(M, N),
            actor_mask=buf_actor_mask.reshape(M, N),
            geo=slot.geo,
            policy_version=int(policy_version),
            world_meta={"K": K, "J": J, "N": N, "M": int(slot.spec["tiles"]), "slot": idx},
            opp_wr_obs={a: float(np.mean(vs)) for a, vs in opp_wr_obs.items()},
            stats={
                "wall_s": round(wall_s, 3),
                "sim_s": round(sim_s, 1),
                "speedup_x": round(sim_s / max(wall_s, 1e-6), 1),
                "agent_steps_s": round((E * H * N) / max(wall_s, 1e-6), 0),
                "warmup_steps": 0,
                "score_outs": score_outs,
                "grpo_ret_mean": round(grpo_ret_mean, 4),
                "grpo_within_std": round(grpo_within_std, 4),
                "main_dscore_mean": round(float(d_scores[:, 0].mean().item()), 2),
                "opp_dscore_mean": round(float(d_scores[:, 1:].mean().item()), 2) if K > 1 else 0.0,
                "seats": list(seat_arms),
            },
        )
