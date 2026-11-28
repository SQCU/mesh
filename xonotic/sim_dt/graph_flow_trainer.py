"""Graph-Native + Exact DPP + 1-NFE Flow Rollout Pool (G > 1 plural counterfactual groups)
and Metal 4 NAX Advantage-Weighted Boltzmann Flow Matching Trainer.
"""
from __future__ import annotations

import math
import random
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from mapgen.bsp_stream import sample_world_spec
from payload.tools.strategy_io_schema import OBS
from solver.strat.train_config import TrainingConfig

from .dataset_sampler import make_rules_for_profile, sample_roster_skills
from .graph_flow_policy import (
    GraphFlowPolicy,
    clone_graph_flow_model,
    compute_ess_boltzmann_weights,
    sample_orthogonal_team_bot_noise,
)
from .metal_graphflow import (
    CompiledGraphFlowOptimizer,
    CompiledGraphFlowPolicy,
    CompiledGraphFlowTrainer,
    GraphFlowWeightBank,
)
from .metal_runtime import MetalCGTRewardComputer, MetalDeltaTimePayloadEngine
from .ppo_trainer import _team_eff_rho_flat
from .world_graph import WorldGraphIR, build_gas_world_graph


@dataclass
class GraphFlowBatch:
    obs: torch.Tensor          # [M, N, 84] bf16
    nav: torch.Tensor          # [M, W_nav, 16] bf16
    carts: torch.Tensor        # [M, J, 18] bf16
    teams: torch.Tensor        # [M, K, 7] bf16
    acc: torch.Tensor          # [M, N, A_dim] fp32
    u_vel: torch.Tensor        # [M, N, A_dim] fp32
    adv: torch.Tensor          # [M, N] fp32
    gae_ret: torch.Tensor      # [M, N] fp32
    obs_drho: torch.Tensor     # [M, N] fp32
    wm_in: torch.Tensor        # [M, N] bool
    actor_mask: torch.Tensor   # [M, N] bool
    active_env: torch.Tensor   # [M] bool (~was_finished)
    nav_kern: torch.Tensor     # [W_nav, W_nav] fp32
    geo: dict[str, torch.Tensor]
    group_size: int
    policy_version: int
    world_meta: dict[str, int]
    opp_wr_obs: dict[str, float]
    stats: dict[str, Any]


def sample_blender_world_spec(seed: int, slot_idx: int) -> dict[str, Any]:
    """Full-blender stratified parametric world sampler spanning the entire Metal-safe domain:
      Tier 0 (Small): K in [2, 4],  N/K in [4, 10], J in [4, 18],  tiles in [2, 4]
      Tier 1 (Mid):   K in [5, 8],  N/K in [4, 9],  J in [9, 32],  tiles in [3, 6]
      Tier 2 (Large): K in [9, 14], N/K in [3, 8],  J in [15, 48], tiles in [6, 9]
    Also randomizes tile_size, joint_width, checkpoints_per_cart, and point_limit_base!
    Guarantees K <= 14, J <= 52, N <= 108, W_nav <= 48, A_dim <= 66, M_ent <= 228.
    """
    rng = random.Random(int(seed))
    tier = int(slot_idx) % 3
    if tier == 0:
        K = rng.choice((2, 3, 4))
        per_team = rng.randint(4, 10)
        mult = rng.uniform(1.5, 4.5)
        tiles = rng.choice((2, 3, 4))
        tile_size = rng.choice((1408.0, 1664.0, 1792.0, 2048.0))
    elif tier == 1:
        K = rng.choice((5, 6, 7, 8))
        per_team = rng.randint(4, 9)
        mult = rng.uniform(1.6, 4.0)
        tiles = rng.choice((3, 4, 6))
        tile_size = rng.choice((1536.0, 1792.0, 2048.0))
    else:
        K = rng.choice((9, 10, 11, 12, 14))
        per_team = rng.randint(3, 8)
        mult = rng.uniform(1.5, 3.6)
        tiles = rng.choice((6, 8, 9))
        tile_size = rng.choice((1664.0, 1792.0, 2176.0))
    N = max(max(12, 3 * K), min(104, K * per_team))
    N = (N // K) * K
    if N < 12:
        N = K * max(3, int(math.ceil(12.0 / K)))
    J = max(K + 1, min(48, int(round(K * mult))))
    return {
        "seed": int(seed),
        "family": "gas_world_bsp",
        "tier": tier,
        "teams": K,
        "carts": J,
        "players": N,
        "tiles": tiles,
        "tile_size": float(tile_size),
        "joint_width": float(rng.choice((288.0, 320.0, 352.0, 384.0))),
        "checkpoints_per_cart": int(rng.choice((3, 4, 5))),
        "point_limit_base": float(rng.choice((120.0, 160.0, 200.0))),
    }


class _GraphFlowSlot:
    """One persistent world slot owning live & counterfactual-fork Metal engines + compiled GraphFlowPolicies."""

    def __init__(
        self,
        world: WorldGraphIR,
        spec: dict[str, Any],
        cfg: TrainingConfig,
        variants: dict[str, GraphFlowPolicy | None],
        *,
        seed_base: int,
        group_size: int,
        physics_profile: str = "baseline",
        roster_mode: str = "mixed_dr",
        tau_rel: float = 2.5,
        n_phys_substeps: int = 2,
    ):
        sc = cfg.sim_dt
        self.world = world
        self.spec = dict(spec)
        self.K = int(world.teams)
        self.J = int(world.carts)
        self.N = int(world.players_default)
        self.W_nav = int(world.num_nav)
        self.A_dim = self.W_nav + self.K + 4
        self.B = int(sc.base_envs)
        self.G = max(1, int(group_size))
        self.E = self.B * self.G
        self.tau_rel = float(tau_rel)
        self.n_phys_substeps = max(1, int(n_phys_substeps))
        self.physics_profile = str(physics_profile)
        self.roster_mode = str(roster_mode)
        self.dr_rng = np.random.default_rng(seed_base + 4242)
        self.noise_gen = torch.Generator(device="cpu").manual_seed(seed_base + 7777)
        pl_base = float(spec.get("point_limit_base", sc.point_limit_base))
        self.rules = make_rules_for_profile(pl_base, self.physics_profile)
        self.geo = {"map_center": world.map_center.clone(), "map_radius": world.map_radius.clone()}
        self.nav_kern = world.nav_kern.detach().cpu().float().contiguous()

        n_live = self.E if self.G == 1 else self.B
        self.live_eng = MetalDeltaTimePayloadEngine(
            world, num_envs=n_live, num_players=self.N, rules=self.rules, base_seed=seed_base,
        )
        self.live_eng.set_skills(sample_roster_skills(self.dr_rng, n_live, self.N, self.K, self.roster_mode))
        self.fork_eng: MetalDeltaTimePayloadEngine | None = (
            MetalDeltaTimePayloadEngine(
                world, num_envs=self.E, num_players=self.N, rules=self.rules, base_seed=seed_base + 50000,
            )
            if self.G > 1
            else None
        )
        self.rew_gpu = MetalCGTRewardComputer(E=self.E, N=self.N, J=self.J, K=self.K)
        self.rew_gpu.set_config(cfg.losses.reward, terminal_win_bonus=float(sc.terminal_win_bonus))
        self._variants = variants
        self._banks: dict[str, GraphFlowWeightBank] = {}
        self._compiled: dict[str, CompiledGraphFlowPolicy] = {}

    def close(self) -> None:
        for c in self._compiled.values():
            c.close()
        self._compiled.clear()
        self.rew_gpu.close()
        self.live_eng.close()
        if self.fork_eng is not None:
            self.fork_eng.close()

    def _get_pol(self, arm: str) -> CompiledGraphFlowPolicy | None:
        mdl = self._variants.get(arm)
        if mdl is None:
            return None
        c = self._compiled.get(arm)
        if c is None:
            bank = self._banks.get(arm)
            c = CompiledGraphFlowPolicy(
                mdl, E=self.E, N=self.N, W_nav=self.W_nav, J=self.J, K=self.K,
                nav_kern=self.nav_kern, weight_bank=bank,
            )
            self._compiled[arm] = c
        return c

    def sync_variant_(self, arm: str, model: GraphFlowPolicy) -> None:
        self._variants[arm] = model
        c = self._compiled.get(arm)
        if c is not None and c._owns_bank:
            c.sync_weights_(model)


class GraphFlowRolloutPool:
    """Persistent multi-world pool running G > 1 plural counterfactual branches with orthogonalized
    Strategy (team-shared) + Tactic (per-bot) base noise draws."""

    def __init__(
        self,
        cfg: TrainingConfig,
        variants: dict[str, GraphFlowPolicy | None],
        *,
        num_worlds: int = 4,
        group_size: int = 4,
        seed_base: int = 30000,
        world_seed_base: int | None = None,
        blender_dr: bool = False,
        self_play_only: bool = False,
        roster_mode: str = "mixed_dr",
        randomize_physics: bool = True,
        tau_rel: float = 2.5,
        n_phys_substeps: int = 2,
        team_corr: float = 0.45,
    ):
        self.cfg = cfg
        self.sc = cfg.sim_dt
        self.variants = variants
        self.G = max(1, int(group_size))
        self.blender_dr = bool(blender_dr)
        self.self_play_only = bool(self_play_only)
        self.roster_mode = str(roster_mode)
        self.randomize_physics = bool(randomize_physics)
        self.tau_rel = float(tau_rel)
        self.n_phys_substeps = int(n_phys_substeps)
        self.team_corr = float(team_corr)
        self.seed_base = int(seed_base)
        self.world_seed_base = int(seed_base if world_seed_base is None else world_seed_base)
        self.rng = np.random.default_rng(seed_base + 777)
        self.slots: list[_GraphFlowSlot] = []
        self._banks: dict[str, GraphFlowWeightBank] = {}
        self._external_banks: set[str] = set()
        for arm, mdl in variants.items():
            if mdl is not None:
                self._banks[arm] = GraphFlowWeightBank(mdl, with_grads=False)
        self._cursor = 0
        self._refresh_gen = 0
        self._phys_cycle = ("baseline", "fast", "slow", "volatile") if randomize_physics else ("baseline",)

        for wi in range(max(1, int(num_worlds))):
            self.slots.append(self._make_slot(wi, self.world_seed_base + (wi + 1) * 101, self.seed_base + wi * 10009))

    def _make_slot(self, wi: int, w_seed: int, m_seed: int) -> _GraphFlowSlot:
        if self.blender_dr:
            spec = sample_blender_world_spec(w_seed, wi)
        else:
            spec = sample_world_spec(w_seed, self.cfg.world)
        map_name = f"gf_w{wi}_s{w_seed}_k{spec['teams']}_j{spec['carts']}_n{spec['players']}"
        world = build_gas_world_graph(
            map_name=map_name,
            teams=int(spec["teams"]),
            carts=int(spec["carts"]),
            players=int(spec["players"]),
            tiles=int(spec["tiles"]),
            tile_size=float(spec.get("tile_size", 1792.0)),
            joint_width=float(spec.get("joint_width", 320.0)),
            checkpoints_per_cart=int(spec.get("checkpoints_per_cart", 4)),
            seed=int(spec["seed"]),
            device=torch.device("cpu"),
        )
        slot = _GraphFlowSlot(
            world, spec, self.cfg, self.variants,
            seed_base=m_seed,
            group_size=self.G,
            physics_profile=self._phys_cycle[(wi + self._refresh_gen) % len(self._phys_cycle)],
            roster_mode=self.roster_mode,
            tau_rel=self.tau_rel,
            n_phys_substeps=self.n_phys_substeps,
        )
        slot._banks = self._banks
        stagger = ((wi + self._refresh_gen) * 12) % 40
        if stagger > 0:
            slot.live_eng.step(self.sc.dt, mode=self.sc.mode, n_steps=stagger)
            fin = slot.live_eng.t_finished
            if bool(fin.any().item()):
                slot.live_eng.reset(fin)
        return slot

    def refresh_all_worlds_(self, gen: int) -> None:
        """Replace all pool slots with fresh parametric worlds (matched across RDMA ranks!)."""
        self._refresh_gen = int(gen)
        n_w = len(self.slots)
        for s in self.slots:
            s.close()
        self.slots.clear()
        for wi in range(n_w):
            w_seed = self.world_seed_base + self._refresh_gen * 10007 + (wi + 1) * 101
            m_seed = self.seed_base + self._refresh_gen * 50021 + wi * 10009
            self.slots.append(self._make_slot(wi, w_seed, m_seed))

    def close(self) -> None:
        for s in self.slots:
            s.close()
        self.slots.clear()
        for arm, b in list(self._banks.items()):
            if arm not in self._external_banks:
                b.close()
        self._banks.clear()

    def attach_weight_bank_(self, arm: str, bank: GraphFlowWeightBank, model: GraphFlowPolicy) -> None:
        """Share a learner's live GraphFlowWeightBank zero-copy across ALL world slots!"""
        old = self._banks.get(arm)
        if old is not None and old is not bank and arm not in self._external_banks:
            old.close()
        self._banks[arm] = bank
        self._external_banks.add(arm)
        self.variants[arm] = model
        for s in self.slots:
            c = s._compiled.pop(arm, None)
            if c is not None:
                c.close()

    def detach_weight_bank_(self, arm: str, model: GraphFlowPolicy) -> None:
        """Replace an external learner bank with a pool-owned frozen bank before closing the learner."""
        self._external_banks.discard(arm)
        self._banks[arm] = GraphFlowWeightBank(model, with_grads=False)
        self.variants[arm] = model
        for s in self.slots:
            c = s._compiled.pop(arm, None)
            if c is not None:
                c.close()

    def sync_variant_(self, arm: str, model: GraphFlowPolicy) -> None:
        self.variants[arm] = model
        bank = self._banks.get(arm)
        if bank is None:
            bank = GraphFlowWeightBank(model, with_grads=False)
            self._banks[arm] = bank
        elif arm not in self._external_banks:
            # Shared bank: repack ONCE instead of num_worlds times!
            bank.sync_from_model_(model)
        # If arm is an external learner bank, CompiledGraphFlowOptimizer/RDMA already updated it in-place!

    def _choose_seats(
        self,
        K: int,
        pfsp_pool: list[str],
        pfsp_wr: dict[str, float],
        *,
        train_arm: str = "main",
        exploiter_target: str | None = None,
    ) -> list[str]:
        if exploiter_target is not None:
            # Adversarial exploiter training: Seat 0 is exploiter, rivals are predominantly the frozen target!
            seats = [train_arm]
            for s_i in range(1, K):
                if s_i == K - 1 and K >= 4 and float(self.rng.random()) < 0.25:
                    seats.append("default")
                else:
                    seats.append(exploiter_target)
            return seats
        if self.self_play_only or K <= 1:
            return [train_arm] * K
        lg = self.cfg.league
        seats = [train_arm]
        cands = sorted(set(pfsp_pool))
        w = np.asarray(
            [max(lg.pfsp_min_weight, 1.0 - float(pfsp_wr.get(a, lg.pfsp_default_winrate))) ** lg.pfsp_power for a in cands],
            dtype=np.float64,
        )
        w /= w.sum()
        if K == 2:
            seats.append(str(self.rng.choice(cands, p=w)))
        else:
            seats.append("default" if float(self.rng.random()) < 0.45 else str(self.rng.choice(cands, p=w)))
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
        train_arm: str = "main",
        exploiter_target: str | None = None,
    ) -> GraphFlowBatch:
        t0 = time.perf_counter()
        idx = self._cursor % len(self.slots) if slot_index is None else (int(slot_index) % len(self.slots))
        self._cursor += 1
        slot = self.slots[idx]
        sc = self.sc
        B, G, E, H = slot.B, slot.G, slot.E, int(sc.rollout_horizon)
        N, Wn, J, K, A_dim = slot.N, slot.W_nav, slot.J, slot.K, slot.A_dim
        lcfg = self.cfg.losses

        if seat_arms is None:
            seat_arms = self._choose_seats(
                K,
                pfsp_pool or ["default", "main_init", "ema", "main"],
                pfsp_wr or {},
                train_arm=train_arm,
                exploiter_target=exploiter_target,
            )

        if G > 1:
            assert slot.fork_eng is not None
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

        buf_obs = torch.empty((H, E, N, 84), dtype=torch.bfloat16)
        buf_nav = torch.empty((H, E, Wn, 16), dtype=torch.bfloat16)
        buf_carts = torch.empty((H, E, J, 18), dtype=torch.bfloat16)
        buf_teams = torch.empty((H, E, K, 7), dtype=torch.bfloat16)
        buf_acc = torch.empty((H, E, N, A_dim), dtype=torch.float32)
        buf_u_vel = torch.empty((H, E, N, A_dim), dtype=torch.float32)
        buf_reward = torch.empty((H, E, N), dtype=torch.float32)
        buf_drho = torch.empty((H, E, N), dtype=torch.float32)
        buf_wm_in = torch.empty((H, E, N), dtype=torch.bool)
        buf_wm_out = torch.empty((H, E, N), dtype=torch.bool)
        buf_val_w = torch.empty((H, E, N), dtype=torch.float32)
        buf_val_l = torch.empty((H, E, N), dtype=torch.float32)
        buf_not_done = torch.empty((H, E), dtype=torch.bool)
        buf_active_env = torch.empty((H, E), dtype=torch.bool)
        buf_actor_mask = torch.empty((H, E, N), dtype=torch.bool)

        arm_to_teams: dict[str, list[int]] = defaultdict(list)
        for ti, arm in enumerate(seat_arms):
            arm_to_teams[arm].append(ti + 1)

        c_main = slot._get_pol(train_arm)
        assert c_main is not None
        d_e = c_main.d_e
        sigma_b = float(c_main.cfg.sigma_base)
        start_scores = eng.t_score.clone()
        score_outs = 0
        gk_hist = np.zeros((4,), dtype=np.int64)
        branch_u_sep_sum = 0.0
        branch_u_sep_cnt = 0
        pickups_tot = 0.0
        sat_cnt = 0
        sat_tot = 0
        u_sq_sum = 0.0
        u_sq_cnt = 0

        obs_in, nav_in, carts_in, teams_in, acc_in, _ = eng.observe_graph_tensors()
        ou_rho = max(0.0, min(0.95, float(getattr(c_main.cfg, "ou_rho", 0.0))))
        ou_scale = math.sqrt(max(0.0, 1.0 - ou_rho * ou_rho))
        q_ou_main: torch.Tensor | None = None
        for t in range(H):
            if G == 1:
                fin_pre = eng.t_finished
                if bool(fin_pre.any().item()):
                    eng.reset(fin_pre)
                    fids = np.nonzero(fin_pre.numpy())[0]
                    eng.set_skills(sample_roster_skills(slot.dr_rng, len(fids), N, K, slot.roster_mode), env_ids=fids)
                    obs_in, nav_in, carts_in, teams_in, acc_in, _ = eng.observe_graph_tensors()
                was_finished = torch.zeros((E,), dtype=torch.bool)
            else:
                was_finished = eng.t_finished.clone()

            obs_c = obs_in.clone()
            nav_c = nav_in.clone()
            carts_c = carts_in.clone()
            teams_c = teams_in.clone()
            acc_c = acc_in.clone()
            p_team = torch.round(obs_c[:, :, OBS["TEAM"]].float()).to(torch.int64)

            # Draw orthogonalized Strategy(team) + Tactic(bot) base noise (with optional OU temporal persistence!)
            eps_raw = sample_orthogonal_team_bot_noise(
                B, G, N, d_e, p_team, K,
                team_corr=self.team_corr, sigma=sigma_b, generator=slot.noise_gen,
            )
            if ou_rho > 0.0 and q_ou_main is not None:
                q_ou_main = ou_rho * q_ou_main + ou_scale * eps_raw
                eps_main = q_ou_main
            else:
                q_ou_main = eps_raw
                eps_main = eps_raw

            u_vel = torch.zeros((E, N, A_dim), dtype=torch.float32)
            has_vel = torch.zeros((E, N), dtype=torch.bool)
            main_mask = torch.zeros((E, N), dtype=torch.bool)
            val_w = torch.zeros((E, N), dtype=torch.float32)
            val_l = torch.zeros((E, N), dtype=torch.float32)

            if len(arm_to_teams) == 1 and train_arm in arm_to_teams:
                res = c_main.sample_velocity(obs_c, nav_c, carts_c, teams_c, acc_c, geo=slot.geo, eps=eps_main)
                u_vel = res["u_vel"]
                has_vel.fill_(True)
                main_mask.fill_(True)
                val_w = res["value_winnie"]
                val_l = res["value_lou"]
            else:
                for arm, t_list in arm_to_teams.items():
                    mask = torch.zeros((E, N), dtype=torch.bool)
                    for tid in t_list:
                        mask = mask | (p_team == tid)
                    if arm == "default":
                        continue  # u = 0 => exact unmodified stock HavocBot!
                    cpol = slot._get_pol(arm)
                    if cpol is None:
                        continue
                    eps_cur = eps_main if arm == train_arm else None
                    res = cpol.sample_velocity(obs_c, nav_c, carts_c, teams_c, acc_c, geo=slot.geo, eps=eps_cur)
                    u_vel = torch.where(mask.unsqueeze(-1), res["u_vel"], u_vel)
                    has_vel = has_vel | mask
                    if arm == train_arm:
                        main_mask = mask
                        val_w = res["value_winnie"]
                        val_l = res["value_lou"]

            # Measure inter-branch velocity separation & box-wall saturation on main actors
            if bool(main_mask.any().item()):
                um = u_vel[main_mask]
                sat_cnt += int((um.abs() >= (float(c_main.cfg.u_clip) - 0.1)).sum().item())
                sat_tot += int(um.numel())
                u_sq_sum += float(um.square().mean(dim=-1).sqrt().sum().item())
                u_sq_cnt += int(um.shape[0])
            if G > 1 and (t % 4 == 0):
                ug = u_vel.view(B, G, N, A_dim)
                diff_g = (ug[:, :, None, :, :] - ug[:, None, :, :, :]).square().mean(dim=-1).sqrt()
                tri = ~torch.eye(G, dtype=torch.bool).view(1, G, G, 1)
                branch_u_sep_sum += float(diff_g.masked_select(tri.expand(B, G, G, N)).mean().item())
                branch_u_sep_cnt += 1

            # Step physics with velocity steering (n_phys_substeps fused into ONE Metal CB!)
            eng.apply_vel(u_vel, has_vel, tau_rel=slot.tau_rel)
            eng.step(float(sc.dt), mode=str(sc.mode), reset_outcomes=True, n_steps=slot.n_phys_substeps, tau_rel=slot.tau_rel)

            # Single observe_graph_tensors() per step — reused as next step's input and final bootstrap!
            obs_out, nav_out, carts_out, teams_out, acc_out, _ = eng.observe_graph_tensors()
            now_finished = eng.t_finished
            score_outs += int(((~was_finished) & now_finished).sum().item())

            rew, wm_in, wm_out = slot.rew_gpu.compute(
                obs_c, carts_c, teams_c, obs_out, carts_out, teams_out,
                was_finished, now_finished, eng.t_winner,
            )
            rc = _team_eff_rho_flat(carts_c, teams_c, obs_c, partial_cp_weight=lcfg.reward.cart_partial_cp_weight)
            rn = _team_eff_rho_flat(carts_out, teams_out, obs_out, partial_cp_weight=lcfg.reward.cart_partial_cp_weight)
            drho = torch.asinh(lcfg.dyn_drho_scale * (rn - rc))

            sv = eng.state_views()
            gk_m = sv["p_goal_kind"][main_mask].numpy()
            if gk_m.size > 0:
                for k_i in range(4):
                    gk_hist[k_i] += int((gk_m == k_i).sum())
            pickups_tot += float(sv["p_pickups"][main_mask].sum().item())

            active_env = ~was_finished
            buf_obs[t] = obs_c
            buf_nav[t] = nav_c
            buf_carts[t] = carts_c
            buf_teams[t] = teams_c
            buf_acc[t] = acc_c
            buf_u_vel[t] = u_vel
            buf_reward[t] = rew
            buf_drho[t] = drho
            buf_wm_in[t] = wm_in
            buf_wm_out[t] = wm_out
            buf_val_w[t] = val_w
            buf_val_l[t] = val_l
            buf_not_done[t] = ~now_finished
            buf_active_env[t] = active_env
            buf_actor_mask[t] = main_mask & active_env[:, None]
            obs_in, nav_in, carts_in, teams_in, acc_in = obs_out, nav_out, carts_out, teams_out, acc_out

        # Bootstrap value at t = H (reusing last obs_in/nav_in/carts_in/teams_in/acc_in!)
        out_f = c_main.sample_velocity(obs_in, nav_in, carts_in, teams_in, acc_in, geo=slot.geo, output_scale=0.0)
        end_finished = eng.t_finished.clone()
        end_scores = eng.t_score.clone()

        if G > 1:
            slot.live_eng._state_u8[:] = slot.fork_eng._state_u8[0::G]

        # Winnie/Lou GAE
        gamma = float(lcfg.gamma)
        lam = float(lcfg.gae_lambda)
        val_in = torch.where(buf_wm_in, buf_val_w, buf_val_l)
        v_boot = torch.where(buf_wm_out[H - 1], out_f["value_winnie"], out_f["value_lou"])
        v_boot = torch.where((~end_finished)[:, None], v_boot, torch.zeros_like(v_boot))

        gae_adv = torch.zeros((H, E, N), dtype=torch.float32)
        next_gae = torch.zeros((E, N), dtype=torch.float32)
        for t in range(H - 1, -1, -1):
            still = buf_not_done[t][:, None]
            same_role = buf_wm_in[t] == buf_wm_out[t]
            v_nxt = val_in[t + 1] if t + 1 < H else v_boot
            v_nxt = torch.where(still & same_role, v_nxt, torch.zeros_like(v_nxt))
            g_nxt = torch.where(still & same_role, next_gae, torch.zeros_like(next_gae))
            delta = buf_reward[t] + gamma * v_nxt - val_in[t]
            cur_gae = delta + gamma * lam * g_nxt
            gae_adv[t] = cur_gae
            next_gae = cur_gae
        gae_ret = gae_adv + val_in

        # Hybrid counterfactual GRPO (group-relative across G branches from identical savestates!) + GAE
        am_f = buf_actor_mask.float()
        am_sum = am_f.sum().clamp_min(1.0)
        g_mean = (gae_adv * am_f).sum() / am_sum
        g_std = (((gae_adv - g_mean).square() * am_f).sum() / am_sum).clamp_min(1e-6).sqrt()
        gae_norm = (gae_adv - g_mean) / g_std

        if G > 1 and float(sc.grpo_weight) > 0.0:
            disc_pows = (gamma ** torch.arange(H, dtype=torch.float32)).view(H, 1, 1)
            m0 = buf_actor_mask[0].float()
            mcnt = m0.sum(dim=-1).clamp_min(1.0)
            ep_ret_ag = (buf_reward * disc_pows).sum(dim=0) + (gamma ** H) * v_boot
            ep_ret_tm = (ep_ret_ag * m0).sum(dim=-1) / mcnt
            grp_ret = ep_ret_tm.view(B, G)
            grp_std = grp_ret.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-3)
            grpo_adv = ((grp_ret - grp_ret.mean(dim=1, keepdim=True)) / grp_std).view(1, E, 1).expand(H, E, N)
            adv = float(sc.grpo_weight) * grpo_adv + float(sc.gae_weight) * gae_norm
            grpo_within_std = float(grp_ret.std(dim=1, unbiased=False).mean().item())
        else:
            adv = gae_norm
            grpo_within_std = 0.0

        d_scores = end_scores - start_scores
        opp_wr_obs: dict[str, list[float]] = defaultdict(list)
        d_sc_np = d_scores.numpy()
        for ti in range(1, K):
            opp_arm = seat_arms[ti]
            wins = np.where(d_sc_np[:, 0] > d_sc_np[:, ti] + 1e-6, 1.0, np.where(d_sc_np[:, ti] > d_sc_np[:, 0] + 1e-6, 0.0, 0.5))
            opp_wr_obs[opp_arm].append(float(wins.mean()))

        wall_s = time.perf_counter() - t0
        sim_s = E * H * slot.n_phys_substeps * float(sc.dt)
        M = H * E
        gk_tot = max(1, int(gk_hist.sum()))
        return GraphFlowBatch(
            obs=buf_obs.reshape(M, N, 84),
            nav=buf_nav.reshape(M, Wn, 16),
            carts=buf_carts.reshape(M, J, 18),
            teams=buf_teams.reshape(M, K, 7),
            acc=buf_acc.reshape(M, N, A_dim),
            u_vel=buf_u_vel.reshape(M, N, A_dim),
            adv=adv.reshape(M, N),
            gae_ret=gae_ret.reshape(M, N),
            obs_drho=buf_drho.reshape(M, N),
            wm_in=buf_wm_in.reshape(M, N),
            actor_mask=buf_actor_mask.reshape(M, N),
            active_env=buf_active_env.reshape(M),
            nav_kern=slot.nav_kern,
            geo=slot.geo,
            group_size=G,
            policy_version=int(policy_version),
            world_meta={"K": K, "J": J, "N": N, "W_nav": Wn, "M": int(slot.spec["tiles"]), "slot": idx},
            opp_wr_obs={a: float(np.mean(vs)) for a, vs in opp_wr_obs.items()},
            stats={
                "wall_s": round(wall_s, 3),
                "sim_s": round(sim_s, 1),
                "speedup_x": round(sim_s / max(wall_s, 1e-6), 1),
                "agent_steps_s": round((E * H * slot.n_phys_substeps * N) / max(wall_s, 1e-6), 0),
                "score_outs": score_outs,
                "grpo_within_std": round(grpo_within_std, 4),
                "branch_u_sep": round(branch_u_sep_sum / max(1, branch_u_sep_cnt), 4),
                "sat_frac": round(float(sat_cnt) / max(1, sat_tot), 4),
                "u_rms": round(float(u_sq_sum) / max(1, u_sq_cnt), 4),
                "main_dscore_mean": round(float(d_scores[:, 0].mean().item()), 2),
                "opp_dscore_mean": round(float(d_scores[:, 1:].mean().item()), 2) if K > 1 else 0.0,
                "gk_pct": [round(100.0 * c / gk_tot, 1) for c in gk_hist.tolist()],
                "pickups_per_match": round(pickups_tot / max(1, E), 2),
                "seats": list(seat_arms),
            },
        )

    @torch.no_grad()
    def sample_batch_multi(
        self,
        *,
        active_arms: list[str],
        seat_arms: list[str],
        policy_versions: dict[str, int] | None = None,
        slot_index: int | None = None,
    ) -> tuple[dict[str, GraphFlowBatch], dict[str, Any]]:
        """ONE shared Metal physics rollout harvesting counterfactual GraphFlowBatches for ALL concurrent active learners!"""
        t0 = time.perf_counter()
        idx = self._cursor % len(self.slots) if slot_index is None else (int(slot_index) % len(self.slots))
        self._cursor += 1
        slot = self.slots[idx]
        sc = self.sc
        B, G, E, H = slot.B, slot.G, slot.E, int(sc.rollout_horizon)
        N, Wn, J, K, A_dim = slot.N, slot.W_nav, slot.J, slot.K, slot.A_dim
        lcfg = self.cfg.losses
        pvers = policy_versions or {}

        # Normalize seat_arms length to K
        seats = list(seat_arms[:K])
        while len(seats) < K:
            seats.append("default")

        arm_to_teams: dict[str, list[int]] = defaultdict(list)
        for ti, arm in enumerate(seats):
            arm_to_teams[arm].append(ti + 1)

        present_active = [a for a in active_arms if a in arm_to_teams]
        if not present_active:
            present_active = [active_arms[0]]
            seats[0] = present_active[0]
            arm_to_teams.clear()
            for ti, arm in enumerate(seats):
                arm_to_teams[arm].append(ti + 1)

        if G > 1:
            assert slot.fork_eng is not None
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

        buf_obs = torch.empty((H, E, N, 84), dtype=torch.bfloat16)
        buf_nav = torch.empty((H, E, Wn, 16), dtype=torch.bfloat16)
        buf_carts = torch.empty((H, E, J, 18), dtype=torch.bfloat16)
        buf_teams = torch.empty((H, E, K, 7), dtype=torch.bfloat16)
        buf_acc = torch.empty((H, E, N, A_dim), dtype=torch.float32)
        buf_u_vel = torch.empty((H, E, N, A_dim), dtype=torch.float32)
        buf_reward = torch.empty((H, E, N), dtype=torch.float32)
        buf_drho = torch.empty((H, E, N), dtype=torch.float32)
        buf_wm_in = torch.empty((H, E, N), dtype=torch.bool)
        buf_wm_out = torch.empty((H, E, N), dtype=torch.bool)
        buf_not_done = torch.empty((H, E), dtype=torch.bool)
        buf_active_env = torch.empty((H, E), dtype=torch.bool)

        buf_val_w = {a: torch.zeros((H, E, N), dtype=torch.float32) for a in present_active}
        buf_val_l = {a: torch.zeros((H, E, N), dtype=torch.float32) for a in present_active}
        buf_amask = {a: torch.zeros((H, E, N), dtype=torch.bool) for a in present_active}

        pol_map = {a: slot._get_pol(a) for a in arm_to_teams if a != "default"}
        start_scores = eng.t_score.clone()
        score_outs = 0
        gk_hist = {a: np.zeros((4,), dtype=np.int64) for a in present_active}
        sat_cnt = {a: 0 for a in present_active}
        sat_tot = {a: 0 for a in present_active}
        u_sq_sum = {a: 0.0 for a in present_active}
        u_sq_cnt = {a: 0 for a in present_active}

        obs_in, nav_in, carts_in, teams_in, acc_in, _ = eng.observe_graph_tensors()
        q_ou_arms: dict[str, torch.Tensor | None] = {a: None for a in present_active}
        for t in range(H):
            if G == 1:
                fin_pre = eng.t_finished
                if bool(fin_pre.any().item()):
                    eng.reset(fin_pre)
                    fids = np.nonzero(fin_pre.numpy())[0]
                    eng.set_skills(sample_roster_skills(slot.dr_rng, len(fids), N, K, slot.roster_mode), env_ids=fids)
                    obs_in, nav_in, carts_in, teams_in, acc_in, _ = eng.observe_graph_tensors()
                was_finished = torch.zeros((E,), dtype=torch.bool)
            else:
                was_finished = eng.t_finished.clone()

            obs_c = obs_in.clone()
            nav_c = nav_in.clone()
            carts_c = carts_in.clone()
            teams_c = teams_in.clone()
            acc_c = acc_in.clone()
            p_team = torch.round(obs_c[:, :, OBS["TEAM"]].float()).to(torch.int64)

            u_vel = torch.zeros((E, N, A_dim), dtype=torch.float32)
            has_vel = torch.zeros((E, N), dtype=torch.bool)
            active_masks_t: dict[str, torch.Tensor] = {}

            for arm, t_list in arm_to_teams.items():
                mask = torch.zeros((E, N), dtype=torch.bool)
                for tid in t_list:
                    mask = mask | (p_team == tid)
                if arm == "default":
                    continue
                cpol = pol_map.get(arm)
                if cpol is None:
                    continue
                eps_cur = None
                if arm in buf_amask:
                    eps_raw = sample_orthogonal_team_bot_noise(
                        B, G, N, cpol.d_e, p_team, K,
                        team_corr=self.team_corr, sigma=float(cpol.cfg.sigma_base), generator=slot.noise_gen,
                    )
                    rho_a = max(0.0, min(0.95, float(getattr(cpol.cfg, "ou_rho", 0.0))))
                    if rho_a > 0.0 and q_ou_arms[arm] is not None:
                        q_ou_arms[arm] = rho_a * q_ou_arms[arm] + math.sqrt(max(0.0, 1.0 - rho_a * rho_a)) * eps_raw
                        eps_cur = q_ou_arms[arm]
                    else:
                        q_ou_arms[arm] = eps_raw
                        eps_cur = eps_raw
                res = cpol.sample_velocity(obs_c, nav_c, carts_c, teams_c, acc_c, geo=slot.geo, eps=eps_cur)
                u_vel = torch.where(mask.unsqueeze(-1), res["u_vel"], u_vel)
                has_vel = has_vel | mask
                if arm in buf_amask:
                    active_masks_t[arm] = mask
                    buf_val_w[arm][t] = res["value_winnie"]
                    buf_val_l[arm][t] = res["value_lou"]
                    if bool(mask.any().item()):
                        um = res["u_vel"][mask]
                        sat_cnt[arm] += int((um.abs() >= (float(cpol.cfg.u_clip) - 0.1)).sum().item())
                        sat_tot[arm] += int(um.numel())
                        u_sq_sum[arm] += float(um.square().mean(dim=-1).sqrt().sum().item())
                        u_sq_cnt[arm] += int(um.shape[0])

            eng.apply_vel(u_vel, has_vel, tau_rel=slot.tau_rel)
            eng.step(float(sc.dt), mode=str(sc.mode), reset_outcomes=True, n_steps=slot.n_phys_substeps, tau_rel=slot.tau_rel)

            obs_out, nav_out, carts_out, teams_out, acc_out, _ = eng.observe_graph_tensors()
            now_finished = eng.t_finished
            score_outs += int(((~was_finished) & now_finished).sum().item())

            rew, wm_in, wm_out = slot.rew_gpu.compute(
                obs_c, carts_c, teams_c, obs_out, carts_out, teams_out,
                was_finished, now_finished, eng.t_winner,
            )
            rc = _team_eff_rho_flat(carts_c, teams_c, obs_c, partial_cp_weight=lcfg.reward.cart_partial_cp_weight)
            rn = _team_eff_rho_flat(carts_out, teams_out, obs_out, partial_cp_weight=lcfg.reward.cart_partial_cp_weight)
            drho = torch.asinh(lcfg.dyn_drho_scale * (rn - rc))

            sv = eng.state_views()
            gk_all = sv["p_goal_kind"].numpy()
            active_env = ~was_finished
            buf_obs[t] = obs_c
            buf_nav[t] = nav_c
            buf_carts[t] = carts_c
            buf_teams[t] = teams_c
            buf_acc[t] = acc_c
            buf_u_vel[t] = u_vel
            buf_reward[t] = rew
            buf_drho[t] = drho
            buf_wm_in[t] = wm_in
            buf_wm_out[t] = wm_out
            buf_not_done[t] = ~now_finished
            buf_active_env[t] = active_env
            for arm in present_active:
                m_a = active_masks_t[arm]
                buf_amask[arm][t] = m_a & active_env[:, None]
                gk_m = gk_all[m_a.numpy()]
                if gk_m.size > 0:
                    for k_i in range(4):
                        gk_hist[arm][k_i] += int((gk_m == k_i).sum())
            obs_in, nav_in, carts_in, teams_in, acc_in = obs_out, nav_out, carts_out, teams_out, acc_out

        end_finished = eng.t_finished.clone()
        end_scores = eng.t_score.clone()
        if G > 1:
            slot.live_eng._state_u8[:] = slot.fork_eng._state_u8[0::G]

        M = H * E
        obs_flat = buf_obs.reshape(M, N, 84)
        nav_flat = buf_nav.reshape(M, Wn, 16)
        carts_flat = buf_carts.reshape(M, J, 18)
        teams_flat = buf_teams.reshape(M, K, 7)
        acc_flat = buf_acc.reshape(M, N, A_dim)
        u_flat = buf_u_vel.reshape(M, N, A_dim)
        drho_flat = buf_drho.reshape(M, N)
        wm_flat = buf_wm_in.reshape(M, N)
        act_flat = buf_active_env.reshape(M)

        gamma = float(lcfg.gamma)
        lam = float(lcfg.gae_lambda)
        d_scores = end_scores - start_scores
        d_sc_np = d_scores.numpy()
        wall_s = time.perf_counter() - t0
        sim_s = E * H * slot.n_phys_substeps * float(sc.dt)
        world_meta = {"K": K, "J": J, "N": N, "W_nav": Wn, "M": int(slot.spec["tiles"]), "slot": idx}

        batches: dict[str, GraphFlowBatch] = {}
        for arm in present_active:
            cpol = pol_map[arm]
            assert cpol is not None
            out_f = cpol.sample_velocity(obs_in, nav_in, carts_in, teams_in, acc_in, geo=slot.geo, output_scale=0.0)
            val_in = torch.where(buf_wm_in, buf_val_w[arm], buf_val_l[arm])
            v_boot = torch.where(buf_wm_out[H - 1], out_f["value_winnie"], out_f["value_lou"])
            v_boot = torch.where((~end_finished)[:, None], v_boot, torch.zeros_like(v_boot))

            gae_adv = torch.zeros((H, E, N), dtype=torch.float32)
            next_gae = torch.zeros((E, N), dtype=torch.float32)
            for t in range(H - 1, -1, -1):
                still = buf_not_done[t][:, None]
                same_role = buf_wm_in[t] == buf_wm_out[t]
                v_nxt = val_in[t + 1] if t + 1 < H else v_boot
                v_nxt = torch.where(still & same_role, v_nxt, torch.zeros_like(v_nxt))
                g_nxt = torch.where(still & same_role, next_gae, torch.zeros_like(next_gae))
                delta = buf_reward[t] + gamma * v_nxt - val_in[t]
                cur_gae = delta + gamma * lam * g_nxt
                gae_adv[t] = cur_gae
                next_gae = cur_gae
            gae_ret = gae_adv + val_in

            am_f = buf_amask[arm].float()
            am_sum = am_f.sum().clamp_min(1.0)
            g_mean = (gae_adv * am_f).sum() / am_sum
            g_std = (((gae_adv - g_mean).square() * am_f).sum() / am_sum).clamp_min(1e-6).sqrt()
            gae_norm = (gae_adv - g_mean) / g_std

            if G > 1 and float(sc.grpo_weight) > 0.0:
                disc_pows = (gamma ** torch.arange(H, dtype=torch.float32)).view(H, 1, 1)
                m0 = buf_amask[arm][0].float()
                mcnt = m0.sum(dim=-1).clamp_min(1.0)
                ep_ret_ag = (buf_reward * disc_pows).sum(dim=0) + (gamma ** H) * v_boot
                ep_ret_tm = (ep_ret_ag * m0).sum(dim=-1) / mcnt
                grp_ret = ep_ret_tm.view(B, G)
                grp_std = grp_ret.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-3)
                grpo_adv = ((grp_ret - grp_ret.mean(dim=1, keepdim=True)) / grp_std).view(1, E, 1).expand(H, E, N)
                adv = float(sc.grpo_weight) * grpo_adv + float(sc.gae_weight) * gae_norm
            else:
                adv = gae_norm

            opp_wr_obs: dict[str, list[float]] = defaultdict(list)
            my_seats = [ti for ti, s in enumerate(seats) if s == arm]
            opp_seats = [ti for ti, s in enumerate(seats) if s != arm]
            for mi in my_seats:
                for oj in opp_seats:
                    w = np.where(d_sc_np[:, mi] > d_sc_np[:, oj] + 1e-6, 1.0,
                                 np.where(d_sc_np[:, oj] > d_sc_np[:, mi] + 1e-6, 0.0, 0.5))
                    opp_wr_obs[seats[oj]].append(float(w.mean()))
            gk_tot = max(1, int(gk_hist[arm].sum()))
            my_sc_mean = float(d_scores[:, my_seats].mean().item()) if my_seats else 0.0
            opp_sc_mean = float(d_scores[:, opp_seats].mean().item()) if opp_seats else 0.0
            batches[arm] = GraphFlowBatch(
                obs=obs_flat, nav=nav_flat, carts=carts_flat, teams=teams_flat, acc=acc_flat, u_vel=u_flat,
                adv=adv.reshape(M, N), gae_ret=gae_ret.reshape(M, N), obs_drho=drho_flat,
                wm_in=wm_flat, actor_mask=buf_amask[arm].reshape(M, N), active_env=act_flat,
                nav_kern=slot.nav_kern, geo=slot.geo, group_size=G,
                policy_version=int(pvers.get(arm, 0)), world_meta=world_meta,
                opp_wr_obs={a: float(np.mean(vs)) for a, vs in opp_wr_obs.items()},
                stats={
                    "wall_s": round(wall_s, 3),
                    "sim_s": round(sim_s, 1),
                    "sat_frac": round(float(sat_cnt[arm]) / max(1, sat_tot[arm]), 4),
                    "u_rms": round(float(u_sq_sum[arm]) / max(1, u_sq_cnt[arm]), 4),
                    "main_dscore_mean": round(my_sc_mean, 2),
                    "opp_dscore_mean": round(opp_sc_mean, 2),
                    "gk_pct": [round(100.0 * c / gk_tot, 1) for c in gk_hist[arm].tolist()],
                    "seats": seats,
                },
            )

        shared_stats = {
            "wall_s": round(wall_s, 3),
            "sim_s": round(sim_s, 1),
            "speedup_x": round(sim_s / max(wall_s, 1e-6), 1),
            "agent_steps": int(E * H * slot.n_phys_substeps * N),
            "score_outs": score_outs,
            "seats": seats,
            "world": world_meta,
            "present_active": present_active,
        }
        return batches, shared_stats


class GraphFlowTrainer:
    """Metal 4 NAX INT8@INT8 Advantage-Weighted Boltzmann Flow Matching + All-GPU Aurora Optimizer."""

    def __init__(
        self,
        model: GraphFlowPolicy,
        ema_model: GraphFlowPolicy | None,
        cfg: TrainingConfig,
        *,
        ema_fast_model: GraphFlowPolicy | None = None,
        lr_mult: float = 1.0,
        max_cache: int = 12,
    ):
        self.model = model.to(device=torch.device("cpu"))
        self.ema_model = ema_model.to(device=torch.device("cpu")) if ema_model is not None else None
        self.ema_fast_model = ema_fast_model.to(device=torch.device("cpu")) if ema_fast_model is not None else None
        self.cfg = cfg
        self.bank = GraphFlowWeightBank(self.model, with_grads=True)
        self.opt = CompiledGraphFlowOptimizer(self.bank, self.model, cfg.optimizer, lr_mult=lr_mult)
        self.policy_version = 0
        self.max_cache = max(2, int(max_cache))
        self._cache: OrderedDict[tuple[int, int, int, int, int], CompiledGraphFlowTrainer] = OrderedDict()

    def close(self) -> None:
        for tr in self._cache.values():
            tr.close()
        self._cache.clear()
        if getattr(self, "opt", None):
            self.opt.close()
            self.opt = None
        if getattr(self, "bank", None):
            self.bank.close()
            self.bank = None

    def _get_trainer(self, mb_envs: int, N: int, Wn: int, J: int, K: int, nav_kern: torch.Tensor) -> CompiledGraphFlowTrainer:
        key = (mb_envs, N, Wn, J, K)
        tr = self._cache.get(key)
        if tr is None:
            while len(self._cache) >= self.max_cache:
                _, old_tr = self._cache.popitem(last=False)
                old_tr.close()
            tr = CompiledGraphFlowTrainer(
                self.model, E_mb=mb_envs, N=N, W_nav=Wn, J=J, K=K,
                nav_kern=nav_kern, weight_bank=self.bank,
            )
            self._cache[key] = tr
        else:
            self._cache.move_to_end(key)
            tr.set_nav_kern(nav_kern)
        return tr

    def reload_after_rdma_(self) -> None:
        self.opt.reload_master_from_model_()

    @torch.no_grad()
    def update(self, batch: GraphFlowBatch, *, epochs: int = 2) -> dict[str, Any]:
        t0 = time.perf_counter()
        M_tot, N, _ = batch.obs.shape
        Wn = int(batch.nav.shape[1])
        J = int(batch.carts.shape[1])
        K = int(batch.teams.shape[1])
        G = max(1, int(batch.group_size))
        entities_per_env = N + Wn + J + K
        lcfg = self.cfg.losses
        sc = self.cfg.sim_dt

        # Drop all-finished (t, b) groups BEFORE microbatching/shuffling so dead post-game ticks never touch GPU or ESS!
        n_groups_all = M_tot // G
        valid_g_mask = batch.actor_mask[: n_groups_all * G].view(n_groups_all, G * N).any(dim=-1)
        valid_g_ids = torch.nonzero(valid_g_mask, as_tuple=False).squeeze(-1)
        n_groups_tot = int(valid_g_ids.numel())
        if n_groups_tot == 0:
            return {
                "update_wall_s": round(time.perf_counter() - t0, 3),
                "opt_steps": 0, "policy_version": int(self.policy_version),
                "mb_envs": 0, "accum_steps": 0, "live_group_frac": 0.0,
                "loss_total": 0.0, "loss_vel": 0.0, "loss_val": 0.0, "loss_dyn": 0.0,
                "ess_frac": 0.0, "alpha": 0.0, "u_norm": 0.0, "v_norm": 0.0, "grad_norm": 0.0,
            }

        # Solve ESS-targeted Boltzmann weights ONCE vectorized over ALL live groups (faster + lower-variance!)
        live_env_ids = (valid_g_ids[:, None] * G + torch.arange(G)[None, :]).reshape(-1)
        adv_c = batch.adv.clamp(-lcfg.adv_clip, lcfg.adv_clip)
        t0_live = (torch.round(batch.obs[live_env_ids, :, 1].float()) - 1.0).clamp(0.0, float(max(1, K) - 1)).to(torch.int64)
        own_idx = (Wn + t0_live).unsqueeze(-1)
        u_sq_live = batch.u_vel[live_env_ids].float().scatter(-1, own_idx, 0.0).square().mean(dim=-1)
        gf_cfg = self.model.cfg
        w_live, batch_ess, batch_alpha = compute_ess_boltzmann_weights(
            adv_c[live_env_ids],
            batch.actor_mask[live_env_ids],
            u_sq_mean=u_sq_live,
            zero_anchor_lambda=gf_cfg.zero_anchor_lambda,
            ess_target=gf_cfg.ess_target,
            w_min=gf_cfg.flow_w_min,
            w_max=gf_cfg.flow_w_max,
            beta_min=gf_cfg.beta_bounds[0],
            beta_max=gf_cfg.beta_bounds[1],
            group_size=G,
        )
        w_boltz_all = torch.zeros((M_tot, N), dtype=torch.float32)
        w_boltz_all[live_env_ids] = w_live

        target_rows = max(512, int(sc.target_microbatch_rows))
        raw_mb = max(G, min(M_tot, int(round(target_rows / max(1, entities_per_env)))))
        mb_envs = max(G, (raw_mb // G) * G)
        groups_per_mb = max(1, mb_envs // G)
        n_micros = max(1, (n_groups_tot + groups_per_mb - 1) // groups_per_mb)
        target_steps = max(1, int(sc.target_opt_steps_per_epoch))
        accum = max(1, int(round(n_micros / float(target_steps))))

        tr = self._get_trainer(mb_envs, N, Wn, J, K, batch.nav_kern)
        tr.set_geo(batch.geo)
        tr.zero_grads_()
        self.opt.reset_snr_()

        diags: list[dict[str, float]] = []
        opt_steps = 0

        for _ in range(max(1, int(epochs))):
            g_perm = valid_g_ids[torch.randperm(n_groups_tot)]
            mb_starts = list(range(0, n_groups_tot, groups_per_mb))
            in_accum = 0
            for mi, g_start in enumerate(mb_starts):
                if in_accum == 0:
                    rem_micros = len(mb_starts) - mi
                    win_size = min(accum, rem_micros)
                g_ids = g_perm[g_start : g_start + groups_per_mb]
                idx = (g_ids[:, None] * G + torch.arange(G)[None, :]).reshape(-1)
                if idx.numel() < G:
                    continue
                sd = tr.microbatch_fwd_bwd(
                    batch.obs[idx], batch.nav[idx], batch.carts[idx], batch.teams[idx], batch.acc[idx],
                    batch.u_vel[idx], adv_c[idx], batch.gae_ret[idx], batch.obs_drho[idx],
                    batch.actor_mask[idx], batch.wm_in[idx],
                    active_env=batch.active_env[idx],
                    w_boltz=w_boltz_all[idx],
                    ess_frac_in=batch_ess,
                    alpha_in=batch_alpha,
                    value_coef=float(lcfg.critic_weight),
                    dyn_coef=float(lcfg.dyn_weight),
                    grad_scale=1.0 / float(max(1, win_size)),
                    group_size=G,
                )
                in_accum += 1
                if in_accum >= win_size or (mi + 1) >= len(mb_starts):
                    # ALL-GPU <1 ms Aurora+Adam+INT8/INT8^T repack + zero_grads + FP32 Step GradSNR!
                    gn = self.opt.step()
                    in_accum = 0
                    opt_steps += 1
                    sd["grad_norm"] = gn
                    diags.append(sd)

        # Export updated FP32 master -> PyTorch model ONCE at end of update()
        self.opt.export_to_model_(self.model)
        if self.ema_model is not None:
            for (n, _), (_, pe) in zip(self.model.named_parameters(), self.ema_model.named_parameters()):
                pe.copy_((0.992 * pe.float() + 0.008 * self.opt.master[n]).to(pe.dtype))
        if self.ema_fast_model is not None:
            for (n, _), (_, pe) in zip(self.model.named_parameters(), self.ema_fast_model.named_parameters()):
                pe.copy_((0.96 * pe.float() + 0.04 * self.opt.master[n]).to(pe.dtype))

        snr_out, coh_out = self.opt.read_snr_(opt_steps)
        self.policy_version += opt_steps
        def _mean(k: str) -> float:
            return round(float(np.mean([d[k] for d in diags])), 5) if diags else 0.0

        return {
            "update_wall_s": round(time.perf_counter() - t0, 3),
            "opt_steps": opt_steps,
            "policy_version": int(self.policy_version),
            "mb_envs": mb_envs,
            "accum_steps": accum,
            "live_group_frac": round(float(n_groups_tot) / float(max(1, n_groups_all)), 4),
            "loss_total": _mean("loss_total"),
            "loss_vel": _mean("loss_vel"),
            "loss_val": _mean("loss_val"),
            "loss_dyn": _mean("loss_dyn"),
            "ess_frac": _mean("ess_frac"),
            "alpha": _mean("alpha"),
            "u_norm": _mean("u_norm"),
            "v_norm": _mean("v_norm"),
            "grad_norm": _mean("grad_norm"),
            "grad_snr": snr_out["all"],
            "grad_coh": coh_out["all"],
            "grad_snr_by_grp": snr_out,
            "grad_coh_by_grp": coh_out,
        }
