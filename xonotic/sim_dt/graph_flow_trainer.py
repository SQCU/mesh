"""Graph-Native + Exact DPP + 1-NFE Flow Rollout Pool (G > 1 plural counterfactual groups)
and Metal 4 NAX Advantage-Weighted Boltzmann Flow Matching Trainer.
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
from solver.strat.train_config import TrainingConfig

from .dataset_sampler import make_rules_for_profile, sample_roster_skills
from .graph_flow_policy import (
    GraphFlowPolicy,
    clone_graph_flow_model,
    sample_orthogonal_team_bot_noise,
)
from .metal_graphflow import (
    CompiledGraphFlowPolicy,
    CompiledGraphFlowTrainer,
    GraphFlowOptimizer,
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
        self.rules = make_rules_for_profile(float(sc.point_limit_base), self.physics_profile)
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
            c = CompiledGraphFlowPolicy(
                mdl, E=self.E, N=self.N, W_nav=self.W_nav, J=self.J, K=self.K, nav_kern=self.nav_kern,
            )
            self._compiled[arm] = c
        return c

    def sync_variant_(self, arm: str, model: GraphFlowPolicy) -> None:
        self._variants[arm] = model
        c = self._compiled.get(arm)
        if c is not None:
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
        self.self_play_only = bool(self_play_only)
        self.roster_mode = str(roster_mode)
        self.team_corr = float(team_corr)
        self.rng = np.random.default_rng(seed_base + 777)
        self.slots: list[_GraphFlowSlot] = []
        self._cursor = 0
        phys_cycle = ("baseline", "fast", "slow", "volatile") if randomize_physics else ("baseline",)
        w_seed = int(seed_base if world_seed_base is None else world_seed_base)

        for wi in range(max(1, int(num_worlds))):
            spec = sample_world_spec(w_seed + (wi + 1) * 101, cfg.world)
            map_name = f"gf_w{wi}_k{spec['teams']}_j{spec['carts']}_n{spec['players']}"
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
                _GraphFlowSlot(
                    world, spec, cfg, variants,
                    seed_base=seed_base + wi * 10009,
                    group_size=self.G,
                    physics_profile=phys_cycle[wi % len(phys_cycle)],
                    roster_mode=self.roster_mode,
                    tau_rel=tau_rel,
                    n_phys_substeps=n_phys_substeps,
                )
            )

        # Stagger initial match phases once at pool creation
        for wi, slot in enumerate(self.slots):
            stagger = (wi * 12) % 40
            if stagger > 0:
                slot.live_eng.step(self.sc.dt, mode=self.sc.mode, n_steps=stagger)
                fin = slot.live_eng.t_finished
                if bool(fin.any().item()):
                    slot.live_eng.reset(fin)

    def close(self) -> None:
        for s in self.slots:
            s.close()
        self.slots.clear()

    def sync_variant_(self, arm: str, model: GraphFlowPolicy) -> None:
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
        seats.append("default")
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
            seat_arms = self._choose_seats(K, pfsp_pool or ["default", "main_init", "ema", "main"], pfsp_wr or {})

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

        c_main = slot._get_pol("main")
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

            # Draw orthogonalized Strategy(team) + Tactic(bot) base noise across the G branches!
            eps_main = sample_orthogonal_team_bot_noise(
                B, G, N, d_e, p_team, K,
                team_corr=self.team_corr, sigma=sigma_b, generator=slot.noise_gen,
            )

            u_vel = torch.zeros((E, N, A_dim), dtype=torch.float32)
            has_vel = torch.zeros((E, N), dtype=torch.bool)
            main_mask = torch.zeros((E, N), dtype=torch.bool)
            val_w = torch.zeros((E, N), dtype=torch.float32)
            val_l = torch.zeros((E, N), dtype=torch.float32)

            if len(arm_to_teams) == 1 and "main" in arm_to_teams:
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
                    eps_cur = eps_main if arm == "main" else None
                    res = cpol.sample_velocity(obs_c, nav_c, carts_c, teams_c, acc_c, geo=slot.geo, eps=eps_cur)
                    u_vel = torch.where(mask.unsqueeze(-1), res["u_vel"], u_vel)
                    has_vel = has_vel | mask
                    if arm == "main":
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

            # Step physics with velocity steering (n_phys_substeps per strategy emission)
            eng.apply_vel(u_vel, has_vel, tau_rel=slot.tau_rel)
            eng.step(float(sc.dt), mode=str(sc.mode), reset_outcomes=True, n_steps=1, tau_rel=slot.tau_rel)
            for _sub in range(1, slot.n_phys_substeps):
                eng.step(float(sc.dt), mode=str(sc.mode), reset_outcomes=False, n_steps=1, tau_rel=slot.tau_rel)

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


class GraphFlowTrainer:
    """Metal 4 NAX Advantage-Weighted Boltzmann Flow Matching + Winnie/Lou Critic Trainer."""

    def __init__(self, model: GraphFlowPolicy, ema_model: GraphFlowPolicy | None, cfg: TrainingConfig):
        self.model = model.to(device=torch.device("cpu"))
        self.ema_model = ema_model.to(device=torch.device("cpu")) if ema_model is not None else clone_graph_flow_model(self.model)
        self.cfg = cfg
        self.opt = GraphFlowOptimizer(self.model, self.ema_model, cfg.optimizer)
        self.policy_version = 0
        self._cache: dict[tuple[int, int, int, int, int], CompiledGraphFlowTrainer] = {}

    def close(self) -> None:
        for tr in self._cache.values():
            tr.close()
        self._cache.clear()

    def _get_trainer(self, mb_envs: int, N: int, Wn: int, J: int, K: int, nav_kern: torch.Tensor) -> CompiledGraphFlowTrainer:
        key = (mb_envs, N, Wn, J, K)
        tr = self._cache.get(key)
        if tr is None:
            tr = CompiledGraphFlowTrainer(self.model, E_mb=mb_envs, N=N, W_nav=Wn, J=J, K=K, nav_kern=nav_kern)
            self._cache[key] = tr
        else:
            tr.set_nav_kern(nav_kern)
            tr.sync_weights_(self.model)
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

        adv_c = batch.adv.clamp(-lcfg.adv_clip, lcfg.adv_clip)
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
                # Expand live group IDs -> contiguous [B_mb * G] env indices!
                idx = (g_ids[:, None] * G + torch.arange(G)[None, :]).reshape(-1)
                if idx.numel() < G:
                    continue
                sd = tr.microbatch_fwd_bwd(
                    batch.obs[idx], batch.nav[idx], batch.carts[idx], batch.teams[idx], batch.acc[idx],
                    batch.u_vel[idx], adv_c[idx], batch.gae_ret[idx], batch.obs_drho[idx],
                    batch.actor_mask[idx], batch.wm_in[idx],
                    active_env=batch.active_env[idx],
                    value_coef=float(lcfg.critic_weight),
                    dyn_coef=float(lcfg.dyn_weight),
                    grad_scale=1.0 / float(max(1, win_size)),
                    group_size=G,
                )
                in_accum += 1
                if in_accum >= win_size or (mi + 1) >= len(mb_starts):
                    self.model.zero_grad(set_to_none=True)
                    tr.export_grads_to_model_(self.model)
                    gn = self.opt.step()
                    tr.sync_weights_(self.model)
                    tr.zero_grads_()
                    in_accum = 0
                    opt_steps += 1
                    sd["grad_norm"] = gn
                    diags.append(sd)

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
        }
