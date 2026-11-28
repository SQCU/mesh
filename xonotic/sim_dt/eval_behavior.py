#!/usr/bin/env python3
"""Eval-Time Policy Behavior Distribution Analyzer across Domain Randomization Regimes.

Evaluates trained PlanWally checkpoints (main_init, main_mid, main_final) and stock `default`
under controlled domain randomization stress tests — explicitly including:
  1. balanced_baseline       : equal bot mechanics (skill=4), baseline physics
  2. stacked_main_weak       : Team 1 stacked with weak-deathmatch-mechanics bots (-8, 0.52x DPS)
                               vs ace enemies (+20, 2.20x DPS = 4.2x DPS disadvantage!)
  3. stacked_main_strong     : Team 1 stacked with ace bots (+20) vs weak enemies (-8)
  4. within_team_carry_weak  : Team 1 has 25% ace carries (+22) + 75% weak-mechanics bots (-8)
                               -> measures within-team role divergence (weak vs carry!)
  5. fast_carts_high_tempo   : fast cart physics perturbation
  6. volatile_contest_slow   : volatile contest/reverse + slow falloff physics perturbation

Measures full action & outcome distributions (mean, std, p10, p50, p90), discrete cart/rival
targeting distributions, squad concentration/entropy, Projected-Winner vs Loser role shifts,
and weak-mechanics vs ace-mechanics behavioral shifts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "xonotic") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "xonotic"))

from payload.tools.strategy_io_schema import CS, OBS, TS
from sim_dt.dataset_sampler import make_rules_for_profile, sample_roster_skills
from sim_dt.metal_runtime import CompiledPlanWally, MetalDeltaTimePayloadEngine
from sim_dt.world_graph import build_gas_world_graph
from solver.strat.plan_policy import PlanWally
from solver.strat.train_config import CONFIGS_DIR, load_training_config


EVAL_REGIMES: list[dict[str, Any]] = [
    {"name": "balanced_baseline",      "roster": "balanced_fixed",          "physics": "baseline"},
    {"name": "stacked_main_weak",      "roster": "stacked_main_weak",       "physics": "baseline"},
    {"name": "mixed_enemies_main_weak","roster": "mixed_enemies_main_weak", "physics": "baseline"},
    {"name": "stacked_main_strong",    "roster": "stacked_main_strong",     "physics": "baseline"},
    {"name": "within_team_carry_weak", "roster": "within_team_carry_weak",  "physics": "baseline"},
    {"name": "fast_carts_high_tempo",  "roster": "balanced_fixed",          "physics": "fast"},
    {"name": "volatile_contest_slow",  "roster": "balanced_fixed",          "physics": "volatile"},
]


def _dist_stats(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    q10, q50, q90 = np.percentile(x, [10, 50, 90])
    return {
        "mean": round(float(np.mean(x)), 4),
        "std": round(float(np.std(x)), 4),
        "p10": round(float(q10), 4),
        "p50": round(float(q50), 4),
        "p90": round(float(q90), 4),
    }


def _projected_leader_idx(teams_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute projected winner index [E] and unique validity mask [E] from teams [E, K, 7]."""
    sc = teams_np[:, :, TS["SCORE"]]
    held = teams_np[:, :, TS["CHECKPOINTS_HELD"]]
    rate = teams_np[:, :, TS["CHECKPOINT_RATE"]]
    lim = np.maximum(teams_np[:, :, TS["SCORE_LIMIT"]], 1e-6)
    rem = np.maximum(lim - sc, 0.0)
    ttw = np.where(held > 0.0, rem / np.maximum(held * rate, 1e-6), np.where(sc >= lim, 0.0, 1e12))
    min_ttw = np.min(ttw, axis=1, keepdims=True)
    is_min = (ttw <= min_ttw + 1e-6) & (min_ttw < 1e11)
    has_pw = (is_min.sum(axis=1) == 1)
    pw_idx = np.argmin(ttw, axis=1)
    # Fallback to score leader when no unique projected winner yet
    score_leader = np.argmax(sc, axis=1)
    eff_leader = np.where(has_pw, pw_idx, score_leader)
    return eff_leader, has_pw


@torch.no_grad()
def evaluate_arm_on_regime(
    arm_name: str,
    model: PlanWally | None,
    opp_model: PlanWally | None,
    world: Any,
    regime: dict[str, Any],
    *,
    cfg: Any,
    E: int,
    H: int,
    seed: int,
) -> dict[str, Any]:
    K = int(world.teams)
    J = int(world.carts)
    N = int(world.players_default)
    rules = make_rules_for_profile(float(cfg.sim_dt.point_limit_base), regime["physics"])
    eng = MetalDeltaTimePayloadEngine(world, num_envs=E, num_players=N, rules=rules, base_seed=seed)
    rng = np.random.default_rng(seed + 999)
    skills = sample_roster_skills(rng, E, N, K, regime["roster"])
    eng.set_skills(skills)
    geo = {"map_center": world.map_center.clone(), "map_radius": world.map_radius.clone()}

    c_main = (
        CompiledPlanWally(
            model, E=E, N=N, J=J, K=K, pack_cfg=cfg.action_pack,
            do_pack=True, stochastic=True, write_sample=True, sample_seed=seed + 17,
        )
        if model is not None
        else None
    )
    c_opp = (
        CompiledPlanWally(
            opp_model, E=E, N=N, J=J, K=K, pack_cfg=cfg.action_pack,
            do_pack=True, stochastic=True, write_sample=True, sample_seed=seed + 71,
        )
        if opp_model is not None
        else None
    )

    p_team_1d = (np.arange(N, dtype=np.int32) % K) + 1
    main_idx = np.nonzero(p_team_1d == 1)[0]
    n_main = len(main_idx)
    main_mask_t = torch.from_numpy(p_team_1d == 1).view(1, N).expand(E, N)
    opp_mask_t = ~main_mask_t

    # Per-team mean skill [E, K] to measure targeting strongest vs weakest enemy team
    team_skill_mean = np.zeros((E, K), dtype=np.float32)
    for k in range(K):
        team_skill_mean[:, k] = skills[:, p_team_1d == (k + 1)].mean(axis=1)
    enemy_skills = team_skill_mean[:, 1:]  # teams 2..K (indices 1..K-1)
    strongest_enemy_t = np.argmax(enemy_skills, axis=1) + 2  # 1-based team ID
    weakest_enemy_t = np.argmin(enemy_skills, axis=1) + 2

    # Behavioral accumulators for Team 1 (main) across [H, E, n_main]
    cart_own = []; cart_neut = []; cart_lead = []; cart_other = []
    squad_max_frac = []; squad_norm_ent = []
    rival_focus_lead = []; rival_focus_strongest = []; rival_focus_weakest = []
    act_commit = []; act_ds = []; act_posture = []; act_push_b = []; act_cont_b = []; act_roam = []
    role_is_pw = []
    bot_is_weak = []
    bot_is_ace = []

    tot_dmg_dealt = np.zeros((E,), dtype=np.float64)
    tot_dmg_taken = np.zeros((E,), dtype=np.float64)
    tot_kills = np.zeros((E,), dtype=np.float64)
    tot_deaths = np.zeros((E,), dtype=np.float64)
    tot_push = np.zeros((E,), dtype=np.float64)
    tot_contest = np.zeros((E,), dtype=np.float64)
    tot_alive_ticks = np.zeros((E,), dtype=np.float64)

    try:
        for _ in range(H):
            obs, carts, teams, _ = eng.observe_tensors()
            obs_c = obs.clone(); carts_c = carts.clone(); teams_c = teams.clone()
            plan = torch.zeros((E, N, 16), dtype=torch.float32)
            hp = torch.zeros((E, N), dtype=torch.bool)

            res_m = None
            if c_main is not None:
                res_m = c_main.forward_sample(obs_c, carts_c, teams_c, geo=geo)
                plan = torch.where(main_mask_t.unsqueeze(-1), res_m["plan"], plan)
                hp = hp | main_mask_t
            if c_opp is not None:
                res_o = c_opp.forward_sample(obs_c, carts_c, teams_c, geo=geo)
                plan = torch.where(opp_mask_t.unsqueeze(-1), res_o["plan"], plan)
                hp = hp | opp_mask_t

            eng.apply_plan(plan, hp)
            eng.step(float(cfg.sim_dt.dt), mode=str(cfg.sim_dt.mode))

            # Collect realized outcome telemetry for Team 1
            sv = eng.state_views()
            tot_dmg_dealt += sv["p_dmg_dealt"][:, main_idx].numpy().sum(axis=1)
            tot_dmg_taken += sv["p_dmg_taken"][:, main_idx].numpy().sum(axis=1)
            tot_kills += sv["p_kills"][:, main_idx].numpy().sum(axis=1)
            tot_deaths += sv["p_deaths"][:, main_idx].numpy().sum(axis=1)
            tot_push += sv["p_cart_push"][:, main_idx].numpy().sum(axis=1)
            tot_contest += sv["p_cart_contest"][:, main_idx].numpy().sum(axis=1)
            tot_alive_ticks += sv["p_alive"][:, main_idx].numpy().mean(axis=1)

            if res_m is not None:
                carts_np = carts_c.float().numpy()
                teams_np = teams_c.float().numpy()
                eff_lead_0, has_pw = _projected_leader_idx(teams_np)  # 0..K-1
                eff_lead_1 = eff_lead_0 + 1  # 1..K
                is_pw_t1 = has_pw & (eff_lead_0 == 0)  # [E]

                cp = res_m["cart_pick"][:, main_idx].numpy().astype(np.int64)  # [E, n_main]
                rp = res_m["rival_pick"][:, main_idx].numpy().astype(np.int64)  # [E, n_main]
                pl = res_m["plan"][:, main_idx, :].numpy()                      # [E, n_main, 16]

                # Cart ownership classification of picked cart
                c_ctrl = np.rint(carts_np[:, :, CS["CONTROL_TEAM"]]).astype(np.int64)  # [E, J]
                picked_owner = np.take_along_axis(c_ctrl, np.clip(cp, 0, J - 1), axis=1)  # [E, n_main]
                c_own = (picked_owner == 1)
                c_neu = (picked_owner == 0)
                c_ld = (picked_owner == eff_lead_1[:, None]) & (eff_lead_1[:, None] != 1)
                c_oth = (~c_own) & (~c_neu) & (~c_ld)

                cart_own.append(c_own)
                cart_neut.append(c_neu)
                cart_lead.append(c_ld)
                cart_other.append(c_oth)

                # Squad coordination / concentration across carts per env
                counts = np.zeros((E, J), dtype=np.float32)
                for mi in range(n_main):
                    counts[np.arange(E), np.clip(cp[:, mi], 0, J - 1)] += 1.0
                probs = counts / max(1.0, float(n_main))
                squad_max_frac.append(probs.max(axis=1))
                ent = -np.sum(np.where(probs > 0, probs * np.log(np.maximum(probs, 1e-12)), 0.0), axis=1)
                squad_norm_ent.append(ent / max(np.log(float(min(J, n_main))), 1e-6))

                # Rival targeting classification
                foc_t1 = np.rint(pl[:, :, 7]).astype(np.int64)  # 1-based focus team
                rival_focus_lead.append(foc_t1 == eff_lead_1[:, None])
                rival_focus_strongest.append(foc_t1 == strongest_enemy_t[:, None])
                rival_focus_weakest.append(foc_t1 == weakest_enemy_t[:, None])

                # Continuous action axes
                act_commit.append(pl[:, :, 2] - pl[:, :, 3])
                act_ds.append(pl[:, :, 10])
                act_posture.append(pl[:, :, 15])
                act_push_b.append(pl[:, :, 5])
                act_cont_b.append(pl[:, :, 6])
                act_roam.append(0.5 * (pl[:, :, 8] + pl[:, :, 9]))

                role_is_pw.append(np.broadcast_to(is_pw_t1[:, None], (E, n_main)).copy())
                sk_m = skills[:, main_idx]
                bot_is_weak.append(sk_m < 0.0)
                bot_is_ace.append(sk_m > 10.0)

        end_scores = eng.t_score.numpy()       # [E, K]
        end_fin = eng.t_finished.numpy()       # [E]
        end_win = eng.t_winner.numpy()         # [E]
    finally:
        if c_main is not None:
            c_main.close()
        if c_opp is not None:
            c_opp.close()
        eng.close()

    # Outcomes for Team 1
    t1_sc = end_scores[:, 0]
    opp_sc_mean = end_scores[:, 1:].mean(axis=1)
    opp_sc_max = end_scores[:, 1:].max(axis=1)
    rank1_win = (t1_sc > opp_sc_max + 1e-6).astype(np.float64) + 0.5 * (np.abs(t1_sc - opp_sc_max) <= 1e-6)
    pw_beats = np.mean(
        np.where(t1_sc[:, None] > end_scores[:, 1:] + 1e-6, 1.0,
                 np.where(end_scores[:, 1:] > t1_sc[:, None] + 1e-6, 0.0, 0.5)),
        axis=1,
    )

    out: dict[str, Any] = {
        "arm": arm_name,
        "regime": regime["name"],
        "outcomes": {
            "first_place_rate": round(float(rank1_win.mean()), 4),
            "pairwise_win_rate": round(float(pw_beats.mean()), 4),
            "main_score_mean": round(float(t1_sc.mean()), 2),
            "opp_score_mean": round(float(opp_sc_mean.mean()), 2),
            "score_margin": round(float((t1_sc - opp_sc_mean).mean()), 2),
            "score_out_frac": round(float(end_fin.mean()), 4),
            "kd_ratio": round(float(tot_kills.sum() / max(tot_deaths.sum(), 1.0)), 3),
            "dmg_dealt_per_match": round(float(tot_dmg_dealt.mean()), 1),
            "dmg_taken_per_match": round(float(tot_dmg_taken.mean()), 1),
            "cart_push_per_match": round(float(tot_push.mean()), 2),
            "cart_contest_s_per_match": round(float(tot_contest.mean()), 2),
            "alive_frac": round(float((tot_alive_ticks / H).mean()), 4),
        },
    }

    if not cart_own:
        return out

    def cat(lst):
        return np.concatenate([a.reshape(-1) for a in lst], axis=0)

    c_own_a = cat(cart_own); c_neu_a = cat(cart_neut); c_ld_a = cat(cart_lead); c_oth_a = cat(cart_other)
    rf_ld_a = cat(rival_focus_lead); rf_st_a = cat(rival_focus_strongest); rf_wk_a = cat(rival_focus_weakest)
    comm_a = cat(act_commit); ds_a = cat(act_ds); post_a = cat(act_posture)
    pb_a = cat(act_push_b); cb_a = cat(act_cont_b); rm_a = cat(act_roam)
    pw_a = cat(role_is_pw); wk_a = cat(bot_is_weak); ace_a = cat(bot_is_ace)

    def _slice_summary(mask: np.ndarray) -> dict[str, Any]:
        if int(mask.sum()) == 0:
            return {"n": 0}
        return {
            "n": int(mask.sum()),
            "p_cart_own": round(float(c_own_a[mask].mean()), 4),
            "p_cart_neutral": round(float(c_neu_a[mask].mean()), 4),
            "p_cart_leader": round(float(c_ld_a[mask].mean()), 4),
            "p_cart_other": round(float(c_oth_a[mask].mean()), 4),
            "p_focus_leader": round(float(rf_ld_a[mask].mean()), 4),
            "p_focus_strongest_mech": round(float(rf_st_a[mask].mean()), 4),
            "p_focus_weakest_mech": round(float(rf_wk_a[mask].mean()), 4),
            "p_vanguard_gt120": round(float((ds_a[mask] > 120.0).mean()), 4),
            "p_rearguard_lt_neg120": round(float((ds_a[mask] < -120.0).mean()), 4),
            "cart_commit": _dist_stats(comm_a[mask]),
            "cart_ds": _dist_stats(ds_a[mask]),
            "posture": _dist_stats(post_a[mask]),
            "push_bias": _dist_stats(pb_a[mask]),
            "contest_bias": _dist_stats(cb_a[mask]),
            "roam_bias": _dist_stats(rm_a[mask]),
        }

    all_m = np.ones_like(pw_a, dtype=bool)
    out["behavior_all"] = _slice_summary(all_m)
    out["coordination"] = {
        "squad_max_cart_frac": _dist_stats(cat(squad_max_frac)),
        "squad_cart_norm_entropy": _dist_stats(cat(squad_norm_ent)),
    }
    out["by_role"] = {
        "projected_winner": _slice_summary(pw_a),
        "loser": _slice_summary(~pw_a),
    }
    if int(wk_a.sum()) > 0 and int(ace_a.sum()) > 0:
        out["within_team_by_mechanics"] = {
            "weak_bots_lt0": _slice_summary(wk_a),
            "ace_bots_gt10": _slice_summary(ace_a),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval-time policy behavior distribution analyzer across DR regimes")
    ap.add_argument("--ckpt-dir", type=str, required=True)
    ap.add_argument("--config", type=str, default=str(CONFIGS_DIR / "sim_dt_grpo_pfsp.json"))
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--teams", type=int, default=6)
    ap.add_argument("--carts", type=int, default=20)
    ap.add_argument("--players", type=int, default=48)
    ap.add_argument("--tiles", type=int, default=4)
    ap.add_argument("--opp", choices=("default", "main_init", "self"), default="default")
    args = ap.parse_args()

    cfg = load_training_config(args.config)
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    world = build_gas_world_graph(
        map_name=f"eval_dr_k{args.teams}_j{args.carts}_n{args.players}",
        teams=args.teams, carts=args.carts, players=args.players, tiles=args.tiles,
        seed=8101, device=torch.device("cpu"),
    )

    models: dict[str, PlanWally | None] = {"default": None}
    for name in ("main_init", "main_mid", "main_final"):
        p = ckpt_dir / f"{name}.pt"
        if p.exists():
            m = PlanWally(model_cfg=cfg.model, norm_cfg=cfg.norm).to(device=torch.device("cpu"), dtype=torch.bfloat16)
            m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
            m.eval()
            models[name] = m

    t0 = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}
    for ri, reg in enumerate(EVAL_REGIMES):
        results[reg["name"]] = {}
        for ai, (arm_name, mdl) in enumerate(models.items()):
            opp_m = None if args.opp == "default" else (models.get("main_init") if args.opp == "main_init" else mdl)
            res = evaluate_arm_on_regime(
                arm_name, mdl, opp_m, world, reg,
                cfg=cfg, E=args.envs, H=args.horizon, seed=90000 + ri * 1000 + ai * 37,
            )
            results[reg["name"]][arm_name] = res

    out_path = ckpt_dir / f"behavior_eval_vs_{args.opp}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[eval-behavior] Done in {time.perf_counter()-t0:.2f}s (opp={args.opp}, E={args.envs}, H={args.horizon}) -> {out_path}\n", flush=True)

    # Print compact readable tables
    print("=== 1. OUTCOME COMPETENCE ACROSS DOMAIN RANDOMIZATION REGIMES ===")
    print(f"{'Regime':24s} | {'Arm':11s} | {'PW-WR':6s} {'1st%':6s} {'Margin':7s} {'K/D':5s} {'Push':6s} {'ContS':6s} {'Alive':5s}")
    print("-" * 92)
    for reg in EVAL_REGIMES:
        rn = reg["name"]
        for arm_name in models:
            o = results[rn][arm_name]["outcomes"]
            print(
                f"{rn:24s} | {arm_name:11s} | {o['pairwise_win_rate']:6.3f} {o['first_place_rate']:6.3f} "
                f"{o['score_margin']:+7.1f} {o['kd_ratio']:5.2f} {o['cart_push_per_match']:6.1f} "
                f"{o['cart_contest_s_per_match']:6.1f} {o['alive_frac']:5.3f}"
            )

    if "main_final" in models:
        print("\n=== 2. POLICY BEHAVIOR DISTRIBUTIONS (main_init vs main_final) ACROSS DR REGIMES ===")
        print(f"{'Regime':24s} | {'Arm':10s} | {'OwnC':5s} {'NeuC':5s} {'LeadC':5s} {'FocLd':5s} {'FocSt':5s} {'FocWk':5s} {'SqMax':5s} {'SqEnt':5s} | {'Commit(p50±sd)':14s} {'CartDS(p10/50/90)':18s} {'Post(p50±sd)':12s}")
        print("-" * 146)
        arms_show = [a for a in ("main_init", "main_final") if a in models]
        for reg in EVAL_REGIMES:
            rn = reg["name"]
            for an in arms_show:
                r = results[rn][an]
                b = r["behavior_all"]
                co = r["coordination"]
                ds = b["cart_ds"]
                print(
                    f"{rn:24s} | {an:10s} | {b['p_cart_own']:5.3f} {b['p_cart_neutral']:5.3f} {b['p_cart_leader']:5.3f} "
                    f"{b['p_focus_leader']:5.3f} {b['p_focus_strongest_mech']:5.3f} {b['p_focus_weakest_mech']:5.3f} "
                    f"{co['squad_max_cart_frac']['mean']:5.3f} {co['squad_cart_norm_entropy']['mean']:5.3f} | "
                    f"{b['cart_commit']['p50']:+6.2f}±{b['cart_commit']['std']:4.2f}   "
                    f"{ds['p10']:+5.0f}/{ds['p50']:+5.0f}/{ds['p90']:+5.0f} "
                    f"{b['posture']['p50']:+5.2f}±{b['posture']['std']:4.2f}"
                )

        print("\n=== 3. ROLE SHIFT (PROJECTED WINNER vs LOSER) & WITHIN-TEAM MECHANICS SHIFT (WEAK vs ACE) ===")
        for rn in ("balanced_baseline", "stacked_main_weak", "mixed_enemies_main_weak", "within_team_carry_weak"):
            r = results[rn]["main_final"]
            for label, sl in [("PW(Winnie)", r["by_role"]["projected_winner"]), ("Loser(Lou)", r["by_role"]["loser"])]:
                if sl.get("n", 0) == 0:
                    continue
                print(
                    f"  [{rn:22s}] {label:11s} (n={sl['n']:6d}): OwnC={sl['p_cart_own']:.3f} NeuC={sl['p_cart_neutral']:.3f} "
                    f"LeadC={sl['p_cart_leader']:.3f} FocLd={sl['p_focus_leader']:.3f} "
                    f"Commit={sl['cart_commit']['mean']:+.2f} CartDS={sl['cart_ds']['mean']:+.1f} "
                    f"(Van={sl['p_vanguard_gt120']:.2f}/Rear={sl['p_rearguard_lt_neg120']:.2f}) Post={sl['posture']['mean']:+.2f}"
                )
            if "within_team_by_mechanics" in r:
                for label, sl in [
                    ("Weak(sk<0)", r["within_team_by_mechanics"]["weak_bots_lt0"]),
                    ("Ace(sk>10)", r["within_team_by_mechanics"]["ace_bots_gt10"]),
                ]:
                    print(
                        f"  [{rn:22s}] {label:11s} (n={sl['n']:6d}): OwnC={sl['p_cart_own']:.3f} NeuC={sl['p_cart_neutral']:.3f} "
                        f"LeadC={sl['p_cart_leader']:.3f} FocLd={sl['p_focus_leader']:.3f} "
                        f"Commit={sl['cart_commit']['mean']:+.2f} CartDS={sl['cart_ds']['mean']:+.1f} "
                        f"(Van={sl['p_vanguard_gt120']:.2f}/Rear={sl['p_rearguard_lt_neg120']:.2f}) Post={sl['posture']['mean']:+.2f}"
                    )


if __name__ == "__main__":
    main()
