from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .grad_snr import pearson_correlation, spearman_correlation


def _safe_mean(values) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else None


def load_telemetry(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_behavioral_motifs(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract outer behavioral motifs from a live Xonotic telemetry trace.

    Measures:
      M1. Cart progression & objective competence per team / arm
      M2. Reverse-induction anti-leader coalition & thermograph alignment motifs
      M3. Role-conditional behavioral asymmetry (projected winner vs loser residual shift)
      M4. Training / value / dynamics / IS-health trajectory summaries
    """
    if not records:
        return {"error": "empty telemetry", "n_records": 0}

    first = records[0]
    last = records[-1]
    k = int(first.get("teams", 0))
    assigned_arms = list(first.get("team_policy_arms") or [])
    if not assigned_arms:
        assigned_arms = [str(first.get("policy_arm", "default"))]
    team_arm = {t: assigned_arms[t % len(assigned_arms)] for t in range(k)}

    cart_init_pos = None
    cart_max_pos = defaultdict(float)
    cart_last_pos = {}
    team_scores_series = defaultdict(list)
    team_rho_series = defaultdict(list)
    team_margin_series = defaultdict(list)
    team_checkpoints_max = defaultdict(int)

    # Reverse-induction / coalition motif accumulators
    pw_steps = defaultdict(int)
    pw_survived = defaultdict(int)
    pw_transitions = defaultdict(int)
    loser_steps = defaultdict(int)
    loser_rank_ups = defaultdict(int)

    # Combat / event targeting relative to projected winner (PW)
    loser_dmg_total = defaultdict(float)
    loser_dmg_to_pw = defaultdict(float)
    loser_kills_total = defaultdict(int)
    loser_kills_to_pw = defaultdict(int)
    winner_dmg_total = defaultdict(float)
    winner_kills_total = defaultdict(int)

    # Spatial proximity of team bots to projected leader's carts vs own/other carts & thermograph weights
    loser_prox_pw_cart = defaultdict(list)
    loser_prox_max_deny_cart = defaultdict(list)
    winner_prox_own_cart = defaultdict(list)
    team_cart_push_events = defaultdict(int)

    # Role-conditional policy residual / velocity energy (winner vs loser)
    arm_residual_winner = defaultdict(list)
    arm_residual_loser = defaultdict(list)

    # Policy update / IS / gradient tracking
    update_summaries = []
    prev_pw = None
    prev_ranks = None
    total_view_faults = 0

    for rec in records:
        gv = rec.get("game_value") or {}
        pw = gv.get("projected_role")
        scores = gv.get("scores") or []
        held = gv.get("checkpoints_held") or []
        rhos = gv.get("finish_rates") or []
        margins = gv.get("winning_margins") or []
        deny_temp = gv.get("lane_thermograph_deny") or []
        role_ranks = gv.get("role_ranks") or {}

        for t in range(min(k, len(scores))):
            team_scores_series[t].append(float(scores[t]))
        for t in range(min(k, len(rhos))):
            team_rho_series[t].append(float(rhos[t]))
        for t in range(min(k, len(margins))):
            team_margin_series[t].append(float(margins[t]))
        for t in range(min(k, len(held))):
            team_checkpoints_max[t] = max(team_checkpoints_max[t], int(held[t]))

        if prev_pw is not None and pw is not None:
            pw_transitions[prev_pw] += 1
            if pw == prev_pw:
                pw_survived[prev_pw] += 1
        if pw is not None:
            pw_steps[int(pw)] += 1
            for t in range(k):
                if t != int(pw):
                    loser_steps[t] += 1
        if prev_ranks is not None and role_ranks:
            for t in range(k):
                r_prev = prev_ranks.get(str(t))
                r_curr = role_ranks.get(str(t))
                if r_prev is not None and r_curr is not None and int(r_curr) > int(r_prev):
                    loser_rank_ups[t] += 1
        prev_pw = pw
        prev_ranks = role_ranks

        carts = rec.get("carts") or []
        if cart_init_pos is None and carts:
            cart_init_pos = [float(c.get("path_position", 0.0)) for c in carts]
        pw_cart_indices = []
        max_deny_idx = int(np.argmax(deny_temp)) if deny_temp and max(deny_temp) > 0.0 else None
        for idx, c in enumerate(carts):
            pos = float(c.get("path_position", 0.0))
            cart_max_pos[idx] = max(cart_max_pos[idx], pos)
            cart_last_pos[idx] = pos
            ctrl = int(round(float(c.get("control_team", 0.0)))) - 1
            if pw is not None and ctrl == int(pw):
                pw_cart_indices.append(idx)
            spd = float(c.get("speed", 0.0))
            if spd > 1.0 and 0 <= ctrl < k:
                team_cart_push_events[ctrl] += 1

        # Player assignments & spatial proximity
        assignments = rec.get("assignments") or []
        edict_team = {}
        for a in assignments:
            team_1 = int(a.get("team", 0))
            t = team_1 - 1
            if not (0 <= t < k):
                continue
            if "edict" in a:
                edict_team[int(a["edict"])] = t
            arm = str(a.get("arm", team_arm.get(t, "default")))
            res_norm = a.get("residual_norm")
            if res_norm is None and isinstance(a.get("residual"), list):
                arr = np.asarray(a["residual"], dtype=np.float64)
                res_norm = float(np.linalg.norm(arr))
            if res_norm is not None:
                if pw is not None and t == int(pw):
                    arm_residual_winner[arm].append(float(res_norm))
                else:
                    arm_residual_loser[arm].append(float(res_norm))

            # Optional spatial proximity if coordinates are present
            origin = a.get("origin") or a.get("pos")
            if origin is not None and len(origin) >= 2 and carts:
                ox, oy = float(origin[0]), float(origin[1])
                dists = []
                for c in carts:
                    cx, cy = float(c.get("x", c.get("origin_x", 0.0))), float(c.get("y", c.get("origin_y", 0.0)))
                    dists.append(math.hypot(ox - cx, oy - cy))
                if dists:
                    nearest = int(np.argmin(dists))
                    if pw is not None and t != int(pw):
                        if pw_cart_indices:
                            loser_prox_pw_cart[t].append(1.0 if nearest in pw_cart_indices else 0.0)
                        if max_deny_idx is not None:
                            loser_prox_max_deny_cart[t].append(1.0 if nearest == max_deny_idx else 0.0)
                    elif pw is not None and t == int(pw) and pw_cart_indices:
                        winner_prox_own_cart[t].append(1.0 if nearest in pw_cart_indices else 0.0)

        # Combat events (damage & kills) relative to PW
        for ev in rec.get("realized_events") or []:
            kind = ev.get("kind")
            actor_t = int(ev.get("actor_team", 0)) - 1
            subj = int(ev.get("subject", -1))
            subj_t = edict_team.get(subj, -1)
            val = float(ev.get("value", 0.0))
            if not (0 <= actor_t < k):
                continue
            if kind == "damage":
                if pw is not None and actor_t != int(pw):
                    loser_dmg_total[actor_t] += val
                    if subj_t == int(pw):
                        loser_dmg_to_pw[actor_t] += val
                elif pw is not None and actor_t == int(pw):
                    winner_dmg_total[actor_t] += val
            elif kind == "kill":
                if pw is not None and actor_t != int(pw):
                    loser_kills_total[actor_t] += 1
                    if subj_t == int(pw):
                        loser_kills_to_pw[actor_t] += 1
                elif pw is not None and actor_t == int(pw):
                    winner_kills_total[actor_t] += 1

        learning = rec.get("learning") or {}
        for arm_state in learning.values():
            total_view_faults = max(total_view_faults, int(arm_state.get("view_faults", 0)))

        for arm, pu in (rec.get("policy_updates") or {}).items():
            if isinstance(pu, dict) and pu:
                ir = pu.get("importance_ratio") or {}
                gs = pu.get("grad_snr") or {}
                update_summaries.append(
                    {
                        "resp_id": rec.get("resp_id"),
                        "arm": arm,
                        "updates": pu.get("updates"),
                        "loss": pu.get("loss"),
                        "loss_pg": pu.get("loss_pg"),
                        "loss_w": pu.get("loss_w"),
                        "loss_l": pu.get("loss_l"),
                        "loss_dynamics": pu.get("loss_dynamics"),
                        "gradient_norm": pu.get("gradient_norm"),
                        "is_mean": ir.get("mean"),
                        "is_floor_fraction": ir.get("numerical_floor_fraction"),
                        "is_clip_fraction": ir.get("clipped_fraction"),
                        "grad_snr_last_cosine": (gs.get("last") or {}).get("cosine"),
                        "grad_snr_window_snr": (gs.get("window") or {}).get("snr"),
                        "grad_snr_window_step_efficiency": (gs.get("window") or {}).get("step_efficiency"),
                    }
                )

    t0 = float(first.get("t", 0.0))
    t1 = float(last.get("t", t0))
    duration_s = max(1e-6, t1 - t0)

    per_team = {}
    for t in range(k):
        sc = team_scores_series[t]
        score_start = sc[0] if sc else 0.0
        score_end = sc[-1] if sc else 0.0
        anti_leader_dmg_share = (
            loser_dmg_to_pw[t] / loser_dmg_total[t] if loser_dmg_total[t] > 0.0 else None
        )
        anti_leader_kill_share = (
            loser_kills_to_pw[t] / loser_kills_total[t] if loser_kills_total[t] > 0 else None
        )
        per_team[t] = {
            "arm": team_arm.get(t, "default"),
            "score_start": score_start,
            "score_end": score_end,
            "score_delta": score_end - score_start,
            "score_rate": (score_end - score_start) / duration_s,
            "max_checkpoints_held": team_checkpoints_max[t],
            "mean_finish_rate": _safe_mean(team_rho_series[t]),
            "mean_winning_margin": _safe_mean(team_margin_series[t]),
            "pw_steps": pw_steps[t],
            "pw_survival_rate": pw_survived[t] / pw_transitions[t] if pw_transitions[t] > 0 else None,
            "loser_steps": loser_steps[t],
            "loser_rank_ups": loser_rank_ups[t],
            "loser_rank_up_rate": loser_rank_ups[t] / loser_steps[t] if loser_steps[t] > 0 else None,
            "cart_push_ticks": team_cart_push_events[t],
            "anti_leader_damage_share": anti_leader_dmg_share,
            "anti_leader_kill_share": anti_leader_kill_share,
            "loser_dmg_total": loser_dmg_total[t],
            "winner_dmg_total": winner_dmg_total[t],
            "loser_prox_pw_cart": _safe_mean(loser_prox_pw_cart[t]),
            "loser_prox_max_deny_cart": _safe_mean(loser_prox_max_deny_cart[t]),
            "winner_prox_own_cart": _safe_mean(winner_prox_own_cart[t]),
        }

    # Aggregate per arm
    by_arm: dict[str, dict[str, Any]] = {}
    arm_teams = defaultdict(list)
    for t, info in per_team.items():
        arm_teams[info["arm"]].append(info)
    for arm, entries in arm_teams.items():
        w_res = _safe_mean(arm_residual_winner[arm])
        l_res = _safe_mean(arm_residual_loser[arm])
        by_arm[arm] = {
            "teams": len(entries),
            "mean_score_end": _safe_mean([e["score_end"] for e in entries]),
            "mean_score_rate": _safe_mean([e["score_rate"] for e in entries]),
            "mean_finish_rate": _safe_mean([e["mean_finish_rate"] for e in entries]),
            "mean_winning_margin": _safe_mean([e["mean_winning_margin"] for e in entries]),
            "total_checkpoints_max": sum(e["max_checkpoints_held"] for e in entries),
            "total_cart_push_ticks": sum(e["cart_push_ticks"] for e in entries),
            "total_pw_steps": sum(e["pw_steps"] for e in entries),
            "pw_survival_rate": _safe_mean([e["pw_survival_rate"] for e in entries]),
            "loser_rank_up_rate": _safe_mean([e["loser_rank_up_rate"] for e in entries]),
            "anti_leader_damage_share": _safe_mean([e["anti_leader_damage_share"] for e in entries]),
            "anti_leader_kill_share": _safe_mean([e["anti_leader_kill_share"] for e in entries]),
            "residual_norm_winner": w_res,
            "residual_norm_loser": l_res,
            "role_residual_asymmetry": (
                abs(w_res - l_res) / max(1e-12, 0.5 * (w_res + l_res))
                if w_res is not None and l_res is not None and (w_res + l_res) > 0
                else None
            ),
        }

    return {
        "n_records": len(records),
        "duration_s": duration_s,
        "teams": k,
        "view_faults": total_view_faults,
        "max_cart_pos": max(cart_max_pos.values(), default=0.0),
        "cart_max_pos": {str(idx): val for idx, val in sorted(cart_max_pos.items())},
        "cart_last_pos": {str(idx): val for idx, val in sorted(cart_last_pos.items())},
        "per_team": per_team,
        "by_arm": by_arm,
        "n_policy_updates": len(update_summaries),
        "last_update": update_summaries[-1] if update_summaries else None,
        "mean_is_floor_fraction": _safe_mean([u["is_floor_fraction"] for u in update_summaries]),
        "mean_is_ratio": _safe_mean([u["is_mean"] for u in update_summaries]),
        "mean_grad_norm": _safe_mean([u["gradient_norm"] for u in update_summaries]),
        "mean_grad_snr_cosine": _safe_mean([u["grad_snr_last_cosine"] for u in update_summaries]),
    }


def paired_seed_divergence(records_a: list[dict[str, Any]], records_b: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare two live Xonotic runs launched with the identical sv_random_seed
    (e.g. default vs matrix_fusion, or pi_0 vs pi_T) to test causal perturbation coupling
    and Policy Improvement Theorem outer return deltas."""
    n = min(len(records_a), len(records_b))
    if n == 0:
        return {"error": "empty trace", "aligned_steps": 0}

    cart_pos_l1 = []
    score_l1 = []
    speed_l1 = []
    for i in range(n):
        ca = records_a[i].get("carts") or []
        cb = records_b[i].get("carts") or []
        m = min(len(ca), len(cb))
        if m > 0:
            pa = np.asarray([float(c.get("path_position", 0.0)) for c in ca[:m]])
            pb = np.asarray([float(c.get("path_position", 0.0)) for c in cb[:m]])
            sa = np.asarray([float(c.get("speed", 0.0)) for c in ca[:m]])
            sb = np.asarray([float(c.get("speed", 0.0)) for c in cb[:m]])
            cart_pos_l1.append(float(np.abs(pb - pa).mean()))
            speed_l1.append(float(np.abs(sb - sa).mean()))
        ga = (records_a[i].get("game_value") or {}).get("scores") or []
        gb = (records_b[i].get("game_value") or {}).get("scores") or []
        ms = min(len(ga), len(gb))
        if ms > 0:
            score_l1.append(float(np.abs(np.asarray(gb[:ms]) - np.asarray(ga[:ms])).mean()))

    motifs_a = extract_behavioral_motifs(records_a)
    motifs_b = extract_behavioral_motifs(records_b)
    return {
        "aligned_steps": n,
        "mean_cart_pos_l1_divergence": _safe_mean(cart_pos_l1),
        "max_cart_pos_l1_divergence": max(cart_pos_l1, default=0.0),
        "mean_cart_speed_l1_divergence": _safe_mean(speed_l1),
        "mean_score_l1_divergence": _safe_mean(score_l1),
        "max_score_l1_divergence": max(score_l1, default=0.0),
        "causally_coupled": bool(max(cart_pos_l1, default=0.0) > 1e-3 or max(score_l1, default=0.0) > 1e-3),
        "motifs_a": motifs_a,
        "motifs_b": motifs_b,
    }


def evaluate_heldout_critics_and_dynamics(learner, heldout_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate held-out critic calibration (MSE, Spearman vs realized return) and
    action-linear dynamics R^2 on held-out transitions."""
    import torch
    from .execution import _team_finish_rates

    mat = [learner.replay.materialize(item) if "chorus_in" not in item else item for item in heldout_items]
    if not mat:
        return {"error": "no heldout items"}

    learner.input_arena.reserve(
        *(frame for item in mat for frame in (item["chorus_in"], item["chorus_out"])),
        count=2,
        capacity=learner.program.capacity,
    )
    learner.program.bind_capacity(learner.input_arena.capacity)
    learner.program.reserve(mat[0]["chorus_in"])

    pred_w, targ_w = [], []
    pred_l, targ_l = [], []
    obs_drho, pred_drho_full, pred_drho_drift = [], [], []

    with torch.no_grad():
        for item in mat:
            before, after, payload = learner._prepare_item(item, 1.0, 1.0, learner.item_storage)
            cur = learner.policy_forward(learner.wally, *before)
            nxt = learner.policy_forward(learner.wally, *after)
            tm = payload["train_mask"].to(torch.bool)
            wm = payload["winner_mask"].to(torch.bool)
            nwm = payload["next_winner_mask"].to(torch.bool)
            succ = torch.clamp(payload["successor_rows"].long(), min=0)
            boot = torch.where(nwm, nxt.value_winnie[succ], nxt.value_lou[succ])
            boot = torch.where(
                ((wm == nwm) | (learner.policy_arm == "terminal_win")) & (payload["discount"] > 0),
                boot,
                0.0,
            )
            targ = payload["reward"] + payload["discount"] * boot
            if bool(payload["target_present"].item()):
                targ = payload["value_target"]

            w_sel = tm & wm
            l_sel = tm & (~wm)
            if bool(w_sel.any()):
                pred_w.extend(cur.value_winnie[w_sel].cpu().tolist())
                targ_w.extend(targ[w_sel].cpu().tolist())
            if bool(l_sel.any()):
                pred_l.extend(cur.value_lou[l_sel].cpu().tolist())
                targ_l.extend(targ[l_sel].cpu().tolist())

            if cur.dyn_drift is not None and cur.dyn_action is not None:
                rho_c = _team_finish_rates(before)
                rho_n = _team_finish_rates(after)[succ]
                drho = torch.asinh(rho_n - rho_c)
                present = (
                    torch.ones_like(cur.rate.mean)
                    if cur.rate.present is None
                    else cur.rate.present.to(cur.rate.mean.dtype)
                )
                p_drift = cur.dyn_drift
                p_full = p_drift + (cur.dyn_action * payload["velocity"] * present).sum(dim=-1)
                d_sel = tm & (payload["successor_rows"] >= 0)
                if bool(d_sel.any()):
                    obs_drho.extend(drho[d_sel].cpu().tolist())
                    pred_drho_full.extend(p_full[d_sel].cpu().tolist())
                    pred_drho_drift.extend(p_drift[d_sel].cpu().tolist())

    def _mse(a, b):
        if not a:
            return None
        diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        return float(np.mean(diff * diff))

    def _r2(y_true, y_pred):
        if len(y_true) < 2:
            return None
        yt = np.asarray(y_true, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)
        var = float(np.var(yt))
        if var <= 1e-18:
            return None
        return float(1.0 - np.mean((yt - yp) ** 2) / var)

    return {
        "n_items": len(mat),
        "winnie_n": len(pred_w),
        "winnie_mse": _mse(pred_w, targ_w),
        "winnie_spearman": spearman_correlation(pred_w, targ_w),
        "lou_n": len(pred_l),
        "lou_mse": _mse(pred_l, targ_l),
        "lou_spearman": spearman_correlation(pred_l, targ_l),
        "dynamics_n": len(obs_drho),
        "dynamics_mse_full": _mse(obs_drho, pred_drho_full),
        "dynamics_mse_drift_only": _mse(obs_drho, pred_drho_drift),
        "dynamics_r2_full": _r2(obs_drho, pred_drho_full),
        "dynamics_r2_drift_only": _r2(obs_drho, pred_drho_drift),
        "dynamics_action_gain": (
            (_mse(obs_drho, pred_drho_drift) - _mse(obs_drho, pred_drho_full))
            if obs_drho
            else None
        ),
    }


def grad_signal_outer_correlation(round_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Direct Xonotic analog of grad_signal_tau_correlation in ~/vllm:
    correlates round-t McCandlish gradient SNR / cosine / step_efficiency against
    next-round outer return delta (delta_return / delta_score_rate)."""
    cos_vals, eff_vals, snr_vals, delta_ret, delta_score = [], [], [], [], []
    for row in round_rows:
        cos_vals.append(row.get("grad_signal_cosine"))
        eff_vals.append(row.get("grad_step_efficiency"))
        snr_vals.append(row.get("grad_snr"))
        delta_ret.append(row.get("next_round_delta_return"))
        delta_score.append(row.get("next_round_delta_score_rate"))
    return {
        "n_rounds": len(round_rows),
        "cosine_vs_delta_return_pearson": pearson_correlation(cos_vals, delta_ret),
        "cosine_vs_delta_return_spearman": spearman_correlation(cos_vals, delta_ret),
        "step_eff_vs_delta_return_pearson": pearson_correlation(eff_vals, delta_ret),
        "step_eff_vs_delta_return_spearman": spearman_correlation(eff_vals, delta_ret),
        "snr_vs_delta_return_spearman": spearman_correlation(snr_vals, delta_ret),
        "cosine_vs_delta_score_rate_spearman": spearman_correlation(cos_vals, delta_score),
    }
