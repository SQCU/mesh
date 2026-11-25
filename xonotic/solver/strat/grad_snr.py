from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch

from .execution import LOSS_WEIGHTS, _team_finish_rates
from .policy_contract import POLICY_LOG_RATIO_BOUND, POLICY_RATIO_CLIP, POLICY_ROLLOUT_IS_THRESHOLD
from .policy_math import clipped_policy_surrogate, rollout_importance_weights
from .strategy import logp_of

MODULE_GROUPS = (
    "encoder",
    "gram_cgt",
    "scale_moe",
    "decoder_trunk",
    "action_head",
    "value_head",
    "dynamics_head",
    "other",
)

LOSS_COMPONENTS = (
    "total",
    "actor_pg",
    "actor_pg_unit_is",
    "critic_winnie",
    "critic_lou",
    "critic_both",
    "dynamics",
    "moe_balance",
)


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _finite_mean(values: Iterable) -> float | None:
    finite = [float(v) for v in values if _is_finite_number(v)]
    return sum(finite) / len(finite) if finite else None


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if _is_finite_number(x) and _is_finite_number(y)]
    if len(pairs) < 3:
        return None
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    mx = sum(x_vals) / len(x_vals)
    my = sum(y_vals) / len(y_vals)
    vx = sum((x - mx) ** 2 for x in x_vals)
    vy = sum((y - my) ** 2 for y in y_vals)
    if vx <= 0.0 or vy <= 0.0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def spearman_correlation(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if _is_finite_number(x) and _is_finite_number(y)]
    if len(pairs) < 3:
        return None
    return pearson_correlation(_average_ranks([p[0] for p in pairs]), _average_ranks([p[1] for p in pairs]))


def module_group_of(param_name: str) -> str:
    root = param_name.split(".", 1)[0]
    if root in ("phil", "quinn", "participant", "cart", "team", "neighborhood", "input_gate", "input_value", "input_out"):
        return "encoder"
    if root in ("ir_query", "team_metric", "rival_metric", "dpp_metric", "dpp_quality", "dpp_out"):
        return "gram_cgt"
    if root in ("scale_in", "scale_router", "scale_w1", "scale_w2", "scale_w3", "scale_out"):
        return "scale_moe"
    if root in ("gia", "uma", "dov"):
        return "decoder_trunk"
    if root in ("dyn_drift", "dyn_action"):
        return "dynamics_head"
    if root == "heads":
        return "heads"
    return "other"


def split_named_gradients_by_group(grads: dict[str, torch.Tensor | None]) -> dict[str, dict[str, torch.Tensor]]:
    grouped: dict[str, dict[str, torch.Tensor]] = {name: {} for name in MODULE_GROUPS}
    for name, grad in grads.items():
        if grad is None or not grad.numel():
            continue
        group = module_group_of(name)
        if group == "heads" and grad.ndim >= 1 and grad.shape[0] > 2:
            grouped["action_head"][name + ":action"] = grad[:-2]
            grouped["value_head"][name + ":value"] = grad[-2:]
        elif group == "heads":
            grouped["action_head"][name] = grad
        else:
            grouped[group][name] = grad
    return grouped


def _grad_norm_sq(grads: dict[str, torch.Tensor | None]) -> float:
    total = 0.0
    for grad in grads.values():
        if grad is not None and grad.numel():
            total += float(grad.detach().cpu().to(torch.float64).square().sum().item())
    return total


def _grad_dot(a: dict[str, torch.Tensor | None], b: dict[str, torch.Tensor | None]) -> float:
    total = 0.0
    for name, ga in a.items():
        gb = b.get(name)
        if ga is None or gb is None or not ga.numel():
            continue
        total += float((ga.detach().cpu().to(torch.float64) * gb.detach().cpu().to(torch.float64)).sum().item())
    return total


def grad_cosine(a: dict[str, torch.Tensor | None], b: dict[str, torch.Tensor | None]) -> float | None:
    na = _grad_norm_sq(a)
    nb = _grad_norm_sq(b)
    if na <= 0.0 or nb <= 0.0:
        return None
    return _grad_dot(a, b) / math.sqrt(na * nb)


def mccandlish_pair_from_grads(
    replica_grads: list[dict[str, torch.Tensor | None]],
    *,
    b0_transitions: float = 1.0,
    b0_rows: float = 1.0,
    generation: int | None = None,
) -> dict | None:
    r = len(replica_grads)
    if r < 2:
        return None
    local_sq_list = [_grad_norm_sq(g) for g in replica_grads]
    local_norms = [math.sqrt(max(0.0, sq)) for sq in local_sq_list]
    mean_local_sq = sum(local_sq_list) / r
    keys = set().union(*(g.keys() for g in replica_grads))
    global_sq = 0.0
    for name in keys:
        tensors = [g[name].detach().cpu().to(torch.float64) for g in replica_grads if g.get(name) is not None and g[name].numel()]
        if not tensors:
            continue
        mean_t = sum(tensors) / r
        global_sq += float(mean_t.square().sum().item())
    noise_sq = (r / (r - 1.0)) * (mean_local_sq - global_sq)
    signal_sq = (r * global_sq - mean_local_sq) / (r - 1.0)
    if r == 2:
        denom = local_norms[0] * local_norms[1]
        cosine = signal_sq / denom if denom > 0.0 else None
    else:
        cosine = signal_sq / mean_local_sq if mean_local_sq > 0.0 else None
    return {
        "generation": generation,
        "replicas": r,
        "mean_local_norm": sum(local_norms) / r,
        "global_norm": math.sqrt(max(0.0, global_sq)),
        "signal_sq": signal_sq,
        "noise_sq": noise_sq,
        "cosine": cosine,
        "b0_transitions": float(b0_transitions),
        "b0_rows": float(b0_rows),
    }


def summarize_mccandlish_pairs(pairs: list[dict]) -> dict | None:
    if not pairs:
        return None
    n = len(pairs)
    mean_r = sum(float(p["replicas"]) for p in pairs) / n
    sum_signal_sq = sum(float(p["signal_sq"]) for p in pairs)
    sum_noise_sq = sum(float(p["noise_sq"]) for p in pairs)
    mean_b0_transitions = _finite_mean([p.get("b0_transitions") for p in pairs])
    mean_b0_rows = _finite_mean([p.get("b0_rows") for p in pairs])
    noise_over_signal = sum_noise_sq / sum_signal_sq if sum_signal_sq > 0.0 else None
    b_noise_ratio = noise_over_signal / mean_r if noise_over_signal is not None else None
    step_efficiency = (
        1.0 / (1.0 + b_noise_ratio)
        if b_noise_ratio is not None and b_noise_ratio >= 0.0
        else None
    )
    return {
        "paired_updates": n,
        "replicas": mean_r,
        "mean_local_grad_norm": _finite_mean([p.get("mean_local_norm") for p in pairs]),
        "mean_global_grad_norm": _finite_mean([p.get("global_norm") for p in pairs]),
        "mean_signal_sq": sum_signal_sq / n,
        "mean_noise_sq": sum_noise_sq / n,
        "mean_cosine": _finite_mean([p.get("cosine") for p in pairs]),
        "snr": sum_signal_sq / sum_noise_sq if sum_noise_sq > 0.0 else None,
        "b_eff_transitions": mean_r * mean_b0_transitions if mean_b0_transitions is not None else None,
        "b_eff_rows": mean_r * mean_b0_rows if mean_b0_rows is not None else None,
        "b_noise_transitions": (
            mean_b0_transitions * noise_over_signal
            if noise_over_signal is not None and mean_b0_transitions is not None
            else None
        ),
        "b_noise_rows": (
            mean_b0_rows * noise_over_signal
            if noise_over_signal is not None and mean_b0_rows is not None
            else None
        ),
        "b_noise_ratio": b_noise_ratio,
        "step_efficiency": step_efficiency,
    }


def mccandlish_from_norm_records(
    records: list[dict],
    *,
    local_key: str = "local_gradient_norm",
    global_key: str = "gradient_norm",
) -> dict | None:
    by_gen: dict[int, list[dict]] = {}
    for entry in records:
        gen = entry.get("generation")
        if not isinstance(gen, int) or isinstance(gen, bool):
            continue
        if not _is_finite_number(entry.get(local_key)) or not _is_finite_number(entry.get(global_key)):
            continue
        by_gen.setdefault(gen, []).append(entry)
    pairs = []
    for gen in sorted(by_gen):
        entries = by_gen[gen]
        r = len(entries)
        if r < 2:
            continue
        locals_ = [float(u[local_key]) for u in entries]
        global_norm = sum(float(u[global_key]) for u in entries) / r
        mean_local_sq = sum(v * v for v in locals_) / r
        global_sq = global_norm * global_norm
        noise_sq = (r / (r - 1.0)) * (mean_local_sq - global_sq)
        signal_sq = (r * global_sq - mean_local_sq) / (r - 1.0)
        if r == 2:
            denom = locals_[0] * locals_[1]
            cosine = signal_sq / denom if denom > 0.0 else None
        else:
            cosine = signal_sq / mean_local_sq if mean_local_sq > 0.0 else None
        pairs.append(
            {
                "generation": gen,
                "replicas": r,
                "mean_local_norm": sum(locals_) / r,
                "global_norm": global_norm,
                "signal_sq": signal_sq,
                "noise_sq": noise_sq,
                "cosine": cosine,
                "b0_transitions": _finite_mean([u.get("b0_transitions") for u in entries]) or 1.0,
                "b0_rows": _finite_mean([u.get("b0_rows") for u in entries]) or 1.0,
            }
        )
    return summarize_mccandlish_pairs(pairs)


def decomposed_tensor_losses(model, forward, policy_arm, current_inputs, next_inputs, item):
    value_scale, actor_scale = item["value_scale"], item["actor_scale"]
    velocity = item["velocity"]
    behavior_logp = item["behavior_logp"]
    train_weight = item["train_mask"].to(torch.float32)
    actor_weight = train_weight * item["actor_mask"].to(torch.float32)
    actor_count = torch.clamp(actor_weight.sum(), min=1.0)
    reward = item["reward"]
    winner_mask = item["winner_mask"].to(torch.bool)
    next_winner_mask = item["next_winner_mask"].to(torch.bool)

    current = forward(model, *current_inputs)
    following = forward(model, *next_inputs)

    actor_active = (actor_weight > 0) & (actor_scale > 0)
    logpi = logp_of(current, velocity, actor_active)

    value = torch.where(winner_mask, current.value_winnie, current.value_lou)
    successor_rows = torch.clamp(item["successor_rows"].long(), min=0)
    next_values_w = following.value_winnie[successor_rows]
    next_values_l = following.value_lou[successor_rows]
    bootstrap = torch.where(next_winner_mask, next_values_w, next_values_l)
    bootstrap = torch.where(
        ((winner_mask == next_winner_mask) | (policy_arm == "terminal_win")) & (item["discount"] > 0),
        bootstrap,
        0.0,
    )
    target = reward + item["discount"] * bootstrap.detach()
    target = torch.where(item["target_present"], item["value_target"], target)
    error = target - value

    present = (
        torch.ones_like(current.rate.mean)
        if current.rate.present is None
        else current.rate.present.to(current.rate.mean.dtype)
    )
    state_count = torch.clamp(present.sum(dim=-1), min=1.0)
    behavior = torch.where(actor_active, behavior_logp, logpi.detach())
    log_ratio = torch.clamp(logpi - behavior, -POLICY_LOG_RATIO_BOUND, POLICY_LOG_RATIO_BOUND)
    rollout_log_ratio = log_ratio.detach()
    ratio = rollout_importance_weights(rollout_log_ratio)
    td_advantage = torch.clamp(error.detach(), -2.0, 2.0)
    surrogate = clipped_policy_surrogate(log_ratio, td_advantage, POLICY_RATIO_CLIP) * torch.sqrt(state_count)
    actor_raw = -(surrogate * actor_weight).sum() / actor_count
    actor_unit_is_raw = -(
        clipped_policy_surrogate(logpi - logpi.detach(), td_advantage, POLICY_RATIO_CLIP) * torch.sqrt(state_count) * actor_weight
    ).sum() / actor_count
    winner_weight = winner_mask.to(torch.float32) * train_weight
    loser_weight = (~winner_mask).to(torch.float32) * train_weight
    winner_huber = torch.nn.functional.smooth_l1_loss(current.value_winnie, target, reduction="none", beta=0.5)
    loser_huber = torch.nn.functional.smooth_l1_loss(current.value_lou, target, reduction="none", beta=0.5)
    winner_loss = (winner_huber * winner_weight).sum() / torch.clamp(winner_weight.sum(), min=1.0)
    loser_loss = (loser_huber * loser_weight).sum() / torch.clamp(loser_weight.sum(), min=1.0)
    if current.dyn_drift is not None and current.dyn_action is not None:
        with torch.no_grad():
            rho_curr = _team_finish_rates(current_inputs)
            rho_next = _team_finish_rates(next_inputs)[successor_rows]
            obs_delta_rho = torch.asinh(10.0 * (rho_next - rho_curr))
            dyn_valid = (item["successor_rows"] >= 0).to(torch.float32) * train_weight
        pred_delta_rho = current.dyn_drift + (current.dyn_action * velocity * present).sum(dim=-1)
        dyn_huber = torch.nn.functional.smooth_l1_loss(pred_delta_rho, obs_delta_rho, reduction="none", beta=0.25)
        dyn_loss = (dyn_huber * dyn_valid).sum() / torch.clamp(dyn_valid.sum(), min=1.0)
    else:
        dyn_loss = torch.zeros((), dtype=actor_raw.dtype, device=actor_raw.device)

    components = {
        "actor_pg": actor_scale * LOSS_WEIGHTS["actor"] * actor_raw,
        "actor_pg_unit_is": actor_scale * LOSS_WEIGHTS["actor"] * actor_unit_is_raw,
        "critic_winnie": value_scale * LOSS_WEIGHTS["winnie"] * winner_loss,
        "critic_lou": value_scale * LOSS_WEIGHTS["lou"] * loser_loss,
        "dynamics": value_scale * LOSS_WEIGHTS["dynamics"] * dyn_loss,
        "moe_balance": value_scale * LOSS_WEIGHTS["moe_balance"] * current.scale_balance,
    }
    components["critic_both"] = components["critic_winnie"] + components["critic_lou"]
    components["total"] = (
        components["actor_pg"]
        + components["critic_winnie"]
        + components["critic_lou"]
        + components["dynamics"]
        + components["moe_balance"]
    )
    with torch.no_grad():
        active_np = actor_active.detach().cpu().numpy().astype(bool)
        lr_np = rollout_log_ratio.detach().cpu().numpy()[active_np] if active_np.any() else np.empty((0,), dtype=np.float32)
        ratio_np = ratio.detach().cpu().numpy()[active_np] if active_np.any() else np.empty((0,), dtype=np.float32)
        coord_counts = present.sum(dim=-1).detach().cpu().numpy()[active_np] if active_np.any() else np.empty((0,), dtype=np.float32)
        adv_np = td_advantage.detach().cpu().numpy()[active_np] if active_np.any() else np.empty((0,), dtype=np.float32)
        diag = {
            "active_actor_rows": int(active_np.sum()),
            "train_rows": int((train_weight > 0).sum().item()),
            "winner_rows": int((winner_weight > 0).sum().item()),
            "loser_rows": int((loser_weight > 0).sum().item()),
            "writable_coords_mean": float(coord_counts.mean()) if coord_counts.size else 0.0,
            "rollout_log_ratio_mean": float(lr_np.mean()) if lr_np.size else None,
            "rollout_log_ratio_min": float(lr_np.min()) if lr_np.size else None,
            "rollout_log_ratio_max": float(lr_np.max()) if lr_np.size else None,
            "is_ratio_mean": float(ratio_np.mean()) if ratio_np.size else None,
            "is_floor_fraction": float(np.mean(lr_np < -POLICY_LOG_RATIO_BOUND)) if lr_np.size else None,
            "is_clip_fraction": float(np.mean(lr_np > math.log(POLICY_ROLLOUT_IS_THRESHOLD))) if lr_np.size else None,
            "advantage_abs_mean": float(np.abs(adv_np).mean()) if adv_np.size else None,
            "advantage_std": float(adv_np.std()) if adv_np.size else None,
            "rate_mean_abs": float(current.rate.mean[present > 0].abs().mean().item()) if bool((present > 0).any()) else 0.0,
            "rate_sigma_mean": float(torch.exp(current.rate.log_scale[present > 0]).mean().item()) if bool((present > 0).any()) else 0.0,
            "velocity_abs_mean": float(velocity[present > 0].abs().mean().item()) if bool((present > 0).any()) else 0.0,
            "state_abs_mean": float(current_inputs.state[present > 0].abs().mean().item()) if bool((present > 0).any()) else 0.0,
        }
    return components, diag


def _accumulate_component_grads(learner, batch_entries, components_to_measure):
    model = learner.wally
    params = list(model.named_parameters())
    param_names = [name for name, _ in params]
    param_tensors = [p for _, p in params]
    sums = {comp: {name: None for name in param_names} for comp in components_to_measure}
    diags = []
    for item, value_scale, actor_scale in batch_entries:
        before, after, payload = learner._prepare_item(item, value_scale, actor_scale, learner.item_storage)
        comps, diag = decomposed_tensor_losses(model, learner.policy_forward, learner.policy_arm, before, after, payload)
        diags.append(diag)
        active_comps = [c for c in components_to_measure if c in comps and comps[c].requires_grad]
        for idx, comp in enumerate(active_comps):
            retain = idx + 1 < len(active_comps)
            grads = torch.autograd.grad(comps[comp], param_tensors, retain_graph=retain, allow_unused=True)
            acc = sums[comp]
            for name, g in zip(param_names, grads):
                if g is None:
                    continue
                g_cpu = g.detach().to(device="cpu", dtype=torch.float32)
                acc[name] = g_cpu if acc[name] is None else (acc[name] + g_cpu)
    return sums, diags


def probe_batch_mccandlish(
    learner,
    items: list[dict],
    *,
    microbatch_size: int = 2,
    max_pairs: int = 8,
    components: tuple[str, ...] = LOSS_COMPONENTS,
    seed: int = 0,
) -> dict:
    """Measure McCandlish (2018) gradient SNR, B_noise, step efficiency, and cross-term
    interference across disjoint paired microbatches of real transitions."""
    mat = [learner.replay.materialize(item) if "chorus_in" not in item else item for item in items]
    if len(mat) < 2:
        return {"error": "need at least 2 transitions", "n_items": len(mat)}
    learner.input_arena.reserve(
        *(frame for item in mat for frame in (item["chorus_in"], item["chorus_out"])),
        count=2,
        capacity=learner.program.capacity,
    )
    learner.program.bind_capacity(learner.input_arena.capacity)
    learner.program.reserve(mat[0]["chorus_in"])

    b0 = max(1, min(int(microbatch_size), len(mat) // 2))
    rng = np.random.default_rng(seed)
    pair_indices = []
    if len(mat) >= 2 * b0:
        order = np.arange(len(mat))
        for start in range(0, len(mat) - 2 * b0 + 1, 2 * b0):
            pair_indices.append((order[start : start + b0].tolist(), order[start + b0 : start + 2 * b0].tolist()))
        while len(pair_indices) < max_pairs and len(mat) >= 2 * b0:
            perm = rng.permutation(len(mat))
            pair_indices.append((perm[:b0].tolist(), perm[b0 : 2 * b0].tolist()))
        pair_indices = pair_indices[:max_pairs]

    comp_pairs = {comp: [] for comp in components}
    group_pairs = {comp: {grp: [] for grp in MODULE_GROUPS} for comp in components}
    cross_cosines = {
        "actor_vs_critic": [],
        "actor_unit_is_vs_critic": [],
        "actor_vs_total": [],
        "actor_unit_is_vs_total": [],
        "critic_w_vs_critic_l": [],
        "dynamics_vs_critic": [],
    }
    norm_ratios = {
        "actor_over_total": [],
        "actor_unit_is_over_total": [],
        "critic_over_total": [],
        "dynamics_over_total": [],
        "moe_over_total": [],
    }
    all_diags = []

    def _pack_half(idxs):
        sub = [mat[i] for i in idxs]
        v_mass = max(sum(float(it.get("value_weight", 1.0)) for it in sub), 1.0)
        a_rows = [int(np.count_nonzero(it["actor_mask"] & it["train_mask"])) for it in sub]
        a_mass = max(sum(a_rows), 1)
        return [
            (it, float(it.get("value_weight", 1.0)) / v_mass, rows / a_mass)
            for it, rows in zip(sub, a_rows)
        ], sum(a_rows)

    for pair_id, (left_idx, right_idx) in enumerate(pair_indices):
        left_entries, left_rows = _pack_half(left_idx)
        right_entries, right_rows = _pack_half(right_idx)
        left_grads, left_diags = _accumulate_component_grads(learner, left_entries, components)
        right_grads, right_diags = _accumulate_component_grads(learner, right_entries, components)
        all_diags.extend(left_diags)
        all_diags.extend(right_diags)
        b0_rows = 0.5 * (left_rows + right_rows)

        for comp in components:
            g0, g1 = left_grads[comp], right_grads[comp]
            pair = mccandlish_pair_from_grads(
                [g0, g1], b0_transitions=float(b0), b0_rows=float(b0_rows), generation=pair_id
            )
            if pair is not None:
                comp_pairs[comp].append(pair)
            gg0 = split_named_gradients_by_group(g0)
            gg1 = split_named_gradients_by_group(g1)
            for grp in MODULE_GROUPS:
                if not gg0[grp] and not gg1[grp]:
                    continue
                gpair = mccandlish_pair_from_grads(
                    [gg0[grp], gg1[grp]], b0_transitions=float(b0), b0_rows=float(b0_rows), generation=pair_id
                )
                if gpair is not None:
                    group_pairs[comp][grp].append(gpair)

        for grads in (left_grads, right_grads):
            if "actor_pg" in grads and "critic_both" in grads:
                c = grad_cosine(grads["actor_pg"], grads["critic_both"])
                if c is not None:
                    cross_cosines["actor_vs_critic"].append(c)
            if "actor_pg_unit_is" in grads and "critic_both" in grads:
                c = grad_cosine(grads["actor_pg_unit_is"], grads["critic_both"])
                if c is not None:
                    cross_cosines["actor_unit_is_vs_critic"].append(c)
            if "actor_pg" in grads and "total" in grads:
                c = grad_cosine(grads["actor_pg"], grads["total"])
                if c is not None:
                    cross_cosines["actor_vs_total"].append(c)
            if "actor_pg_unit_is" in grads and "total" in grads:
                c = grad_cosine(grads["actor_pg_unit_is"], grads["total"])
                if c is not None:
                    cross_cosines["actor_unit_is_vs_total"].append(c)
            if "critic_winnie" in grads and "critic_lou" in grads:
                c = grad_cosine(grads["critic_winnie"], grads["critic_lou"])
                if c is not None:
                    cross_cosines["critic_w_vs_critic_l"].append(c)
            if "dynamics" in grads and "critic_both" in grads:
                c = grad_cosine(grads["dynamics"], grads["critic_both"])
                if c is not None:
                    cross_cosines["dynamics_vs_critic"].append(c)
            n_tot = math.sqrt(max(0.0, _grad_norm_sq(grads.get("total", {}))))
            if n_tot > 0.0:
                norm_ratios["actor_over_total"].append(math.sqrt(max(0.0, _grad_norm_sq(grads.get("actor_pg", {})))) / n_tot)
                norm_ratios["actor_unit_is_over_total"].append(
                    math.sqrt(max(0.0, _grad_norm_sq(grads.get("actor_pg_unit_is", {})))) / n_tot
                )
                norm_ratios["critic_over_total"].append(math.sqrt(max(0.0, _grad_norm_sq(grads.get("critic_both", {})))) / n_tot)
                norm_ratios["dynamics_over_total"].append(math.sqrt(max(0.0, _grad_norm_sq(grads.get("dynamics", {})))) / n_tot)
                norm_ratios["moe_over_total"].append(math.sqrt(max(0.0, _grad_norm_sq(grads.get("moe_balance", {})))) / n_tot)

    return {
        "n_items": len(mat),
        "microbatch_size": b0,
        "pairs_evaluated": len(pair_indices),
        "by_component": {comp: summarize_mccandlish_pairs(comp_pairs[comp]) for comp in components},
        "by_component_and_group": {
            comp: {
                grp: summary
                for grp in MODULE_GROUPS
                if (summary := summarize_mccandlish_pairs(group_pairs[comp][grp])) is not None
            }
            for comp in components
        },
        "cross_term_cosines": {k: _finite_mean(v) for k, v in cross_cosines.items()},
        "norm_ratios_to_total": {k: _finite_mean(v) for k, v in norm_ratios.items()},
        "diagnostics": {
            "writable_coords_mean": _finite_mean([d["writable_coords_mean"] for d in all_diags]),
            "rollout_log_ratio_mean": _finite_mean([d["rollout_log_ratio_mean"] for d in all_diags]),
            "rollout_log_ratio_min": min((d["rollout_log_ratio_min"] for d in all_diags if d["rollout_log_ratio_min"] is not None), default=None),
            "rollout_log_ratio_max": max((d["rollout_log_ratio_max"] for d in all_diags if d["rollout_log_ratio_max"] is not None), default=None),
            "is_ratio_mean": _finite_mean([d["is_ratio_mean"] for d in all_diags]),
            "is_floor_fraction": _finite_mean([d["is_floor_fraction"] for d in all_diags]),
            "is_clip_fraction": _finite_mean([d["is_clip_fraction"] for d in all_diags]),
            "advantage_abs_mean": _finite_mean([d["advantage_abs_mean"] for d in all_diags]),
            "advantage_std": _finite_mean([d["advantage_std"] for d in all_diags]),
            "rate_mean_abs": _finite_mean([d["rate_mean_abs"] for d in all_diags]),
            "rate_sigma_mean": _finite_mean([d["rate_sigma_mean"] for d in all_diags]),
            "velocity_abs_mean": _finite_mean([d["velocity_abs_mean"] for d in all_diags]),
            "state_abs_mean": _finite_mean([d["state_abs_mean"] for d in all_diags]),
        },
    }


def probe_cross_server_mccandlish(
    learner,
    items: list[dict],
    *,
    components: tuple[str, ...] = LOSS_COMPONENTS,
    max_per_server: int = 8,
) -> dict:
    """True i.i.d. cross-server r=G McCandlish (2018) estimator: each concurrent server
    session (`rollout_group`) forms one independent gradient replica g_1, ..., g_G,
    eliminating single-server temporal autocorrelation bias."""
    mat = [learner.replay.materialize(item) if "chorus_in" not in item else item for item in items]
    by_server: dict[int, list[dict]] = {}
    for it in mat:
        gid = int(it.get("rollout_group", 0))
        by_server.setdefault(gid, []).append(it)
    active_servers = [gid for gid, lst in sorted(by_server.items()) if lst]
    if len(active_servers) < 2:
        return {"error": f"need >=2 distinct rollout_group servers, found {len(active_servers)}", "servers": len(active_servers)}

    per_srv = min(max_per_server, min(len(by_server[gid]) for gid in active_servers))
    selected = {gid: by_server[gid][-per_srv:] for gid in active_servers}
    flat = [it for lst in selected.values() for it in lst]
    learner.input_arena.reserve(
        *(frame for item in flat for frame in (item["chorus_in"], item["chorus_out"])),
        count=2,
        capacity=learner.program.capacity,
    )
    learner.program.bind_capacity(learner.input_arena.capacity)
    learner.program.reserve(flat[0]["chorus_in"])

    replica_grads_by_comp: dict[str, list[dict[str, torch.Tensor | None]]] = {c: [] for c in components}
    replica_rows = []
    all_diags = []
    for gid in active_servers:
        sub = selected[gid]
        v_mass = max(sum(float(it.get("value_weight", 1.0)) for it in sub), 1.0)
        a_rows = [int(np.count_nonzero(it["actor_mask"] & it["train_mask"])) for it in sub]
        a_mass = max(sum(a_rows), 1)
        entries = [(it, float(it.get("value_weight", 1.0)) / v_mass, rows / a_mass) for it, rows in zip(sub, a_rows)]
        grads, diags = _accumulate_component_grads(learner, entries, components)
        all_diags.extend(diags)
        replica_rows.append(sum(a_rows))
        for c in components:
            replica_grads_by_comp[c].append(grads[c])

    mean_rows = float(np.mean(replica_rows)) if replica_rows else 1.0
    by_component = {}
    by_component_and_group = {}
    for c in components:
        pair = mccandlish_pair_from_grads(
            replica_grads_by_comp[c],
            b0_transitions=float(per_srv),
            b0_rows=mean_rows,
            generation=0,
        )
        by_component[c] = summarize_mccandlish_pairs([pair]) if pair is not None else None
        grouped_reps = [split_named_gradients_by_group(g) for g in replica_grads_by_comp[c]]
        g_out = {}
        for grp in MODULE_GROUPS:
            sub_reps = [gr[grp] for gr in grouped_reps]
            if not any(sub_reps):
                continue
            gp = mccandlish_pair_from_grads(sub_reps, b0_transitions=float(per_srv), b0_rows=mean_rows, generation=0)
            if gp is not None:
                g_out[grp] = summarize_mccandlish_pairs([gp])
        by_component_and_group[c] = g_out

    return {
        "mode": "cross_server_iid",
        "servers": len(active_servers),
        "server_ids": active_servers,
        "transitions_per_server": per_srv,
        "total_transitions": len(flat),
        "by_component": by_component,
        "by_component_and_group": by_component_and_group,
        "diagnostics": {
            "writable_coords_mean": _finite_mean([d["writable_coords_mean"] for d in all_diags]),
            "rollout_log_ratio_mean": _finite_mean([d["rollout_log_ratio_mean"] for d in all_diags]),
            "is_ratio_mean": _finite_mean([d["is_ratio_mean"] for d in all_diags]),
            "is_floor_fraction": _finite_mean([d["is_floor_fraction"] for d in all_diags]),
            "advantage_abs_mean": _finite_mean([d["advantage_abs_mean"] for d in all_diags]),
        },
    }


class OnlineMcCandlishTracker:
    """Zero-extra-backward online split-half McCandlish gradient SNR tracker.
    Whenever an update processes >= 2 microbatch items, captures half-0 gradient snapshot
    after the midpoint and derives half-1 = total - half-0 after the full accumulation,
    recording exact r=2 McCandlish signal/noise/cosine/B_noise globally and per module group."""

    def __init__(self, maxlen: int = 64):
        self.maxlen = max(1, int(maxlen))
        self.pairs: list[dict] = []
        self.group_pairs: dict[str, list[dict]] = {grp: [] for grp in MODULE_GROUPS}

    def record_split_update(
        self,
        half0_grads: dict[str, torch.Tensor],
        total_grads: dict[str, torch.Tensor | None],
        *,
        generation: int,
        b0_transitions: float,
        b0_rows: float,
    ) -> dict | None:
        g0 = {k: v.detach().cpu().to(torch.float64) for k, v in half0_grads.items() if v is not None}
        g1 = {}
        for k, vt in total_grads.items():
            if vt is None:
                continue
            vt_cpu = vt.detach().cpu().to(torch.float64)
            g1[k] = vt_cpu - g0[k] if k in g0 else vt_cpu
        # Note: total_grads is the sum of half0 and half1, so average is 0.5*(g0+g1)
        pair = mccandlish_pair_from_grads([g0, g1], b0_transitions=b0_transitions, b0_rows=b0_rows, generation=generation)
        if pair is None:
            return None
        self.pairs.append(pair)
        if len(self.pairs) > self.maxlen:
            self.pairs.pop(0)
        gg0 = split_named_gradients_by_group(g0)
        gg1 = split_named_gradients_by_group(g1)
        per_group = {}
        for grp in MODULE_GROUPS:
            if not gg0[grp] and not gg1[grp]:
                continue
            gp = mccandlish_pair_from_grads([gg0[grp], gg1[grp]], b0_transitions=b0_transitions, b0_rows=b0_rows, generation=generation)
            if gp is not None:
                self.group_pairs[grp].append(gp)
                if len(self.group_pairs[grp]) > self.maxlen:
                    self.group_pairs[grp].pop(0)
                per_group[grp] = {
                    "cosine": gp["cosine"],
                    "signal_sq": gp["signal_sq"],
                    "noise_sq": gp["noise_sq"],
                }
        return {
            "last": {
                "generation": generation,
                "cosine": pair["cosine"],
                "signal_sq": pair["signal_sq"],
                "noise_sq": pair["noise_sq"],
                "local_grad_norm": pair["mean_local_norm"],
                "global_grad_norm": pair["global_norm"],
                "by_group": per_group,
            },
            "window": summarize_mccandlish_pairs(self.pairs),
            "window_by_group": {
                grp: summary
                for grp in MODULE_GROUPS
                if (summary := summarize_mccandlish_pairs(self.group_pairs[grp])) is not None
            },
        }
