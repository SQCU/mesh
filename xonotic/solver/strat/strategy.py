from __future__ import annotations

import math
from typing import NamedTuple

import mlx.core as mx

from .cast_header import (
    Wally,
    actuator,
    dee,
    dina_action,
    dina_drift,
    dina_matrix,
    dina_readout,
    dina_state,
    gia_uma_dov,
    participant_gram_matrix,
    ir_query,
    ir_value,
    kay,
    lou,
    norm,
    phil,
    quinn,
    scale_fuse,
    tau,
    val,
    vera_lou,
    vera_winnie,
    winnie,
)
from .matmul import batched_matrix_vector, matrix_multiply, matrix_multiply_transpose_right
__all__ = [
    "Dynamics", "Strategy", "strategy", "dynamics", "log_probs", "logp_of",
    "control_logp_of", "sample_controls", "act", "integrate", "measure_log_density",
]

class Dynamics(NamedTuple):
    mean: mx.array
    first: mx.array
    second: mx.array
    matrix: mx.array
    disagreement: mx.array

class Strategy(NamedTuple):
    dw_dt: mx.array
    logits: mx.array
    ir: mx.array
    query: mx.array
    value_winnie: mx.array
    value_lou: mx.array
    aux_winnie: mx.array
    aux_lou: mx.array
    coupling: mx.array
    belief: mx.array
    pooled: mx.array
    weights: mx.array
    dee: mx.array
    guidance: mx.array
    uncertainty: mx.array
    scale_matrix_stats: mx.array
    scale_expert_load: mx.array
    controls: mx.array
    control_log_scale: mx.array
    control_density: mx.array

def measure_log_density(scores, action_mass):
    return scores + mx.log(action_mass)

def normalized_log_probs(logits):
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

def strategy(
    wally: Wally,
    xan: mx.array,
    zed: mx.array,
    cell_slots: mx.array,
    gigi: mx.array,
    semantics: mx.array,
    team_ids: mx.array,
    weights: mx.array,
    action_mass: mx.array,
    delta: mx.array,
    control_weight: mx.array,
    exploration_weight: mx.array,
    *,
    participant_fusion_scale=None,
    residual_fusion_scale=None,
    execute_remote_scale=True,
) -> Strategy:
    bea = norm(matrix_multiply(gigi, phil(wally, cell_slots)))

    q = norm(quinn(wally, xan, bea, semantics))

    k = norm(kay(wally, zed))
    v = norm(val(wally, zed))

    score = matrix_multiply_transpose_right(q, k) / (wally.w.d ** 0.5)

    participant_fusion_scale = float(
        getattr(wally, "participant_fusion_scale", 1.0)
        if participant_fusion_scale is None else participant_fusion_scale
    )
    same_team = team_ids[:, None] == team_ids[None, :]
    coupling = participant_gram_matrix(wally, q, team_ids) * participant_fusion_scale

    quality = mx.mean(mx.logaddexp(score, 0), axis=0)
    inclusion = dee(quality, k)
    projected = ir_query(wally, q)
    same_count = mx.sum(same_team, axis=1, keepdims=True)
    rival_count = q.shape[0] - same_count
    normalizer = mx.where(
        same_team, same_count, mx.maximum(rival_count, 1),
    )
    mixed = matrix_multiply(coupling / normalizer, projected)
    ir = (
        mixed[:, None, :]
        + (score * inclusion[None, :])[:, :, None]
        * ir_value(wally, v)[None, :, :]
    )
    scale_delta, scale_matrix_stats, scale_expert_load = scale_fuse(
        wally, ir, execute_remote=execute_remote_scale,
        residual_fusion_scale=residual_fusion_scale,
    )
    ir = norm(ir + scale_delta)

    dw_dt = gia_uma_dov(wally, ir)

    weights_next = integrate(weights, dw_dt, delta)
    control = dynamics(wally, q[:, None, :], ir)
    future_query = norm(q[:, None, :] + dina_readout(wally, control.mean))
    winner_guidance = vera_winnie(wally, future_query) - vera_winnie(wally, q)[:, None]
    loser_guidance = vera_lou(wally, future_query) - vera_lou(wally, q)[:, None]
    guidance = mx.where(semantics[:, 5:6] >= 0.5, winner_guidance, loser_guidance)
    uncertainty = mx.tanh(mx.sqrt(mx.maximum(control.disagreement, 0)))
    anticipated = (
        weights_next
        + control_weight * guidance
        + exploration_weight * uncertainty
    )
    logits = measure_log_density(anticipated / tau(wally), action_mass)

    allocation = mx.exp(normalized_log_probs(logits))
    pooled = norm(mx.sum(ir * allocation[:, :, None], axis=1))
    actuator_parameters = actuator(wally, ir)
    return Strategy(
        dw_dt=dw_dt,
        logits=logits,
        ir=ir,
        query=q,
        value_winnie=winnie(wally, pooled),
        value_lou=lou(wally, pooled),
        aux_winnie=vera_winnie(wally, q),
        aux_lou=vera_lou(wally, q),
        coupling=coupling,
        belief=bea,
        pooled=pooled,
        weights=weights_next,
        dee=inclusion,
        guidance=guidance,
        uncertainty=uncertainty,
        scale_matrix_stats=scale_matrix_stats,
        scale_expert_load=scale_expert_load,
        controls=actuator_parameters[..., :3],
        control_log_scale=actuator_parameters[..., 3:],
        control_density=mx.ones((xan.shape[0],), dtype=xan.dtype),
    )

def dynamics(wally: Wally, y: mx.array, u: mx.array) -> Dynamics:
    state = norm(dina_state(wally, y))
    action = norm(dina_action(wally, u))
    drift_first, drift_second = dina_drift(wally, state)
    matrix_first, matrix_second = dina_matrix(wally, state)
    first = drift_first + batched_matrix_vector(
        matrix_first, action,
    ) / (wally.w.d_u ** 0.5)
    second = drift_second + batched_matrix_vector(
        matrix_second, action,
    ) / (wally.w.d_u ** 0.5)
    matrix = 0.5 * (matrix_first + matrix_second)
    return Dynamics(
        mean=0.5 * (first + second),
        first=first,
        second=second,
        matrix=matrix,
        disagreement=mx.mean(mx.square(first - second), axis=-1),
    )

def log_probs(out: Strategy) -> mx.array:
    return normalized_log_probs(out.logits)

def control_logp_of(out: Strategy, actions: mx.array, controls: mx.array) -> mx.array:
    mean = mx.take_along_axis(
        out.controls, actions[:, None, None], axis=1,
    )[:, 0, :]
    log_scale = mx.take_along_axis(
        out.control_log_scale, actions[:, None, None], axis=1,
    )[:, 0, :]
    standardized = (controls - mean) * mx.exp(-log_scale)
    density_logp = -0.5 * mx.sum(
        mx.square(standardized) + 2 * log_scale + math.log(2 * math.pi), axis=-1,
    )
    return density_logp * out.control_density

def sample_controls(out: Strategy, actions: mx.array, key: mx.array):
    mean = mx.take_along_axis(
        out.controls, actions[:, None, None], axis=1,
    )[:, 0, :]
    log_scale = mx.take_along_axis(
        out.control_log_scale, actions[:, None, None], axis=1,
    )[:, 0, :]
    controls = mean + out.control_density[:, None] * mx.exp(log_scale) * mx.random.normal(mean.shape, key=key)
    return controls, control_logp_of(out, actions, controls)

def logp_of(out: Strategy, actions: mx.array, controls=None) -> mx.array:
    discrete = mx.take_along_axis(log_probs(out), actions[:, None], axis=-1)[:, 0]
    return discrete if controls is None else discrete + control_logp_of(out, actions, controls)

def act(out: Strategy, key: mx.array):
    action_key, control_key = mx.random.split(key)
    actions = mx.random.categorical(out.logits, key=action_key)
    controls, control_logp = sample_controls(out, actions, control_key)
    return actions, controls, logp_of(out, actions) + control_logp

def integrate(w: mx.array, dw_dt: mx.array, delta: float) -> mx.array:
    return mx.tanh(w + dw_dt * delta)
