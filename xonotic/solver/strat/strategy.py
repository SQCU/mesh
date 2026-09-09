from __future__ import annotations

from typing import NamedTuple

from . import tensor as mx

from . import paged_matrix
from .cast_header import Wally, encode_rows, gia_uma_dov, ir_query, norm, scale_fuse
from .inputs import ChorusArrays
from .matmul import matrix_multiply, matrix_multiply_transpose_left, linear
from .state_steering import StateRate, rate_distribution, logp_of as rate_logp, sample, policy_inputs


class Strategy(NamedTuple):
    rate: StateRate
    ir: mx.array
    value_winnie: mx.array
    value_lou: mx.array
    coupling: mx.array
    local_neighborhood: mx.array
    scale_residual_stats: mx.array
    scale_expert_load: mx.array
    scale_balance: mx.array


class Encoded(NamedTuple):
    ir: mx.array
    coupling: mx.array
    valid: mx.array
    own_rows: mx.array
    page_rows: mx.array
    local: mx.array


def embedded_rows(model, frame, width, nonlinear=True):
    pages, observations, present = policy_inputs(frame, width)
    valid = mx.concatenate((frame.observation_present,
        (mx.any(present, axis=-1) & frame.participant_present[:, None]).reshape(-1),
        frame.cart_present, frame.team_present))
    rows = mx.concatenate((linear(model.participant, observations), linear(model.quinn, pages).reshape(-1, model.participant.weight.shape[0]),
        linear(model.cart, frame.carts), linear(model.team, frame.teams)))
    source_valid = mx.concatenate((frame.context_present, frame.navigation_node_present,
        frame.navigation_edge_present, frame.navigation_cell_present))
    sources = mx.concatenate((linear(model.phil, frame.context),
        model.neighborhood.encode(frame.navigation_nodes, frame.navigation_edges, frame.navigation_cells)))
    sources = mx.where(source_valid[:, None], mx.arcsinh(sources) if nonlinear else sources, 0)
    rows = mx.where(valid[:, None], mx.arcsinh(rows) if nonlinear else rows, 0)
    owner_context = mx.broadcast_to(rows[frame.participant_rows, None, :], (*pages.shape[:2], rows.shape[-1])).reshape(-1, rows.shape[-1])
    rows = rows + mx.concatenate((mx.zeros_like(rows[:observations.shape[0]]), owner_context,
        mx.zeros((frame.carts.shape[0] + frame.teams.shape[0], rows.shape[-1]), dtype=rows.dtype)))
    local = model.neighborhood(rows[:observations.shape[0]], sources, frame.neighborhood_indices,
        frame.neighborhood_distances, frame.neighborhood_present & source_valid[frame.neighborhood_indices], frame.neighborhood_radius,
        source_times=mx.concatenate((frame.context[:, 1], mx.zeros((sources.shape[0] - frame.context.shape[0],), dtype=rows.dtype))),
        observer_times=frame.observations[:, 21], source_temporal=mx.arange(sources.shape[0]) < frame.context.shape[0], gram=nonlinear)
    local = mx.where(frame.observation_present[:, None], local, 0)
    row_local = mx.concatenate((local, mx.broadcast_to(local[frame.participant_rows, None, :],
        (*pages.shape[:2], local.shape[-1])).reshape(-1, local.shape[-1]),
        mx.zeros((frame.carts.shape[0] + frame.teams.shape[0], local.shape[-1]), dtype=rows.dtype)))
    return mx.where(valid[:, None], rows + (mx.arcsinh(row_local) if nonlinear else row_local), 0), valid, local


def row_gram_context(model, teams, rivals, values, frame, global_context):
    n, count = frame.layout.shape[:2]
    global_values = matrix_multiply(rivals, global_context) / model.w.r_e ** .5
    o, p, c = frame.observations.shape[0], n * count, frame.carts.shape[0]
    spans = (slice(0, o), slice(o, o + p), slice(o + p, o + p + c), slice(o + p + c, None))
    observation, pages, carts, team_rows = (teams[span] for span in spans)
    observation_values, page_values, cart_values, team_values = (values[span] for span in spans)
    grouped = mx.concatenate((observation[:, :, None] * observation_values[:, None, :],
        paged_matrix.batched_cross(pages.reshape(n, count, model.w.r), page_values.reshape(n, count, model.w.d_ir)),
        carts[:, :, None] * cart_values[:, None, :], team_rows[:, :, None] * team_values[:, None, :]))
    labels = mx.concatenate((frame.observations[:, 1], frame.team_ids, frame.carts[:, 3], frame.teams[:, 0]))
    membership = (labels[:, None] == frame.teams[None, :, 0]) & frame.team_present[None, :]
    team_context = matrix_multiply(membership.astype(values.dtype).T, grouped.reshape(grouped.shape[0], -1))
    contexts = matrix_multiply(membership.astype(values.dtype), team_context).reshape(grouped.shape)
    outputs = mx.concatenate((mx.sum(observation[:, :, None] * contexts[:o], axis=1),
        paged_matrix.batched_multiply(pages.reshape(n, count, model.w.r), contexts[o:o + n]).reshape(p, model.w.d_ir),
        mx.sum(carts[:, :, None] * contexts[o + n:o + n + c], axis=1),
        mx.sum(team_rows[:, :, None] * contexts[o + n + c:], axis=1)))
    return global_values + outputs / model.w.r ** .5, mx.concatenate((teams, rivals), axis=-1)


def encode_source(wally, *values):
    frame = ChorusArrays(*values)
    rows, valid, local = embedded_rows(wally, frame, wally.w.d_x)
    query = mx.where(valid[:, None], encode_rows(wally, rows), 0)
    projected = ir_query(wally, query)
    normalized = norm(query)
    return projected, local, valid, linear(wally.team_metric, normalized), linear(wally.rival_metric, normalized), norm(projected)


def encode_finish(wally, source, values, strength, global_context):
    frame = ChorusArrays(*values)
    projected, local, valid, teams, rivals, payload = source
    fused, coupling = row_gram_context(wally, teams, rivals, payload, frame, global_context)
    ir = mx.where(valid[:, None], projected + norm(fused) * strength, 0)
    pages = mx.arange(frame.observations.shape[0], frame.observations.shape[0] + frame.layout.shape[0] * frame.layout.shape[1]).reshape(frame.layout.shape[:2])
    return Encoded(ir, coupling, valid, frame.participant_rows, pages, local[frame.participant_rows])


def encode(wally, *values):
    source = encode_source(wally, *values[:-1])
    cross = getattr(wally, 'cross_executor', matrix_multiply_transpose_left)
    return encode_finish(wally, source, values[:-1], values[-1], cross(*source[-2:]))


def read_heads(model, encoded, rows, frame, statistics, load, balance):
    own = mx.concatenate((encoded.own_rows[:, None], encoded.page_rows), axis=1)
    valid = encoded.valid[own] & frame.participant_present[:, None]
    ir = mx.where(valid[..., None], rows[own], 0)
    projected = linear(model.heads, ir)
    coordinates = projected[:, 1:, :-2] + projected[:, :1, :-2]
    values = mx.sum(projected[..., -2:], axis=1)
    rate = rate_distribution(coordinates, frame.layout, frame.participant_present)
    return Strategy(rate, ir.reshape(frame.state.shape[0], -1), values[:, 0], values[:, 1],
        mx.where(valid[..., None], encoded.coupling[own], 0),
        encoded.local * frame.participant_present[:, None], statistics, load, balance)


def decode(wally, encoded, scale_delta, statistics, load, balance, frame):
    mixed = mx.where(encoded.valid[:, None], encoded.ir + scale_delta, 0)
    rows = mixed + gia_uma_dov(wally, mixed)
    return read_heads(wally, encoded, rows, frame, statistics, load, balance)


def strategy(wally: Wally, *values, participant_fusion_scale=None, residual_fusion_scale=None, execute_remote_scale=True):
    frame = ChorusArrays(*values)
    strength = mx.array(getattr(wally, 'participant_fusion_scale', 1.0)
        if participant_fusion_scale is None else participant_fusion_scale, dtype=frame.state.dtype)
    row_encode = getattr(wally, 'row_encode', lambda *args: encode(wally, *args))
    row_decode = getattr(wally, 'row_decode', lambda *args: decode(wally, *args))
    encoded = row_encode(*values, strength)
    scale_delta, statistics, load, balance = scale_fuse(wally, encoded.ir, execute_remote=execute_remote_scale,
        residual_fusion_scale=residual_fusion_scale, valid=encoded.valid)
    return row_decode(encoded, scale_delta, statistics, load, balance, frame)


def logp_of(out, velocity, row_mask=None):
    return rate_logp(out.rate, velocity, row_mask)


def act(out, key):
    return sample(out.rate, key)
