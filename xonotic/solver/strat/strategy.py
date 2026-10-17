from __future__ import annotations

from typing import NamedTuple

import torch

from .cast_header import Wally, encode_rows, gia_uma_dov, ir_query, norm, scale_fuse
from .inputs import ChorusArrays
from .matmul import matrix_multiply, matrix_multiply_transpose_left, linear
from .state_steering import StateRate, rate_distribution, logp_of as rate_logp, sample, policy_inputs


class Strategy(NamedTuple):
    rate: StateRate
    ir: torch.Tensor
    value_winnie: torch.Tensor
    value_lou: torch.Tensor
    coupling: torch.Tensor
    local_neighborhood: torch.Tensor
    scale_residual_stats: torch.Tensor
    scale_expert_load: torch.Tensor
    scale_balance: torch.Tensor


class Encoded(NamedTuple):
    ir: torch.Tensor
    coupling: torch.Tensor
    valid: torch.Tensor
    own_rows: torch.Tensor
    page_rows: torch.Tensor
    local: torch.Tensor


def embedded_rows(model, frame, width, nonlinear=True):
    pages, observations, present = policy_inputs(frame, width)
    valid = torch.cat((frame.observation_present,
        (present.any(dim=-1) & frame.participant_present[:, None]).reshape(-1),
        frame.cart_present, frame.team_present))
    rows = torch.cat((linear(model.participant, observations), linear(model.quinn, pages).reshape(-1, model.participant.weight.shape[0]),
        linear(model.cart, frame.carts), linear(model.team, frame.teams)))
    source_valid = torch.cat((frame.context_present, frame.navigation_node_present,
        frame.navigation_edge_present, frame.navigation_cell_present))
    sources = torch.cat((linear(model.phil, frame.context),
        model.neighborhood.encode(frame.navigation_nodes, frame.navigation_edges, frame.navigation_cells)))
    sources = torch.where(source_valid[:, None], torch.asinh(sources) if nonlinear else sources, 0.0)
    rows = torch.where(valid[:, None], torch.asinh(rows) if nonlinear else rows, 0.0)
    participant_rows = frame.participant_rows.long()
    owner_context = rows[participant_rows, None, :].expand(*pages.shape[:2], rows.shape[-1]).reshape(-1, rows.shape[-1])
    rows = rows + torch.cat((torch.zeros_like(rows[:observations.shape[0]]), owner_context,
        rows.new_zeros((frame.carts.shape[0] + frame.teams.shape[0], rows.shape[-1]))))
    indices = frame.neighborhood_indices.long()
    local = model.neighborhood(rows[:observations.shape[0]], sources, indices,
        frame.neighborhood_distances, frame.neighborhood_present & source_valid[indices], frame.neighborhood_radius,
        source_times=torch.cat((frame.context[:, 1], rows.new_zeros((sources.shape[0] - frame.context.shape[0],)))),
        observer_times=frame.observations[:, 21],
        source_temporal=torch.arange(sources.shape[0], device=rows.device) < frame.context.shape[0], gram=nonlinear)
    local = torch.where(frame.observation_present[:, None], local, 0.0)
    row_local = torch.cat((local, local[participant_rows, None, :].expand(*pages.shape[:2], local.shape[-1]).reshape(-1, local.shape[-1]),
        local.new_zeros((frame.carts.shape[0] + frame.teams.shape[0], local.shape[-1]))))
    return torch.where(valid[:, None], rows + (torch.asinh(row_local) if nonlinear else row_local), 0.0), valid, local


def row_gram_context(model, teams, rivals, values, frame, global_context):
    n, count = frame.layout.shape[:2]
    global_values = matrix_multiply(rivals, global_context) / model.w.r_e ** .5
    o, p, c = frame.observations.shape[0], n * count, frame.carts.shape[0]
    spans = (slice(0, o), slice(o, o + p), slice(o + p, o + p + c), slice(o + p + c, None))
    observation, pages, carts, team_rows = (teams[span] for span in spans)
    observation_values, page_values, cart_values, team_values = (values[span] for span in spans)
    grouped = torch.cat((observation[:, :, None] * observation_values[:, None, :],
        matrix_multiply_transpose_left(pages.reshape(n, count, model.w.r), page_values.reshape(n, count, model.w.d_ir)),
        carts[:, :, None] * cart_values[:, None, :], team_rows[:, :, None] * team_values[:, None, :]))
    labels = torch.cat((frame.observations[:, 1], frame.team_ids.to(values.dtype), frame.carts[:, 3], frame.teams[:, 0]))
    membership = (labels[:, None] == frame.teams[None, :, 0]) & frame.team_present[None, :]
    team_context = matrix_multiply(membership.to(values.dtype).T, grouped.reshape(grouped.shape[0], -1))
    contexts = matrix_multiply(membership.to(values.dtype), team_context).reshape(grouped.shape)
    outputs = torch.cat(((observation[:, :, None] * contexts[:o]).sum(dim=1),
        matrix_multiply(pages.reshape(n, count, model.w.r), contexts[o:o + n]).reshape(p, model.w.d_ir),
        (carts[:, :, None] * contexts[o + n:o + n + c]).sum(dim=1),
        (team_rows[:, :, None] * contexts[o + n + c:]).sum(dim=1)))
    return global_values + outputs / model.w.r ** .5, torch.cat((teams, rivals), dim=-1)


def encode_source(wally, *values):
    frame = ChorusArrays(*values)
    rows, valid, local = embedded_rows(wally, frame, wally.w.d_x)
    query = torch.where(valid[:, None], encode_rows(wally, rows), 0.0)
    projected = ir_query(wally, query)
    normalized = norm(query)
    return projected, local, valid, linear(wally.team_metric, normalized), linear(wally.rival_metric, normalized), norm(projected)


def encode_finish(wally, source, values, strength, global_context):
    frame = ChorusArrays(*values)
    projected, local, valid, teams, rivals, payload = source
    fused, coupling = row_gram_context(wally, teams, rivals, payload, frame, global_context)
    ir = torch.where(valid[:, None], projected + norm(fused) * strength, 0.0)
    o = frame.observations.shape[0]
    pages = torch.arange(o, o + frame.layout.shape[0] * frame.layout.shape[1], device=ir.device).reshape(frame.layout.shape[:2])
    participant_rows = frame.participant_rows.long()
    return Encoded(ir, coupling, valid, participant_rows, pages, local[participant_rows])


def encode(wally, *values):
    source = encode_source(wally, *values[:-1])
    cross = getattr(wally, 'cross_executor', matrix_multiply_transpose_left)
    return encode_finish(wally, source, values[:-1], values[-1], cross(*source[-2:]))


def read_heads(model, encoded, rows, frame, statistics, load, balance):
    own = torch.cat((encoded.own_rows[:, None], encoded.page_rows), dim=1)
    valid = encoded.valid[own] & frame.participant_present[:, None]
    ir = torch.where(valid[..., None], rows[own], 0.0)
    projected = linear(model.heads, ir)
    coordinates = projected[:, 1:, :-2] + projected[:, :1, :-2]
    values = projected[..., -2:].sum(dim=1)
    rate = rate_distribution(coordinates, frame.layout, frame.participant_present)
    return Strategy(rate, ir.reshape(frame.state.shape[0], -1), values[:, 0], values[:, 1],
        torch.where(valid[..., None], encoded.coupling[own], 0.0),
        encoded.local * frame.participant_present[:, None].to(encoded.local.dtype), statistics, load, balance)


def decode(wally, encoded, scale_delta, statistics, load, balance, frame):
    mixed = torch.where(encoded.valid[:, None], encoded.ir + scale_delta, 0.0)
    rows = mixed + gia_uma_dov(wally, mixed)
    return read_heads(wally, encoded, rows, frame, statistics, load, balance)


def strategy(wally: Wally, *values, participant_fusion_scale=None, residual_fusion_scale=None, execute_remote_scale=True):
    frame = ChorusArrays(*values)
    strength = torch.tensor(getattr(wally, 'participant_fusion_scale', 1.0)
        if participant_fusion_scale is None else participant_fusion_scale, dtype=frame.state.dtype, device=frame.state.device)
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
