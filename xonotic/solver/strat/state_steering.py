from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch

from .policy_math import control_log_scale

STATE_HEADER = dict(ID=0, SCHEMA=1, ENGINE_TIME=2, APPLIED_SEQUENCE=3,
    ENTITY=4, OFFSET=5, GENERATION=6, OWNER_GENERATION=7, SESSION_HI=8, SESSION_LO=9,
    COMMAND_SOURCE_TIME=10, COMMAND_DURATION=11, COMMAND_TAU=12, RECEIVED_SEQUENCE=13, FORCING=14)
READ_ONLY = 1 << 21


def advance_key(key, destination):
    np.copyto(destination, key)
    low = int(key[1]) + 1
    key[1] = low & 0xffffffff
    key[0] = (int(key[0]) + (low >> 32)) & 0xffffffff


def writable_words(layout, module=torch):
    tags = layout[..., 3:]
    if module is np or isinstance(tags, np.ndarray):
        return (tags > 0) & ((tags.astype(np.uint32) & READ_ONLY) == 0)
    return (tags > 0) & ((tags.to(torch.int64) & READ_ONLY) == 0)

class StateRate(NamedTuple):
    mean: torch.Tensor
    log_scale: torch.Tensor
    density: torch.Tensor
    present: torch.Tensor | None = None
    factor: torch.Tensor | None = None


def relaxation_coefficients(delta, tau):
    elapsed = torch.as_tensor(delta) / torch.as_tensor(tau)
    return torch.exp(-elapsed), -torch.as_tensor(tau) * torch.expm1(-elapsed)


def integrate_anticipatory(residual, fast_state, velocity, delta, tau, lead=0.0, tau_fast=None):
    decay, gain = relaxation_coefficients(delta, tau)
    slow_next = decay * residual + gain * velocity
    if not lead:
        fast_next = torch.zeros_like(slow_next) if fast_state is None else fast_state
        return slow_next, slow_next, fast_next
    tf = torch.as_tensor(0.25 * tau if tau_fast is None else tau_fast, dtype=slow_next.dtype, device=slow_next.device)
    elapsed_f = torch.as_tensor(delta, dtype=slow_next.dtype, device=slow_next.device) / tf
    m_decay = torch.exp(-elapsed_f)
    m_gain = float(lead) * (-torch.expm1(-elapsed_f))
    prev_fast = torch.zeros_like(slow_next) if fast_state is None else fast_state
    fast_next = m_decay * prev_fast + m_gain * velocity
    return slow_next + fast_next, slow_next, fast_next


def integrate(residual, velocity, delta, tau, lead=0.0, fast_state=None, tau_fast=None):
    imposed, _, _ = integrate_anticipatory(residual, fast_state, velocity, delta, tau, lead, tau_fast)
    return imposed


def _spd_cholesky_logdet_solve(matrix, rhs, eps=1e-6):
    k = matrix.shape[-1]
    cols = []
    for j in range(k):
        if j == 0:
            diag = torch.sqrt(torch.clamp(matrix[..., 0, 0], min=eps))
            col = [diag] + [matrix[..., i, 0] / diag for i in range(1, k)]
        else:
            prev = torch.stack(cols, dim=-1)
            diag_sq = matrix[..., j, j] - (prev[..., j, :] * prev[..., j, :]).sum(dim=-1)
            diag = torch.sqrt(torch.clamp(diag_sq, min=eps))
            col = [torch.zeros_like(diag) for _ in range(j)] + [diag]
            for i in range(j + 1, k):
                dot = (prev[..., i, :] * prev[..., j, :]).sum(dim=-1)
                col.append((matrix[..., i, j] - dot) / diag)
        cols.append(torch.stack(col, dim=-1))
    lower = torch.stack(cols, dim=-1)
    logdet = 2.0 * torch.diagonal(lower, dim1=-2, dim2=-1).log().sum(dim=-1)
    ys = []
    for i in range(k):
        acc = rhs[..., i]
        for j in range(i):
            acc = acc - lower[..., i, j] * ys[j]
        ys.append(acc / lower[..., i, i])
    xs = [None] * k
    for i in range(k - 1, -1, -1):
        acc = ys[i]
        for j in range(i + 1, k):
            acc = acc - lower[..., j, i] * xs[j]
        xs[i] = acc / lower[..., i, i]
    return logdet, torch.stack(xs, dim=-1)


def rate_distribution(coordinates, layout, participant_present, k_cov=0):
    width = layout.shape[-1] - 3
    shape = (layout.shape[0], layout.shape[1] * width)
    present = (writable_words(layout) & participant_present[:, None, None]).reshape(shape)
    mean = (0.01 * coordinates[..., :width]).reshape(shape)
    log_scale = (math.log(0.01) + control_log_scale(coordinates[..., width:2 * width])).reshape(shape)
    factor = None
    if k_cov > 0:
        raw_factor = (0.01 / math.sqrt(max(1, shape[1] * k_cov))) * coordinates[..., 2 * width:(2 + k_cov) * width]
        factor = raw_factor.reshape(shape[0], shape[1], k_cov)
        factor = torch.where(present.unsqueeze(-1), factor, 0.0)
    return StateRate(torch.where(present, mean, 0.0), torch.where(present, log_scale, 0.0),
        participant_present.to(mean.dtype), present, factor)


def policy_inputs(frame, width):
    shape = (*frame.layout.shape[:2], width)
    present = frame.layout[..., 3:] > 0
    native = tuple(torch.where(present, value.reshape(shape), 0.0)
        for value in (frame.state, frame.residual, frame.word_read_times, frame.held_velocity, frame.command_residual))
    timing = torch.stack((frame.delta.reshape(()), frame.relaxation_time.reshape(()))).to(frame.state.dtype)
    pages = torch.cat((*native[:2], frame.layout, timing.expand(*shape[:-1], 2),
        frame.page_headers, *native[2:]), dim=-1)
    marker = torch.cat((frame.observations, timing.expand(frame.observations.shape[0], 2)), dim=-1)
    return pages, marker, present


def logp_of(rate, velocity, active=None):
    active = rate.density > 0 if active is None else active
    present = torch.ones_like(rate.mean) if rate.present is None else rate.present.to(rate.mean.dtype)
    mass = present * active.to(rate.mean.dtype)[:, None]
    scale = torch.where(mass > 0, rate.log_scale, 0.0)
    delta = torch.where(mass > 0, velocity - rate.mean, 0.0)
    if rate.factor is None:
        error = delta * torch.exp(-scale)
        terms = error.square() + 2 * scale + math.log(2 * math.pi)
        return -0.5 * (terms * mass).sum(dim=-1) * rate.density
    inv_var = torch.where(mass > 0, torch.exp(-2.0 * scale), 0.0)
    factor = rate.factor * mass.unsqueeze(-1)
    k = factor.shape[-1]
    eye = torch.eye(k, dtype=rate.mean.dtype, device=rate.mean.device)
    scaled_f = factor * inv_var.unsqueeze(-1)
    capacitance = eye + (factor.unsqueeze(-1) * scaled_f.unsqueeze(-2)).sum(dim=-3)
    u = delta * inv_var
    ft_u = (factor * u.unsqueeze(-1)).sum(dim=-2)
    logdet_m, solved = _spd_cholesky_logdet_solve(capacitance, ft_u)
    quad = (delta * u).sum(dim=-1) - (ft_u * solved).sum(dim=-1)
    logdet_d = (2.0 * scale * mass).sum(dim=-1)
    dim = mass.sum(dim=-1)
    row_active = active.to(rate.mean.dtype)
    return -0.5 * (quad + logdet_d + logdet_m * row_active + dim * math.log(2 * math.pi)) * rate.density


def normal(shape, key, device):
    """A standard normal of `shape` from the 64-bit sampling key (the uint64 counter schedule of advance_key): drawn
    on the host's generator, so every rank given the key draws the same values."""
    generator = torch.Generator().manual_seed((int(key[0]) << 32) | int(key[1]))
    return torch.randn(tuple(shape), generator=generator).to(device)


def sample(rate, key):
    present = torch.ones_like(rate.mean) if rate.present is None else rate.present.to(rate.mean.dtype)
    active = (rate.density > 0).to(rate.mean.dtype)[:, None] * present
    eps1 = normal(rate.mean.shape, key, rate.mean.device)
    raw = rate.mean + torch.exp(torch.where(active > 0, rate.log_scale, 0.0)) * eps1
    if rate.factor is not None:
        key2 = (int(key[0]) ^ 0x9e3779b9, (int(key[1]) + 0x85ebca6b) & 0xffffffff)
        eps2 = normal((rate.mean.shape[0], rate.factor.shape[-1]), key2, rate.mean.device)
        raw = raw + (rate.factor * eps2.unsqueeze(-2)).sum(dim=-1)
    value = torch.where(active > 0, raw, 0.0)
    return value, logp_of(rate, value)


PAGE_WIDTH = 256
PAGE_STATE_HEADER = len(STATE_HEADER)
PAGE_RESPONSE_HEADER = 12
PAGE_STATE_WIDTH = PAGE_STATE_HEADER + 6 * PAGE_WIDTH


class StatePages(NamedTuple):
    headers: np.ndarray
    state: np.ndarray
    residual: np.ndarray
    layout: np.ndarray
    page_headers: np.ndarray
    locations: np.ndarray
    word_read_times: np.ndarray
    held_velocity: np.ndarray
    command_residual: np.ndarray

    def response(self, velocity, duration, relaxation_time, lead=0.0, fast_state=None):
        source = self.page_headers
        metadata = np.column_stack((source[:, 0:3], np.full(len(source), duration),
                                    np.full(len(source), relaxation_time), source[:, 4:10], source[:, 3]))
        residual_pages = self.residual.reshape(len(self.state), -1, PAGE_WIDTH)
        velocity_pages = np.asarray(velocity).reshape(residual_pages.shape)
        writable = writable_words(self.layout, np)
        if lead:
            tf = max(1e-6, 0.25 * float(relaxation_time))
            m_decay = math.exp(-float(duration) / tf)
            m_gain = float(lead) * (-math.expm1(-float(duration) / tf))
            prev_m = np.zeros_like(residual_pages) if fast_state is None else np.asarray(fast_state).reshape(residual_pages.shape)
            lead_term = m_decay * prev_m + m_gain * velocity_pages
        else:
            lead_term = 0.0
        residual_pages = np.where(writable, residual_pages + lead_term, 0)
        velocity_pages = np.where(writable, velocity_pages, 0)
        bots, pages = self.locations.T
        return np.concatenate((metadata, residual_pages[bots, pages], velocity_pages[bots, pages]), axis=-1).astype(np.float32)


def unpack_pages(rows, participant_ids):
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != PAGE_STATE_WIDTH or not np.isfinite(rows).all():
        raise ValueError('native view frame requires finite typed state pages')
    selected = rows[np.isin(rows[:, 0], participant_ids)]
    identities = np.unique(selected[:, 0]).astype(np.int64)
    groups = [selected[selected[:, 0] == identity] for identity in identities]
    counts = [len(group) for group in groups]
    pages = max(counts, default=0)
    state = np.zeros((len(groups), pages, PAGE_WIDTH), dtype=np.float32)
    residual = np.zeros_like(state)
    word_read_times = np.zeros_like(state)
    held_velocity = np.zeros_like(state)
    command_residual = np.zeros_like(state)
    layout = np.zeros((len(groups), pages, 3 + PAGE_WIDTH), dtype=np.float32)
    headers, page_headers, locations = [], [], []
    for bot, group in enumerate(groups):
        group = group[np.lexsort((group[:, 5], group[:, 4]))]
        if len(np.unique(group[:, [1, 2, 3, 7, 8, 9, 10, 11, 12, 13]], axis=0)) != 1:
            raise ValueError('native view pages disagree about their source snapshot')
        if len(np.unique(group[:, [4, 5]], axis=0)) != len(group):
            raise ValueError('native view frame repeats a memory page')
        count = len(group)
        headers.append(group[0, :PAGE_STATE_HEADER])
        page_headers.extend(group[:, :PAGE_STATE_HEADER])
        locations.extend((bot, page) for page in range(count))
        state[bot, :count] = group[:, PAGE_STATE_HEADER:PAGE_STATE_HEADER + PAGE_WIDTH]
        residual[bot, :count] = group[:, PAGE_STATE_HEADER + PAGE_WIDTH:PAGE_STATE_HEADER + 2 * PAGE_WIDTH]
        layout[bot, :count, :3] = group[:, 4:7]
        layout[bot, :count, 3:] = group[:, PAGE_STATE_HEADER + 2 * PAGE_WIDTH:PAGE_STATE_HEADER + 3 * PAGE_WIDTH]
        word_read_times[bot, :count] = group[:, PAGE_STATE_HEADER + 3 * PAGE_WIDTH:PAGE_STATE_HEADER + 4 * PAGE_WIDTH]
        held_velocity[bot, :count] = group[:, PAGE_STATE_HEADER + 4 * PAGE_WIDTH:PAGE_STATE_HEADER + 5 * PAGE_WIDTH]
        command_residual[bot, :count] = group[:, PAGE_STATE_HEADER + 5 * PAGE_WIDTH:]
    return StatePages(np.asarray(headers, dtype=np.float32).reshape(-1, PAGE_STATE_HEADER), state.reshape(len(groups), pages * PAGE_WIDTH),
                      residual.reshape(len(groups), pages * PAGE_WIDTH), layout,
                      np.asarray(page_headers, dtype=np.float32).reshape(-1, PAGE_STATE_HEADER), np.asarray(locations, dtype=np.int64).reshape(-1, 2),
                      word_read_times.reshape(len(groups), pages * PAGE_WIDTH), held_velocity.reshape(len(groups), pages * PAGE_WIDTH),
                      command_residual.reshape(len(groups), pages * PAGE_WIDTH))


def decode_state_words(state, tags):
    state, tags = np.asarray(state, dtype=np.float32), np.asarray(tags, dtype=np.uint32)
    encoded = (tags & (1 << 20)) != 0
    bits = (((tags >> 3) & 0xffff) << 16) | np.where(encoded, state, 0).astype(np.uint32)
    return np.where(encoded, bits, state.view(np.uint32)).astype(np.uint32), tags & 7


def state_labels(layout):
    labels = []
    for page in np.asarray(layout):
        entity, offset, generation = map(int, page[:3])
        arena = 'global' if entity < 0 else f'entity.{entity}.generation.{generation}'
        labels.extend(f'{arena}.word.{offset + i}' if tag > 0 else '' for i, tag in enumerate(page[3:]))
    return labels


def latent_labels(layout, width, page_capacity):
    labels = [f'participant.ir.{index}' for index in range(width)]
    for page in range(page_capacity):
        present = page < len(layout) and np.any(layout[page, 3:] > 0)
        address = '.'.join(str(int(value)) for value in layout[page, :3]) if present else ''
        labels.extend(f'page.{address}.ir.{index}' if present else '' for index in range(width))
    return labels
