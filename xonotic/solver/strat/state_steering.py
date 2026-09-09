from __future__ import annotations

import math
from typing import NamedTuple

from . import tensor as mx
import numpy as np

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


def writable_words(layout, module=mx):
    tags = layout[..., 3:]
    return (tags > 0) & ((tags.astype(module.uint32) & READ_ONLY) == 0)

class StateRate(NamedTuple):
    mean: mx.array
    log_scale: mx.array
    density: mx.array
    present: mx.array | None = None


def relaxation_coefficients(delta, tau):
    elapsed = mx.array(delta) / mx.array(tau)
    return mx.exp(-elapsed), -mx.array(tau) * mx.expm1(-elapsed)


def integrate(residual, velocity, delta, tau):
    decay, gain = relaxation_coefficients(delta, tau)
    return decay * residual + gain * velocity


def rate_distribution(coordinates, layout, participant_present):
    width = layout.shape[-1] - 3
    shape = (layout.shape[0], layout.shape[1] * width)
    present = (writable_words(layout) & participant_present[:, None, None]).reshape(shape)
    mean = (0.01 * coordinates[..., :width]).reshape(shape)
    log_scale = (math.log(0.01) + control_log_scale(coordinates[..., width:])).reshape(shape)
    return StateRate(mx.where(present, mean, 0), mx.where(present, log_scale, 0),
        participant_present.astype(mean.dtype), present)


def policy_inputs(frame, width):
    shape = (*frame.layout.shape[:2], width)
    present = frame.layout[..., 3:] > 0
    native = tuple(mx.where(present, value.reshape(shape), 0)
        for value in (frame.state, frame.residual, frame.word_read_times, frame.held_velocity, frame.command_residual))
    timing = mx.stack((mx.array(frame.delta), mx.array(frame.relaxation_time)))
    pages = mx.concatenate((*native[:2], frame.layout, mx.broadcast_to(timing, (*shape[:-1], 2)),
        frame.page_headers, *native[2:]), axis=-1)
    marker = mx.concatenate((frame.observations, mx.broadcast_to(timing, (frame.observations.shape[0], 2))), axis=-1)
    return pages, marker, present


def logp_of(rate, velocity, active=None):
    active = rate.density > 0 if active is None else active
    present = mx.ones_like(rate.mean) if rate.present is None else rate.present
    mass = present * active[:, None]
    scale = mx.where(mass > 0, rate.log_scale, 0)
    error = mx.where(mass > 0, velocity - rate.mean, 0) * mx.exp(-scale)
    terms = mx.square(error) + 2 * scale + math.log(2 * math.pi)
    return -0.5 * mx.sum(terms * mass, axis=-1) * rate.density


def sample(rate, key):
    present = mx.ones_like(rate.mean) if rate.present is None else rate.present
    active = (rate.density > 0)[:, None] * present
    value = mx.where(active, rate.mean + mx.exp(mx.where(active, rate.log_scale, 0)) *
                     mx.random.normal(rate.mean.shape, key=key), 0)
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

    def response(self, velocity, duration, relaxation_time):
        source = self.page_headers
        metadata = np.column_stack((source[:, 0:3], np.full(len(source), duration),
                                    np.full(len(source), relaxation_time), source[:, 4:10], source[:, 3]))
        residual_pages = self.residual.reshape(len(self.state), -1, PAGE_WIDTH)
        velocity_pages = np.asarray(velocity).reshape(residual_pages.shape)
        writable = writable_words(self.layout, np)
        residual_pages = np.where(writable, residual_pages, 0)
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
