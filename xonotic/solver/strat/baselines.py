from __future__ import annotations

from . import tensor as mx
import mlx.nn as nn

from .inputs import ChorusArrays
from .neighborhood import LocalNeighborhood
from .state_steering import StateRate, writable_words
from payload.tools.strategy_io_schema import OBS_WIDTH
from .strategy import Strategy, Encoded, embedded_rows, read_heads


class BaselinePolicy(nn.Module):
    def __init__(self, arm, state_width, context_width, hidden_width=256):
        super().__init__()
        self.arm = arm
        self.state_width = state_width
        self.quinn = nn.Linear(6 * state_width + 20, hidden_width)
        self.phil = nn.Linear(context_width, hidden_width)
        self.participant = nn.Linear(OBS_WIDTH + 2, hidden_width)
        self.cart = nn.Linear(18, hidden_width)
        self.team = nn.Linear(7, hidden_width)
        self.neighborhood = LocalNeighborhood(hidden_width, gram=False)
        self.heads = nn.Linear(hidden_width, 2 * state_width + 2, bias=False)


def baseline_strategy(policy, *values):
    frame = ChorusArrays(*values)
    hidden, valid, local = embedded_rows(policy, frame, policy.state_width, nonlinear=False)
    global_context = mx.sum(hidden, axis=0)
    hidden = hidden + global_context[None, :]
    hidden = hidden * mx.sigmoid(hidden) if policy.arm == 'ffn' else hidden
    hidden = mx.where(valid[:, None], hidden, 0)
    page_rows = mx.arange(frame.observations.shape[0], frame.observations.shape[0] + frame.layout.shape[0] * frame.layout.shape[1]).reshape(frame.layout.shape[:2])
    encoded = Encoded(hidden, mx.zeros((hidden.shape[0], 0)), valid, frame.participant_rows,
        page_rows, local[frame.participant_rows])
    return read_heads(policy, encoded, hidden, frame, mx.zeros(0), mx.zeros(0), mx.sum(mx.zeros(0)))


def default_strategy(policy, *values):
    frame = ChorusArrays(*values)
    zeros = mx.zeros_like(frame.state)
    values = mx.zeros(frame.state.shape[0])
    present = writable_words(frame.layout).reshape(frame.state.shape)
    rate = StateRate(zeros, zeros, values, present)
    return Strategy(rate, mx.zeros((frame.state.shape[0], 0)), values, values,
        mx.zeros((frame.state.shape[0], 1 + frame.layout.shape[1], 0)), mx.zeros((frame.state.shape[0], 0)),
        mx.zeros(0), mx.zeros(0), mx.sum(mx.zeros(0)))
