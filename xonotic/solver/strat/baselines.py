from __future__ import annotations

import torch
from torch import nn

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
        self.sharded = frozenset()


def baseline_strategy(policy, *values):
    frame = ChorusArrays(*values)
    hidden, valid, local = embedded_rows(policy, frame, policy.state_width, nonlinear=False)
    global_context = hidden.sum(dim=0)
    hidden = hidden + global_context[None, :]
    hidden = hidden * torch.sigmoid(hidden) if policy.arm == 'ffn' else hidden
    hidden = torch.where(valid[:, None], hidden, 0.0)
    o = frame.observations.shape[0]
    page_rows = torch.arange(o, o + frame.layout.shape[0] * frame.layout.shape[1], device=hidden.device).reshape(frame.layout.shape[:2])
    participant_rows = frame.participant_rows.long()
    encoded = Encoded(hidden, hidden.new_zeros((hidden.shape[0], 0)), valid, participant_rows,
        page_rows, local[participant_rows])
    zero = hidden.new_zeros(0)
    return read_heads(policy, encoded, hidden, frame, zero, zero, zero.sum())


def default_strategy(policy, *values):
    frame = ChorusArrays(*values)
    zeros = torch.zeros_like(frame.state)
    values = frame.state.new_zeros(frame.state.shape[0])
    present = writable_words(frame.layout).reshape(frame.state.shape)
    rate = StateRate(zeros, zeros, values, present)
    zero = frame.state.new_zeros(0)
    return Strategy(rate, frame.state.new_zeros((frame.state.shape[0], 0)), values, values,
        frame.state.new_zeros((frame.state.shape[0], 1 + frame.layout.shape[1], 0)), frame.state.new_zeros((frame.state.shape[0], 0)),
        zero, zero, zero.sum())
