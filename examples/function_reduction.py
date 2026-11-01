"""A supplied reduction compiled into the ordinary collective-heavy function.

The caller supplies its communicator and existing page operands. Preparation
captures the function and records it; the returned Function is its only runtime.
"""
import torch

from torch_mesh.declarations import Function
from torch_mesh.reduction import Reduction


def prepare(group, values, destination, *, topology=None, configuration=None):
    # Each row is one reduction value. This callable is captured here and does
    # not run from a callback while values are being processed.
    combine = Reduction(torch.add, 0, dtype=values.dtype,
                        element_shape=tuple(values.shape[1:]),
                        reassociate=True, topology=topology)

    def source(x):
        return combine(x, group)

    function = Function(source, values, configuration=configuration)
    calls = function.declare(group, destinations=(destination,))
    group.plan(calls)
    function.prepare()
    return function
