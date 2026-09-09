import time
from typing import NamedTuple

import numpy as np

from .inputs import ChorusArrays, capacity_for, frame_shapes, fill_frame
from .strategy import Strategy
from .state_steering import StateRate, writable_words
from .storage import TensorStorage
from .persistent_policy import PersistentPolicy


class PolicyEmission(NamedTuple):
    outputs: dict
    comparisons: dict
    velocity: np.ndarray
    logp: np.ndarray
    residual: np.ndarray
    preparation_seconds: float
    inference_seconds: float


class InputArena:
    def __init__(self):
        self.capacity = (1,) * 10
        self.slots = []
        self.allocations = self.loads = 0

    def reserve(self, *frames, count, capacity=(1,) * 10):
        realized = tuple(max(a, b, c) for a, b, c in zip(self.capacity, capacity, capacity_for(*frames)))
        self.capacity = realized
        shapes = frame_shapes(frames[0], self.capacity)
        while len(self.slots) < count:
            self.slots.append((TensorStorage(), np.empty((0, 0), dtype=bool)))
        for slot, (storage, scratch) in enumerate(self.slots):
            before = storage.allocations
            storage.realize({name: (shape, value.dtype)
                for name, shape, value in zip(frames[0]._fields, shapes, frames[0])})
            if scratch.shape != shapes.neighborhood_indices:
                scratch = np.empty(shapes.neighborhood_indices, dtype=bool)
                self.allocations += 1
            self.slots[slot] = storage, scratch
            self.allocations += storage.allocations - before

    def load(self, frame, slot=0):
        storage, scratch = self.slots[slot]
        fill_frame(ChorusArrays(**storage.host), frame, scratch)
        self.loads += 1
        return ChorusArrays(**storage.values)

    def stage(self, *frames, capacity=(1,) * 10):
        unique = dict((id(frame), frame) for frame in frames)
        self.reserve(*frames, count=len(unique), capacity=capacity)
        return {identity: self.load(frame, slot) for slot, (identity, frame) in enumerate(unique.items())}

    def report(self):
        return {'slots': len(self.slots), 'allocations': self.allocations, 'loads': self.loads,
            'bytes': sum(storage.nbytes + scratch.nbytes for storage, scratch in self.slots)}


def prepare_policies(programs, frame):
    started = time.perf_counter()
    owner = next(iter(programs.values()))
    capacity = tuple(map(max, zip(*(program.capacity for program in programs.values()))))
    values = owner.inputs.stage(frame, capacity=capacity)[id(frame)]
    for program in programs.values():
        program.bind_capacity(owner.inputs.capacity)
        program.reserve(frame)
    shape = values.state.shape
    if owner.command is None:
        owner.command = TensorStorage()
    owner.command.realize({'velocity': (shape, np.float32), 'residual': (shape, np.float32),
        'logp': ((shape[0],), np.float32)})
    return values, owner.command, time.perf_counter() - started


def emit_policies(programs, prepared, frame, arms, key, recorded=None):
    values, command, preparation = prepared
    started = time.perf_counter()
    outputs, comparisons = {}, {}
    n, width = frame.state.shape
    velocity, residual = (command.host[name][:n, :width] for name in ('velocity', 'residual'))
    logp = command.host['logp'][:n]
    for value in command.host.values():
        value.fill(0)
    for arm, program in programs.items():
        output, sampled, likelihood, integrated = program.emit(values, frame, key)
        sampled, likelihood, integrated = map(np.asarray, (sampled, likelihood, integrated))
        selected = arms == arm
        velocity[selected], logp[selected], residual[selected] = sampled[selected], likelihood[selected], integrated[selected]
        outputs[arm] = output
        comparisons[arm] = {'rate_mean': np.asarray(output.rate.mean), 'rate_log_scale': np.asarray(output.rate.log_scale),
            'velocity': sampled, 'residual': integrated, 'present': np.asarray(output.rate.present),
            'density': np.asarray(output.rate.density), 'winner_value': np.asarray(output.value_winnie),
            'loser_value': np.asarray(output.value_lou)}
    if recorded is not None:
        np.copyto(velocity, recorded['velocity'])
        np.copyto(logp, recorded['behavior_logp'])
        arm = next(iter(programs))
        integrated = programs[arm].native.integrate(values, command.values['velocity'])
        np.copyto(residual, np.asarray(integrated)[:n, :width])
    writable = writable_words(frame.layout, np).reshape(n, width)
    np.copyto(velocity, 0, where=~writable)
    np.copyto(residual, 0, where=~writable)
    return PolicyEmission(outputs, comparisons, velocity, logp, residual,
        preparation, time.perf_counter() - started)


class PolicyProgram:
    def __init__(self, model, forward, learner=None):
        self.model, self.forward = model, forward
        self.capacity = (1,) * 10
        self.inputs = InputArena()
        self.command = None
        self.capacity_changes = 0
        self.external = any(getattr(model, name, None) is not None for name in ('scale_executor', 'cross_executor'))
        self.native = PersistentPolicy(model, forward, learner)
        self.learner = learner

    def bind_capacity(self, capacity):
        self.capacity_changes += self.capacity != tuple(capacity)
        self.capacity = tuple(capacity)

    def reserve(self, frame):
        if self.learner is not None:
            self.learner.realize_items(frame)
        self.native.reserve(frame, self.capacity)

    def infer(self, values, frame):
        return self.crop(self.native.infer(values), frame)

    def emit(self, values, frame, key):
        out, sampled, likelihood, integrated = self.native.emit(values, key)
        n, width = frame.state.shape
        return self.crop(out, frame), sampled[:n, :width].copy(), likelihood[:n].copy(), integrated[:n, :width].copy()

    def crop(self, out, frame):
        n, width = frame.state.shape
        pages = 1 + frame.layout.shape[1]
        latent_width = out.ir.shape[-1] // (1 + self.capacity[1])
        ir = out.ir[:n].reshape(n, 1 + self.capacity[1], latent_width)[:, :pages].reshape(n, pages * latent_width)
        return Strategy(StateRate(out.rate.mean[:n, :width].copy(), out.rate.log_scale[:n, :width].copy(),
            out.rate.density[:n].copy(), out.rate.present[:n, :width].copy()), ir.copy(),
            out.value_winnie[:n].copy(), out.value_lou[:n].copy(), out.coupling[:n, :pages].copy(), out.local_neighborhood[:n].copy(),
            out.scale_residual_stats.copy(), out.scale_expert_load.copy(), out.scale_balance.copy())

    def report(self):
        return {'capacity': dict(zip(('participants', 'state_pages', 'events', 'observations', 'carts', 'teams', 'navigation_nodes', 'navigation_edges', 'navigation_cells', 'neighbors'), self.capacity)),
                'capacity_changes': self.capacity_changes, 'input_storage': self.inputs.report(),
                **self.native.report()}
