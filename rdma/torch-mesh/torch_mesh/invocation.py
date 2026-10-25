"""Persistent whole-sequence feedback at a service's invocation boundary."""
import json
from pathlib import Path

import torch.distributed as dist

import controller as policy
from controller import WholeFunctionController
from mesh_source import revision
from rollout import RolloutStats
from . import partition


class Invocation:
    """Restore placement before operands are built; observe only the full call.

    Bootstrap realizes configuration. The asynchronous statistics worker observes
    completed whole functions and persists the next configuration. Numerical
    execution never gathers elapsed times.
    """
    serial = 0

    def __init__(self, mesh, name, workload, state_path):
        self.mesh, self.name, self.workload = mesh, name, workload
        self.implementation = revision(Path(policy.__file__).resolve().parents[1])
        self.group, self.path = mesh.get_group(name), Path(state_path).expanduser()
        self.rank, self.size = self.group.rank(), self.group.size()
        self.store = dist.distributed_c10d._get_default_store()
        Invocation.serial += 1
        self.key = f'mesh/invocation/{self.group.group_name}/{name}/{Invocation.serial}'
        if self.rank == 0:
            self.store.set(self.key + '/state', self.path.read_bytes() if self.path.exists() else b'null')
        state = json.loads(self.store.get(self.key + '/state'))
        self.controller = WholeFunctionController.of(state) if state is not None else None
        self.proposal, self.preparations = None, None
        self.sequence, self.feedback = 0, None
        if self.controller is not None:
            self.controller.bind_implementation(self.implementation)
            if self.controller.workload != workload:
                raise ValueError('stored invocation state describes another complete function')
            parts = self.controller.proposal()['parts'][name]
            operand = partition.operand(mesh, name)
            if len(parts) != self.size or any(int(p) != p or p < 0 for p in parts) or sum(parts) != operand.units:
                raise ValueError('stored invocation placement does not fit this function and communicator')
            partition._attach(mesh, name, list(map(int, parts)))

    def realize(self, groups, prepare=None):
        """Declare the complete input-to-output graph and install its route vector."""
        partition.plan(self.mesh, self.name, groups)
        if self.controller is None:
            operand = partition.operand(self.mesh, self.name)
            decisions = {self.name: {'parts': operand.parts.tolist(), 'grain': 1,
                                    'low': [0] * self.size, 'high': [operand.units] * self.size}}
            self.controller = WholeFunctionController.from_comm(decisions, self.group.comm, workload=self.workload)
        else:
            self.controller = self.controller.bind_fabric(self.group.comm)
        self.controller.bind_implementation(self.implementation)
        self.proposal = self.controller.proposal()
        self.group.reset_fused_configuration()
        self.controller.install_routes(self.group.comm, self.proposal)
        partition._replan(self.mesh, self.name)
        self.group.plan_active(False)
        self.preparations = prepare
        if prepare is not None:
            prepare()
        self.feedback = RolloutStats(self.store, self.key + '/rollouts', self.rank, self.size,
                                     self.controller, state_path=self.path)
        workers = getattr(self.group, '_rollout_workers', None)
        if workers is None:
            self.group._rollout_workers = workers = []
        workers.append(self.feedback)
        return self.proposal

    def begin(self):
        if self.proposal is None:
            raise RuntimeError('realize the complete invocation before starting it')
        self.group.plan_active(True)

    def completed(self, seconds, **evidence):
        """Enqueue this rank's evidence without accessing the store or peers."""
        self.group.plan_active(False)
        episode = f'{self.proposal["id"]}:{self.sequence}'
        self.sequence += 1
        self.feedback.submit([{'episode': episode, 'proposal': self.proposal,
                               'workload': self.workload, 'seconds': float(seconds), 'metrics': evidence}])
        return {'executed_proposal': self.proposal['id'], 'local_seconds': seconds,
                'episode': episode, 'proposal': self.proposal, 'metrics': evidence, 'state': str(self.path)}

    def run(self, call, *, calls=1, **evidence):
        """Measure a complete service call through the shared sequence runner."""
        from .benchmarks import sequence
        self.begin()
        measured = sequence(call, calls, 1)
        measured['feedback'] = self.completed(measured['host_ns'] * 1e-9, **evidence)
        return measured
