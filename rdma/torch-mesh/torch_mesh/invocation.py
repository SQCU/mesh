"""Persistent whole-sequence feedback at a service's invocation boundary."""
import hashlib
import json
from pathlib import Path

import torch.distributed as dist

from controller import WholeFunctionController
from . import partition


class Invocation:
    """Restore placement before operands are built; observe only the full call.

    The bootstrap store carries configuration/state and completed elapsed times.
    It carries no tensor-consumption receipts and supplies no page-reuse proof.
    A proposal is persisted for the next invocation; it never changes a running
    sequence or ranks alternatives by accepting/rejecting a measured candidate.
    """
    serial = 0

    def __init__(self, mesh, name, workload, state_path):
        self.mesh, self.name, self.workload = mesh, name, workload
        self.group, self.path = mesh.get_group(name), Path(state_path)
        self.rank, self.size = self.group.rank(), self.group.size()
        self.store = dist.distributed_c10d._get_default_store()
        Invocation.serial += 1
        identity = hashlib.sha256(str(self.path).encode()).hexdigest()[:16]
        self.key = f'mesh/invocation/{self.group.group_name}/{identity}/{Invocation.serial}'
        if self.rank == 0:
            self.store.set(self.key + '/state', self.path.read_bytes() if self.path.exists() else b'null')
        state = json.loads(self.store.get(self.key + '/state'))
        self.controller = WholeFunctionController.of(state) if state is not None else None
        self.proposal, self.preparations = None, None
        if self.controller is not None:
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
            topology = self.controller.topology
            canonical = lambda value: json.dumps(value, sort_keys=True)
            if (topology is None or topology['nodes'] != self.size
                    or canonical(topology['cables']) != canonical(self.group.fabric.cables)):
                raise ValueError('stored invocation routes describe another physical fabric')
            self.controller.topology['demands'] = [[a, b, count] for (a, b), count in self.group.plan_demands().items()]
        self.proposal = self.controller.proposal()
        self.group.reset_fused_configuration()
        self.controller.install_routes(self.group.comm, self.proposal)
        partition._replan(self.mesh, self.name)
        self.group.plan_active(False)
        self.preparations = prepare
        if prepare is not None:
            prepare()
        return self.proposal

    def begin(self):
        if self.proposal is None:
            raise RuntimeError('realize the complete invocation before starting it')
        self.group.plan_active(True)

    def completed(self, seconds, **evidence):
        """Submit completed rank-max latency, then persist the next configuration."""
        self.group.plan_active(False)
        self.store.set(f'{self.key}/seconds/{self.rank}', json.dumps(float(seconds)).encode())
        durations = [float(self.store.get(f'{self.key}/seconds/{rank}')) for rank in range(self.size)]
        next_proposal = self.controller.completed(max(durations), proposal=self.proposal,
                                                   finishes={'ranks': durations}, **evidence)
        if self.rank == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + '.tmp')
            temporary.write_text(json.dumps(self.controller.state(), sort_keys=True) + '\n')
            temporary.replace(self.path)
        return {'executed_proposal': self.proposal['id'], 'seconds': max(durations),
                'next_proposal': next_proposal['id'], 'state': str(self.path)}
