#!/usr/bin/env python3
# docs/measurement.md#the-torch-parallel-runs
import os

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as collective
import torch.nn.functional as F
import torch_mesh
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard
from torch_mesh import partition
from torch_mesh.configuration import placement, bind, record
from mesh_statistics import now

from tp import DIM, STEPS, drawn

FFN, EXPERTS, TOP_K, TOKENS = 2816, 8, 2, 4096
SLOTS = TOKENS * TOP_K * 5 // (4 * EXPERTS)


class MoE:
    # docs/measurement.md#the-torch-parallel-runs
    def __init__(self, mesh):
        with torch.device("meta"):
            tokens, router = torch.empty(TOKENS, DIM), torch.empty(EXPERTS, DIM)
            experts = torch.empty(EXPERTS, FFN, DIM), torch.empty(EXPERTS, FFN, DIM), torch.empty(EXPERTS, DIM, FFN)
        shard = lambda t: drawn(partition.distribute_tensor(t, mesh, [Shard(0)], src_data_rank=None))
        self.mesh, self.x, self.router, self.w = mesh, shard(tokens), drawn(router), [shard(w) for w in experts]
        self.sent, self.back = (partition.empty(mesh, "ep", (EXPERTS, SLOTS, DIM), 1) for _ in range(2))
        self.received, self.results = (partition.empty(mesh, "ep", (EXPERTS, SLOTS, DIM), 0) for _ in range(2))
        self.combined = partition.empty(mesh, "ep", (TOKENS, DIM), 0)

    def __call__(self):
        x, w = self.x.to_local(), [w.to_local() for w in self.w]
        sent, back, received, results = (t.to_local().view(-1, DIM) for t in (self.sent, self.back, self.received, self.results))
        mine, slots = len(w[0]), self.sent.to_local().shape[1]
        out, into = partition.split(self.mesh, "ep", EXPERTS * slots), partition.split(self.mesh, "ep", mine * SLOTS)
        weights, experts = (x @ self.router.mT).softmax(-1).topk(TOP_K, dim=-1)
        counts = F.one_hot(experts.flatten(), EXPERTS)
        place = counts.cumsum(0).gather(1, experts.view(-1, 1)).view_as(experts) - 1
        kept = place < slots
        slot = torch.where(kept, experts * slots + place, counts.sum(0).argmin() * slots + slots - 1)
        sent[slot] = x[:, None].expand(-1, TOP_K, -1)
        received = collective.wait_tensor(collective.all_to_all_single(sent, into, out, self.mesh.get_group("ep")))
        w1, w3, w2 = (t.mT for t in w)
        for got, made in zip(received.split(into), results.split(into)):
            got, made = got.view(mine, -1, DIM), made.view(mine, -1, DIM)
            gate, up = torch.bmm(got, w1), torch.bmm(got, w3)
            made.copy_(torch.bmm(F.silu(gate) * up, w2))
        back = collective.wait_tensor(collective.all_to_all_single(results, out, into, self.mesh.get_group("ep")))
        rows = torch.index_select(back, 0, slot.flatten()).view(-1, TOP_K, DIM)
        return torch.sum(rows.mul_((weights / weights.sum(-1, keepdim=True) * kept)[..., None]), 1, out=self.combined.to_local())


def main():
    dist.init_process_group(backend=os.environ.get("BACKEND", "mesh"))
    mesh = DeviceMesh.from_group(dist.group.WORLD, "mps", mesh_dim_names=("ep",))
    partition.attach(mesh, ep=EXPERTS)
    configuration = placement(mesh, 'ep',
        {'function': 'moe_dispatch_experts_combine', 'steps': STEPS, 'tokens': TOKENS, 'width': DIM,
         'experts': EXPERTS, 'ffn': FFN, 'top_k': TOP_K, 'slots': SLOTS},
        os.environ.get('MESH_OBSERVATIONS', '~/.cache/mesh/observations/ep.jsonl'))
    with torch.no_grad():
        from torch_mesh.declarations import Function, execution_sequence
        model = MoE(mesh)
        function = Function(model, configuration=configuration['assignment']['operands'])
        program = execution_sequence([function] * STEPS)
        bind(configuration, lambda: function.declare(mesh.get_group("ep")), program.realize)
        started = now()
        program.forward()
        completion = program.completion()
        record(configuration, completion, started, STEPS * TOKENS * DIM * model.x.element_size(),
               {'program': 'ep', 'steps': STEPS, 'parts': configuration['assignment']['parts']['ep']})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
