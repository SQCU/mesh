#!/usr/bin/env python3
# design/the-compiler.md, order 3; Lepikhin et al., GShard, arXiv:2006.16668, Algorithm 1
import os

import torch
import torch.nn.functional as F

from tp import arguments, execute


DIM, FFN, EXPERTS, TOP_K, TOKENS = (int(os.environ.get(k, v)) for k, v in
    (('DIM', 2048), ('FFN', 2816), ('EXPERTS', 8), ('TOP_K', 2), ('TOKENS', 4096)))


class Dispatch(torch.nn.Module):
    def __init__(self, participants, rows, slots):
        super().__init__()
        self.participants, self.rows, self.slots = participants, rows, slots
        self.router = torch.nn.Parameter(torch.randn(EXPERTS, DIM) / DIM ** 0.5)

    def forward(self, x):
        weights, experts = (x @ self.router.mT).softmax(-1).topk(TOP_K, dim=-1)
        ids = experts.flatten(1)
        valid = (torch.arange(self.participants * self.rows, device=x.device) < TOKENS).view(
            self.participants, self.rows, 1).expand(-1, -1, TOP_K).flatten(1)
        counts = F.one_hot(ids, EXPERTS) * valid.unsqueeze(-1)
        place = counts.cumsum(1).gather(2, ids.unsqueeze(-1)).squeeze(-1) - 1
        kept = valid & (place >= 0) & (place < self.slots)
        slots = torch.where(kept, ids * self.slots + place, EXPERTS * self.slots)
        values = x.unsqueeze(2).expand(-1, -1, TOP_K, -1).flatten(1, 2) * kept.unsqueeze(-1)
        sent = x.new_zeros(self.participants, EXPERTS * self.slots + 1, DIM).scatter_add(
            1, slots.unsqueeze(-1).expand(-1, -1, DIM), values)
        packed = sent[:, :EXPERTS * self.slots].reshape(self.participants, EXPERTS, self.slots, DIM).permute(1, 0, 2, 3)
        weights = weights / weights.sum(-1, keepdim=True) * kept.view(self.participants, self.rows, TOP_K)
        return packed, slots, weights


class Experts(torch.nn.Module):
    def __init__(self, participants):
        super().__init__()
        self.participants = participants
        self.w1 = torch.nn.Parameter(torch.randn(EXPERTS, FFN, DIM) / DIM ** 0.5)
        self.w3 = torch.nn.Parameter(torch.randn(EXPERTS, FFN, DIM) / DIM ** 0.5)
        self.w2 = torch.nn.Parameter(torch.randn(EXPERTS, DIM, FFN) / FFN ** 0.5)

    def forward(self, received):
        results = []
        for p in range(self.participants):
            got = received[:, p]
            gate, up = torch.bmm(got, self.w1.mT), torch.bmm(got, self.w3.mT)
            results.append(torch.bmm(F.silu(gate) * up, self.w2.mT))
        return torch.stack(results, 1)


class MoE(torch.nn.Module):
    def __init__(self, participants):
        super().__init__()
        self.participants = participants
        self.rows = (TOKENS + participants - 1) // participants
        self.slots = max(1, self.rows * TOP_K * 5 // (4 * EXPERTS))
        self.dispatch = Dispatch(participants, self.rows, self.slots)
        self.experts = Experts(participants)

    def forward(self, x):
        packed, slots, weights = self.dispatch(x)
        back = self.experts(packed).permute(1, 0, 2, 3).reshape(self.participants, EXPERTS * self.slots, DIM)
        back = torch.cat((back, back.new_zeros(self.participants, 1, DIM)), 1)
        selected = back.gather(1, slots.unsqueeze(-1).expand(-1, -1, DIM))
        result = (selected.view(self.participants, self.rows, TOP_K, DIM) * weights.unsqueeze(-1)).sum(2)
        return result.flatten(0, 1)[:TOKENS]


# design/unequal-shards-change-request.md item 9
def placement(processors, slots):
    def choose(node):
        if node.name == 'x' or node.name.startswith('p_experts_'):
            return 0, processors, None
        value = node.meta.get('val')
        if isinstance(value, torch.Tensor) and tuple(value.shape) == (EXPERTS, len(processors), slots, DIM):
            modules = [{path for path, _ in n.meta.get('nn_module_stack', {}).values()} for n in (node, *node.users)]
            for dimension, name in enumerate(('dispatch', 'experts')):
                if name in modules[0] and not any(name in read for read in modules[1:]):
                    return dimension, processors, None
        return processors[0]
    return choose


def main():
    args = arguments('Capacity-limited top-k routing, expert all-to-all, batched FFN and weighted combine, the tokens '
                     'and the experts split over processors.')
    torch.manual_seed(0)
    model = MoE(len(args.nodes)).eval()
    x = torch.randn(len(args.nodes), model.rows, DIM) / DIM ** 0.5
    return execute(model, (x,), lambda processors: placement(processors, model.slots), args)


if __name__ == '__main__':
    main()
