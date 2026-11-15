#!/usr/bin/env python3
# design/the-compiler.md, order 3; design/unequal-shards-change-request.md items 8 and 9
import os

import torch
import torch.nn.functional as F

from parallel import Attention, DIM, arguments, execute


SEQ, VOCAB = (int(os.environ.get(k, v)) for k, v in (('SEQ', 8192), ('VOCAB', 8192)))
STRIPED = ('x', 'ids', 'positions', 'targets')


# Brandon et al., Striped Attention, arXiv:2311.09431, section 2.2.1
class Causal(torch.nn.Module):
    def __init__(self, sequence=SEQ):
        super().__init__()
        self.tokens, self.positions = torch.nn.Embedding(VOCAB, DIM), torch.nn.Embedding(sequence, DIM)
        self.attention, self.head = Attention(), torch.nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, ids, positions, targets):
        h = self.tokens(ids) + self.positions(positions)
        logits = self.head(h + self.attention(h))
        return F.cross_entropy(logits.transpose(1, 2), targets), logits.detach()


def placement(processors, units='striped', sequence=None):
    def choose(node):
        if node.name in STRIPED:
            return 1, processors, units
        if node.target is torch.ops.aten.bmm.default:
            return 1 if sequence in (None, node.meta['val'].shape[1]) else 2, processors, units
        return processors[0]
    return choose


def main():
    args = arguments('A causal language model forward and backward, its tokens, positions and targets split over '
                     'processors in stripes of the sequence.')
    torch.manual_seed(0)
    ids, targets = torch.randint(0, VOCAB, (1, SEQ)), torch.randint(0, VOCAB, (1, SEQ))
    inputs = ids, torch.arange(SEQ)[None], targets
    # compile-time callback (callable supplied to execute(Causal().eval(), inputs, lambda processors: placement(processors, sequence=SEQ), a): callback inversion of control is outside the working
    # specification, and runtime callback overhead is inadmissible in performance-critical code.
    return execute(Causal().eval(), inputs, lambda processors: mesh_callbacks_are_outside_spec(placement(processors, sequence=SEQ)), args, loss=0)


if __name__ == '__main__':
    main()
