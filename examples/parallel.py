# design/the-compiler.md, order 3
import os

import torch
import torch.nn.functional as F


DIM, HEADS, FFN, LAYERS = (int(os.environ.get(k, v)) for k, v in
    (('DIM', 2048), ('HEADS', 16), ('FFN', 5632), ('LAYERS', 4)))


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = DIM // HEADS
        self.wq, self.wk, self.wv, self.wo = (torch.nn.Linear(DIM, DIM, bias=False) for _ in range(4))

    def forward(self, x):
        batch, sequence, _ = x.shape
        def heads(weight):
            return weight(x).view(batch, sequence, HEADS, self.head_dim).transpose(1, 2)
        q, k, v = (heads(weight) for weight in (self.wq, self.wk, self.wv))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).reshape(batch, sequence, DIM))


class FeedForward(torch.nn.Module):
    def __init__(self, width=FFN):
        super().__init__()
        self.w1, self.w3 = torch.nn.Linear(DIM, width, bias=False), torch.nn.Linear(DIM, width, bias=False)
        self.w2 = torch.nn.Linear(width, DIM, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm, self.ffn_norm = torch.nn.RMSNorm(DIM), torch.nn.RMSNorm(DIM)
        self.attention, self.feed_forward = Attention(), FeedForward()

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        return h + self.feed_forward(self.ffn_norm(h))


class Training(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(*(Block() for _ in range(LAYERS)))

    def forward(self, x):
        out = self.model(x)
        return out.sum(), out.detach()


# M. Shoeybi et al., Megatron-LM, arXiv:1909.08053, section 3 and Figure 3; design/unequal-shards-change-request.md
# items 1, 5, 6 and 15
def megatron(processors, columns, rows, units={}):
    def placement(node):
        if node.op == 'placeholder':
            for dimension, layers in enumerate((columns, rows)):
                if any(node.target.endswith(f'_{layer}_weight') for layer in layers):
                    return dimension, processors, next((u for part, u in units.items() if f'_{part}_' in node.target),
                                                       None)
        return processors[0]
    return placement


def placement(processors, units={}):
    return megatron(processors, ('wq', 'wk', 'wv', 'w1', 'w3'), ('wo', 'w2'), units)
