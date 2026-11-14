#!/usr/bin/env python3
# design/parallel-layers-example.md; design/the-compiler.md, order 3
import os

import torch
import torch.nn.functional as F

from parallel import arguments, chosen, execute, megatron


DIM, HEADS, FF_MULT, LAYERS, BATCH, SEQ, VOCAB = (int(os.environ.get(k, v)) for k, v in
    (('DIM', 2048), ('HEADS', 16), ('FF_MULT', 4), ('LAYERS', 4), ('BATCH', 2), ('SEQ', 1024), ('VOCAB', 8192)))
ROTARY_BASE = 1000


# dff_sparsify legacy/pgpt/pgptlformer.py rotarizer and apply_rotarizer_emb
class Rotary(torch.nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        inverse = (ROTARY_BASE ** (torch.arange(0, head_dim, 2).float() / head_dim)) ** -1
        angles = torch.outer(torch.arange(SEQ).float(), inverse)
        self.register_buffer('cos', angles.cos()[None, :, None, :], persistent=False)
        self.register_buffer('sin', angles.sin()[None, :, None, :], persistent=False)

    def forward(self, x):
        half = x.shape[3] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * self.cos + x2 * self.sin, x1 * -self.sin + x2 * self.cos], 3)


# dff_sparsify legacy/pgpt/pgptlformer.py vit22_tformer.self_attn
class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = DIM // HEADS
        self.queryproj, self.keyproj, self.valueproj = (torch.nn.Linear(DIM, DIM, bias=False) for _ in range(3))
        self.attnoutproj = torch.nn.Linear(DIM, DIM, bias=True)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        batch, sequence, _ = x.shape
        query, key, value = (proj(x).view(batch, sequence, HEADS, self.head_dim)
                             for proj in (self.queryproj, self.keyproj, self.valueproj))
        query, key = self.rotary(query), self.rotary(key)
        y = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                                           is_causal=True)
        return self.attnoutproj(y.transpose(1, 2).reshape(batch, sequence, DIM))


# dff_sparsify legacy/pgpt/pgptlformer.py vit22_tformer.feedfor and swiglu
class FeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.denseproj_in = torch.nn.Linear(DIM, DIM * FF_MULT, bias=True)
        self.denseproj_gate = torch.nn.Linear(DIM, DIM * FF_MULT, bias=True)
        self.denseproj_out = torch.nn.Linear(DIM * FF_MULT, DIM, bias=True)

    def forward(self, x):
        return self.denseproj_out(F.silu(self.denseproj_gate(x)) * self.denseproj_in(x))


# dff_sparsify legacy/pgpt/pgptlformer.py vit22_tformer.forward; B. Wang and A. Komatsuzaki, GPT-J-6B, 2021,
# mesh-transformer-jax mesh_transformer/layers.py TransformerLayerShard.__call__; A. Chowdhery et al., PaLM,
# arXiv:2204.02311, section 2, "Parallel Layers"
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layerwisenorm = torch.nn.RMSNorm(DIM, elementwise_affine=False)
        self.attention, self.feed_forward = Attention(), FeedForward()
        self.learnedlambda = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, h_states):
        inner_states = self.layerwisenorm(h_states)
        attn_out = self.attention(inner_states)
        dense_out = self.feed_forward(inner_states)
        return h_states * self.learnedlambda + dense_out + attn_out


# dff_sparsify legacy/pgpt/pgptlformer.py PGPT_Lformer.forward
class Inference(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = torch.nn.Embedding(VOCAB, DIM)
        self.blocks = torch.nn.Sequential(*(Block() for _ in range(LAYERS)))
        self.tokenpicker_head = torch.nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, input_ids):
        h_states = F.rms_norm(self.embedder(input_ids), (DIM,))
        h_states = F.rms_norm(self.blocks(h_states), (DIM,))
        logits = self.tokenpicker_head(h_states[:, -1:, :])
        return 30 * torch.tanh(logits / 30)


# parallel.py megatron
def placement(processors, units={}):
    return megatron(processors, ('queryproj', 'keyproj', 'valueproj', 'denseproj_in', 'denseproj_gate'),
                    ('attnoutproj', 'denseproj_out'), units)


def main():
    args = arguments('Inference of a randomly initialized language model whose blocks add attention and '
                     'feed-forward outputs computed from one normalized input, its weights split over processors.')
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))
    model = Inference().eval()
    given = chosen(model, (input_ids,), placement, args, HEADS)
    return execute(model, (input_ids,), lambda processors: placement(processors, given), args, units=given)


if __name__ == '__main__':
    main()
