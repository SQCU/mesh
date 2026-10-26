# design/whole-function.md
import numpy as np
import torch
import torch_mesh
from . import _stream, fused
from .engines import host

class DNN:
    """docs/measurement.md#physical-fabric-dnn"""

    def __init__(self, args, group, fabric):
        self.args, self.group = args, group
        self.channels = len(fabric.adjacent[group.rank()])
        B, H, K, L = args.rows, args.width, args.expansion, args.layers
        self.input = torch_mesh.empty(B, H)
        rng = np.random.default_rng(1729)
        rng.standard_normal(out=host(self.input).numpy(), dtype=np.float32)
        self.weights = []
        for _ in range(L):
            a, b = torch_mesh.empty(H, K, device='cpu'), torch_mesh.empty(K, H, device='cpu')
            rng.standard_normal(out=a.numpy(), dtype=np.float32)
            rng.standard_normal(out=b.numpy(), dtype=np.float32)
            a.mul_(H ** -0.5); b.mul_(K ** -0.5)
            self.weights.append((a, b))
        self.x = torch_mesh.empty(B, H)
        self.partial = [torch_mesh.empty(B, H) for _ in range(L)]
        self.pairs = [(a, b) for a in range(group.size()) for b in range(group.size()) if a != b]
        self.landing = [[torch_mesh.empty(B, H) for _ in self.pairs] for _ in range(L)]
        peers = group.size() - 1
        inputs = ', '.join(f'volatile coherent(system) device const float *p{k} [[buffer({k + 1})]]' for k in range(peers))
        addition = ' + '.join(['own[i]'] + [f'mesh_payload(p{k} + i)' for k in range(peers)])
        vectors = ' + '.join(['((device const float4 *)own)[i / 4]'] +
                             [f'mesh_payload((volatile coherent(system) device const float4 *)p{k} + i / 4)' for k in range(peers)])
        fold_args = 'struct FoldArgs { uint n, pad; mesh_step_binding land[CHANNELS], publish[CHANNELS]; };'
        fold_wait = r'''
    for (uint k = 0; k < CHANNELS; k++) if (s.land[k].p) {
        auto b = s.land[k];
        mesh_step_enter(*b.p, b.v, b.progress, tid == 0);
        if (local == 0) mesh_step_counts(*b.p, b.a, b.v);
    }
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE));
'''
        source = fused.steps() + f'#define CHANNELS {self.channels}\n' + fold_args + f'''
kernel void fold(device const float *own [[buffer(0)]], {inputs}, device float *out [[buffer({peers + 1})]],
                 constant FoldArgs &s [[buffer({peers + 2})]], uint tid [[thread_position_in_grid]],
                 uint local [[thread_position_in_threadgroup]], uint stride [[threads_per_grid]]) {{
    {fold_wait}
    for (uint i = 4 * tid; i < s.n; i += 4 * stride) {{
        if (i + 3 < s.n) ((device float4 *)out)[i / 4] = tanh({vectors});
        else for (; i < s.n; i++) out[i] = tanh({addition});
    }}
}}
'''
        self.fold = _stream.pipeline(source, 'fold')
        self.arrivals = [torch.zeros(1, dtype=torch.int32, device='mps') for _ in range(L)]
        self.parts = None
        self.tiles = [(first, min(args.tile_rows, B - first)) for first in range(0, B, args.tile_rows)]

    def prepare(self, parts):
        rank, B, H = self.group.rank(), self.args.rows, self.args.width
        share = parts['compute']
        if share != self.parts:
            first, width = sum(share[:rank]), share[rank]
            self.local = []
            for a, b in self.weights:
                ag, bg = torch_mesh.empty(H, width), torch_mesh.empty(width, H)
                ag.copy_(a[:, first:first + width]); bg.copy_(b[first:first + width, :])
                self.local.append((ag, bg))
            self.hidden = torch_mesh.empty(B, width)
            self.parts = list(share)
        schedule = _stream.Schedule()
        for first, rows in self.tiles:
            x = self.input.narrow(0, first, rows)
            result = self.x.narrow(0, first, rows)
            hidden = self.hidden.narrow(0, first, rows)
            for layer, (a, b) in enumerate(self.local):
                partial = self.partial[layer].narrow(0, first, rows)
                landings = [t.narrow(0, first, rows) for t in self.landing[layer]]
                crossing = fused.writes(schedule, partial, landings, self.pairs, self.group)
                from .matrix import product
                schedule.append(_stream.Commands([
                    product(x, a, hidden, act=True),
                    product(hidden, b, partial, publication=crossing, arrivals=self.arrivals[layer])]))
                inputs = [t for (_, destination), t in zip(self.pairs, landings) if destination == rank]
                output = len(inputs) + 1
                schedule.compute(self.fold, [partial, *inputs, result],
                                 fused._encode_words([result.numel(), 0], self.channels, crossing, None),
                                 output + 1, (result.numel() + 1023) // 1024, 1, 256,
                                 ((1 << output) - 1) | (1 << (32 + output)))
                x = result
        schedule.finish()
        self.schedule = schedule
        self.completion = schedule.completion()
        self.run = schedule.enqueue

    def declare(self):
        from .backend import Call, WRITE
        return [([Call(WRITE, rows * self.args.width, torch.float32, root=a, destination=b)
                  for a, b in self.pairs], self.args.layers) for _, rows in self.tiles]
