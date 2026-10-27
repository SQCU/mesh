# design/whole-function.md
import numpy as np
import torch
import torch_mesh
from . import _stream, fused
from .engines import host
from .matrix import product, row_tiles

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
        fold_wait = r'''
    constant mesh_step_binding *land = (constant mesh_step_binding *)(s + 4) + tile * CHANNELS;
    for (uint k = 0; k < CHANNELS; k++) if (land[k].p) {
        auto b = land[k];
        mesh_step_enter(*b.p, b.v, b.progress, tid == 0 && local == 0);
        if (local == 0) mesh_step_counts(*b.p, b.a, b.v);
    }
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE));
'''
        source = fused.steps() + f'#define CHANNELS {self.channels}\n' + f'''
kernel void fold(device const float *own [[buffer(0)]], {inputs}, device float *out [[buffer({peers + 1})]],
                 constant uint *s [[buffer({peers + 2})]], uint2 pos [[thread_position_in_grid]],
                 uint local [[thread_index_in_threadgroup]], uint2 grid [[threads_per_grid]],
                 uint2 group [[threadgroup_position_in_grid]]) {{
    const uint tid = pos.x, stride = grid.x, tile = group.y;
    const uint begin = tile * s[1] * s[2], end = min(s[0], begin + s[1] * s[2]);
    {fold_wait}
    for (uint i = begin + 4 * tid; i < end; i += 4 * stride) {{
        if (i % 4 == 0 && i + 3 < end) ((device float4 *)out)[i / 4] = tanh({vectors});
        else {{
            const uint stop = min(i + 4, end);
            for (uint j = i; j < stop; j++) {{ const uint i = j; out[i] = tanh({addition}); }}
        }}
    }}
}}
'''
        self.fold = _stream.pipeline(source, 'fold')
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
                tiles = row_tiles(rows)
                ready = torch_mesh.empty(len(tiles), dtype=torch.int64)
                arrivals = torch_mesh.empty(len(tiles), dtype=torch.int32)
                host(ready).zero_(); host(arrivals).zero_()
                words = [partial.numel(), H, tiles[0][1], 0]
                for index, (at, count) in enumerate(tiles):
                    crossing = fused.writes(schedule, partial.narrow(0, at, count),
                        [t.narrow(0, at, count) for t in landings], self.pairs, self.group,
                        ready=ready.narrow(0, index, 1), ready_value=count * H * partial.element_size())
                    words += [w for k in range(self.channels) for w in fused._words(
                        crossing.land[k] if k < len(crossing.land) else fused._NONE)]
                schedule.append(_stream.Commands([
                    product(x, a, hidden, act=True),
                    product(hidden, b, partial, ready=ready, arrivals=arrivals)]))
                inputs = [t for (_, destination), t in zip(self.pairs, landings) if destination == rank]
                output = len(inputs) + 1
                schedule.compute(self.fold, [partial, *inputs, result],
                                 words, output + 1, (tiles[0][1] * H + 1023) // 1024, len(tiles), 256,
                                 ((1 << output) - 1) | (1 << (32 + output)))
                x = result
        schedule.finish()
        self.schedule = schedule
        self.completion = schedule.completion()
        self.run = schedule.enqueue

    def declare(self):
        from .backend import Call, WRITE
        return [([Call(WRITE, count * self.args.width, torch.float32, root=a, destination=b)
                  for a, b in self.pairs], 1) for _, rows in self.tiles for _ in range(self.args.layers)
                for _, count in row_tiles(rows)]
