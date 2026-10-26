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
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3));
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
        # The producer stores its result once, through the existing transport
        # payload accessor. No staging copy or per-operation host wait.
        product = fused.steps() + f"#define CHANNELS {self.channels}\n#define FUSED 1\n" + r"""
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
struct ProductArgs {
    uint4 shape;
#if FUSED
    mesh_step_binding land[CHANNELS], publish[CHANNELS];
#endif
};
kernel void product(device float *X [[buffer(0)]], device float *W [[buffer(1)]], device uchar *Y [[buffer(2)]],
                    device atomic_uint *arrivals [[buffer(3)]], constant ProductArgs &s [[buffer(4)]],
                    uint3 tg [[threadgroup_position_in_grid]], uint3 lane [[thread_position_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint local = lane.x;
    const int M=s.shape.x, K=s.shape.y, N=s.shape.z;
#if FUSED
    for (uint k = 0; k < CHANNELS; k++) if (s.publish[k].p) {
        auto b = s.publish[k];
        mesh_step_enter(*b.p, b.v, b.progress, tg.x == 0 && tg.y == 0 && local == 0);
    }
#endif
    const int row=int(tg.x)*64+(sg/2)*32, col=int(tg.y)*64+(sg%2)*32;
    auto A=tensor(X, dextents<int,2>{K,M}, array<int,2>{1,K});
    auto B=tensor(W, dextents<int,2>{N,K}, array<int,2>{1,N});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(A),decltype(B),float>();
    for(uint i=0;i<acc.get_capacity();i++)acc[i]=0;
    for(int at=0;at<K;at+=32){auto a=A.slice<32,32>(at,row);auto b=B.slice<32,32>(col,at);op.run(a,b,acc);}
    for(uint i=0;i<acc.get_capacity();i++)if(acc.is_valid_element(i)){
        auto e=acc.get_multidimensional_index(i);
        if(row+e[1]<M && col+e[0]<N)mesh_step_write<float>(Y,(row+e[1])*N+col+e[0],ACTIVATE ? tanh(acc[i]) : acc[i]);
    }
#if FUSED
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3));
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    threadgroup uint last[1];
    if (local == 0) last[0] = atomic_fetch_add_explicit(arrivals, 1u, memory_order_relaxed) == uint(((M+63)/64)*((N+63)/64)-1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (last[0]) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3));
        for (uint k = 0; k < CHANNELS; k++) if (s.publish[k].p)
            mesh_step_release(*s.publish[k].p, s.publish[k].a, local);
        if (local == 0) atomic_store_explicit(arrivals, 0u, memory_order_relaxed);
    }
#endif
}
"""
        self.arrivals = [torch.zeros(1, dtype=torch.int32, device='mps') for _ in range(L)]
        self.product = _stream.pipeline('#define ACTIVATE 0\n' + product, 'product')
        self.activation = _stream.pipeline('#define ACTIVATE 1\n' + product.replace('#define FUSED 1', '#define FUSED 0'), 'product')
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
                schedule.compute(self.activation, [x, a, hidden, self.arrivals[layer]],
                                 [rows, self.args.width, hidden.shape[1], 0], 4,
                                 (rows + 63) // 64, (hidden.shape[1] + 63) // 64, 128,
                                 3 | (1 << (32 + 2)))
                schedule.compute(self.product, [hidden, b, partial, self.arrivals[layer]],
                                 fused._encode_words([rows, hidden.shape[1], self.args.width, 0], self.channels, None, crossing),
                                 4, (rows + 63) // 64, (self.args.width + 63) // 64, 128,
                                 3 | (12 << 32))
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
