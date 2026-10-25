# design/whole-function.md
import numpy as np
import torch
import torch_mesh
from . import _stream, fused

class DNN:
    """docs/measurement.md#physical-fabric-dnn"""

    def __init__(self, args, group, fabric):
        self.args, self.group = args, group
        self.channels = len(fabric.adjacent[group.rank()])
        B, H, K, L = args.rows, args.width, args.expansion, args.layers
        self.input = torch_mesh.empty(B, H, device='cpu')
        rng = np.random.default_rng(1729)
        rng.standard_normal(out=self.input.numpy(), dtype=np.float32)
        self.negative_input = torch_mesh.empty(B, H, device='cpu')
        torch.neg(self.input, out=self.negative_input)
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
        fold_args = 'struct FoldArgs { uint n, pad; };'
        ready_args = 'struct ReadyArgs { uint n, pad; mesh_step_binding land[CHANNELS], publish[CHANNELS]; };'
        fold_wait = r'''
    for (uint k = 0; k < CHANNELS; k++) if (s.land[k].p) {
        auto b = s.land[k];
        mesh_step_enter(*b.p, b.v, b.progress, tid == 0);
        if (local == 0) mesh_step_counts(*b.p, b.a, b.v);
    }
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3));
'''
        source = fused.steps() + f'#define CHANNELS {self.channels}\n' + fold_args + ready_args + f'''
kernel void land_ready(constant ReadyArgs &s [[buffer(0)]], uint tid [[thread_position_in_grid]],
                       uint local [[thread_position_in_threadgroup]]) {{
    {fold_wait}
}}
kernel void fold(device const float *own [[buffer(0)]], {inputs}, device float *out [[buffer({peers + 1})]],
                 constant FoldArgs &s [[buffer({peers + 2})]], uint tid [[thread_position_in_grid]],
                 uint local [[thread_position_in_threadgroup]], uint stride [[threads_per_grid]]) {{
    for (uint i = 4 * tid; i < s.n; i += 4 * stride) {{
        if (i + 3 < s.n) ((device float4 *)out)[i / 4] = tanh({vectors});
        else for (; i < s.n; i++) out[i] = tanh({addition});
    }}
}}
'''
        self.fold = _stream.pipeline(source, 'fold')
        self.land_ready = _stream.pipeline(source, 'land_ready')
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
        if(row+e[1]<M && col+e[0]<N)mesh_step_write<float>(Y,(row+e[1])*N+col+e[0],acc[i]);
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
        self.product = _stream.pipeline(product, 'product')
        self.parts = None

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
        self.crossings = [fused.writes(partial, landings, self.pairs, self.group)
                          for partial, landings in zip(self.partial, self.landing)]

    def run(self):
        self.x.copy_(self.input)
        for layer, (a, b) in enumerate(self.local):
            partial, landings, crossing = self.partial[layer], self.landing[layer], self.crossings[layer]
            torch.mm(self.x, a, out=self.hidden)
            self.hidden.tanh_()
            fused.encode(self.product, [self.hidden, b, partial, self.arrivals[layer]],
                         [self.args.rows, self.hidden.shape[1], self.args.width, 0], 4,
                         (self.args.rows + 63) // 64, (self.args.width + 63) // 64, 128,
                         self.channels, publish=crossing)
            inputs = [t for (_, destination), t in zip(self.pairs, landings) if destination == self.group.rank()]
            fused.encode(self.land_ready, [], [self.x.numel(), 0], 0, 1, 1, 32,
                         self.channels, land=crossing)
            _stream.encode(self.fold, [partial, *inputs, self.x], [self.x.numel(), 0], len(inputs) + 2,
                           (self.x.numel() + 1023) // 1024, 1, 256)
