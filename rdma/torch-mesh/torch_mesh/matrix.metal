// Shared matrix product from dnn.py, with explicit type, view and epilogue operands.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
struct ProductArgs {
    uint4 shape;
#if FUSED
    mesh_step_binding land[CHANNELS], publish[CHANNELS];
#endif
};
kernel void product(device INPUT *X [[buffer(0)]], device INPUT *W [[buffer(1)]], device uchar *Y [[buffer(2)]],
                    device atomic_uint *arrivals [[buffer(3)]], constant ProductArgs &s [[buffer(4)]], device OUTPUT *bias [[buffer(5)]],
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
    auto A=tensor(X + tg.z * ABATCH, dextents<int,2>{K,M}, array<int,2>{AK,AM});
    auto B=tensor(W + tg.z * BBATCH, dextents<int,2>{N,K}, array<int,2>{BN,BK});
    constexpr auto desc=matmul2d_descriptor(32,32,32,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(A),decltype(B),float>();
    for(uint i=0;i<acc.get_capacity();i++)acc[i]=0;
    for(int at=0;at<K;at+=32){auto a=A.slice<32,32>(at,row);auto b=B.slice<32,32>(col,at);op.run(a,b,acc);}
    for(uint i=0;i<acc.get_capacity();i++)if(acc.is_valid_element(i)){
        auto e=acc.get_multidimensional_index(i);
        if(row+e[1]<M && col+e[0]<N) {
            float value=ALPHA*acc[i];
#if BIAS
            value+=BETA*float(bias[(row+e[1])*CM+(col+e[0])*CN]);
#endif
            mesh_step_write<OUTPUT>(Y,tg.z*YBATCH+(row+e[1])*YM+(col+e[0])*YN,
                                    OUTPUT(ACTIVATE ? tanh(value) : value));
        }
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
