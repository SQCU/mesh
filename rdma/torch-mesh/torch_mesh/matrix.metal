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
                    device INPUT *X1 [[buffer(6)]],
                    uint3 tg [[threadgroup_position_in_grid]], uint3 lane [[thread_position_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int M=s.shape.x, N=s.shape.z;
#if FUSED
    const uint local = lane.x;
    for (uint k = 0; k < CHANNELS; k++) if (s.publish[k].p) {
        auto b = s.publish[k];
        mesh_step_enter(*b.p, b.v, b.progress, tg.x == 0 && tg.y == 0 && tg.z == 0 && local == 0);
    }
#endif
    const int row=int(tg.x)*ROWS*ROW_GROUPS+(sg/COLUMN_GROUPS)*ROWS, col=int(tg.y)*COLUMNS*COLUMN_GROUPS+(sg%COLUMN_GROUPS)*COLUMNS;
    auto A=tensor(X + tg.z * ABATCH, dextents<int,2>{K0,M}, array<int,2>{AK,AM});
    auto B=tensor(W + tg.z * BBATCH, dextents<int,2>{N,K0}, array<int,2>{BN,BK});
    constexpr auto desc=matmul2d_descriptor(ROWS,COLUMNS,INNER,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(A),decltype(B),float>();
    for(uint i=0;i<acc.get_capacity();i++)acc[i]=0;
    for(int at=0;at<K0;at+=INNER){auto a=A.slice<INNER,ROWS>(at,row);auto b=B.slice<COLUMNS,INNER>(col,at);op.run(a,b,acc);}
#if SEGMENTED
    const int K=s.shape.y;
    auto A1=tensor(X1 + tg.z * A1BATCH, dextents<int,2>{K-K0,M}, array<int,2>{A1K,A1M});
    auto B1=tensor(W + tg.z * BBATCH + K0 * BK, dextents<int,2>{N,K-K0}, array<int,2>{BN,BK});
    for(int at=0;at<K-K0;at+=INNER){auto a=A1.slice<INNER,ROWS>(at,row);auto b=B1.slice<COLUMNS,INNER>(col,at);op.run(a,b,acc);}
#endif
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
    if (local == 0) last[0] = atomic_fetch_add_explicit(arrivals, 1u, memory_order_relaxed) == uint(((M+ROWS*ROW_GROUPS-1)/(ROWS*ROW_GROUPS))*((N+COLUMNS*COLUMN_GROUPS-1)/(COLUMNS*COLUMN_GROUPS))*s.shape.w-1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (last[0]) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3));
        for (uint k = 0; k < CHANNELS; k++) if (s.publish[k].p)
            mesh_step_release(*s.publish[k].p, s.publish[k].a, local);
        if (local == 0) atomic_store_explicit(arrivals, 0u, memory_order_relaxed);
    }
#endif
}
