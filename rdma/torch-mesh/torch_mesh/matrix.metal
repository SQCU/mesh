// Shared matrix product from dnn.py, with explicit type, view and epilogue operands.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
struct ProductArgs {
    uint4 shape;
};
kernel void product(device INPUT *X [[buffer(0)]], device WEIGHT *W [[buffer(1)]], device uchar *Y [[buffer(2)]],
                    device atomic_uint *arrivals [[buffer(3)]], constant ProductArgs &s [[buffer(4)]], device OUTPUT *bias [[buffer(5)]],
                    device INPUT *X1 [[buffer(6)]], device ulong *ready [[buffer(7)]],
                    uint3 tg [[threadgroup_position_in_grid]], uint3 lane [[thread_position_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int M=s.shape.x, N=s.shape.z;
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
            value+=BETA*float(bias[mesh_index<5>((BIAS_ROWS > 1 ? row+e[1] : 0)*BIAS_COLUMNS + (BIAS_COLUMNS > 1 ? col+e[0] : 0))]);
#endif
            mesh_step_write<OUTPUT>(Y,mesh_index<2>((ulong(tg.z)*M+row+e[1])*N+col+e[0]),
                                    OUTPUT(ACTIVATE ? tanh(value) : value));
        }
    }
#if PRODUCED
    const uint local = lane.x, rowGroups = (M+ROWS*ROW_GROUPS-1)/(ROWS*ROW_GROUPS),
        columnGroups = (N+COLUMNS*COLUMN_GROUPS-1)/(COLUMNS*COLUMN_GROUPS), slot = tg.z*rowGroups+tg.x;
    threadgroup uint last[1];
    if (mesh_last_arrival<true>(arrivals, slot, columnGroups, local, last)) {
        // The publication value is a marker shared by this family. The final
        // short tile still transfers only its masked payload bytes.
        const ulong bytes = ulong(min(uint(ROWS*ROW_GROUPS), uint(M)))*N*sizeof(OUTPUT);
        mesh_source_release(ready, slot, bytes, local);
        mesh_arrival_reset(arrivals, slot, local);
    }
#endif
}
