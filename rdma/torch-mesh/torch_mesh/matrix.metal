// Shared matrix product from dnn.py, with explicit type, view and epilogue operands.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
kernel void product(device const INPUT *X [[buffer(0)]], device const WEIGHT *W [[buffer(1)]], device uchar *Y [[buffer(2)]],
                    device atomic_uint *arrivals [[buffer(3)]], device OUTPUT *bias [[buffer(5)]],
                    device const INPUT *X1 [[buffer(6)]], device ulong *ready [[buffer(7)]],
                    uint3 tg [[threadgroup_position_in_grid]], uint3 lane [[thread_position_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int M=PRODUCT_ROWS, N=PRODUCT_COLUMNS, K=PRODUCT_INNER;
    const int row=int(tg.x)*ROWS*ROW_GROUPS+(sg/COLUMN_GROUPS)*ROWS, col=int(tg.y)*COLUMNS*COLUMN_GROUPS+(sg%COLUMN_GROUPS)*COLUMNS;
    constexpr auto desc=matmul2d_descriptor(ROWS,COLUMNS,INNER,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    const ulong abase=BATCH_A ? ulong(tg.z)*M*K0 : 0, bbase=BATCH_B ? ulong(tg.z)*K*N : 0;
    using Left=decltype(mesh_matrix_operand<true,0,AFFINE_A,ROWS,INNER,AM,AK,INPUT,WEIGHT>(op,X,abase,int2(K0,M),int2(0,row)));
    using Right=decltype(mesh_matrix_operand<false,1,AFFINE_B,INNER,COLUMNS,BK,BN,INPUT,WEIGHT>(op,W,bbase,int2(N,K0),int2(col,0)));
    auto acc=op.get_destination_cooperative_tensor<Left,Right,float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();i++)acc[i]=0;
    for(int at=0;at<K0;at+=INNER){
        auto a=mesh_matrix_operand<true,0,AFFINE_A,ROWS,INNER,AM,AK,INPUT,WEIGHT>(op,X,abase,int2(K0,M),int2(at,row));
        auto b=mesh_matrix_operand<false,1,AFFINE_B,INNER,COLUMNS,BK,BN,INPUT,WEIGHT>(op,W,bbase,int2(N,K0),int2(col,at));
        op.run(a,b,acc);
    }
#if SEGMENTED
    const ulong a1base=BATCH_A1 ? ulong(tg.z)*M*(K-K0) : 0;
    for(int at=0;at<K-K0;at+=INNER){
        auto a=mesh_matrix_operand<true,6,AFFINE_A1,ROWS,INNER,A1M,A1K,INPUT,WEIGHT>(op,X1,a1base,int2(K-K0,M),int2(at,row));
        auto b=mesh_matrix_operand<false,1,AFFINE_B,INNER,COLUMNS,BK,BN,INPUT,WEIGHT>(op,W,bbase+ulong(K0)*N,int2(N,K-K0),int2(col,at));
        op.run(a,b,acc);
    }
#endif
    #pragma unroll
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
    if (mesh_last_arrival<true>(arrivals + mesh_index<3>(slot), 0, columnGroups, local, last)) {
        // The publication value is a marker shared by this family. The final
        // short tile still transfers only its masked payload bytes.
        const ulong bytes = ulong(min(uint(ROWS*ROW_GROUPS), uint(M)-tg.x*ROWS*ROW_GROUPS))*N*sizeof(OUTPUT);
        mesh_source_release(ready + mesh_index<7>(slot), 0, bytes, local);
        mesh_arrival_reset(arrivals + mesh_index<3>(slot), 0, local);
    }
#endif
}
