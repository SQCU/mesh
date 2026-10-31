// Shared matrix product from dnn.py, with explicit type, view and epilogue operands.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#ifndef MESH_MATRIX_PARAMETERS
#define MESH_MATRIX_PARAMETERS
#endif
#ifndef MESH_MATRIX_X_READ
#define MESH_MATRIX_X_READ mesh_page_read<0,INPUT>{X}
#endif
#ifndef MESH_MATRIX_W_READ
#define MESH_MATRIX_W_READ mesh_page_read<1,WEIGHT>{W}
#endif
#ifndef MESH_MATRIX_X1_READ
#define MESH_MATRIX_X1_READ mesh_page_read<6,INPUT>{X1}
#endif
#ifndef MESH_MATRIX_BIAS_READ
#define MESH_MATRIX_BIAS_READ mesh_page_read<5,OUTPUT>{bias}
#endif
#if !PRODUCT_ROW_KERNEL
kernel void product(device const INPUT *X [[buffer(0)]], device const WEIGHT *W [[buffer(1)]], device uchar *Y [[buffer(2)]],
                    device atomic_uint *arrivals [[buffer(3)]], device OUTPUT *bias [[buffer(5)]],
                    device const INPUT *X1 [[buffer(6)]], device ulong *ready [[buffer(7)]], MESH_MATRIX_PARAMETERS
                    uint3 tg [[threadgroup_position_in_grid]], uint3 lane [[thread_position_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]]) {
    const auto readX=MESH_MATRIX_X_READ;const auto readW=MESH_MATRIX_W_READ;
#if BIAS
    const auto readBias=MESH_MATRIX_BIAS_READ;
#endif
#if SEGMENTED
    const auto readX1=MESH_MATRIX_X1_READ;
#endif
    constexpr int M=PRODUCT_ROWS, N=PRODUCT_COLUMNS, K=PRODUCT_INNER;
    const int row=int(tg.x)*ROWS*ROW_GROUPS+(sg/COLUMN_GROUPS)*ROWS, col=int(tg.y)*COLUMNS*COLUMN_GROUPS+(sg%COLUMN_GROUPS)*COLUMNS;
    constexpr auto desc=matmul2d_descriptor(ROWS,COLUMNS,INNER,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    const ulong abase=BATCH_A ? ulong(tg.z)*M*PRODUCT_FIRST : 0, bbase=BATCH_B ? ulong(tg.z)*K*N : 0;
    using Left=decltype(mesh_matrix_operand<true,0,AFFINE_A,ROWS,INNER,AM,AK,INPUT,WEIGHT>(op,X,abase,int2(PRODUCT_FIRST,M),int2(0,row),PRODUCT_FIRST,readX));
    using Right=decltype(mesh_matrix_operand<false,1,AFFINE_B,INNER,COLUMNS,BK,BN,INPUT,WEIGHT>(op,W,bbase,int2(N,PRODUCT_FIRST),int2(col,0),N,readW));
    auto acc=op.get_destination_cooperative_tensor<Left,Right,float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();i++)acc[i]=0;
    for(int at=0;at<PRODUCT_FIRST;at+=INNER){
        auto a=mesh_matrix_operand<true,0,AFFINE_A,ROWS,INNER,AM,AK,INPUT,WEIGHT>(op,X,abase,int2(PRODUCT_FIRST,M),int2(at,row),PRODUCT_FIRST,readX);
        auto b=mesh_matrix_operand<false,1,AFFINE_B,INNER,COLUMNS,BK,BN,INPUT,WEIGHT>(op,W,bbase,int2(N,PRODUCT_FIRST),int2(col,at),N,readW);
        op.run(a,b,acc);
    }
#if SEGMENTED
    const ulong a1base=BATCH_A1 ? ulong(tg.z)*M*(K-PRODUCT_FIRST) : 0;
    for(int at=0;at<K-PRODUCT_FIRST;at+=INNER){
        auto a=mesh_matrix_operand<true,6,AFFINE_A1,ROWS,INNER,A1M,A1K,INPUT,WEIGHT>(op,X1,a1base,int2(K-PRODUCT_FIRST,M),int2(at,row),K-PRODUCT_FIRST,readX1);
        auto b=mesh_matrix_operand<false,1,AFFINE_B,INNER,COLUMNS,BK,BN,INPUT,WEIGHT>(op,W,bbase+ulong(PRODUCT_FIRST)*N,int2(N,K-PRODUCT_FIRST),int2(col,at),N,readW);
        op.run(a,b,acc);
    }
#endif
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();i++)if(acc.is_valid_element(i)){
        auto e=acc.get_multidimensional_index(i);
        if(row+e[1]<M && col+e[0]<N) {
            float value=ALPHA*acc[i];
#if BIAS
            value+=BETA*float(readBias.load((BIAS_ROWS > 1 ? row+e[1] : 0)*BIAS_COLUMNS + (BIAS_COLUMNS > 1 ? col+e[0] : 0)));
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

#else
kernel void product(device const INPUT* X [[buffer(0)]],device const WEIGHT* W [[buffer(1)]],device uchar* Y [[buffer(2)]],
    device atomic_uint* arrivals [[buffer(3)]],device const OUTPUT* bias [[buffer(5)]],
    device const INPUT* X1 [[buffer(6)]],device ulong* ready [[buffer(7)]], MESH_MATRIX_PARAMETERS
    uint3 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    const auto readX=MESH_MATRIX_X_READ;const auto readW=MESH_MATRIX_W_READ;const auto readBias=MESH_MATRIX_BIAS_READ;
    constexpr uint M=PRODUCT_ROWS,N=PRODUCT_COLUMNS,K=PRODUCT_INNER;
    threadgroup float part[ROW_SPLIT*MESH_LINEAR_ROWS*MESH_LINEAR_COLUMNS];
    const linear_dims dims={M,K,N,BIAS,0,PRODUCT_FIRST,0,0,0,0};
    const matrix_row_read<0,INPUT,false,PRODUCT_FIRST,AFFINE_A,AM,AK,AB,decltype(readX)> x{readX,BATCH_A?ulong(tg.z)*M*PRODUCT_FIRST:0};
    const matrix_row_read<1,WEIGHT,true,N,AFFINE_B,BN,BK,BB,decltype(readW)> w{readW,BATCH_B?ulong(tg.z)*K*N:0};
#if SEGMENTED
    const auto readX1=MESH_MATRIX_X1_READ;
    const matrix_row_read<6,INPUT,false,K-PRODUCT_FIRST,AFFINE_A1,A1M,A1K,A1B,decltype(readX1)> x1{readX1,BATCH_A1?ulong(tg.z)*M*(K-PRODUCT_FIRST):0};
#else
    const auto x1=x;
#endif
    const uint columnGroups=(N+MESH_LINEAR_COLUMNS-1)/MESH_LINEAR_COLUMNS;
    linear_body_read<MESH_LINEAR_COLUMNS,ROW_SPLIT,MESH_LINEAR_LOADS>(x,x1,w,dims,matrix_row_store<2,OUTPUT,decltype(readBias),PRODUCT_ROWS,PRODUCT_COLUMNS,BIAS_ROWS,BIAS?BIAS_COLUMNS:0,ACTIVATE>{Y,readBias,tg.z,ALPHA,BETA},
        part,sg,lane,tg.x*columnGroups+tg.y,tid);
#if PRODUCED
    const uint rowGroups=(M+MESH_LINEAR_ROWS-1)/MESH_LINEAR_ROWS,slot=tg.z*rowGroups+tg.x;
    threadgroup uint last[1];
    if(mesh_last_arrival<true>(arrivals+mesh_index<3>(slot),0,columnGroups,tid,last)){
        const ulong bytes=ulong(min(uint(MESH_LINEAR_ROWS),M-tg.x*MESH_LINEAR_ROWS))*N*sizeof(OUTPUT);
        mesh_source_release(ready+mesh_index<7>(slot),0,bytes,tid);
        mesh_arrival_reset(arrivals+mesh_index<3>(slot),0,tid);
    }
#endif
}
#endif
