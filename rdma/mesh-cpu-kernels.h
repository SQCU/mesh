#pragma once
#include "mesh-coreml.h"
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>

/* Shared numerical bodies. Source preparation places page effects in the same
   entry; native declarations and generated entries use these exact operations. */
static inline void mesh_cpu_product_body(const void *arguments) {
      const MeshCpuCommand *c=(const MeshCpuCommand *)arguments;
      cblas_sgemm(CblasRowMajor,(enum CBLAS_TRANSPOSE)c->input_transpose,(enum CBLAS_TRANSPOSE)c->weight_transpose,c->rows,c->columns,c->inner,
                  1,c->input,c->input_stride,c->weight,c->weight_stride,0,c->output,c->output_stride);
}
typedef struct {
  BNNSNDArrayDescriptor input, weight, output;
  void *workspace;
} MeshCpuProduct;

static inline void mesh_cpu_product_strided_body(const void *arguments) {
  const MeshCpuCommand *c=(const MeshCpuCommand *)arguments;
  MeshCpuProduct *p=(MeshCpuProduct *)c->scratch;
  if(BNNSMatMul(false,true,1,&p->input,&p->weight,&p->output,p->workspace,NULL)) {
    fputs("mesh: prepared CPU contraction failed\n",stderr);abort();
  }
}

static inline void mesh_cpu_gelu_body(const void *arguments) {
      const MeshCpuCommand *c=(const MeshCpuCommand *)arguments;
      const int n=c->rows*c->columns;
      for(int i=0;i<n;i++) {
        const float x=c->input[i];c->scratch[i]=0.7978845608028654f*(x+0.044715f*x*x*x);
      }
      vvtanhf(c->scratch,c->scratch,&n);
      for(int i=0;i<n;i++)c->output[i]=0.5f*c->input[i]*(1+c->scratch[i])*c->weight[i];
}
