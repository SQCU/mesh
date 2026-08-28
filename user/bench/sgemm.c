#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <mach/mach_time.h>
static double now(void){static mach_timebase_info_data_t t;if(!t.denom)mach_timebase_info(&t);return mach_absolute_time()*(double)t.numer/t.denom/1e9;}
int main(int argc,char**argv){
 int N=4096; int Ks[]={1,2,4,8,16,32,64,128,256,512,1024,4096};
 float*A=malloc((size_t)N*N*4);
 for(size_t i=0;i<(size_t)N*N;i++)A[i]=(float)((i*2654435761u)%1000)/1000.f;
 for(int ki=0;ki<12;ki++){int K=Ks[ki];
  float*B=calloc((size_t)N*K,4),*C=calloc((size_t)N*K,4);
  for(size_t i=0;i<(size_t)N*K;i++)B[i]=0.5f;
  // warm
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,K,N,1.f,A,N,B,K,0.f,C,K);
  int reps = K<=16?20:(K<=256?5:2);
  double t0=now();
  for(int r=0;r<reps;r++) cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,K,N,1.f,A,N,B,K,0.f,C,K);
  double dt=(now()-t0)/reps;
  double fl=2.0*N*N*K;
  printf("K=%5d  t=%9.3f ms  %8.1f GFLOP/s  bytes_moved=%.0f MB  AI=%.2f flop/byte\n",
    K,dt*1e3,fl/dt/1e9,((double)N*N*4+2.0*N*K*4)/1e6, fl/((double)N*N*4+2.0*N*K*4));
  free(B);free(C);}
 return 0;}
