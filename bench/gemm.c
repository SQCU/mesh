#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
static double now(void){struct timeval t;gettimeofday(&t,NULL);return t.tv_sec+t.tv_usec/1e6;}
int main(int argc,char**argv){
  int N = argc>1?atoi(argv[1]):4096;
  printf("  N=%d  A=%.1f MB\n", N, (double)N*N*4/1e6);
  float *A=malloc((size_t)N*N*4);
  for(size_t i=0;i<(size_t)N*N;i++) A[i]=0.001f*(i%97);
  printf("  %-6s %-10s %-12s %-10s %s\n","K","GFLOP/it","time ms","GFLOP/s","AI flop/byte");
  int Ks[]={1,8,64,256,512,1024,0};
  for(int k=0;Ks[k];k++){
    int K=Ks[k];
    float *X=malloc((size_t)N*K*4), *Y=malloc((size_t)N*K*4);
    for(size_t i=0;i<(size_t)N*K;i++){X[i]=0.5f;Y[i]=0.f;}
    int iters = K<=8?200: K<=64?60: K<=256?24: 12;
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,K,N,1.f,A,N,X,K,0.f,Y,K);
    double t=now();
    for(int i=0;i<iters;i++)
      cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,K,N,1.f,A,N,X,K,0.f,Y,K);
    double e=(now()-t)/iters;
    double fl=2.0*N*N*K;
    double ai=fl/(4.0*N*N+8.0*N*K);
    printf("  %-6d %-10.2f %-12.3f %-10.1f %.1f\n",K,fl/1e9,e*1e3,fl/e/1e9,ai);
    free(X);free(Y);
  }
  free(A); return 0;
}
