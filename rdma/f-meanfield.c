// mean-field congestion planner -- see design/xonotic-bot-compute.md
#include "mesh-f.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define N     4096
#define KPB   8
#define M     32
#define EVERY 4
#define GAMMA 0.92f
#define MU    0.35f

static float *A, *V, *R, *Z, *rho_local, *rho_total;
static int K, bots, iter, ready;
static unsigned long long solves;

static void init(void){
  const char *b = getenv("MESH_BOTS");
  bots = b ? atoi(b) : 24;
  K = bots * KPB;
  A = malloc((size_t)N*N*sizeof(float));
  V = calloc((size_t)N*K, sizeof(float));
  R = malloc((size_t)N*K*sizeof(float));
  Z = malloc((size_t)N*K*sizeof(float));
  rho_local = calloc(N, sizeof(float));
  rho_total = calloc(N, sizeof(float));
  if(!A||!V||!R||!Z||!rho_local||!rho_total){ fprintf(stderr,"meanfield: alloc failed\n"); exit(1); }
  unsigned s = 12345;
  for(size_t i=0;i<(size_t)N*N;i++){ s=s*1103515245u+12345u; A[i]=(float)((s>>16)&0xff)/65536.0f; }
  for(int r=0;r<N;r++){ float t=0; for(int c=0;c<N;c++) t+=A[(size_t)r*N+c];
    if(t>0) for(int c=0;c<N;c++) A[(size_t)r*N+c]/=t; }
  for(size_t i=0;i<(size_t)N*K;i++){ s=s*1103515245u+12345u; R[i]=(float)((s>>16)&0xff)/32768.0f-0.5f; }
  ready=1;
  fprintf(stderr,"meanfield: bots=%d K=%d N=%d A=%.0fMB V=%.0fMB\n",
          bots,K,N,(double)N*N*4/1e6,(double)N*K*4/1e6);
}

static void step(void){
  for(int c=0;c<K;c++)
    for(int r=0;r<N;r++){
      float x = R[(size_t)r*K+c] + V[(size_t)r*K+c] - MU*rho_total[r];
      Z[(size_t)r*K+c] = x>20.0f ? x : logf(1.0f+expf(x));
    }
  for(int r=0;r<N;r++){
    const float *a = A + (size_t)r*N;
    for(int c=0;c<K;c++){
      float acc=0;
      for(int k=0;k<N;k++) acc += a[k]*Z[(size_t)k*K+c];
      V[(size_t)r*K+c] = GAMMA*acc;
    }
  }
  for(int r=0;r<N;r++){ float t=0; for(int c=0;c<K;c++) t+=Z[(size_t)r*K+c]; rho_local[r]=t; }
}

void mesh_f(struct miov *iov, int niov, uint32_t bytes, struct wire *h, int node_idx){
  if(!ready) init();
  if(!(h->flags & F_META)) return;
  if(bytes < N*sizeof(float)) return;

  size_t off=0;
  for(int i=0;i<niov && off<N*sizeof(float); i++){
    size_t n = iov[i].len; if(off+n > N*sizeof(float)) n = N*sizeof(float)-off;
    const float *p = iov[i].base;
    for(size_t j=0;j<n/sizeof(float);j++) rho_total[off/sizeof(float)+j] = rho_local[off/sizeof(float)+j] + p[j];
    off += n;
  }

  for(int s=0;s<EVERY;s++) step();
  iter += EVERY;
  if(iter >= M){ iter = 0; solves++; }

  off=0;
  for(int i=0;i<niov && off<N*sizeof(float); i++){
    size_t n = iov[i].len; if(off+n > N*sizeof(float)) n = N*sizeof(float)-off;
    memcpy(iov[i].base, (char*)rho_local + off, n);
    off += n;
  }
  h->src = (uint16_t)node_idx;
  h->dst = (uint16_t)(node_idx ^ 1);
}
