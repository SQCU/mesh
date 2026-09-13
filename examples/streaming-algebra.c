#include "mesh-algebra.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* design/algorithm-sources.md#streaming-algebra */
static void check(int error) { if(error){fprintf(stderr,"streaming algebra: %d\n",error);exit(1);} }
/* design/algorithm-sources.md#streaming-algebra */
static double now(void) { struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9; }
/* design/algorithm-sources.md#streaming-algebra */
static struct mesh_tensor *tensor(struct mesh_algebra *a,size_t count,size_t rows,size_t columns,int transfer) {
  struct mesh_shape shapes[count];for(size_t i=0;i<count;i++)shapes[i]=(struct mesh_shape){rows,columns,MESH_F32};
  struct mesh_tensor *t=mesh_tensor_create(a,shapes,count,transfer);if(!t)check(errno);return t;
}
/* design/algorithm-sources.md#streaming-algebra */
static float input(int rank,size_t extent,size_t row,size_t column) {
  return (float)((int)((row*17+column*3+extent*11)%97)-48)/128+(float)rank/16;
}
/* design/algorithm-sources.md#streaming-algebra */
static float weight(size_t part,size_t row,size_t column) {
  return (float)((int)((row*5+column*7+part*13)%31)-15)/256;
}
/* design/algorithm-sources.md#streaming-algebra */
static void produce(struct mesh_tensor *t,int rank,uint32_t extent,size_t rows,size_t columns,size_t trial) {
  check(mesh_tensor_issue(t,extent)?0:EBUSY);float *data=mesh_tensor_data(t,extent);
  for(size_t r=0;r<rows;r++)for(size_t c=0;c<columns;c++)data[r*columns+c]=input(rank,extent,r,c)+(float)trial/256;
  mesh_tensor_complete(t,extent);
}
/* design/algorithm-sources.md#online-percentage-moments */
static void moment(size_t n,double x,double *mean,double *m2) { double d=x-*mean;*mean+=d/n;*m2+=d*(x-*mean); }

struct program { struct mesh_tensor *x,*p,*result,*gathered,*statistics,*normalized; };

/* design/algorithm-sources.md#pallas-indexed-destinations */
static struct program configure(struct mesh_algebra *a,int rank,size_t rows,size_t k,size_t n,struct mesh_tensor *weights) {
  struct mesh_tensor *x=tensor(a,4,rows,k,0),*p=tensor(a,4,rows,k,1),*remote=tensor(a,4,rows,k,1);
  struct mesh_tensor *sum=tensor(a,4,rows,k,0),*activated=tensor(a,4,rows,k,0);
  struct mesh_tensor *partial=tensor(a,4,rows,n,0),*result=tensor(a,2,rows,n,1),*gathered=tensor(a,2,rows,n,1);
  struct mesh_tensor *statistics=tensor(a,2,rows,1,0);
  struct mesh_tensor *squared=tensor(a,2,rows,n,0),*sumsq=tensor(a,2,rows,1,0);
  struct mesh_tensor *meansq=tensor(a,2,rows,1,0),*inverse=tensor(a,2,rows,1,0),*normalized=tensor(a,2,rows,n,0);
  for(size_t i=0;i<4;i++) {
    struct mesh_view xv=mesh_tensor_view(x,(uint32_t)i),pv=mesh_tensor_view(p,(uint32_t)i),rv=mesh_tensor_view(remote,(uint32_t)i);
    struct mesh_view sv=mesh_tensor_view(sum,(uint32_t)i),av=mesh_tensor_view(activated,(uint32_t)i),cv=mesh_tensor_view(partial,(uint32_t)i);
    check(mesh_algebra_bind(a,MESH_AFFINE,xv,(struct mesh_view){0},pv,0.5f,0.125f));
    check(mesh_algebra_bind(a,MESH_ADD,pv,rv,sv,1,1));
    check(mesh_algebra_bind(a,MESH_TANH,sv,(struct mesh_view){0},av,0,0));
    check(mesh_algebra_bind(a,MESH_CONTRACT,av,mesh_view_transpose(mesh_tensor_view(weights,(uint32_t)(i/2))),cv,1,0));
  }
  for(uint32_t group=0;group<2;group++) {
    for(uint32_t peer=0;peer<2;peer++)check(mesh_algebra_copy(a,(struct mesh_endpoint){p,peer,group,2},(struct mesh_endpoint){remote,1-peer,group,2},2,(uint16_t)group));
    check(mesh_algebra_bind(a,MESH_ADD,mesh_tensor_view(partial,group),mesh_tensor_view(partial,group+2),mesh_tensor_view(result,group),1,1));
    check(mesh_algebra_bind(a,MESH_SUM,mesh_tensor_view(result,group),(struct mesh_view){0},mesh_tensor_view(statistics,group),0,0));
    struct mesh_view v=mesh_tensor_view(result,group);
    check(mesh_algebra_bind(a,MESH_MULTIPLY,v,v,mesh_tensor_view(squared,group),0,0));
    check(mesh_algebra_bind(a,MESH_SUM,mesh_tensor_view(squared,group),(struct mesh_view){0},mesh_tensor_view(sumsq,group),0,0));
    check(mesh_algebra_bind(a,MESH_AFFINE,mesh_tensor_view(sumsq,group),(struct mesh_view){0},mesh_tensor_view(meansq,group),1.0f/n,1e-5f));
    check(mesh_algebra_bind(a,MESH_RSQRT,mesh_tensor_view(meansq,group),(struct mesh_view){0},mesh_tensor_view(inverse,group),0,0));
    check(mesh_algebra_bind(a,MESH_MULTIPLY,v,mesh_view_broadcast(mesh_tensor_view(inverse,group),rows,n),mesh_tensor_view(normalized,group),0,0));
    check(mesh_algebra_return(a,result,group));
  }
  for(uint32_t peer=0;peer<2;peer++)check(mesh_algebra_copy(a,(struct mesh_endpoint){result,peer,peer,1},(struct mesh_endpoint){gathered,1-peer,peer,1},1,(uint16_t)peer));
  check(mesh_algebra_return(a,gathered,(uint32_t)(1-rank)));
  for(uint32_t group=0;group<2;group++)check(mesh_algebra_return(a,statistics,group));
  check(mesh_algebra_return(a,p,0));
  for(uint32_t group=0;group<2;group++)check(mesh_algebra_return(a,normalized,group));
  return (struct program){x,p,result,gathered,statistics,normalized};
}

/* design/algorithm-sources.md#streaming-algebra */
static int verify(struct program p,int rank,size_t rows,size_t k,size_t n,size_t trial,double *maxError) {
    for(size_t group=0;group<2;group++) {
      float *actual=mesh_tensor_data(p.result,(uint32_t)group);
      float *peer=group==(size_t)(1-rank)?mesh_tensor_data(p.gathered,(uint32_t)group):actual;
      float *stat=mesh_tensor_data(p.statistics,(uint32_t)group);
      float *normal=mesh_tensor_data(p.normalized,(uint32_t)group);
      for(size_t r=0;r<rows;r++) {
        double total=0,square=0;
        for(size_t c=0;c<n;c++) {
          double expected=0;
          for(size_t part=0;part<2;part++)for(size_t j=0;j<k;j++) {
            size_t extent=group+part*2;
            float v=(0.5f*(input(0,extent,r,j)+(float)trial/256)+0.125f)+(0.5f*(input(1,extent,r,j)+(float)trial/256)+0.125f);
            expected+=(double)tanhf(v)*weight(part,j,c);
          }
          double error=fabs(actual[r*n+c]-expected);if(error>*maxError)*maxError=error;
          if(!isfinite(actual[r*n+c]) || error>2e-4 || peer[r*n+c]!=actual[r*n+c]){fprintf(stderr,"numerical mismatch %zu %zu %zu %.9g %.9g\n",group,r,c,actual[r*n+c],expected);return 6;}
          total+=actual[r*n+c];square+=(double)actual[r*n+c]*actual[r*n+c];
        }
        for(size_t c=0;c<n;c++)if(!isfinite(normal[r*n+c]) || fabs(normal[r*n+c]-actual[r*n+c]/sqrt(square/n+1e-5))>2e-5){fprintf(stderr,"normalization composition mismatch\n");return 8;}
        if(!isfinite(stat[r]) || fabs(stat[r]-total)>2e-4){fprintf(stderr,"row reduction mismatch\n");return 7;}
      }
    }
  return 0;
}

/* design/algorithm-sources.md#pallas-indexed-destinations */
int main(int argc,char **argv) {
  int rank=argc>1?atoi(argv[1]):0;
  size_t invocations=argc>2?(size_t)atoi(argv[2]):32;
  size_t rows=argc>3?(size_t)atoi(argv[3]):128,depth=argc>4?(size_t)atoi(argv[4]):8,k=128,n=64;
  if((rank!=0 && rank!=1) || !invocations || !rows || !depth)return 2;
  if(depth>invocations)depth=invocations;
  struct mesh_ctx context={0};check(mesh_attach(&context,NULL));
  if(context.M->qps<2){fprintf(stderr,"acceptance requires two independent configured transport queues\n");mesh_detach(&context);return 2;}
  struct mesh_algebra *a=mesh_algebra_create(&context);if(!a)check(errno);
  struct program *programs=calloc(depth,sizeof *programs);if(!programs)check(ENOMEM);
  struct mesh_tensor *weights=tensor(a,2,n,k,0);
  for(size_t part=0;part<2;part++) {
    float *data=mesh_tensor_data(weights,(uint32_t)part);
    for(size_t r=0;r<k;r++)for(size_t c=0;c<n;c++)data[c*k+r]=weight(part,r,c);
    check(mesh_tensor_constant(weights,(uint32_t)part));
  }
  for(size_t slot=0;slot<depth;slot++)programs[slot]=configure(a,rank,rows,k,n,weights);
  check(mesh_algebra_realize(a));
  for(size_t slot=0;slot<depth;slot++)for(uint32_t i=slot?0:1;i<4;i++)produce(programs[slot].x,rank,i,rows,k,slot);
  double maxError=0,earlyMean=0,earlyM2=0,start=now();size_t windows=0,later=0;
  for(size_t trial=0;trial<invocations;trial++) {
    size_t slot=trial%depth,base=8*slot;double began=now();
    if(!slot) {
      size_t width=invocations-trial<depth?invocations-trial:depth;
      for(;;) {
        mesh_algebra_scan(a);int ready=1;
        for(size_t j=0;j<width;j++)ready=ready && mesh_algebra_available(a,8*j+1);
        check((int)mesh_algebra_report(a).code);check((int)mesh_link_metadata(&context,0).code);
        if(ready)break;
        if(now()-began>30){fprintf(stderr,"independent invocation outputs did not complete\n");return 3;}
      }
      if(mesh_algebra_available(a,5) || mesh_algebra_available(a,0)){fprintf(stderr,"premature publication\n");return 4;}
      moment(++windows,(now()-began)*1e3,&earlyMean,&earlyM2);later+=width-1;
      produce(programs[slot].x,rank,0,rows,k,trial);
    }
    for(;;) {
      mesh_algebra_scan(a);int ready=1;
      for(size_t j=0;j<8;j++)ready=ready && mesh_algebra_available(a,base+j);
      check((int)mesh_algebra_report(a).code);check((int)mesh_link_metadata(&context,0).code);
      if(ready)break;
      if(now()-began>30){fprintf(stderr,"completion timeout invocation %zu\n",trial);return 5;}
    }
    check(verify(programs[slot],rank,rows,k,n,trial,&maxError));
    for(size_t j=0;j<8;j++)mesh_algebra_consume(a,base+j);
    size_t next=trial+depth;
    if(next<invocations)for(uint32_t i=slot?0:1;i<4;i++)produce(programs[slot].x,rank,i,rows,k,next);
  }
  struct mesh_algebra_report report=mesh_algebra_report(a);
  while(report.completed!=report.submitted){mesh_algebra_scan(a);report=mesh_algebra_report(a);check((int)report.code);}
  char variance[32];snprintf(variance,sizeof variance,windows>1?"%.9g":"null",windows>1?earlyM2/(windows-1):0);
  printf("{\"rank\":%d,\"invocations\":%zu,\"depth\":%zu,\"rows\":%zu,\"k\":%zu,\"n\":%zu,\"delayed_windows\":%zu,\"later_invocation_completions\":%zu,\"max_absolute_error\":%.9g,\"early_window_ms\":{\"count\":%zu,\"mean\":%.9g,\"sample_variance\":%s},\"allocated_pages\":%u,\"commands\":%llu,\"gpu_seconds\":%.9g,\"wall_seconds\":%.9g}\n",rank,invocations,depth,rows,2*k,n,windows,later,maxError,windows,earlyMean,variance,context.arena,(unsigned long long)report.completed,report.gpu_seconds,now()-start);
  free(programs);mesh_algebra_destroy(a);check(mesh_detach(&context));return 0;
}
