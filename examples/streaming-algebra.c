#include "mesh-algebra.h"
#include "streaming-expression.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

/* design/algorithm-sources.md#streaming-algebra */
static void check(int error) { if(error){fprintf(stderr,"streaming algebra: %d\n",error);exit(1);} }
/* design/algorithm-sources.md#streaming-algebra */
static double now(void) { struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9; }
/* design/algorithm-sources.md#streaming-algebra */
static struct mesh_tensor *typed_tensor(struct mesh_algebra *a,size_t count,size_t rows,size_t columns,int transfer,enum mesh_scalar scalar) {
  struct mesh_shape shapes[count];for(size_t i=0;i<count;i++)shapes[i]=(struct mesh_shape){rows,columns,scalar};
  struct mesh_tensor *t=mesh_tensor_create(a,shapes,count,transfer);if(!t)check(errno);return t;
}
/* design/algorithm-sources.md#streaming-algebra */
static struct mesh_tensor *tensor(struct mesh_algebra *a,size_t count,size_t rows,size_t columns,int transfer) {
  return typed_tensor(a,count,rows,columns,transfer,MESH_F32);
}
/* design/algorithm-sources.md#coreml-partial-execution */
static float activation(float x,int half) {return half?(float)(_Float16)tanhf(x):tanhf(x);}
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

struct program { struct mesh_tensor *x,*p,*result,*gathered,*statistics,*normalized,*partial,*peer_partial; struct mesh_row_map prefix,tail; struct mesh_row_function prefix_function,tail_function; };

/* design/algorithm-sources.md#pallas-indexed-destinations */
static struct program configure(struct mesh_algebra *a,int rank,size_t rows,size_t k,size_t n,struct mesh_tensor *weights,int symbolic,int half) {
  struct mesh_tensor *x=tensor(a,4,rows,k,0),*p=tensor(a,4,rows,k,1),*remote=tensor(a,4,rows,k,1);
  struct mesh_tensor *sum=symbolic?NULL:tensor(a,4,rows,k,0),*activated=symbolic?NULL:typed_tensor(a,4,rows,k,0,half?MESH_F16:MESH_F32);
  struct mesh_tensor *partial=NULL,*result=tensor(a,2,rows,n,1),*gathered=tensor(a,2,rows,n,1);
  struct mesh_tensor *statistics=tensor(a,2,rows,1,0);
  struct mesh_tensor *squared=tensor(a,2,rows,n,0),*sumsq=tensor(a,2,rows,1,0);
  struct mesh_tensor *meansq=tensor(a,2,rows,1,0),*inverse=tensor(a,2,rows,1,0),*normalized=tensor(a,2,rows,n,0);
  for(size_t i=0;!symbolic && i<4;i++) {
    struct mesh_view xv=mesh_tensor_view(x,(uint32_t)i),pv=mesh_tensor_view(p,(uint32_t)i),rv=mesh_tensor_view(remote,(uint32_t)i);
    struct mesh_view sv=mesh_tensor_view(sum,(uint32_t)i),av=mesh_tensor_view(activated,(uint32_t)i);
    check(mesh_algebra_bind(a,MESH_AFFINE,xv,(struct mesh_view){0},pv,0.5f,0.125f));
    check(mesh_algebra_bind(a,MESH_ADD,pv,rv,sv,1,1));
    check(mesh_algebra_bind(a,MESH_TANH,sv,(struct mesh_view){0},av,0,0));
  }
  for(uint32_t group=0;group<2;group++) {
    for(uint32_t q=0;q<2;q++)for(uint32_t peer=0;peer<2;peer++) {
      uint32_t i=group+2*(1-q);
      check(mesh_algebra_copy(a,(struct mesh_endpoint){p,peer,i,1},(struct mesh_endpoint){remote,1-peer,i,1},1,(uint16_t)group));
    }
    struct mesh_view av[2],wv[2],xv[2],rv[2],pv[2];
    for(uint32_t q=0;q<2;q++) {
      uint32_t i=group+2*(1-q);
      xv[q]=mesh_tensor_view(x,i);rv[q]=mesh_tensor_view(remote,i);pv[q]=mesh_tensor_view(p,i);
      if(!symbolic)av[q]=mesh_tensor_view(activated,i);
      wv[q]=mesh_view_transpose(mesh_tensor_view(weights,1-q));
    }
    struct mesh_tensor *contributions=symbolic?(half?symbolic_contract_f16:symbolic_contract)(a,2,xv,rv,wv,pv,mesh_tensor_view(result,group)):mesh_algebra_contract(a,av,wv,2,mesh_tensor_view(result,group),1);
    if(!contributions)check(errno);if(!group)partial=contributions;
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
  return (struct program){.x=x,.p=p,.result=result,.gathered=gathered,.statistics=statistics,.normalized=normalized,.partial=partial,.peer_partial=tensor(a,2,rows,n,1)};
}

/* design/algorithm-sources.md#streaming-algebra */
static int verify(struct program p,int rank,size_t rows,size_t k,size_t n,size_t trial,double *maxError,int half) {
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
            expected+=(double)activation(v,half)*weight(part,j,c);
          }
          double error=fabs(actual[r*n+c]-expected);if(error>*maxError)*maxError=error;
          if(!isfinite(actual[r*n+c]) || error>(half?1e-3:2e-4) || (!isfinite(peer[r*n+c]) || (half?fabs(peer[r*n+c]-expected)>1e-3:peer[r*n+c]!=actual[r*n+c]))){fprintf(stderr,"numerical mismatch %zu %zu %zu %.9g %.9g\n",group,r,c,actual[r*n+c],expected);return 6;}
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
  int rank=argc>1?atoi(argv[1]):0,symbolic=argc>5?atoi(argv[5]):0,half=argc>9?atoi(argv[9]):0;
  size_t invocations=argc>2?(size_t)atoi(argv[2]):32;
  size_t rows=argc>3?(size_t)atoi(argv[3]):257,depth=argc>4?(size_t)atoi(argv[4]):8,k=128,n=64;
  if((rank!=0 && rank!=1) || !invocations || !rows || !depth)return 2;
  if(depth>invocations)depth=invocations;
  struct mesh_ctx context={0};check(mesh_attach(&context,NULL));
  if(context.M->qps<3){fprintf(stderr,"acceptance requires three independent configured transport queues\n");mesh_detach(&context);return 2;}
  int cpu=argc>6 && !strcmp(argv[6],"cpu");
  struct mesh_algebra *a=cpu?mesh_algebra_create_cpu(&context):mesh_algebra_create(&context);if(!a)check(errno);
  if(argc>6 && !cpu)check(mesh_algebra_coreml(a,argv[6],argc>7?argv[7]:"rdma/mesh_coreml.py",argc>8?argv[8]:"rdma/.build/coreml-parts"));
  struct program *programs=calloc(depth,sizeof *programs);if(!programs)check(ENOMEM);
  struct mesh_tensor *weights=typed_tensor(a,2,n,k,0,half?MESH_F16:MESH_F32);
  for(size_t part=0;part<2;part++) {
    void *data=mesh_tensor_data(weights,(uint32_t)part);
    for(size_t r=0;r<k;r++)for(size_t c=0;c<n;c++){if(half)((_Float16 *)data)[c*k+r]=(_Float16)weight(part,r,c);else ((float *)data)[c*k+r]=weight(part,r,c);}
    check(mesh_tensor_constant(weights,(uint32_t)part));
  }
  for(size_t slot=0;slot<depth;slot++)programs[slot]=configure(a,rank,rows,k,n,weights,symbolic,half);
  size_t prefix_rows=(size_t)context.M->block*context.M->pgsz/(n*sizeof(float));
  for(size_t slot=0;slot<depth;slot++) {
    struct program *p=&programs[slot];
    for(uint32_t q=0;q<2;q++)for(uint32_t peer=0;peer<2;peer++)check(mesh_algebra_copy(a,(struct mesh_endpoint){p->partial,peer,q,1},(struct mesh_endpoint){p->peer_partial,1-peer,q,1},1,2));
    check(mesh_algebra_return(a,p->peer_partial,1));check(mesh_algebra_return_part(a,p->peer_partial,1,0));check(mesh_algebra_return(a,p->peer_partial,0));
    p->prefix=mesh_tensor_rows(p->x,0);p->tail=p->prefix;
    uint32_t prefix_pages=(uint32_t)(prefix_rows*k*sizeof(float)/context.M->pgsz);
    if(prefix_pages<p->prefix.count)p->prefix.count=prefix_pages;
    p->tail.first+=p->prefix.count;p->tail.count-=p->prefix.count;
    p->prefix_function=(struct mesh_row_function){.output=&p->prefix,.outputs=1,.rows=1};
    p->tail_function=(struct mesh_row_function){.output=&p->tail,.outputs=1,.rows=1};
  }
  check(mesh_algebra_realize(a));
  for(size_t slot=0;slot<depth;slot++)for(uint32_t i=slot?0:1;i<4;i++)produce(programs[slot].x,rank,i,rows,k,slot);
  double maxError=0,earlyMean=0,earlyM2=0,start=now();size_t windows=0,later=0,prefixes=0,early_k=0;
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
      while(!mesh_algebra_available(a,8*depth+2)) {
        mesh_algebra_scan(a);check((int)mesh_algebra_report(a).code);check((int)mesh_link_metadata(&context,0).code);
        if(now()-began>30){fprintf(stderr,"K contribution did not reach peer before missing panel\n");return 13;}
      }
      float *early=mesh_tensor_data(programs[0].peer_partial,0);
      for(size_t r=0;r<rows;r++)for(size_t c=0;c<n;c++) {
        double expected=0;
        for(size_t j=0;j<k;j++)expected+=(double)activation((0.5f*(input(0,2,r,j)+(float)trial/256)+0.125f)+(0.5f*(input(1,2,r,j)+(float)trial/256)+0.125f),half)*weight(1,j,c);
        double error=fabs(early[r*n+c]-expected);if(error>maxError)maxError=error;
        if(!isfinite(early[r*n+c]) || error>(half?1e-3:2e-4)){fprintf(stderr,"early K contribution mismatch row %zu column %zu actual %.9g expected %.9g error %.9g\n",r,c,early[r*n+c],expected,error);return 14;}
      }
      early_k++;
      if(rows>prefix_rows) {
        struct program *p=&programs[slot];uint32_t zero=0;
        check(mesh_tensor_issue(p->x,0)?0:EBUSY);float *data=mesh_tensor_data(p->x,0);
        for(size_t r=0;r<prefix_rows;r++)for(size_t c=0;c<k;c++)data[r*k+c]=input(rank,0,r,c)+(float)trial/256;
        mesh_complete(&context,&p->prefix_function,&zero,1);
        while(!mesh_algebra_available(a,8*depth+1)) {
          mesh_algebra_scan(a);check((int)mesh_algebra_report(a).code);check((int)mesh_link_metadata(&context,0).code);
          if(now()-began>30){fprintf(stderr,"contraction prefix did not reach peer before input tail\n");return 9;}
        }
        if(mesh_algebra_available(a,8*depth)){fprintf(stderr,"contraction tail published without input\n");return 10;}
        float *received=mesh_tensor_data(p->peer_partial,1);
        for(size_t r=0;r<prefix_rows;r++)for(size_t c=0;c<n;c++) {
          double expected=0;
          for(size_t j=0;j<k;j++)expected+=(double)activation((0.5f*(input(0,0,r,j)+(float)trial/256)+0.125f)+(0.5f*(input(1,0,r,j)+(float)trial/256)+0.125f),half)*weight(0,j,c);
          double error=fabs(received[r*n+c]-expected);if(error>maxError)maxError=error;
          if(!isfinite(received[r*n+c]) || error>(half?1e-3:2e-4)){fprintf(stderr,"received contraction prefix mismatch\n");return 11;}
        }
        prefixes++;
        for(size_t r=prefix_rows;r<rows;r++)for(size_t c=0;c<k;c++)data[r*k+c]=input(rank,0,r,c)+(float)trial/256;
        mesh_complete(&context,&p->tail_function,&zero,1);
      } else produce(programs[slot].x,rank,0,rows,k,trial);
    }
    for(;;) {
      mesh_algebra_scan(a);int ready=1;
      for(size_t j=0;j<8;j++)ready=ready && mesh_algebra_available(a,base+j);
      ready=ready && mesh_algebra_available(a,8*depth+3*slot);
      check((int)mesh_algebra_report(a).code);check((int)mesh_link_metadata(&context,0).code);
      if(ready)break;
      if(now()-began>30){fprintf(stderr,"completion timeout invocation %zu\n",trial);return 5;}
    }
    check(verify(programs[slot],rank,rows,k,n,trial,&maxError,half));
    float *sent=mesh_tensor_data(programs[slot].partial,1),*received=mesh_tensor_data(programs[slot].peer_partial,1);
    for(size_t i=0;i<rows*n;i++)if(!isfinite(received[i]) || fabs(received[i]-sent[i])>(half?2e-3:2e-4)){fprintf(stderr,"contraction peer copy mismatch\n");return 12;}
    for(size_t j=0;j<8;j++)mesh_algebra_consume(a,base+j);
    mesh_algebra_consume(a,8*depth+3*slot);mesh_algebra_consume(a,8*depth+3*slot+1);mesh_algebra_consume(a,8*depth+3*slot+2);
    size_t next=trial+depth;
    if(next<invocations)for(uint32_t i=slot?0:1;i<4;i++)produce(programs[slot].x,rank,i,rows,k,next);
  }
  struct mesh_algebra_report report=mesh_algebra_report(a);
  while(report.completed!=report.submitted){mesh_algebra_scan(a);report=mesh_algebra_report(a);check((int)report.code);}
  char variance[32];snprintf(variance,sizeof variance,windows>1?"%.9g":"null",windows>1?earlyM2/(windows-1):0);
  printf("{\"rank\":%d,\"symbolic\":%d,\"half_contraction_inputs\":%d,\"invocations\":%zu,\"depth\":%zu,\"rows\":%zu,\"k\":%zu,\"n\":%zu,\"delayed_windows\":%zu,\"later_invocation_completions\":%zu,\"received_prefixes_before_tail\":%zu,\"received_k_contributions_before_panel\":%zu,\"max_absolute_error\":%.9g,\"early_window_ms\":{\"count\":%zu,\"mean\":%.9g,\"sample_variance\":%s},\"allocated_pages\":%u,\"commands\":%llu,\"cpu_submissions\":%llu,\"native_submissions\":%llu,\"native_output_backings\":%llu,\"ne_planned_operations\":%llu,\"gpu_seconds\":%.9g,\"wall_seconds\":%.9g}\n",rank,symbolic,half,invocations,depth,rows,2*k,n,windows,later,prefixes,early_k,maxError,windows,earlyMean,variance,context.arena,(unsigned long long)report.completed,(unsigned long long)report.cpu_submitted,(unsigned long long)report.native_submitted,(unsigned long long)report.native_backings,(unsigned long long)report.ne_planned_operations,report.gpu_seconds,now()-start);
  free(programs);mesh_algebra_destroy(a);check(mesh_detach(&context));return 0;
}
