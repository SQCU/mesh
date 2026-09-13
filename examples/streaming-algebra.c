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
/* design/algorithm-sources.md#online-percentage-moments */
static void moment(size_t n,double x,double *mean,double *m2) { double d=x-*mean;*mean+=d/n;*m2+=d*(x-*mean); }

/* design/algorithm-sources.md#streaming-algebra */
int main(int argc,char **argv) {
  int rank=argc>1?atoi(argv[1]):0;
  size_t repetitions=argc>2?(size_t)atoi(argv[2]):20;
  size_t rows=argc>3?(size_t)atoi(argv[3]):128,k=128,n=64;
  if((rank!=0 && rank!=1) || !repetitions || !rows)return 2;
  struct mesh_ctx context={0};check(mesh_attach(&context,NULL));
  if(context.M->qps<2){fprintf(stderr,"acceptance requires two independent configured transport queues\n");mesh_detach(&context);return 2;}
  struct mesh_algebra *a=mesh_algebra_create(&context);if(!a)check(errno);
  struct mesh_tensor *x=tensor(a,4,rows,k,0),*p=tensor(a,4,rows,k,1),*remote=tensor(a,4,rows,k,1);
  struct mesh_tensor *sum=tensor(a,4,rows,k,0),*activated=tensor(a,4,rows,k,0),*weights=tensor(a,2,n,k,0);
  struct mesh_tensor *partial=tensor(a,4,rows,n,0),*result=tensor(a,2,rows,n,1),*gathered=tensor(a,2,rows,n,1);
  struct mesh_tensor *statistics=tensor(a,2,rows,1,0);
  struct mesh_tensor *squared=tensor(a,2,rows,n,0),*sumsq=tensor(a,2,rows,1,0);
  struct mesh_tensor *meansq=tensor(a,2,rows,1,0),*inverse=tensor(a,2,rows,1,0),*normalized=tensor(a,2,rows,n,0);
  for(size_t i=0;i<4;i++) {
    float *data=mesh_tensor_data(x,(uint32_t)i);
    for(size_t r=0;r<rows;r++)for(size_t c=0;c<k;c++)data[r*k+c]=input(rank,i,r,c);
    struct mesh_view xv=mesh_tensor_view(x,(uint32_t)i),pv=mesh_tensor_view(p,(uint32_t)i),rv=mesh_tensor_view(remote,(uint32_t)i);
    struct mesh_view sv=mesh_tensor_view(sum,(uint32_t)i),av=mesh_tensor_view(activated,(uint32_t)i),cv=mesh_tensor_view(partial,(uint32_t)i);
    check(mesh_algebra_bind(a,MESH_AFFINE,xv,(struct mesh_view){0},pv,0.5f,0.125f));
    check(mesh_algebra_transfer(a,p,(uint32_t)i,(uint32_t)i,(uint16_t)(i%2),0));
    check(mesh_algebra_transfer(a,remote,(uint32_t)i,(uint32_t)i,(uint16_t)(i%2),1));
    check(mesh_algebra_bind(a,MESH_ADD,pv,rv,sv,1,1));
    check(mesh_algebra_bind(a,MESH_TANH,sv,(struct mesh_view){0},av,0,0));
    check(mesh_algebra_bind(a,MESH_CONTRACT,av,mesh_view_transpose(mesh_tensor_view(weights,(uint32_t)(i/2))),cv,1,0));
  }
  for(size_t part=0;part<2;part++) {
    float *data=mesh_tensor_data(weights,(uint32_t)part);
    for(size_t r=0;r<k;r++)for(size_t c=0;c<n;c++)data[c*k+r]=weight(part,r,c);
    check(mesh_tensor_constant(weights,(uint32_t)part));
  }
  for(uint32_t group=0;group<2;group++) {
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
  check(mesh_algebra_transfer(a,result,(uint32_t)rank,100+(uint32_t)rank,(uint16_t)rank,0));
  check(mesh_algebra_transfer(a,gathered,(uint32_t)(1-rank),100+(uint32_t)(1-rank),(uint16_t)(1-rank),1));
  check(mesh_algebra_return(a,gathered,(uint32_t)(1-rank)));
  for(uint32_t group=0;group<2;group++)check(mesh_algebra_return(a,statistics,group));
  check(mesh_algebra_return(a,p,0));
  for(uint32_t group=0;group<2;group++)check(mesh_algebra_return(a,normalized,group));
  check(mesh_algebra_realize(a));
  double means[2]={0},m2[2]={0},earlyMean=0,earlyM2=0,maxError=0;
  size_t independent=0;double start=now();
  for(size_t trial=0;trial<2*repetitions+4;trial++) {
    int delayed=trial>=4 && (trial-4)%2;
    double began=now();
    for(uint32_t i=delayed?1:0;i<4;i++)check(mesh_tensor_publish(x,i)?0:EBUSY);
    if(delayed) {
      while(!mesh_algebra_available(a,1)) {
        mesh_algebra_scan(a);
        if(now()-began>30){fprintf(stderr,"independent extent did not complete while extent zero was absent\n");return 3;}
      }
      if(mesh_algebra_available(a,5) || mesh_algebra_available(a,0)){fprintf(stderr,"premature publication\n");return 4;}
      independent++;
      moment(independent,(now()-began)*1e3,&earlyMean,&earlyM2);
      check(mesh_tensor_publish(x,0)?0:EBUSY);
    }
    for(;;) {
      mesh_algebra_scan(a);
      struct mesh_algebra_report report=mesh_algebra_report(a);
      check((int)report.code);
      struct mesh_row_metadata link=mesh_link_metadata(&context,0);check((int)link.code);
      int ready=report.submitted==report.completed;
      for(size_t i=0;i<8;i++)ready=ready && mesh_algebra_available(a,i);
      if(ready)break;
      if(now()-began>30){
        fprintf(stderr,"completion timeout trial %zu commands %llu/%llu returns",trial,(unsigned long long)report.completed,(unsigned long long)report.submitted);
        for(size_t i=0;i<8;i++)fprintf(stderr," %d",mesh_algebra_available(a,i));
        fprintf(stderr,"\n");
        struct mesh_tensor *tensors[]={x,p,remote,sum,activated,partial};
        for(size_t t=0;t<6;t++){for(uint32_t i=0;i<4;i++)fprintf(stderr," %d",mesh_present(&context,mesh_tensor_rows(tensors[t],i),0));fprintf(stderr,"\n");}
        return 5;
      }
    }
    if(trial>=4)moment((trial-4)/2+1,(now()-began)*1e3,&means[delayed],&m2[delayed]);
    for(size_t group=0;group<2;group++) {
      float *actual=mesh_tensor_data(result,(uint32_t)group);
      float *peer=group==(size_t)(1-rank)?mesh_tensor_data(gathered,(uint32_t)group):actual;
      float *stat=mesh_tensor_data(statistics,(uint32_t)group);
      float *normal=mesh_tensor_data(normalized,(uint32_t)group);
      for(size_t r=0;r<rows;r++) {
        double total=0,square=0;
        for(size_t c=0;c<n;c++) {
          double expected=0;
          for(size_t part=0;part<2;part++)for(size_t j=0;j<k;j++) {
            size_t extent=group+part*2;
            float v=(0.5f*input(0,extent,r,j)+0.125f)+(0.5f*input(1,extent,r,j)+0.125f);
            expected+=(double)tanhf(v)*weight(part,j,c);
          }
          double error=fabs(actual[r*n+c]-expected);if(error>maxError)maxError=error;
          if(!isfinite(actual[r*n+c]) || error>2e-4 || peer[r*n+c]!=actual[r*n+c]){fprintf(stderr,"numerical mismatch %zu %zu %zu %.9g %.9g\n",group,r,c,actual[r*n+c],expected);return 6;}
          total+=actual[r*n+c];square+=(double)actual[r*n+c]*actual[r*n+c];
        }
        for(size_t c=0;c<n;c++)if(!isfinite(normal[r*n+c]) || fabs(normal[r*n+c]-actual[r*n+c]/sqrt(square/n+1e-5))>2e-5){fprintf(stderr,"normalization composition mismatch\n");return 8;}
        if(!isfinite(stat[r]) || fabs(stat[r]-total)>2e-4){fprintf(stderr,"row reduction mismatch\n");return 7;}
      }
    }
    for(size_t i=0;i<8;i++)mesh_algebra_consume(a,i);
  }
  struct mesh_algebra_report report=mesh_algebra_report(a);
  printf("{\"rank\":%d,\"rows\":%zu,\"k\":%zu,\"n\":%zu,\"independent_completions\":%zu,\"max_absolute_error\":%.9g,\"normal_ms\":{\"count\":%zu,\"mean\":%.9g,\"sample_variance\":%.9g},\"delayed_ms\":{\"count\":%zu,\"mean\":%.9g,\"sample_variance\":%.9g},\"early_ms\":{\"count\":%zu,\"mean\":%.9g,\"sample_variance\":%.9g},\"commands\":%llu,\"gpu_seconds\":%.9g,\"wall_seconds\":%.9g}\n",rank,rows,2*k,n,independent,maxError,repetitions,means[0],repetitions>1?m2[0]/(repetitions-1):0,repetitions,means[1],repetitions>1?m2[1]/(repetitions-1):0,independent,earlyMean,independent>1?earlyM2/(independent-1):0,(unsigned long long)report.completed,report.gpu_seconds,now()-start);
  mesh_algebra_destroy(a);check(mesh_detach(&context));return 0;
}
