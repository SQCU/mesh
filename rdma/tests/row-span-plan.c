/* cc -std=c11 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined
      -Irdma rdma/tests/row-span-plan.c -o /tmp/mesh-row-span-plan-test */
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static size_t attempts,fail_at=SIZE_MAX,live;
static void *test_malloc(size_t bytes){
  if(attempts++==fail_at)return NULL;
  void *p=malloc(bytes);if(p)live++;return p;
}
static void *test_realloc(void *old,size_t bytes){
  if(attempts++==fail_at)return NULL;
  void *p=realloc(old,bytes);if(p && !old)live++;return p;
}
static void test_free(void *p){if(p){assert(live);live--;}free(p);}
#define malloc test_malloc
#define realloc test_realloc
#define free test_free
#include "mesh-row-span-plan.h"
#undef malloc
#undef realloc
#undef free

/* Independent byte oracle: a cut is required at a row edge, a row-relative
   packet grid, or any participant's actual registration crossing in this row.
   In particular, a crossing in another row does not split this row. */
static void oracle(const uint64_t *phases,const uint64_t *pitches,uint32_t participants,
                   uint64_t width,uint64_t rows,uint32_t packet,const struct mesh_span_plan *plan){
  const uint64_t bytes=width*rows;
  assert(plan->count && plan->cuts[0]==0 && plan->cuts[plan->count-1]==bytes);
  if(!bytes){assert(plan->count==1);return;}
  size_t next=1;
  for(uint64_t k=1;k<=bytes;k++){
    const uint64_t row=k/width,within=k%width;
    int boundary=!within || !(within%packet);
    for(uint32_t p=0;p<participants;p++)
      if((phases[p]+row*(pitches[p]%MESH_DMA_EXTENT)+within)%MESH_DMA_EXTENT==0)boundary=1;
    if(boundary){assert(next<plan->count && plan->cuts[next]==k);next++;}
  }
  assert(next==plan->count);
  for(size_t i=1;i<plan->count;i++){
    const uint64_t first=plan->cuts[i-1],length=plan->cuts[i]-first;
    assert(length && length<=packet && first/width==(plan->cuts[i]-1)/width);
    for(uint32_t p=0;p<participants;p++){
      const uint64_t start=(phases[p]+(first/width)*(pitches[p]%MESH_DMA_EXTENT)+first%width)%MESH_DMA_EXTENT;
      assert(length<=MESH_DMA_EXTENT-start);
    }
  }
}
static void example(void){
  const uint64_t phases[]={MESH_DMA_EXTENT-7,MESH_DMA_EXTENT-19,3};
  const uint64_t pitches[]={101,137,64};
  struct mesh_span_plan plan={0};
  assert(!mesh_row_span_plan_create(phases,pitches,3,64,9,23,&plan));
  oracle(phases,pitches,3,64,9,23,&plan);mesh_span_plan_free(&plan);assert(!live);
}
static void many_rows(void){
  const uint64_t phases[]={MESH_DMA_EXTENT-64,0},pitches[]={MESH_DMA_EXTENT+1,MESH_DMA_EXTENT+257};
  struct mesh_span_plan plan={0};
  assert(!mesh_row_span_plan_create(phases,pitches,2,64,10000,64,&plan));
  /* Only the first 63 nonzero row phases cross an MR inside this row. A
     union across all row phases would emit 640000 units instead of 10063. */
  assert(plan.count==10064);
  oracle(phases,pitches,2,64,10000,64,&plan);mesh_span_plan_free(&plan);assert(!live);
}
static void wide_row(void){
  const uint64_t phases[]={0,MESH_DMA_EXTENT-17},pitches[]={3*MESH_DMA_EXTENT,4*MESH_DMA_EXTENT};
  struct mesh_span_plan plan={0};
  assert(!mesh_row_span_plan_create(phases,pitches,2,2*MESH_DMA_EXTENT+9,2,UINT32_MAX,&plan));
  assert(plan.cuts[plan.count-1]==4*MESH_DMA_EXTENT+18);
  for(size_t i=1;i<plan.count;i++){
    const uint64_t a=plan.cuts[i-1],b=plan.cuts[i],row=a/(2*MESH_DMA_EXTENT+9),within=a%(2*MESH_DMA_EXTENT+9);
    assert(a<b && row==(b-1)/(2*MESH_DMA_EXTENT+9));
    for(unsigned p=0;p<2;p++)assert(b-a<=MESH_DMA_EXTENT-(phases[p]+row*pitches[p]+within)%MESH_DMA_EXTENT);
  }
  mesh_span_plan_free(&plan);assert(!live);
}
static uint64_t state=UINT64_C(0x65f2c31a71);
static uint32_t random32(void){state^=state<<13;state^=state>>7;state^=state<<17;return (uint32_t)state;}
static void randomized(void){
  for(unsigned test=0;test<400;test++){
    uint64_t phase[4],pitch[4];const uint32_t n=1+random32()%4,packet=1+random32()%80;
    const uint64_t width=1+random32()%100,rows=1+random32()%23;
    for(uint32_t p=0;p<n;p++){
      phase[p]=(MESH_DMA_EXTENT-1-random32()%300)%MESH_DMA_EXTENT;
      pitch[p]=random32()%2?random32()%200:MESH_DMA_EXTENT-random32()%200;
    }
    struct mesh_span_plan plan={0};assert(!mesh_row_span_plan_create(phase,pitch,n,width,rows,packet,&plan));
    oracle(phase,pitch,n,width,rows,packet,&plan);mesh_span_plan_free(&plan);assert(!live);
  }
}
static void failures(void){
  uint64_t phase=0,pitch=1;struct mesh_span_plan out={0};
  assert(mesh_row_span_plan_create(NULL,&pitch,1,1,1,1,&out)==EINVAL);
  assert(mesh_row_span_plan_create(&phase,&pitch,0,1,1,1,&out)==EINVAL);
  assert(mesh_row_span_plan_create(&phase,&pitch,1,1,1,0,&out)==EINVAL);
  assert(mesh_row_span_plan_create(&phase,&pitch,1,1,1,1,NULL)==EINVAL);
  phase=MESH_DMA_EXTENT;assert(mesh_row_span_plan_create(&phase,&pitch,1,1,1,1,&out)==EINVAL);phase=0;
  assert(mesh_row_span_plan_create(&phase,&pitch,1,UINT64_MAX,2,1,&out)==EOVERFLOW);
  assert(mesh_row_span_plan_create(&phase,&pitch,1,1,UINT64_MAX,1,&out)==EOVERFLOW);
  pitch=UINT64_MAX;assert(mesh_row_span_plan_create(&phase,&pitch,1,2,2,1,&out)==EOVERFLOW);
  assert(!out.cuts && !out.count && !live);
  pitch=0;assert(!mesh_row_span_plan_create(&phase,&pitch,1,5,0,1,&out));assert(out.count==1 && !out.cuts[0]);
  uint64_t *kept=out.cuts;assert(mesh_row_span_plan_create(&phase,&pitch,1,1,1,1,&out)==EINVAL && out.cuts==kept);
  mesh_span_plan_free(&out);
  pitch=UINT64_MAX;assert(!mesh_row_span_plan_create(&phase,&pitch,1,0,7,1,&out));
  assert(out.count==1 && !out.cuts[0]);mesh_span_plan_free(&out);
  /* Every allocation failure cleans both current-row and accumulated plans. */
  const uint64_t phases[]={MESH_DMA_EXTENT-4,3},pitches[]={17,23};
  attempts=0;assert(!mesh_row_span_plan_create(phases,pitches,2,31,8,19,&out));
  const size_t allocations=attempts;mesh_span_plan_free(&out);assert(!live);
  for(size_t k=0;k<allocations;k++){
    attempts=0;fail_at=k;
    assert(mesh_row_span_plan_create(phases,pitches,2,31,8,19,&out)==ENOMEM);
    assert(!out.cuts && !out.count && !live);
  }
  fail_at=SIZE_MAX;
}
int main(void){example();many_rows();wide_row();randomized();failures();puts("affine row span plan PASS");return 0;}
