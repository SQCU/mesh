#include "mesh-plan.h"
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void writes(void){
  const struct mesh_cable cables[]={
    {11,1,{0,1}},{12,2,{0,1}},{13,4,{0,1}},{14,1,{0,2}},{15,2,{1,2}}
  };
  const struct mesh_fabric f={3,5,cables};
  const uint32_t first[]={0,1,2,3,5},path[]={0,1,2,3,4};
  const double weight[]={0,1,2,3};
  const struct mesh_paths paths={4,first,path,weight};
  assert(!mesh_paths_check(&f,0,1,&paths));
  for(uint64_t elements=1;elements<100000;elements+=137){
    const struct mesh_operand operand={0,4,elements};
    struct mesh_transfer *steps[3];
    int count[3];
    unsigned char *coverage=calloc(elements,1);
    for(uint32_t r=0;r<3;r++){
      count[r]=mesh_write_compile(&f,r,0,1,&paths,operand,16384,2,NULL,0);
      assert(count[r]>=0);
      steps[r]=calloc((size_t)count[r]+1,sizeof **steps);
      assert(mesh_write_compile(&f,r,0,1,&paths,operand,16384,2,steps[r],count[r])==count[r]);
    }
    unsigned used=0;
    for(uint32_t r=0;r<3;r++)for(int k=0;k<count[r];k++){
      const struct mesh_transfer *t=steps[r]+k;
      const struct mesh_step *s=&t->step;
      assert(s->first%4096==0 && s->piece.elements>0 && s->piece.elements<=8192);
      assert(s->first+s->piece.elements<=elements);
      int matches=0;
      for(int j=0;j<count[s->peer];j++){
        const struct mesh_transfer *u=steps[s->peer]+j;
        matches+=u->cable==t->cable && u->step.peer==r && u->step.op!=s->op && u->step.round==s->round &&
          u->step.first==s->first && u->step.piece.elements==s->piece.elements;
      }
      assert(matches==1);
      if(r==1 && s->op==MESH_STEP_COPY)for(uint64_t i=s->first;i<s->first+s->piece.elements;i++)coverage[i]++;
      if(r==2 && s->op==MESH_STEP_SEND){
        int dependency=0;
        for(int j=0;j<k;j++)dependency+=steps[r][j].step.op==MESH_STEP_COPY && steps[r][j].step.first==s->first &&
          steps[r][j].step.piece.elements==s->piece.elements;
        assert(dependency==1);
      }
      used|=1u<<t->cable;
    }
    for(uint64_t i=0;i<elements;i++)assert(coverage[i]==1);
    if(elements>3*4096)assert(used==31);
    for(int r=0;r<3;r++)free(steps[r]);
    free(coverage);
  }
  const uint32_t bad_first[]={0,3},bad_cable[]={3,3,0};
  const struct mesh_paths loop={1,bad_first,bad_cable,weight};
  assert(mesh_paths_check(&f,0,1,&loop)==EINVAL);
  assert(mesh_paths_check(&f,0,0,&paths)==EINVAL);
  puts("one-way paths: reciprocal entries, unique destination coverage, relay dependencies, ragged tails passed");
}

int main(void){
  const struct mesh_cable cables[]={
    {11,1,{0,1}},{12,2,{0,1}},{13,4,{0,1}},{14,1,{0,2}},{15,2,{1,2}}
  };
  const struct mesh_fabric f={3,5,cables};
  assert(!mesh_fabric_check(&f));
  for(uint64_t elements=1;elements<100000;elements+=137){
    const struct mesh_step send={MESH_STEP_SEND,1,0,0,4096,{0,4,elements}};
    const struct mesh_step recv={MESH_STEP_COPY,0,0,0,4096,{0,4,elements}};
    struct mesh_transfer a[5],b[5];
    const int n=mesh_fabric_stripe(&f,0,&send,16384,NULL,a,5);
    const int m=mesh_fabric_stripe(&f,1,&recv,16384,NULL,b,5);
    assert(n>0 && n==m && n<=3);
    uint64_t at=4096;
    for(int i=0;i<n;i++){
      assert(a[i].cable==b[i].cable && a[i].step.first==b[i].step.first);
      assert(a[i].step.piece.elements==b[i].step.piece.elements);
      assert(a[i].step.first==at && at%4096==0 && a[i].step.piece.elements>0);
      at+=a[i].step.piece.elements;
    }
    assert(at==4096+elements);
    if(elements>8192)assert(n==3);
  }
  struct mesh_transfer out[5];
  const struct mesh_step send={MESH_STEP_SEND,1,0,0,0,{0,4,4096*703}};
  assert(mesh_fabric_stripe(&f,0,&send,16384,NULL,out,5)==3);
  assert(out[0].step.piece.elements==4096*101);
  assert(out[1].step.piece.elements==4096*201);
  assert(out[2].step.piece.elements==4096*401);
  assert(mesh_fabric_stripe(&f,0,&send,16384,NULL,out,2)==-ENOSPC);
  puts("physical fabric: reciprocal schedules, ragged conservation, all cables, unequal capacities passed");
  writes();
}
