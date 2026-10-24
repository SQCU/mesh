#include "mesh-plan.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct mesh_transfer transfer(uint32_t op,uint32_t buffer,uint64_t first,uint64_t elements,uint32_t cable){
  return (struct mesh_transfer){.step={op,2,7,buffer,first,{9,4,elements}},.cable=cable};
}

static void refinement(void){
  struct mesh_transfer in[16],out[32];
  for(unsigned k=0;k<4;k++)in[k]=transfer(MESH_STEP_COPY,0,4096*k,4096,k);
  in[3].step.piece.elements-=17; /* A ragged final page remains the final page. */
  in[4]=transfer(MESH_STEP_SEND,0,0,8192,4);
  in[5]=transfer(MESH_STEP_SEND,0,8192,8192-17,5);
  assert(mesh_fabric_refine(in,6,16384,NULL,0)==8);
  assert(mesh_fabric_refine(in,6,16384,out,32)==8);
  for(unsigned k=0;k<4;k++)assert(!memcmp(in+k,out+k,sizeof *in));
  for(unsigned k=0;k<4;k++){
    assert(out[k+4].step.first==4096*k);
    assert(out[k+4].step.piece.elements==4096-(k==3?17:0));
    assert(out[k+4].cable==4+k/2 && out[k+4].step.round==7 && out[k+4].step.peer==2);
  }
  assert(mesh_fabric_refine(in,6,16384,out,7)==-ENOSPC);
  /* Overlapping latest COPYs partition the source; an older, now hidden
     non-page boundary must neither split the SEND nor make it invalid. */
  in[0]=transfer(MESH_STEP_COPY,0,0,4096*4,0);
  in[1]=transfer(MESH_STEP_COPY,0,4097,3,1);
  in[2]=transfer(MESH_STEP_COPY,0,4096,8192,2);
  in[3]=transfer(MESH_STEP_COPY,1,4099,3,3); /* another buffer is irrelevant */
  in[4]=transfer(MESH_STEP_SEND,0,0,4096*4,4);
  assert(mesh_fabric_refine(in,5,16384,out,32)==7);
  assert(out[4].step.first==0 && out[4].step.piece.elements==4096);
  assert(out[5].step.first==4096 && out[5].step.piece.elements==8192);
  assert(out[6].step.first==12288 && out[6].step.piece.elements==4096);
  in[2].step.piece.element_bytes=8; /* other element coordinates do not participate */
  assert(mesh_fabric_refine(in,5,16384,out,32)==-EINVAL);
  /* A latest REDUCE blocks identity forwarding only on its own pages. */
  in[0]=transfer(MESH_STEP_COPY,0,0,4096*4,0);
  in[1]=transfer(MESH_STEP_REDUCE,0,4096,8192,1);
  in[2]=transfer(MESH_STEP_SEND,0,0,4096*4,2);
  assert(mesh_fabric_refine(in,3,16384,out,32)==5);
  assert(out[2].step.piece.elements==4096 && out[3].step.piece.elements==8192 && out[4].step.piece.elements==4096);
  /* A COPY meeting only a portion of a SEND leaves both uncovered sides. */
  in[0]=transfer(MESH_STEP_COPY,0,4096,4096,0);
  in[1]=transfer(MESH_STEP_SEND,0,0,4096*3-1,1);
  assert(mesh_fabric_refine(in,2,16384,out,32)==4);
  assert(out[1].step.first==0 && out[1].step.piece.elements==4096);
  assert(out[2].step.first==4096 && out[2].step.piece.elements==4096);
  assert(out[3].step.first==8192 && out[3].step.piece.elements==4095);
  in[0].step.first++;
  assert(mesh_fabric_refine(in,2,16384,out,32)==-EINVAL);
  in[0]=transfer(MESH_STEP_COPY,0,1,4096*4,0);
  in[1]=transfer(MESH_STEP_SEND,0,4096,8192,1);
  assert(mesh_fabric_refine(in,2,16384,out,32)==-EINVAL);
  /* Exhaust ragged extents and overlapping page intervals. Every useful
     element and every wire page remains in the same SEND/cable order. */
  for(uint64_t pages=2;pages<=12;pages++)for(uint64_t tail=1;tail<=4096;tail+=257){
    const uint64_t elements=(pages-1)*4096+tail;
    in[0]=transfer(MESH_STEP_COPY,0,0,elements,0);
    in[1]=transfer(MESH_STEP_COPY,0,4096,elements-4096,1);
    in[2]=transfer(MESH_STEP_SEND,0,0,elements,2);
    const int n=mesh_fabric_refine(in,3,16384,out,32);
    assert(n==4);
    uint64_t at=0,wire_pages=0;
    for(int k=2;k<n;k++){
      const struct mesh_step *s=&out[k].step;
      assert(s->first==at && s->piece.elements && out[k].cable==2);
      at+=s->piece.elements;wire_pages+=(s->piece.elements+4095)/4096;
    }
    assert(at==elements && wire_pages==pages);
  }
  in[0]=transfer(MESH_STEP_SEND,0,UINT64_MAX,1,0);
  assert(mesh_fabric_refine(in,1,16384,out,32)==-EOVERFLOW);
  assert(mesh_fabric_refine(NULL,0,16384,NULL,0)==0);
  puts("relay refinement: latest COPY intervals, overlapping writes, ragged conservation, unchanged wire pages passed");
}

static void point_layout(void){
  const struct mesh_step send[]={
    {MESH_STEP_SEND,1,0,0,0,{0,4,4096}},
    {MESH_STEP_SEND,1,0,0,4096,{0,4,8192}},
    {MESH_STEP_SEND,1,0,0,12288,{0,4,4095}},
    {MESH_STEP_SEND,1,0,0,16383,{0,4,1}}
  };
  struct mesh_step recv[4];
  for(unsigned k=0;k<4;k++){recv[k]=send[k];recv[k].op=MESH_STEP_COPY;recv[k].peer=0;}
  const uint32_t counts[]={3,3},last_counts[]={1,1};
  const struct mesh_step *steps[]={send,recv},*last[]={send+3,recv+3};
  const struct mesh_group_call calls[]={
    {MESH_GROUP_POINT,0,counts,steps},{MESH_GROUP_POINT,0,last_counts,last}
  };
  const struct mesh_group group={2,2,calls,16384};
  struct mesh_place scratch[4];
  struct mesh_group_place made[2][4];
  uint32_t placed[2];
  for(unsigned r=0;r<2;r++){
    assert(mesh_group_lay(&group,r,1-r,!r,9,scratch,made[r],placed+r)==13);
    assert(placed[r]==4);
    for(unsigned k=0;k<4;k++){
      assert(made[r][k].call==(k==3));
      assert(made[r][k].place.step==(k==3?0:k));
      assert(made[r][k].place.entry==9+k);
    }
  }
  puts("point layout: every physical stripe retained in both directions and across calls passed");
}

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
      count[r]=mesh_write_compile(&f,r,0,1,&paths,operand,16384,NULL,0);
      assert(count[r]>=0);
      steps[r]=calloc((size_t)count[r]+1,sizeof **steps);
      assert(mesh_write_compile(&f,r,0,1,&paths,operand,16384,steps[r],count[r])==count[r]);
    }
    unsigned used=0;
    for(uint32_t r=0;r<3;r++)for(int k=0;k<count[r];k++){
      const struct mesh_transfer *t=steps[r]+k;
      const struct mesh_step *s=&t->step;
      assert(s->first%4096==0 && s->piece.elements>0);
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
    assert(used);
    for(int r=0;r<3;r++)free(steps[r]);
    free(coverage);
  }
  const uint32_t bad_first[]={0,3},bad_cable[]={3,3,0};
  const struct mesh_paths loop={1,bad_first,bad_cable,weight};
  assert(mesh_paths_check(&f,0,1,&loop)==EINVAL);
  assert(mesh_paths_check(&f,0,0,&paths)==EINVAL);
  /* A negligible relay share must not take a mandatory page away from the
     requested 1:2:3 direct allocation. Every physical direct cable is used. */
  const double exact_weights[]={0,log(2),log(3),log(1e-10)};
  const struct mesh_paths exact={4,first,path,exact_weights};
  struct mesh_transfer exact_steps[4];
  assert(mesh_write_compile(&f,0,0,1,&exact,(struct mesh_operand){0,4,6*4096},16384,exact_steps,4)==3);
  for(int k=0;k<3;k++)assert(exact_steps[k].cable==(unsigned)k && exact_steps[k].step.piece.elements==(uint64_t)(k+1)*4096);
  assert(mesh_write_compile(&f,2,0,1,&exact,(struct mesh_operand){0,4,6*4096},16384,NULL,0)==0);
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
  refinement();
  point_layout();
}
