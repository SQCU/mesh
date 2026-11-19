#include "mesh-plan.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N=6, E=6, VALUES=37 };
#define POISON INT64_C(-123456789012345)
static uint32_t links[E][2]={{0,1},{0,1},{1,2},{1,3},{3,4},{3,5}};
static uint64_t capacity[E]={8000000000ull,16000000000ull,8000000000ull,24000000000ull,16000000000ull,8000000000ull};
static uint32_t roots[N*N],parents[N*N*N];
static double weights[N*N];
static struct mesh_trees trees;

struct compiled {
  struct mesh_program program;
  uint32_t *segment,*first;
  double *weight;
  struct mesh_move *moves;
  struct mesh_step *steps[N];
  uint32_t count[N];
};

static struct compiled compile(uint32_t what,uint32_t root,int whole,uint64_t input,uint64_t output,
                               const uint64_t *counts,uint64_t elements){
  struct compiled c={0};uint32_t parts,moves;
  mesh_compile_room(&trees,&parts,&moves);
  c.segment=calloc(parts,sizeof *c.segment);c.first=calloc(N+1,sizeof *c.first);
  c.weight=calloc(parts,sizeof *c.weight);c.moves=calloc(moves,sizeof *c.moves);
  assert(c.segment&&c.first&&c.weight&&c.moves);
  assert(!mesh_compile_members(&trees,what,root,whole,&input,&output,&c.program,c.segment,c.weight,c.first,c.moves));
  for(uint32_t r=0;r<N;r++){
    c.steps[r]=calloc(moves,sizeof *c.steps[r]);assert(c.steps[r]);
    c.count[r]=mesh_program_steps_cut(&c.program,r,(struct mesh_operand){0,8,elements},counts,1,c.steps[r]);
  }
  return c;
}

static void release(struct compiled *c){
  for(uint32_t r=0;r<N;r++)free(c->steps[r]);
  free(c->segment);free(c->first);free(c->weight);free(c->moves);
}

static int64_t value(uint32_t rank,uint64_t at){return 100*(rank+1)+(int64_t)at;}

/* Evaluate the actual compiler steps. SENDs read the previous round's state;
   receives run in the emitted order. Every noncontributor buffer starts poison,
   including scratch, so a REDUCE before first COPY or a relay operand read fails.
   This proves logical dataflow, not asynchronous NIC retirement. */
static void check(uint32_t what,uint32_t root,int whole,uint64_t input,uint64_t output,const uint64_t *counts){
  uint64_t elements=VALUES,starts[N+1]={0};
  if(counts){for(uint32_t r=0;r<N;r++)starts[r+1]=starts[r]+counts[r];elements=starts[N];}
  assert(elements<=VALUES);
  struct compiled c=compile(what,root,whole,input,output,counts,elements);
  assert(!mesh_program_check_members(&c.program,(struct mesh_operand){0,8,elements},counts,1,&input,&output));
  const uint32_t buffers=c.program.scratch+1;
  int64_t *data[N];int64_t **message[N];unsigned char *delivered[N];uint32_t last=0;
  uint32_t pure_copy=0,pure_reduce=0;
  for(uint32_t r=0;r<N;r++){
    data[r]=malloc((size_t)buffers*VALUES*sizeof *data[r]);assert(data[r]);
    for(uint32_t b=0;b<buffers;b++)for(uint64_t x=0;x<VALUES;x++){
      const int own=(input>>r&1) && (what!=MESH_ALLGATHER || (x>=starts[r]&&x<starts[r+1]));
      data[r][b*VALUES+x]=own?value(r,x):POISON;
    }
    message[r]=calloc(c.count[r]+1,sizeof *message[r]);delivered[r]=calloc(c.count[r]+1,1);
    assert(message[r]&&delivered[r]);
    for(uint32_t k=0;k<c.count[r];k++){
      const struct mesh_step *s=c.steps[r]+k;
      assert(s->peer<N&&s->buffer<buffers&&s->first+s->piece.elements<=elements);
      if(s->round>last)last=s->round;
      if(!(input>>r&1)){pure_copy+=s->op==MESH_STEP_COPY;pure_reduce+=s->op==MESH_STEP_REDUCE;}
    }
  }
  for(uint32_t round=0;round<=last;round++){
    for(uint32_t r=0;r<N;r++)for(uint32_t k=0;k<c.count[r];k++){
      const struct mesh_step *s=c.steps[r]+k;
      if(s->op!=MESH_STEP_SEND || s->round!=round)continue;
      message[r][k]=malloc(s->piece.elements*sizeof(int64_t));assert(message[r][k]);
      for(uint64_t x=0;x<s->piece.elements;x++){
        const int64_t v=data[r][s->buffer*VALUES+s->first+x];assert(v!=POISON);message[r][k][x]=v;
      }
    }
    for(uint32_t r=0;r<N;r++)for(uint32_t k=0;k<c.count[r];k++){
      const struct mesh_step *s=c.steps[r]+k;
      if(s->op==MESH_STEP_SEND || s->round!=round)continue;
      const uint32_t from=s->peer;uint32_t found=c.count[from];
      for(uint32_t j=0;j<c.count[from];j++){
        const struct mesh_step *sent=c.steps[from]+j;
        if(sent->op==MESH_STEP_SEND && sent->peer==r && sent->round==round &&
           sent->first==s->first && sent->piece.elements==s->piece.elements && !delivered[from][j]){found=j;break;}
      }
      assert(found<c.count[from]&&message[from][found]);delivered[from][found]=1;
      for(uint64_t x=0;x<s->piece.elements;x++){
        int64_t *at=&data[r][s->buffer*VALUES+s->first+x];
        if(s->op==MESH_STEP_REDUCE){assert(*at!=POISON);*at+=message[from][found][x];}
        else {assert(s->op==MESH_STEP_COPY);*at=message[from][found][x];}
      }
    }
  }
  for(uint32_t r=0;r<N;r++){
    const int output_here=what==MESH_REDUCE?r==root:(output>>r&1);
    if(output_here)for(uint64_t x=0;x<elements;x++){
      if(what==MESH_REDUCE_SCATTER&&(x<starts[r]||x>=starts[r+1]))continue;
      int64_t expected=0;
      if(what==MESH_BROADCAST)expected=value(root,x);
      else if(what==MESH_ALLGATHER){
        uint32_t source=0;while(source<N&&x>=starts[source+1])source++;
        assert(source<N&&(input>>source&1));expected=value(source,x);
      }else for(uint32_t source=0;source<N;source++)if(input>>source&1)expected+=value(source,x);
      assert(data[r][x]==expected);
    }
    for(uint32_t k=0;k<c.count[r];k++){
      if(c.steps[r][k].op==MESH_STEP_SEND)assert(delivered[r][k]);free(message[r][k]);
    }
    free(message[r]);free(delivered[r]);free(data[r]);
  }
  assert(pure_copy);
  if(what==MESH_REDUCE || what==MESH_REDUCE_SCATTER || what==MESH_ALLREDUCE)assert(pure_reduce);
  if(what==MESH_ALLGATHER || what==MESH_REDUCE_SCATTER){
    assert(c.program.segments==N);
    const uint64_t owners=what==MESH_ALLGATHER?input:output;
    for(uint32_t p=0;p<c.program.parts;p++)assert(owners>>c.segment[p]&1);
  }
  release(&c);
}

static void numbering_and_legacy(void){
  const uint64_t inputs=(1u<<0)|(1u<<2),outputs=(1u<<4)|(1u<<5),counts[N]={3,0,5,0,0,0};
  struct compiled c=compile(MESH_ALLGATHER,0,0,inputs,outputs,counts,8);
  int seen0=0,seen2=0;
  for(uint32_t p=0;p<c.program.parts;p++){seen0|=c.segment[p]==0;seen2|=c.segment[p]==2;}
  assert(seen0&&seen2&&c.program.segments==N);
  /* A nonzero trailing count outside contributors must not disappear. */
  const uint64_t invalid[N]={3,0,5,0,0,1};
  struct mesh_step temporary[1024];
  assert(!mesh_program_steps_cut(&c.program,4,(struct mesh_operand){0,8,9},invalid,1,temporary));
  assert(!mesh_compile(&trees,MESH_ALLGATHER,0,0,&inputs,&c.program,c.segment,c.weight,c.first,c.moves));
  assert(!c.program.segments);int seen1=0;
  for(uint32_t p=0;p<c.program.parts;p++){assert(c.segment[p]<2);seen1|=c.segment[p]==1;}
  assert(seen1);release(&c);
}

static void high_rank_bits(void){
  enum { nodes=70 };uint32_t root=65,parent[nodes];double weight=0;
  for(uint32_t r=0;r<nodes;r++)parent[r]=r<root?r+1:r>root?r-1:r;
  const struct mesh_trees tree={nodes,1,&root,parent,&weight};
  uint64_t input[2]={0,1u<<1},output[2]={0,1u<<5};uint32_t parts,moves;
  mesh_compile_room(&tree,&parts,&moves);
  uint32_t segment,first[nodes+1];struct mesh_move *move=calloc(moves,sizeof *move);struct mesh_program program;
  assert(!mesh_compile_members(&tree,MESH_ALLGATHER,0,0,input,output,&program,&segment,&weight,first,move));
  assert(segment==65&&program.segments==nodes&&first[nodes]==8);
  for(uint32_t r=0;r<65;r++)assert(first[r]==first[r+1]);
  free(move);
}

static void immutable_source_for_recipient(void){
  const uint64_t members=5;
  struct compiled c=compile(MESH_ALLREDUCE,0,1,members,members,NULL,VALUES);
  for(uint32_t r=0;r<N;r++)if(members>>r&1){
    int original_send=0,result_receive=0;
    for(uint32_t k=0;k<c.count[r];k++){
      const struct mesh_step *s=c.steps[r]+k;
      if(s->op==MESH_STEP_SEND){assert(s->buffer!=0);original_send|=s->buffer==1;}
      else result_receive|=s->buffer==0;
    }
    assert(original_send&&result_receive);
  }
  release(&c);
}

/* The earlier direct exchange submitted SEND from buffer0 and then REDUCEd
   into that same operand. Complete B->A first, so A's result is modified before
   the NIC reads A->B. Both public compiler entry points must keep A's original
   source intact; merely publishing its SEND earlier is insufficient. */
static void delayed_source_read(void){
  const uint32_t root[2]={0,1},parent[4]={0,0,1,1};const double weight[2]={0,0};
  const struct mesh_trees tree={2,2,root,parent,weight};
  uint32_t room_parts,room_moves;mesh_compile_room(&tree,&room_parts,&room_moves);
  uint32_t *segment=calloc(room_parts,sizeof *segment),first[3];
  double *shares=calloc(room_parts,sizeof *shares);struct mesh_move *move=calloc(room_moves,sizeof *move);
  assert(segment&&shares&&move);
  for(int legacy=0;legacy<2;legacy++){
    struct mesh_program program;
    const int made=legacy?mesh_compile(&tree,MESH_ALLREDUCE,0,1,NULL,&program,segment,shares,first,move):
      mesh_compile_members(&tree,MESH_ALLREDUCE,0,1,NULL,NULL,&program,segment,shares,first,move);
    assert(!made&&program.scratch==1);
    struct mesh_step steps[2][4];uint32_t count[2];
    for(uint32_t r=0;r<2;r++)count[r]=mesh_program_steps_cut(&program,r,(struct mesh_operand){0,8,1},NULL,1,steps[r]);
    for(uint32_t early=0;early<2;early++){
      int64_t data[2][2]={{3,3},{5,5}};
      for(uint32_t order=0;order<2;order++){
        const uint32_t from=order?1-early:early,to=1-from;
        const struct mesh_step *send=NULL,*receive=NULL;
        for(uint32_t k=0;k<count[from];k++)if(steps[from][k].op==MESH_STEP_SEND)send=&steps[from][k];
        for(uint32_t k=0;k<count[to];k++)if(steps[to][k].op==MESH_STEP_REDUCE)receive=&steps[to][k];
        assert(send&&receive&&send->buffer==1&&receive->buffer==0);
        /* This read occurs after the reverse receive has already changed the
           source rank's result when order==1. */
        data[to][receive->buffer]+=data[from][send->buffer];
      }
      assert(data[0][0]==8&&data[1][0]==8&&data[0][1]==3&&data[1][1]==5);
    }
  }
  free(segment);free(shares);free(move);
}

static void custom_program_members(void){
  uint32_t segment[3]={0,0,0},first[4]={0,1,3,4};double weights[3]={0,0,0};
  struct mesh_move moves[8]={
    {MESH_STEP_SEND,1,0,0,0,3},
    {MESH_STEP_COPY,0,0,0,0,3},{MESH_STEP_SEND,2,1,0,0,3},
    {MESH_STEP_COPY,1,1,0,0,3}};
  struct mesh_program program={.what=MESH_REDUCE,.root=2,.nodes=3,.parts=3,.segments=1,
    .segment=segment,.log_weight=weights,.first=first,.move=moves};
  const struct mesh_operand operand={0,4,9};const uint64_t input=1,output=4,none=0;
  assert(!mesh_program_check_members(&program,operand,NULL,1,&input,&output));
  /* Reduce's explicit root is required even if the recipient mask is empty. */
  assert(!mesh_program_check_members(&program,operand,NULL,1,&input,&none));
  moves[3].op=MESH_STEP_REDUCE;
  assert(mesh_program_check_members(&program,operand,NULL,1,&input,&output)==EINVAL);
  moves[3].op=MESH_STEP_COPY;
  moves[1].op=MESH_STEP_REDUCE;
  assert(mesh_program_check_members(&program,operand,NULL,1,&input,&output)==EINVAL);
  moves[1].op=MESH_STEP_COPY;
  struct mesh_move temporary=moves[1];moves[1]=moves[2];moves[2]=temporary;
  assert(mesh_program_check_members(&program,operand,NULL,1,&input,&output)==EINVAL);
  moves[2]=moves[1];moves[1]=temporary;
  /* A valid interval in another scratch buffer cannot initialize this read. */
  program.scratch=1;moves[1].buffer=1;
  assert(mesh_program_check_members(&program,operand,NULL,1,&input,&output)==EINVAL);
  moves[2].buffer=1;
  assert(!mesh_program_check_members(&program,operand,NULL,1,&input,&output));
  moves[1].buffer=moves[2].buffer=0;program.scratch=0;
  /* Copying disjoint head/tail intervals leaves an actual hole. */
  first[1]=2;first[2]=5;first[3]=7;
  moves[0]=(struct mesh_move){MESH_STEP_SEND,1,0,0,0,1};
  moves[1]=(struct mesh_move){MESH_STEP_SEND,1,0,0,2,1};
  moves[2]=(struct mesh_move){MESH_STEP_COPY,0,0,0,0,1};
  moves[3]=(struct mesh_move){MESH_STEP_COPY,0,0,0,2,1};
  moves[4]=(struct mesh_move){MESH_STEP_SEND,2,1,0,0,3};
  moves[5]=(struct mesh_move){MESH_STEP_COPY,1,1,0,0,1};
  moves[6]=(struct mesh_move){MESH_STEP_COPY,1,1,0,2,1};
  assert(mesh_program_check_members(&program,operand,NULL,1,&input,&output)==EINVAL);
  /* Removing the invalid relay SEND still cannot manufacture a complete output. */
  moves[4].parts=1;
  assert(mesh_program_check_members(&program,operand,NULL,1,&input,&output)==EINVAL);
  /* Adjacent and overlapping COPY intervals coalesce without defining a gap. */
  moves[0].parts=2;moves[2].parts=2;moves[4].parts=3;moves[5].parts=2;
  struct mesh_move before[8];memcpy(before,moves,sizeof moves);
  assert(!mesh_program_check_members(&program,operand,NULL,1,&input,&output));
  assert(!memcmp(before,moves,sizeof moves));
  /* Malformed extents and descriptors fail without interpreting their data. */
  assert(mesh_program_check_members(&program,(struct mesh_operand){0,4,UINT64_MAX},NULL,1,&input,&output)==EOVERFLOW);
  const uint64_t bad_cut=8;
  assert(mesh_program_check_members(&program,operand,&bad_cut,1,&input,&output)==EINVAL);
  moves[4].parts=UINT32_MAX;
  assert(mesh_program_check_members(&program,operand,NULL,1,&input,&output)==EINVAL);
  moves[4].parts=3;weights[0]=weights[1]=weights[2]=-INFINITY;
  assert(mesh_program_check_members(&program,operand,NULL,1,&input,&output)==EINVAL);
}

int main(void){
  const struct mesh_link_map map={.kind=MESH_LINKS_GRAPH,.nodes=N,.links=E,.link=links,.bandwidth=capacity};
  const uint32_t count=mesh_trees_pack(&map,NULL,0,N,roots,parents,weights);assert(count);
  trees=(struct mesh_trees){N,count,roots,parents,weights};assert(!mesh_trees_check(&map,&trees));
  const uint64_t input=5,output=48,gather[N]={3,0,5,0,0,0},scatter[N]={0,0,0,0,3,5};
  check(MESH_REDUCE,4,0,input,1u<<4,NULL);
  check(MESH_ALLGATHER,0,0,input,output,gather);
  check(MESH_REDUCE_SCATTER,0,0,input,output,scatter);
  check(MESH_ALLREDUCE,0,0,input,output,NULL);
  check(MESH_ALLREDUCE,0,1,input,output,NULL);
  /* Some contributors also receive, while another recipient has no input. */
  check(MESH_ALLREDUCE,0,0,input,input|(1u<<4),NULL);
  check(MESH_ALLREDUCE,0,1,input,input|(1u<<4),NULL);
  check(MESH_BROADCAST,2,0,1u<<2,output,NULL);
  const uint64_t reverse[N]={3,0,0,0,5,0};
  check(MESH_REDUCE_SCATTER,0,0,(1u<<2)|(1u<<5),(1u<<0)|(1u<<4),reverse);
  numbering_and_legacy();high_rank_bits();immutable_source_for_recipient();delayed_source_read();custom_program_members();
  struct compiled empty=compile(MESH_ALLGATHER,0,0,0,output,gather,8);
  assert(!empty.program.parts&&empty.first[N]==0);release(&empty);
  empty=compile(MESH_ALLREDUCE,0,0,input,0,NULL,VALUES);
  assert(!empty.program.parts&&empty.first[N]==0);
  const uint64_t none=0;
  assert(mesh_compile_members(&trees,MESH_REDUCE,4,0,&none,&output,&empty.program,
    empty.segment,empty.weight,empty.first,empty.moves)==EINVAL);
  release(&empty);
  puts("distinct contributors/recipients, relay poison, global zero gaps PASS");return 0;
}
