#include "mesh-plan.h"
#include "mesh-write-routes.h"
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N=4, E=6, ELEMENTS=103 };
static const struct mesh_cable cables[E]={
  {1,8000000000ull,{0,1}},{2,16000000000ull,{0,1}},{3,8000000000ull,{1,2}},
  {4,24000000000ull,{2,3}},{5,8000000000ull,{0,3}},{6,16000000000ull,{1,3}}
};
static const struct mesh_fabric fabric={N,E,cables};
static struct mesh_paths paths[N*N];
static uint32_t path_first[N*N][128],path_edges[N*N][512];
static double weights[N*N][128];
static uint32_t stack[N];

static void enumerate(uint32_t source,uint32_t dest,uint32_t at,uint32_t seen,uint32_t depth){
  const uint32_t pair=source*N+dest;
  if(at==dest){
    const uint32_t p=paths[pair].count++;
    assert(p<127);
    const uint32_t offset=path_first[pair][p];
    assert(offset+depth<512);
    memcpy(path_edges[pair]+offset,stack,depth*sizeof *stack);
    path_first[pair][p+1]=offset+depth;
    /* Distinct positive proposals, including long alternate paths. */
    weights[pair][p]=(double)((p+3*source+dest)%5)/7;
    return;
  }
  for(uint32_t e=0;e<E;e++){
    if(cables[e].node[0]!=at && cables[e].node[1]!=at)continue;
    const uint32_t next=cables[e].node[cables[e].node[0]==at];
    if(seen&(1u<<next))continue;
    stack[depth]=e;enumerate(source,dest,next,seen|(1u<<next),depth+1);
  }
}

static int overlap(const struct mesh_step *a,const struct mesh_step *b){
  return a->buffer==b->buffer && a->first<b->first+b->piece.elements && b->first<a->first+a->piece.elements;
}

static int ready(const struct mesh_routed_rank *r,const unsigned char *done,uint32_t k){
  for(uint32_t j=0;j<k;j++)
    if(r->transfer[j].step.op!=MESH_STEP_SEND && overlap(&r->transfer[j].step,&r->transfer[k].step) && !done[j])return 0;
  return 1;
}

/* Model publication, physical arrival and local consumption separately.
   Wire arrivals remain FIFO; the consumer's order is the compiler event order.
   REDUCE payloads land in distinct staging, as in the production executor. */
static uint32_t waited_for_dma;
static void simulate_cables(const struct mesh_routed_program *p,int64_t ***buffers,uint32_t expected_cables){
  unsigned char *done[N],*published[N],*arrived[N],*deferred[N];int64_t **staged[N];uint32_t pending=0;
  for(uint32_t r=0;r<N;r++){
    const size_t n=p->rank[r].count+1;
    done[r]=calloc(n,1);published[r]=calloc(n,1);arrived[r]=calloc(n,1);deferred[r]=calloc(n,1);staged[r]=calloc(n,sizeof *staged[r]);
    for(uint32_t k=0;k<p->rank[r].count;k++)pending+=p->rank[r].transfer[k].step.op!=MESH_STEP_SEND;
  }
  uint32_t iteration=0,used=0;
  while(pending){
    int advanced=0;
    for(uint32_t r=0;r<N;r++)for(uint32_t k=0;k<p->rank[r].count;k++){
      const struct mesh_transfer *t=p->rank[r].transfer+k;const struct mesh_step *s=&t->step;
      if(s->op!=MESH_STEP_SEND || published[r][k] || !ready(p->rank+r,done[r],k))continue;
      int head=1;
      for(uint32_t j=0;j<p->rank[r].count;j++)
        if(p->rank[r].transfer[j].step.op==MESH_STEP_SEND && p->rank[r].transfer[j].cable==t->cable &&
           p->rank[r].transfer[j].step.round<s->round && !published[r][j])head=0;
      if(head){published[r][k]=1;advanced=1;}
    }
    for(uint32_t rr=0;rr<N;rr++){
      const uint32_t r=(rr+iteration)%N;const struct mesh_routed_rank *rank=p->rank+r;
      for(uint32_t kk=rank->count;kk>0;){
        const uint32_t k=--kk;const struct mesh_transfer *t=rank->transfer+k;const struct mesh_step *s=&t->step;
        if(s->op!=MESH_STEP_SEND || !published[r][k] || arrived[r][k])continue;
        /* Deliberately separate release from the NIC reading its source. */
        if(!deferred[r][k]){deferred[r][k]=1;advanced=1;continue;}
        int head=1;
        for(uint32_t j=0;j<rank->count;j++)
          if(rank->transfer[j].step.op==MESH_STEP_SEND && rank->transfer[j].cable==t->cable &&
             rank->transfer[j].step.round<s->round && !arrived[r][j])head=0;
        if(!head)continue;
        const struct mesh_routed_rank *peer=p->rank+s->peer;uint32_t match=peer->count;
        for(uint32_t j=0;j<peer->count;j++){
          const struct mesh_transfer *v=peer->transfer+j;
          if(v->step.op!=MESH_STEP_SEND && v->step.peer==r && v->cable==t->cable && v->step.round==s->round){assert(match==peer->count);match=j;}
        }
        assert(match<peer->count);const struct mesh_step *receive=&peer->transfer[match].step;
        assert(receive->piece.elements==s->piece.elements && receive->piece.element_bytes==s->piece.element_bytes);
        assert((s->first+s->piece.elements)*8<=rank->extent[s->buffer]);
        assert((receive->first+receive->piece.elements)*8<=peer->extent[receive->buffer]);
        staged[s->peer][match]=malloc(s->piece.elements*sizeof(int64_t));
        for(uint64_t x=0;x<s->piece.elements;x++){
          const int64_t value=buffers[r][s->buffer][s->first+x];assert(value!=-123456789);
          staged[s->peer][match][x]=value;
          if(receive->op==MESH_STEP_COPY)buffers[s->peer][receive->buffer][receive->first+x]=value;
        }
        arrived[r][k]=arrived[s->peer][match]=1;advanced=1;used|=1u<<t->cable;
      }
    }
    for(uint32_t r=0;r<N;r++)for(uint32_t k=0;k<p->rank[r].count;k++){
      const struct mesh_routed_rank *rank=p->rank+r;const struct mesh_transfer *t=rank->transfer+k;const struct mesh_step *s=&t->step;
      if(s->op==MESH_STEP_SEND || done[r][k] || !arrived[r][k] || !ready(rank,done[r],k))continue;
      int head=1;
      for(uint32_t j=0;j<rank->count;j++){
        const struct mesh_transfer *u=rank->transfer+j;
        if(u->cable!=t->cable)continue;
        if(u->step.op!=MESH_STEP_SEND && rank->consume[j]<rank->consume[k] && !done[r][j])head=0;
        if(s->op==MESH_STEP_REDUCE && j<k && u->step.op==MESH_STEP_SEND && overlap(&u->step,s) && !published[r][j])head=0;
      }
      if(!head)continue;
      if(s->op==MESH_STEP_REDUCE)for(uint32_t j=0;j<k;j++){
        const struct mesh_step *source=&rank->transfer[j].step;
        if(source->op==MESH_STEP_SEND && overlap(source,s) && !arrived[r][j]){head=0;waited_for_dma++;}
      }
      if(!head)continue;
      if(s->op==MESH_STEP_REDUCE)for(uint64_t x=0;x<s->piece.elements;x++)buffers[r][s->buffer][s->first+x]+=staged[r][k][x];
      done[r][k]=1;pending--;advanced=1;
    }
    assert(advanced);iteration++;
  }
  assert(used==expected_cables);
  for(uint32_t r=0;r<N;r++){
    for(uint32_t k=0;k<p->rank[r].count;k++)free(staged[r][k]);
    free(staged[r]);free(done[r]);free(published[r]);free(arrived[r]);free(deferred[r]);
  }
}
static void simulate(const struct mesh_routed_program *p,int64_t ***buffers){
  simulate_cables(p,buffers,(1u<<E)-1);
}

static void check_collective(uint32_t what,uint32_t root){
  uint32_t link[E][2];uint64_t bandwidth[E];
  for(uint32_t e=0;e<E;e++){memcpy(link[e],cables[e].node,sizeof link[e]);bandwidth[e]=cables[e].bandwidth;}
  const struct mesh_link_map map={.kind=MESH_LINKS_GRAPH,.nodes=N,.links=E,.link=link,.bandwidth=bandwidth};
  uint32_t roots[N*N],parents[N*N*N];double weight[N*N];
  const uint32_t trees=mesh_trees_pack(&map,NULL,0,N,roots,parents,weight);
  assert(trees);
  const struct mesh_trees tree={N,trees,roots,parents,weight};
  uint32_t parts,moves;mesh_compile_room(&tree,&parts,&moves);
  uint32_t *segment=calloc(parts,sizeof *segment),*first=calloc(N+1,sizeof *first);
  double *shares=calloc(parts,sizeof *shares);
  struct mesh_move *move=calloc(moves,sizeof *move);struct mesh_program program;
  assert(!mesh_compile(&tree,what,root,0,NULL,&program,segment,shares,first,move));
  struct mesh_step *logical=calloc(moves,sizeof *logical);uint32_t offsets[N+1]={0},initialized[N];
  const uint64_t ragged[N]={17,0,53,33};
  for(uint32_t r=0;r<N;r++){
    initialized[r]=1+program.scratch;
    offsets[r+1]=offsets[r]+mesh_program_steps_cut(&program,r,(struct mesh_operand){7,8,ELEMENTS},
      what==MESH_ALLGATHER || what==MESH_REDUCE_SCATTER?ragged:NULL,16,logical+offsets[r]);
  }
  struct mesh_routed_program physical;
  assert(!mesh_routed_compile(&fabric,paths,offsets,logical,initialized,16,&physical));
  int64_t **buffers[N];
  for(uint32_t r=0;r<N;r++){
    const struct mesh_routed_rank *rank=physical.rank+r;
    buffers[r]=calloc(rank->buffers,sizeof *buffers[r]);
    for(uint32_t b=0;b<rank->buffers;b++){
      const size_t elements=rank->extent[b]/8;
      buffers[r][b]=calloc(elements+ELEMENTS,sizeof **buffers);
      for(size_t x=0;x<elements+ELEMENTS;x++)buffers[r][b][x]=b<rank->initialized?100*r+(int64_t)x:-123456789;
    }
  }
  simulate(&physical,buffers);
  uint64_t own=0;
  for(uint32_t r=0;r<N;r++){
    if(what!=MESH_REDUCE || r==root)for(uint64_t x=0;x<ELEMENTS;x++){
      if(what==MESH_REDUCE_SCATTER && (x<own || x>=own+ragged[r]))continue;
      int64_t expected;
      if(what==MESH_BROADCAST)expected=100*root+(int64_t)x;
      else if(what==MESH_ALLGATHER){uint32_t source=0;uint64_t at=0;while(x>=at+ragged[source])at+=ragged[source++];expected=100*source+(int64_t)x;}
      else expected=600+N*(int64_t)x;
      assert(buffers[r][0][x]==expected);
    }
    own+=ragged[r];
    for(uint32_t b=0;b<physical.rank[r].buffers;b++)free(buffers[r][b]);free(buffers[r]);
  }
  uint64_t wanted=0,observed=0;
  for(uint32_t k=0;k<offsets[N];k++)if(logical[k].op==MESH_STEP_SEND)wanted+=logical[k].piece.elements*8;
  for(uint32_t k=0;k<N*N;k++)observed+=physical.demand[k];
  assert(wanted==observed);
  mesh_routed_free(&physical);free(segment);free(first);free(shares);free(move);free(logical);
}

static void endpoint_offsets(void){
  /* No direct 0--2 cable; two independent transactions, including a REDUCE,
     have distinct offsets and buffer IDs at their endpoints. */
  const uint32_t first[]={0,2,2,4,4},initialized[]={2,1,2,1};
  struct mesh_step steps[]={
    {MESH_STEP_SEND,2,0,0,3,{7,8,103}}, {MESH_STEP_SEND,2,0,1,5,{7,8,47}},
    {MESH_STEP_COPY,0,0,1,9,{7,8,103}}, {MESH_STEP_REDUCE,0,0,0,7,{7,8,47}}
  };
  struct mesh_routed_program result;
  assert(!mesh_routed_compile(&fabric,paths,first,steps,initialized,16,&result));
  assert(result.demand[2]==150*8);
  assert(result.rank[1].buffers>1 && result.rank[3].buffers>1);
  for(uint32_t r=1;r<N;r+=2)for(uint32_t b=1;b<result.rank[r].buffers;b++)
    assert(result.rank[r].extent[b]==103*8 || result.rank[r].extent[b]==47*8);
  mesh_routed_free(&result);
  steps[3].piece.elements--;
  assert(mesh_routed_compile(&fabric,paths,first,steps,initialized,16,&result)==EPROTO);
  assert(!result.rank && !result.demand);
}

static void counted(void){
  uint64_t matrix[N*N]={0},segments[2*N];
  /* Rank 3 owns no payload but provides alternate inbound/outbound capacity. */
  for(uint32_t s=0;s<N-1;s++)for(uint32_t d=0;d<N-1;d++)matrix[s*N+d]=41+7*s+11*d;
  struct mesh_step logical[2*N*N];uint32_t first[N+1]={0},initialized[N];
  for(uint32_t r=0;r<N;r++){
    initialized[r]=2;
    for(uint32_t q=0;q<N;q++){segments[q]=matrix[r*N+q];segments[N+q]=matrix[q*N+r];}
    first[r+1]=first[r]+mesh_counted_steps(N,r,segments,(struct mesh_operand){7,8,0},logical+first[r]);
  }
  struct mesh_routed_program physical;
  assert(!mesh_routed_compile(&fabric,paths,first,logical,initialized,16,&physical));
  assert(first[N]==first[N-1] && physical.rank[N-1].count && physical.rank[N-1].buffers>2);
  int64_t **buffers[N];
  for(uint32_t r=0;r<N;r++){
    const struct mesh_routed_rank *rank=physical.rank+r;
    buffers[r]=calloc(rank->buffers,sizeof *buffers[r]);
    for(uint32_t b=0;b<rank->buffers;b++){
      buffers[r][b]=malloc((rank->extent[b]/8+1024)*sizeof(int64_t));
      for(uint64_t x=0;x<rank->extent[b]/8+1024;x++)buffers[r][b][x]=-123456789;
    }
    uint64_t at=0;
    for(uint32_t d=0;d<N;d++)for(uint64_t x=0;x<matrix[r*N+d];x++)buffers[r][0][at++]=100000*r+1000*d+(int64_t)x;
  }
  simulate(&physical,buffers);
  for(uint32_t d=0;d<N;d++){
    uint64_t at=0;
    for(uint32_t s=0;s<N;s++){
      assert(physical.demand[s*N+d]==(s==d?0:8*matrix[s*N+d]));
      for(uint64_t x=0;x<matrix[s*N+d];x++,at++)if(s!=d)assert(buffers[d][1][at]==100000*s+1000*d+(int64_t)x);
    }
    for(uint32_t b=0;b<physical.rank[d].buffers;b++)free(buffers[d][b]);free(buffers[d]);
  }
  mesh_routed_free(&physical);
}

static void relayed_predecessor(void){
  /* 0's COPY via 1 must precede 1's REDUCE into the same bytes at 2.
     Both logical transfers are round zero. Sorting physical hops first puts
     the direct REDUCE ahead of its predecessor on cable 1--2 and deadlocks. */
  const uint32_t first[]={0,1,2,4,4},initialized[]={1,1,1,1};
  const struct mesh_step logical[]={
    {MESH_STEP_SEND,2,0,0,0,{7,8,17}},
    {MESH_STEP_SEND,2,0,0,0,{7,8,17}},
    {MESH_STEP_COPY,0,0,0,0,{7,8,17}},
    {MESH_STEP_REDUCE,1,0,0,0,{7,8,17}}
  };
  const uint32_t a_first[]={0,2},a_edge[]={0,2},b_first[]={0,1},b_edge[]={2};
  const double weight[]={0};
  struct mesh_paths fixed[N*N]={0};
  fixed[2]=(struct mesh_paths){1,a_first,a_edge,weight};
  fixed[N+2]=(struct mesh_paths){1,b_first,b_edge,weight};
  struct mesh_routed_program p;
  assert(!mesh_routed_compile(&fabric,fixed,first,logical,initialized,16,&p));
  int64_t **data[N];
  for(uint32_t r=0;r<N;r++){
    data[r]=calloc(p.rank[r].buffers,sizeof *data[r]);
    for(uint32_t b=0;b<p.rank[r].buffers;b++){
      data[r][b]=malloc((p.rank[r].extent[b]+8));
      for(uint64_t x=0;x<p.rank[r].extent[b]/8;x++)data[r][b][x]=b? -123456789:100*r+(int64_t)x;
    }
  }
  simulate_cables(&p,data,(1u<<0)|(1u<<2));
  for(uint32_t x=0;x<17;x++)assert(data[2][0][x]==100+2*(int64_t)x);
  for(uint32_t r=0;r<N;r++){for(uint32_t b=0;b<p.rank[r].buffers;b++)free(data[r][b]);free(data[r]);}
  mesh_routed_free(&p);
}

static void publication_before_combine(void){
  const uint32_t first[]={0,3,6,6,6},initialized[]={1,3,1,1};
  const struct mesh_step steps[]={
    {MESH_STEP_COPY,1,0,0,0,{7,8,1}}, {MESH_STEP_SEND,1,0,0,0,{7,8,3}}, {MESH_STEP_REDUCE,1,0,0,1,{7,8,2}},
    {MESH_STEP_SEND,0,0,1,1,{7,8,2}}, {MESH_STEP_SEND,0,0,0,0,{7,8,1}}, {MESH_STEP_COPY,0,0,2,0,{7,8,3}}
  };
  const uint32_t one[]={0,1},edge[]={0};const double w[]={0};struct mesh_paths routes[N*N]={0};
  routes[1]=routes[N]=(struct mesh_paths){1,one,edge,w};
  struct mesh_routed_program p;assert(!mesh_routed_compile(&fabric,routes,first,steps,initialized,8,&p));
  /* B is allowed to arrive before A, but A must be consumed before B: C's
     publication depends on A and B's combine depends on C's publication. */
  assert(p.rank[0].transfer[2].step.round<p.rank[0].transfer[0].step.round);
  assert(p.rank[0].consume[0]<p.rank[0].consume[2]);
  int64_t **data[N];
  for(uint32_t r=0;r<N;r++){
    data[r]=calloc(p.rank[r].buffers,sizeof *data[r]);
    for(uint32_t b=0;b<p.rank[r].buffers;b++){
      data[r][b]=malloc(p.rank[r].extent[b]+8);
      for(uint64_t x=0;x<p.rank[r].extent[b]/8;x++)data[r][b][x]=100*r+10*b+(int64_t)x;
    }
  }
  waited_for_dma=0;simulate_cables(&p,data,1);assert(waited_for_dma);
  assert(data[0][0][0]==100 && data[0][0][1]==112 && data[0][0][2]==114);
  assert(data[1][2][0]==100 && data[1][2][1]==1 && data[1][2][2]==2);
  for(uint32_t r=0;r<N;r++){for(uint32_t b=0;b<p.rank[r].buffers;b++)free(data[r][b]);free(data[r]);}
  mesh_routed_free(&p);
}

/* The Studios' ring (ranks 0 A, 1 B, 2 D, 3 C; A-B 3 cables, B-D 2, D-C 4, C-A 1): mesh_trees_shortest's trees reach
   every rank in two hops at most, the opposite pair through the wider side, and an all-gather along them lowers to
   SEND/COPY fragments with no relay storage where each adjacent pair's write routes are its own cables (what a fused
   crossing needs: nccl-mesh.c fusable); the default routes, every simple path, put relay storage at the ranks between. */
static void shortest_ring(void){
  enum { R=4, C=10 };
  static const struct mesh_cable ring[C]={
    {11,8000000000ull,{0,1}},{12,8000000000ull,{0,1}},{13,8000000000ull,{0,1}},{14,8000000000ull,{1,2}},{15,8000000000ull,{1,2}},
    {16,8000000000ull,{2,3}},{17,8000000000ull,{2,3}},{18,8000000000ull,{2,3}},{19,8000000000ull,{2,3}},{20,8000000000ull,{3,0}}};
  const struct mesh_fabric f={R,C,ring};
  uint32_t link[C][2];uint64_t bandwidth[C];
  for(uint32_t e=0;e<C;e++){memcpy(link[e],ring[e].node,sizeof link[e]);bandwidth[e]=ring[e].bandwidth;}
  const struct mesh_link_map map={.kind=MESH_LINKS_GRAPH,.nodes=R,.links=C,.link=link,.bandwidth=bandwidth};
  uint32_t root[R],parent[R*R];double weight[R];
  assert(mesh_trees_shortest(&map,root,parent,weight)==R);
  const struct mesh_trees trees={R,R,root,parent,weight};
  assert(!mesh_trees_check(&map,&trees));
  /* A: B, C its children, D under B (B-D 2 cables against C-A's 1); B: D under it, C under D; D: A under B; C: B under D */
  static const uint32_t want[R*R]={0,0,1,0, 1,1,1,2, 1,2,2,2, 3,2,3,3};
  for(uint32_t k=0;k<R*R;k++)assert(parent[k]==want[k]);
  uint32_t parts,moves;mesh_compile_room(&trees,&parts,&moves);
  uint32_t *segment=calloc(parts,sizeof *segment),*first=calloc(R+1,sizeof *first);
  double *shares=calloc(parts,sizeof *shares);
  struct mesh_move *move=calloc(moves,sizeof *move);struct mesh_program program;
  assert(!mesh_compile(&trees,MESH_ALLGATHER,0,0,NULL,&program,segment,shares,first,move));
  struct mesh_step *logical=calloc(moves,sizeof *logical);uint32_t offsets[R+1]={0},initialized[R];
  for(uint32_t r=0;r<R;r++){
    initialized[r]=1+program.scratch;
    offsets[r+1]=offsets[r]+mesh_program_steps_cut(&program,r,(struct mesh_operand){7,2,4*3072},NULL,16384,logical+offsets[r]);
  }
  for(uint32_t k=0;k<offsets[R];k++)assert(logical[k].op==MESH_STEP_SEND || logical[k].op==MESH_STEP_COPY);
  /* the adjacent pairs' routes their own cables, each a path of one cable */
  struct mesh_paths direct[R*R]={0};uint32_t one[R*R][C+1],edges[R*R][C];double zero[C]={0};
  for(uint32_t a=0;a<R;a++)for(uint32_t b=0;b<R;b++){
    uint32_t k=0;one[a*R+b][0]=0;
    for(uint32_t e=0;e<C;e++)if((ring[e].node[0]==a && ring[e].node[1]==b) || (ring[e].node[0]==b && ring[e].node[1]==a)){
      edges[a*R+b][k]=e;k++;one[a*R+b][k]=k;
    }
    if(k){direct[a*R+b]=(struct mesh_paths){k,one[a*R+b],edges[a*R+b],zero};assert(!mesh_paths_check(&f,a,b,direct+a*R+b));}
  }
  struct mesh_routed_program physical;
  assert(!mesh_routed_compile(&f,direct,offsets,logical,initialized,16384,&physical));
  for(uint32_t r=0;r<R;r++)assert(physical.rank[r].buffers==initialized[r]);
  mesh_routed_free(&physical);
  struct mesh_paths *every=NULL;assert(!mesh_write_routes_default(&f,&every));
  assert(!mesh_routed_compile(&f,every,offsets,logical,initialized,16384,&physical));
  uint32_t relays=0;for(uint32_t r=0;r<R;r++)relays+=physical.rank[r].buffers>initialized[r];
  assert(relays);
  mesh_routed_free(&physical);mesh_write_routes_free(every,R);
  free(segment);free(first);free(shares);free(move);free(logical);
}

int main(void){
  for(uint32_t s=0;s<N;s++)for(uint32_t d=0;d<N;d++)if(s!=d){
    paths[s*N+d]=(struct mesh_paths){0,path_first[s*N+d],path_edges[s*N+d],weights[s*N+d]};
    enumerate(s,d,s,1u<<s,0);assert(!mesh_paths_check(&fabric,s,d,paths+s*N+d));
  }
  for(uint32_t what=MESH_ALLREDUCE;what<=MESH_ALLGATHER;what++)for(uint32_t root=0;root<N;root++)check_collective(what,root);
  endpoint_offsets();
  counted();
  relayed_predecessor();
  publication_before_combine();
  shortest_ring();
  puts("routed collectives: the shortest ring trees' all-gather with no relay storage over direct routes, 20 arithmetic/gather cases, counted exchange with a relay-only rank, ragged zero shares, parallel cables, alternate relays, compact leases, exact demands and mismatched declarations passed");
  return 0;
}
