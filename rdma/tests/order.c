/* design/node-addressed-transfers.md: the shared order's unit tests (mesh-order.h).  Two ends' regions are made in
   memory; the same declared transfers (equal span lengths, unequal offsets and scatter) are appended at both; each
   end's walkers are run in a different interleaving, with gates opened in a different order and an epoch added mid
   order (a cable joining, from the boundary of the two ends' frontiers), and every queue pair's sequence of piece
   lengths and transfers must be equal at the two ends.  Then publication: transfers sharing a done word complete out of
   order and the word rises only as a prefix; one of another generation is lost; a restart loses what is not done */
#include "../mesh-order.h"
#include <assert.h>
#include <stdio.h>

#define OWNER ((UINT64_C(1)<<32)|123)
enum { PG=16384, PAGES=2048, CABLES=3, LANES=2, MOST=4, TRANSFERS=300, LOG=1<<16 };
struct end { struct hdr *m; struct mesh_node *x; uint64_t cursor; };
struct record { uint64_t bytes,slot; };
struct fifo { struct record r[LOG]; uint32_t n; };
static struct fifo seen[2][2][MOST][8];   /* end, way, cable, lane */
static struct order_walker walkers[2][2][MOST][8];

static void end_make(struct end *e){
  struct hdr geometry={0};
  const uint64_t length=mesh_layout(&geometry,PG,1,PAGES);
  e->m=calloc(1,length);assert(e->m);
  *e->m=geometry;e->m->version=MESH_LAYOUT;
  e->x=mesh_node_at(e->m,0);
  e->x->ring=e->m->data_off;           /* the ring in the arena's first pages */
  order_rings_new(e->m,e->x);
  atomic_store(&e->x->generation,1);atomic_store(&e->x->node,7);
  e->cursor=e->m->data_off+2*(uint64_t)MESH_RING*sizeof(struct mesh_slot);
  atomic_store(&mesh_client_at(e->m,OWNER)->owner,OWNER);  /* the appending client, alive: its gates hold */
}
static uint64_t take(struct end *e,uint64_t bytes){ const uint64_t at=e->cursor; e->cursor+=(bytes+63)/64*64; assert(e->cursor<e->m->length); return at; }
/* transfer i's spans: its lengths are the program's (equal at both ends), its offsets each end's own */
static uint64_t length_of(uint32_t i,uint32_t s){ return 1+((uint64_t)i*7919+s*104729)%(3*PG+11); }
static uint32_t spans_of(uint32_t i){ return 1+i%3; }
static _Atomic uint64_t *word(struct end *e,uint64_t at){ return (_Atomic uint64_t *)((char *)e->m+at); }
static uint64_t gates[2][TRANSFERS];
static void append(struct end *e,int end,int way,uint32_t i){
  const uint32_t n=spans_of(i);
  const uint64_t table=take(e,n*sizeof(struct mesh_dma_span));
  struct mesh_dma_span *spans=(struct mesh_dma_span *)((char *)e->m+table);
  for(uint32_t s=0;s<n;s++){spans[s].bytes=length_of(i,s);spans[s].offset=take(e,spans[s].bytes+(end?17:3));}
  gates[end][i]=take(e,8);
  const struct mesh_crossing t={.spans=table,.units=n,.gate=i%5?gates[end][i]:0,.base=(uint64_t)way,.generation=1};
  const uint64_t at=atomic_load(&e->x->tail[way]);
  struct mesh_slot *slot=mesh_ring(e->m,e->x,way,at);
  assert(atomic_load(&slot->sequence)==2*at);
  slot->t=t;atomic_store(&slot->owner,OWNER);atomic_store(&e->x->tail[way],at+1);atomic_store(&slot->sequence,2*at+1);
}
struct context { int end,way; uint32_t cable; };
static int post(void *context,int way,uint32_t lane,struct mesh_dma_span span,int dud,uint64_t slot){
  const struct context *c=context;
  assert(!dud && way==c->way);
  struct fifo *f=&seen[c->end][way][c->cable][lane];
  assert(f->n<LOG);
  f->r[f->n++]=(struct record){span.bytes,slot};
  return 1;
}
static void walk(struct end *e,int end,uint32_t cables,uint32_t lanes,uint32_t rotate){
  for(uint32_t q=0;q<cables*lanes*2;q++){
    const uint32_t at=(q+rotate)%(cables*lanes*2),way=at%2,c=at/2%cables,j=at/2/cables;
    struct context context={end,(int)way,c};
    order_walk(e->m,e->x,(int)way,c,j,&walkers[end][way][c][j],post,&context);
  }
}
static uint64_t frontier(int end,int way){
  uint64_t f=0;
  for(uint32_t c=0;c<MOST;c++)for(uint32_t j=0;j<8;j++)if(walkers[end][way][c][j].posted>f)f=walkers[end][way][c][j].posted;
  return f;
}

static void placement(void){
  struct mesh_epoch e={.first={10,20},.cables=CABLES};
  for(uint32_t c=0;c<CABLES;c++){e.cable[c].port=c;e.cable[c].lanes=LANES;}
  for(uint64_t n=10;n<40;n++){
    const struct order_place p=order_place(&e,0,n,5);
    assert(p.cable==(n-10)%CABLES && p.lane==1);
  }
  /* consecutive transfers take a cable's queue pairs in turn */
  assert(order_place(&e,1,20,4).lane!=order_place(&e,1,20,5).lane);
  struct mesh_node x={0};
  x.epoch[0]=(struct mesh_epoch){.first={0,0},.cables=1};x.epoch[1]=(struct mesh_epoch){.first={100,50},.cables=2};
  assert(order_epoch(&x,2,0,99)==x.epoch && order_epoch(&x,2,0,100)==x.epoch+1 && order_epoch(&x,2,1,50)==x.epoch+1);
  assert(!order_epoch(&x,0,0,0));
  assert(order_boundary(5,9)==9 && order_boundary(9,5)==9);
  printf("{\"test\":\"placement\",\"passed\":true}\n");
}

/* both ends' queue pairs carry the same sequence, though their walks interleave apart and their gates open apart,
   with a second epoch (a cable more) added at the boundary of the two ends' frontiers */
static void agreement(void){
  struct end ends[2];
  for(int k=0;k<2;k++)end_make(ends+k);
  uint32_t port[MOST],lanes[MOST];uint64_t wire[MOST];
  for(uint32_t c=0;c<MOST;c++){port[c]=c;lanes[c]=LANES;wire[c]=c+1;}
  const uint64_t zero[2]={0,0};
  for(int k=0;k<2;k++)assert(!order_extend(ends[k].x,zero,CABLES-1,port,lanes,wire));
  /* end 0 sends what end 1 receives: way 0 at end 0, way 1 at end 1 */
  for(uint32_t i=0;i<TRANSFERS/2;i++){append(ends,0,0,i);append(ends+1,1,1,i);}
  for(uint32_t i=0;i<TRANSFERS/2;i++){
    if(i%2)atomic_store(word(ends,gates[0][i]),UINT64_MAX-1);
    if(i%3)atomic_store(word(ends+1,gates[1][i]),UINT64_MAX-1);
    walk(ends,0,CABLES-1,LANES,i);walk(ends+1,1,CABLES-1,LANES,3*i+1);
  }
  /* a cable joins: both freeze, tell their frontiers, append the same epoch from the later of them */
  for(int k=0;k<2;k++)atomic_fetch_add(&ends[k].x->freeze,1);
  walk(ends,0,CABLES,LANES,0);walk(ends+1,1,CABLES,LANES,0);
  const uint64_t boundary=order_boundary(frontier(0,0),frontier(1,1));
  const uint64_t first[2]={boundary,boundary};
  for(int k=0;k<2;k++){assert(!order_extend(ends[k].x,first,CABLES,port,lanes,wire));atomic_fetch_add(&ends[k].x->freeze,1);}
  for(uint32_t i=TRANSFERS/2;i<TRANSFERS;i++){append(ends,0,0,i);append(ends+1,1,1,i);}
  for(uint32_t i=0;i<TRANSFERS;i++){
    atomic_store(word(ends,gates[0][i]),UINT64_MAX-1);
    if(i%4)atomic_store(word(ends+1,gates[1][i]),UINT64_MAX-1);
    walk(ends,0,CABLES,LANES,i*5);walk(ends+1,1,CABLES,LANES,i);
  }
  for(uint32_t i=0;i<TRANSFERS;i++)atomic_store(word(ends+1,gates[1][i]),UINT64_MAX-1);
  for(int r=0;r<4;r++){walk(ends,0,CABLES,LANES,r);walk(ends+1,1,CABLES,LANES,r);}
  uint64_t pieces=0,expected=0;
  for(uint32_t i=0;i<TRANSFERS;i++)for(uint32_t s=0;s<spans_of(i);s++)expected+=(length_of(i,s)+PG-1)/PG;
  for(uint32_t c=0;c<CABLES;c++)for(uint32_t j=0;j<LANES;j++){
    const struct fifo *a=&seen[0][0][c][j],*b=&seen[1][1][c][j];
    assert(a->n==b->n);
    for(uint32_t u=0;u<a->n;u++)assert(a->r[u].bytes==b->r[u].bytes && a->r[u].slot==b->r[u].slot && a->r[u].bytes<=PG);
    pieces+=a->n;
    assert(a->n);
  }
  assert(pieces==expected);
  printf("{\"test\":\"agreement\",\"transfers\":%u,\"pieces\":%llu,\"boundary\":%llu,\"passed\":true}\n",TRANSFERS,
         (unsigned long long)pieces,(unsigned long long)boundary);
  for(int k=0;k<2;k++)free(ends[k].m);
}

/* done words rise as prefixes; another generation's transfer is lost; a restart loses what is not done; a retired
   slot is free for its next lap */
static void publication(void){
  struct end e;end_make(&e);
  uint32_t port[1]={0},lanes[1]={1};uint64_t wire[1]={1};const uint64_t zero[2]={0,0};
  assert(!order_extend(e.x,zero,1,port,lanes,wire));
  const uint64_t shared=take(&e,8),own=take(&e,8),stale=take(&e,8),late=take(&e,8);
  const uint64_t table=take(&e,sizeof(struct mesh_dma_span));
  ((struct mesh_dma_span *)((char *)e.m+table))->offset=take(&e,100);((struct mesh_dma_span *)((char *)e.m+table))->bytes=100;
  const struct mesh_crossing t[]={{.spans=table,.units=1,.done=shared,.value=1,.generation=1},
                                  {.spans=table,.units=1,.done=shared,.value=2,.generation=1},
                                  {.spans=table,.units=1,.done=own,.value=9,.generation=1},
                                  {.spans=table,.units=1,.done=stale,.value=4,.generation=0},
                                  {.spans=table,.units=1,.done=late,.value=5,.generation=1}};
  for(uint64_t i=0;i<5;i++){
    struct mesh_slot *s=mesh_ring(e.m,e.x,0,i);
    s->t=t[i];atomic_store(&s->pieces,2);atomic_store(&s->sequence,2*i+1);
  }
  atomic_store(&e.x->tail[0],5);
  order_completed(e.m,e.x,0,1);order_completed(e.m,e.x,0,2);
  assert(atomic_load(word(&e,shared))==0 && atomic_load(word(&e,own))==9 && atomic_load(word(&e,stale))==UINT64_MAX);
  order_completed(e.m,e.x,0,0);
  assert(atomic_load(word(&e,shared))==2);
  order_retire(e.m,e.x,0);
  assert(atomic_load(&e.x->retired[0])==4 && atomic_load(&e.x->anchor[0][0])==3 && atomic_load(&e.x->anchor[0][1])==3);
  assert(atomic_load(&mesh_ring(e.m,e.x,0,0)->sequence)==2*MESH_RING);
  order_restart(e.m,e.x);
  assert(atomic_load(word(&e,late))==UINT64_MAX && !atomic_load(&e.x->live) && atomic_load(&e.x->generation)==2 && !atomic_load(&e.x->epochs));
  printf("{\"test\":\"publication\",\"passed\":true}\n");
  free(e.m);
}
int main(void){
  placement();
  agreement();
  publication();
  return 0;
}
