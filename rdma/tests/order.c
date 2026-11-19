/* design/declared-programs.md "Testing": the placement's unit tests (mesh-order.h, mesh-call.c mesh_declare), two
   ends' regions in memory over the simulated wire of tests/wire.h, which runs the bridge's own steps and ends the
   process at any SEND that meets a receive of another piece, transfer, execution, program or length.

   - Mixed shapes in one period: a program of period 4 (steps of one, two and three spans, unequal at each step, spans
     up to one piece), equal at the two ends but for its offsets and its directions swapped, run ahead by its client
     (every execution issued at once) with SEND gates opened in a different order at each end.
   - Several programs sharing a node: a second client's program (key below the first's, two executions a round) joins
     at a round boundary both ends reach; their rounds interleave on every queue pair.
   - A cable joining mid-run: an epoch from the later of the two ends' bounds (tests/wire.h wire_epoch).
   - Run-ahead past 10,000 transfers with one queue pair served once in 64 turns, the rest under jitter.
   - A client's redeclaration at its execution boundary (its old program closed where it ended).
   - Done words: a per-transfer count reaching the executions and a word shared by a whole direction rising only as a
     prefix; a restart loses every program (every done word ~0).
   - Refusals: a span longer than a piece, a declaration mid-execution, a closing program of a client gone with pieces
     outstanding (a restart asked). */
#include "wire.h"
#include "../mesh-call.h"
#include <assert.h>

enum { PG=16384, PAGES=4096, CABLES=3, LANES=2, PERIOD=4 };
#define CLIENT_A ((UINT64_C(1)<<32)|101)
#define CLIENT_B ((UINT64_C(2)<<32)|102)
#define CLIENT_C ((UINT64_C(3)<<32)|103)
static struct wire w;
static uint64_t cursor[2];
static uint64_t take(int n,uint64_t bytes){
  struct hdr *m=w.node[n].m;
  if(!cursor[n])cursor[n]=m->data_off;
  const uint64_t at=cursor[n];cursor[n]+=(bytes+63)/64*64;
  assert(cursor[n]<m->length);
  return at;
}
static _Atomic uint64_t *word(int n,uint64_t at){ return (_Atomic uint64_t *)((char *)w.node[n].m+at); }
static void region_make(int n,uint32_t node,uint32_t peer){
  struct hdr geometry={0};
  const uint64_t length=mesh_layout(&geometry,PG,1,PAGES);
  struct hdr *m=calloc(1,length);assert(m);
  *m=geometry;m->version=MESH_LAYOUT;m->node=node;m->magic=MESH_MAGIC;
  w.node[n].m=m;w.node[n].x=mesh_node_at(m,0);
  order_restart(m,w.node[n].x);
  atomic_store(&w.node[n].x->node,peer);
  const uint64_t clients[]={CLIENT_A,CLIENT_B,CLIENT_C};
  for(int k=0;k<3;k++)atomic_store(&mesh_client_at(m,clients[k])->owner,clients[k]);
}
static struct mesh_ctx client(int n,uint64_t owner){ return (struct mesh_ctx){.M=w.node[n].m,.client=owner,.fd=-1}; }
static struct mesh_order order_of(int n){
  struct mesh_ctx c=client(n,CLIENT_A);struct mesh_order o;
  assert(!mesh_order_open(&c,w.node[1-n].m->node,&o));
  return o;
}

/* A program's shape, the same at both ends: each logical direction's transfers (0: node 0 to node 1), each a step and
   its span lengths.  Node 0's direction 0 is logical 0; node 1's direction 0 is logical 1 */
struct shape { uint32_t transfers[2],step[2][64],spans[2][64]; uint64_t bytes[2][64][3]; };
struct made { struct mesh_declaration program; uint64_t done[2][64],shared[2],gate[2][64]; uint32_t gated[2][64]; int sharing; };
static void shape_make(struct shape *s,uint32_t seed,uint32_t transfers,uint32_t period){
  memset(s,0,sizeof *s);
  for(int l=0;l<2;l++){
    s->transfers[l]=transfers+(uint32_t)l;
    for(uint32_t i=0;i<s->transfers[l];i++){
      s->step[l][i]=i*period/s->transfers[l];
      s->spans[l][i]=1+(i+seed+(uint32_t)l)%3;
      for(uint32_t k=0;k<s->spans[l][i];k++)s->bytes[l][i][k]=1+((uint64_t)(i+1)*7919+k*104729+seed*31+(uint64_t)l*17)%PG;
    }
  }
}
/* node n's program of shape s: its tables, spans (offsets its own, `pad` apart), a done count a transfer (one an
   execution) and a word each direction shared by all its transfers (rising by one a transfer), SEND gates on two of
   every three transfers */
static void program_make(int n,const struct shape *s,uint64_t key,uint32_t period,uint32_t count,struct made *out){
  memset(out,0,sizeof *out);
  out->program=(struct mesh_declaration){.key=key,.period=period,.count=count};
  for(int d=0;d<2;d++){
    const int l=n?1-d:d;
    const uint32_t t=s->transfers[l];
    const uint64_t table=take(n,t*sizeof(struct mesh_crossing));
    struct mesh_crossing *x=(struct mesh_crossing *)((char *)w.node[n].m+table);
    out->shared[d]=take(n,8);
    uint64_t first=0;
    for(uint32_t i=0;i<t;i++){
      const uint64_t spans=take(n,s->spans[l][i]*sizeof(struct mesh_dma_span));
      struct mesh_dma_span *span=(struct mesh_dma_span *)((char *)w.node[n].m+spans);
      for(uint32_t k=0;k<s->spans[l][i];k++)span[k]=(struct mesh_dma_span){take(n,s->bytes[l][i][k]+(n?29:5)),s->bytes[l][i][k]};
      out->done[d][i]=take(n,8);
      out->gated[d][i]=!d && i%3;
      if(out->gated[d][i])out->gate[d][i]=take(n,8);
      x[i]=(struct mesh_crossing){.spans=spans,.units=s->spans[l][i],.first=first,.step=s->step[l][i],
        .gate=out->gate[d][i],.gate_base=0,.gate_step=s->spans[l][i],.done=out->done[d][i],.done_value=1,.done_step=1};
      first+=s->spans[l][i];
    }
    out->program.table[d]=table;out->program.transfers[d]=t;out->program.pieces[d]=first;
  }
}
/* a direction's shared word: one more done word, a transfer of its own (its last), rising by one a transfer */
static void shared_word(int n,struct made *p){
  p->sharing=1;
  for(int d=0;d<2;d++){
    const uint32_t t=(uint32_t)p->program.transfers[d];
    struct mesh_crossing *x=(struct mesh_crossing *)((char *)w.node[n].m+p->program.table[d]);
    /* the shared count rides on every transfer's done word in place of its own on even transfers */
    for(uint32_t i=0;i<t;i+=2){x[i].done=p->shared[d];x[i].done_value=i/2+1;x[i].done_step=(t+1)/2;}
  }
}
/* the SEND gates of node n's program opened through execution e (each a count of its spans an execution), in an
   order its own (`rotate`) */
static void gates_open(int n,const struct made *p,uint64_t e,uint32_t rotate){
  const uint32_t t=(uint32_t)p->program.transfers[0];
  for(uint32_t k=0;k<t;k++){
    const uint32_t i=(k+rotate)%t;
    if(!p->gated[0][i])continue;
    const struct mesh_crossing *x=(const struct mesh_crossing *)((char *)w.node[n].m+p->program.table[0])+i;
    atomic_store(word(n,p->gate[0][i]),(e+1)*x->units);
  }
}
/* turns of the wire until every queue is empty and nothing more is posted (at most `turns`) */
static void run(uint64_t turns,uint32_t lag){
  for(uint64_t k=0;k<turns;k++){
    wire_turn(&w,k,lag,64);
    if(k%64)continue;
    int empty=1;
    for(int d=0;d<2;d++)for(uint32_t c=0;c<CABLES;c++)for(uint32_t j=0;j<LANES;j++){
      const struct wire_fifo *f=&w.fifo[d][c][j];
      empty&=f->sent_head==f->sent_tail && f->received_head==f->received_tail;
    }
    if(empty){wire_turn(&w,k+1,lag,64);wire_turn(&w,k+2,lag,64);break;}
  }
}
/* every done count of node n's program at `executions`, and its shared words at the last transfer's value */
static void check_done(int n,const struct made *p,uint64_t executions){
  for(int d=0;d<2;d++){
    const uint32_t t=(uint32_t)p->program.transfers[d];
    for(uint32_t i=1;i<t;i+=2)
      if(atomic_load(word(n,p->done[d][i]))!=executions){
        fprintf(stderr,"order: node %d direction %d transfer %u done %llu, not %llu\n",n,d,i,(unsigned long long)atomic_load(word(n,p->done[d][i])),
                (unsigned long long)executions);
        exit(1);
      }
    if(p->sharing)assert(atomic_load(word(n,p->shared[d]))==(executions-1)*((t+1)/2)+(t-1)/2+1);
  }
}
int main(void){
  wire_make(&w,CABLES,LANES,256);
  region_make(0,11,12);region_make(1,12,11);
  /* the first epoch over two cables, from piece 0 */
  wire_epoch(&w,2,0);
  assert(w.node[0].x->epoch[0].first[0]==0 && order_qps(w.node[0].x->epoch)==4);
  struct mesh_order o[2]={order_of(0),order_of(1)};

  /* refusals: a span longer than a piece */
  {
    struct shape s;shape_make(&s,1,2,1);s.bytes[0][0][0]=PG+1;
    struct made p;program_make(0,&s,9,1,1,&p);
    struct mesh_ctx c=client(0,CLIENT_A);struct mesh_run r;
    assert(mesh_declare(&c,o,&p.program,&r)==EINVAL);
  }

  /* program A: period 4, mixed shapes, run ahead; its shared prefix words */
  struct shape a;shape_make(&a,3,9,PERIOD);
  struct made pa[2];struct mesh_run ra[2];
  for(int n=0;n<2;n++){
    program_make(n,&a,20,PERIOD,1,pa+n);shared_word(n,pa+n);
    struct mesh_ctx c=client(n,CLIENT_A);
    assert(!mesh_declare(&c,o+n,&pa[n].program,ra+n));
  }
  enum { E1=40 };
  for(int n=0;n<2;n++)mesh_issue(ra+n,E1*PERIOD);   /* every execution issued at once */
  uint64_t last_shared[2]={0,0};
  for(uint64_t e=0;e<E1;e++){
    for(int n=0;n<2;n++)gates_open(n,pa+n,e,(uint32_t)(n*5+e));
    for(int k=0;k<40;k++){
      wire_turn(&w,e*40+(uint64_t)k,UINT32_MAX,1);
      for(int n=0;n<2;n++)for(int d=0;d<2;d++){
        const uint64_t v=atomic_load(word(n,pa[n].shared[d]));
        assert(v!=UINT64_MAX);
        if(!d && v<last_shared[n]){fprintf(stderr,"order: a shared done word fell\n");return 1;}
        if(!d)last_shared[n]=v;
      }
    }
  }
  run(4000,UINT32_MAX);
  for(int n=0;n<2;n++)check_done(n,pa+n,E1);
  /* a declaration mid-execution is refused */
  for(int n=0;n<2;n++){
    mesh_issue(ra+n,E1*PERIOD+1);
    struct mesh_ctx c=client(n,CLIENT_A);struct mesh_run r;
    struct made p;struct shape s;shape_make(&s,5,2,1);program_make(n,&s,30,1,1,&p);
    assert(mesh_declare(&c,o+n,&p.program,&r)==EBUSY);
    mesh_issue(ra+n,(E1+1)*PERIOD);
    gates_open(n,pa+n,E1,0);
  }
  run(2000,UINT32_MAX);
  for(int n=0;n<2;n++)check_done(n,pa+n,E1+1);

  /* program B of another client (key 10: before A in a round), two executions a round, joins at A's round E1 + 1 */
  struct shape b;shape_make(&b,7,5,2);
  struct made pb[2];struct mesh_run rb[2];
  for(int n=0;n<2;n++){
    program_make(n,&b,10,2,2,pb+n);
    struct mesh_ctx c=client(n,CLIENT_B);
    assert(!mesh_declare(&c,o+n,&pb[n].program,rb+n));
    const struct mesh_segment *g=order_segment(w.node[n].x,atomic_load(&w.node[n].x->segments)-1);
    assert(g->programs==2 && w.node[n].x->program[g->slot[0]].program.key==10 && g->base[1]==E1+1);
  }
  /* run ahead past 10,000 transfers: both clients issue every round at once, one queue pair served once in 64 turns,
     a third cable joining after the first rounds */
  enum { ROUNDS=420 };
  const uint64_t transfers=(uint64_t)ROUNDS*(a.transfers[0]+a.transfers[1]+2*(b.transfers[0]+b.transfers[1]));
  assert(transfers>10000);
  for(int n=0;n<2;n++){mesh_issue(ra+n,(E1+1+ROUNDS)*PERIOD);mesh_issue(rb+n,2*ROUNDS*2);}
  for(uint64_t r=0;r<ROUNDS;r++){
    for(int n=0;n<2;n++){gates_open(n,pa+n,E1+1+r,(uint32_t)(r*3+(uint64_t)n));gates_open(n,pb+n,2*r,(uint32_t)r);gates_open(n,pb+n,2*r+1,(uint32_t)r+1);}
    for(int k=0;k<12;k++)wire_turn(&w,r*12+(uint64_t)k,1,64);
    if(r==ROUNDS/4){
      wire_epoch(&w,3,512);
      assert(atomic_load(&w.node[0].x->epochs)==2 && order_qps(w.node[0].x->epoch+1)==6);
      assert(w.node[0].x->epoch[1].first[0]==w.node[1].x->epoch[1].first[1] && w.node[0].x->epoch[1].first[1]==w.node[1].x->epoch[1].first[0]);
    }
  }
  run(4000000,1);
  for(int n=0;n<2;n++){check_done(n,pa+n,E1+1+ROUNDS);check_done(n,pb+n,2*ROUNDS);}
  for(int n=0;n<2;n++)for(int d=0;d<2;d++)assert(w.node[n].done[d][0][1]==w.node[n].posted[d][0][1]);
  printf("order: %llu transfers run ahead over two programs sharing the node, a cable joining, one queue pair lagging: "
         "%llu pieces, %llu bytes matched\n",(unsigned long long)transfers,(unsigned long long)w.matched,(unsigned long long)w.bytes);

  /* client A redeclares at its execution boundary: A closes where it ended, A2 joins B's rounds */
  struct shape a2;shape_make(&a2,11,4,1);
  struct made pa2[2];struct mesh_run ra2[2];
  for(int n=0;n<2;n++){
    program_make(n,&a2,30,1,1,pa2+n);
    struct mesh_ctx c=client(n,CLIENT_A);
    assert(!mesh_declare(&c,o+n,&pa2[n].program,ra2+n));
    assert(atomic_load(&w.node[n].x->program[ra[n].program].closed)==E1+1+ROUNDS);
    mesh_issue(ra2+n,50);mesh_issue(rb+n,2*(ROUNDS+50)*2);
  }
  for(uint64_t r=0;r<50;r++)for(int n=0;n<2;n++){gates_open(n,pa2+n,r,0);gates_open(n,pb+n,2*(ROUNDS+r),1);gates_open(n,pb+n,2*(ROUNDS+r)+1,2);}
  run(20000,UINT32_MAX);
  for(int n=0;n<2;n++){check_done(n,pa2+n,50);check_done(n,pb+n,2*(ROUNDS+50));}
  for(int n=0;n<2;n++){struct mesh_ctx c=client(n,CLIENT_A);assert(mesh_run_done(&c,ra+n));assert(!mesh_run_done(&c,ra2+n));}

  /* client B gone with its rounds outstanding at node 0 only (node 1's B issued one round fewer): the next declaration
     toward the node asks a restart, which loses every program */
  mesh_issue(rb+0,2*(ROUNDS+51)*2);
  atomic_store(&mesh_client_at(w.node[0].m,CLIENT_B)->owner,0);
  {
    struct mesh_ctx c=client(0,CLIENT_C);struct mesh_run r;
    struct made p;struct shape s;shape_make(&s,13,2,1);program_make(0,&s,40,1,1,&p);
    assert(mesh_declare(&c,o,&p.program,&r)==ENOLINK && atomic_load(&w.node[0].x->restart));
  }
  for(int n=0;n<2;n++)order_restart(w.node[n].m,w.node[n].x);
  for(int n=0;n<2;n++){
    assert(atomic_load(word(n,pa2[n].done[0][1]))==UINT64_MAX && atomic_load(word(n,pb[n].done[1][1]))==UINT64_MAX);
    struct mesh_ctx c=client(n,CLIENT_A);
    assert(!mesh_order_live(&c,o+n));
  }
  printf("order: placement, composition, epochs, run-ahead, redeclaration, done words and restart passed\n");
  return 0;
}
