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
/* The agreement of epochs (mesh-order.h order_agree_turn) between two ends whose cables come up at times of their own,
   in orders of their own, the messages between them each delayed (in order, as a control socket's), and each end
   posting (its frontier moving) below its bound: every seed's two ends append the same epochs (the same cables, each
   end's first sends the other's first receives) and end on every cable */
struct agree_end { struct hdr *m; struct mesh_node *x; struct order_agree a; uint64_t frontier[2]; uint64_t up_at[3];
  struct order_message box[64]; uint32_t head,tail; uint64_t ready[64]; };
static uint64_t agree_frontier(const struct mesh_node *x,int way,void *context){ (void)x; return ((struct agree_end *)context)->frontier[way]; }
static void agree_test(void){
  static const uint64_t wires[3]={0x30,0x10,0x20};
  for(uint32_t seed=1;seed<=400;seed++){
    struct agree_end e[2];
    uint64_t random=seed*0x9e3779b97f4a7c15u;
    for(int n=0;n<2;n++){
      struct hdr geometry={0};
      const uint64_t length=mesh_layout(&geometry,PG,1,64);
      struct hdr *m=calloc(1,length);assert(m);
      *m=geometry;m->version=MESH_LAYOUT;m->node=21+(uint32_t)n;m->magic=MESH_MAGIC;
      e[n]=(struct agree_end){.m=m,.x=mesh_node_at(m,0)};
      order_restart(m,e[n].x);atomic_store(&e[n].x->node,22-(uint32_t)n);
      for(int c=0;c<3;c++){random^=random<<13;random^=random>>7;random^=random<<17;e[n].up_at[c]=random%300;}
    }
    for(uint64_t t=0;t<2000;t++)for(int n=0;n<2;n++){
      struct agree_end *v=e+n,*peer=e+1-n;
      uint64_t up[3];uint32_t port[3],lanes[3],k=0;
      for(int c=0;c<3;c++)if(t>=v->up_at[c] && t>=peer->up_at[c]/2){up[k]=wires[c];port[k]=(uint32_t)c;lanes[k]=2;k++;}
      /* sorted by identity, as order_cables sorts them */
      for(uint32_t i=1;i<k;i++)for(uint32_t j=i;j>0 && up[j]<up[j-1];j--){
        uint64_t w0=up[j];up[j]=up[j-1];up[j-1]=w0;uint32_t p0=port[j];port[j]=port[j-1];port[j-1]=p0;
      }
      while(v->head!=v->tail && v->ready[v->head%64]<=t){order_agree_heard(&v->a,v->x,&v->box[v->head%64]);v->head++;}
      struct order_message say;
      const int done=order_agree_turn(v->x,n==0,&v->a,k,up,port,lanes,agree_frontier,v,&say);
      assert(!(done&ORDER_LOST));
      if(done&ORDER_SAY){
        random^=random<<13;random^=random>>7;random^=random<<17;
        const uint64_t at=t+1+random%40,last=peer->tail>peer->head?peer->ready[(peer->tail-1)%64]:0;
        peer->box[peer->tail%64]=say;peer->ready[peer->tail%64]=at>last?at:last;peer->tail++;
      }
      /* posting goes on below the bound */
      for(int way=0;way<2;way++){
        const uint64_t limit=atomic_load(&v->x->limit[way]),epochs=atomic_load(&v->x->epochs);
        if(epochs && v->frontier[way]+1<limit && (t+(uint64_t)way)%3)v->frontier[way]++;
      }
    }
    const uint32_t epochs=atomic_load(&e[0].x->epochs);
    if(epochs!=atomic_load(&e[1].x->epochs) || !epochs || e[0].x->epoch[epochs-1].cables!=3){
      fprintf(stderr,"order: seed %u: epochs %u and %u, the last over %u cables\n",seed,epochs,atomic_load(&e[1].x->epochs),
              epochs?e[0].x->epoch[epochs-1].cables:0);
      exit(1);
    }
    for(uint32_t k=0;k<epochs;k++){
      const struct mesh_epoch *a=e[0].x->epoch+k,*b=e[1].x->epoch+k;
      assert(a->cables==b->cables && a->first[0]==b->first[1] && a->first[1]==b->first[0]);
      for(uint32_t c=0;c<a->cables;c++)assert(a->cable[c].wire==b->cable[c].wire);
      /* nothing either end posted lies past an epoch's beginning but in the epochs before */
      if(k)assert(a->first[0]>=e[0].x->epoch[k-1].first[0]);
    }
    for(int n=0;n<2;n++)free(e[n].m);
  }
  printf("order: epochs agreed over 400 orders of cables coming up and delayed messages: passed\n");
}
int main(void){
  agree_test();
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

  /* gates that are their transfers' issue (MESH_GATE_ISSUES), and done words raised word by word: a program of two
     words alternating transfer by transfer (two micro-batches' frames, frame-major), every transfer gated each way,
     nothing issued while it runs; the second word's gates held shut, the first's executions cross and its word rises
     through them while the second's stays put; then the second's open, the client counts its executions issued, and
     the program closes */
  {
    enum { FRAMES=12, RUNS=4 };
    struct made p[2];struct mesh_run r[2];uint64_t gate[2][2];
    for(int n=0;n<2;n++){
      p[n].program=(struct mesh_declaration){.key=5,.period=1,.count=1};
      for(int k=0;k<2;k++){gate[n][k]=take(n,8);atomic_store(word(n,gate[n][k]),0);}
      for(int d=0;d<2;d++){
        const uint64_t table=take(n,2*FRAMES*sizeof(struct mesh_crossing));
        struct mesh_crossing *x=(struct mesh_crossing *)((char *)w.node[n].m+table);
        const uint64_t words[2]={take(n,8),take(n,8)};
        p[n].done[d][0]=words[0];p[n].done[d][1]=words[1];
        for(uint32_t i=0;i<2*FRAMES;i++){
          const uint64_t spans=take(n,sizeof(struct mesh_dma_span));
          *(struct mesh_dma_span *)((char *)w.node[n].m+spans)=(struct mesh_dma_span){take(n,4096),4096};
          const uint32_t k=i%2,f=i/2;
          /* SEND span: gate word k past frame f (gate_base + 0 + 1 + e FRAMES, MESH_GATE_WHOLE); a receive: word k at
             frame f's value; a done word a micro-batch, its frame count */
          x[i]=(struct mesh_crossing){.spans=spans,.units=1,.first=i,.step=0,.flags=MESH_GATE_ISSUES|(d?0:MESH_GATE_WHOLE),
            .gate=gate[n][k],.gate_base=f+1,.gate_step=FRAMES,.done=words[k],.done_value=f+1,.done_step=FRAMES};
        }
        p[n].program.table[d]=table;p[n].program.transfers[d]=2*FRAMES;p[n].program.pieces[d]=2*FRAMES;
      }
      struct mesh_ctx c=client(n,CLIENT_C);
      assert(!mesh_declare(&c,o+n,&p[n].program,r+n));
    }
    for(int n=0;n<2;n++)atomic_store(word(n,gate[n][0]),RUNS*FRAMES);
    run(20000,UINT32_MAX);
    for(int n=0;n<2;n++)for(int d=0;d<2;d++){
      if(atomic_load(word(n,p[n].done[d][0]))!=RUNS*FRAMES || atomic_load(word(n,p[n].done[d][1]))!=0){
        fprintf(stderr,"order: node %d direction %d: words %llu and %llu, not %u and 0 (issued 0, the second word's gates shut)\n",n,d,
                (unsigned long long)atomic_load(word(n,p[n].done[d][0])),(unsigned long long)atomic_load(word(n,p[n].done[d][1])),RUNS*FRAMES);
        return 1;
      }
    }
    for(int n=0;n<2;n++)atomic_store(word(n,gate[n][1]),RUNS*FRAMES);
    run(20000,UINT32_MAX);
    for(int n=0;n<2;n++){
      for(int d=0;d<2;d++)assert(atomic_load(word(n,p[n].done[d][1]))==RUNS*FRAMES);
      mesh_issue(r+n,RUNS);
      struct mesh_ctx c=client(n,CLIENT_C);
      assert(!mesh_close(&c,r+n));
    }
    run(200,UINT32_MAX);
    for(int n=0;n<2;n++){struct mesh_ctx c=client(n,CLIENT_C);assert(mesh_run_done(&c,r+n));}
    printf("order: gates that are their issue, words raised word by word: passed\n");
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
