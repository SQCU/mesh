/* design/declared-programs.md: the placement of every piece of a node pair's declared programs on its cables' queue
   pairs, as pure functions of the region's declarations (struct mesh_node: its programs, segments and epochs).  The
   two ends of a cable evaluate the same functions on equal declarations, so each queue pair's SENDs and receives are
   the same sequence at both ends without either telling the other anything.  The bridge's port children
   (mesh-link.c), the clients' declarations (mesh-call.c) and the two-node simulator (tests/sim-wire.c) are its callers */
#ifndef MESH_ORDER_H
#define MESH_ORDER_H
#include "mesh.h"
#include <errno.h>
#include <string.h>

/* A piece: at most one allocation block (the region's `block` pages of `pgsz`, equal at both ends of every cable:
   port_heard), and every declared span is one piece (mesh-call.c mesh_declare) */
static inline uint64_t order_piece_bytes(const struct hdr *m){ return (uint64_t)m->block*m->pgsz; }
#define ORDER_NONE UINT64_MAX

static inline _Atomic uint64_t *order_word(void *m,uint64_t at){ return (_Atomic uint64_t *)((char *)m+at); }
/* the word at region offset `at` raised to `value` where it is lower (release) */
static inline void order_raise(void *m,uint64_t at,uint64_t value){
  if(!at)return;
  _Atomic uint64_t *word=order_word(m,at);
  uint64_t old=atomic_load_explicit(word,memory_order_relaxed);
  while(old<value && !atomic_compare_exchange_weak_explicit(word,&old,value,memory_order_release,memory_order_relaxed)){}
}

/* ---- Epochs: which queue pair a piece is on ----------------------------------------------------------------------
   An epoch's queue pairs are taken lane by lane across its cables: lane 0 of every cable in the epoch's order, then lane 1
   of every cable that has two, and so on, so consecutive pieces cross every cable before any cable's second queue pair.
   Piece n of direction d, in the epoch holding it (the latest whose first[d] is at most n), is on queue pair
   (n - first[d]) mod Q of that epoch, Q its queue pairs */
static inline uint32_t order_qps(const struct mesh_epoch *e){
  uint32_t q=0;
  for(uint32_t c=0;c<e->cables && c<MESH_PORTS;c++)q+=e->cable[c].lanes;
  return q;
}
/* the index of cable c's lane l among the epoch's queue pairs; UINT32_MAX where cable c holds no lane l */
static inline uint32_t order_qp(const struct mesh_epoch *e,uint32_t c,uint32_t l){
  if(c>=e->cables || c>=MESH_PORTS || l>=e->cable[c].lanes)return UINT32_MAX;
  uint32_t q=0;
  for(uint32_t k=0;k<e->cables && k<MESH_PORTS;k++){
    const uint32_t n=e->cable[k].lanes;
    q+=n<l?n:l;
    if(k<c && n>l)q++;
  }
  return q;
}
/* the epoch's cable on the bridge's port `port`; UINT32_MAX where it has none */
static inline uint32_t order_cable(const struct mesh_epoch *e,uint32_t port){
  for(uint32_t c=0;c<e->cables && c<MESH_PORTS;c++)if(e->cable[c].port==port)return c;
  return UINT32_MAX;
}
/* The piece that is unit u of direction d on the queue pair `lane` of the cable on port `port` (u counted from the
   cable's connect: a cable is in every epoch of its generation from the first that names it, and its loss restarts the
   order), over the first `epochs` epochs; ORDER_NONE where no epoch names the cable */
static inline uint64_t order_unit_piece(const struct mesh_node *x,uint32_t epochs,int d,uint32_t port,uint32_t lane,uint64_t u){
  if(epochs>MESH_EPOCHS)epochs=MESH_EPOCHS;
  for(uint32_t e=0;e<epochs;e++){
    const struct mesh_epoch *at=x->epoch+e;
    const uint32_t c=order_cable(at,port),q=c==UINT32_MAX?UINT32_MAX:order_qp(at,c,lane);
    if(q==UINT32_MAX)continue;
    const uint64_t Q=order_qps(at),first=at->first[d];
    if(e+1==epochs)return first+q+u*Q;
    const uint64_t next=x->epoch[e+1].first[d];
    const uint64_t count=next>first+q?(next-first-q+Q-1)/Q:0;
    if(u<count)return first+q+u*Q;
    u-=count;
  }
  return ORDER_NONE;
}
/* the first piece of direction d not yet complete, every queue pair of the last epoch holding `done(q)` units done:
   the least over its queue pairs of the piece of each one's first unit not done (each queue pair's pieces rise, so every
   piece below that least one is done on its own queue pair).  `done` names the counts as the region holds them: a
   lane's records (mesh.h struct mesh_lanes) by port and lane */
static inline uint64_t order_done_through(const struct mesh_node *x,uint32_t epochs,int d,const uint64_t *done_count){
  if(!epochs)return 0;
  if(epochs>MESH_EPOCHS)epochs=MESH_EPOCHS;
  const struct mesh_epoch *last=x->epoch+epochs-1;
  uint64_t least=ORDER_NONE;
  for(uint32_t c=0,k=0;c<last->cables && c<MESH_PORTS;c++)
    for(uint32_t l=0;l<last->cable[c].lanes;l++,k++){
      const uint64_t n=order_unit_piece(x,epochs,d,last->cable[c].port,l,done_count[k]);
      if(n<least)least=n;
    }
  return least;
}

/* the piece past every piece of direction d posted on the last epoch's queue pairs, each holding `posted(q)` units
   posted (in the epoch's cable and lane order): the bound a new epoch begins at or after (mesh-link.c order_turn) */
static inline uint64_t order_posted_through(const struct mesh_node *x,uint32_t epochs,int d,const uint64_t *posted){
  if(!epochs)return 0;
  if(epochs>MESH_EPOCHS)epochs=MESH_EPOCHS;
  const struct mesh_epoch *last=x->epoch+epochs-1;
  uint64_t through=0;
  for(uint32_t c=0,k=0;c<last->cables && c<MESH_PORTS;c++)
    for(uint32_t l=0;l<last->cable[c].lanes;l++,k++){
      if(!posted[k])continue;
      const uint64_t n=order_unit_piece(x,epochs,d,last->cable[c].port,l,posted[k]-1);
      if(n!=ORDER_NONE && n+1>through)through=n+1;
    }
  return through;
}

/* ---- Segments: which program, execution and transfer a piece is ------------------------------------------------
   A node's pieces of direction d are its segments' in turn.  Segment s holds rounds from start[d]; a round is, for each
   of its programs in key order, `count` executions of that program's direction-d table, execution after execution; a
   program's executions in the segment begin at its `base`.  Where piece n lies */
struct order_at {
  uint64_t segment,execution,span;
  uint32_t slot,transfer;
  const struct mesh_crossing *t;
};
static inline const struct mesh_segment *order_segment(const struct mesh_node *x,uint64_t s){ return x->segment+s%MESH_SEGMENTS; }
/* the program table of direction d of program slot `slot`, NULL where the region does not hold it */
static inline const struct mesh_crossing *order_table(const struct hdr *m,const struct mesh_declared *p,int d){
  const uint64_t at=p->program.table[d],n=p->program.transfers[d];
  if(!at || !n || at%8 || at<m->data_off || at>m->length || n>(m->length-at)/sizeof(struct mesh_crossing))return NULL;
  return (const struct mesh_crossing *)((const char *)m+at);
}
/* 1 where piece n of direction d lies in a segment of the `segments` declared (each segment's start read from its
   slot; `hint`, the caller's segment to look from, moved to the one found), filling *at; 0 where it lies past the
   declarations as they stand (the last segment closed and the next not yet published) or in no table */
static inline int order_locate(const struct hdr *m,const struct mesh_node *x,uint64_t segments,int d,uint64_t n,uint64_t *hint,
                               struct order_at *at){
  if(!segments)return 0;
  const uint64_t oldest=segments>MESH_SEGMENTS?segments-MESH_SEGMENTS:0;
  uint64_t s=*hint<oldest?oldest:*hint>=segments?segments-1:*hint;
  while(s>oldest && n<order_segment(x,s)->start[d])s--;
  while(s+1<segments && order_segment(x,s+1)->start[d]<=n)s++;
  *hint=s;
  const struct mesh_segment *g=order_segment(x,s);
  if(n<g->start[d] || !g->round[d])return 0;
  const uint64_t off=n-g->start[d],round=off/g->round[d],rounds=atomic_load_explicit(&g->rounds,memory_order_acquire);
  if(round>=rounds)return 0;
  uint64_t within=off%g->round[d];
  for(uint32_t k=0;k<g->programs && k<MESH_SEGMENT_PROGRAMS;k++){
    const uint32_t slot=g->slot[k];
    if(slot>=MESH_PROGRAMS)return 0;
    const struct mesh_declared *p=x->program+slot;
    const uint64_t pieces=p->program.pieces[d],width=(uint64_t)p->program.count*pieces;
    if(within>=width){within-=width;continue;}
    const struct mesh_crossing *t=order_table(m,p,d);
    if(!t)return 0;
    const uint64_t piece=within%pieces;
    uint32_t lo=0,hi=(uint32_t)p->program.transfers[d];
    while(hi-lo>1){const uint32_t mid=lo+(hi-lo)/2;if(t[mid].first<=piece)lo=mid;else hi=mid;}
    if(piece<t[lo].first || piece-t[lo].first>=t[lo].units)return 0;
    *at=(struct order_at){.segment=s,.execution=g->base[k]+round*p->program.count+within/pieces,.span=piece-t[lo].first,
                          .slot=slot,.transfer=lo,.t=t+lo};
    return 1;
  }
  return 0;
}
/* whether the piece at `at` of direction d may be posted: its step issued (its program's counter past it) and its gate
   holding (a SEND of span s: the word at gate_base + s + 1 + e gate_step; a receive: gate_base + e gate_step) */
static inline int order_ready(const struct hdr *m,const struct mesh_node *x,const struct order_at *at,int d){
  const struct mesh_declared *p=x->program+at->slot;
  const struct mesh_crossing *t=at->t;
  if(atomic_load_explicit(&p->issued,memory_order_acquire)<=at->execution*p->program.period+t->step)return 0;
  if(!t->gate)return 1;
  if(t->gate%8 || t->gate+8>m->length)return 1;
  const uint64_t threshold=t->gate_base+at->execution*t->gate_step+(d || t->flags&MESH_GATE_WHOLE?0:at->span+1);
  return atomic_load_explicit((const _Atomic uint64_t *)((const char *)m+t->gate),memory_order_acquire)>=threshold;
}
/* every done word of the transfers of direction d wholly below piece `through`, from piece `from` (a transfer's first
   piece), raised in the pieces' order to its execution's value; the piece it reached (the first transfer's first piece
   not wholly below `through`, or past the declarations as they stand) */
static inline uint64_t order_raise_through(struct hdr *m,const struct mesh_node *x,uint64_t segments,int d,uint64_t from,uint64_t through,
                                           uint64_t *hint){
  uint64_t n=from;
  while(n<through){
    struct order_at at;
    if(!order_locate(m,x,segments,d,n,hint,&at))break;
    const uint64_t last=n-at.span+at.t->units-1;
    if(last>=through)break;
    if(at.t->done && at.t->done%8==0 && at.t->done+8<=m->length)order_raise(m,at.t->done,at.t->done_value+at.execution*at.t->done_step);
    n=last+1;
  }
  return n;
}

/* ---- The bridge's two steps (mesh-link.c order_pass; tests/sim-wire.c runs the same) -----------------------------
   A lane's next piece to post each way: the piece its unit u is (order_unit_piece), where it lies (order_locate) and
   whether it may go (order_ready, below the order's posting bound `limit`): 1 with *n and *at filled, else 0 (the lane
   waits at that piece: nothing is skipped) */
static inline int order_next(const struct hdr *m,const struct mesh_node *x,uint32_t epochs,uint64_t segments,int d,uint32_t port,
                             uint32_t lane,uint64_t u,uint64_t *hint,uint64_t *n,struct order_at *at){
  *n=order_unit_piece(x,epochs,d,port,lane,u);
  return *n!=ORDER_NONE && *n<atomic_load_explicit(&x->limit[d],memory_order_acquire) && order_locate(m,x,segments,d,*n,hint,at) &&
         order_ready(m,x,at,d);
}
/* queue pair q of epoch e as its cable and lane (order_qp's inverse): 1, else 0 where the epoch has no queue pair q */
static inline int order_qp_cable(const struct mesh_epoch *e,uint32_t q,uint32_t *cable,uint32_t *lane){
  for(uint32_t l=0;;l++){
    uint32_t round=0;
    for(uint32_t c=0;c<e->cables && c<MESH_PORTS;c++)
      if(e->cable[c].lanes>l){if(q==round){*cable=c;*lane=l;return 1;}round++;}
    if(!round)return 0;
    q-=round;
  }
}
/* whether piece n of direction d is done: the queue pair it is on (its epoch's: the latest whose first[d] is at most
   n) holding more units done (`done`, in the last epoch's cable and lane order) than its unit's index there */
static inline int order_piece_done(const struct mesh_node *x,uint32_t epochs,int d,uint64_t n,const uint64_t *done){
  if(!epochs)return 0;
  if(epochs>MESH_EPOCHS)epochs=MESH_EPOCHS;
  uint32_t e=epochs;
  while(e>1 && x->epoch[e-1].first[d]>n)e--;
  const struct mesh_epoch *at=x->epoch+e-1;
  const uint64_t Q=order_qps(at);
  uint32_t c,lane;
  if(n<at->first[d] || !Q || !order_qp_cable(at,(uint32_t)((n-at->first[d])%Q),&c,&lane))return 0;
  const uint32_t port=at->cable[c].port;
  /* its unit: those before it in this epoch, then every unit of it in the epochs before */
  uint64_t u=(n-at->first[d])/Q;
  for(uint32_t k=0;k+1<e;k++){
    const struct mesh_epoch *before=x->epoch+k;
    const uint32_t b=order_cable(before,port),q=b==UINT32_MAX?UINT32_MAX:order_qp(before,b,lane);
    if(q==UINT32_MAX)continue;
    const uint64_t B=order_qps(before),first=before->first[d],next=x->epoch[k+1].first[d];
    u+=next>first+q?(next-first-q+B-1)/B:0;
  }
  const struct mesh_epoch *last=x->epoch+epochs-1;
  for(uint32_t k=0,i=0;i<last->cables && i<MESH_PORTS;k+=last->cable[i++].lanes)
    if(last->cable[i].port==port)return lane<last->cable[i].lanes && done[k+lane]>u;
  return 0;
}
/* The done words of direction d raised, by every process that serves the order (each port's child, on every pass in
   which some queue pair's done count moved; tests/wire.h's one step), whichever sees a completion first: those of
   every transfer wholly below the pieces every queue pair of the last epoch has done (`done`, a count each in the
   epoch's cable and lane order: order_done_through), in the pieces' order, with the piece below which all are raised
   published (`raised`); then, of the next ORDER_AHEAD transfers, each one wholly done whose done word no earlier one of
   them not yet done names (a word several transfers count on rises only as their prefix; a transfer held by its gate
   holds back no other's word).  Raisers need no agreement: every done count only rises, so each one's reading of them
   is true of the queue pairs (a transfer it finds done is done, and one it finds before another not done was not done
   when read), every word is raised to the greatest value any raiser computes (order_raise) and `raised` is moved only
   up (order_raised_to) */
#define ORDER_AHEAD 64u
static inline void order_raised_to(_Atomic uint64_t *raised,uint64_t value){
  uint64_t old=atomic_load_explicit(raised,memory_order_relaxed);
  while(old<value && !atomic_compare_exchange_weak_explicit(raised,&old,value,memory_order_release,memory_order_relaxed)){}
}
static inline void order_raise_pass(struct hdr *m,struct mesh_node *x,uint32_t epochs,uint64_t segments,int d,const uint64_t *done,
                                    uint64_t *hint,uint64_t *seen){
  /* nothing to raise where no queue pair has done more since the last pass and no segment was declared */
  uint64_t sum=segments;
  const struct mesh_epoch *final=x->epoch+(epochs<MESH_EPOCHS?epochs:MESH_EPOCHS)-1;
  for(uint32_t c=0,k=0;c<final->cables && c<MESH_PORTS;c++)for(uint32_t l=0;l<final->cable[c].lanes;l++,k++)sum+=done[k]*(k+1);
  if(sum==*seen)return;
  *seen=sum;
  uint64_t raised=atomic_load_explicit(&x->raised[d],memory_order_acquire);
  const uint64_t through=order_done_through(x,epochs,d,done);
  if(through>raised){
    const uint64_t reached=order_raise_through(m,x,segments,d,raised,through,hint);
    if(reached>raised)order_raised_to(&x->raised[d],raised=reached);
  }
  /* past the latest piece any queue pair has done, no transfer is whole: the look ahead ends there */
  uint64_t latest=0;
  for(uint32_t c=0,k=0;c<final->cables && c<MESH_PORTS;c++)for(uint32_t l=0;l<final->cable[c].lanes;l++,k++)
    if(done[k]){const uint64_t p=order_unit_piece(x,epochs,d,final->cable[c].port,l,done[k]-1);if(p!=ORDER_NONE && p+1>latest)latest=p+1;}
  uint64_t held[ORDER_AHEAD],n=raised,look=*hint;
  uint32_t holding=0;
  for(uint32_t k=0;k<ORDER_AHEAD && n<latest;k++){
    struct order_at at;
    if(!order_locate(m,x,segments,d,n,&look,&at))break;
    const uint64_t last=n-at.span+at.t->units-1;
    if(last>=latest)break;
    /* each queue pair's pieces are done in their order, so a transfer is done where the last piece of it on each queue
       pair is: its last Q pieces in the last epoch, and every piece of it before that epoch */
    const uint64_t Q=order_qps(final),boundary=final->first[d];
    int whole=1;
    for(uint64_t p=last+1;whole && p>n && p>boundary && last+1-p<Q;p--)whole=order_piece_done(x,epochs,d,p-1,done);
    for(uint64_t p=n;whole && p<boundary && p<=last;p++)whole=order_piece_done(x,epochs,d,p,done);
    const uint64_t word=at.t->done;
    int behind=0;
    for(uint32_t h=0;h<holding && !behind;h++)behind=held[h]==word;
    if(word && whole && !behind && word%8==0 && word+8<=m->length)order_raise(m,word,at.t->done_value+at.execution*at.t->done_step);
    else if(word && !whole && !behind)held[holding++]=word;
    n=last+1;
  }
}

/* ---- Composition: a declaration's next segment (mesh-call.c mesh_declare, under the node's `declaring`) ----------
   The node's last segment closes at the round its programs have reached, each at a whole round (its issued steps a
   whole number of executions, `count` a round; every one at the same round: the clients agreed it), and the next
   begins there with the programs that go on, without those closing, and with the one declared, in key order.  0,
   EBUSY where they are not at one round boundary, ENOSPC where the segment's slot or the programs' room is taken */
struct order_next { uint64_t rounds,start[2]; uint32_t programs,slot[MESH_SEGMENT_PROGRAMS]; uint64_t base[MESH_SEGMENT_PROGRAMS]; };
static inline int order_round_of(const struct mesh_node *x,const struct mesh_segment *g,uint32_t k,uint64_t *rounds){
  const struct mesh_declared *p=x->program+g->slot[k];
  const uint64_t issued=atomic_load_explicit(&p->issued,memory_order_acquire),period=p->program.period,count=p->program.count;
  if(!period || !count || issued%period)return EBUSY;
  const uint64_t executions=issued/period;
  if(executions<g->base[k] || (executions-g->base[k])%count)return EBUSY;
  *rounds=(executions-g->base[k])/count;
  return 0;
}
/* the next segment of node x: `closing` (a bit a program slot) leave, `joining` (MESH_PROGRAMS none) joins */
static inline int order_compose(const struct mesh_node *x,uint32_t closing,uint32_t joining,struct order_next *next){
  const uint64_t segments=atomic_load_explicit(&x->segments,memory_order_acquire);
  *next=(struct order_next){0};
  uint32_t slots[MESH_SEGMENT_PROGRAMS+1],count=0;uint64_t base[MESH_SEGMENT_PROGRAMS+1];
  if(segments){
    const struct mesh_segment *g=order_segment(x,segments-1);
    uint64_t rounds=UINT64_MAX;
    for(uint32_t k=0;k<g->programs && k<MESH_SEGMENT_PROGRAMS;k++){
      uint64_t r;
      if(order_round_of(x,g,k,&r))return EBUSY;
      if(rounds!=UINT64_MAX && r!=rounds)return EBUSY;
      rounds=r;
    }
    if(rounds==UINT64_MAX)rounds=0;
    next->rounds=rounds;
    for(int d=0;d<2;d++)next->start[d]=g->start[d]+rounds*g->round[d];
    for(uint32_t k=0;k<g->programs && k<MESH_SEGMENT_PROGRAMS;k++){
      if(closing>>g->slot[k]&1)continue;
      slots[count]=g->slot[k];base[count++]=g->base[k]+rounds*x->program[g->slot[k]].program.count;
    }
  }
  if(joining<MESH_PROGRAMS){slots[count]=joining;base[count++]=0;}
  if(count>MESH_SEGMENT_PROGRAMS)return ENOSPC;
  /* key order, alike at both ends */
  for(uint32_t a=1;a<count;a++)
    for(uint32_t b=a;b>0 && x->program[slots[b]].program.key<x->program[slots[b-1]].program.key;b--){
      const uint32_t s=slots[b];slots[b]=slots[b-1];slots[b-1]=s;
      const uint64_t e=base[b];base[b]=base[b-1];base[b-1]=e;
    }
  next->programs=count;
  for(uint32_t k=0;k<count;k++){next->slot[k]=slots[k];next->base[k]=base[k];}
  return 0;
}
/* segment `s` (the node's next) written from `next`, the last segment closed at its rounds and the count published */
static inline void order_publish_segment(struct mesh_node *x,uint64_t s,const struct order_next *next){
  struct mesh_segment *g=x->segment+s%MESH_SEGMENTS;
  g->start[0]=next->start[0];g->start[1]=next->start[1];
  g->round[0]=g->round[1]=0;
  g->programs=next->programs;
  for(uint32_t k=0;k<MESH_SEGMENT_PROGRAMS;k++){
    g->slot[k]=k<next->programs?next->slot[k]:MESH_PROGRAMS;g->base[k]=k<next->programs?next->base[k]:0;
    if(k<next->programs)for(int d=0;d<2;d++)
      g->round[d]+=(uint64_t)x->program[next->slot[k]].program.count*x->program[next->slot[k]].program.pieces[d];
  }
  atomic_store_explicit(&g->rounds,UINT64_MAX,memory_order_release);
  if(s)atomic_store_explicit(&x->segment[(s-1)%MESH_SEGMENTS].rounds,next->rounds,memory_order_release);
  atomic_store_explicit(&x->segments,s+1,memory_order_release);
}
/* whether segment s of node x is wholly done each way (every done word of its pieces raised) */
static inline int order_segment_done(const struct mesh_node *x,uint64_t s){
  const uint64_t segments=atomic_load_explicit(&x->segments,memory_order_acquire);
  if(s+1>=segments)return 0;
  const struct mesh_segment *after=order_segment(x,s+1);
  return atomic_load_explicit(&x->raised[0],memory_order_acquire)>=after->start[0] &&
         atomic_load_explicit(&x->raised[1],memory_order_acquire)>=after->start[1];
}

/* ---- The order's life (the bridge's parent: mesh-link.c order_turn) --------------------------------------------- */
/* every done word of program slot p raised to UINT64_MAX (it is lost) */
static inline void order_program_lost(struct hdr *m,struct mesh_node *x,uint32_t p){
  const struct mesh_declared *declared=x->program+p;
  for(int d=0;d<2;d++){
    const struct mesh_crossing *t=order_table(m,declared,d);
    for(uint64_t i=0;t && i<declared->program.transfers[d];i++)
      if(t[i].done && t[i].done%8==0 && t[i].done+8<=m->length)order_raise(m,t[i].done,UINT64_MAX);
  }
}
/* The order restarted (a cable of its epoch lost, its epochs exhausted, the region adopted, a client's declaration
   that met a program of a client gone with pieces outstanding): its generation moved, every program of it lost (its
   done words ~0) and its slot vacant, no segment and no epoch until the two bridges agree one */
static inline void order_restart(struct hdr *m,struct mesh_node *x){
  atomic_store_explicit(&x->live,0,memory_order_seq_cst);
  atomic_fetch_add_explicit(&x->generation,1,memory_order_seq_cst);
  atomic_store_explicit(&x->epochs,0,memory_order_seq_cst);
  for(uint32_t p=0;p<MESH_PROGRAMS;p++)if(atomic_load(&x->program[p].owner))order_program_lost(m,x,p);
  for(uint32_t p=0;p<MESH_PROGRAMS;p++){
    struct mesh_declared *declared=x->program+p;
    atomic_store(&declared->owner,0);atomic_store(&declared->issued,0);atomic_store(&declared->closed,UINT64_MAX);
    atomic_store(&declared->end[0],UINT64_MAX);atomic_store(&declared->end[1],UINT64_MAX);
  }
  atomic_store(&x->segments,0);atomic_store(&x->declaring,0);atomic_store(&x->restart,0);
  for(int d=0;d<2;d++){atomic_store(&x->limit[d],UINT64_MAX);atomic_store(&x->raised[d],0);}
}
/* the next epoch appended: from `first` each way, over `cables` (both ends' agreed set, sorted by identity); 0, or
   ENOSPC where the generation holds MESH_EPOCHS already (the caller restarts the order) */
static inline int order_extend(struct mesh_node *x,const uint64_t first[2],uint32_t cables,const uint32_t *port,
                               const uint32_t *lanes,const uint64_t *wire){
  const uint32_t e=atomic_load_explicit(&x->epochs,memory_order_acquire);
  if(e>=MESH_EPOCHS)return ENOSPC;
  struct mesh_epoch *at=x->epoch+e;
  *at=(struct mesh_epoch){.first={first[0],first[1]},.cables=cables};
  for(uint32_t c=0;c<cables && c<MESH_PORTS;c++){at->cable[c].port=port[c];at->cable[c].lanes=lanes[c];at->cable[c].wire=wire[c];}
  atomic_store_explicit(&x->epochs,e+1,memory_order_release);
  atomic_store_explicit(&x->live,1,memory_order_release);
  return 0;
}
#endif
