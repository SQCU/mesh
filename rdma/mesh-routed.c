#include "mesh-plan.h"
#include "mesh-graph.h"
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* A logical transaction has two endpoint views. Its physical fragments also
   have two views; order both from one global key, not from a local peer list. */
struct routed_piece {
  struct mesh_transfer transfer;
  uint32_t rank,local,transaction,round,hop,consume;
  uint64_t offset;
  uint32_t *retired,retires;
  unsigned char witnessed;
  struct mesh_packet_view view;
};

static int physical_order(const void *a,const void *b){
  const struct routed_piece *x=a,*y=b;
  const uint32_t left[]={x->round,x->hop,x->transaction,x->transfer.cable};
  const uint32_t right[]={y->round,y->hop,y->transaction,y->transfer.cable};
  for(unsigned k=0;k<4;k++)if(left[k]!=right[k])return left[k]<right[k]?-1:1;
  if(x->offset!=y->offset)return x->offset<y->offset?-1:1;
  return (x->transfer.step.op!=MESH_STEP_SEND)-(y->transfer.step.op!=MESH_STEP_SEND);
}

static int local_order(const void *a,const void *b){
  const struct routed_piece *x=a,*y=b;
  if(x->rank!=y->rank)return x->rank<y->rank?-1:1;
  if(x->local!=y->local)return x->local<y->local?-1:1;
  if(x->transfer.step.round!=y->transfer.step.round)return x->transfer.step.round<y->transfer.step.round?-1:1;
  return (x->transfer.step.op!=MESH_STEP_SEND)-(y->transfer.step.op!=MESH_STEP_SEND);
}

static int number_pieces(struct routed_piece *pieces,size_t count){
  qsort(pieces,count,sizeof *pieces,physical_order);
  for(size_t k=0;k<count;k+=2){
    if(k+1==count)return EPROTO;
    struct routed_piece *a=pieces+k,*b=a+1;
    if(a->transfer.step.op!=MESH_STEP_SEND || b->transfer.step.op==MESH_STEP_SEND ||
       a->transaction!=b->transaction || a->hop!=b->hop || a->offset!=b->offset || a->transfer.cable!=b->transfer.cable ||
       a->rank!=b->transfer.step.peer || b->rank!=a->transfer.step.peer ||
       a->transfer.step.piece.elements!=b->transfer.step.piece.elements)return EPROTO;
    a->transfer.step.round=b->transfer.step.round=(uint32_t)(k/2);
  }
  qsort(pieces,count,sizeof *pieces,local_order);
  return 0;
}

/* Restore streaming refinement before proving lifetimes. Split both endpoint
   views together, so each publication/arrival event still names one exact byte
   interval. Repeat until upstream cuts have propagated through all relays.
   Each change adds an existing page boundary, hence the finite page graph
   bounds this closure; no transport hop or path is excluded. */
static int refine_pieces(struct routed_piece **storage,size_t *length,uint64_t page){
  int status=0,changed=1;
  while(changed && !status){
    struct routed_piece *pieces=*storage;
    const size_t count=*length,pairs=count/2;
    struct cuts { uint64_t *end;uint32_t count; };
    struct cuts *cuts=calloc(pairs+1,sizeof *cuts);
    if(!cuts)return ENOMEM;
    size_t extra=0;
    changed=0;
    for(size_t first=0;first<count && !status;){
      size_t last=first+1;
      while(last<count && pieces[last].rank==pieces[first].rank)last++;
      const uint32_t n=(uint32_t)(last-first);
      struct mesh_transfer *input=malloc((size_t)n*sizeof *input);
      if(!input){status=ENOMEM;break;}
      for(uint32_t k=0;k<n;k++)input[k]=pieces[first+k].transfer;
      const int room=mesh_fabric_refine(input,n,page,NULL,0);
      /* Non-page-aligned dependencies retain their existing whole-piece proof.
         This pass only refines representable streaming boundaries. */
      if(room==-EINVAL){free(input);first=last;continue;}
      struct mesh_transfer *output=room<0?NULL:malloc(((size_t)room+1)*sizeof *output);
      if(room<0 || !output){free(input);status=room<0?-room:ENOMEM;break;}
      if(mesh_fabric_refine(input,n,page,output,(uint32_t)room)!=room)status=EPROTO;
      for(uint32_t k=0,at=0;k<n && !status;k++){
        const struct mesh_step *original=&input[k].step;
        uint32_t begin=at;
        uint64_t extent=0;
        do { extent+=output[at++].step.piece.elements; } while(at<(uint32_t)room && extent<original->piece.elements);
        if(extent!=original->piece.elements){status=EPROTO;break;}
        if(at==begin+1)continue;
        struct cuts *cut=cuts+original->round;
        cut->count=at-begin;
        cut->end=malloc((size_t)cut->count*sizeof *cut->end);
        if(!cut->end){status=ENOMEM;break;}
        extent=0;
        for(uint32_t j=begin;j<at;j++){extent+=output[j].step.piece.elements;cut->end[j-begin]=extent;}
        extra+=2*((size_t)cut->count-1);changed=1;
      }
      free(input);free(output);first=last;
    }
    if(!status && changed){
      if(extra>UINT32_MAX-count || count+extra>SIZE_MAX/sizeof **storage)status=EOVERFLOW;
      struct routed_piece *next=status?NULL:malloc((count+extra)*sizeof *next);
      if(!status && !next)status=ENOMEM;
      if(!status){
        size_t written=0;
        for(size_t k=0;k<count;k++){
          const struct cuts *cut=cuts+pieces[k].transfer.step.round;
          if(!cut->count){next[written++]=pieces[k];continue;}
          uint64_t first=0;
          for(uint32_t j=0;j<cut->count;j++){
            struct routed_piece p=pieces[k];
            p.offset+=first;p.transfer.step.first+=first;p.view.first+=first*p.transfer.step.piece.element_bytes;
            p.transfer.step.piece.elements=cut->end[j]-first;
            next[written++]=p;first=cut->end[j];
          }
        }
        free(pieces);*storage=next;*length=count+extra;
        status=number_pieces(next,*length);
      }
    }
    for(size_t k=0;k<pairs;k++)free(cuts[k].end);
    free(cuts);
  }
  return status;
}

struct routed_edge { uint32_t before,after; };
static int edge_order(const void *a,const void *b){
  const struct routed_edge *x=a,*y=b;
  if(x->before!=y->before)return x->before<y->before?-1:1;
  return (x->after>y->after)-(x->after<y->after);
}

/* A COPY can overwrite an old SEND generation when its arrival is a causal
   witness that the NIC has already read that generation. Keep first-publication,
   full-arrival and local-consumption events separate: forwarding can start before
   a whole COPY is consumed. Only strict data edges, never optional publication
   guards, establish this lifetime certificate. */
static int proof_edge(struct routed_edge **edges,size_t *used,size_t *room,uint32_t before,uint32_t after){
  if(*used==*room){
    const size_t next=*room?*room*2:64;
    if(next<*room || next>SIZE_MAX/sizeof **edges)return EOVERFLOW;
    struct routed_edge *grown=realloc(*edges,next*sizeof **edges);
    if(!grown)return ENOMEM;
    *edges=grown;*room=next;
  }
  /* Reverse edges: queries walk from an incoming first publication to its
     strict causal ancestors, looking for a prior outgoing full arrival. */
  (*edges)[(*used)++]=(struct routed_edge){after,before};
  return 0;
}

struct byte_witness {
  uint32_t transaction,event,point;
  uint64_t first,last;
};
static int byte_witness_add(struct byte_witness **states,size_t *count,size_t *room,struct byte_witness next){
  for(size_t k=0;k<*count;k++){
    const struct byte_witness *s=*states+k;
    if(s->transaction==next.transaction && s->event==next.event && s->point==next.point &&
       s->first==next.first && s->last==next.last)return 0;
  }
  if(*count==*room){
    const size_t n=*room?2**room:32;
    if(n<*room || n>SIZE_MAX/sizeof **states)return EOVERFLOW;
    struct byte_witness *more=realloc(*states,n*sizeof **states);
    if(!more)return ENOMEM;
    *states=more;*room=n;
  }
  (*states)[(*count)++]=next;return 0;
}
static int pieces_meet(const struct mesh_step *a,const struct mesh_step *b){
  return a->buffer==b->buffer && a->first<b->first+b->piece.elements && b->first<a->first+a->piece.elements;
}

/* Whole-entry arrival is stronger than a streaming overwrite needs. For each
   byte x in the overwritten intersection, walk the strict ancestry of that
   incoming byte's publication. Point states preserve x's affine byte offset
   through identity relays. A consumed predecessor instead establishes a whole
   interval before every x. Only arrival of the old outgoing byte is a witness;
   publication alone never retires a source. This is a compiler proof, not an
   extra execution dependency or a wait for an entire forwarding entry. */
static int byte_retired(const struct routed_piece *pieces,const size_t *send,const size_t *land,
                        size_t incoming,size_t outgoing,uint64_t page,int *proven){
  const struct mesh_step *copy=&pieces[incoming].transfer.step,*old=&pieces[outgoing].transfer.step;
  const uint64_t cfirst=copy->first*copy->piece.element_bytes,ofirst=old->first*old->piece.element_bytes;
  const uint64_t clast=cfirst+copy->piece.elements*copy->piece.element_bytes;
  const uint64_t olast=ofirst+old->piece.elements*old->piece.element_bytes;
  const uint64_t begin=cfirst>ofirst?cfirst:ofirst,end=clast<olast?clast:olast;
  struct byte_witness *states=NULL,*covered=NULL;size_t count=0,room=0,covers=0,cover_room=0;
  *proven=0;
  int status=byte_witness_add(&states,&count,&room,(struct byte_witness){copy->round,0,1,begin-cfirst,end-cfirst});
  for(size_t at=0;at<count && !status && !*proven;at++){
    const struct byte_witness state=states[at];
    const size_t k=state.event==0?send[state.transaction]:land[state.transaction];
    const struct mesh_step *s=&pieces[k].transfer.step;
    if(state.event==1){
      if(state.transaction==old->round){
        if(state.point){
          /* The same byte, not merely another byte in the same transaction. */
          if(state.first==begin-ofirst)*proven=1;
        }else{
          const uint64_t lo=ofirst+state.first>begin?ofirst+state.first:begin;
          const uint64_t hi=ofirst+state.last<end?ofirst+state.last:end;
          if(lo<hi)status=byte_witness_add(&covered,&covers,&cover_room,(struct byte_witness){0,0,0,lo,hi});
        }
      }
      if(!status)status=byte_witness_add(&states,&count,&room,(struct byte_witness){state.transaction,0,state.point,state.first,state.last});
      continue;
    }
    size_t start=k;while(start && pieces[start-1].rank==pieces[k].rank)start--;
    size_t latest=k;
    for(size_t j=start;j<k;j++)if(pieces[j].transfer.step.op!=MESH_STEP_SEND && pieces_meet(&pieces[j].transfer.step,s))latest=j;
    const struct mesh_step *a=&pieces[latest].transfer.step;
    const int forwarding=state.event==0 && latest<k && a->op==MESH_STEP_COPY &&
      a->piece.element_bytes==s->piece.element_bytes && a->first<=s->first &&
      s->first+s->piece.elements<=a->first+a->piece.elements &&
      (s->first-a->first)*s->piece.element_bytes%page==0;
    if(forwarding){
      const uint64_t shift=(s->first-a->first)*s->piece.element_bytes;
      status=byte_witness_add(&states,&count,&room,(struct byte_witness){a->round,1,state.point,shift+state.first,shift+state.last});
    }else{
      if(state.event==2)status=byte_witness_add(&states,&count,&room,
        (struct byte_witness){state.transaction,1,0,0,s->piece.elements*s->piece.element_bytes});
      for(size_t j=start;j<k && !status;j++){
        a=&pieces[j].transfer.step;
        if(a->op!=MESH_STEP_SEND && pieces_meet(a,s))status=byte_witness_add(&states,&count,&room,
          (struct byte_witness){a->round,2,0,0,a->piece.elements*a->piece.element_bytes});
      }
    }
  }
  /* Different complete predecessor slices may jointly certify the overlap. */
  uint64_t frontier=begin;
  for(size_t pass=0;pass<covers && frontier<end;pass++){
    const uint64_t prior=frontier;
    for(size_t k=0;k<covers;k++)if(covered[k].first<=frontier && covered[k].last>frontier)frontier=covered[k].last;
    if(frontier==prior)break;
  }
  if(frontier==end)*proven=1;
  free(states);free(covered);return status;
}
/* edge k's event entered (mesh-graph.h's edge heads, over the edges' compressed rows) */
static uint32_t routed_after(const void *context,size_t k){ return ((const struct routed_edge *)context)[k].after; }
static int retirement_proofs(struct routed_piece *pieces,size_t count,uint64_t page){
  if(count/2>UINT32_MAX/3)return EOVERFLOW;
  const uint32_t events=(uint32_t)(count/2)*3;
  struct routed_edge *edges=NULL;size_t used=0,room=0;
  int status=0;
  for(size_t begin=0,k=0;k<count && !status;k++){
    if(k && pieces[k].rank!=pieces[k-1].rank)begin=k;
    const struct mesh_step *s=&pieces[k].transfer.step;
    const uint32_t event=3*s->round;
    if(s->op!=MESH_STEP_SEND){
      if((status=proof_edge(&edges,&used,&room,event,event+1)) ||
         (status=proof_edge(&edges,&used,&room,event+1,event+2)))break;
    }
    size_t latest=k;
    for(size_t j=begin;j<k;j++){
      const struct mesh_step *a=&pieces[j].transfer.step;
      if(a->op!=MESH_STEP_SEND && a->buffer==s->buffer &&
         a->first<s->first+s->piece.elements && s->first<a->first+a->piece.elements)latest=j;
    }
    const struct mesh_step *a=&pieces[latest].transfer.step;
    const int forwarding=s->op==MESH_STEP_SEND && latest<k && a->op==MESH_STEP_COPY &&
      a->first<=s->first && s->first+s->piece.elements<=a->first+a->piece.elements &&
      (s->first-a->first)*s->piece.element_bytes%page==0;
    if(forwarding){
      status=proof_edge(&edges,&used,&room,3*a->round,event);
      if(!status && a->first==s->first && a->piece.elements==s->piece.elements)
        status=proof_edge(&edges,&used,&room,3*a->round+1,event+1);
    }else for(size_t j=begin;j<k && !status;j++){
      a=&pieces[j].transfer.step;
      if(a->op==MESH_STEP_SEND || a->buffer!=s->buffer ||
         a->first>=s->first+s->piece.elements || s->first>=a->first+a->piece.elements)continue;
      status=proof_edge(&edges,&used,&room,3*a->round+2,event+(s->op!=MESH_STEP_SEND?2:0));
    }
  }
  size_t *first=calloc((size_t)events+1,sizeof *first);
  uint32_t *seen=calloc((size_t)events+1,sizeof *seen),*stack=malloc(((size_t)events+1)*sizeof *stack);
  size_t *send=malloc((count/2+1)*sizeof *send),*land=malloc((count/2+1)*sizeof *land);
  if(!first || !seen || !stack || !send || !land)status=ENOMEM;
  if(!status){
    for(size_t k=0;k<count;k++)(pieces[k].transfer.step.op==MESH_STEP_SEND?send:land)[pieces[k].transfer.step.round]=k;
    if(used)qsort(edges,used,sizeof *edges,edge_order);
    for(size_t k=0;k<used;k++)first[edges[k].before+1]++;
    for(uint32_t k=0;k<events;k++)first[k+1]+=first[k];
    for(size_t begin=0,k=0;k<count && !status;k++){
      if(k && pieces[k].rank!=pieces[k-1].rank)begin=k;
      const struct mesh_step *s=&pieces[k].transfer.step;
      if(s->op==MESH_STEP_SEND)continue; /* a COPY landing in place, or a combine into its buffer */
      const uint32_t stamp=(uint32_t)k+1,root=3*s->round;
      stack[0]=root;seen[root]=stamp;
      mesh_graph_mark(first,routed_after,edges,seen,stamp,stack,1);
      for(size_t j=begin;j<k;j++){
        const struct mesh_step *a=&pieces[j].transfer.step;
        if(a->op!=MESH_STEP_SEND || !pieces_meet(a,s))continue;
        int proven=seen[3*a->round+1]==stamp;
        if(!proven)status=byte_retired(pieces,send,land,k,j,page,&proven);
        if(status)break;
        if(!proven)continue;
        uint32_t *more=realloc(pieces[k].retired,((size_t)pieces[k].retires+1)*sizeof *more);
        if(!more){status=ENOMEM;break;}
        pieces[k].retired=more;more[pieces[k].retires++]=(uint32_t)(j-begin);
      }
    }
    /* A SEND is witnessed where its arrival is a strict causal ancestor of a
       later landing at its own rank: that landing proves the peer read it. */
    for(size_t begin=0,end=0;!status && begin<count;begin=end){
      while(end<count && pieces[end].rank==pieces[begin].rank)end++;
      const uint32_t stamp=(uint32_t)(count+1+begin);
      size_t pending=0;
      for(size_t k=begin;k<end;k++){
        const uint32_t root=3*pieces[k].transfer.step.round;
        if(pieces[k].transfer.step.op!=MESH_STEP_SEND && seen[root]!=stamp){seen[root]=stamp;stack[pending++]=root;}
      }
      mesh_graph_mark(first,routed_after,edges,seen,stamp,stack,pending);
      for(size_t k=begin;k<end;k++)
        pieces[k].witnessed=pieces[k].transfer.step.op==MESH_STEP_SEND && seen[3*pieces[k].transfer.step.round+1]==stamp;
    }
  }
  free(edges);free(first);free(seen);free(stack);free(send);free(land);return status;
}

/* Publications and consumptions are distinct events: publishing a SEND does
   not complete its remote arrival or retire its source. Physical wire lists
   follow publication order. Each local consumer follows consumption order,
   which may differ for independently staged arrivals on one physical cable. */
static int dependency_order(struct routed_piece *pieces,size_t count){
  if(count>UINT32_MAX)return EOVERFLOW;
  const uint32_t events=(uint32_t)count;
  struct routed_edge *edges=NULL;
  size_t used=0,room=0;
  int status=0;
  for(size_t begin=0,k=0;k<count && !status;k++){
    if(k && pieces[k].rank!=pieces[k-1].rank)begin=k;
    const struct mesh_step *s=&pieces[k].transfer.step;
    /* A physical arrival follows its source publication. The bridge may
       satisfy this edge page by page; no remote software ACK is involved. */
    for(size_t j=begin;j<=k;j++){
      const struct mesh_step *prior=&pieces[j].transfer.step;
      uint32_t before,after;
      if(j==k){
        if(s->op==MESH_STEP_SEND)continue;
        before=2*s->round;after=before+1;
      }else{
        if(prior->buffer!=s->buffer || prior->first>=s->first+s->piece.elements ||
           s->first>=prior->first+prior->piece.elements)continue;
        if(prior->op==MESH_STEP_SEND){
          /* The existing in-place combine guard requires publication, not
             transfer completion. Keep the two directions of an exchange
             executable instead of contracting them into a false cycle. */
          if(s->op!=MESH_STEP_REDUCE)continue;
          before=2*prior->round;after=2*s->round+1;
        }else{
          before=2*prior->round+1;after=2*s->round+(s->op!=MESH_STEP_SEND);
        }
      }
      if(before==after){status=EPROTO;break;}
      if(used==room){
        const size_t next=room?room*2:64;
        if(next<room || next>SIZE_MAX/sizeof *edges){status=EOVERFLOW;break;}
        struct routed_edge *grown=realloc(edges,next*sizeof *edges);
        if(!grown){status=ENOMEM;break;}
        edges=grown;room=next;
      }
      edges[used++]=(struct routed_edge){before,after};
    }
  }
  uint32_t *indegree=calloc((size_t)events+1,sizeof *indegree);
  uint32_t *order=malloc(((size_t)events+1)*sizeof *order);
  size_t *first=calloc((size_t)events+1,sizeof *first);
  if(!indegree || !order || !first)status=ENOMEM;
  if(!status){
    if(used)qsort(edges,used,sizeof *edges,edge_order);
    size_t unique=0;
    for(size_t k=0;k<used;k++)if(!k || edge_order(edges+k,edges+k-1))edges[unique++]=edges[k];
    used=unique;
    for(size_t k=0;k<used;k++)first[edges[k].before+1]++;
    for(uint32_t k=0;k<events;k++)first[k+1]+=first[k];
    /* the events' order (mesh-graph.h): each next place the lowest-numbered event whose priors are placed */
    status=mesh_graph_order(events,first,routed_after,edges,indegree,order);
    if(!status)for(size_t k=0;k<count;k++){
      const uint32_t transaction=pieces[k].transfer.step.round;
      pieces[k].transfer.step.round=order[2*transaction];
      pieces[k].consume=order[2*transaction+1];
    }
  }
  free(edges);free(indegree);free(order);free(first);return status;
}

void mesh_routed_free(struct mesh_routed_program *p){
  if(!p)return;
  for(uint32_t r=0;p->rank && r<p->nodes;r++){
    free(p->rank[r].transfer);free(p->rank[r].consume);free(p->rank[r].extent);
    free(p->rank[r].retired_first);free(p->rank[r].retired);free(p->rank[r].witnessed);
    free(p->rank[r].views);
  }
  free(p->rank);free(p->demand);free(p->route_units);*p=(struct mesh_routed_program){0};
}

int mesh_routed_compile(const struct mesh_fabric *f,const struct mesh_paths *paths,
                         const uint32_t *first,const struct mesh_step *steps,
                         const uint32_t *initialized,uint64_t page,enum mesh_packet_coordinates coordinates,struct mesh_routed_program *out){
  if(!out)return EINVAL;
  *out=(struct mesh_routed_program){0};
  if(mesh_fabric_check(f) || !paths || !first || !steps || !initialized || !page || first[0])return EINVAL;
  const uint32_t N=f->nodes;
  if((uint64_t)N*N>SIZE_MAX/sizeof(uint64_t))return EOVERFLOW;
  for(uint32_t r=0;r<N;r++)if(first[r+1]<first[r] || !initialized[r])return EINVAL;
  const uint32_t logical=first[N];
  unsigned char *matched=calloc((size_t)logical+1,1);
  uint32_t *relay=malloc((size_t)N*sizeof *relay);
  struct routed_piece *pieces=NULL;
  size_t count=0;
  struct mesh_routed_program made={.nodes=N,.rank=calloc(N,sizeof *made.rank),.demand=calloc((size_t)N*N,sizeof *made.demand),
                                  .route_units=calloc((size_t)N*N,sizeof *made.route_units)};
  int status=matched && relay && made.rank && made.demand && made.route_units?0:ENOMEM;
  for(uint32_t r=0;r<N && !status;r++){
    struct mesh_routed_rank *rank=made.rank+r;
    rank->buffers=rank->initialized=initialized[r];
    rank->extent=calloc(rank->buffers,sizeof *rank->extent);
    if(!rank->extent){status=ENOMEM;break;}
    for(uint32_t k=first[r];k<first[r+1];k++){
      const struct mesh_step *s=steps+k;
      if(s->peer>=N || s->peer==r || s->buffer>=rank->buffers || s->op>MESH_STEP_COPY || !s->piece.element_bytes){status=EINVAL;break;}
      if(s->first>UINT64_MAX-s->piece.elements || s->first+s->piece.elements>UINT64_MAX/s->piece.element_bytes){status=EOVERFLOW;break;}
      const uint64_t extent=(s->first+s->piece.elements)*s->piece.element_bytes;
      if(extent>rank->extent[s->buffer])rank->extent[s->buffer]=extent;
    }
  }
  uint32_t transaction=0;
  for(uint32_t source=0;source<N && !status;source++)for(uint32_t k=first[source];k<first[source+1] && !status;k++){
    const struct mesh_step *send=steps+k;
    if(send->op!=MESH_STEP_SEND || !send->piece.elements)continue;
    const uint32_t destination=send->peer;
    uint32_t receive=first[destination];
    for(;receive<first[destination+1];receive++){
      const struct mesh_step *s=steps+receive;
      if(!matched[receive] && s->op!=MESH_STEP_SEND && s->peer==source && s->round==send->round &&
         s->piece.elements==send->piece.elements && s->piece.element_bytes==send->piece.element_bytes && s->piece.type==send->piece.type)break;
    }
    if(receive==first[destination+1]){status=EPROTO;break;}
    matched[receive]=1;
    const struct mesh_step *land=steps+receive;
    const uint64_t bytes=send->piece.elements*send->piece.element_bytes;
    uint64_t *demand=made.demand+(size_t)source*N+destination;
    if(*demand>UINT64_MAX-bytes){status=EOVERFLOW;break;}
    *demand+=bytes;
    uint64_t divisor=page,remainder=send->piece.element_bytes;
    while(remainder){const uint64_t next=divisor%remainder;divisor=remainder;remainder=next;}
    const uint64_t units=(send->piece.elements-1)/(page/divisor)+1;
    uint64_t *resolution=made.route_units+(size_t)source*N+destination;
    if(units>*resolution)*resolution=units;
    for(uint32_t r=0;r<N;r++)relay[r]=UINT32_MAX;
    for(uint32_t r=0;r<N && !status;r++){
      const struct mesh_paths *along=paths+(size_t)source*N+destination;
      const int needed=mesh_write_compile(f,r,source,destination,along,send->piece,page,0,NULL,0);
      if(needed<0){status=-needed;break;}
      if(!needed)continue;
      if((size_t)needed>UINT32_MAX-count || count+(size_t)needed>SIZE_MAX/sizeof *pieces){status=EOVERFLOW;break;}
      struct mesh_transfer *physical=malloc((size_t)needed*sizeof *physical);
      struct routed_piece *grown=realloc(pieces,(count+(size_t)needed)*sizeof *pieces);
      if(grown)pieces=grown;
      if(!physical || !grown){free(physical);status=ENOMEM;break;}
      const int written=mesh_write_compile(f,r,source,destination,along,send->piece,page,0,physical,(uint32_t)needed);
      if(written!=needed){free(physical);status=written<0?-written:EPROTO;break;}
      if(r!=source && r!=destination){
        struct mesh_routed_rank *rank=made.rank+r;
        if(rank->buffers==UINT32_MAX){free(physical);status=EOVERFLOW;break;}
        uint64_t *extent=realloc(rank->extent,((size_t)rank->buffers+1)*sizeof *extent);
        if(!extent){free(physical);status=ENOMEM;break;}
        rank->extent=extent;relay[r]=rank->buffers++;
        /* Each commodity owns a distinct generation at a relay. No local
           tensor operand is overwritten just because a path crosses here. */
        rank->extent[relay[r]]=bytes;
      }
      for(int j=0;j<written;j++){
        struct mesh_transfer t=physical[j];
        const uint64_t offset=t.step.first;
        const uint32_t hop=t.step.round;
        uint32_t local=UINT32_MAX;
        if(r==source){t.step.buffer=send->buffer;t.step.first+=send->first;local=k-first[r];}
        else if(r==destination){t.step.op=land->op;t.step.buffer=land->buffer;t.step.first+=land->first;local=receive-first[r];}
        else t.step.buffer=relay[r];
        pieces[count++]=(struct routed_piece){t,r,local,transaction,send->round,hop,0,offset,NULL,0,0,
          {(offset+(coordinates==MESH_PACKET_OPERAND?send->first:0))*send->piece.element_bytes,
           coordinates==MESH_PACKET_OPERAND?0:transaction,0}};
        made.rank[r].count++;
      }
      free(physical);
    }
    transaction++;
  }
  for(uint32_t k=0;k<logical && !status;k++)
    if(steps[k].op!=MESH_STEP_SEND && steps[k].piece.elements && !matched[k])status=EPROTO;
  if(!status)status=number_pieces(pieces,count);
  if(!status)status=refine_pieces(&pieces,&count,page);
  if(!status){
    for(uint32_t r=0;r<N;r++)made.rank[r].count=0;
    for(size_t k=0;k<count;k++)made.rank[pieces[k].rank].count++;
    status=retirement_proofs(pieces,count,page);
    if(!status)status=dependency_order(pieces,count);
  }
  if(!status){
    size_t at=0;
    for(uint32_t r=0;r<N;r++){
      struct mesh_routed_rank *rank=made.rank+r;
      rank->transfer=calloc((size_t)rank->count+1,sizeof *rank->transfer);
      rank->consume=calloc((size_t)rank->count+1,sizeof *rank->consume);
      rank->retired_first=calloc((size_t)rank->count+1,sizeof *rank->retired_first);
      rank->witnessed=calloc((size_t)rank->count+1,1);
      rank->views=calloc((size_t)rank->count+1,sizeof *rank->views);
      size_t total=0;for(uint32_t k=0;k<rank->count;k++)total+=pieces[at+k].retires;
      if(total>UINT32_MAX){status=EOVERFLOW;break;}
      rank->retired=malloc((total+1)*sizeof *rank->retired);
      if(!rank->transfer || !rank->consume || !rank->retired_first || !rank->retired || !rank->witnessed || !rank->views){status=ENOMEM;break;}
      uint32_t retired=0;
      for(uint32_t k=0;k<rank->count;k++,at++){
        rank->transfer[k]=pieces[at].transfer;rank->consume[k]=pieces[at].consume;rank->witnessed[k]=pieces[at].witnessed;
        rank->views[k]=pieces[at].view;
        rank->retired_first[k]=retired;
        if(pieces[at].retires)memcpy(rank->retired+retired,pieces[at].retired,pieces[at].retires*sizeof *rank->retired);
        retired+=pieces[at].retires;
      }
      rank->retired_first[rank->count]=retired;
    }
  }
  for(size_t k=0;k<count;k++)free(pieces[k].retired);
  free(pieces);free(matched);free(relay);
  made.domains=coordinates==MESH_PACKET_OPERAND?!!transaction:transaction;
  if(status)mesh_routed_free(&made);else *out=made;
  return status;
}
