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

/* Restore streaming refinement before ordering. Split both endpoint
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

/* Publications and consumptions are distinct events: publishing a SEND does
   not complete its remote arrival or retire its source. Physical wire lists
   follow publication order. Each local consumer follows consumption order,
   which may differ for independently staged arrivals on one physical cable. */
struct transfer_endpoint {
  uint32_t group,source,destination,cable,round,type,width,entry;
  uint64_t elements;
};
static int endpoint_order(const void *left,const void *right){
  const struct transfer_endpoint *a=left,*b=right;
  const uint32_t x[]={a->group,a->source,a->destination,a->cable,a->round,a->type,a->width};
  const uint32_t y[]={b->group,b->source,b->destination,b->cable,b->round,b->type,b->width};
  for(unsigned i=0;i<sizeof x/sizeof *x;i++)if(x[i]!=y[i])return x[i]<y[i]?-1:1;
  if(a->elements!=b->elements)return a->elements<b->elements?-1:1;
  return (a->entry>b->entry)-(a->entry<b->entry);
}
static int order_edge(struct mesh_graph_follow **edges,size_t *used,size_t *room,
                      uint32_t before,uint32_t after,uint32_t offset,enum mesh_graph_wait wait){
  if(*used==*room){
    const size_t next=*room?2**room:1;
    if(next<*room||next>SIZE_MAX/sizeof **edges)return EOVERFLOW;
    struct mesh_graph_follow *made=realloc(*edges,next*sizeof *made);
    if(!made)return ENOMEM;
    *edges=made;*room=next;
  }
  (*edges)[(*used)++]=(struct mesh_graph_follow){before,after,offset,wait};
  return 0;
}

/* Each transfer a move of page units on its directed cable (a receive or combine an event on none), each follow a
   unit's real wait: a pair's receive unit on its SEND unit, a forward's or a combine's unit on the landing unit it
   reads, and the SEND that fills a landing writing bytes an earlier SEND of its rank reads on that SEND's last unit
   (design/pages-and-functions.md Freeing 6-7: the bridge posts no unit of the landing before that SEND completed,
   mesh.h struct mesh_zwoggle_entry `after`, and the hardware's credit holds the filling SEND until it is posted), so
   an exchange in place, each side's landing waiting on its own SEND, is a cycle (EDEADLK), refused here.
   mesh_graph_schedule fills every cable from the units ready on it, highest first, so a relay offsets its pages by
   its hops and takes no cable's time from it. */
int mesh_transfer_order(const struct mesh_fabric *fabric,uint32_t groups,const uint32_t *first,
                        const struct mesh_transfer *transfer,const struct mesh_transfer_production *production,
                        uint64_t page,uint32_t *publish,uint32_t *consume,double *finish){
  if(mesh_fabric_check(fabric)||!first||!page||first[0])return EINVAL;
  const size_t rows=(size_t)groups*fabric->nodes;
  for(size_t row=0;row<rows;row++)if(first[row]>first[row+1])return EINVAL;
  const uint32_t count=first[rows],arcs=2*fabric->cables;
  if(fabric->cables>UINT32_MAX/2)return EOVERFLOW;
  if(count&&(!transfer||!publish||!consume))return EINVAL;
  struct mesh_graph_move *moves=calloc((size_t)count+1,sizeof *moves);
  struct transfer_endpoint *sent=malloc(((size_t)count+1)*sizeof *sent),
                           *received=malloc(((size_t)count+1)*sizeof *received);
  double *capacity=malloc(((size_t)arcs+1)*sizeof *capacity);
  uint32_t *order=malloc(((size_t)count+1)*sizeof *order);
  uint32_t *filled=malloc(((size_t)count+1)*sizeof *filled);
  struct mesh_graph_follow *edges=NULL,*overwrites=NULL;
  size_t used=0,room=0,overwritten=0,overwrite_room=0;
  uint32_t sends=0,receives=0;
  int status=moves&&sent&&received&&capacity&&order&&filled?0:ENOMEM;
  for(uint32_t cable=0;!status&&cable<fabric->cables;cable++)
    capacity[2*cable]=capacity[2*cable+1]=fabric->cable[cable].bandwidth/8.0;
  for(size_t row=0;!status&&row<rows;row++){
    const uint32_t rank=(uint32_t)(row%fabric->nodes),group=(uint32_t)(row/fabric->nodes);
    for(uint32_t k=first[row];!status&&k<first[row+1];k++){
      const struct mesh_transfer *t=transfer+k;
      const struct mesh_step *s=&t->step;
      if(t->cable>=fabric->cables||s->op>MESH_STEP_COPY||!s->piece.element_bytes||
         s->peer>=fabric->nodes||s->peer==rank||s->first>UINT64_MAX-s->piece.elements||
         s->first+s->piece.elements>UINT64_MAX/s->piece.element_bytes){status=EINVAL;break;}
      const struct mesh_cable *cable=fabric->cable+t->cable;
      if(!((cable->node[0]==rank&&cable->node[1]==s->peer)||
           (cable->node[1]==rank&&cable->node[0]==s->peer))){status=EINVAL;break;}
      const int sending=s->op==MESH_STEP_SEND;
      struct transfer_endpoint endpoint={group,sending?rank:s->peer,sending?s->peer:rank,
        t->cable,s->round,s->piece.type,s->piece.element_bytes,k,s->piece.elements};
      if(sending)sent[sends++]=endpoint;else received[receives++]=endpoint;
      const uint64_t bytes=mesh_step_bytes(s),units=bytes?(bytes+page-1)/page:1;
      if(units>UINT32_MAX){status=EOVERFLOW;break;}
      const uint32_t arc=sending&&bytes?2*t->cable+(rank==cable->node[1]):arcs;
      moves[k]=(struct mesh_graph_move){.arc=arc,.units=(uint32_t)units,
        .production=production?production[k].order:0,.release=production?production[k].seconds:0,
        .unit_seconds=arc<arcs?(double)page/capacity[arc]:0,
        .last_seconds=arc<arcs?(double)(bytes-(units-1)*page)/capacity[arc]:0};
      for(uint32_t j=first[row];!status&&j<k;j++){
        const struct mesh_step *prior=&transfer[j].step;
        if(prior->buffer!=s->buffer||!prior->piece.elements||!s->piece.elements)continue;
        const uint64_t a=prior->first*prior->piece.element_bytes,b=s->first*s->piece.element_bytes,
                       an=mesh_step_bytes(prior),bn=mesh_step_bytes(s);
        if(a>=b+bn||b>=a+an)continue;
        if(prior->op==MESH_STEP_SEND){
          /* a landing (a COPY's, or a combine's into these bytes) after the SEND's last unit, the SEND filling it
             found once the pairs are */
          if(s->op!=MESH_STEP_SEND)status=order_edge(&overwrites,&overwritten,&overwrite_room,j,k,0,MESH_GRAPH_COMPLETE);
        }else if(a<=b&&b+bn<=a+an&&(b-a)%page==0){
          if((b-a)/page>UINT32_MAX){status=EOVERFLOW;break;}
          status=order_edge(&edges,&used,&room,j,k,(uint32_t)((b-a)/page),MESH_GRAPH_STREAMED);
        }else status=order_edge(&edges,&used,&room,j,k,0,MESH_GRAPH_COMPLETE);
      }
    }
  }
  if(!status&&sends!=receives)status=EPROTO;
  if(!status){
    qsort(sent,sends,sizeof *sent,endpoint_order);qsort(received,receives,sizeof *received,endpoint_order);
    for(uint32_t k=0;k<sends&&!status;k++){
      struct transfer_endpoint a=sent[k],b=received[k];
      a.entry=b.entry=0;
      if(endpoint_order(&a,&b)){status=EPROTO;break;}
      status=order_edge(&edges,&used,&room,sent[k].entry,received[k].entry,0,MESH_GRAPH_STREAMED);
      filled[received[k].entry]=sent[k].entry;
    }
    for(size_t e=0;e<overwritten&&!status;e++)
      status=order_edge(&edges,&used,&room,overwrites[e].before,filled[overwrites[e].after],0,MESH_GRAPH_COMPLETE);
  }
  if(!status)status=mesh_graph_schedule(moves,count,arcs,edges,used,order,finish);
  if(!status)for(uint32_t k=0;k<sends;k++){
    const uint32_t send=sent[k].entry,receive=received[k].entry;
    publish[send]=publish[receive]=order[send];
    consume[send]=consume[receive]=order[receive];
  }
  free(moves);free(sent);free(received);free(capacity);free(order);free(edges);free(overwrites);free(filled);return status;
}

static int dependency_order(const struct mesh_fabric *fabric,struct routed_piece *pieces,size_t count,uint64_t page){
  if(count>UINT32_MAX)return EOVERFLOW;
  uint32_t *first=calloc((size_t)fabric->nodes+1,sizeof *first),
           *publication=malloc((count+1)*sizeof *publication),*consumption=malloc((count+1)*sizeof *consumption);
  struct mesh_transfer *transfers=malloc((count+1)*sizeof *transfers);
  int status=first&&publication&&consumption&&transfers?0:ENOMEM;
  if(!status){
    for(size_t k=0;k<count;k++){first[pieces[k].rank+1]++;transfers[k]=pieces[k].transfer;}
    for(uint32_t rank=0;rank<fabric->nodes;rank++)first[rank+1]+=first[rank];
    status=mesh_transfer_order(fabric,1,first,transfers,NULL,page,publication,consumption,NULL);
    if(!status)for(size_t k=0;k<count;k++){
      pieces[k].transfer.step.round=publication[k];pieces[k].consume=consumption[k];
    }
  }
  free(first);free(publication);free(consumption);free(transfers);return status;
}

void mesh_routed_free(struct mesh_routed_program *p){
  if(!p)return;
  for(uint32_t r=0;p->rank && r<p->nodes;r++){
    free(p->rank[r].transfer);free(p->rank[r].consume);free(p->rank[r].extent);
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
        pieces[count++]=(struct routed_piece){t,r,local,transaction,send->round,hop,0,offset,
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
    status=dependency_order(f,pieces,count,page);
  }
  if(!status){
    size_t at=0;
    for(uint32_t r=0;r<N;r++){
      struct mesh_routed_rank *rank=made.rank+r;
      rank->transfer=calloc((size_t)rank->count+1,sizeof *rank->transfer);
      rank->consume=calloc((size_t)rank->count+1,sizeof *rank->consume);
      rank->views=calloc((size_t)rank->count+1,sizeof *rank->views);
      if(!rank->transfer || !rank->consume || !rank->views){status=ENOMEM;break;}
      for(uint32_t k=0;k<rank->count;k++,at++){
        rank->transfer[k]=pieces[at].transfer;rank->consume[k]=pieces[at].consume;rank->views[k]=pieces[at].view;
      }
    }
  }
  free(pieces);free(matched);free(relay);
  made.domains=coordinates==MESH_PACKET_OPERAND?!!transaction:transaction;
  if(status)mesh_routed_free(&made);else *out=made;
  return status;
}
