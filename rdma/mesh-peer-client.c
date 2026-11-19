#include "mesh-peer.h"
#include "mesh-order.h"
#include <stdlib.h>
#include <string.h>

/* design/declared-programs.md: the peer service over declared programs, all of it the client's: no message to any
   bridge, no installation agreed.  An installation is its declared program and the lease of its tables */
struct installation { struct mesh_run run; struct mesh_section tables; };

int mesh_peer_capable(struct mesh_ctx *ctx,uint32_t peer){
  struct mesh_order order;
  return mesh_order_open(ctx,peer,&order);
}
static int match_order(const void *a,const void *b){
  const struct mesh_peer_transfer *x=*(const struct mesh_peer_transfer *const *)a,*y=*(const struct mesh_peer_transfer *const *)b;
  return (x->match>y->match)-(x->match<y->match);
}
/* the pieces of a record's spans (each cut into pieces of at most `piece` bytes) */
static uint64_t record_pieces(const struct hdr *m,const struct mesh_peer_transfer *r,uint64_t piece){
  const struct mesh_dma_span *spans=(const struct mesh_dma_span *)((const char *)m+r->spans);
  uint64_t n=0;
  for(uint64_t k=0;k<r->units;k++)n+=(spans[k].bytes+piece-1)/piece;
  return n;
}
int mesh_peer_install(struct mesh_ctx *ctx,uint32_t peer,uint64_t group,uint64_t records,uint32_t count,uint64_t *installation){
  if(!count || !installation || !peer || !ctx || !ctx->M || records<ctx->M->data_off || records%8 ||
     records>ctx->M->length || count>(ctx->M->length-records)/sizeof(struct mesh_peer_transfer))return EINVAL;
  struct hdr *m=ctx->M;
  const uint64_t piece=order_piece_bytes(m);
  const struct mesh_peer_transfer *record=(const struct mesh_peer_transfer *)((const char *)m+records);
  uint64_t transfers[2]={0},pieces[2]={0};
  for(uint32_t i=0;i<count;i++){
    const struct mesh_peer_transfer *r=record+i;
    if(r->land>1 || r->reserved || !r->units || !mesh_region_holds(m,r->spans,r->units*sizeof(struct mesh_dma_span),8))return EINVAL;
    transfers[r->land]++;pieces[r->land]+=record_pieces(m,r,piece);
  }
  struct installation *made=calloc(1,sizeof *made);
  const struct mesh_peer_transfer **sorted=calloc(count,sizeof *sorted);
  struct mesh_order order;
  int error=!made || !sorted?ENOMEM:mesh_order_open(ctx,peer,&order);
  const uint64_t table_bytes=(transfers[0]+transfers[1])*sizeof(struct mesh_crossing);
  if(!error)error=mesh_section_create(ctx,table_bytes+(pieces[0]+pieces[1])*sizeof(struct mesh_dma_span),1,&made->tables);
  if(!error){
    for(uint32_t i=0;i<count;i++)sorted[i]=record+i;
    qsort(sorted,count,sizeof *sorted,match_order);
    char *base=mesh_section_address(ctx,made->tables,0);
    const uint64_t at=(uint64_t)(base-(char *)m);
    struct mesh_crossing *table[2]={(struct mesh_crossing *)base,(struct mesh_crossing *)base+transfers[0]};
    struct mesh_dma_span *span=(struct mesh_dma_span *)(base+table_bytes);
    struct mesh_declaration program={.key=group,.period=1,.count=1,
      .table={transfers[0]?at:0,transfers[1]?at+transfers[0]*sizeof(struct mesh_crossing):0},
      .transfers={transfers[0],transfers[1]},.pieces={pieces[0],pieces[1]}};
    uint64_t filled[2]={0},first[2]={0};
    for(uint32_t i=0;i<count;i++){
      const struct mesh_peer_transfer *r=sorted[i];
      const struct mesh_dma_span *spans=(const struct mesh_dma_span *)((const char *)m+r->spans);
      struct mesh_crossing *t=table[r->land]+filled[r->land]++;
      *t=(struct mesh_crossing){.spans=(uint64_t)((char *)span-(char *)m),.first=first[r->land],.flags=MESH_GATE_WHOLE,
        .gate=r->ready,.gate_base=r->threshold,.done=r->completed,.done_value=r->value};
      for(uint64_t k=0;k<r->units;k++)
        for(uint64_t from=0;from<spans[k].bytes;from+=piece){
          const uint64_t rest=spans[k].bytes-from;
          *span++=(struct mesh_dma_span){spans[k].offset+from,rest<piece?rest:piece};
          t->units++;
        }
      first[r->land]+=t->units;
    }
    error=mesh_declare(ctx,&order,&program,&made->run);
    if(!error)mesh_issue(&made->run,1);
  }
  free(sorted);
  if(error){if(made && made->tables.count)mesh_lease_return(ctx,made->tables.first);free(made);return error;}
  *installation=(uint64_t)(uintptr_t)made;
  return 0;
}
int mesh_peer_wait(struct mesh_ctx *ctx,uint64_t installation,uint64_t timeout_ns){
  (void)timeout_ns;
  const struct installation *i=(const struct installation *)(uintptr_t)installation;
  return !i?EINVAL:mesh_order_live(ctx,&i->run.order)?0:ENOLINK;
}
int mesh_peer_release(struct mesh_ctx *ctx,uint64_t installation){
  struct installation *i=(struct installation *)(uintptr_t)installation;
  if(!ctx || !i)return EINVAL;
  const int closed=mesh_order_live(ctx,&i->run.order)?mesh_close(ctx,&i->run):0;
  const int error=mesh_lease_return(ctx,i->tables.first);
  free(i);
  return closed?closed:error;
}
