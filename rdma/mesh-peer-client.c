#include "mesh-peer.h"
#include <stdlib.h>
#include <string.h>

/* design/node-addressed-transfers.md: the peer service over the node's order, all of it the client's: no message to
   any bridge, no installation agreed.  An installation is the order it was appended to and a lease of the records'
   gates (each a mesh_gate the order reads) */
struct installation { struct mesh_order order; struct mesh_section gates; };

int mesh_peer_capable(struct mesh_ctx *ctx,uint32_t peer){
  struct mesh_order order;
  return mesh_order_open(ctx,peer,&order);
}
static int match_order(const void *a,const void *b){
  const struct mesh_peer_transfer *x=*(const struct mesh_peer_transfer *const *)a,*y=*(const struct mesh_peer_transfer *const *)b;
  return (x->match>y->match)-(x->match<y->match);
}
int mesh_peer_install(struct mesh_ctx *ctx,uint32_t peer,uint64_t group,uint64_t records,uint32_t count,uint64_t *installation){
  (void)group;
  if(!count || !installation || !peer || !ctx || !ctx->M || records<ctx->M->data_off || records%8 ||
     records>ctx->M->length || count>(ctx->M->length-records)/sizeof(struct mesh_peer_transfer))return EINVAL;
  struct installation *made=calloc(1,sizeof *made);
  const struct mesh_peer_transfer **sorted=calloc(count,sizeof *sorted);
  int error=!made || !sorted?ENOMEM:mesh_order_open(ctx,peer,&made->order);
  const struct mesh_peer_transfer *record=(const struct mesh_peer_transfer *)((const char *)ctx->M+records);
  for(uint32_t i=0;i<count;i++)if(!error && record[i].land>1)error=EINVAL;
  if(!error)error=mesh_section_create(ctx,count*sizeof(struct mesh_gate),1,&made->gates);
  if(!error){
    for(uint32_t i=0;i<count;i++)sorted[i]=record+i;
    qsort(sorted,count,sizeof *sorted,match_order);
    struct mesh_gate *gates=mesh_section_address(ctx,made->gates,0);
    const uint64_t at=(uint64_t)((char *)gates-(char *)ctx->M);
    for(uint32_t i=0;i<count && !error;i++){
      const struct mesh_peer_transfer *r=sorted[i];
      struct mesh_crossing t={.spans=r->spans,.units=r->units,.done=r->completed,.value=r->value};
      if(r->reserved){t.gate=r->ready;t.gates=r->reserved;}
      else if(r->ready){gates[i]=(struct mesh_gate){r->ready,r->threshold};t.gate=at+i*sizeof *gates;t.gates=1;}
      while((error=mesh_order_append(ctx,&made->order,(int)r->land,&t))==EAGAIN){}
    }
  }
  free(sorted);
  if(error){if(made && made->gates.count)mesh_lease_return(ctx,made->gates.first);free(made);return error;}
  *installation=(uint64_t)(uintptr_t)made;
  return 0;
}
int mesh_peer_wait(struct mesh_ctx *ctx,uint64_t installation,uint64_t timeout_ns){
  (void)timeout_ns;
  const struct installation *i=(const struct installation *)(uintptr_t)installation;
  return !i?EINVAL:mesh_order_live(ctx,&i->order)?0:ENOLINK;
}
int mesh_peer_release(struct mesh_ctx *ctx,uint64_t installation){
  struct installation *i=(struct installation *)(uintptr_t)installation;
  if(!ctx || !i)return EINVAL;
  const int error=mesh_lease_return(ctx,i->gates.first);
  free(i);
  return error;
}
