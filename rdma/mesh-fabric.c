#include "mesh-plan.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* design/whole-function.md#fabric */
int mesh_fabric_check(const struct mesh_fabric *f){
  if(!f || !f->nodes || (f->cables && !f->cable))return EINVAL;
  for(uint32_t e=0;e<f->cables;e++){
    const struct mesh_cable *c=f->cable+e;
    if(!c->wire || c->node[0]>=f->nodes || c->node[1]>=f->nodes || c->node[0]==c->node[1])return EINVAL;
    for(uint32_t j=0;j<e;j++)if(f->cable[j].wire==c->wire)return EINVAL;
  }
  return 0;
}

int mesh_fabric_stripe(const struct mesh_fabric *f,uint32_t rank,const struct mesh_step *step,uint64_t page,
                       const uint64_t *weights,struct mesh_transfer *out,uint32_t room){
  if(mesh_fabric_check(f) || !step || rank>=f->nodes || step->peer>=f->nodes || rank==step->peer ||
     step->op>MESH_STEP_COPY || !step->piece.element_bytes)return -EINVAL;
  if(step->piece.elements>UINT64_MAX-step->first)return -EOVERFLOW;
  if(!step->piece.elements)return 0;
  uint32_t n=0;
  for(uint32_t e=0;e<f->cables;e++){
    const struct mesh_cable *c=f->cable+e;
    if(!((c->node[0]==rank && c->node[1]==step->peer) || (c->node[1]==rank && c->node[0]==step->peer)))continue;
    const uint64_t w=weights? weights[e]:(c->bandwidth? c->bandwidth:1);
    (void)w;
    n++;
  }
  if(!n)return -ENOLINK;
  uint32_t *first=calloc((size_t)n+1,sizeof *first),*cable=calloc(n,sizeof *cable);
  uint64_t *logs=calloc(n,sizeof *logs);
  if(!first || !cable || !logs){free(first);free(cable);free(logs);return -ENOMEM;}
  /* A logical collective edge supplies a direct-path configuration to the
     same compiler as an explicitly routed WRITE. There is no second flow
     allocator, compulsory per-cable page, or first-cable selection here.
     Relay participation is declared by the enclosing logical dataflow; the
     WRITE interface also admits relay paths inside an individual transaction. */
  for(uint32_t e=0,k=0;e<f->cables;e++){
    const struct mesh_cable *c=f->cable+e;
    if(!((c->node[0]==rank && c->node[1]==step->peer) || (c->node[1]==rank && c->node[0]==step->peer)))continue;
    const uint64_t w=weights? weights[e]:(c->bandwidth? c->bandwidth:1);
    first[k]=k;cable[k]=e;logs[k]=w;k++;
  }
  first[n]=n;
  const struct mesh_paths paths={n,first,cable,logs};
  const uint32_t source=step->op==MESH_STEP_SEND?rank:step->peer;
  const uint32_t destination=step->op==MESH_STEP_SEND?step->peer:rank;
  const int written=mesh_write_compile(f,rank,source,destination,&paths,step->piece,page?page:1,0,out,room);
  for(int k=0;out && k<written;k++){
    const uint64_t offset=out[k].step.first,elements=out[k].step.piece.elements;
    out[k].step=*step;out[k].step.first+=offset;out[k].step.piece.elements=elements;
  }
  free(first);free(cable);free(logs);
  return written;
}

static int mesh_cut_order(const void *a,const void *b){
  const uint64_t x=*(const uint64_t *)a,y=*(const uint64_t *)b;
  return (x>y)-(x<y);
}

int mesh_fabric_refine(const struct mesh_transfer *in,uint32_t count,uint64_t page,
                       struct mesh_transfer *out,uint32_t room){
  if((count && !in) || !page || (out && out==in))return -EINVAL;
  if(count>INT_MAX)return -EOVERFLOW;
  if((uint64_t)count*2+2>SIZE_MAX/sizeof(uint64_t))return -EOVERFLOW;
  for(uint32_t k=0;k<count;k++){
    const struct mesh_step *s=&in[k].step;
    if(s->op>MESH_STEP_COPY || !s->piece.element_bytes)return -EINVAL;
    if(s->piece.elements>UINT64_MAX-s->first ||
       s->first+s->piece.elements>UINT64_MAX/s->piece.element_bytes)return -EOVERFLOW;
  }
  uint64_t *cuts=malloc(((size_t)count*2+2)*sizeof *cuts);
  if(!cuts)return -ENOMEM;
  uint64_t written=0;
  int error=0;
  for(uint32_t k=0;k<count && !error;k++){
    if(written>=INT_MAX){error=-EOVERFLOW;break;}
    const struct mesh_step *s=&in[k].step;
    if(s->op!=MESH_STEP_SEND || !s->piece.elements){
      if(out && written<room)out[written]=in[k];
      written++;
      continue;
    }
    const uint64_t end=s->first+s->piece.elements;
    size_t n=0;
    cuts[n++]=s->first;cuts[n++]=end;
    for(uint32_t j=0;j<k;j++){
      const struct mesh_step *a=&in[j].step;
      if(a->op==MESH_STEP_SEND || a->buffer!=s->buffer || a->piece.element_bytes!=s->piece.element_bytes ||
         !a->piece.elements || a->first>=end || a->first+a->piece.elements<=s->first)continue;
      if(a->first>s->first)cuts[n++]=a->first;
      if(a->first+a->piece.elements<end)cuts[n++]=a->first+a->piece.elements;
    }
    qsort(cuts,n,sizeof *cuts,mesh_cut_order);
    const uint64_t quantum=mesh_page_quantum(page,s->piece.element_bytes);
    uint64_t first=s->first;
    int64_t owner=-2;
    for(size_t i=0;i+1<n;i++){
      if(cuts[i]==cuts[i+1])continue;
      int64_t next=-1;
      for(uint32_t j=k;j>0;){
        const struct mesh_step *prior=&in[--j].step;
        if(prior->op==MESH_STEP_SEND || prior->buffer!=s->buffer ||
           prior->piece.element_bytes!=s->piece.element_bytes || prior->first>cuts[i] ||
           prior->first+prior->piece.elements<=cuts[i])continue;
        if(prior->op==MESH_STEP_COPY)next=j;
        break;
      }
      if(next>=0 && next!=owner && (cuts[i]-in[next].step.first)%quantum){error=-EINVAL;break;}
      if(owner!=-2 && next!=owner){
        if((cuts[i]-s->first)%quantum){error=-EINVAL;break;}
        if(out && written<room){
          out[written]=in[k];out[written].step.first=first;
          out[written].step.piece.elements=cuts[i]-first;
        }
        written++;first=cuts[i];
      }
      owner=next;
    }
    if(!error){
      if(out && written<room){
        out[written]=in[k];out[written].step.first=first;
        out[written].step.piece.elements=end-first;
      }
      written++;
    }
    if(written>INT_MAX)error=-EOVERFLOW;
  }
  free(cuts);
  return error?error:out && written>room?-ENOSPC:(int)written;
}

/* design/whole-function.md#one-way-writes */
int mesh_paths_check(const struct mesh_fabric *f,uint32_t source,uint32_t destination,const struct mesh_paths *paths){
  if(mesh_fabric_check(f) || source>=f->nodes || destination>=f->nodes || source==destination || !paths || !paths->count ||
     !paths->first || !paths->cable || !paths->weight || paths->first[0])return EINVAL;
  unsigned char *seen=calloc(f->nodes,1);
  if(!seen)return ENOMEM;
  int status=0;
  uint64_t total=0;
  for(uint32_t p=0;!status && p<paths->count;p++){
    memset(seen,0,f->nodes);
    uint32_t at=source;
    seen[at]=1;
    if(paths->first[p+1]<=paths->first[p] || paths->first[p+1]-paths->first[p]>=f->nodes){
      status=EINVAL;break;
    }
    if(paths->weight[p]>UINT64_MAX-total){status=EOVERFLOW;break;}
    total+=paths->weight[p];
    for(uint32_t k=paths->first[p];k<paths->first[p+1];k++){
      const uint32_t e=paths->cable[k];
      if(e>=f->cables){status=EINVAL;break;}
      const struct mesh_cable *c=f->cable+e;
      if(c->node[0]!=at && c->node[1]!=at){status=EINVAL;break;}
      at=c->node[c->node[0]==at];
      if(seen[at]){status=EINVAL;break;}
      seen[at]=1;
    }
    if(at!=destination)status=EINVAL;
  }
  free(seen);
  return status?status:total?0:EINVAL;
}

int mesh_write_compile(const struct mesh_fabric *f,uint32_t rank,uint32_t source,uint32_t destination,
                        const struct mesh_paths *paths,struct mesh_operand operand,uint64_t page,uint64_t tile,
                        struct mesh_transfer *out,uint32_t room){
  const int check=mesh_paths_check(f,source,destination,paths);
  if(check || rank>=f->nodes || !operand.element_bytes || !page)return -(check?check:EINVAL);
  if(!operand.elements)return 0;
  const uint32_t P=paths->count;
  const uint64_t q=mesh_page_quantum(page,operand.element_bytes),units=(operand.elements-1)/q+1;
  uint64_t *count=calloc(P,sizeof *count),*base=calloc(P,sizeof *base);
  uint64_t *remainder=calloc(P,sizeof *remainder);
  if(!count || !base || !remainder){free(count);free(base);free(remainder);return -ENOMEM;}
  uint64_t total=0;
  for(uint32_t p=0;p<P;p++)total+=paths->weight[p];
  uint64_t assigned=0;
  for(uint32_t p=0;p<P;p++){
    /* Weights describe total flow, not surplus after a page on every path.
       A sub-page path can round to zero; cable coverage belongs to the joint
       flow allocation, not to an extra relay page in every commodity. */
    const __uint128_t product=(__uint128_t)units*paths->weight[p];
    count[p]=(uint64_t)(product/total);remainder[p]=(uint64_t)(product%total);assigned+=count[p];
  }
  while(assigned<units){
    uint32_t best=0;
    for(uint32_t p=1;p<P;p++)if(remainder[p]>remainder[best])best=p;
    count[best]++;remainder[best]=0;assigned++;
  }
  for(uint32_t p=1;p<P;p++)base[p]=base[p-1]+count[p-1];
  uint32_t written=0;
  for(uint32_t p=0;p<P;p++){
    if(!count[p])continue;
    const uint64_t lo=base[p]*q;
    const uint64_t amount=count[p]>(operand.elements-lo)/q?operand.elements-lo:count[p]*q;
    uint32_t at=source;
    for(uint32_t k=paths->first[p];k<paths->first[p+1];k++){
      const uint32_t e=paths->cable[k],next=f->cable[e].node[f->cable[e].node[0]==at];
      if(rank==at || rank==next){
        const uint64_t fragments=tile?(lo+amount-1)/tile-lo/tile+1:1;
        if(fragments>INT_MAX-written){free(count);free(base);free(remainder);return -EOVERFLOW;}
        if(!out)written+=(uint32_t)fragments;
        else for(uint64_t first=lo,left=amount;left;){
          uint64_t n=tile?tile-first%tile:left;
          if(n>left)n=left;
          if(out && written<room)out[written]=(struct mesh_transfer){
            .step={rank==at?MESH_STEP_SEND:MESH_STEP_COPY,rank==at?next:at,
                   k-paths->first[p],0,first,{operand.type,operand.element_bytes,n}},.cable=e};
          written++;first+=n;left-=n;
        }
      }
      at=next;
    }
  }
  free(count);free(base);free(remainder);
  return out && written>room?-ENOSPC:(int)written;
}
