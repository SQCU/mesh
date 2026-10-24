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
                       const double *weights,struct mesh_transfer *out,uint32_t room){
  if(!f || !step || rank>=f->nodes || step->peer>=f->nodes || !step->piece.element_bytes)return -EINVAL;
  if(!step->piece.elements)return 0;
  uint64_t a=page? page:1,b=step->piece.element_bytes;
  while(b){const uint64_t r=a%b;a=b;b=r;}
  const uint64_t quantum=(page? page:1)/a,units=(step->piece.elements-1)/quantum+1;
  uint32_t n=0;
  double total=0;
  for(uint32_t e=0;e<f->cables;e++){
    const struct mesh_cable *c=f->cable+e;
    if(!((c->node[0]==rank && c->node[1]==step->peer) || (c->node[1]==rank && c->node[0]==step->peer)))continue;
    const double w=weights? weights[e]:(c->bandwidth? (double)c->bandwidth:1);
    if(!(w>0) || !isfinite(w))return -EINVAL;
    total+=w;n++;
  }
  if(!n)return -ENOLINK;
  if(n>room)return -ENOSPC;
  uint64_t *parts=calloc(n,sizeof *parts);
  double *remainder=calloc(n,sizeof *remainder);
  if(!parts || !remainder){free(parts);free(remainder);return -ENOMEM;}
  const uint64_t floor=units>=n,remaining=units-floor*n;
  uint64_t assigned=0;
  for(uint32_t e=0,k=0;e<f->cables;e++){
    const struct mesh_cable *c=f->cable+e;
    if(!((c->node[0]==rank && c->node[1]==step->peer) || (c->node[1]==rank && c->node[0]==step->peer)))continue;
    const double w=weights? weights[e]:(c->bandwidth? (double)c->bandwidth:1);
    const double ideal=(double)remaining*w/total;
    parts[k]=floor+(uint64_t)ideal;remainder[k]=ideal-(double)(uint64_t)ideal;
    assigned+=parts[k];out[k].cable=e;k++;
  }
  while(assigned<units){
    uint32_t best=0;
    for(uint32_t k=1;k<n;k++)if(remainder[k]>remainder[best])best=k;
    parts[best]++;remainder[best]=-1;assigned++;
  }
  uint64_t at=0;
  uint32_t written=0;
  for(uint32_t k=0;k<n;k++){
    if(!parts[k])continue;
    const uint64_t available=step->piece.elements-at;
    const uint64_t count=parts[k]>available/quantum? available:parts[k]*quantum;
    const uint32_t cable=out[k].cable;
    out[written]=(struct mesh_transfer){.step=*step,.cable=cable};
    out[written].step.first+=at;out[written].step.piece.elements=count;
    at+=count;written++;
  }
  free(parts);free(remainder);
  return (int)written;
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
    uint64_t a=page,b=s->piece.element_bytes;
    while(b){const uint64_t r=a%b;a=b;b=r;}
    const uint64_t quantum=page/a;
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
     !paths->first || !paths->cable || !paths->log_weight || paths->first[0])return EINVAL;
  unsigned char *seen=calloc(f->nodes,1);
  if(!seen)return ENOMEM;
  int status=0;
  for(uint32_t p=0;!status && p<paths->count;p++){
    memset(seen,0,f->nodes);
    uint32_t at=source;
    seen[at]=1;
    if(paths->first[p+1]<=paths->first[p] || paths->first[p+1]-paths->first[p]>=f->nodes || !isfinite(paths->log_weight[p])){
      status=EINVAL;break;
    }
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
  return status;
}

int mesh_write_compile(const struct mesh_fabric *f,uint32_t rank,uint32_t source,uint32_t destination,
                        const struct mesh_paths *paths,struct mesh_operand operand,uint64_t page,
                        struct mesh_transfer *out,uint32_t room){
  const int check=mesh_paths_check(f,source,destination,paths);
  if(check || rank>=f->nodes || !operand.element_bytes || !page)return -(check?check:EINVAL);
  if(!operand.elements)return 0;
  const uint32_t P=paths->count;
  uint64_t a=page,b=operand.element_bytes;
  while(b){const uint64_t r=a%b;a=b;b=r;}
  const uint64_t q=page/a,units=(operand.elements-1)/q+1;
  uint64_t *count=calloc(P,sizeof *count),*base=calloc(P,sizeof *base);
  double *remainder=calloc(P,sizeof *remainder);
  if(!count || !base || !remainder){free(count);free(base);free(remainder);return -ENOMEM;}
  double maximum=paths->log_weight[0],total=0;
  for(uint32_t p=1;p<P;p++)if(paths->log_weight[p]>maximum)maximum=paths->log_weight[p];
  for(uint32_t p=0;p<P;p++)total+=exp(paths->log_weight[p]-maximum);
  uint64_t assigned=0;
  for(uint32_t p=0;p<P;p++){
    /* Weights describe total flow, not surplus after a page on every path.
       A sub-page path can round to zero; cable coverage belongs to the joint
       flow allocation, not to an extra relay page in every commodity. */
    const double real=(double)units*exp(paths->log_weight[p]-maximum)/total;
    count[p]=(uint64_t)real;remainder[p]=real-(uint64_t)real;assigned+=count[p];
  }
  while(assigned<units){
    uint32_t best=0;
    for(uint32_t p=1;p<P;p++)if(remainder[p]>remainder[best])best=p;
    count[best]++;remainder[best]=-1;assigned++;
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
        if(out && written<room)out[written]=(struct mesh_transfer){
          .step={rank==at?MESH_STEP_SEND:MESH_STEP_COPY,rank==at?next:at,
                 k-paths->first[p],0,lo,{operand.type,operand.element_bytes,amount}},.cable=e};
        written++;
      }
      at=next;
    }
  }
  free(count);free(base);free(remainder);
  return out && written>room?-ENOSPC:(int)written;
}
