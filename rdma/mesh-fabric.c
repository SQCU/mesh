#include "mesh-plan.h"
#include <errno.h>
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
    break;
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
    break;
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
                        const struct mesh_paths *paths,struct mesh_operand operand,uint64_t page,uint32_t chunk_pages,
                        struct mesh_transfer *out,uint32_t room){
  const int check=mesh_paths_check(f,source,destination,paths);
  if(check || rank>=f->nodes || !operand.element_bytes || !page || !chunk_pages)return -(check?check:EINVAL);
  if(!operand.elements)return 0;
  const uint32_t P=paths->count;
  uint64_t a=page,b=operand.element_bytes;
  while(b){const uint64_t r=a%b;a=b;b=r;}
  const uint64_t q=page/a,units=(operand.elements-1)/q+1,floor=units>=P;
  uint64_t *count=calloc(P,sizeof *count),*base=calloc(P,sizeof *base);
  double *remainder=calloc(P,sizeof *remainder);
  if(!count || !base || !remainder){free(count);free(base);free(remainder);return -ENOMEM;}
  double maximum=paths->log_weight[0],total=0;
  for(uint32_t p=1;p<P;p++)if(paths->log_weight[p]>maximum)maximum=paths->log_weight[p];
  for(uint32_t p=0;p<P;p++)total+=exp(paths->log_weight[p]-maximum);
  uint64_t assigned=0,chunks=0;
  for(uint32_t p=0;p<P;p++){
    const double real=(double)(units-floor*P)*exp(paths->log_weight[p]-maximum)/total;
    count[p]=floor+(uint64_t)real;remainder[p]=real-(uint64_t)real;assigned+=count[p];
  }
  while(assigned<units){
    uint32_t best=0;
    for(uint32_t p=1;p<P;p++)if(remainder[p]>remainder[best])best=p;
    count[best]++;remainder[best]=-1;assigned++;
  }
  for(uint32_t p=0;p<P;p++){
    if(p)base[p]=base[p-1]+count[p-1];
    const uint64_t c=(count[p]+chunk_pages-1)/chunk_pages;
    if(c>chunks)chunks=c;
  }
  uint32_t written=0;
  for(uint64_t chunk=0;chunk<chunks;chunk++)for(uint32_t p=0;p<P;p++){
    const uint64_t unit=chunk*chunk_pages;
    if(unit>=count[p])continue;
    const uint64_t lo=(base[p]+unit)*q,part=count[p]-unit<chunk_pages?count[p]-unit:chunk_pages;
    const uint64_t amount=part> (operand.elements-lo)/q?operand.elements-lo:part*q;
    uint32_t at=source;
    for(uint32_t k=paths->first[p];k<paths->first[p+1];k++){
      const uint32_t e=paths->cable[k],next=f->cable[e].node[f->cable[e].node[0]==at];
      if(rank==at || rank==next){
        if(out && written<room)out[written]=(struct mesh_transfer){
          .step={rank==at?MESH_STEP_SEND:MESH_STEP_COPY,rank==at?next:at,
                 (uint32_t)(chunk*f->nodes+k-paths->first[p]),0,lo,{operand.type,operand.element_bytes,amount}},.cable=e};
        written++;
      }
      at=next;
    }
  }
  free(count);free(base);free(remainder);
  return out && written>room?-ENOSPC:(int)written;
}
