/* Common exact-DMA cuts for affine rows. One compiled table covers the whole
   logical payload; row boundaries are ordinary cuts, not transport calls. */
#ifndef MESH_ROW_SPAN_PLAN_H
#define MESH_ROW_SPAN_PLAN_H
#include "mesh-span-plan.h"

/* Rank p maps packed byte k to (k / row_bytes)*pitches[p] + k%row_bytes,
   relative to its original buffer. phases[p] is that buffer's arena offset
   modulo MESH_DMA_EXTENT. Per-row physical phases can differ across ranks.

   Compile each row at its ACTUAL phase, not the union of every row's phase:
   imposing one common shape on all rows can turn a long odd-pitch tensor into
   byte-sized packets. The output is one flat cuts[] in packed logical bytes;
   every row boundary appears, and unit counts per row may differ. Every
   participant uses the same sequence. The packet limit applies within rows.

   Positive errno on failure, leaving the initially empty out unchanged.
   Empty payload (zero rows or zero row_bytes) has the single endpoint 0.
   This helper proves packet boundaries and arithmetic representability, not
   allocation ownership or disjoint destination writes; the binding layer
   checks those against the actual buffers. Pitches may describe overlapping
   reads, so no blanket pitch >= row_bytes restriction is imposed. */
static inline int mesh_row_span_plan_create(const uint64_t *phases,const uint64_t *pitches,uint32_t participants,
                                            uint64_t row_bytes,uint64_t rows,uint32_t packet,
                                            struct mesh_span_plan *out){
  if(!out || out->cuts || out->count || !phases || !pitches || !participants || !packet)return EINVAL;
  for(uint32_t p=0;p<participants;p++){
    if(phases[p]>=MESH_DMA_EXTENT)return EINVAL;
    if(rows && row_bytes && pitches[p] && rows-1>(UINT64_MAX-row_bytes)/pitches[p])return EOVERFLOW;
  }
  if(rows && row_bytes>UINT64_MAX/rows)return EOVERFLOW;
  const size_t maximum=SIZE_MAX/sizeof(uint64_t);
  if(!rows || !row_bytes){
    uint64_t *cuts=malloc(sizeof *cuts);
    if(!cuts)return ENOMEM;
    cuts[0]=0;*out=(struct mesh_span_plan){cuts,1};return 0;
  }
  const uint64_t least=(row_bytes-1)/packet+1;
  if(rows>(maximum-1)/least || participants>maximum)return EOVERFLOW;
  size_t room=(size_t)(rows*least)+1,count=1;
  uint64_t *cuts=malloc(room*sizeof *cuts),*phase=malloc((size_t)participants*sizeof *phase);
  if(!cuts || !phase){free(cuts);free(phase);return ENOMEM;}
  cuts[0]=0;
  for(uint32_t p=0;p<participants;p++)phase[p]=phases[p];
  int error=0;
  for(uint64_t row=0;row<rows;row++){
    struct mesh_span_plan current={0};
    error=mesh_span_plan_create(phase,pitches,participants,row_bytes,1,packet,&current);
    if(error)break;
    const size_t units=current.count-1;
    if(units>maximum-count){mesh_span_plan_free(&current);error=EOVERFLOW;break;}
    const size_t needed=count+units;
    if(needed>room){
      size_t grown=room<=maximum/2?room*2:maximum;
      if(grown<needed)grown=needed;
      uint64_t *more=realloc(cuts,grown*sizeof *more);
      if(!more){mesh_span_plan_free(&current);error=ENOMEM;break;}
      cuts=more;room=grown;
    }
    const uint64_t base=row*row_bytes;
    for(size_t k=1;k<current.count;k++)cuts[count++]=base+current.cuts[k];
    mesh_span_plan_free(&current);
    for(uint32_t p=0;p<participants;p++)phase[p]=(phase[p]+pitches[p]%MESH_DMA_EXTENT)%MESH_DMA_EXTENT;
  }
  free(phase);
  if(error){free(cuts);return error;}
  *out=(struct mesh_span_plan){cuts,count};return 0;
}
#endif
