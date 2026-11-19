#include "../mesh-peer-plan.h"
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>

static void inspect(int scatter,uint32_t forward_limit){
  const uint64_t start=16384,bytes=65539;
  struct mesh_endpoint left={18,100,2,0,1,0},right={18,100,1,0,1,0};
  struct peer_limit ll={{32768,8192}},rl={{16384,forward_limit}};
  struct mesh_peer_transfer tx={.land=0},rx={.land=1};
  struct mesh_dma_span source={start+100,bytes},dest[3]={{start+500,bytes}};
  if(scatter){
    dest[0]=(struct mesh_dma_span){start+MESH_DMA_EXTENT-3,17001};
    dest[1]=(struct mesh_dma_span){start+99881,22998};
    dest[2]=(struct mesh_dma_span){start+453097,25540};
  }
  uint32_t units=scatter?3:1;struct mesh_dma_span remote_source={100,bytes},remote_dest[3];
  for(uint32_t k=0;k<units;k++)remote_dest[k]=(struct mesh_dma_span){(dest[k].offset-start)%MESH_DMA_EXTENT,dest[k].bytes};
  struct peer_record arow={.match=17,.bytes=bytes,.units=1,.source=1};
  struct peer_record brow={.match=17,.bytes=bytes,.units=units,.source=1};
  struct peer_layout a={{1,1,1,0},&left,&ll,&arow,&source},
                     b={{1,1,units,0},&right,&rl,&brow,remote_dest};
  struct peer_plan aplan={0},bplan={0};
  assert(!peer_plan_compile(&aplan,1,2,a,b,&tx,start,4096));
  b.spans=dest;a.spans=&remote_source;
  assert(!peer_plan_compile(&bplan,2,1,b,a,&rx,start,4096));
  assert(aplan.packet[0][0]==forward_limit && bplan.packet[0][1]==forward_limit);
  assert(aplan.packet[0][1]==8192 && bplan.packet[0][0]==8192 && aplan.count==bplan.count);
  uint64_t total=0,largest=0;
  for(size_t i=0;i<aplan.count;i++){
    struct mesh_dma_span x=aplan.piece[i].span,y=bplan.piece[i].span;
    assert(x.bytes>0 && x.bytes<=forward_limit && x.bytes==y.bytes);
    assert(x.bytes<=MESH_DMA_EXTENT-(x.offset-start)%MESH_DMA_EXTENT && y.bytes<=MESH_DMA_EXTENT-(y.offset-start)%MESH_DMA_EXTENT);
    total+=x.bytes;if(x.bytes>largest)largest=x.bytes;
  }
  assert(total==bytes && largest==forward_limit);
  if(!scatter){assert(aplan.count==(bytes+forward_limit-1)/forward_limit);assert(aplan.piece[aplan.count-1].span.bytes==bytes%forward_limit);}
  else assert(aplan.piece[0].span.bytes==3);
  printf("%u byte limit, %" PRIu64 " payload bytes, %zu requests, final %" PRIu64 " bytes\n",
         forward_limit,total,aplan.count,aplan.piece[aplan.count-1].span.bytes);
  free(aplan.row);free(aplan.piece);free(bplan.row);free(bplan.piece);
}
int main(void){inspect(0,12288);inspect(1,12288);inspect(0,4096);return 0;}
