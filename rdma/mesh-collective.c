#include "mesh-collective.h"
#include <stdio.h>
#include <string.h>

/* The link map is configuration, read as written: the MLX JACCL hostfile idea
   (github.com/ml-explore/mlx docs/src/usage/distributed.rst) reduced to a kind, a node count and links. */
int mesh_link_map_read(const char *path,struct mesh_link_map *map){
  static const char *const kinds[]={"mesh","ring","tree"};
  FILE *file=fopen(path,"r");
  if(!file)return errno;
  char line[128],kind[8]="";
  uint32_t a,b;
  *map=(struct mesh_link_map){.kind=UINT32_MAX};
  while(fgets(line,sizeof line,file)){
    if(sscanf(line,"%u %u",&a,&b)==2 && map->links<MESH_LINK_MAP_NODES*(MESH_LINK_MAP_NODES-1)/2){
      map->link[map->links][0]=a;map->link[map->links++][1]=b;
    }else if(map->kind==UINT32_MAX && sscanf(line,"%7s %u",kind,&a)==2)
      for(uint32_t k=0;k<3;k++)if(!strcmp(kind,kinds[k])){map->kind=k;map->nodes=a;}
  }
  fclose(file);
  /* Refuse a map its algorithm cannot run, instead of indexing past the planner's arrays or
     walking a broken ring forever: every node < nodes and at most one link into it; a ring is one
     cycle through every node; a tree has nodes-1 links and every node reaches the root. */
  uint32_t n=map->nodes,up[MESH_LINK_MAP_NODES],at,steps;
  if(map->kind==UINT32_MAX || !n || n>MESH_LINK_MAP_NODES)return EINVAL;
  if(map->kind==MESH_LINKS_MESH)return 0;
  for(uint32_t v=0;v<n;v++)up[v]=n;
  for(uint32_t l=0;l<map->links;l++){
    if(map->link[l][0]>=n || map->link[l][1]>=n || up[map->link[l][1]]<n)return EINVAL;
    up[map->link[l][1]]=map->link[l][0];
  }
  if(map->kind==MESH_LINKS_RING){
    at=0;steps=0;
    do{at=up[at];steps++;}while(at<n && at && steps<n);
    return at==0 && steps==n?0:EINVAL;
  }
  if(map->links!=n-1)return EINVAL;
  for(uint32_t v=0;v<n;v++){
    for(at=v,steps=0;up[at]<n && steps<n;steps++)at=up[at];
    if(up[at]<n)return EINVAL;
  }
  return 0;
}

static struct mesh_step mesh_piece(uint32_t op,uint32_t peer,uint32_t round,uint64_t first,uint64_t elements,struct mesh_operand operand){
  operand.elements=elements;
  return (struct mesh_step){.op=op,.peer=peer,.round=round,.first=first,.piece=operand};
}

/* Ring all-reduce: Patarasuk and Yuan, "Bandwidth optimal all-reduce algorithms for clusters of
   workstations", JPDC 69(2) 2009, the ring reduce-scatter + all-gather (Rabenseifner, ICCS 2004).
   Body transcribed from baidu-research/baidu-allreduce collectives.cu RingAllreduce (Gibiansky 2017):
   segment sizes, recv_from/send_to, and the two loops, with `rank` read as the ring position and
   each MPI call replaced by the step it performs.  Every rank sends and receives 2(size-1)/size of
   the operand. */
static uint32_t mesh_ring_allreduce(const struct mesh_link_map *map,uint32_t node,struct mesh_operand operand,struct mesh_step *out){
  const uint32_t size=map->nodes;
  uint32_t next[MESH_LINK_MAP_NODES],previous[MESH_LINK_MAP_NODES],rank=0,count=0;
  for(uint32_t l=0;l<map->links;l++){next[map->link[l][0]]=map->link[l][1];previous[map->link[l][1]]=map->link[l][0];}
  for(uint32_t at=map->link[0][0];at!=node;at=next[at])rank++;
  // Compute the sizes of the chunks, and where each chunk ends.
  uint64_t segment_sizes[MESH_LINK_MAP_NODES],segment_ends[MESH_LINK_MAP_NODES];
  const uint64_t segment_size=operand.elements/size,residual=operand.elements%size;
  for(uint32_t i=0;i<size;i++)segment_sizes[i]=segment_size+(i<residual);
  segment_ends[0]=segment_sizes[0];
  for(uint32_t i=1;i<size;i++)segment_ends[i]=segment_sizes[i]+segment_ends[i-1];
  // Receive from your left neighbor; send to your right neighbor.
  const uint32_t recv_from=previous[node],send_to=next[node];
  // At the i'th iteration, sends segment (rank - i) and receives segment (rank - i - 1).
  for(uint32_t i=0;i<size-1;i++){
    uint32_t recv_chunk=(rank-i-1+size)%size,send_chunk=(rank-i+size)%size;
    out[count++]=mesh_piece(MESH_STEP_SEND,send_to,i,segment_ends[send_chunk]-segment_sizes[send_chunk],segment_sizes[send_chunk],operand);
    out[count++]=mesh_piece(MESH_STEP_REDUCE,recv_from,i,segment_ends[recv_chunk]-segment_sizes[recv_chunk],segment_sizes[recv_chunk],operand);
  }
  // Pipelined ring allgather: at the i'th iteration, sends segment (rank + 1 - i), receives (rank - i).
  for(uint32_t i=0;i<size-1;i++){
    uint32_t send_chunk=(rank-i+1+size)%size,recv_chunk=(rank-i+size)%size;
    out[count++]=mesh_piece(MESH_STEP_SEND,send_to,size-1+i,segment_ends[send_chunk]-segment_sizes[send_chunk],segment_sizes[send_chunk],operand);
    out[count++]=mesh_piece(MESH_STEP_COPY,recv_from,size-1+i,segment_ends[recv_chunk]-segment_sizes[recv_chunk],segment_sizes[recv_chunk],operand);
  }
  return count;
}

/* Tree all-reduce: MPICH MPIR_Reduce_intra_binomial followed by MPIR_Bcast_intra_binomial
   (src/mpi/coll/reduce/reduce_intra_binomial.c, src/mpi/coll/bcast/bcast_intra_binomial.c), with the
   children and parent that MPICH derives from rank bits read from the configured tree instead.
   Reduce: receive and reduce each child's whole operand, then send the result to the parent.
   Bcast: receive the whole result from the parent, then send it to each child.  On a star the
   root sends and receives (nodes-1) operands and every leaf one each way. */
static uint32_t mesh_tree_allreduce(const struct mesh_link_map *map,uint32_t rank,struct mesh_operand operand,struct mesh_step *out){
  uint32_t count=0;
  for(uint32_t l=0;l<map->links;l++)if(map->link[l][0]==rank)
    out[count++]=mesh_piece(MESH_STEP_REDUCE,map->link[l][1],0,0,operand.elements,operand);
  for(uint32_t l=0;l<map->links;l++)if(map->link[l][1]==rank){
    out[count++]=mesh_piece(MESH_STEP_SEND,map->link[l][0],0,0,operand.elements,operand);
    out[count++]=mesh_piece(MESH_STEP_COPY,map->link[l][0],1,0,operand.elements,operand);
  }
  for(uint32_t l=0;l<map->links;l++)if(map->link[l][0]==rank)
    out[count++]=mesh_piece(MESH_STEP_SEND,map->link[l][1],1,0,operand.elements,operand);
  return count;
}

/* Direct exchange on a full mesh: each rank sends its whole partial once to every other rank and
   sums what arrives, in any order (Megatron-LM's all-reduce of row-parallel partials, Shoeybi et al.
   2019, over MPI_Allreduce's semantics).  At two nodes this is the pair's existing exchange and has
   the ring's per-rank bytes.  Its SENDs may still be reading the operand when a REDUCE lands, so
   the sum goes to the caller's result operand, as in the pair's existing programs. */
static uint32_t mesh_direct_allreduce(const struct mesh_link_map *map,uint32_t rank,struct mesh_operand operand,struct mesh_step *out){
  uint32_t count=0;
  for(uint32_t peer=0;peer<map->nodes;peer++)if(peer!=rank)out[count++]=mesh_piece(MESH_STEP_SEND,peer,0,0,operand.elements,operand);
  for(uint32_t peer=0;peer<map->nodes;peer++)if(peer!=rank)out[count++]=mesh_piece(MESH_STEP_REDUCE,peer,0,0,operand.elements,operand);
  return count;
}

uint32_t mesh_allreduce_plan(const struct mesh_link_map *map,uint32_t rank,struct mesh_operand operand,struct mesh_step *steps){
  return map->kind==MESH_LINKS_RING?mesh_ring_allreduce(map,rank,operand,steps):
    map->kind==MESH_LINKS_TREE?mesh_tree_allreduce(map,rank,operand,steps):mesh_direct_allreduce(map,rank,operand,steps);
}

/* Each step becomes one prepared transfer, bound exactly as the pair's direct push binds its
   partials: a section over the typed piece, the peer's channel on queue identity%qps, and the
   same identity on the sending and the receiving rank. */
int mesh_allreduce_bind(struct mesh_ctx *context,const struct mesh_step *steps,uint32_t count,uint32_t identity,
  struct mesh_section operand,const struct mesh_section *received,uint32_t invocations,uint32_t invocation_pages,
  struct mesh_section *pieces){
  for(uint32_t i=0,reduced=0;i<count;i++){
    const struct mesh_step *step=steps+i;
    struct mesh_section piece;
    if(step->op==MESH_STEP_REDUCE)piece=received[reduced++];
    else{
      int status=mesh_section_slice(context,operand,step->first*step->piece.element_bytes,
        step->piece.elements*step->piece.element_bytes,invocations,invocation_pages,&piece);
      if(status)return status;
    }
    if(pieces)pieces[i]=piece;
    uint32_t transfer=identity+step->round;
    int status=mesh_transfer_bind(context,mesh_peer_channel(context,step->peer,transfer%context->M->qps),
      step->op==MESH_STEP_SEND?MESH_SEND:MESH_RECEIVE,transfer,piece,piece.pages);
    if(status)return status;
  }
  return 0;
}

/* The word layout and cancellation ranges of mesh_metal_transport_create, without the Metal
   buffers: link p's words follow link p-1's, invocation-major, one per received chunk in the
   frame order mesh_transfers_prepare assigned. */
int mesh_host_inputs(struct mesh_ctx *context){
  struct hdr *m=context->M;
  uint64_t frames[m->links?m->links:1],words=0;
  for(uint32_t p=0;p<m->links;p++){
    struct mesh_tx *tx=mesh_events(m,mesh_notice_queue(m,context->client,p));
    frames[p]=0;
    for(uint32_t q=p*m->qps;q<(p+1)*m->qps;q++){
      struct mesh_transfer *in=mesh_transfers(m,context->client,q,MESH_RECEIVE);
      for(uint32_t i=0;i<atomic_load(mesh_order_length(m,context->client,q,MESH_RECEIVE));i++)
        frames[p]+=(uint64_t)mesh_row_chunks(m,in[i].local_row,in[i].bytes)*in[i].count;
    }
    words+=frames[p]*tx->invocations;
  }
  struct mesh_section input,stop;
  int status=mesh_section_create(context,sizeof(uint64_t)*(words?words:1),1,0,&input);
  if(!status)status=mesh_section_create(context,sizeof(struct mesh_cancellation)+m->links*sizeof(struct mesh_cancel_range),1,0,&stop);
  if(status)return status;
  uint64_t *word=mesh_section_address(context,input,0);
  struct mesh_cancellation *cancel=mesh_section_address(context,stop,0);
  memset(word,0,input.bytes);memset(cancel,0,stop.bytes);
  for(uint32_t p=0;p<m->links;p++){
    struct mesh_tx *tx=mesh_events(m,mesh_notice_queue(m,context->client,p));
    tx->cancel=(uintptr_t)cancel-(uintptr_t)m;
    cancel->ranges[p]=(struct mesh_cancel_range){.offset=(uintptr_t)word-(uintptr_t)m,.count=frames[p]*tx->invocations};
    for(uint32_t q=p*m->qps;q<(p+1)*m->qps;q++){
      struct mesh_transfer *in=mesh_transfers(m,context->client,q,MESH_RECEIVE);
      for(uint32_t i=0;i<atomic_load(mesh_order_length(m,context->client,q,MESH_RECEIVE));i++){
        uint32_t chunks=mesh_row_chunks(m,in[i].local_row,in[i].bytes);
        for(uint32_t s=0;s<in[i].count;s++)for(uint32_t k=0;k<chunks;k++){
          struct mesh_publication *delivery=mesh_publication_at(m,in[i].local_row+s*in[i].stride+k);
          delivery->device_input=(uintptr_t)(word+in[i].first+s*chunks+k)-(uintptr_t)m;
          delivery->device_stride=sizeof(uint64_t)*frames[p];
        }
      }
    }
    word+=frames[p]*tx->invocations;
  }
  return 0;
}

/* What the GPU publication kernel stores (M04/M10): each prepared cell's continuation, into the
   cell of invocation t, released after the caller's writes to the section. */
void mesh_host_publish(struct mesh_ctx *context,struct mesh_section section,uint32_t invocation){
  uint32_t count=mesh_publication_prepare(context->M,section.first,NULL);
  struct prepared_publication records[count?count:1];
  mesh_publication_prepare(context->M,section.first,records);
  for(uint32_t i=0;i<count;i++)
    atomic_store_explicit((_Atomic uint64_t *)(uintptr_t)(records[i].destination+(uint64_t)invocation*sizeof(struct mesh_send)),
      records[i].argument,memory_order_release);
}

/* The last chunk's word, as mesh_metal_receive_prepare reads it: chunks of one receive land in order. */
uint64_t mesh_host_arrived(struct mesh_ctx *context,struct mesh_section section,uint32_t invocation){
  struct hdr *m=context->M;
  struct mesh_publication *delivery=mesh_publication_at(m,section.first+mesh_row_chunks(m,section.first,section.bytes)-1);
  return atomic_load_explicit((_Atomic uint64_t *)((char *)m+delivery->device_input+(uint64_t)invocation*delivery->device_stride),
    memory_order_acquire);
}
