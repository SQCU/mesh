#include "nccl.h"
#include "mesh-call.h"
#include "mesh-collective.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* libnccl-mesh (nccl.h) on the prepared transport's communicator streams (mesh.h mesh_ring).  The process attaches
   to its bridge's region once and, at its first communicator, binds STREAMS streams on every link and starts one
   session: the bridges pair once.  Communicator c takes streams 2c (its collectives) and 2c + 1 (its point-to-point
   calls) on each link, in the order the communicators are made, which is the same on every rank (NCCL's
   ncclCommInitRank and torch.distributed's new_group are collective), so each stream carries one communicator's
   calls of one channel in the order every rank issues them, and the k-th message on a stream is the peer's k-th.
     A call is its plan's steps (mesh_collective_choose and mesh_collective_plan over the communicator's link map:
   SEND, REDUCE, COPY), each step's piece cut into chunks alike on both ends (from its end, TAIL, TAIL, 2 TAIL, ...;
   the tag pre-revert-20261001's cut), each chunk one message.  A group of calls is one GPU program, its dispatches
   in one serial compute encoder: the operands placed in window memory (a send buffer's bytes stored
   system-coherent, so the NIC reads what the GPU wrote), every receive of the group described in its stream's ring,
   then each call's steps in plan order (a SEND's message described once its bytes are stored; a REDUCE's wait for
   its message and for the earlier SENDs reading its range, then its combine, stored system-coherent where a later
   SEND reads it), a wait for every message of the group, the results copied out where an operand was not the
   caller's, and the program's done word.  Every sequence number is a counter the GPU keeps a ring (the post kernel
   advances it in stream order), every wait the word the bridge stores, so a recording of the program replays as it
   ran (metal-microbench metal_recording.h), and no library thread runs.
     The window: slabs of SLAB bytes (or one of an allocation's size, a power of two) placed at a multiple of their
   size, within one registered region (mesh_link_info.extent), so no message straddles a region; an allocation is a
   span of a slab with a Metal buffer of its own (the caller's: ncclMeshMemAllocBuffer, freed once it is
   deallocated; or the library's: ncclMemAlloc, freed by ncclMemFree). */

/* nccl-mesh-metal.m */
int nccl_mesh_gpu_attach(char *error,size_t size);
void *nccl_mesh_buffer(const void *pointer,size_t bytes,uint64_t options,void (*gone)(void *),void *argument);
void *nccl_mesh_buffer_over(const void *pointer,size_t bytes,uint64_t *offset);
void *nccl_mesh_queue_create(void);
void nccl_mesh_release(void *object);
void *nccl_mesh_command_buffer(void *queue);
void *nccl_mesh_encoder(void *commandBuffer);
void nccl_mesh_encoder_end(void *encoder);
void nccl_mesh_commit_wait(void *commandBuffer,char *error,size_t size);
void nccl_mesh_kernel(void *encoder,int k,void *to,uint64_t dst,void *from,uint64_t src,uint64_t n,int type,int op,int nranks,
  uint64_t scalar,int published,void *other,uint64_t at);
void nccl_mesh_copy(void *encoder,void *to,uint64_t dst,void *from,uint64_t src,uint64_t bytes,int received,int published);
int nccl_mesh_post(void *encoder,void *rings,void *words,const uint64_t *list,size_t count,void **held);
int nccl_mesh_wait(void *encoder,void *words,const uint64_t *list,size_t count,void **held);
void nccl_mesh_publish(void *encoder,void *buffer,uint64_t at,uint64_t value);
void nccl_mesh_use(void *encoder,void *buffer);
enum { KERNEL_COMBINE, KERNEL_PREMULTIPLY, KERNEL_POSTDIVIDE };

#define STREAMS_DEFAULT 4
#define ENTRIES 4096
#define CHUNK (UINT32_C(4)<<20)
#define SLAB ((size_t)64<<20)
#define KEPT 65536
#define TAIL ((size_t)1<<20)
#define UID_MAGIC 0x4d4e4353u

/* ---- errors ---- */
static __thread char last_error[512];
static ncclResult_t fail(ncclResult_t result,const char *format,...){
  va_list a;va_start(a,format);vsnprintf(last_error,sizeof last_error,format,a);va_end(a);
  if(getenv("MESH_NCCL_DEBUG"))fprintf(stderr,"nccl-mesh: %s\n",last_error);
  return result;
}

/* ---- the process's attach, session and window ---- */
struct span { size_t at,bytes; int used; struct span *next; };
struct slab { unsigned char *base; size_t bytes; uint64_t window; struct span *spans; struct slab *next; };
struct allocation { unsigned char *base; size_t bytes; void *buffer; int owned; struct slab *slab; struct allocation *next; };
static struct {
  pthread_mutex_t lock;
  int attached,started;
  struct mesh_ctx context;
  uint32_t links,streams;
  struct mesh_ring **rings;
  void *region;                 /* a Metal buffer over the whole region: the rings, the counters, the words */
  struct mesh_section control;  /* counters (a word a ring), the rings' words (ENTRIES a ring), kept sequences, done words */
  uint64_t control_at,kept_at,kept_next,done_at,done_next;
  uint64_t extent;
  struct slab *slabs;
  struct allocation *allocations;
  int comms;
  void *queue;
} G={.lock=PTHREAD_MUTEX_INITIALIZER};

static struct hdr *region(void){return G.context.M;}
static uint64_t region_offset(const void *p){return (uint64_t)((const unsigned char *)p-(const unsigned char *)region());}
static uint32_t ring_index(uint32_t link,uint32_t stream,int direction){return (link*G.streams+stream)*2+(uint32_t)direction;}
static uint64_t counter_of(uint32_t ring){return G.control_at+(uint64_t)ring*8;}
static uint64_t words_of(uint32_t ring){return G.control_at+(uint64_t)G.links*G.streams*2*8+(uint64_t)ring*ENTRIES*8;}

/* The attach (MESH_REGION) and the GPU's kernels, once. */
static ncclResult_t attach(void){
  if(G.attached)return ncclSuccess;
  char error[256];
  if(nccl_mesh_gpu_attach(error,sizeof error))return fail(ncclSystemError,"Metal: %s",error);
  int status=mesh_attach(&G.context,NULL);
  if(status)return fail(ncclSystemError,"mesh_attach (MESH_REGION %s): %s",getenv("MESH_REGION")?:"/mesh0",strerror(status));
  struct hdr *m=region();
  G.links=m->links;
  const char *streams=getenv("MESH_NCCL_STREAMS");
  G.streams=streams && atoi(streams)>0?(uint32_t)atoi(streams):STREAMS_DEFAULT;
  G.extent=G.links?__atomic_load_n(&mesh_links(m)[0].extent,__ATOMIC_ACQUIRE):0;
  for(uint32_t p=1;p<G.links;p++){
    uint64_t e=__atomic_load_n(&mesh_links(m)[p].extent,__ATOMIC_ACQUIRE);
    if(e && (!G.extent || e<G.extent))G.extent=e;
  }
  if(!G.extent)G.extent=(uint64_t)1<<30;
  G.region=nccl_mesh_buffer(m,G.context.len,0,NULL,NULL);
  G.queue=nccl_mesh_queue_create();
  if(!G.region || !G.queue)return fail(ncclSystemError,"Metal buffers over the region");
  G.attached=1;
  return ncclSuccess;
}

/* The session: STREAMS streams on every link and their control words, then the start (the bridges pair).  Every
   rank starts it at its first communicator. */
static ncclResult_t session(void){
  if(G.started)return ncclSuccess;
  struct hdr *m=region();
  uint64_t rings=(uint64_t)G.links*G.streams*2;
  size_t bytes=(size_t)(rings*8+rings*ENTRIES*8+KEPT*8+4096*8);
  int status=mesh_section_create(&G.context,bytes,1,0,&G.control);
  if(status)return fail(ncclSystemError,"the control words: %s",strerror(status));
  unsigned char *control=mesh_section_address(&G.context,G.control,0);
  memset(control,0,bytes);
  G.control_at=region_offset(control);
  G.kept_at=G.control_at+rings*8+rings*ENTRIES*8;
  G.done_at=G.kept_at+(uint64_t)KEPT*8;
  status=mesh_transfers_prepare(&G.context,1,1,1);
  if(status)return fail(ncclSystemError,"mesh_transfers_prepare: %s",strerror(status));
  G.rings=calloc(rings?rings:1,sizeof *G.rings);
  if(!G.rings)return fail(ncclSystemError,"allocation");
  for(uint32_t p=0;p<G.links;p++){
    struct mesh_section storage;
    status=mesh_streams_bind(&G.context,mesh_links(m)[p].peer,G.streams,ENTRIES,CHUNK,G.rings+(size_t)p*G.streams*2,&storage);
    if(status)return fail(ncclSystemError,"mesh_streams_bind (link %u): %s",p,strerror(status));
  }
  status=mesh_transfers_start(&G.context);
  if(status)return fail(ncclRemoteError,"mesh_transfers_start: %s",strerror(status));
  G.started=1;
  return ncclSuccess;
}

/* A slab of `bytes` (SLAB, or a power of two for a larger allocation), in the registered window, placed at a
   multiple of its size within one registered region. */
static struct slab *slab_make(size_t bytes){
  struct hdr *m=region();
  size_t page=m->pgsz,align=bytes;
  if(align>G.extent)return NULL;
  struct mesh_section section;
  if(mesh_section_aligned(&G.context,bytes,(uint32_t)(align/page),1,&section))return NULL;
  const uint64_t window=(uint64_t)((unsigned char *)mesh_section_address(&G.context,section,0)-mesh_at(m,0));
  if(window/G.extent!=(window+bytes-1)/G.extent)return NULL;  /* the provider's regions are not a multiple of the slab */
  struct slab *s=calloc(1,sizeof *s);
  struct span *whole=calloc(1,sizeof *whole);
  if(!s || !whole){free(s);free(whole);return NULL;}
  s->base=mesh_section_address(&G.context,section,0);s->bytes=bytes;
  s->window=(uint64_t)(s->base-mesh_at(m,0));
  *whole=(struct span){.at=0,.bytes=bytes};
  s->spans=whole;s->next=G.slabs;G.slabs=s;
  return s;
}
/* `bytes` (whole pages) of a slab, first fit; NULL where the window has no room. */
static unsigned char *window_take(size_t bytes,struct slab **in){
  for(int fresh=0;fresh<2;fresh++){
    for(struct slab *s=G.slabs;s;s=s->next)for(struct span *p=s->spans;p;p=p->next){
      if(p->used || p->bytes<bytes)continue;
      if(p->bytes>bytes){
        struct span *rest=calloc(1,sizeof *rest);
        if(!rest)return NULL;
        *rest=(struct span){.at=p->at+bytes,.bytes=p->bytes-bytes,.next=p->next};
        p->next=rest;p->bytes=bytes;
      }
      p->used=1;*in=s;
      return s->base+p->at;
    }
    if(fresh)break;
    size_t size=SLAB;
    while(size<bytes)size<<=1;
    if(!slab_make(size))return NULL;
  }
  return NULL;
}
static void window_give(struct slab *s,unsigned char *at){
  for(struct span *p=s->spans;p;p=p->next)if(s->base+p->at==at){
    p->used=0;
    while(p->next && !p->next->used){struct span *n=p->next;p->bytes+=n->bytes;p->next=n->next;free(n);}
    return;
  }
}
static struct allocation *allocation_of(const void *p){
  for(struct allocation *a=G.allocations;a;a=a->next)
    if((const unsigned char *)p>=a->base && (const unsigned char *)p<a->base+a->bytes)return a;
  return NULL;
}
static size_t pages(size_t bytes){size_t page=(size_t)getpagesize();return (bytes?bytes:1)+page-1&~(page-1);}
/* An allocation of `bytes`, its Metal buffer made with `options`, owned by the caller (`owned`: freed once the
   buffer is deallocated) or by the library. */
static void allocation_gone(void *argument);
static struct allocation *allocate(size_t bytes,uint64_t options,int owned){
  if(attach() || !G.started)return NULL;
  struct slab *s;
  size_t size=pages(bytes);
  unsigned char *at=window_take(size,&s);
  if(!at)return NULL;
  struct allocation *a=calloc(1,sizeof *a);
  if(!a){window_give(s,at);return NULL;}
  *a=(struct allocation){.base=at,.bytes=size,.slab=s,.owned=owned};
  a->buffer=nccl_mesh_buffer(at,size,options,owned?allocation_gone:NULL,a);
  if(!a->buffer){window_give(s,at);free(a);return NULL;}
  a->next=G.allocations;G.allocations=a;
  return a;
}
static void allocation_free(struct allocation *a){
  struct allocation **p=&G.allocations;
  while(*p && *p!=a)p=&(*p)->next;
  if(*p)*p=a->next;
  window_give(a->slab,a->base);
  free(a);
}
static void allocation_gone(void *argument){
  pthread_mutex_lock(&G.lock);allocation_free(argument);pthread_mutex_unlock(&G.lock);
}

/* ---- datatypes and operators (the tag pre-revert-20261001's) ---- */
static const size_t type_bytes[ncclNumTypes]={1,1,4,4,8,8,2,4,8,2,1,1};
static int integral(ncclDataType_t t){return t<=ncclUint64;}
struct small { int e,m,fn,saturate; };
static const struct small BF16={8,7,0,0},E4M3={4,3,1,1},E5M2={5,2,0,1};
static uint32_t small_encode(double x,struct small f){
  uint32_t sign=signbit(x)?1u<<(f.e+f.m):0,top=(1u<<f.e)-1;
  int bias=(1<<(f.e-1))-1;
  uint32_t largest=f.fn?(top<<f.m)|((1u<<f.m)-2):((top-1)<<f.m)|((1u<<f.m)-1);
  if(isnan(x))return f.fn?sign|(top<<f.m)|((1u<<f.m)-1):sign|(top<<f.m)|(1u<<(f.m-1));
  double a=fabs(x);
  if(isinf(a))return f.saturate?sign|largest:sign|(top<<f.m);
  if(a==0)return sign;
  int exponent;frexp(a,&exponent);exponent-=1;
  if(exponent<1-bias)exponent=1-bias;
  double q=nearbyint(ldexp(a,f.m-exponent));
  if(q>=ldexp(1.0,f.m+1)){q=ldexp(1.0,f.m);exponent++;}
  uint32_t field=q<ldexp(1.0,f.m)?0:(uint32_t)(exponent+bias),mantissa=(uint32_t)q&((1u<<f.m)-1);
  uint32_t bits=(field<<f.m)|mantissa;
  if(field>top || (!f.fn && field==top) || (f.fn && field==top && mantissa==(1u<<f.m)-1) || bits>largest)
    return f.saturate?sign|largest:sign|(top<<f.m);
  return sign|bits;
}
static void reciprocal(unsigned char *scalar,ncclDataType_t t,int nranks){
  double v=1.0/nranks;
  memset(scalar,0,8);
  if(t==ncclFloat32){float k=(float)v;memcpy(scalar,&k,4);}
  else if(t==ncclFloat64)memcpy(scalar,&v,8);
  else if(t==ncclFloat16){_Float16 k=(_Float16)v;memcpy(scalar,&k,2);}
  else{uint32_t bits=small_encode(v,t==ncclBfloat16?BF16:t==ncclFloat8e4m3?E4M3:E5M2);memcpy(scalar,&bits,type_bytes[t]);}
}
/* PreMulSum operators: their scalars, by handle (ncclNumOps + i). */
#define CUSTOM 64
static struct { int used; ncclDataType_t type; unsigned char scalar[8]; } custom[CUSTOM];

/* ---- communicators ---- */
struct ncclComm {
  uint64_t key; int rank,nranks,index;
  int *link_of;                  /* rank r's link from this process (-1: this rank) */
  struct mesh_link_map map;      /* over the communicator's ranks */
  uint32_t (*pairs)[2];
  ncclResult_t async;
};
struct uid { uint32_t magic,version; uint64_t key; };
_Static_assert(sizeof(struct uid)<=NCCL_UNIQUE_ID_BYTES,"unique id");

/* The communicator's map over its ranks from the link-map file `path` over nodes (each rank's node `nodes[r]`),
   and each rank's link: the bridge link whose peer is its node. */
static ncclResult_t links_for(struct ncclComm *c,const char *path,const int *nodes){
  struct mesh_link_map file;
  int status=mesh_link_map_read(path,&file);
  if(status)return fail(ncclInvalidArgument,"the link map %s: %s",path,strerror(status));
  c->pairs=calloc(file.links?file.links:1,sizeof *c->pairs);
  c->link_of=calloc((size_t)c->nranks,sizeof *c->link_of);
  if(!c->pairs || !c->link_of){mesh_link_map_free(&file);return fail(ncclSystemError,"allocation");}
  int *rank_of=calloc(file.nodes?file.nodes:1,sizeof *rank_of);
  if(!rank_of){mesh_link_map_free(&file);return fail(ncclSystemError,"allocation");}
  for(uint32_t v=0;v<file.nodes;v++)rank_of[v]=-1;
  for(int r=0;r<c->nranks;r++){
    if(nodes[r]<0 || (uint32_t)nodes[r]>=file.nodes || rank_of[nodes[r]]>=0){
      free(rank_of);mesh_link_map_free(&file);return fail(ncclInvalidArgument,"rank %d's node %d: outside the map or another rank's",r,nodes[r]);
    }
    rank_of[nodes[r]]=r;
  }
  uint32_t n=0;
  for(uint32_t l=0;l<file.links;l++){
    int a=rank_of[file.link[l][0]],b=rank_of[file.link[l][1]];
    if(a>=0 && b>=0){c->pairs[n][0]=(uint32_t)a;c->pairs[n][1]=(uint32_t)b;n++;}
  }
  c->map=(struct mesh_link_map){.kind=file.kind,.nodes=(uint32_t)c->nranks,.links=n,.link=c->pairs};
  if(file.kind!=MESH_LINKS_MESH && (uint32_t)c->nranks!=file.nodes)c->map.kind=MESH_LINKS_TREE;
  free(rank_of);mesh_link_map_free(&file);
  struct hdr *m=region();
  for(int r=0;r<c->nranks;r++){
    c->link_of[r]=-1;
    if(r==c->rank)continue;
    for(uint32_t p=0;p<m->links;p++)if(mesh_links(m)[p].peer==(uint32_t)nodes[r])c->link_of[r]=(int)p;
  }
  return ncclSuccess;
}

static ncclResult_t comm_make(ncclComm_t *out,int nranks,uint64_t key,int rank,const ncclMeshConfig_t *mesh){
  if(nranks<1 || rank<0 || rank>=nranks)return fail(ncclInvalidArgument,"rank %d of %d",rank,nranks);
  if(!mesh || !mesh->links || !mesh->nodes)
    return fail(ncclInvalidUsage,"no link map: the communicator takes an ncclMeshConfig_t (base.size sizeof(ncclMeshConfig_t)) naming a link-map file and each rank's node in it");
  pthread_mutex_lock(&G.lock);
  ncclResult_t result=attach();
  if(result==ncclSuccess)result=session();
  if(result!=ncclSuccess){pthread_mutex_unlock(&G.lock);return result;}
  struct ncclComm *c=calloc(1,sizeof *c);
  if(!c){pthread_mutex_unlock(&G.lock);return fail(ncclSystemError,"allocation");}
  c->key=key;c->rank=rank;c->nranks=nranks;
  c->index=G.comms;
  if((uint32_t)(2*c->index+1)>=G.streams){
    pthread_mutex_unlock(&G.lock);free(c);
    return fail(ncclInvalidUsage,"communicator %d: its streams are past the session's %u a link (MESH_NCCL_STREAMS)",G.comms,G.streams);
  }
  G.comms++;
  result=links_for(c,mesh->links,mesh->nodes);
  pthread_mutex_unlock(&G.lock);
  if(result!=ncclSuccess){free(c->pairs);free(c->link_of);free(c);return result;}
  for(int r=0;r<nranks;r++)if(r!=rank && c->link_of[r]<0)
    return fail(ncclInvalidUsage,"rank %d's node has no bridge link from this node",r);
  *out=c;
  return ncclSuccess;
}

/* ---- calls ---- */
enum { K_SEND=MESH_ALLGATHER+1, K_RECV, K_LOCAL };
/* An operand: a Metal buffer and an offset in it (`buffer` NULL: none). */
struct place { void *buffer; uint64_t offset; uint64_t window; int in_window; unsigned char *at; };
struct call {
  int kind; const void *send; void *recv; size_t count; ncclDataType_t type; int op,root,peer;
  struct ncclComm *comm; cudaStream_t stream; uint32_t how; uint64_t *segments;
  int combine,premultiply,postdivide; unsigned char scalar[8];
};
struct group { struct call *calls; int n,capacity,depth; };
static __thread struct group pending;

/* Where `p` (bytes of it) lies: its allocation's buffer and offset; outside the window, a Metal buffer over its
   pages (held by the caller until its program has run: `held`). */
static struct place place_of(const void *p,size_t bytes,void ***held,int *nheld){
  struct place out={0};
  struct allocation *a=allocation_of(p);
  if(a && (const unsigned char *)p+bytes<=a->base+a->bytes){
    out=(struct place){.buffer=a->buffer,.offset=(uint64_t)((const unsigned char *)p-a->base),.in_window=1,
      .window=a->slab->window+(uint64_t)(a->base-a->slab->base)+(uint64_t)((const unsigned char *)p-a->base),.at=(unsigned char *)p};
    return out;
  }
  uint64_t offset;
  void *buffer=nccl_mesh_buffer_over(p,bytes,&offset);
  if(buffer){(*held)[(*nheld)++]=buffer;}
  out=(struct place){.buffer=buffer,.offset=offset,.at=(unsigned char *)p};
  return out;
}

/* The tag's cut of a step's piece into chunks: from its end TAIL, TAIL, 2 TAIL, ... bytes, the first the rest. */
static uint32_t chunk_doublings(uint64_t elements,size_t e){
  const uint64_t t=TAIL/e?TAIL/e:1;
  uint32_t m=0;
  while(m<40 && (t<<(m+1))<=elements)m++;
  return m;
}
static uint32_t chunks_of(uint64_t elements,size_t e){return chunk_doublings(elements,e)+1;}
static uint64_t chunk_first(uint64_t elements,size_t e,uint32_t j){
  const uint32_t m=chunk_doublings(elements,e);
  const uint64_t t=TAIL/e?TAIL/e:1;
  return j?elements-(t<<(m-j)):0;
}
static uint64_t chunk_count(uint64_t elements,size_t e,uint32_t j){
  const uint32_t m=chunk_doublings(elements,e);
  const uint64_t t=TAIL/e?TAIL/e:1;
  if(!j)return m?elements-(t<<(m-1)):elements;
  return j==m?t:t<<(m-j-1);
}

/* A message of a program: its ring, its kept sequence's slot, its call, the operand range it moves (byte offsets
   in the call's operand: a SEND's reads, a COPY's writes; a REDUCE's lands in scratch, its combine writes the
   range), whether its word has been waited for. */
struct message { uint32_t ring; uint64_t kept,lo,hi; int call,send,reduce,waited; };
struct program {
  void *encoder;
  struct message *messages; int n,capacity;
  uint64_t *list; size_t nlist,listcap;
  void **held; int nheld,heldcap;
};
static int program_hold(struct program *g,void *object){
  if(!object)return 0;
  if(g->nheld==g->heldcap){
    int cap=g->heldcap?2*g->heldcap:16;
    void **grown=realloc(g->held,(size_t)cap*sizeof *grown);
    if(!grown)return -1;
    g->held=grown;g->heldcap=cap;
  }
  g->held[g->nheld++]=object;
  return 0;
}
static int list_push(struct program *g,uint64_t v){
  if(g->nlist==g->listcap){
    size_t cap=g->listcap?2*g->listcap:64;
    uint64_t *grown=realloc(g->list,cap*sizeof *grown);
    if(!grown)return -1;
    g->list=grown;g->listcap=cap;
  }
  g->list[g->nlist++]=v;
  return 0;
}
static int message_add(struct program *g,uint32_t ring,int call,uint64_t lo,uint64_t hi,int send,int reduce){
  if(g->n==g->capacity){
    int cap=g->capacity?2*g->capacity:64;
    struct message *grown=realloc(g->messages,(size_t)cap*sizeof *grown);
    if(!grown)return -1;
    g->messages=grown;g->capacity=cap;
  }
  uint64_t kept=G.kept_at+(G.kept_next++%KEPT)*8;
  g->messages[g->n++]=(struct message){.ring=ring,.kept=kept,.call=call,.lo=lo,.hi=hi,.send=send,.reduce=reduce};
  return g->n-1;
}
/* Messages described by one post kernel: the program's messages [first, first + count), whose window offsets and
   bytes are `window`[i] and `bytes`[i]. */
#define BATCH 256
struct batch { int first,count; uint64_t window[BATCH],bytes[BATCH]; };
static int post(struct program *g,struct batch *b){
  if(!b->count)return 0;
  g->nlist=0;
  if(list_push(g,(uint64_t)b->count) || list_push(g,ENTRIES))return -1;
  for(int i=0;i<b->count;i++){
    struct message *x=g->messages+b->first+i;
    const struct mesh_ring *ring=G.rings[x->ring];
    if(list_push(g,region_offset(ring)) || list_push(g,counter_of(x->ring)) || list_push(g,b->window[i]) || list_push(g,b->bytes[i]) ||
       list_push(g,words_of(x->ring)) || list_push(g,x->kept))return -1;
  }
  void *held=NULL;
  if(nccl_mesh_post(g->encoder,G.region,G.region,g->list,g->nlist,&held))return -1;
  b->first+=b->count;b->count=0;
  return program_hold(g,held);
}
static int describe(struct program *g,struct batch *b,uint32_t ring,int call,uint64_t lo,uint64_t hi,int send,int reduce,uint64_t window){
  int i=message_add(g,ring,call,lo,hi,send,reduce);
  if(i<0)return -1;
  if(!b->count)b->first=i;
  b->window[b->count]=window;b->bytes[b->count]=hi-lo;b->count++;
  return b->count==BATCH?post(g,b):0;
}
/* One wait kernel for the listed messages (indices), each marked waited. */
static int wait_for(struct program *g,const int *which,int count){
  if(!count)return 0;
  g->nlist=0;
  if(list_push(g,(uint64_t)count) || list_push(g,ENTRIES))return -1;
  for(int i=0;i<count;i++){
    struct message *x=g->messages+which[i];
    if(list_push(g,x->kept) || list_push(g,words_of(x->ring)))return -1;
    x->waited=1;
  }
  void *held=NULL;
  if(nccl_mesh_wait(g->encoder,G.region,g->list,g->nlist,&held))return -1;
  return program_hold(g,held);
}
/* Waits for the unwaited messages of call `call` overlapping [lo, hi) that `access` conflicts with: before a SEND
   reads a range, the receives writing it (COPY); before a combine writes a range, the receives writing it and the
   SENDs reading it; plus `also` (a message of its own, or -1). */
static int wait_hazards(struct program *g,int call,uint64_t lo,uint64_t hi,int writes,int also){
  int which[256],count=0;
  if(also>=0)which[count++]=also;
  for(int i=0;i<g->n;i++){
    struct message *y=g->messages+i;
    if(y->waited || y->call!=call || i==also || y->reduce || !(y->lo<hi && lo<y->hi))continue;
    if(y->send && !writes)continue;
    if(count==256){if(wait_for(g,which,count))return -1;count=0;}
    which[count++]=i;
  }
  return wait_for(g,which,count);
}

/* The communicator's stream for a call: its collectives' or its point-to-point calls', on rank `peer`'s link. */
static uint32_t ring_for(const struct call *k,int peer,int direction){
  int channel=k->kind==K_SEND || k->kind==K_RECV;
  return ring_index((uint32_t)k->comm->link_of[peer],(uint32_t)(2*k->comm->index+channel),direction);
}

/* A call of a program: its plan's steps, its operands, the window operand the steps work on, its scratch (where a
   REDUCE step's pieces land, each at pieces[step]). */
struct planned {
  struct call *k; struct mesh_step *steps; uint32_t nsteps;
  struct place in,out,operand; struct allocation *scratch;
  uint64_t *pieces;
  size_t element;
};
static ncclResult_t plan(struct planned *p){
  struct call *k=p->k;
  struct ncclComm *c=k->comm;
  p->element=type_bytes[k->type];
  if(k->kind==K_LOCAL)return ncclSuccess;
  if(k->kind==K_SEND || k->kind==K_RECV){
    p->nsteps=1;p->steps=calloc(1,sizeof *p->steps);
    if(!p->steps)return fail(ncclSystemError,"allocation");
    p->steps[0]=(struct mesh_step){.op=k->kind==K_SEND?MESH_STEP_SEND:MESH_STEP_COPY,.peer=(uint32_t)k->peer,
      .piece={.type=(uint32_t)k->type,.element_bytes=(uint32_t)p->element,.elements=k->count}};
    return ncclSuccess;
  }
  struct mesh_operand operand={.type=(uint32_t)k->type,.element_bytes=(uint32_t)p->element,
    .elements=k->kind==MESH_ALLGATHER || k->kind==MESH_REDUCE_SCATTER?(k->segments?0:k->count*(uint64_t)c->nranks):k->count};
  if(k->segments)for(int r=0;r<c->nranks;r++)operand.elements+=k->segments[r];
  struct mesh_collective wanted={.what=(uint32_t)k->kind,.how=k->how,.root=(uint32_t)k->root,.segments=k->segments};
  struct mesh_collective chosen=mesh_collective_choose(&c->map,wanted,operand,0,0);
  if(chosen.how==MESH_UNAVAILABLE)return fail(ncclInvalidUsage,"no algorithm of the planner carries this call on the communicator's map");
  chosen.segments=k->segments;
  p->steps=calloc(MESH_COLLECTIVE_STEPS(c->nranks),sizeof *p->steps);
  if(!p->steps)return fail(ncclSystemError,"allocation");
  p->nsteps=mesh_collective_plan(&c->map,(uint32_t)c->rank,chosen,operand,p->steps);
  if(!p->nsteps)return fail(ncclInvalidUsage,"the planner has no steps for rank %d",c->rank);
  return ncclSuccess;
}
static uint64_t place_window(const struct allocation *a){return a->slab->window+(uint64_t)(a->base-a->slab->base);}

/* A call's receives, every REDUCE and COPY chunk in plan order (its peers' SENDs' order), described. */
static ncclResult_t encode_receives(struct program *g,struct planned *p,int call,struct batch *b){
  struct call *k=p->k;
  const size_t e=p->element;
  for(uint32_t s=0;s<p->nsteps;s++){
    const struct mesh_step *step=p->steps+s;
    if(step->op==MESH_STEP_SEND || !step->piece.elements)continue;
    for(uint32_t j=0;j<chunks_of(step->piece.elements,e);j++){
      const uint64_t first=chunk_first(step->piece.elements,e,j)*e,lo=step->first*e+first,hi=lo+chunk_count(step->piece.elements,e,j)*e;
      const uint64_t window=step->op==MESH_STEP_REDUCE?place_window(p->scratch)+p->pieces[s]+first:p->operand.window+lo;
      if(describe(g,b,ring_for(k,(int)step->peer,MESH_RECEIVE),call,lo,hi,0,step->op==MESH_STEP_REDUCE,window))
        return fail(ncclSystemError,"allocation");
    }
  }
  return ncclSuccess;
}
/* A call's steps in plan order: a SEND chunk described once the receives writing its range have landed (a COPY a
   tree forwards) and its bytes are stored system-coherent (the operand's were when it was placed, a combine's are
   when it is stored); a REDUCE chunk's wait (its message, and the unwaited receives and SENDs of its range), then
   its combine, stored system-coherent where a later SEND reads it.  `next` is the call's first receive. */
static ncclResult_t encode_steps(struct program *g,struct planned *p,int call,int next){
  struct call *k=p->k;
  struct ncclComm *c=k->comm;
  const size_t e=p->element;
  struct batch b={0};
  for(uint32_t s=0;s<p->nsteps;s++){
    const struct mesh_step *step=p->steps+s;
    if(!step->piece.elements)continue;
    const uint32_t chunks=chunks_of(step->piece.elements,e);
    for(uint32_t j=0;j<chunks;j++){
      const uint64_t first=chunk_first(step->piece.elements,e,j)*e,lo=step->first*e+first,hi=lo+chunk_count(step->piece.elements,e,j)*e;
      if(step->op==MESH_STEP_SEND){
        int pending=0;
        for(int i=0;i<g->n && !pending;i++){
          struct message *y=g->messages+i;
          pending=!y->waited && !y->send && y->call==call && !y->reduce && y->lo<hi && lo<y->hi;
        }
        if(pending && (post(g,&b) || wait_hazards(g,call,lo,hi,0,-1)))return fail(ncclSystemError,"wait");
        if(describe(g,&b,ring_for(k,(int)step->peer,MESH_SEND),call,lo,hi,1,0,p->operand.window+lo))return fail(ncclSystemError,"post");
        continue;
      }
      const int me=next++;
      if(step->op!=MESH_STEP_REDUCE)continue;
      if(post(g,&b) || wait_hazards(g,call,lo,hi,1,me))return fail(ncclSystemError,"wait");
      int published=0;
      for(uint32_t t=s+1;t<p->nsteps && !published;t++)
        published=p->steps[t].op==MESH_STEP_SEND && p->steps[t].first*e<hi && lo<(p->steps[t].first+p->steps[t].piece.elements)*e;
      nccl_mesh_kernel(g->encoder,KERNEL_COMBINE,p->operand.buffer,p->operand.offset+lo,p->scratch->buffer,p->pieces[s]+first,(hi-lo)/e,
        (int)k->type,k->combine,c->nranks,0,published,NULL,0);
    }
  }
  if(post(g,&b))return fail(ncclSystemError,"post");
  return ncclSuccess;
}

static ncclResult_t check_call(struct call *k){
  if(!k->comm)return fail(ncclInvalidArgument,"no communicator");
  if((int)k->type<0 || k->type>=ncclNumTypes)return fail(ncclInvalidArgument,"datatype %d",(int)k->type);
  if(k->kind!=K_SEND && k->kind!=K_RECV && k->kind!=MESH_ALLGATHER && k->kind!=MESH_BROADCAST){
    if(k->op>=ncclNumOps && (k->op-ncclNumOps>=CUSTOM || !custom[k->op-ncclNumOps].used))return fail(ncclInvalidArgument,"operator %d",k->op);
  }
  if((k->kind==K_SEND || k->kind==K_RECV) && (k->peer<0 || k->peer>=k->comm->nranks || k->peer==k->comm->rank))
    return fail(ncclInvalidArgument,"peer %d",k->peer);
  /* the operator: ncclAvg premultiplies floating types by 1/n and post-divides integers; PreMulSum premultiplies */
  k->combine=k->op<ncclNumOps?(k->op==ncclAvg?ncclSum:k->op):ncclSum;
  if(k->op==ncclAvg){if(integral(k->type))k->postdivide=1;else{k->premultiply=1;reciprocal(k->scalar,k->type,k->comm->nranks);}}
  else if(k->op>=ncclNumOps){k->premultiply=1;memcpy(k->scalar,custom[k->op-ncclNumOps].scalar,8);}
  return ncclSuccess;
}

/* The bytes of a call's send and receive buffers. */
static size_t in_bytes(const struct call *k){
  size_t e=type_bytes[k->type];
  if(k->kind==MESH_REDUCE_SCATTER){
    if(k->segments){size_t t=0;for(int r=0;r<k->comm->nranks;r++)t+=k->segments[r];return t*e;}
    return k->count*(size_t)k->comm->nranks*e;
  }
  if(k->kind==MESH_ALLGATHER)return (k->segments?k->segments[k->comm->rank]:k->count)*e;
  return k->count*e;
}
static size_t out_bytes(const struct call *k){
  size_t e=type_bytes[k->type];
  if(k->kind==MESH_ALLGATHER){
    if(k->segments){size_t t=0;for(int r=0;r<k->comm->nranks;r++)t+=k->segments[r];return t*e;}
    return k->count*(size_t)k->comm->nranks*e;
  }
  if(k->kind==MESH_REDUCE_SCATTER)return (k->segments?k->segments[k->comm->rank]:k->count)*e;
  return k->count*e;
}
/* rank r's segment's first element of an all-gather's or reduce-scatter's operand */
static uint64_t segment_first(const struct call *k,int r){
  if(!k->segments)return (uint64_t)r*k->count;
  uint64_t t=0;for(int q=0;q<r;q++)t+=k->segments[q];
  return t;
}

/* The operand of a call, placed (nothing encoded): the window bytes its steps work on.  An all-reduce, broadcast,
   all-gather or receive works in its receive buffer where that lies in the window, a send in its send buffer
   there; anything else in an operand of its own, copied out at the end.  Its scratch: where a REDUCE step's
   pieces land. */
static ncclResult_t place_call(struct program *g,struct planned *p,struct allocation **own){
  struct call *k=p->k;
  const size_t e=p->element,in=in_bytes(k),out=out_bytes(k);
  int nheld=0;void *over[2]={0};void **at=over;
  if(k->send && in)p->in=place_of(k->send,in,&at,&nheld);
  if(k->recv && out)p->out=place_of(k->recv,out,&at,&nheld);
  for(int i=0;i<nheld;i++)if(program_hold(g,over[i]))return fail(ncclSystemError,"allocation");
  if((k->send && in && !p->in.buffer) || (k->recv && out && !p->out.buffer))return fail(ncclSystemError,"a Metal buffer over a call's operand");
  if(k->kind==K_LOCAL){p->operand=p->out;return ncclSuccess;}
  const size_t operand_bytes=k->kind==MESH_REDUCE_SCATTER || k->kind==MESH_REDUCE || k->kind==K_SEND?in:out;
  if((k->kind==MESH_ALLREDUCE || k->kind==MESH_BROADCAST || k->kind==MESH_ALLGATHER || k->kind==K_RECV) && p->out.in_window)p->operand=p->out;
  else if(k->kind==K_SEND && p->in.in_window && !k->premultiply)p->operand=p->in;
  else {
    pthread_mutex_lock(&G.lock);
    *own=allocate(operand_bytes,0,0);
    pthread_mutex_unlock(&G.lock);
    if(!*own)return fail(ncclSystemError,"no room in the window for a call's operand (%zu bytes)",operand_bytes);
    p->operand=(struct place){.buffer=(*own)->buffer,.in_window=1,.window=place_window(*own),.at=(*own)->base};
  }
  size_t scratch=0;
  for(uint32_t s=0;s<p->nsteps;s++)if(p->steps[s].op==MESH_STEP_REDUCE)scratch+=pages(p->steps[s].piece.elements*e);
  if(scratch){
    pthread_mutex_lock(&G.lock);
    p->scratch=allocate(scratch,0,0);
    pthread_mutex_unlock(&G.lock);
    if(!p->scratch)return fail(ncclSystemError,"no room in the window for a call's received pieces (%zu bytes)",scratch);
  }
  p->pieces=calloc(p->nsteps?p->nsteps:1,sizeof *p->pieces);
  if(!p->pieces)return fail(ncclSystemError,"allocation");
  uint64_t offset=0;
  for(uint32_t s=0;s<p->nsteps;s++)if(p->steps[s].op==MESH_STEP_REDUCE){p->pieces[s]=offset;offset+=pages(p->steps[s].piece.elements*e);}
  return ncclSuccess;
}
/* The call's input stored into its operand system-coherent (a SEND reads what the GPU stored only once it is so; in
   place, stored again so), the premultiplication folded in; a local copy's bytes. */
static void load_call(struct program *g,struct planned *p){
  struct call *k=p->k;
  struct ncclComm *c=k->comm;
  const size_t e=p->element,in=in_bytes(k);
  uint64_t into=0,length=0;
  switch(k->kind){
  case MESH_ALLREDUCE: case MESH_REDUCE: case MESH_REDUCE_SCATTER: case K_SEND: case K_LOCAL: length=in;break;
  case MESH_BROADCAST: if(c->rank==k->root)length=in;break;
  case MESH_ALLGATHER: into=segment_first(k,c->rank)*e;length=in;break;
  default: break;
  }
  if(!length)return;
  if(k->premultiply)
    nccl_mesh_kernel(g->encoder,KERNEL_PREMULTIPLY,p->operand.buffer,p->operand.offset+into,p->in.buffer,p->in.offset,length/e,(int)k->type,0,
      c->nranks,*(uint64_t *)k->scalar,1,NULL,0);
  else nccl_mesh_copy(g->encoder,p->operand.buffer,p->operand.offset+into,p->in.buffer,p->in.offset,length,0,k->kind!=K_LOCAL);
}

/* The results out of the operand (where it was not the receive buffer), and an integer ncclAvg's division. */
static void finish_call(struct program *g,struct planned *p){
  struct call *k=p->k;
  struct ncclComm *c=k->comm;
  const size_t e=p->element;
  if(k->kind==K_SEND || k->kind==K_LOCAL)return;
  if(k->kind==MESH_REDUCE && c->rank!=k->root)return;
  uint64_t from=0,length=out_bytes(k);
  if(k->kind==MESH_REDUCE_SCATTER)from=segment_first(k,c->rank)*e;
  if(p->operand.buffer!=p->out.buffer || p->operand.offset!=p->out.offset)
    nccl_mesh_copy(g->encoder,p->out.buffer,p->out.offset,p->operand.buffer,p->operand.offset+from,length,1,0);
  if(k->postdivide)
    nccl_mesh_kernel(g->encoder,KERNEL_POSTDIVIDE,p->out.buffer,p->out.offset,p->out.buffer,p->out.offset,length/e,(int)k->type,0,c->nranks,0,0,NULL,0);
}

/* A deferred stream's kept work: programs to encode, each its calls (copied) */
struct kept { struct call *calls; int n; struct kept *next; };

/* The program of a group's `calls` into `commandBuffer`: every call planned and placed first (nothing encoded
   where that fails), then its inputs loaded, every receive of the group described, each call's steps, one wait for
   every message not yet waited for, the results finished, and the done word (`done`: a window word) set to
   `value` last.  What the program holds (its operands' and scratch allocations, the Metal objects it made) is the
   caller's to release once it has run. */
static ncclResult_t encode_group(void *commandBuffer,struct call *calls,int n,uint64_t *done,uint64_t value,
                                  struct allocation ***scratch,int *nscratch,void ***held,int *nheld){
  struct program g={0};
  struct planned *p=calloc((size_t)n,sizeof *p);
  struct allocation **owned=calloc((size_t)n*2,sizeof *owned);
  ncclResult_t result=p && owned?ncclSuccess:fail(ncclSystemError,"allocation");
  for(int i=0;result==ncclSuccess && i<n;i++){p[i].k=calls+i;result=plan(p+i);}
  for(int i=0;result==ncclSuccess && i<n;i++){result=place_call(&g,p+i,owned+2*i);owned[2*i+1]=p[i].scratch;}
  for(int i=0;result!=ncclSuccess && owned && i<n;i++)owned[2*i+1]=p[i].scratch;
  if(result==ncclSuccess){
    g.encoder=nccl_mesh_encoder(commandBuffer);
    for(int i=0;i<n;i++){
      if(p[i].in.buffer)nccl_mesh_use(g.encoder,p[i].in.buffer);
      if(p[i].out.buffer)nccl_mesh_use(g.encoder,p[i].out.buffer);
      if(p[i].operand.buffer)nccl_mesh_use(g.encoder,p[i].operand.buffer);
      if(p[i].scratch)nccl_mesh_use(g.encoder,p[i].scratch->buffer);
    }
    for(int i=0;i<n;i++)load_call(&g,p+i);
    int *first=calloc((size_t)n,sizeof *first);
    struct batch *b=calloc(1,sizeof *b);
    if(!first || !b)result=fail(ncclSystemError,"allocation");
    for(int i=0;result==ncclSuccess && i<n;i++){first[i]=g.n;result=encode_receives(&g,p+i,i,b);}
    if(result==ncclSuccess && post(&g,b))result=fail(ncclSystemError,"post");
    for(int i=0;result==ncclSuccess && i<n;i++)result=encode_steps(&g,p+i,i,first[i]);
    free(first);free(b);
    if(result==ncclSuccess){
      int *all=malloc(((size_t)g.n+1)*sizeof *all),count=0;
      if(!all)result=fail(ncclSystemError,"allocation");
      for(int i=0;all && i<g.n;i++)if(!g.messages[i].waited)all[count++]=i;
      if(all && wait_for(&g,all,count))result=fail(ncclSystemError,"wait");
      free(all);
    }
    for(int i=0;result==ncclSuccess && i<n;i++)finish_call(&g,p+i);
    if(result==ncclSuccess && done)nccl_mesh_publish(g.encoder,G.region,region_offset(done),value);
    nccl_mesh_encoder_end(g.encoder);
  }
  for(int i=0;p && i<n;i++){free(p[i].steps);free(p[i].pieces);}
  free(p);free(g.messages);free(g.list);
  *scratch=owned;*nscratch=owned?2*n:0;
  *held=g.held;*nheld=g.nheld;
  return result;
}
static void release_group(struct allocation **scratch,int nscratch,void **held,int nheld){
  pthread_mutex_lock(&G.lock);
  for(int i=0;i<nscratch;i++)if(scratch[i])allocation_free(scratch[i]);
  pthread_mutex_unlock(&G.lock);
  for(int i=0;i<nheld;i++)nccl_mesh_release(held[i]);
  free(scratch);free(held);
}

/* A deferred stream's program waiting to be released: its scratch and held objects, freed once its done word
   reaches its value. */
struct ran { uint64_t *done; uint64_t value; struct allocation **scratch; int nscratch; void **held; int nheld; struct ran *next; };
static struct ran *running;
static void reap(void){
  struct ran **r=&running;
  while(*r){
    struct ran *x=*r;
    if(atomic_load_explicit((_Atomic uint64_t *)x->done,memory_order_acquire)>=x->value){
      *r=x->next;release_group(x->scratch,x->nscratch,x->held,x->nheld);free(x);
    } else r=&x->next;
  }
}

/* A group's calls: on the NULL stream run now on the library's queue and waited for; on a stream kept for its
   ncclMeshStreamEncode. */
static ncclResult_t run(struct call *calls,int n){
  if(!n)return ncclSuccess;
  for(int i=0;i<n;i++){ncclResult_t r=check_call(calls+i);if(r!=ncclSuccess)return r;}
  cudaStream_t stream=calls[0].stream;
  for(int i=1;i<n;i++)if(calls[i].stream!=stream)return fail(ncclInvalidUsage,"a group's calls on more than one stream");
  if(stream){
    struct kept *x=calloc(1,sizeof *x);
    struct call *copy=malloc((size_t)n*sizeof *copy);
    if(!x || !copy){free(x);free(copy);return fail(ncclSystemError,"allocation");}
    memcpy(copy,calls,(size_t)n*sizeof *copy);
    for(int i=0;i<n;i++)if(calls[i].segments){
      copy[i].segments=malloc((size_t)calls[i].comm->nranks*sizeof(uint64_t));
      if(!copy[i].segments)return fail(ncclSystemError,"allocation");
      memcpy(copy[i].segments,calls[i].segments,(size_t)calls[i].comm->nranks*sizeof(uint64_t));
    }
    x->calls=copy;x->n=n;
    struct kept **tail=(struct kept **)&stream->pending;
    while(*tail)tail=&(*tail)->next;
    *tail=x;
    return ncclSuccess;
  }
  void *command=nccl_mesh_command_buffer(G.queue);
  struct allocation **scratch;int nscratch;void **held;int nheld;
  ncclResult_t result=encode_group(command,calls,n,NULL,0,&scratch,&nscratch,&held,&nheld);
  char error[256]="";
  if(result==ncclSuccess)nccl_mesh_commit_wait(command,error,sizeof error);
  else nccl_mesh_release(command);
  release_group(scratch,nscratch,held,nheld);
  if(result==ncclSuccess && error[0])result=fail(ncclUnhandledCudaError,"a GPU program failed: %s",error);
  return result;
}

static ncclResult_t issue(struct call k){
  if(pending.depth){
    if(pending.n==pending.capacity){
      int cap=pending.capacity?2*pending.capacity:16;
      struct call *grown=realloc(pending.calls,(size_t)cap*sizeof *grown);
      if(!grown)return fail(ncclSystemError,"allocation");
      pending.calls=grown;pending.capacity=cap;
    }
    if(k.segments){
      uint64_t *copy=malloc((size_t)k.comm->nranks*sizeof *copy);
      if(!copy)return fail(ncclSystemError,"allocation");
      memcpy(copy,k.segments,(size_t)k.comm->nranks*sizeof *copy);k.segments=copy;
    }
    pending.calls[pending.n++]=k;
    return ncclSuccess;
  }
  return run(&k,1);
}

/* ---- the API ---- */
#define NCCL_API(ret,name,...) ret name(__VA_ARGS__) __attribute__((visibility("default")))
ncclResult_t ncclGetVersion(int *version){if(!version)return ncclInvalidArgument;*version=NCCL_VERSION_CODE;return ncclSuccess;}
ncclResult_t ncclMeshUniqueIdOf(uint64_t key,ncclUniqueId *id){
  if(!id || !key)return fail(ncclInvalidArgument,"key");
  memset(id,0,sizeof *id);
  struct uid u={UID_MAGIC,NCCL_VERSION_CODE,key};memcpy(id->internal,&u,sizeof u);
  return ncclSuccess;
}
ncclResult_t ncclGetUniqueId(ncclUniqueId *id){
  uint64_t key=0;
  while(!key)arc4random_buf(&key,sizeof key);
  return ncclMeshUniqueIdOf(key,id);
}
ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm,int nranks,ncclUniqueId commId,int rank,ncclConfig_t *config){
  struct uid u;memcpy(&u,commId.internal,sizeof u);
  if(!comm || u.magic!=UID_MAGIC)return fail(ncclInvalidArgument,"unique id");
  const ncclMeshConfig_t *mesh=config && config->size==sizeof(ncclMeshConfig_t)?(const ncclMeshConfig_t *)config:NULL;
  return comm_make(comm,nranks,u.key,rank,mesh);
}
ncclResult_t ncclCommInitRank(ncclComm_t *comm,int nranks,ncclUniqueId commId,int rank){return ncclCommInitRankConfig(comm,nranks,commId,rank,NULL);}
ncclResult_t ncclCommInitAll(ncclComm_t *comm,int ndev,const int *devlist){(void)comm;(void)ndev;(void)devlist;return fail(ncclInvalidUsage,"ncclCommInitAll: a rank is a process on a node");}
ncclResult_t ncclCommFinalize(ncclComm_t comm){return comm?comm->async:ncclInvalidArgument;}
ncclResult_t ncclCommDestroy(ncclComm_t comm){
  if(!comm)return ncclInvalidArgument;
  free(comm->pairs);free(comm->link_of);free(comm);
  return ncclSuccess;
}
ncclResult_t ncclCommAbort(ncclComm_t comm){return ncclCommDestroy(comm);}
ncclResult_t ncclCommSplit(ncclComm_t comm,int color,int key,ncclComm_t *newcomm,ncclConfig_t *config){
  (void)comm;(void)color;(void)key;(void)newcomm;(void)config;
  return fail(ncclInvalidUsage,"ncclCommSplit: make the communicator with its own ncclMeshConfig_t");
}
const char *ncclGetErrorString(ncclResult_t result){
  static const char *names[]={"no error","unhandled cuda error","unhandled system error","internal error","invalid argument",
    "invalid usage","remote process exited or there was a network error","NCCL operation in progress"};
  return (unsigned)result<sizeof names/sizeof *names?names[result]:"unknown result code";
}
const char *ncclGetLastError(ncclComm_t comm){(void)comm;return last_error;}
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm,ncclResult_t *asyncError){if(!comm || !asyncError)return ncclInvalidArgument;*asyncError=comm->async;return ncclSuccess;}
ncclResult_t ncclCommCount(const ncclComm_t comm,int *count){if(!comm || !count)return ncclInvalidArgument;*count=comm->nranks;return ncclSuccess;}
ncclResult_t ncclCommCuDevice(const ncclComm_t comm,int *device){if(!comm || !device)return ncclInvalidArgument;*device=0;return ncclSuccess;}
ncclResult_t ncclCommUserRank(const ncclComm_t comm,int *rank){if(!comm || !rank)return ncclInvalidArgument;*rank=comm->rank;return ncclSuccess;}
ncclResult_t ncclCommRegister(const ncclComm_t comm,void *buff,size_t size,void **handle){(void)comm;(void)size;if(handle)*handle=buff;return ncclSuccess;}
ncclResult_t ncclCommDeregister(const ncclComm_t comm,void *handle){(void)comm;(void)handle;return ncclSuccess;}
ncclResult_t ncclMemAlloc(void **ptr,size_t size){
  if(!ptr)return ncclInvalidArgument;
  pthread_mutex_lock(&G.lock);
  ncclResult_t r=attach();
  if(r==ncclSuccess)r=session();
  struct allocation *a=r==ncclSuccess?allocate(size,0,0):NULL;
  pthread_mutex_unlock(&G.lock);
  if(!a)return r!=ncclSuccess?r:fail(ncclSystemError,"no room in the window for %zu bytes",size);
  *ptr=a->base;
  return ncclSuccess;
}
ncclResult_t ncclMemFree(void *ptr){
  pthread_mutex_lock(&G.lock);
  struct allocation *a=allocation_of(ptr);
  if(a && !a->owned){nccl_mesh_release(a->buffer);allocation_free(a);}
  pthread_mutex_unlock(&G.lock);
  return a?ncclSuccess:fail(ncclInvalidArgument,"%p is no allocation",ptr);
}
ncclResult_t ncclMeshMemAllocBuffer(void **ptr,size_t size,uint64_t options,void **buffer){
  if(!ptr || !buffer)return ncclInvalidArgument;
  pthread_mutex_lock(&G.lock);
  ncclResult_t r=attach();
  if(r==ncclSuccess)r=session();
  struct allocation *a=r==ncclSuccess?allocate(size,options,1):NULL;
  pthread_mutex_unlock(&G.lock);
  if(!a)return r!=ncclSuccess?r:fail(ncclSystemError,"no room in the window for %zu bytes",size);
  *ptr=a->base;*buffer=a->buffer;
  return ncclSuccess;
}
ncclResult_t ncclMeshMemBuffer(const void *ptr,void **buffer,size_t *offset){
  pthread_mutex_lock(&G.lock);
  struct allocation *a=allocation_of(ptr);
  pthread_mutex_unlock(&G.lock);
  if(!a)return ncclInvalidArgument;
  if(buffer)*buffer=a->buffer;
  if(offset)*offset=(size_t)((const unsigned char *)ptr-a->base);
  return ncclSuccess;
}
ncclResult_t ncclMeshMemUsage(size_t *used,size_t *window){
  pthread_mutex_lock(&G.lock);
  size_t u=0;
  for(struct allocation *a=G.allocations;a;a=a->next)u+=a->bytes;
  size_t w=G.attached?(size_t)region()->wire_pages*region()->pgsz:0;
  pthread_mutex_unlock(&G.lock);
  if(used)*used=u;
  if(window)*window=w;
  return ncclSuccess;
}
ncclResult_t ncclMeshCommMembers(ncclComm_t comm,int *members){
  if(!comm || !members)return ncclInvalidArgument;
  struct hdr *m=region();
  for(int r=0;r<comm->nranks;r++){
    int l=comm->link_of[r];
    members[r]=l<0 || atomic_load(&mesh_links(m)[l].port.phase)!=MESH_STOPPED || !atomic_load(&mesh_links(m)[l].port.code);
  }
  return ncclSuccess;
}
ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op,void *scalar,ncclDataType_t datatype,ncclScalarResidence_t residence,ncclComm_t comm){
  (void)comm;
  if(!op || !scalar || residence!=ncclScalarHostImmediate || (int)datatype<0 || datatype>=ncclNumTypes)return fail(ncclInvalidArgument,"PreMulSum");
  for(int i=0;i<CUSTOM;i++)if(!custom[i].used){
    custom[i].used=1;custom[i].type=datatype;memset(custom[i].scalar,0,8);memcpy(custom[i].scalar,scalar,type_bytes[datatype]);
    *op=(ncclRedOp_t)(ncclNumOps+i);
    return ncclSuccess;
  }
  return fail(ncclInvalidUsage,"too many PreMulSum operators");
}
ncclResult_t ncclRedOpDestroy(ncclRedOp_t op,ncclComm_t comm){
  (void)comm;
  if((int)op<ncclNumOps || (int)op-ncclNumOps>=CUSTOM)return ncclInvalidArgument;
  custom[op-ncclNumOps].used=0;
  return ncclSuccess;
}
ncclResult_t ncclGroupStart(void){pending.depth++;return ncclSuccess;}
ncclResult_t ncclGroupEnd(void){
  if(!pending.depth)return fail(ncclInvalidUsage,"ncclGroupEnd without ncclGroupStart");
  if(--pending.depth)return ncclSuccess;
  int n=pending.n;pending.n=0;
  ncclResult_t r=ncclSuccess;
  /* consecutive calls on one stream are one program */
  for(int i=0;i<n && r==ncclSuccess;){
    int j=i+1;
    while(j<n && pending.calls[j].stream==pending.calls[i].stream)j++;
    r=run(pending.calls+i,j-i);
    i=j;
  }
  for(int i=0;i<n;i++)free(pending.calls[i].segments);
  return r;
}
static ncclResult_t collective(int kind,const void *send,void *recv,size_t count,ncclDataType_t type,int op,int root,ncclComm_t comm,
                               cudaStream_t stream,const ncclCollConfig_t *config,const size_t *counts){
  if(!comm)return fail(ncclInvalidArgument,"no communicator");
  struct call k={.kind=kind,.send=send,.recv=recv,.count=count,.type=type,.op=op,.root=root,.comm=comm,.stream=stream};
  if(config && config->algSelection && config->algSelection!=NCCL_CONFIG_UNDEF_PTR){
    static const char *names[]={"direct","ring","tree","binomial"};
    for(int a=0;a<4;a++)if(strstr(config->algSelection,names[a]))k.how|=1u<<a;
  }
  uint64_t segments[256];
  if(counts){
    if(comm->nranks>256)return fail(ncclInvalidUsage,"V-forms of more than 256 ranks");
    int equal=1;
    for(int r=0;r<comm->nranks;r++){segments[r]=counts[r];equal&=counts[r]==counts[0];if(!counts[r])return fail(ncclInvalidArgument,"a segment of 0");}
    if(equal)k.count=counts[0];else k.segments=segments;
  }
  if(!count && !counts)return ncclSuccess;
  return issue(k);
}
ncclResult_t ncclAllReduce(const void *s,void *r,size_t n,ncclDataType_t t,ncclRedOp_t op,ncclComm_t c,cudaStream_t st){return collective(MESH_ALLREDUCE,s,r,n,t,op,0,c,st,NULL,NULL);}
ncclResult_t ncclAllReduceConfig(const void *s,void *r,size_t n,ncclDataType_t t,ncclRedOp_t op,ncclComm_t c,cudaStream_t st,const ncclCollConfig_t *k){return collective(MESH_ALLREDUCE,s,r,n,t,op,0,c,st,k,NULL);}
ncclResult_t ncclReduce(const void *s,void *r,size_t n,ncclDataType_t t,ncclRedOp_t op,int root,ncclComm_t c,cudaStream_t st){return collective(MESH_REDUCE,s,r,n,t,op,root,c,st,NULL,NULL);}
ncclResult_t ncclReduceConfig(const void *s,void *r,size_t n,ncclDataType_t t,ncclRedOp_t op,int root,ncclComm_t c,cudaStream_t st,const ncclCollConfig_t *k){return collective(MESH_REDUCE,s,r,n,t,op,root,c,st,k,NULL);}
ncclResult_t ncclBroadcast(const void *s,void *r,size_t n,ncclDataType_t t,int root,ncclComm_t c,cudaStream_t st){return collective(MESH_BROADCAST,s,r,n,t,0,root,c,st,NULL,NULL);}
ncclResult_t ncclBroadcastConfig(const void *s,void *r,size_t n,ncclDataType_t t,int root,ncclComm_t c,cudaStream_t st,const ncclCollConfig_t *k){return collective(MESH_BROADCAST,s,r,n,t,0,root,c,st,k,NULL);}
ncclResult_t ncclBcast(void *b,size_t n,ncclDataType_t t,int root,ncclComm_t c,cudaStream_t st){return collective(MESH_BROADCAST,b,b,n,t,0,root,c,st,NULL,NULL);}
ncclResult_t ncclReduceScatter(const void *s,void *r,size_t n,ncclDataType_t t,ncclRedOp_t op,ncclComm_t c,cudaStream_t st){return collective(MESH_REDUCE_SCATTER,s,r,n,t,op,0,c,st,NULL,NULL);}
ncclResult_t ncclReduceScatterConfig(const void *s,void *r,size_t n,ncclDataType_t t,ncclRedOp_t op,ncclComm_t c,cudaStream_t st,const ncclCollConfig_t *k){return collective(MESH_REDUCE_SCATTER,s,r,n,t,op,0,c,st,k,NULL);}
ncclResult_t ncclAllGather(const void *s,void *r,size_t n,ncclDataType_t t,ncclComm_t c,cudaStream_t st){return collective(MESH_ALLGATHER,s,r,n,t,0,0,c,st,NULL,NULL);}
ncclResult_t ncclAllGatherConfig(const void *s,void *r,size_t n,ncclDataType_t t,ncclComm_t c,cudaStream_t st,const ncclCollConfig_t *k){return collective(MESH_ALLGATHER,s,r,n,t,0,0,c,st,k,NULL);}
ncclResult_t ncclMeshAllGatherV(const void *s,void *r,const size_t *counts,ncclDataType_t t,ncclComm_t c,cudaStream_t st){return collective(MESH_ALLGATHER,s,r,0,t,0,0,c,st,NULL,counts);}
ncclResult_t ncclMeshReduceScatterV(const void *s,void *r,const size_t *counts,ncclDataType_t t,ncclRedOp_t op,ncclComm_t c,cudaStream_t st){return collective(MESH_REDUCE_SCATTER,s,r,0,t,op,0,c,st,NULL,counts);}
ncclResult_t ncclSend(const void *s,size_t n,ncclDataType_t t,int peer,ncclComm_t c,cudaStream_t st){
  if(!n)return ncclSuccess;
  return issue((struct call){.kind=K_SEND,.send=s,.count=n,.type=t,.peer=peer,.comm=c,.stream=st});
}
ncclResult_t ncclRecv(void *r,size_t n,ncclDataType_t t,int peer,ncclComm_t c,cudaStream_t st){
  if(!n)return ncclSuccess;
  return issue((struct call){.kind=K_RECV,.recv=r,.count=n,.type=t,.peer=peer,.comm=c,.stream=st});
}
/* A rank's own block of an all-to-all, gather or scatter: a copy on the GPU, in its group's program. */
static ncclResult_t local(const void *s,void *r,size_t n,ncclDataType_t t,ncclComm_t c,cudaStream_t st){
  if(!n || s==r)return ncclSuccess;
  return issue((struct call){.kind=K_LOCAL,.send=s,.recv=r,.count=n,.type=t,.comm=c,.stream=st});
}
/* NCCL 2.32's lowering of all-to-all, gather and scatter to grouped point-to-point calls. */
ncclResult_t ncclAlltoAll(const void *s,void *r,size_t n,ncclDataType_t t,ncclComm_t c,cudaStream_t st){
  if(!c)return ncclInvalidArgument;
  const size_t e=type_bytes[t];
  ncclGroupStart();
  for(int q=0;q<c->nranks;q++){
    if(q==c->rank)continue;
    ncclSend((const char *)s+(size_t)q*n*e,n,t,q,c,st);
    ncclRecv((char *)r+(size_t)q*n*e,n,t,q,c,st);
  }
  local((const char *)s+(size_t)c->rank*n*e,(char *)r+(size_t)c->rank*n*e,n,t,c,st);
  return ncclGroupEnd();
}
ncclResult_t ncclGather(const void *s,void *r,size_t n,ncclDataType_t t,int root,ncclComm_t c,cudaStream_t st){
  if(!c)return ncclInvalidArgument;
  const size_t e=type_bytes[t];
  ncclGroupStart();
  if(c->rank!=root)ncclSend(s,n,t,root,c,st);
  else {
    for(int q=0;q<c->nranks;q++)if(q!=root)ncclRecv((char *)r+(size_t)q*n*e,n,t,q,c,st);
    local(s,(char *)r+(size_t)root*n*e,n,t,c,st);
  }
  return ncclGroupEnd();
}
ncclResult_t ncclScatter(const void *s,void *r,size_t n,ncclDataType_t t,int root,ncclComm_t c,cudaStream_t st){
  if(!c)return ncclInvalidArgument;
  const size_t e=type_bytes[t];
  ncclGroupStart();
  if(c->rank==root){
    for(int q=0;q<c->nranks;q++)if(q!=root)ncclSend((const char *)s+(size_t)q*n*e,n,t,q,c,st);
    local((const char *)s+(size_t)root*n*e,r,n,t,c,st);
  }
  else ncclRecv(r,n,t,root,c,st);
  return ncclGroupEnd();
}
ncclResult_t ncclAlltoAllConfig(const void *s,void *r,size_t n,ncclDataType_t t,ncclComm_t c,cudaStream_t st,const ncclCollConfig_t *k){(void)k;return ncclAlltoAll(s,r,n,t,c,st);}
ncclResult_t ncclGatherConfig(const void *s,void *r,size_t n,ncclDataType_t t,int root,ncclComm_t c,cudaStream_t st,const ncclCollConfig_t *k){(void)k;return ncclGather(s,r,n,t,root,c,st);}
ncclResult_t ncclScatterConfig(const void *s,void *r,size_t n,ncclDataType_t t,int root,ncclComm_t c,cudaStream_t st,const ncclCollConfig_t *k){(void)k;return ncclScatter(s,r,n,t,root,c,st);}

/* ---- streams ---- */
ncclResult_t ncclMeshStreamCreate(cudaStream_t *stream,void *queue){
  (void)queue;
  if(!stream)return ncclInvalidArgument;
  pthread_mutex_lock(&G.lock);
  ncclResult_t r=attach();
  if(r==ncclSuccess)r=session();
  uint64_t *done=NULL;
  if(r==ncclSuccess){
    if(G.done_next>=4096)r=fail(ncclInvalidUsage,"too many streams");
    else done=(uint64_t *)((unsigned char *)region()+G.done_at+8*G.done_next++);
  }
  pthread_mutex_unlock(&G.lock);
  if(r!=ncclSuccess)return r;
  struct ncclMeshStream *s=calloc(1,sizeof *s);
  if(!s)return fail(ncclSystemError,"allocation");
  s->done=done;*done=0;s->refs=1;
  *stream=s;
  return ncclSuccess;
}
ncclResult_t ncclMeshStreamDestroy(cudaStream_t stream){
  if(!stream)return ncclInvalidArgument;
  ncclMeshStreamSynchronize(stream);
  free(stream);
  return ncclSuccess;
}
ncclResult_t ncclMeshStreamEncode(cudaStream_t stream,void *commandBuffer){
  if(!stream || !commandBuffer)return ncclInvalidArgument;
  ncclResult_t result=ncclSuccess;
  pthread_mutex_lock(&G.lock);reap();pthread_mutex_unlock(&G.lock);
  while(stream->pending && result==ncclSuccess){
    struct kept *x=stream->pending;stream->pending=x->next;
    struct ran *r=calloc(1,sizeof *r);
    if(!r)return fail(ncclSystemError,"allocation");
    stream->value++;
    result=encode_group(commandBuffer,x->calls,x->n,stream->done,stream->value,&r->scratch,&r->nscratch,&r->held,&r->nheld);
    r->done=stream->done;r->value=stream->value;
    pthread_mutex_lock(&G.lock);r->next=running;running=r;pthread_mutex_unlock(&G.lock);
    for(int i=0;i<x->n;i++)free(x->calls[i].segments);
    free(x->calls);free(x);
  }
  return result;
}
ncclResult_t ncclMeshStreamQuery(cudaStream_t stream){
  if(!stream)return ncclInvalidArgument;
  return atomic_load_explicit((_Atomic uint64_t *)stream->done,memory_order_acquire)>=stream->value?ncclSuccess:ncclInProgress;
}
ncclResult_t ncclMeshStreamSynchronize(cudaStream_t stream){
  if(!stream)return ncclInvalidArgument;
  if(stream->pending)return fail(ncclInvalidUsage,"a stream's kept programs are encoded (ncclMeshStreamEncode) before it is synchronized");
  while(atomic_load_explicit((_Atomic uint64_t *)stream->done,memory_order_acquire)<stream->value)sched_yield();
  pthread_mutex_lock(&G.lock);reap();pthread_mutex_unlock(&G.lock);
  return ncclSuccess;
}
