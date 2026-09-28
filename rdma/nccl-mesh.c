#include "nccl.h"
#include "mesh-net.h"
#include "mesh-collective.h"
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

/* libnccl-mesh (nccl.h): NCCL's calls as calls of the mesh's one collective planner, run on the bridge's
   communicator sessions (mesh-net.h).  A communicator is a clique key (the unique id's), this rank,
   the link map over its ranks, and two connections each way with every rank the map links it to:
   channel 0 carries the collectives' steps, channel 1 the point-to-point calls, so that each keeps
   NCCL's matching rule (the k-th send on a connection is the peer's k-th receive on it) whatever the
   other does.  A collective is mesh_collective_choose's algorithm, and this rank's steps from
   mesh_collective_plan run in plan order: a SEND an isend of the operand's piece, a COPY an irecv into
   the operand, a REDUCE an irecv into scratch and its combine into the operand, each piece written
   only once no isend still reads it.  A group is split by communicator; each communicator's worker
   thread runs its part in issue order, after its streams' prior work. */

enum { CH_COLL, CH_P2P, CHANNELS };
enum { K_SEND=MESH_ALLGATHER+1, K_RECV };
#define UID_MAGIC 0x4d4e434cu
struct uid { uint32_t magic,version; uint64_t key; };
_Static_assert(sizeof(struct uid)<=NCCL_UNIQUE_ID_BYTES,"unique id");

struct peer { int dev; void *send[CHANNELS],*recv[CHANNELS]; };
struct op_entry { int used; ncclDataType_t type; unsigned char scalar[8]; };
struct call {
  int kind; const void *send; void *recv; size_t count; ncclDataType_t type; int op,root,peer;
  struct ncclComm *comm; struct ncclMeshStream *stream; uint32_t how; int force;
  /* resolved when the group ends */
  int combine,premultiply,postdivide; unsigned char scalar[8];
  uint64_t elements; struct mesh_collective chosen; struct mesh_step *steps; uint32_t nsteps;
};
struct mark { struct ncclMeshStream *stream; uint64_t wait,done; };
struct launch { pthread_mutex_t lock; pthread_cond_t cond; int items,sync,nmarks; ncclResult_t result; struct mark *marks; };
struct item { struct call *calls; int n; struct launch *launch; struct item *next; };
struct ncclComm {
  uint64_t key; int rank,nranks;
  struct mesh_link_map map; double alpha,beta;
  void *net; struct peer *peers;
  uint64_t splits;
  struct op_entry *ops; int nops;
  unsigned char *stage; size_t stage_bytes; void *stage_mh;
  pthread_t worker; int started,stopping,finalized;
  pthread_mutex_t lock; pthread_cond_t cond;
  struct item *head,*tail; int busy;
  _Atomic int aborting,broken,async;
  char error[512];
};

__attribute__((visibility("hidden"))) void *nccl_mesh_event_create(void *queue);
__attribute__((visibility("hidden"))) void nccl_mesh_event_release(void *event);
__attribute__((visibility("hidden"))) uint64_t nccl_mesh_event_value(void *event);
__attribute__((visibility("hidden"))) void nccl_mesh_event_signal(void *event,uint64_t value);
__attribute__((visibility("hidden"))) int nccl_mesh_queue_signal(void *queue,void *event,uint64_t value);
__attribute__((visibility("hidden"))) int nccl_mesh_queue_wait(void *queue,void *event,uint64_t value);

/* ---- errors ---- */
static _Thread_local char last_error[512];
static void fail_message(struct ncclComm *c,const char *format,...){
  char text[512];va_list list;va_start(list,format);vsnprintf(text,sizeof text,format,list);va_end(list);
  snprintf(last_error,sizeof last_error,"%s",text);
  if(c){pthread_mutex_lock(&c->lock);snprintf(c->error,sizeof c->error,"%s",text);pthread_mutex_unlock(&c->lock);}
  if(getenv("MESH_NCCL_DEBUG"))fprintf(stderr,"nccl-mesh: %s\n",text);
}
#define FAIL(c,result,...) (fail_message((c),__VA_ARGS__),(ncclResult_t)(result))
static ncclResult_t net_failure(struct ncclComm *c,int result,const char *what){
  int error=mesh_net_error();
  return FAIL(c,result?result:ncclSystemError,"%s: %s (%s)",what,ncclGetErrorString((ncclResult_t)result),strerror(error));
}

static uint64_t now_ns(void){return clock_gettime_nsec_np(CLOCK_MONOTONIC);}
static uint64_t deadline_after(void){
  const char *text=getenv("MESH_NCCL_TIMEOUT");double seconds=text?atof(text):0;
  return now_ns()+(uint64_t)((seconds>0?seconds:300)*1e9);
}
static uint64_t mix(uint64_t x){
  x+=UINT64_C(0x9e3779b97f4a7c15);
  x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
  return x^(x>>31);
}

/* ---- datatypes and operators ---- */
static const size_t type_bytes[ncclNumTypes]={1,1,4,4,8,8,2,4,8,2,1,1};
static int integral(ncclDataType_t t){return t<=ncclUint64;}

/* bfloat16 and the two fp8 formats: decoded to double, and a double rounded to nearest even into
   them (one rounding); fp8 saturates to its largest finite value as NCCL's __NV_SATFINITE does. */
struct small { int e,m,fn,saturate; };
static const struct small BF16={8,7,0,0},E4M3={4,3,1,1},E5M2={5,2,0,1};
static double small_decode(uint32_t bits,struct small f){
  uint32_t top=(1u<<f.e)-1,exponent=(bits>>f.m)&top,mantissa=bits&((1u<<f.m)-1);
  int bias=(1<<(f.e-1))-1;double sign=(bits>>(f.e+f.m))&1?-1.0:1.0;
  if(!f.fn && exponent==top)return mantissa?NAN:sign*INFINITY;
  if(f.fn && exponent==top && mantissa==(1u<<f.m)-1)return NAN;
  if(!exponent)return sign*ldexp((double)mantissa,1-bias-f.m);
  return sign*ldexp((double)(mantissa|(1u<<f.m)),(int)exponent-bias-f.m);
}
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
static const struct small *small_of(ncclDataType_t t){return t==ncclBfloat16?&BF16:t==ncclFloat8e4m3?&E4M3:t==ncclFloat8e5m2?&E5M2:NULL;}

/* dst = dst op src, every element one operation of the type (NCCL's FuncSum/FuncProd/FuncMax/FuncMin):
   integers wrap; float and double natively; half, bfloat16 and fp8 exactly in double, rounded once. */
#define INTEGER_LOOP(T,U) { T *d=dst;const T *s=src; switch(op){ \
  case ncclSum: for(size_t i=0;i<n;i++)d[i]=(T)((U)d[i]+(U)s[i]); break; \
  case ncclProd: for(size_t i=0;i<n;i++)d[i]=(T)((U)d[i]*(U)s[i]); break; \
  case ncclMax: for(size_t i=0;i<n;i++)if(s[i]>d[i])d[i]=s[i]; break; \
  default: for(size_t i=0;i<n;i++)if(s[i]<d[i])d[i]=s[i]; } break; }
/* max and min of floating values: a NaN propagates, and -0 orders below +0 */
#define MAX_OF(a,b) (isnan(b) || (b)>(a) || ((b)==(a) && !signbit(b) && signbit(a))?(b):(a))
#define MIN_OF(a,b) (isnan(b) || (b)<(a) || ((b)==(a) && signbit(b) && !signbit(a))?(b):(a))
#define NATIVE_LOOP(T) { T *d=dst;const T *s=src; switch(op){ \
  case ncclSum: for(size_t i=0;i<n;i++)d[i]=d[i]+s[i]; break; \
  case ncclProd: for(size_t i=0;i<n;i++)d[i]=d[i]*s[i]; break; \
  case ncclMax: for(size_t i=0;i<n;i++)d[i]=MAX_OF(d[i],s[i]); break; \
  default: for(size_t i=0;i<n;i++)d[i]=MIN_OF(d[i],s[i]); } break; }
static double apply(int op,double a,double b){
  switch(op){case ncclSum:return a+b;case ncclProd:return a*b;case ncclMax:return MAX_OF(a,b);default:return MIN_OF(a,b);}
}
static void combine(void *dst,const void *src,size_t n,ncclDataType_t t,int op){
  switch(t){
  case ncclInt8: INTEGER_LOOP(int8_t,uint32_t)
  case ncclUint8: INTEGER_LOOP(uint8_t,uint32_t)
  case ncclInt32: INTEGER_LOOP(int32_t,uint32_t)
  case ncclUint32: INTEGER_LOOP(uint32_t,uint32_t)
  case ncclInt64: INTEGER_LOOP(int64_t,uint64_t)
  case ncclUint64: INTEGER_LOOP(uint64_t,uint64_t)
  case ncclFloat32: NATIVE_LOOP(float)
  case ncclFloat64: NATIVE_LOOP(double)
  case ncclFloat16: { _Float16 *d=dst;const _Float16 *s=src;
    for(size_t i=0;i<n;i++)d[i]=(_Float16)apply(op,(double)d[i],(double)s[i]); break; }
  default: { const struct small *f=small_of(t);
    if(type_bytes[t]==2){uint16_t *d=dst;const uint16_t *s=src;
      for(size_t i=0;i<n;i++)d[i]=(uint16_t)small_encode(apply(op,small_decode(d[i],*f),small_decode(s[i],*f)),*f);}
    else{uint8_t *d=dst;const uint8_t *s=src;
      for(size_t i=0;i<n;i++)d[i]=(uint8_t)small_encode(apply(op,small_decode(d[i],*f),small_decode(s[i],*f)),*f);}
  }
  }
}
/* dst = src x scalar, one operation of the type (a premultiplication: ncclAvg on floating types,
   PreMulSum). */
static void premultiply(void *dst,const void *src,size_t n,ncclDataType_t t,const unsigned char *scalar){
  switch(t){
#define SCALE_INT(T,U) { T *d=dst;const T *s=src;T k;memcpy(&k,scalar,sizeof k); for(size_t i=0;i<n;i++)d[i]=(T)((U)s[i]*(U)k); break; }
  case ncclInt8: SCALE_INT(int8_t,uint32_t)
  case ncclUint8: SCALE_INT(uint8_t,uint32_t)
  case ncclInt32: SCALE_INT(int32_t,uint32_t)
  case ncclUint32: SCALE_INT(uint32_t,uint32_t)
  case ncclInt64: SCALE_INT(int64_t,uint64_t)
  case ncclUint64: SCALE_INT(uint64_t,uint64_t)
  case ncclFloat32: { float *d=dst;const float *s=src;float k;memcpy(&k,scalar,sizeof k); for(size_t i=0;i<n;i++)d[i]=s[i]*k; break; }
  case ncclFloat64: { double *d=dst;const double *s=src;double k;memcpy(&k,scalar,sizeof k); for(size_t i=0;i<n;i++)d[i]=s[i]*k; break; }
  case ncclFloat16: { _Float16 *d=dst;const _Float16 *s=src;_Float16 k;memcpy(&k,scalar,sizeof k);
    for(size_t i=0;i<n;i++)d[i]=(_Float16)((double)s[i]*(double)k); break; }
  default: { const struct small *f=small_of(t);
    if(type_bytes[t]==2){uint16_t *d=dst;const uint16_t *s=src;uint16_t k;memcpy(&k,scalar,sizeof k);double v=small_decode(k,*f);
      for(size_t i=0;i<n;i++)d[i]=(uint16_t)small_encode(small_decode(s[i],*f)*v,*f);}
    else{uint8_t *d=dst;const uint8_t *s=src;double v=small_decode(scalar[0],*f);
      for(size_t i=0;i<n;i++)d[i]=(uint8_t)small_encode(small_decode(s[i],*f)*v,*f);}
  }
#undef SCALE_INT
  }
}
/* ncclAvg on integer types: the wrapped sum divided by n, truncating toward zero (NCCL's
   FuncSumPostDiv::divide: the magnitude divided, the sign kept). */
static void postdivide(void *data,size_t n,ncclDataType_t t,int nranks){
  switch(t){
#define DIVIDE_SIGNED(T,U) { T *d=data; for(size_t i=0;i<n;i++){U u=d[i]<0?(U)0-(U)d[i]:(U)d[i];U q=u/(U)nranks;d[i]=d[i]<0?(T)((U)0-q):(T)q;} break; }
#define DIVIDE_UNSIGNED(T) { T *d=data; for(size_t i=0;i<n;i++)d[i]=(T)(d[i]/(T)nranks); break; }
  case ncclInt8: DIVIDE_SIGNED(int8_t,uint8_t)
  case ncclInt32: DIVIDE_SIGNED(int32_t,uint32_t)
  case ncclInt64: DIVIDE_SIGNED(int64_t,uint64_t)
  case ncclUint8: DIVIDE_UNSIGNED(uint8_t)
  case ncclUint32: DIVIDE_UNSIGNED(uint32_t)
  case ncclUint64: DIVIDE_UNSIGNED(uint64_t)
  default: break;
#undef DIVIDE_SIGNED
#undef DIVIDE_UNSIGNED
  }
}
/* 1/nranks rounded to the type (ncclAvg's premultiplier on floating types: 1/n as a double, then
   the type). */
static void reciprocal(unsigned char *scalar,ncclDataType_t t,int nranks){
  double v=1.0/nranks;
  memset(scalar,0,8);
  if(t==ncclFloat32){float k=(float)v;memcpy(scalar,&k,4);}
  else if(t==ncclFloat64)memcpy(scalar,&v,8);
  else if(t==ncclFloat16){_Float16 k=(_Float16)v;memcpy(scalar,&k,2);}
  else{uint32_t bits=small_encode(v,*small_of(t));memcpy(scalar,&bits,type_bytes[t]);}
}

/* ---- the link map over a communicator's ranks ---- */
static int linked(const struct mesh_link_map *map,int a,int b){
  if(a==b || a<0 || b<0 || (uint32_t)a>=map->nodes || (uint32_t)b>=map->nodes)return 0;
  if(map->kind==MESH_LINKS_MESH)return 1;
  for(uint32_t l=0;l<map->links;l++)
    if(((int)map->link[l][0]==a && (int)map->link[l][1]==b) || ((int)map->link[l][0]==b && (int)map->link[l][1]==a))return 1;
  return 0;
}
/* MESH_LINKS (a link map file over the ranks: its first link line's third and fourth fields the
   alpha-beta cost) or, without it, every pair of the nranks ranks linked at no cost. */
static ncclResult_t map_for(struct ncclComm *c){
  const char *path=getenv("MESH_LINKS");
  if(!path || !*path){c->map=(struct mesh_link_map){.kind=MESH_LINKS_MESH,.nodes=(uint32_t)c->nranks};return ncclSuccess;}
  int error=mesh_link_map_read(path,&c->map);
  if(error)return FAIL(c,ncclInvalidArgument,"MESH_LINKS %s: %s",path,strerror(error));
  if(c->map.nodes!=(uint32_t)c->nranks)return FAIL(c,ncclInvalidUsage,"MESH_LINKS %s has %u nodes, the communicator %d ranks",path,c->map.nodes,c->nranks);
  FILE *file=fopen(path,"r");char line[256];unsigned a,b;double alpha,beta;
  while(file && fgets(line,sizeof line,file))if(sscanf(line,"%u %u %lf %lf",&a,&b,&alpha,&beta)==4){c->alpha=alpha;c->beta=beta;break;}
  if(file)fclose(file);
  return ncclSuccess;
}

/* ---- connections ---- */
static uint64_t listen_key(uint64_t key,int channel,int from,int to){
  return mix(key^mix(((uint64_t)channel<<48)|((uint64_t)(uint32_t)from<<24)|(uint32_t)to))|1;
}
static void close_peers(struct ncclComm *c){
  for(int p=0;c->peers && p<c->nranks;p++)for(int ch=0;ch<CHANNELS;ch++){
    if(c->peers[p].send[ch])mesh_net_close_send(c->peers[p].send[ch]);
    if(c->peers[p].recv[ch])mesh_net_close_recv(c->peers[p].recv[ch]);
    c->peers[p].send[ch]=c->peers[p].recv[ch]=NULL;
  }
}
/* Both channels each way with every rank the map links this one to, on whichever of the bridge's links
   reaches it: a listen on every link for each such rank's connections (keys both ends derive from
   the clique key), a connect on every link to each such rank's; the one its bridge accepts is the
   link.  The probing connects that met no listen are withdrawn with their context. */
static ncclResult_t connect_peers(struct ncclComm *c){
  c->peers=calloc((size_t)c->nranks,sizeof *c->peers);
  if(!c->peers)return FAIL(c,ncclSystemError,"peer table allocation");
  for(int p=0;p<c->nranks;p++)c->peers[p].dev=-1;
  int wanted=0;
  for(int p=0;p<c->nranks;p++)wanted+=linked(&c->map,c->rank,p);
  if(!wanted)return ncclSuccess;
  struct mesh_link_view views[64];uint32_t node;
  int links=mesh_observe(getenv("MESH_REGION"),views,64,&node);
  if(links<0)return FAIL(c,ncclSystemError,"the bridge region %s: %s",getenv("MESH_REGION")?getenv("MESH_REGION"):"/mesh0",strerror(-links));
  if(!links)return FAIL(c,ncclInvalidUsage,"this node's bridge has no links, and rank %d is linked to %d ranks",c->rank,wanted);
  if(links>64)links=64;
  void *probe=NULL;
  int result=mesh_net_init(&probe,c->key,NULL,NULL,NULL);
  if(result)return net_failure(c,result,"mesh_net_init");
  size_t count=(size_t)c->nranks*CHANNELS*(size_t)links;
  void **listens=calloc(count,sizeof *listens);
  struct mesh_net_handle *targets=calloc(count,sizeof *targets);
  ncclResult_t status=listens && targets?ncclSuccess:FAIL(c,ncclSystemError,"connection table allocation");
  for(int p=0;p<c->nranks && !status;p++)if(linked(&c->map,c->rank,p))for(int ch=0;ch<CHANNELS && !status;ch++)for(int l=0;l<links && !status;l++){
    size_t at=((size_t)p*CHANNELS+(size_t)ch)*(size_t)links+(size_t)l;
    unsigned char handle[MESH_NET_HANDLE_BYTES]={0};
    struct mesh_net_handle named={MESH_NET_HANDLE_MAGIC,0,listen_key(c->key,ch,p,c->rank)};
    memcpy(handle,&named,sizeof named);
    if((result=mesh_net_listen(c->net,l,handle,listens+at)))status=net_failure(c,result,"mesh_net_listen");
    targets[at]=(struct mesh_net_handle){MESH_NET_HANDLE_MAGIC,views[l].peer,listen_key(c->key,ch,c->rank,p)};
  }
  uint64_t deadline=deadline_after();
  for(int done=0;!status && !done;){
    done=1;
    for(int p=0;p<c->nranks && !status;p++)if(linked(&c->map,c->rank,p))for(int ch=0;ch<CHANNELS && !status;ch++){
      struct peer *e=c->peers+p;
      for(int l=0;l<links && !e->send[ch] && !status;l++){
        size_t at=((size_t)p*CHANNELS+(size_t)ch)*(size_t)links+(size_t)l;
        unsigned char handle[MESH_NET_HANDLE_BYTES]={0};memcpy(handle,targets+at,sizeof targets[at]);
        void *send=NULL;
        if((result=mesh_net_connect(probe,l,handle,&send,NULL)))status=net_failure(c,result,"mesh_net_connect");
        else if(send){e->send[ch]=send;if(e->dev<0)e->dev=l;}
      }
      for(int l=0;l<links && !e->recv[ch] && !status;l++){
        size_t at=((size_t)p*CHANNELS+(size_t)ch)*(size_t)links+(size_t)l;
        void *recv=NULL;
        if((result=mesh_net_accept(listens[at],&recv,NULL)))status=net_failure(c,result,"mesh_net_accept");
        else if(recv)e->recv[ch]=recv;
      }
      done&=e->send[ch] && e->recv[ch];
    }
    if(!done && !status){
      if(now_ns()>deadline){
        int missing=-1;
        for(int p=0;p<c->nranks && missing<0;p++)if(linked(&c->map,c->rank,p))
          for(int ch=0;ch<CHANNELS;ch++)if(!c->peers[p].send[ch] || !c->peers[p].recv[ch])missing=p;
        status=FAIL(c,ncclTimeout,"rank %d: no connection with rank %d by the deadline (MESH_NCCL_TIMEOUT): no bridge link reaches it, or it has not joined",c->rank,missing);
      } else usleep(200);
    }
  }
  for(size_t i=0;listens && i<count;i++)if(listens[i])mesh_net_close_listen(listens[i]);
  free(listens);free(targets);
  mesh_net_finalize(probe);
  return status;
}

/* ---- window memory ---- */
static pthread_mutex_t global_lock=PTHREAD_MUTEX_INITIALIZER;
static void *global_net;
static ncclResult_t global_attach(void){
  pthread_mutex_lock(&global_lock);
  int result=global_net?0:mesh_net_init(&global_net,0,NULL,NULL,NULL);
  pthread_mutex_unlock(&global_lock);
  return result?net_failure(NULL,result,"mesh_net_init (the bridge region MESH_REGION)"):ncclSuccess;
}
static int in_window(const void *pointer,size_t bytes,void **mhandle){
  *mhandle=NULL;
  return bytes && !mesh_net_reg_mr(NULL,(void *)pointer,bytes,MESH_NET_PTR_HOST,mhandle);
}
static ncclResult_t stage_reserve(struct ncclComm *c,size_t bytes){
  if(bytes<=c->stage_bytes)return ncclSuccess;
  if(c->stage){mesh_net_dereg_mr(NULL,c->stage_mh);mesh_net_mem_free(c->stage);c->stage=NULL;c->stage_bytes=0;}
  size_t size=(bytes+(1u<<20)-1)&~(size_t)((1u<<20)-1);
  void *pointer=NULL;
  int result=mesh_net_mem_alloc(&pointer,size);
  if(result)return net_failure(c,result,"staging memory from the bridge's registered window");
  if((result=mesh_net_reg_mr(NULL,pointer,size,MESH_NET_PTR_HOST,&c->stage_mh))){mesh_net_mem_free(pointer);return net_failure(c,result,"mesh_net_reg_mr");}
  c->stage=pointer;c->stage_bytes=size;
  return ncclSuccess;
}
static size_t aligned(size_t bytes){return (bytes+4095)&~(size_t)4095;}

/* ---- requests ---- */
struct pending { size_t lo,hi; void *request; };
static int stopped(struct ncclComm *c,uint64_t deadline){return atomic_load(&c->aborting) || now_ns()>deadline;}
static ncclResult_t stop_reason(struct ncclComm *c,const char *what){
  return atomic_load(&c->aborting)?FAIL(c,ncclInvalidUsage,"%s: the communicator was aborted",what):
    FAIL(c,ncclTimeout,"%s: not done by the deadline (MESH_NCCL_TIMEOUT)",what);
}
/* Reaps every pending isend that is done (its ring slot free again). */
static ncclResult_t reap(struct ncclComm *c,struct pending *sends,int *count){
  for(int i=0;i<*count;){
    int done=0,size=0,result=mesh_net_test(sends[i].request,&done,&size);
    if(result)return net_failure(c,result,"an isend");
    if(done){sends[i]=sends[--*count];continue;}
    i++;
  }
  return ncclSuccess;
}
static ncclResult_t await(struct ncclComm *c,void *request,size_t bytes,struct pending *sends,int *count,uint64_t deadline,const char *what){
  for(;;){
    int done=0,size=0,result=mesh_net_test(request,&done,&size);
    if(result)return net_failure(c,result,what);
    if(done){
      if(bytes<=INT32_MAX && (size_t)size!=bytes)return FAIL(c,ncclInvalidUsage,"%s: %d bytes arrived, %zu expected (the peer's count or datatype differs)",what,size,bytes);
      return ncclSuccess;
    }
    if(sends){ncclResult_t r=reap(c,sends,count);if(r)return r;}
    if(stopped(c,deadline))return stop_reason(c,what);
    sched_yield();
  }
}
/* An isend or irecv posted, waiting while its connection's request ring is full. */
static ncclResult_t post(struct ncclComm *c,int send,void *comm,void *data,size_t bytes,void *mhandle,
  struct pending *sends,int *count,uint64_t deadline,void **request){
  for(*request=NULL;;){
    int result=send?mesh_net_isend(comm,data,bytes,0,mhandle,NULL,request):
      mesh_net_irecv(comm,1,&data,&bytes,(int[]){0},&mhandle,NULL,request);
    if(result)return net_failure(c,result,send?"mesh_net_isend":"mesh_net_irecv");
    if(*request)return ncclSuccess;
    if(sends){ncclResult_t r=reap(c,sends,count);if(r)return r;}
    if(stopped(c,deadline))return stop_reason(c,send?"an isend":"an irecv");
    sched_yield();
  }
}

/* ---- one collective ---- */
/* The operand: in recvbuff itself where recvbuff lies in the registered window and holds the whole
   result (all-reduce, broadcast, all-gather, a reduce's root), else window memory; stage in (the
   contribution, premultiplied where the operator says), this rank's plan, stage out (the result,
   post-divided where the operator says). */
static ncclResult_t run_collective(struct ncclComm *c,struct call *k,unsigned char *stage,uint64_t deadline){
  const size_t e=type_bytes[k->type],bytes=(size_t)k->elements*e,part=k->count*e;
  const int r=c->rank,root=k->root,what=k->kind;
  unsigned char *operand=stage,*scratch=stage+aligned(bytes);
  void *operand_mh=c->stage_mh,*own=NULL;
  if(k->recv && (what==MESH_ALLREDUCE || what==MESH_BROADCAST || what==MESH_ALLGATHER || (what==MESH_REDUCE && r==root)) &&
     in_window(k->recv,bytes,&own)){operand=k->recv;operand_mh=own;}
  const unsigned char *send=k->send;
  if(what==MESH_ALLREDUCE || what==MESH_REDUCE || what==MESH_REDUCE_SCATTER){
    if(k->premultiply)premultiply(operand,send,k->elements,k->type,k->scalar);
    else if(operand!=send)memmove(operand,send,bytes);
  } else if(what==MESH_BROADCAST){if(r==root && operand!=send)memmove(operand,send,bytes);}
  else if(operand+(size_t)r*part!=send)memmove(operand+(size_t)r*part,send,part);
  ncclResult_t status=ncclSuccess;
  struct pending *sends=calloc(k->nsteps?k->nsteps:1,sizeof *sends);int count=0;
  void *used[64];int nused=0;
  if(!sends)status=FAIL(c,ncclSystemError,"allocation");
  for(uint32_t i=0;i<k->nsteps && !status;i++){
    const struct mesh_step *s=k->steps+i;
    const size_t lo=s->first*e,length=s->piece.elements*e;
    struct peer *p=c->peers+s->peer;
    void *request=NULL;
    if(!length)continue;
    if(s->op==MESH_STEP_SEND){
      if(!(status=post(c,1,p->send[CH_COLL],operand+lo,length,operand_mh,sends,&count,deadline,&request)))
        sends[count++]=(struct pending){lo,lo+length,request};
      continue;
    }
    int seen=0;
    for(int u=0;u<nused;u++)seen|=used[u]==p->recv[CH_COLL];
    if(!seen && nused<64)used[nused++]=p->recv[CH_COLL];
    /* a piece is written only once no isend still reads it */
    for(int j=0;j<count && !status && s->op==MESH_STEP_COPY;)
      if(sends[j].lo<lo+length && lo<sends[j].hi){status=await(c,sends[j].request,SIZE_MAX,NULL,NULL,deadline,"an isend");sends[j]=sends[--count];}
      else j++;
    if(status)break;
    unsigned char *into=s->op==MESH_STEP_COPY?operand:scratch;
    if((status=post(c,0,p->recv[CH_COLL],into+lo,length,s->op==MESH_STEP_COPY?operand_mh:c->stage_mh,sends,&count,deadline,&request)))break;
    if((status=await(c,request,length,sends,&count,deadline,s->op==MESH_STEP_COPY?"a COPY step's irecv":"a REDUCE step's irecv")))break;
    if(s->op==MESH_STEP_REDUCE){
      for(int j=0;j<count && !status;)
        if(sends[j].lo<lo+length && lo<sends[j].hi){status=await(c,sends[j].request,SIZE_MAX,NULL,NULL,deadline,"an isend");sends[j]=sends[--count];}
        else j++;
      if(!status)combine(operand+lo,scratch+lo,s->piece.elements,k->type,k->combine);
    }
  }
  while(count && !status){status=await(c,sends[count-1].request,SIZE_MAX,NULL,NULL,deadline,"an isend");count--;}
  for(int u=0;u<nused && !status;u++){
    void *flush=NULL;int result=mesh_net_iflush(used[u],1,NULL,NULL,NULL,&flush);
    if(result)status=net_failure(c,result,"mesh_net_iflush");
    else if(flush)status=await(c,flush,SIZE_MAX,NULL,NULL,deadline,"an iflush");
  }
  free(sends);
  if(!status){
    unsigned char *result=what==MESH_REDUCE_SCATTER?operand+(size_t)r*part:operand;
    size_t length=what==MESH_REDUCE_SCATTER?part:bytes;
    if(what==MESH_REDUCE && r!=root)length=0;
    if(length && k->postdivide)postdivide(result,length/e,k->type,c->nranks);
    if(length && result!=(unsigned char *)k->recv)memmove(k->recv,result,length);
  }
  if(own)mesh_net_dereg_mr(NULL,own);
  return status;
}

/* ---- one communicator's part of a group ---- */
struct transfer { int send; struct peer *peer; unsigned char *window,*user; size_t bytes; void *mhandle,*request; int state; };
/* Point-to-point calls first posted (receives and sends each in call order on their connection, as
   many as the rings take), the collectives run in issue order, then the point-to-point calls
   finished; a send to and a receive from this rank itself are one copy, the k-th with the k-th. */
static ncclResult_t run_item(struct ncclComm *c,struct item *it,uint64_t deadline){
  if(atomic_load(&c->broken))return FAIL(c,ncclRemoteError,"the communicator failed earlier: %s",c->error);
  size_t p2p=0,largest=0;
  int transfers=0;
  for(int i=0;i<it->n;i++){
    struct call *k=it->calls+i;
    size_t bytes=k->count*type_bytes[k->type];void *mh;
    if(k->kind>=K_SEND){
      if(k->peer==c->rank)continue;
      transfers++;
      if(in_window(k->kind==K_SEND?k->send:k->recv,bytes,&mh))mesh_net_dereg_mr(NULL,mh);
      else p2p+=aligned(bytes);
    } else {
      size_t operand=aligned((size_t)k->elements*type_bytes[k->type]);
      if(2*operand>largest)largest=2*operand;
    }
  }
  ncclResult_t status=stage_reserve(c,p2p+largest);
  if(status)return status;
  struct transfer *t=calloc(transfers?(size_t)transfers:1,sizeof *t);
  if(!t)return FAIL(c,ncclSystemError,"allocation");
  unsigned char *at=c->stage+largest;
  int n=0;
  for(int i=0;i<it->n;i++){
    struct call *k=it->calls+i;
    if(k->kind<K_SEND)continue;
    size_t bytes=k->count*type_bytes[k->type];
    if(k->peer==c->rank){
      if(k->kind!=K_RECV)continue;
      int sends=0,receives=0;const struct call *mine=NULL;
      for(int j=0;j<i;j++)receives+=it->calls[j].kind==K_RECV && it->calls[j].peer==c->rank;
      for(int j=0;j<it->n && !mine;j++)if(it->calls[j].kind==K_SEND && it->calls[j].peer==c->rank && sends++==receives)mine=it->calls+j;
      if(!mine){status=FAIL(c,ncclInvalidUsage,"a receive from this rank itself has no matching send to itself in the group");break;}
      if(mine->count*type_bytes[mine->type]!=bytes){status=FAIL(c,ncclInvalidUsage,"a send to itself and its receive differ in size");break;}
      memmove(k->recv,mine->send,bytes);
      continue;
    }
    struct transfer *x=t+n++;
    *x=(struct transfer){.send=k->kind==K_SEND,.peer=c->peers+k->peer,.user=k->kind==K_SEND?(unsigned char *)k->send:k->recv,.bytes=bytes};
    if(in_window(x->user,bytes,&x->mhandle))x->window=x->user;
    else{x->window=at;at+=aligned(bytes);x->mhandle=c->stage_mh;x->state=0;if(x->send)memcpy(x->window,x->user,bytes);}
  }
  /* posts every transfer its connection's ring takes, in order per connection, and tests the posted */
  int remaining=n;
  int *blocked=calloc((size_t)c->nranks*2,sizeof *blocked);
  if(!blocked && !status)status=FAIL(c,ncclSystemError,"allocation");
  #define PROGRESS() do{ \
    memset(blocked,0,sizeof *blocked*(size_t)c->nranks*2); \
    for(int j=0;j<n && !status;j++){ \
      struct transfer *x=t+j;int slot=(int)(x->peer-c->peers)*2+x->send; \
      if(x->state==0 && !blocked[slot]){ \
        void *comm=x->send?x->peer->send[CH_P2P]:x->peer->recv[CH_P2P];void *data=x->window;size_t size=x->bytes; \
        int result=x->send?mesh_net_isend(comm,data,size,0,x->mhandle,NULL,&x->request): \
          mesh_net_irecv(comm,1,&data,&size,(int[]){0},&x->mhandle,NULL,&x->request); \
        if(result)status=net_failure(c,result,x->send?"a send's isend":"a receive's irecv"); \
        else if(x->request)x->state=1;else blocked[slot]=1; \
      } \
      if(x->state==1){ \
        int done=0,size=0,result=mesh_net_test(x->request,&done,&size); \
        if(result)status=net_failure(c,result,x->send?"a send":"a receive"); \
        else if(done){ \
          x->state=2;remaining--; \
          if(!x->send && x->bytes<=INT32_MAX && (size_t)size!=x->bytes) \
            status=FAIL(c,ncclInvalidUsage,"a receive: %d bytes arrived, %zu expected (the peer's count or datatype differs)",size,x->bytes); \
        } \
      } \
    } \
  }while(0)
  if(!status)PROGRESS();
  for(int i=0;i<it->n && !status;i++)if(it->calls[i].kind<K_SEND)status=run_collective(c,it->calls+i,c->stage,deadline);
  while(remaining && !status){
    PROGRESS();
    if(remaining && !status){if(stopped(c,deadline))status=stop_reason(c,"a point-to-point call");else sched_yield();}
  }
  #undef PROGRESS
  for(int j=0;j<n && !status;j++)if(!t[j].send && t[j].window!=t[j].user)memcpy(t[j].user,t[j].window,t[j].bytes);
  for(int j=0;j<n;j++)if(t[j].window==t[j].user && t[j].mhandle)mesh_net_dereg_mr(NULL,t[j].mhandle);
  free(blocked);free(t);
  if(status && status!=ncclInvalidArgument)atomic_store(&c->broken,1);
  return status;
}

/* ---- the worker: a communicator's parts of groups, in issue order ---- */
static void finish(struct ncclComm *c,struct item *it,ncclResult_t result){
  struct launch *l=it->launch;
  pthread_mutex_lock(&l->lock);
  if(result && !l->result)l->result=result;
  int last=!--l->items;
  if(last)for(int m=0;m<l->nmarks;m++)nccl_mesh_event_signal(l->marks[m].stream->event,l->marks[m].done);
  if(result && !l->sync){int none=0;atomic_compare_exchange_strong(&c->async,&none,(int)result);}
  int sync=l->sync;
  if(last && sync)pthread_cond_broadcast(&l->cond);
  pthread_mutex_unlock(&l->lock);
  if(last && !sync){pthread_mutex_destroy(&l->lock);pthread_cond_destroy(&l->cond);free(l->marks);free(l);}
  for(int i=0;i<it->n;i++)free(it->calls[i].steps);
  free(it->calls);free(it);
}
static void *worker(void *argument){
  struct ncclComm *c=argument;
  pthread_setname_np("nccl-mesh.comm");
  for(;;){
    pthread_mutex_lock(&c->lock);
    while(!c->head && !c->stopping)pthread_cond_wait(&c->cond,&c->lock);
    struct item *it=c->head;
    if(!it){pthread_mutex_unlock(&c->lock);break;}
    c->head=it->next;if(!c->head)c->tail=NULL;
    c->busy=1;
    pthread_mutex_unlock(&c->lock);
    uint64_t deadline=deadline_after();
    ncclResult_t result=ncclSuccess;
    for(int m=0;m<it->launch->nmarks && !result;m++)
      while(nccl_mesh_event_value(it->launch->marks[m].stream->event)<it->launch->marks[m].wait){
        if(stopped(c,deadline)){result=stop_reason(c,"waiting for the stream's prior work");break;}
        usleep(20);
      }
    if(!result)result=run_item(c,it,deadline);
    finish(c,it,result);
    pthread_mutex_lock(&c->lock);
    c->busy=0;pthread_cond_broadcast(&c->cond);
    pthread_mutex_unlock(&c->lock);
  }
  return NULL;
}
static void drain(struct ncclComm *c){
  pthread_mutex_lock(&c->lock);
  while(c->head || c->busy)pthread_cond_wait(&c->cond,&c->lock);
  pthread_mutex_unlock(&c->lock);
}

/* ---- communicators ---- */
static struct ncclComm *comm_new(uint64_t key,int rank,int nranks){
  struct ncclComm *c=calloc(1,sizeof *c);
  if(!c)return NULL;
  c->key=key;c->rank=rank;c->nranks=nranks;
  pthread_mutex_init(&c->lock,NULL);pthread_cond_init(&c->cond,NULL);
  return c;
}
static void comm_free(struct ncclComm *c){
  if(c->started){
    pthread_mutex_lock(&c->lock);c->stopping=1;pthread_cond_broadcast(&c->cond);pthread_mutex_unlock(&c->lock);
    pthread_join(c->worker,NULL);
  }
  close_peers(c);
  if(c->stage){mesh_net_dereg_mr(NULL,c->stage_mh);mesh_net_mem_free(c->stage);}
  if(c->net)mesh_net_finalize(c->net);
  mesh_link_map_free(&c->map);
  pthread_mutex_destroy(&c->lock);pthread_cond_destroy(&c->cond);
  free(c->peers);free(c->ops);free(c);
}
/* Attached to the bridge, connected to every linked rank, its worker started. */
static ncclResult_t comm_start(struct ncclComm *c){
  ncclResult_t status=global_attach();
  int result=status?0:mesh_net_init(&c->net,c->key,NULL,NULL,NULL);
  if(!status && result)status=net_failure(c,result,"mesh_net_init (the bridge region MESH_REGION)");
  if(!status)status=connect_peers(c);
  if(!status){
    if(pthread_create(&c->worker,NULL,worker,c))status=FAIL(c,ncclSystemError,"worker thread");
    else c->started=1;
  }
  return status;
}

ncclResult_t ncclGetVersion(int *version){
  if(!version)return FAIL(NULL,ncclInvalidArgument,"ncclGetVersion: version is NULL");
  *version=NCCL_VERSION_CODE;
  return ncclSuccess;
}
ncclResult_t ncclGetUniqueId(ncclUniqueId *id){
  if(!id)return FAIL(NULL,ncclInvalidArgument,"ncclGetUniqueId: uniqueId is NULL");
  struct uid made={UID_MAGIC,NCCL_VERSION_CODE,0};
  while(!made.key)made.key=((uint64_t)arc4random()<<32)|arc4random();
  memset(id,0,sizeof *id);memcpy(id->internal,&made,sizeof made);
  return ncclSuccess;
}
ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm,int nranks,ncclUniqueId commId,int rank,ncclConfig_t *config){
  if(!comm)return FAIL(NULL,ncclInvalidArgument,"ncclCommInitRank: comm is NULL");
  *comm=NULL;
  struct uid id;memcpy(&id,commId.internal,sizeof id);
  if(id.magic!=UID_MAGIC || !id.key)return FAIL(NULL,ncclInvalidArgument,"ncclCommInitRank: not a unique id from ncclGetUniqueId");
  if(nranks<1 || rank<0 || rank>=nranks)return FAIL(NULL,ncclInvalidArgument,"ncclCommInitRank: rank %d of %d",rank,nranks);
  if(config && config->magic!=NCCL_API_MAGIC)return FAIL(NULL,ncclInvalidArgument,"ncclCommInitRankConfig: config not initialized with NCCL_CONFIG_INITIALIZER");
  struct ncclComm *c=comm_new(id.key,rank,nranks);
  if(!c)return FAIL(NULL,ncclSystemError,"allocation");
  ncclResult_t status=map_for(c);
  if(!status)status=comm_start(c);
  if(status){snprintf(last_error,sizeof last_error,"%s",c->error);comm_free(c);return status;}
  *comm=c;
  return ncclSuccess;
}
ncclResult_t ncclCommInitRank(ncclComm_t *comm,int nranks,ncclUniqueId commId,int rank){
  return ncclCommInitRankConfig(comm,nranks,commId,rank,NULL);
}
/* One process holds one rank of a node: a node has one Metal device. */
ncclResult_t ncclCommInitAll(ncclComm_t *comm,int ndev,const int *devlist){
  if(!comm || ndev<1)return FAIL(NULL,ncclInvalidArgument,"ncclCommInitAll: %d devices",ndev);
  if(ndev!=1 || (devlist && devlist[0]!=0))
    return FAIL(NULL,ncclInvalidArgument,"ncclCommInitAll: a node has one Metal device (device 0); ranks on other nodes join with ncclCommInitRank");
  ncclUniqueId id;ncclGetUniqueId(&id);
  return ncclCommInitRank(comm,1,id,0);
}
ncclResult_t ncclCommFinalize(ncclComm_t comm){
  if(!comm)return FAIL(NULL,ncclInvalidArgument,"ncclCommFinalize: comm is NULL");
  drain(comm);
  comm->finalized=1;
  return (ncclResult_t)atomic_load(&comm->async);
}
ncclResult_t ncclCommDestroy(ncclComm_t comm){
  if(!comm)return ncclSuccess;
  drain(comm);
  comm_free(comm);
  return ncclSuccess;
}
ncclResult_t ncclCommAbort(ncclComm_t comm){
  if(!comm)return ncclSuccess;
  atomic_store(&comm->aborting,1);
  drain(comm);
  comm_free(comm);
  return ncclSuccess;
}
ncclResult_t ncclCommCount(const ncclComm_t comm,int *count){
  if(!comm || !count)return FAIL(NULL,ncclInvalidArgument,"ncclCommCount: NULL argument");
  *count=comm->nranks;return ncclSuccess;
}
ncclResult_t ncclCommUserRank(const ncclComm_t comm,int *rank){
  if(!comm || !rank)return FAIL(NULL,ncclInvalidArgument,"ncclCommUserRank: NULL argument");
  *rank=comm->rank;return ncclSuccess;
}
ncclResult_t ncclCommCuDevice(const ncclComm_t comm,int *device){
  if(!comm || !device)return FAIL(NULL,ncclInvalidArgument,"ncclCommCuDevice: NULL argument");
  *device=0;return ncclSuccess;
}
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm,ncclResult_t *asyncError){
  if(!comm || !asyncError)return FAIL(NULL,ncclInvalidArgument,"ncclCommGetAsyncError: NULL argument");
  *asyncError=(ncclResult_t)atomic_load(&comm->async);return ncclSuccess;
}
const char *ncclGetErrorString(ncclResult_t result){
  switch(result){
  case ncclSuccess: return "no error";
  case ncclUnhandledCudaError: return "unhandled Metal error";
  case ncclSystemError: return "unhandled system error (set MESH_NCCL_DEBUG=1 for details)";
  case ncclInternalError: return "internal error";
  case ncclInvalidArgument: return "invalid argument (set MESH_NCCL_DEBUG=1 for details)";
  case ncclInvalidUsage: return "invalid usage (set MESH_NCCL_DEBUG=1 for details)";
  case ncclRemoteError: return "remote process exited or there was a network error";
  case ncclInProgress: return "NCCL operation in progress";
  case ncclTimeout: return "operation timed out (MESH_NCCL_TIMEOUT)";
  default: return "unknown result code";
  }
}
const char *ncclGetLastError(ncclComm_t comm){
  return comm && comm->error[0]?comm->error:last_error;
}
struct registration { void *buffer; size_t size; };
ncclResult_t ncclCommRegister(const ncclComm_t comm,void *buff,size_t size,void **handle){
  if(!comm || !handle)return FAIL(comm,ncclInvalidArgument,"ncclCommRegister: NULL argument");
  struct registration *r=malloc(sizeof *r);
  if(!r)return FAIL(comm,ncclSystemError,"allocation");
  *r=(struct registration){buff,size};*handle=r;
  return ncclSuccess;
}
ncclResult_t ncclCommDeregister(const ncclComm_t comm,void *handle){
  (void)comm;free(handle);return ncclSuccess;
}
ncclResult_t ncclMemAlloc(void **ptr,size_t size){
  if(!ptr)return FAIL(NULL,ncclInvalidArgument,"ncclMemAlloc: ptr is NULL");
  ncclResult_t status=global_attach();
  if(status)return status;
  int result=mesh_net_mem_alloc(ptr,size?size:1);
  return result?net_failure(NULL,result,"ncclMemAlloc from the bridge's registered window"):ncclSuccess;
}
ncclResult_t ncclMemFree(void *ptr){
  if(!ptr)return ncclSuccess;
  int result=mesh_net_mem_free(ptr);
  return result?net_failure(NULL,result,"ncclMemFree"):ncclSuccess;
}
ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op,void *scalar,ncclDataType_t datatype,ncclScalarResidence_t residence,ncclComm_t comm){
  if(!op || !scalar || !comm || (unsigned)datatype>=ncclNumTypes)return FAIL(comm,ncclInvalidArgument,"ncclRedOpCreatePreMulSum: invalid argument");
  if(residence!=ncclScalarHostImmediate)return FAIL(comm,ncclInvalidArgument,"ncclRedOpCreatePreMulSum: the host path reads the scalar now (ncclScalarHostImmediate)");
  int slot=0;
  while(slot<comm->nops && comm->ops[slot].used)slot++;
  if(slot==comm->nops){
    struct op_entry *grown=realloc(comm->ops,(size_t)(comm->nops+16)*sizeof *grown);
    if(!grown)return FAIL(comm,ncclSystemError,"allocation");
    memset(grown+comm->nops,0,16*sizeof *grown);comm->ops=grown;comm->nops+=16;
  }
  comm->ops[slot].used=1;comm->ops[slot].type=datatype;
  memset(comm->ops[slot].scalar,0,8);memcpy(comm->ops[slot].scalar,scalar,type_bytes[datatype]);
  *op=(ncclRedOp_t)(ncclNumOps+slot);
  return ncclSuccess;
}
ncclResult_t ncclRedOpDestroy(ncclRedOp_t op,ncclComm_t comm){
  int slot=(int)op-ncclNumOps;
  if(!comm || slot<0 || slot>=comm->nops || !comm->ops[slot].used)return FAIL(comm,ncclInvalidArgument,"ncclRedOpDestroy: no such operator");
  comm->ops[slot].used=0;
  return ncclSuccess;
}

/* ---- groups ---- */
static _Thread_local struct { int depth,n,capacity; struct call *calls; ncclResult_t error; } group;
static _Thread_local struct { int n; int how[4096],root[4096]; } plans;
static pthread_mutex_t stream_lock=PTHREAD_MUTEX_INITIALIZER;

ncclResult_t ncclGroupStart(void){group.depth++;return ncclSuccess;}

/* The operator, the algorithm and this rank's plan of a collective; a point-to-point call's peer
   checked. */
static ncclResult_t resolve(struct call *k){
  struct ncclComm *c=k->comm;
  if(k->kind>=K_SEND){
    if(k->peer!=c->rank && !c->peers[k->peer].send[CH_P2P])
      return FAIL(c,ncclInvalidUsage,"%s rank %d: no link joins rank %d to it (the mesh does not forward point-to-point)",k->kind==K_SEND?"ncclSend to":"ncclRecv from",k->peer,c->rank);
    return ncclSuccess;
  }
  k->elements=(uint64_t)k->count*(k->kind==MESH_REDUCE_SCATTER || k->kind==MESH_ALLGATHER?(uint64_t)c->nranks:1);
  k->combine=ncclSum;
  if(k->kind==MESH_ALLREDUCE || k->kind==MESH_REDUCE || k->kind==MESH_REDUCE_SCATTER){
    if(k->op<ncclAvg)k->combine=k->op;
    else if(k->op==ncclAvg){
      if(integral(k->type))k->postdivide=1;
      else{k->premultiply=1;reciprocal(k->scalar,k->type,c->nranks);}
    } else {
      int slot=k->op-ncclNumOps;
      if(slot<0 || slot>=c->nops || !c->ops[slot].used)return FAIL(c,ncclInvalidArgument,"operator %d was not created on this communicator",k->op);
      if(c->ops[slot].type!=k->type)return FAIL(c,ncclInvalidArgument,"a PreMulSum operator reduces the datatype it was created for");
      k->premultiply=1;memcpy(k->scalar,c->ops[slot].scalar,8);
    }
  }
  if(c->nranks==1)return ncclSuccess;
  struct mesh_operand operand={(uint32_t)k->type,(uint32_t)type_bytes[k->type],k->elements};
  struct mesh_collective wanted={.what=(uint32_t)k->kind,.how=k->how,.root=(uint32_t)k->root};
  k->chosen=mesh_collective_choose(&c->map,wanted,operand,c->alpha,c->beta);
  if(k->chosen.how==MESH_UNAVAILABLE && k->how && !k->force){wanted.how=0;k->chosen=mesh_collective_choose(&c->map,wanted,operand,c->alpha,c->beta);}
  if(k->chosen.how==MESH_UNAVAILABLE)
    return FAIL(c,ncclInvalidUsage,"no algorithm of the selection %#x carries this collective (%d) of %llu elements on the communicator's link map",
      k->how,k->kind,(unsigned long long)k->elements);
  k->steps=calloc(MESH_COLLECTIVE_STEPS(c->nranks),sizeof *k->steps);
  if(!k->steps)return FAIL(c,ncclSystemError,"allocation");
  k->nsteps=mesh_collective_plan(&c->map,(uint32_t)c->rank,k->chosen,operand,k->steps);
  if(!k->nsteps)return FAIL(c,ncclInternalError,"the planner gave rank %d no steps",c->rank);
  for(uint32_t i=0;i<k->nsteps;i++)if(!c->peers[k->steps[i].peer].send[CH_COLL])
    return FAIL(c,ncclInvalidUsage,"the plan uses rank %u, which no connection reaches",k->steps[i].peer);
  return ncclSuccess;
}

static void calls_free(struct call *calls,int n){
  for(int i=0;i<n;i++)free(calls[i].steps);
  free(calls);
}
ncclResult_t ncclGroupEnd(void){
  if(group.depth<=0)return FAIL(NULL,ncclInvalidUsage,"ncclGroupEnd without ncclGroupStart");
  if(--group.depth)return ncclSuccess;
  struct call *calls=group.calls;int n=group.n;ncclResult_t status=group.error;
  group.calls=NULL;group.n=group.capacity=0;group.error=ncclSuccess;
  for(int i=0;i<n && !status;i++)status=resolve(calls+i);
  plans.n=0;
  for(int i=0;i<n && !status && plans.n<4096;i++,plans.n++){
    int local=calls[i].kind>=K_SEND || calls[i].comm->nranks==1;
    plans.how[plans.n]=local?-1:(int)calls[i].chosen.how;plans.root[plans.n]=local?0:(int)calls[i].chosen.root;
  }
  if(status || !n){calls_free(calls,n);return status;}
  struct launch *l=calloc(1,sizeof *l);
  struct ncclComm **comms=calloc((size_t)n,sizeof *comms);int ncomms=0;
  if(!l || !comms){free(l);free(comms);calls_free(calls,n);return FAIL(NULL,ncclSystemError,"allocation");}
  pthread_mutex_init(&l->lock,NULL);pthread_cond_init(&l->cond,NULL);
  l->marks=calloc((size_t)n,sizeof *l->marks);
  for(int i=0;i<n;i++){
    int seen=0;
    for(int j=0;j<ncomms;j++)seen|=comms[j]==calls[i].comm;
    if(!seen)comms[ncomms++]=calls[i].comm;
    if(!calls[i].stream)l->sync=1;
    else{
      seen=0;
      for(int m=0;m<l->nmarks;m++)seen|=l->marks[m].stream==calls[i].stream;
      if(!seen)l->marks[l->nmarks++].stream=calls[i].stream;
    }
  }
  /* each stream: its prior work's value (the queue signals it, or the caller has), then the value the
     group's completion signals, which the queue's later work waits for */
  pthread_mutex_lock(&stream_lock);
  for(int m=0;m<l->nmarks;m++){
    struct mark *mark=l->marks+m;struct ncclMeshStream *s=mark->stream;
    if(s->queue){mark->wait=++s->value;nccl_mesh_queue_signal(s->queue,s->event,mark->wait);}
    else mark->wait=s->value;
    mark->done=++s->value;
    if(s->queue)nccl_mesh_queue_wait(s->queue,s->event,mark->done);
  }
  pthread_mutex_unlock(&stream_lock);
  l->items=ncomms;
  for(int j=0;j<ncomms;j++){
    struct item *it=calloc(1,sizeof *it);
    it->calls=calloc((size_t)n,sizeof *it->calls);it->launch=l;
    for(int i=0;i<n;i++)if(calls[i].comm==comms[j]){it->calls[it->n++]=calls[i];calls[i].steps=NULL;}
    struct ncclComm *c=comms[j];
    pthread_mutex_lock(&c->lock);
    if(c->tail)c->tail->next=it;else c->head=it;
    c->tail=it;
    pthread_cond_signal(&c->cond);
    pthread_mutex_unlock(&c->lock);
  }
  free(comms);free(calls);
  if(!l->sync)return ncclSuccess;
  pthread_mutex_lock(&l->lock);
  while(l->items)pthread_cond_wait(&l->cond,&l->lock);
  status=l->result;
  pthread_mutex_unlock(&l->lock);
  pthread_mutex_destroy(&l->lock);pthread_cond_destroy(&l->cond);free(l->marks);free(l);
  return status;
}

/* A call joins the open group, or is a group of one. */
static ncclResult_t enqueue(struct call k,const ncclCollConfig_t *config){
  struct ncclComm *c=k.comm;
  ncclResult_t status=ncclSuccess;
  if(!c)status=FAIL(NULL,ncclInvalidArgument,"comm is NULL");
  else if((unsigned)k.type>=ncclNumTypes)status=FAIL(c,ncclInvalidArgument,"datatype %d",k.type);
  else if(k.kind>=K_SEND && (k.peer<0 || k.peer>=c->nranks))status=FAIL(c,ncclInvalidArgument,"peer %d of %d ranks",k.peer,c->nranks);
  else if((k.kind==MESH_BROADCAST || k.kind==MESH_REDUCE) && (k.root<0 || k.root>=c->nranks))status=FAIL(c,ncclInvalidArgument,"root %d of %d ranks",k.root,c->nranks);
  else if(k.count && k.kind!=K_RECV && !k.send && !(k.kind==MESH_BROADCAST && c->rank!=k.root))status=FAIL(c,ncclInvalidArgument,"sendbuff is NULL");
  else if(k.count && k.kind!=K_SEND && !k.recv && !(k.kind==MESH_REDUCE && c->rank!=k.root))status=FAIL(c,ncclInvalidArgument,"recvbuff is NULL");
  else if(k.stream && !k.stream->event)status=FAIL(c,ncclInvalidArgument,"the stream has no event (ncclMeshStreamCreate)");
  else if(atomic_load(&c->aborting))status=FAIL(c,ncclInvalidUsage,"the communicator was aborted");
  else if(k.op<0 || (k.op>=ncclNumOps && k.op-ncclNumOps>=c->nops))status=FAIL(c,ncclInvalidArgument,"operator %d",k.op);
  if(!status && config){
    if(config->magic!=NCCL_API_MAGIC)status=FAIL(c,ncclInvalidArgument,"ncclCollConfig_t not initialized with NCCL_COLLCONFIG_INITIALIZER");
    else if(config->launchCompletionEvent)status=FAIL(c,ncclInvalidUsage,"launchCompletionEvent is not supported (use the stream's event)");
    else if(config->algSelection && *config->algSelection){
      static const char *const names[]={"direct","ring","tree","binomial"};
      char text[256];snprintf(text,sizeof text,"%s",config->algSelection);
      k.force=config->forceAlgSelection!=0;
      for(char *save=NULL,*name=strtok_r(text,", \t",&save);name && !status;name=strtok_r(NULL,", \t",&save)){
        int found=-1;
        for(int a=0;a<4;a++)if(!strcasecmp(name,names[a]))found=a;
        if(found>=0)k.how|=1u<<found;
        else if(k.force)status=FAIL(c,ncclInvalidArgument,"algSelection \"%s\": the mesh's algorithms are direct, ring, tree, binomial",name);
      }
    }
  }
  if(!status && !k.count)return ncclSuccess;
  ncclGroupStart();
  if(status){if(!group.error)group.error=status;}
  else {
    if(group.n==group.capacity){
      int capacity=group.capacity?2*group.capacity:16;
      struct call *grown=realloc(group.calls,(size_t)capacity*sizeof *grown);
      if(!grown){if(!group.error)group.error=FAIL(c,ncclSystemError,"allocation");}
      else{group.calls=grown;group.capacity=capacity;}
    }
    if(group.n<group.capacity)group.calls[group.n++]=k;
  }
  ncclResult_t ended=ncclGroupEnd();
  return status?status:ended;
}
#define CALL(...) ((struct call){__VA_ARGS__})

ncclResult_t ncclAllReduceConfig(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  return enqueue(CALL(.kind=MESH_ALLREDUCE,.send=sendbuff,.recv=recvbuff,.count=count,.type=datatype,.op=op,.comm=comm,.stream=stream),config);
}
ncclResult_t ncclAllReduce(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,cudaStream_t stream){
  return ncclAllReduceConfig(sendbuff,recvbuff,count,datatype,op,comm,stream,NULL);
}
ncclResult_t ncclReduceConfig(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,ncclRedOp_t op,int root,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  return enqueue(CALL(.kind=MESH_REDUCE,.send=sendbuff,.recv=recvbuff,.count=count,.type=datatype,.op=op,.root=root,.comm=comm,.stream=stream),config);
}
ncclResult_t ncclReduce(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,ncclRedOp_t op,int root,ncclComm_t comm,cudaStream_t stream){
  return ncclReduceConfig(sendbuff,recvbuff,count,datatype,op,root,comm,stream,NULL);
}
ncclResult_t ncclBroadcastConfig(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,int root,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  return enqueue(CALL(.kind=MESH_BROADCAST,.send=sendbuff,.recv=recvbuff,.count=count,.type=datatype,.root=root,.comm=comm,.stream=stream),config);
}
ncclResult_t ncclBroadcast(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,int root,ncclComm_t comm,cudaStream_t stream){
  return ncclBroadcastConfig(sendbuff,recvbuff,count,datatype,root,comm,stream,NULL);
}
ncclResult_t ncclBcast(void *buff,size_t count,ncclDataType_t datatype,int root,ncclComm_t comm,cudaStream_t stream){
  return ncclBroadcast(buff,buff,count,datatype,root,comm,stream);
}
ncclResult_t ncclReduceScatterConfig(const void *sendbuff,void *recvbuff,size_t recvcount,ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  return enqueue(CALL(.kind=MESH_REDUCE_SCATTER,.send=sendbuff,.recv=recvbuff,.count=recvcount,.type=datatype,.op=op,.comm=comm,.stream=stream),config);
}
ncclResult_t ncclReduceScatter(const void *sendbuff,void *recvbuff,size_t recvcount,ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,cudaStream_t stream){
  return ncclReduceScatterConfig(sendbuff,recvbuff,recvcount,datatype,op,comm,stream,NULL);
}
ncclResult_t ncclAllGatherConfig(const void *sendbuff,void *recvbuff,size_t sendcount,ncclDataType_t datatype,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  return enqueue(CALL(.kind=MESH_ALLGATHER,.send=sendbuff,.recv=recvbuff,.count=sendcount,.type=datatype,.comm=comm,.stream=stream),config);
}
ncclResult_t ncclAllGather(const void *sendbuff,void *recvbuff,size_t sendcount,ncclDataType_t datatype,ncclComm_t comm,cudaStream_t stream){
  return ncclAllGatherConfig(sendbuff,recvbuff,sendcount,datatype,comm,stream,NULL);
}
ncclResult_t ncclSend(const void *sendbuff,size_t count,ncclDataType_t datatype,int peer,ncclComm_t comm,cudaStream_t stream){
  return enqueue(CALL(.kind=K_SEND,.send=sendbuff,.count=count,.type=datatype,.peer=peer,.comm=comm,.stream=stream),NULL);
}
ncclResult_t ncclRecv(void *recvbuff,size_t count,ncclDataType_t datatype,int peer,ncclComm_t comm,cudaStream_t stream){
  return enqueue(CALL(.kind=K_RECV,.recv=recvbuff,.count=count,.type=datatype,.peer=peer,.comm=comm,.stream=stream),NULL);
}
/* NCCL 2.32's own lowering of these three (src/enqueue/task_prep/task_classify.cc
   classifyCollToP2pTasks): grouped sends and receives. */
ncclResult_t ncclAlltoAllConfig(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  (void)config;
  if(!comm)return FAIL(NULL,ncclInvalidArgument,"comm is NULL");
  if((unsigned)datatype>=ncclNumTypes)return FAIL(comm,ncclInvalidArgument,"datatype %d",datatype);
  size_t bytes=count*type_bytes[datatype];
  ncclResult_t status=ncclSuccess;
  ncclGroupStart();
  for(int r=0;r<comm->nranks;r++){
    ncclResult_t a=ncclSend((const char *)sendbuff+(size_t)r*bytes,count,datatype,r,comm,stream);
    ncclResult_t b=ncclRecv((char *)recvbuff+(size_t)r*bytes,count,datatype,r,comm,stream);
    if(!status)status=a?a:b;
  }
  ncclResult_t ended=ncclGroupEnd();
  return status?status:ended;
}
ncclResult_t ncclAlltoAll(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,ncclComm_t comm,cudaStream_t stream){
  return ncclAlltoAllConfig(sendbuff,recvbuff,count,datatype,comm,stream,NULL);
}
ncclResult_t ncclGatherConfig(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,int root,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  (void)config;
  if(!comm)return FAIL(NULL,ncclInvalidArgument,"comm is NULL");
  if((unsigned)datatype>=ncclNumTypes || root<0 || root>=comm->nranks)return FAIL(comm,ncclInvalidArgument,"datatype %d, root %d",datatype,root);
  size_t bytes=count*type_bytes[datatype];
  ncclGroupStart();
  ncclResult_t status=ncclSend(sendbuff,count,datatype,root,comm,stream);
  for(int r=0;comm->rank==root && r<comm->nranks;r++){
    ncclResult_t b=ncclRecv((char *)recvbuff+(size_t)r*bytes,count,datatype,r,comm,stream);
    if(!status)status=b;
  }
  ncclResult_t ended=ncclGroupEnd();
  return status?status:ended;
}
ncclResult_t ncclGather(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,int root,ncclComm_t comm,cudaStream_t stream){
  return ncclGatherConfig(sendbuff,recvbuff,count,datatype,root,comm,stream,NULL);
}
ncclResult_t ncclScatterConfig(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,int root,ncclComm_t comm,
  cudaStream_t stream,const ncclCollConfig_t *config){
  (void)config;
  if(!comm)return FAIL(NULL,ncclInvalidArgument,"comm is NULL");
  if((unsigned)datatype>=ncclNumTypes || root<0 || root>=comm->nranks)return FAIL(comm,ncclInvalidArgument,"datatype %d, root %d",datatype,root);
  size_t bytes=count*type_bytes[datatype];
  ncclResult_t status=ncclSuccess;
  ncclGroupStart();
  for(int r=0;comm->rank==root && r<comm->nranks;r++){
    ncclResult_t a=ncclSend((const char *)sendbuff+(size_t)r*bytes,count,datatype,r,comm,stream);
    if(!status)status=a;
  }
  ncclResult_t b=ncclRecv(recvbuff,count,datatype,root,comm,stream);
  if(!status)status=b;
  ncclResult_t ended=ncclGroupEnd();
  return status?status:ended;
}
ncclResult_t ncclScatter(const void *sendbuff,void *recvbuff,size_t count,ncclDataType_t datatype,int root,ncclComm_t comm,cudaStream_t stream){
  return ncclScatterConfig(sendbuff,recvbuff,count,datatype,root,comm,stream,NULL);
}

/* NCCL's split: every rank's (color, key) all-gathered on `comm`, the ranks of this color ordered by
   key, then rank; the new clique's key derived from the parent's and its count of splits (every rank
   splits in the same order).  The new map is the parent's links among them, relabelled: the same
   kind where every rank stays, a mesh where every pair is linked (one rank alone). */
ncclResult_t ncclCommSplit(ncclComm_t comm,int color,int key,ncclComm_t *newcomm,ncclConfig_t *config){
  if(!comm || !newcomm)return FAIL(comm,ncclInvalidArgument,"ncclCommSplit: NULL argument");
  if(config && config->magic!=NCCL_API_MAGIC)return FAIL(comm,ncclInvalidArgument,"ncclCommSplit: config not initialized");
  *newcomm=NULL;
  const int n=comm->nranks;
  int64_t mine[2]={color,key},*every=calloc((size_t)n*2,sizeof *every);
  if(!every)return FAIL(comm,ncclSystemError,"allocation");
  /* the all-gather is its own group: the caller's open group (if any) is set aside meanwhile */
  __typeof__(group) open=group;
  memset(&group,0,sizeof group);
  ncclResult_t status=ncclAllGather(mine,every,2,ncclInt64,comm,NULL);
  group=open;
  uint64_t split=comm->splits++;
  if(status || color==NCCL_SPLIT_NOCOLOR){free(every);return status;}
  int *ranks=calloc((size_t)n,sizeof *ranks),*relabel=calloc((size_t)n,sizeof *relabel),size=0;
  if(!ranks || !relabel){free(every);free(ranks);free(relabel);return FAIL(comm,ncclSystemError,"allocation");}
  for(int r=0;r<n;r++)if(every[2*r]==color)ranks[size++]=r;
  for(int i=1;i<size;i++)for(int j=i;j>0;j--){
    int a=ranks[j-1],b=ranks[j];
    if(every[2*a+1]>every[2*b+1] || (every[2*a+1]==every[2*b+1] && a>b)){ranks[j-1]=b;ranks[j]=a;}
  }
  for(int r=0;r<n;r++)relabel[r]=-1;
  for(int i=0;i<size;i++)relabel[ranks[i]]=i;
  struct ncclComm *c=comm_new(mix(comm->key^mix(split))^mix((uint64_t)(uint32_t)color<<1)^1,relabel[comm->rank],size);
  if(!c){free(every);free(ranks);free(relabel);return FAIL(comm,ncclSystemError,"allocation");}
  c->alpha=comm->alpha;c->beta=comm->beta;
  uint32_t (*kept)[2]=calloc(comm->map.links?comm->map.links:1,sizeof *kept),links=0;
  for(uint32_t l=0;kept && l<comm->map.links;l++){
    int a=relabel[comm->map.link[l][0]],b=relabel[comm->map.link[l][1]];
    if(a>=0 && b>=0){kept[links][0]=(uint32_t)a;kept[links++][1]=(uint32_t)b;}
  }
  int every_pair=1;
  struct mesh_link_map probe={comm->map.kind,(uint32_t)size,links,kept};
  for(int a=0;a<size;a++)for(int b=a+1;b<size;b++)every_pair&=linked(&probe,a,b);
  if(!kept)status=FAIL(comm,ncclSystemError,"allocation");
  else if(size==n && comm->map.kind!=MESH_LINKS_MESH)c->map=probe;
  else if(comm->map.kind==MESH_LINKS_MESH || every_pair){c->map=(struct mesh_link_map){.kind=MESH_LINKS_MESH,.nodes=(uint32_t)size};free(kept);}
  else{free(kept);status=FAIL(comm,ncclInvalidUsage,"ncclCommSplit: the ranks of color %d are not every pair linked, nor every rank of the parent",color);}
  free(every);free(ranks);free(relabel);
  if(!status)status=comm_start(c);
  if(status){snprintf(last_error,sizeof last_error,"%s",c->error[0]?c->error:comm->error);comm_free(c);return status;}
  *newcomm=c;
  return ncclSuccess;
}

/* ---- the stream ---- */
ncclResult_t ncclMeshStreamCreate(cudaStream_t *stream,void *queue){
  if(!stream)return FAIL(NULL,ncclInvalidArgument,"ncclMeshStreamCreate: stream is NULL");
  struct ncclMeshStream *s=calloc(1,sizeof *s);
  if(!s)return FAIL(NULL,ncclSystemError,"allocation");
  s->queue=queue;s->event=nccl_mesh_event_create(queue);
  if(!s->event){free(s);return FAIL(NULL,ncclUnhandledCudaError,"no Metal device for the stream's event");}
  *stream=s;
  return ncclSuccess;
}
ncclResult_t ncclMeshStreamDestroy(cudaStream_t stream){
  if(!stream)return ncclSuccess;
  nccl_mesh_event_release(stream->event);free(stream);
  return ncclSuccess;
}
ncclResult_t ncclMeshStreamSynchronize(cudaStream_t stream){
  if(!stream || !stream->event)return FAIL(NULL,ncclInvalidArgument,"ncclMeshStreamSynchronize: no stream");
  uint64_t deadline=deadline_after(),value;
  pthread_mutex_lock(&stream_lock);value=stream->value;pthread_mutex_unlock(&stream_lock);
  while(nccl_mesh_event_value(stream->event)<value){
    if(now_ns()>deadline)return FAIL(NULL,ncclTimeout,"ncclMeshStreamSynchronize: the stream did not reach %llu",(unsigned long long)value);
    usleep(20);
  }
  return ncclSuccess;
}
ncclResult_t ncclMeshGroupPlans(int *algorithms,int *roots,int capacity,int *count){
  if(!count)return FAIL(NULL,ncclInvalidArgument,"ncclMeshGroupPlans: count is NULL");
  *count=plans.n;
  for(int i=0;i<plans.n && i<capacity;i++){if(algorithms)algorithms[i]=plans.how[i];if(roots)roots[i]=plans.root[i];}
  return ncclSuccess;
}
