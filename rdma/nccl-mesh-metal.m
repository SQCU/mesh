#import <Metal/Metal.h>
#import <objc/runtime.h>
#include "nccl-mesh-metal.h"
#include "nccl-mesh-kernels.h"
#include "mesh_metal.h"
#include "metal-capabilities.h"
#include "nccl.h"
#include "metal-recording.h"
#include "mesh-landing.h"
#include "mesh-fabric.h"
#include "mesh-graph.h"
#include "mesh-publication.h"
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* a typed kernel's kind by its operation and its ncclDataType_t (int8 uint8 int32 uint32 int64 uint64 float16 float32
   float64 bfloat16 e4m3 e5m2, then the partials merge's rows, METAL_MERGE), KINDS where none is compiled: float64's,
   truncdiv's but the integers', premul's of the merge */
enum { OP_PREMUL, OP_TRUNCDIV, OPS };
#define INTEGERS(op) KIND_##op##_i8, KIND_##op##_u8, KIND_##op##_i32, KIND_##op##_u32, KIND_##op##_i64, KIND_##op##_u64
#define FLOATS(op) KIND_##op##_f16, KIND_##op##_f32, KINDS, KIND_##op##_bf16, KIND_##op##_e4m3, KIND_##op##_e5m2
static const uint8_t TYPED[OPS][METAL_MERGE + 1] = {
  {INTEGERS(premul), FLOATS(premul), KINDS},
  {INTEGERS(truncdiv), KINDS, KINDS, KINDS, KINDS, KINDS, KINDS, KINDS}};
/* The arithmetic operation is part of the compiled entry point, never a
   selector read for each payload element. */
#define REDUCTION_TYPES(op) {KIND_zelloth_i8_##op,KIND_zelloth_u8_##op,KIND_zelloth_i32_##op,KIND_zelloth_u32_##op, \
  KIND_zelloth_i64_##op,KIND_zelloth_u64_##op,KIND_zelloth_f16_##op,KIND_zelloth_f32_##op,KINDS, \
  KIND_zelloth_bf16_##op,KIND_zelloth_e4m3_##op,KIND_zelloth_e5m2_##op}
static const uint8_t REDUCTIONS[4][ncclNumTypes]={REDUCTION_TYPES(sum),REDUCTION_TYPES(prod),REDUCTION_TYPES(max),REDUCTION_TYPES(min)};
/* a copy's kind by its width (0 words of 4 bytes, 1 bytes) and its loads (METAL_PLAIN, METAL_LAND) */
static const uint8_t COPY[2][METAL_LAND + 1][2] = {
  {{KIND_copy_plain_4,KIND_copy_ready_plain_4},{KIND_copy_land_4,KIND_copy_ready_land_4}},
  {{KIND_copy_plain_1,KIND_copy_ready_plain_1},{KIND_copy_land_1,KIND_copy_ready_land_1}}};

ARGS(args, uint64_t, uint32_t);
/* Kernel lookup is shared; compiled argument pages, pieces of the region (nccl-mesh.c metal_piece_hold), belong to
   their function's residency set, are never recycled while that function can execute, and are given back once that
   set ends. */
#define RECORD MESH_METAL_ARGUMENT_BYTES
/* Prepared pipelines; launch extent comes from the compiled page cells. */
struct kernels { id<MTLDevice> device; id<MTLComputePipelineState> pipelines[KINDS]; };
@interface MeshArgumentPages : NSObject {
@public
  struct metal_bytes page;
  size_t bytes, used, count, room;
  void **held;
}
@end
@implementation MeshArgumentPages
- (void)dealloc { for (size_t k = 0; k < count; k++) metal_piece_give(held[k]); free(held); [super dealloc]; }
@end
static char argumentPagesKey;
_Static_assert(sizeof(struct args) <= RECORD, "an argument record");

static id<MTLDevice> device_;
static NSMutableDictionary *kernels_;
/* the text of this thread's last failure here that is composed (say): a result's message (struct metal_result), its
   caller's to carry before the thread's next step */
static _Thread_local char said_[MESH_NCCL_DIAGNOSTIC_BYTES];

static const char *say(const char *format, ...) __attribute__((format(printf, 1, 2)));
static const char *say(const char *format, ...) {
  va_list arguments;
  va_start(arguments, format);
  vsnprintf(said_, sizeof said_, format, arguments);
  va_end(arguments);
  return said_;
}

void *metal_device(void) {
  static dispatch_once_t once;
  dispatch_once(&once, ^{ device_ = MTLCreateSystemDefaultDevice(); });
  return device_;
}

/* `device`'s kernels at its first use, none compiled here: the library from the AIR its text was compiled to when this
   library was built, and every kind's pipeline (NAME) from this node's binary archive of them, made by its install,
   each looked up with MTLPipelineOptionFailOnBinaryArchiveMiss.  Both files sit beside this library's image, named by
   the kernels' SHA-256 this image was compiled with (KERNELS_SHA256: of the text and its compile line, the
   Makefile's): libnccl-mesh.<SHA-256>.metallib and libnccl-mesh.<SHA-256>.binarchive.  NULL, *message the rebuild
   that makes them and what failed, where a file is absent (an install of other kernels, or none) or the archive lacks
   a kind (its GPU compiler not the system's: an OS update since the install); the caller's collective fails as a
   value, and nothing is kept, so a later call reads the files again (rdma/NCCL.md "The library's kernels") */
static struct kernels *kernels_of(id<MTLDevice> device, const char **message) {
  @autoreleasepool {
    if (!kernels_) kernels_ = [NSMutableDictionary new];
    NSValue *key = [NSValue valueWithPointer:(__bridge const void *)device];
    NSValue *held = kernels_[key];
    if (held) return held.pointerValue;
    Dl_info image;
    if (!dladdr((const void *)kernels_of, &image)) { *message = "dladdr: no image holds libnccl-mesh's code"; return NULL; }
    NSString *stem = [[@(image.dli_fname) stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"libnccl-mesh." KERNELS_SHA256];
    NSString *air = [stem stringByAppendingString:@".metallib"], *binaries = [stem stringByAppendingString:@".binarchive"];
    NSError *error = nil;
    id<MTLLibrary> library = [[device newLibraryWithURL:[NSURL fileURLWithPath:air] error:&error] autorelease];
    MTLBinaryArchiveDescriptor *archived = [[MTLBinaryArchiveDescriptor new] autorelease];
    archived.url = [NSURL fileURLWithPath:binaries];
    id<MTLBinaryArchive> archive = library ? [[device newBinaryArchiveWithDescriptor:archived error:&error] autorelease] : nil;
    if (!archive) {
      *message = say("make -C rdma install, on this node, installs %s: %s", (library ? binaries : air).UTF8String,
                     error ? error.localizedDescription.UTF8String : "absent");
      return NULL;
    }
    struct kernels *made = calloc(1, sizeof *made);
    if (!made) { *message = "out of memory"; return NULL; }
    MTLComputePipelineDescriptor *described = [[MTLComputePipelineDescriptor new] autorelease];
    described.binaryArchives = @[archive];
    for (uint32_t kind = 0; kind < KINDS; kind++) {
      described.computeFunction = [[library newFunctionWithName:@(NAME[kind])] autorelease];
      error = nil;
      made->pipelines[kind] = described.computeFunction ? [device newComputePipelineStateWithDescriptor:described
                                                                                            options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                                                         reflection:nil error:&error] : nil;
      if (made->pipelines[kind]) continue;
      *message = say("make -C rdma install, on this node, makes %s anew: kernel %s: %s", binaries.UTF8String, NAME[kind],
                     error ? error.localizedDescription.UTF8String : "absent from the library");
      for (uint32_t k = 0; k < kind; k++) [made->pipelines[k] release];
      free(made);
      return NULL;
    }
    made->device = [device retain];
    kernels_[key] = [NSValue valueWithPointer:made];
    return made;
  }
}

/* the program's device's kernels (kernels_of), found at its first step and kept with it */
static struct kernels *program_kernels(struct metal_program *program, const char **message) {
  if (!program->kernels)
    program->kernels = kernels_of(program->residency.device, message);
  return program->kernels;
}

void *metal_wrap(void *address, size_t bytes) {
  return [(id<MTLDevice>)metal_device() newBufferWithBytesNoCopy:address length:bytes options:MTLResourceStorageModeShared deallocator:nil];
}

void *metal_event(void) { return [(id<MTLDevice>)metal_device() newSharedEvent]; }
uint64_t metal_signaled(void *event) { return ((id<MTLSharedEvent>)event).signaledValue; }
int metal_wait(void *event, uint64_t value, uint64_t ns) { return [(id<MTLSharedEvent>)event waitUntilSignaledValue:value timeoutMS:ns / 1000000]; }
void metal_release(void *object) { [(id)object release]; }
void *metal_retain(void *object) { return [(id)object retain]; }
/* The occurrence already owns its effective observation budget. Binding does
   not reinterpret configuration or start any observation. */
struct metal_program *metal_begin(struct metal_program *program, void *encoder, void *table, void *residency, void *violation, uint32_t polls,
                                  const char **message) {
  if (!kernels_of(((id<MTLResidencySet>)residency).device, message)) return NULL;
  *program = (struct metal_program){0};
  program->encoder = [(id<MTL4ComputeCommandEncoder>)encoder retain];
  program->table = [(id<MTL4ArgumentTable>)table retain];
  program->residency = [(id<MTLResidencySet>)residency retain];
  program->violation = [(id<MTLBuffer>)violation retain];
  program->polls = polls;
  program->accesses = [NSMutableData new];
  metal_resident(program, violation);
  return program;
}


void metal_resident(struct metal_program *program, void *buffer) {
  if (!program->residency || !buffer || [program->residency containsAllocation:(id<MTLBuffer>)buffer]) return;
  [program->residency addAllocation:(id<MTLBuffer>)buffer];
  program->added = 1;
}

void metal_gate(struct metal_program *program, void *violation) {
  if ((id<MTLBuffer>)violation == program->violation) return;
  [(id<MTLBuffer>)violation retain];
  [program->violation release];
  program->violation = (id<MTLBuffer>)violation;
  metal_resident(program, violation);
}

/* All argument tables and records use these function-owned pages. */
static int argument_pages(struct metal_program *program,size_t bytes,struct metal_bytes *made) {
  MeshArgumentPages *pages=objc_getAssociatedObject(program->residency,&argumentPagesKey);
  if(!pages) {
    pages=[MeshArgumentPages new];
    objc_setAssociatedObject(program->residency,&argumentPagesKey,pages,OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [pages release];
  }
  const size_t size=(bytes+RECORD-1)/RECORD*RECORD;
  if(!pages->count||size>pages->bytes-pages->used) {
    if(pages->count==pages->room) {
      const size_t room=pages->room?2*pages->room:1;
      void **held=realloc(pages->held,room*sizeof *held);
      if(!held)return ENOMEM;
      pages->held=held;pages->room=room;
    }
    void *piece=metal_piece_hold(&pages->page,&pages->bytes,size);
    if(!piece)return ENOMEM;
    pages->held[pages->count++]=piece;pages->used=0;
    metal_resident(program,pages->page.buffer);
  }
  *made=(struct metal_bytes){pages->page.buffer,pages->page.offset+pages->used,(char *)pages->page.contents+pages->used};
  pages->used+=size;
  return 0;
}
static struct metal_result record4(struct metal_program *program,id<MTLBuffer> d,id<MTLBuffer> s,id<MTLBuffer> a,id<MTLBuffer> words,
    struct args p,uint64_t bound[6],struct MetalCommandBinding *bindings) {
  struct metal_bytes record;
  const int failed=argument_pages(program,sizeof p,&record);
  if(failed)return metal_failed(failed);
  id<MTLBuffer> v = words ? words : program->violation ? program->violation : d;
  p.polls = v != d ? program->polls : 0;
  p.sequence=0;
  const uint64_t argument=((id<MTLBuffer>)record.buffer).gpuAddress+record.offset;
  memcpy(record.contents,&p,sizeof p);
  const id<MTLBuffer> used[4] = {d, s ? s : d, a ? a : d, v};
  for (int k = 0; k < 4; k++) metal_resident(program, used[k]);
  const uint64_t made[6] = {d.gpuAddress,(s?s:d).gpuAddress,argument,(a?a:d).gpuAddress,v.gpuAddress,argument};
  memcpy(bound, made, sizeof made);
  if(bindings) {
    const id<MTLBuffer> buffers[6]={d,s?s:d,(id<MTLBuffer>)record.buffer,a?a:d,v,(id<MTLBuffer>)record.buffer};
    for(uint32_t k=0;k<6;k++)bindings[k]=(struct MetalCommandBinding){.buffer=buffers[k],
      .offset=(k==2||k==5)?record.offset:0,.length=k==2?sizeof p:0,
      .index=k,.access=k==2?MetalReachRead:0};
  }
  return METAL_DONE;
}

/* Transport leaves enter the same prepared command graph as numerical leaves.
   Bindings carry storage; accesses name the exact data relation. */
static void command_emit(struct metal_program *program,id<MTLComputePipelineState> pipeline,
    const struct MetalCommandBinding *bindings,MTLSize grid,MTLSize threads,BOOL threadGrid) {
  NSMutableData *accesses=(NSMutableData *)program->accesses;
  const struct MetalCommandDescription command={.pipeline=pipeline,.bindings=bindings,.bindingCount=6,
    .reaches=accesses.bytes,.reachCount=accesses.length/sizeof(struct MetalReach),
    .groups=grid,.threads=threads,.threadGrid=threadGrid};
  MetalEncoderCompute(program->encoder,&command);
  [accesses setLength:0];
}

static struct metal_result dispatch4(struct metal_program *program,id<MTLComputePipelineState> state,id<MTLBuffer> d,id<MTLBuffer> s,
    id<MTLBuffer> a,id<MTLBuffer> words,struct args p,uint64_t units,int group) {
  uint64_t bound[6];struct MetalCommandBinding bindings[6];
  const struct metal_result recorded=record4(program,d,s,a,words,p,bound,bindings);
  if(recorded.status)return recorded;
  const NSUInteger most=group?MESH_TRANSPORT_GROUP_THREADS:MESH_TRANSPORT_ELEMENT_THREADS;
  const NSUInteger width=MIN(state.maxTotalThreadsPerThreadgroup,most);
  command_emit(program,state,bindings,MTLSizeMake(group?1:MIN(units,MESH_TRANSPORT_GRID_THREADS),1,1),
    MTLSizeMake(width,1,1),!group);
  return METAL_DONE;
}

/* kernel `kind` (KINDS_OF) on buffers d, s, a and v (0, 1, 3 and 4; an absent one d, and v, where none is given, the
   program's failure words: metal_gate) and the arguments p (2, its polls the program's where v is such words), the
   progress word (5): a GROUP kernel (`group`: publish,
   spin) as one threadgroup of the configured width, any other over `units` threads */
static struct metal_result dispatch(struct metal_program *program, uint32_t kind, void *d, void *s, void *a, void *v, struct args p,
                                    uint64_t units, int group) {
  if (kind >= KINDS) return (struct metal_result){EINVAL, "no kernel of this operation and data type (nccl-mesh-metal.m TYPED)"};
  if (!units) return METAL_DONE;
  const char *message = NULL;
  struct kernels *k = program_kernels(program, &message);
  if (!k) return (struct metal_result){EINVAL, message};
  id<MTLComputePipelineState> state = k->pipelines[kind];
  @autoreleasepool {
    return dispatch4(program, state, (id<MTLBuffer>)d, (id<MTLBuffer>)s, (id<MTLBuffer>)a, (id<MTLBuffer>)v, p, units, group);
  }
}

struct metal_result metal_copy(struct metal_program *program, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset,
                              size_t bytes,const struct metal_copy_relation *relation) {
  const int words = !((dst_offset | src_offset | bytes) & 3);
  if (mode < METAL_PLAIN || mode > METAL_LAND) return metal_failed(EINVAL);
  const struct metal_wait *ready=relation?relation->ready:NULL;
  const size_t observed=(ready?1:0)+(relation?relation->count:0);
  const ncclMeshAccess uses[]={{{src,src_offset},bytes,ncclMeshAccessRead|(relation?relation->source_access:0)},
    {{dst,dst_offset},bytes,ncclMeshAccessWritten}};
  metal_accesses(program,uses,sizeof uses/sizeof *uses);
  struct args args={.dst=dst_offset,.src=src_offset,.n=words?bytes/4:bytes};
  struct metal_bytes table={0};
  if(observed) {
    if(argument_pages(program,observed*6*sizeof(uint64_t),&table))return metal_failed(ENOMEM);
    uint64_t *values=table.contents;
    for(size_t k=0;k<observed;k++) {
      const struct metal_wait *wait=ready&&k==0?ready:relation->observations+k-(ready?1:0);
      values[6*k]=((id<MTLBuffer>)wait->words).gpuAddress+wait->word;
      values[6*k+1]=wait->target.n;
      values[6*k+2]=((id<MTLBuffer>)wait->violation).gpuAddress;
      values[6*k+3]=program->polls;values[6*k+4]=wait->channel;values[6*k+5]=wait->at;
      const ncclMeshAccess observation={{wait->words,wait->word},sizeof(uint64_t),ncclMeshAccessRead|ncclMeshAccessObserve};
      metal_accesses(program,&observation,1);metal_resident(program,wait->violation);
    }
    args.aux=table.offset;args.count=(uint32_t)observed;
  }
  return dispatch(program,COPY[!words][mode][observed!=0],dst,src,table.buffer,NULL,args,
                  words?bytes/4:bytes,0);
}

struct metal_result metal_fill_pattern(struct metal_program *program,void *dst,size_t dst_offset,
    size_t bytes,uint64_t scalar,size_t element_bytes,size_t unit_bytes) {
  const ncclMeshAccess output={{dst,dst_offset},bytes,ncclMeshAccessWritten};
  metal_accesses(program,&output,1);
  return dispatch(program,KIND_fill_pattern,dst,NULL,NULL,NULL,
    (struct args){.dst=dst_offset,.n=bytes,.scalar=scalar,.step=unit_bytes,.input=element_bytes},bytes,0);
}

/* operation `op`'s kernel of `type` (TYPED), KINDS where none is compiled */
static uint32_t typed(int op, int type) { return (unsigned)type <= METAL_MERGE ? TYPED[op][type] : KINDS; }

/* CPU arithmetic is another prepared leaf over the same region pages. The
   recorder owns dependencies, submission and completion for every backend. */
struct native_task { void (^operation)(void); MeshHostCompletion complete; void *context; };
static void native_submit(void *opaque) {
  struct native_task *task=opaque;task->operation();task->complete(task->context,NULL);
}
static void native_release(void *opaque) {
  struct native_task *task=opaque;[task->operation release];free(task);
}
static struct metal_result native_emit(struct metal_program *program,void (^operation)(void)) {
  NSData *accesses=(NSData *)program->accesses;
  MetalEncoderHost(program->encoder,accesses.bytes,accesses.length/sizeof(struct MetalReach),
    ^const char *(MeshHostCompletion complete,void *context,MeshHostTask *out) {
      struct native_task *task=calloc(1,sizeof *task);
      if(!task)return "allocating native command binding";
      task->operation=[operation copy];task->complete=complete;task->context=context;
      *out=(MeshHostTask){.context=task,.submit=native_submit,.release=native_release};return NULL;
    });
  [(NSMutableData *)program->accesses setLength:0];
  return METAL_DONE;
}

struct metal_result metal_premultiply(struct metal_program *program,int type,void *dst,size_t dst_offset,
    void *src,size_t src_offset,size_t n,size_t element_bytes,struct metal_scalar scalar) {
  if(!scalar.buffer) {
    struct metal_bytes constant;
    if(argument_pages(program,sizeof scalar.bits,&constant))return metal_failed(ENOMEM);
    memcpy(constant.contents,&scalar.bits,sizeof scalar.bits);
    scalar.buffer=constant.buffer;scalar.offset=constant.offset;
  }
  const ncclMeshAccess uses[]={{{src,src_offset},n*element_bytes,ncclMeshAccessRead},
    {{scalar.buffer,scalar.offset},element_bytes,ncclMeshAccessRead},{{dst,dst_offset},n*element_bytes,ncclMeshAccessWritten}};
  metal_accesses(program,uses,3);
  if(type==ncclFloat64) {
    double *out=(double *)((char *)metal_contents(dst)+dst_offset);
    const double *input=(const double *)((const char *)metal_contents(src)+src_offset);
    const double *factor=(const double *)((const char *)metal_contents(scalar.buffer)+scalar.offset);
    return native_emit(program,^{const double value=*factor;for(size_t k=0;k<n;k++)out[k]=input[k]*value;});
  }
  return dispatch(program,typed(OP_PREMUL,type),dst,src,scalar.buffer,NULL,
    (struct args){.dst=dst_offset,.src=src_offset,.n=n,.aux=scalar.offset},n,0);
}

struct metal_result metal_truncdiv(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                   size_t element_bytes, uint64_t divisor) {
  const ncclMeshAccess uses[]={{{src,src_offset},n*element_bytes,ncclMeshAccessRead},{{dst,dst_offset},n*element_bytes,ncclMeshAccessWritten}};
  metal_accesses(program,uses,sizeof uses/sizeof *uses);
  return dispatch(program, typed(OP_TRUNCDIV, type), dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = n, .scalar = divisor},
                  n, 0);
}

/* a wait's arguments (nccl-mesh-metal.h struct metal_wait): `landed` at aux until scalar,
   the landed bytes at src, dst the destination's offset, its entry and channel for its failure's record.  Its page
   counts become the record's words here and in release_args alone */
static struct args wait_args(const struct metal_wait *w, size_t dst_offset, size_t n, int op) {
  return (struct args){.dst = dst_offset, .src = w->data_offset, .n = n, .scalar = pages_to_word(w->target), .aux = w->word, .op = (uint32_t)op,
                       .at = w->at, .channel = w->channel, .stride = (uint32_t)pages_to_word(w->stride)};
}

/* a publication's arguments (struct metal_release): its count at aux stored scalar; dst the operand's offset (a deferred one's) */
static struct args release_args(const struct metal_release *r, size_t dst_offset) {
  return (struct args){.dst = dst_offset, .scalar = pages_to_word(r->value), .aux = r->word, .stride = (uint32_t)pages_to_word(r->stride)};
}

/* An access is a compiler operand; no overlap decision occurs in execution. */
void metal_accesses(struct metal_program *program,const ncclMeshAccess *uses,size_t n) {
  NSMutableData *accesses=(NSMutableData *)program->accesses;
  for(size_t k=0;k<n;k++)if(uses[k].bytes) {
    metal_resident(program,uses[k].storage.buffer);
    const uint32_t mode=(uses[k].access&ncclMeshAccessRead?MetalReachRead:0)|
      (uses[k].access&ncclMeshAccessWritten?MetalReachWritten:0)|(uses[k].access&ncclMeshAccessObserve?MetalReachObserve:0)|(uses[k].access&ncclMeshAccessNext?MetalReachNext:0)|(uses[k].access&ncclMeshAccessAtomic?MetalReachAtomic:0);
    const struct MetalReach reach={(id<MTLBuffer>)uses[k].storage.buffer,uses[k].storage.offset,uses[k].bytes,mode,-1};
    [accesses appendBytes:&reach length:sizeof reach];
  }
}
struct metal_result metal_publish(struct metal_program *program, const struct metal_release *r) {
  const ncclMeshAccess use={{r->words,r->word},sizeof(uint64_t),ncclMeshAccessWritten|ncclMeshAccessAtomic};metal_accesses(program,&use,1);
  return dispatch(program, KIND_publish, r->words, NULL, NULL, NULL, release_args(r, 0), 1, 1);
}

/* Many overwritten packets may name one completion word. Preparation joins
   their exact output ranges into one threshold column for that word. Runtime
   scans words relevant to the cell, never all packets of the tensor. */
struct output_observation { struct metal_wait wait; size_t word; };
struct output_observations { const struct metal_wait *word; size_t count,spans;
  const struct mesh_graph_span *span; const uint64_t *reach; const struct output_observation *values; };
static int output_observation_order(const void *left,const void *right) {
  const struct output_observation *a=left,*b=right;
  return (a->wait.data_offset>b->wait.data_offset)-(a->wait.data_offset<b->wait.data_offset);
}
static struct output_observations output_observations(const struct metal_zelloth_sum *sum) {
  NSMutableData *words=[NSMutableData data],*values=[NSMutableData dataWithLength:sum->waits*sizeof(struct output_observation)];
  struct output_observation *v=values.mutableBytes;
  for(size_t i=0;i<sum->waits;i++) {
    const struct metal_wait wait=sum->wait[i];const struct metal_wait *known=words.bytes;
    size_t word=0,count=words.length/sizeof wait;
    while(word<count&&(known[word].words!=wait.words||known[word].word!=wait.word))word++;
    if(word==count)[words appendBytes:&wait length:sizeof wait];
    v[i]=(struct output_observation){wait,word};
  }
  qsort(v,sum->waits,sizeof *v,output_observation_order);
  NSMutableData *spans=[NSMutableData dataWithLength:sum->waits*sizeof(struct mesh_graph_span)],
    *reach=[NSMutableData dataWithLength:sum->waits*sizeof(uint64_t)];
  struct mesh_graph_span *span=spans.mutableBytes;
  for(size_t i=0;i<sum->waits;i++)span[i]=(struct mesh_graph_span){v[i].wait.data_offset-sum->out_offset,
    v[i].wait.data_offset-sum->out_offset+v[i].wait.bytes};
  mesh_graph_span_reach(span,sum->waits,reach.mutableBytes);
  return (struct output_observations){words.bytes,words.length/sizeof(struct metal_wait),sum->waits,
    spans.bytes,reach.bytes,values.bytes};
}
struct output_observation_meet { const struct output_observations *observations; uint64_t *threshold; };
static void output_observation_met(void *context,size_t i) {
  const struct output_observation_meet *meet=context;
  const struct output_observation value=meet->observations->values[i];
  const uint64_t target=pages_to_word(value.wait.target);
  if(target>meet->threshold[value.word])meet->threshold[value.word]=target;
}
struct landing_cells_output { uint64_t *at; size_t inputs; size_t element_bytes; const struct output_observations *observations; };
static void landing_cell(void *context,uint64_t first,uint64_t elements,const uint64_t *through) {
  struct landing_cells_output *out=context;
  out->at[0]=first;out->at[1]=elements;
  memcpy(out->at+2,through,out->inputs*sizeof *through);
  memset(out->at+2+out->inputs,0,out->observations->count*sizeof *out->at);
  struct output_observation_meet meet={out->observations,out->at+2+out->inputs};
  mesh_graph_meets(out->observations->span,out->observations->reach,out->observations->spans,
    (struct mesh_graph_span){first*out->element_bytes,(first+elements)*out->element_bytes},output_observation_met,&meet);
  out->at+=2+out->inputs+out->observations->count;
}
typedef void (*native_fold)(double *,const double *,const double *const *,size_t,size_t,size_t);
#define NATIVE_FOLD(name,expression) \
static void native_##name(double *out,const double *own,const double *const *inputs,size_t K,size_t first,size_t count) { \
  for(size_t j=first;j<first+count;j++){double value=own[j];for(size_t k=0;k<K;k++){const double next=inputs[k][j];value=(expression);}out[j]=value;} \
}
NATIVE_FOLD(sum,value+next)
NATIVE_FOLD(product,value*next)
NATIVE_FOLD(maximum,fmax(value,next))
NATIVE_FOLD(minimum,fmin(value,next))
#undef NATIVE_FOLD
/* Concurrent cells publish a monotone prefix to the ordered wire. The state
   records only completed values; it never assigns or chooses numerical work. */
static void native_prefix(uint32_t *state,uint32_t *prefix,uint32_t count,uint32_t cell,
    const uint64_t *cells,size_t K,size_t observations,uint64_t *const *words) {
  __atomic_store_n(state+cell,2,__ATOMIC_RELEASE);
  const uint32_t at=mesh_graph_prefix(state,0,count,prefix);
  if(!at)return;
  const uint64_t *through=cells+(2+observations)*(at-1)+2;
  for(size_t k=0;k<K;k++)if(words[k])mesh_publish(words[k],through[k]);
}
static struct metal_result native_zelloth(struct metal_program *program,int op,const struct metal_zelloth_sum *sum,
    uint32_t channel,uint64_t entry) {
  static native_fold const folds[]={native_sum,native_product,native_maximum,native_minimum};
  const struct output_observations observations=output_observations(sum);
  const size_t K=sum->inputs,W=observations.count,O=K+W;
  struct mesh_landing_map maps[K];size_t cursor[K];uint64_t through[K];
  for(size_t k=0;k<K;k++)maps[k]=(struct mesh_landing_map){sum->in[k].runs,sum->in[k].count,
    pages_to_word(pages_sub(sum->in[k].target,sum->in[k].packets))};
  const uint32_t count=(uint32_t)mesh_landing_cells(maps,K,MESH_TRANSPORT_ELEMENT_THREADS*MESH_TRANSPORT_VALUES_PER_THREAD,cursor,through,NULL,NULL);
  if(!count)return METAL_DONE;
  const size_t words=(2+O)*count,states=words+(count+2)/2;
  struct metal_bytes table;
  if(argument_pages(program,states*sizeof(uint64_t),&table))return metal_failed(ENOMEM);
  uint64_t *cells=table.contents;
  struct landing_cells_output output={cells,K,sizeof(double),&observations};
  mesh_landing_cells(maps,K,MESH_TRANSPORT_ELEMENT_THREADS*MESH_TRANSPORT_VALUES_PER_THREAD,cursor,through,landing_cell,&output);
  uint32_t *state=(uint32_t *)(cells+words),*prefix=state+count;
  memset(state,0,(count+1)*sizeof *state);
  struct metal_bytes pointers;
  if(argument_pages(program,(3*K+W)*sizeof(void *),&pointers))return metal_failed(ENOMEM);
  const double **inputs=pointers.contents;
  uint64_t **progress=(uint64_t **)(inputs+K),**ready=progress+K;
  void *violation=metal_contents(program->violation);
  const uint32_t polls=program->polls;
  for(size_t k=0;k<K;k++) {
    inputs[k]=(const double *)((const char *)metal_contents(sum->in[k].data)+sum->in[k].data_offset);
    progress[k]=sum->in[k].summed?(uint64_t *)((char *)metal_contents(sum->in[k].summed)+sum->in[k].summed_offset):NULL;
    ready[k]=(uint64_t *)((char *)metal_contents(sum->in[k].words)+sum->in[k].word);
    metal_resident(program,sum->in[k].summed);metal_resident(program,sum->in[k].words);
  }
  for(size_t k=0;k<W;k++)ready[K+k]=(uint64_t *)((char *)metal_contents(observations.word[k].words)+observations.word[k].word);
  struct metal_bytes waits={0};
  if(W&&argument_pages(program,W*sizeof(struct metal_wait),&waits))return metal_failed(ENOMEM);
  if(W)memcpy(waits.contents,observations.word,W*sizeof(struct metal_wait));
  const struct metal_wait *overwrite=waits.contents;
  const double *own=(const double *)((const char *)metal_contents(sum->own)+sum->own_offset);
  double *out=(double *)((char *)metal_contents(sum->out)+sum->out_offset);
  native_fold fold=folds[op];metal_resident(program,table.buffer);
  for(uint32_t q=0;q<count;q++) {
    const uint64_t *cell=cells+(2+O)*q;const size_t first=cell[0],elements=cell[1],offset=first*sizeof(double),bytes=elements*sizeof(double);
    const ncclMeshAccess uses[]={{{sum->own,sum->own_offset+offset},bytes,ncclMeshAccessRead},
      {{sum->out,sum->out_offset+offset},bytes,ncclMeshAccessWritten}};
    metal_accesses(program,uses,2);
    for(size_t k=0;k<K;k++) {
      const ncclMeshAccess input={{sum->in[k].data,sum->in[k].data_offset+offset},bytes,ncclMeshAccessRead};
      metal_accesses(program,&input,1);
      const ncclMeshAccess words[]={
        {{sum->in[k].words,sum->in[k].word},sizeof(uint64_t),ncclMeshAccessRead|ncclMeshAccessObserve},
        {{sum->in[k].summed,sum->in[k].summed_offset},sum->in[k].summed?sizeof(uint64_t):0,ncclMeshAccessWritten|ncclMeshAccessAtomic}};
      metal_accesses(program,words,2);
    }
    for(size_t k=0;k<W;k++)if(cell[2+K+k]) {
      const ncclMeshAccess use={{observations.word[k].words,observations.word[k].word},sizeof(uint64_t),ncclMeshAccessRead|ncclMeshAccessObserve};
      metal_accesses(program,&use,1);
    }
    const ncclMeshAccess completion={
      {table.buffer,table.offset+words*sizeof(uint64_t)},(count+1)*sizeof(uint32_t),
      ncclMeshAccessRead|ncclMeshAccessWritten|ncclMeshAccessAtomic};
    metal_accesses(program,&completion,1);
    struct metal_result made=native_emit(program,^{
      for(size_t k=0;k<K;k++)mesh_step_observe(ready[k],cell[2+k],violation,polls,channel,entry);
      for(size_t k=0;k<W;k++)if(cell[2+K+k])mesh_step_observe(ready[K+k],cell[2+K+k],violation,polls,overwrite[k].channel,overwrite[k].at);
      fold(out,own,inputs,K,first,elements);
      native_prefix(state,prefix,count,q,cells,K,O,progress);
    });
    if(made.status)return made;
  }
  return METAL_DONE;
}

/* One ordinary command per value version. Each threadgroup has its prepared
   cell, with every input threshold already resolved. There is no task claim,
   run search or dependency interpreter in the numerical kernel. */
struct metal_result metal_zelloth(struct metal_program *program,int type,int op,size_t element_bytes,uint32_t S,
    const struct metal_zelloth_sum *sums,uint32_t channel,uint64_t at) {
  if(type==ncclFloat64) {
    for(uint32_t s=0;s<S;s++){struct metal_result made=native_zelloth(program,op,sums+s,channel,at);if(made.status)return made;}
    return METAL_DONE;
  }
  const char *message=NULL;
  struct kernels *kernels=program_kernels(program,&message);
  if(!kernels)return (struct metal_result){EINVAL,message};
  const uint32_t kind=type==METAL_MERGE?KIND_zelloth_merge:
    (unsigned)type<ncclNumTypes&&(unsigned)op<4?REDUCTIONS[op][type]:KINDS;
  if(kind>=KINDS)return metal_failed(EINVAL);
  id<MTLComputePipelineState> pipeline=kernels->pipelines[kind];
  const uint64_t width=MIN(pipeline.maxTotalThreadsPerThreadgroup,MESH_TRANSPORT_GROUP_THREADS);
  const uint64_t block=(type==METAL_MERGE?1:MESH_TRANSPORT_VALUES_PER_THREAD)*width;
  for(uint32_t s=0;s<S;s++) {
    const struct metal_zelloth_sum *sum=sums+s;
    const struct output_observations observations=output_observations(sum);
    const size_t K=sum->inputs,W=observations.count,O=K+W;
    struct mesh_landing_map *maps=malloc((K+1)*sizeof *maps);
    size_t *cursor=malloc((K+1)*sizeof *cursor);
    uint64_t *through=malloc((K+1)*sizeof *through);
    if(!maps||!cursor||!through){free(maps);free(cursor);free(through);return metal_failed(ENOMEM);}
    for(size_t k=0;k<K;k++)maps[k]=(struct mesh_landing_map){sum->in[k].runs,sum->in[k].count,
      pages_to_word(pages_sub(sum->in[k].target,sum->in[k].packets))};
    const uint64_t blocks=mesh_landing_cells(maps,K,block,cursor,through,NULL,NULL);
    if(!blocks){free(maps);free(cursor);free(through);continue;}
    const size_t cells=8+3*O,state=cells+(2+O)*blocks,prefix=state+(blocks+1)/2,total=prefix+1;
    struct metal_bytes table;
    if(argument_pages(program,total*sizeof(uint64_t),&table)){free(maps);free(cursor);free(through);return metal_failed(ENOMEM);}
    uint64_t *z=table.contents;
    z[0]=K;z[1]=blocks;z[2]=((id<MTLBuffer>)sum->own).gpuAddress+sum->own_offset;
    z[3]=((id<MTLBuffer>)sum->out).gpuAddress+sum->out_offset;
    z[4]=cells;z[5]=state;z[6]=prefix;z[7]=W;
    struct landing_cells_output output={z+cells,K,element_bytes,&observations};
    mesh_landing_cells(maps,K,block,cursor,through,landing_cell,&output);
    free(maps);free(cursor);free(through);
    memset(z+state,0,(total-state)*sizeof *z);
    const struct mesh_landing_run *last=sum->in[0].runs+sum->in[0].count-1;
    const size_t bytes=(last->first+last->elements)*element_bytes;
    const ncclMeshAccess own[]={{{sum->own,sum->own_offset},bytes,ncclMeshAccessRead},
      {{sum->out,sum->out_offset},bytes,ncclMeshAccessWritten}};
    metal_accesses(program,own,2);
    for(size_t k=0;k<K;k++) {
      const struct metal_zelloth_input *input=sum->in+k;
      z[8+3*k]=((id<MTLBuffer>)input->data).gpuAddress+input->data_offset;
      z[9+3*k]=((id<MTLBuffer>)input->words).gpuAddress+input->word;
      z[10+3*k]=input->summed?((id<MTLBuffer>)input->summed).gpuAddress+input->summed_offset:0;
      const ncclMeshAccess uses[]={{{input->data,input->data_offset},bytes,ncclMeshAccessRead},
        {{input->words,input->word},sizeof(uint64_t),ncclMeshAccessRead|ncclMeshAccessObserve},
        {{input->summed,input->summed_offset},input->summed?sizeof(uint64_t):0,ncclMeshAccessWritten|ncclMeshAccessAtomic}};
      metal_accesses(program,uses,3);
    }
    for(size_t k=0;k<W;k++) {
      const struct metal_wait *wait=observations.word+k;
      z[8+3*(K+k)]=((id<MTLBuffer>)wait->words).gpuAddress+wait->word;
      z[9+3*(K+k)]=wait->channel;z[10+3*(K+k)]=wait->at;
      const ncclMeshAccess use={{wait->words,wait->word},sizeof(uint64_t),ncclMeshAccessRead|ncclMeshAccessObserve};
      metal_accesses(program,&use,1);
    }
    metal_resident(program,table.buffer);
    uint64_t bound[6];struct MetalCommandBinding bindings[6];
    const struct args args={.src=table.offset,.op=(uint32_t)op,.channel=channel,.at=at,.polls=program->polls};
    const struct metal_result recorded=record4(program,(id<MTLBuffer>)sum->out,(id<MTLBuffer>)table.buffer,nil,
      program->violation,args,bound,bindings);
    if(recorded.status)return recorded;
    command_emit(program,pipeline,bindings,MTLSizeMake(blocks,1,1),MTLSizeMake(width,1,1),NO);
  }
  return METAL_DONE;
}

struct metal_result metal_open_gate(struct metal_program *program, void *words, size_t word, uint64_t value) {
  const ncclMeshAccess use={{words,word},sizeof(uint64_t),ncclMeshAccessWritten|ncclMeshAccessAtomic};metal_accesses(program,&use,1);
  return dispatch(program, KIND_publish, words, NULL, NULL, NULL, (struct args){.scalar = value, .aux = word}, 1, 1);
}

struct metal_result metal_spin(struct metal_program *program, const struct metal_wait *w) {
  const ncclMeshAccess uses[]={{{w->words,w->word},sizeof(uint64_t),ncclMeshAccessRead|ncclMeshAccessObserve},
    {{w->data,w->data_offset},w->bytes,ncclMeshAccessWritten}};
  metal_accesses(program,uses,sizeof uses/sizeof *uses);
  return dispatch(program, KIND_spin, w->words, w->words, NULL, w->violation, wait_args(w, 0, 1, 0), 1, 1);
}
/* Deferred steps retain their compiled records through the function's residency set. */
struct metal_result metal_send_deferred(struct metal_program *program, void *operand, size_t offset, const struct metal_release *r, uint64_t bound[6]) {
  return record4(program, (id<MTLBuffer>)operand, nil, (id<MTLBuffer>)r->words, nil, release_args(r, offset), bound, NULL);
}

struct metal_result metal_land_deferred(struct metal_program *program, void *landing, size_t offset, const struct metal_wait *w, uint64_t bound[6]) {
  return record4(program, (id<MTLBuffer>)landing, (id<MTLBuffer>)landing, (id<MTLBuffer>)w->words, (id<MTLBuffer>)w->violation,
                 wait_args(w, offset, 0, 0), bound, NULL);
}


void *metal_contents(void *buffer) { return buffer ? ((id<MTLBuffer>)buffer).contents : NULL; }

/* mesh.metal's text as this library was built (.build/mesh_metal.h, xxd of it), then STEPS: made once, kept */
const char *metal_prelude(void) {
  static NSString *text;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    NSString *words = [[[NSString alloc] initWithBytes:mesh_metal length:mesh_metal_len encoding:NSUTF8StringEncoding] autorelease];
    words = [@(MetalCompilationPrelude()) stringByAppendingString:words];
    text = [[NSString alloc] initWithFormat:@"%@\n%@", words, STEPS];
  });
  return text.UTF8String;
}

void metal_end(struct metal_program *program) {
  if (program->added) [program->residency commit];
  [(id)program->accesses release];
  [program->encoder release]; [program->table release]; [program->residency release]; [program->violation release];
}

