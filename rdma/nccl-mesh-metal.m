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
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/* a typed kernel's kind by its operation and its ncclDataType_t (int8 uint8 int32 uint32 int64 uint64 float16 float32
   float64 bfloat16 e4m3 e5m2, then the partials merge's rows, METAL_MERGE), KINDS where none is compiled: float64's,
   truncdiv's but the integers', premul's of the merge */
enum { OP_PREMUL, OP_TRUNCDIV, OP_ZELLOTH, OPS };
#define INTEGERS(op) KIND_##op##_i8, KIND_##op##_u8, KIND_##op##_i32, KIND_##op##_u32, KIND_##op##_i64, KIND_##op##_u64
#define FLOATS(op) KIND_##op##_f16, KIND_##op##_f32, KINDS, KIND_##op##_bf16, KIND_##op##_e4m3, KIND_##op##_e5m2
static const uint8_t TYPED[OPS][METAL_MERGE + 1] = {
  {INTEGERS(premul), FLOATS(premul), KINDS},
  {INTEGERS(truncdiv), KINDS, KINDS, KINDS, KINDS, KINDS, KINDS, KINDS}, {INTEGERS(zelloth), FLOATS(zelloth), KIND_zelloth_merge}};
/* a copy's kind by its width (0 words of 4 bytes, 1 bytes) and its loads (METAL_PLAIN, METAL_LAND) */
static const uint8_t COPY[2][METAL_LAND + 1] = {{KIND_copy_plain_4, KIND_copy_land_4}, {KIND_copy_plain_1, KIND_copy_land_1}};

ARGS(args, uint64_t, uint32_t);
/* Kernel lookup is shared; compiled argument pages, pieces of the region (nccl-mesh.c metal_piece_hold), belong to
   their function's residency set, are never recycled while that function can execute, and are given back once that
   set ends. */
#define RECORD MESH_METAL_ARGUMENT_BYTES
/* a device's kernels, and its GPU's cores (mesh-fabric.c mesh_gpu_cores): zelloth's most threadgroups, one a core,
   so every one is resident beside every other and beside the GPU's other clients (metal_zelloth) */
struct kernels { id<MTLDevice> device; id<MTLComputePipelineState> pipelines[KINDS]; uint32_t cores; };
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
    const uint32_t cores = mesh_gpu_cores();
    if (!cores) { *message = "IORegistry's AGXAccelerator states no gpu-core-count (mesh_gpu_cores)"; return NULL; }
    struct kernels *made = calloc(1, sizeof *made);
    if (!made) { *message = "out of memory"; return NULL; }
    made->cores = cores;
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
/* a stream's polls as a program's bound: 0, or more than NCCL_MESH_POLLS, is NCCL_MESH_POLLS
   (design/transport.md) */
static uint32_t bound_of(uint32_t polls) { return polls && polls < NCCL_MESH_POLLS ? polls : NCCL_MESH_POLLS; }

struct metal_program *metal_begin(struct metal_program *program, void *encoder, void *table, void *residency, void *violation, uint32_t polls,
                                  const char **message) {
  if (!kernels_of(((id<MTLResidencySet>)residency).device, message)) return NULL;
  *program = (struct metal_program){0};
  program->encoder = [(id<MTL4ComputeCommandEncoder>)encoder retain];
  program->table = [(id<MTL4ArgumentTable>)table retain];
  program->residency = [(id<MTLResidencySet>)residency retain];
  program->violation = [(id<MTLBuffer>)violation retain];
  program->polls = bound_of(polls);
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

/* Immutable argument records in pages owned by this function's residency set.
   Bound addresses use the shared six-field step layout. No record is selected
   for reuse from a GPU progress counter. */
static struct metal_result record4(struct metal_program *program, id<MTLBuffer> d, id<MTLBuffer> s, id<MTLBuffer> a, id<MTLBuffer> words,
                                   struct args p, uint64_t bound[6]) {
  MeshArgumentPages *pages=objc_getAssociatedObject(program->residency,&argumentPagesKey);
  if(!pages) {
    pages=[MeshArgumentPages new];
    objc_setAssociatedObject(program->residency,&argumentPagesKey,pages,OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [pages release];
  }
  if(!pages->count || pages->used+RECORD>pages->bytes) {
    void **grown=pages->count<pages->room?pages->held:realloc(pages->held,(pages->count+4)*sizeof *grown);
    if(grown&&grown!=pages->held){pages->held=grown;pages->room=pages->count+4;}
    void *piece=grown?metal_piece_hold(&pages->page,&pages->bytes):NULL;
    if(!piece)return metal_failed(ENOMEM);
    pages->held[pages->count++]=piece;pages->used=0;
    metal_resident(program,pages->page.buffer);
  }
  id<MTLBuffer> v = words ? words : program->violation ? program->violation : d;
  p.polls = v != d ? program->polls : 0;
  p.sequence=0;
  const uint64_t argument=((id<MTLBuffer>)pages->page.buffer).gpuAddress+pages->page.offset+pages->used;
  memcpy((char *)pages->page.contents+pages->used,&p,sizeof p);pages->used+=RECORD;
  const id<MTLBuffer> used[4] = {d, s ? s : d, a ? a : d, v};
  for (int k = 0; k < 4; k++) metal_resident(program, used[k]);
  const uint64_t made[6] = {d.gpuAddress,(s?s:d).gpuAddress,argument,(a?a:d).gpuAddress,v.gpuAddress,argument};
  memcpy(bound, made, sizeof made);
  return METAL_DONE;
}

/* design/runtime-waits-2026-10-30.md: the barriers a group's dispatches take are fixed by their kinds where the group
   is compiled, never found by comparing access ranges while encoding: one before the group's first present (its
   receives let the NIC land into bytes earlier work used) and before every other direct dispatch (an operand copy, a
   gate, a guard's spin, a fused landing, an epilogue: each orders after what came before it), none before a later
   present (presents write disjoint lists and wait on their predecessors' words) nor before a group's landing waits
   (metal_spins).  3f2a90b7's placement less its barrier at an empty begin. */
static void metal_barrier(struct metal_program *program) {
  [(id<MTL4ComputeCommandEncoder>)program->encoder barrierAfterEncoderStages:MTLStageDispatch|MTLStageBlit
      beforeEncoderStages:MTLStageDispatch|MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
}
static void metal_order(struct metal_program *program) {
  if (program->presenting && program->presented++) return;
  metal_barrier(program);
}

/* The same prepared range frontier orders direct dispatches and recorded work. */
static struct metal_result dispatch4(struct metal_program *program, id<MTLComputePipelineState> state, id<MTLBuffer> d, id<MTLBuffer> s,
                                     id<MTLBuffer> a, id<MTLBuffer> words, struct args p, uint64_t units, int group) {
  uint64_t bound[6];
  const struct metal_result recorded = record4(program, d, s, a, words, p, bound);
  if (recorded.status) return recorded;
  metal_order(program);
  for (NSUInteger k = 0; k < 6; k++) [program->table setAddress:bound[k] atIndex:k];
  id<MTL4ComputeCommandEncoder> encoder = program->encoder;
  [encoder setComputePipelineState:state];
  [encoder setArgumentTable:program->table];
  const NSUInteger most = group ? MESH_TRANSPORT_GROUP_THREADS : MESH_TRANSPORT_ELEMENT_THREADS, width = state.maxTotalThreadsPerThreadgroup < most ? state.maxTotalThreadsPerThreadgroup : most;
  if (group) [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  else [encoder dispatchThreads:MTLSizeMake(units < MESH_TRANSPORT_GRID_THREADS ? (NSUInteger)units : MESH_TRANSPORT_GRID_THREADS, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
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

struct metal_result metal_copy(struct metal_program *program, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes) {
  const int words = !((dst_offset | src_offset | bytes) & 3);
  if (mode < METAL_PLAIN || mode > METAL_LAND) return metal_failed(EINVAL);
  const ncclMeshAccess uses[]={{{src,src_offset},bytes,ncclMeshAccessRead},{{dst,dst_offset},bytes,ncclMeshAccessWritten}};
  metal_accesses(program,uses,sizeof uses/sizeof *uses);
  return dispatch(program, COPY[!words][mode], dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = words ? bytes / 4 : bytes},
                  words ? bytes / 4 : bytes, 0);
}

/* operation `op`'s kernel of `type` (TYPED), KINDS where none is compiled */
static uint32_t typed(int op, int type) { return (unsigned)type <= METAL_MERGE ? TYPED[op][type] : KINDS; }

struct metal_result metal_premultiply(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                      size_t element_bytes, uint64_t scalar) {
  const ncclMeshAccess uses[]={{{src,src_offset},n*element_bytes,ncclMeshAccessRead},{{dst,dst_offset},n*element_bytes,ncclMeshAccessWritten}};
  metal_accesses(program,uses,sizeof uses/sizeof *uses);
  return dispatch(program, typed(OP_PREMUL, type), dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = n, .scalar = scalar},
                  n, 0);
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

/* nccl-mesh-metal.h metal_boundary: the group's operands entered in the caller's schedule's frontier, which bars the
   caller's later work that meets them after the group's waits */
void metal_boundary(struct metal_program *program,const ncclMeshAccess *uses,size_t n) {
  struct MetalReach reach[n?n:1];
  for(size_t k=0;k<n;k++)reach[k]=(struct MetalReach){(id)uses[k].storage.buffer,uses[k].storage.offset,uses[k].bytes,
      (uses[k].access&ncclMeshAccessRead?MetalReachRead:0)|(uses[k].access&ncclMeshAccessWritten?MetalReachWritten:0),-1};
  if(n)MetalEncoderOrder(program->encoder,reach,n);
}
/* a dispatch's buffers made resident; their ranges order nothing here (metal_order) */
void metal_accesses(struct metal_program *program,const ncclMeshAccess *uses,size_t n) {
  for(size_t k=0;k<n;k++)if(uses[k].bytes)metal_resident(program,uses[k].storage.buffer);
}
struct metal_result metal_publish(struct metal_program *program, const struct metal_release *r) {
  const ncclMeshAccess use={{r->words,r->word},sizeof(uint64_t),ncclMeshAccessWritten};metal_accesses(program,&use,1);
  return dispatch(program, KIND_publish, r->words, NULL, NULL, NULL, release_args(r, 0), 1, 1);
}

struct metal_result metal_publish_presented(struct metal_program *program, const struct metal_release *r) {
  const ncclMeshAccess use={{r->words,r->word},sizeof(uint64_t),ncclMeshAccessWritten};metal_accesses(program,&use,1);
  program->presenting = 1;
  const struct metal_result result = dispatch(program, KIND_publish, r->words, NULL, NULL, NULL, release_args(r, 0), 1, 1);
  program->presenting = 0;
  return result;
}

struct metal_result metal_zelloth(struct metal_program *program, int type, int op, size_t element_bytes, uint32_t S,
                                  const struct metal_zelloth_sum *sum, uint32_t channel, uint64_t at) {
  const uint32_t kind = typed(OP_ZELLOTH, type);
  if (kind >= KINDS || !S) return metal_failed(EINVAL);
  const char *message = NULL;
  struct kernels *k = program_kernels(program, &message);
  if (!k) return (struct metal_result){EINVAL, message};
  id<MTLComputePipelineState> state = k->pipelines[kind];
  const uint64_t width = state.maxTotalThreadsPerThreadgroup < MESH_TRANSPORT_GROUP_THREADS ? state.maxTotalThreadsPerThreadgroup
                                                                                           : MESH_TRANSPORT_GROUP_THREADS;
  /* a block's elements: a combine's lanes a thread, a merge's one row a thread */
  const uint64_t block = (type == METAL_MERGE ? 1 : MESH_COMBINE_LANES) * width;
  size_t words = 4 + 9 * (size_t)S, runs = 0;
  for (uint32_t s = 0; s < S; s++) {
    if (!sum[s].inputs || !sum[s].in[0].count) return metal_failed(EINVAL);
    uint64_t mine = 0;
    for (size_t r = 0; r < sum[s].in[0].count; r++) mine += (sum[s].in[0].runs[r].elements + block - 1) / block;
    words += 3 * (size_t)sum[s].inputs + sum[s].in[0].count * (2 + 4 * (size_t)sum[s].inputs) +
             (sum[s].waits ? 1 + (2 + mine) * (size_t)sum[s].waits : 0);
    runs += sum[s].in[0].count;
  }
  /* the blocks' cells in their global order: every sum's runs in order, a segment each (mesh-graph.h
     mesh_graph_cells), each cell naming its sum's and its run's words (`of`, a run's two) */
  uint64_t *length = malloc((runs + 1) * sizeof *length), *of = malloc((2 * runs + 1) * sizeof *of);
  if (!length || !of) { free(length); free(of); return metal_failed(ENOMEM); }
  size_t n = 0;
  for (uint32_t s = 0; s < S; s++)
    for (size_t r = 0; r < sum[s].in[0].count; r++) length[n++] = sum[s].in[0].runs[r].elements;
  const uint64_t blocks = mesh_graph_cells(length, runs, block, NULL, NULL);
  uint64_t *segment = malloc((blocks + 1) * sizeof *segment), *place = malloc((blocks + 1) * sizeof *place);
  struct metal_bytes table;
  /* after the cells: each sum's done word, the threadgroups' `ended`, `low`, then each block's state, each sum's
     prefix and its first unclaimed block (32-bit words), all zero (nccl-mesh-kernels.h ZELLOTH_OF) */
  const size_t states = 2 + ((size_t)blocks + 2 * (size_t)S + 1) / 2;
  int status = !segment || !place || metal_carve((words + 3 * blocks + S + states) * sizeof(uint64_t), &table) ? ENOMEM : 0;
  uint64_t *z = status ? NULL : table.contents;
  const uint64_t base = status ? 0 : ((id<MTLBuffer>)table.buffer).gpuAddress + table.offset;
  if (!status) metal_resident(program, table.buffer);
  const size_t done = words + 3 * blocks;
  uint64_t first = 0;
  size_t off = 4 + 9 * (size_t)S, g = 0;
  for (uint32_t s = 0; !status && s < S; s++) {
    const struct metal_zelloth_sum *x = sum + s;
    const uint32_t K = x->inputs;
    const size_t R = x->in[0].count;
    uint64_t mine = 0, elements = 0;
    for (size_t r = 0; r < R; r++) { mine += (x->in[0].runs[r].elements + block - 1) / block; elements = x->in[0].runs[r].first + x->in[0].runs[r].elements; }
    uint64_t *h = z + 4 + 9 * (size_t)s;
    z[done + s] = 0;
    h[0] = K; h[1] = R; h[2] = ((id<MTLBuffer>)x->own).gpuAddress + x->own_offset; h[3] = ((id<MTLBuffer>)x->out).gpuAddress + x->out_offset;
    h[4] = base + (done + s) * sizeof(uint64_t); h[5] = first; h[6] = mine; h[7] = off;
    const ncclMeshAccess uses[] = {{{x->own, x->own_offset}, elements * element_bytes, ncclMeshAccessRead},
                                   {{x->out, x->out_offset}, elements * element_bytes, ncclMeshAccessWritten}};
    metal_accesses(program, uses, sizeof uses / sizeof *uses);
    uint64_t *in = z + off;
    for (uint32_t i = 0; !status && i < K; i++) {
      const struct metal_zelloth_input *y = x->in + i;
      if (y->count != R) { status = EINVAL; break; }
      in[3 * i] = ((id<MTLBuffer>)y->data).gpuAddress + y->data_offset;
      in[3 * i + 1] = ((id<MTLBuffer>)y->words).gpuAddress + y->word;
      in[3 * i + 2] = y->summed ? ((id<MTLBuffer>)y->summed).gpuAddress + y->summed_offset : 0;
      const ncclMeshAccess inputs[] = {{{y->data, y->data_offset}, elements * element_bytes, ncclMeshAccessRead},
        {{y->words, y->word}, sizeof(uint64_t), ncclMeshAccessRead | ncclMeshAccessObserve},
        {{y->summed, y->summed_offset}, y->summed ? sizeof(uint64_t) : 0, ncclMeshAccessWritten}};
      metal_accesses(program, inputs, sizeof inputs / sizeof *inputs);
    }
    for (size_t r = 0; !status && r < R; r++, g++) {
      const size_t at = off + 3 * (size_t)K + r * (2 + 4 * (size_t)K);
      uint64_t *run = z + at;
      of[2 * g] = 4 + 9 * (size_t)s; of[2 * g + 1] = at;
      run[0] = x->in[0].runs[r].first; run[1] = x->in[0].runs[r].elements;
      for (uint32_t i = 0; i < K; i++) {
        const struct mesh_landing_run *y = x->in[i].runs + r;
        if (y->first != run[0] || y->elements != run[1]) { status = EINVAL; break; }
        uint64_t *w = run + 2 + 4 * (size_t)i;
        w[0] = pages_to_word(pages_sub(x->in[i].target, x->in[i].packets)) + y->packet; w[1] = y->phase; w[2] = y->step; w[3] = y->shift;
      }
    }
    first += mine;
    off += 3 * (size_t)K + R * (2 + 4 * (size_t)K);
    /* its waits (nccl-mesh-metal.h struct metal_zelloth_wait): each an earlier sum of the same buffer, its done word
       and, for each of this sum's blocks, the count of that sum's blocks starting at or before its last element
       (mesh-graph.h mesh_graph_through), as the pool's combine followed an earlier one chunk by chunk; that sum's
       whole where its blocks do not start in order (nccl-mesh-kernels.h zelloth_needs) */
    h[8] = x->waits ? off : 0;
    if (x->waits) z[off] = x->waits;
    size_t thresholds = off + 1 + 2 * (size_t)x->waits;
    for (uint32_t d = 0; !status && d < x->waits; d++, thresholds += mine) {
      const ptrdiff_t a = x->wait[d].after - sum;
      const int64_t apart = (int64_t)x->out_offset - (int64_t)sum[a < 0 ? 0 : a].out_offset;
      if (a < 0 || a >= (ptrdiff_t)s || sum[a].out != x->out || apart % (int64_t)element_bytes) { status = EINVAL; break; }
      z[off + 1 + 2 * d] = base + (done + (size_t)a) * sizeof(uint64_t); z[off + 2 + 2 * d] = thresholds;
      /* the runs as segments (mesh-graph.h mesh_graph_cells_through) */
      const size_t RA = sum[a].in[0].count;
      uint64_t *runs_of = malloc((2 * (RA + R) + 1) * sizeof *runs_of);
      if (!runs_of) { status = ENOMEM; break; }
      for (size_t r = 0; r < RA; r++) { runs_of[r] = sum[a].in[0].runs[r].first; runs_of[RA + r] = sum[a].in[0].runs[r].elements; }
      for (size_t r = 0; r < R; r++) { runs_of[2 * RA + r] = x->in[0].runs[r].first; runs_of[2 * RA + R + r] = x->in[0].runs[r].elements; }
      (void)mesh_graph_cells_through(runs_of, runs_of + RA, RA, runs_of + 2 * RA, runs_of + 2 * RA + R, R,
                                     apart / (int64_t)element_bytes, block, z + thresholds);
      free(runs_of);
    }
    off += x->waits ? 1 + (2 + mine) * (size_t)x->waits : 0;
  }
  if (!status) {
    mesh_graph_cells(length, runs, block, segment, place);
    for (uint64_t q = 0; q < blocks; q++) {
      uint64_t *cell = z + words + 3 * q;
      cell[0] = of[2 * segment[q]]; cell[1] = of[2 * segment[q] + 1]; cell[2] = place[q];
    }
    z[0] = S; z[1] = block; z[2] = blocks; z[3] = words;
    memset(z + done + S, 0, states * sizeof(uint64_t));
    /* the table read until every threadgroup has left it */
    metal_done((const _Atomic uint64_t *)(z + done + S), 1, blocks < k->cores ? blocks : k->cores);
  }
  free(length); free(of); free(segment); free(place);
  if (status) return metal_failed(status);
  uint64_t bound[6];
  struct args p = {.src = table.offset, .op = (uint32_t)op, .polls = program->polls, .channel = channel, .at = at};
  const struct metal_result recorded = record4(program, (id<MTLBuffer>)sum[0].out, (id<MTLBuffer>)table.buffer, nil, program->violation, p, bound);
  if (recorded.status) return recorded;
  metal_order(program);
  for (NSUInteger a = 0; a < 6; a++) [program->table setAddress:bound[a] atIndex:a];
  id<MTL4ComputeCommandEncoder> encoder = program->encoder;
  [encoder setComputePipelineState:state];
  [encoder setArgumentTable:program->table];
  const uint64_t groups = blocks < k->cores ? blocks : k->cores;
  [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  return METAL_DONE;
}

struct metal_result metal_open_gate(struct metal_program *program, void *words, size_t word, uint64_t value) {
  const ncclMeshAccess use={{words,word},sizeof(uint64_t),ncclMeshAccessWritten};metal_accesses(program,&use,1);
  return dispatch(program, KIND_publish, words, NULL, NULL, NULL, (struct args){.scalar = value, .aux = word}, 1, 1);
}

/* nccl-mesh-metal.h metal_link: presenting, so metal_order bars the first store alone */
struct metal_result metal_link(struct metal_program *program, const struct metal_link *links, size_t n) {
  struct metal_result made = METAL_DONE;
  program->presenting = 1; program->presented = 0;
  for (size_t k = 0; k < n && !made.status; k++) made = metal_open_gate(program, links[k].words, links[k].word, links[k].value);
  program->presenting = 0;
  return made;
}

struct metal_result metal_spin(struct metal_program *program, const struct metal_wait *w) {
  const ncclMeshAccess uses[]={{{w->words,w->word},sizeof(uint64_t),ncclMeshAccessRead|ncclMeshAccessObserve},
    {{w->data,w->data_offset},w->bytes,ncclMeshAccessWritten}};
  metal_accesses(program,uses,sizeof uses/sizeof *uses);
  return dispatch(program, KIND_spin, w->words, w->words, NULL, w->violation, wait_args(w, 0, 1, 0), 1, 1);
}
/* nccl-mesh-metal.h metal_spins: presenting with one already presented, so metal_order bars none of them */
struct metal_result metal_spins(struct metal_program *program, const struct metal_wait *w, size_t n) {
  const int presenting = program->presenting, presented = program->presented;
  program->presenting = 1; program->presented = 1;
  struct metal_result made = METAL_DONE;
  for (size_t k = 0; k < n && !made.status; k++) made = metal_spin(program, w + k);
  program->presenting = presenting; program->presented = presented;
  return made;
}

/* Deferred steps retain their compiled records through the function's residency set. */
struct metal_result metal_send_deferred(struct metal_program *program, void *operand, size_t offset, const struct metal_release *r, uint64_t bound[6]) {
  return record4(program, (id<MTLBuffer>)operand, nil, (id<MTLBuffer>)r->words, nil, release_args(r, offset), bound);
}

struct metal_result metal_land_deferred(struct metal_program *program, void *landing, size_t offset, const struct metal_wait *w, uint64_t bound[6]) {
  return record4(program, (id<MTLBuffer>)landing, (id<MTLBuffer>)landing, (id<MTLBuffer>)w->words, (id<MTLBuffer>)w->violation,
                 wait_args(w, offset, 0, 0), bound);
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
  [program->encoder release]; [program->table release]; [program->residency release]; [program->violation release];
}

