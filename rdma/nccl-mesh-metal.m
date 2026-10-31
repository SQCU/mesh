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
enum { OP_LAND, OP_PREMUL, OP_TRUNCDIV, OP_ZELLOTH, OPS };
#define INTEGERS(op) KIND_##op##_i8, KIND_##op##_u8, KIND_##op##_i32, KIND_##op##_u32, KIND_##op##_i64, KIND_##op##_u64
#define FLOATS(op) KIND_##op##_f16, KIND_##op##_f32, KINDS, KIND_##op##_bf16, KIND_##op##_e4m3, KIND_##op##_e5m2
static const uint8_t TYPED[OPS][METAL_MERGE + 1] = {
  {INTEGERS(land), FLOATS(land), KIND_land_merge}, {INTEGERS(premul), FLOATS(premul), KINDS},
  {INTEGERS(truncdiv), KINDS, KINDS, KINDS, KINDS, KINDS, KINDS, KINDS}, {INTEGERS(zelloth), FLOATS(zelloth), KIND_zelloth_merge}};
/* a copy's kind by its width (0 words of 4 bytes, 1 bytes) and its loads (METAL_PLAIN, METAL_LAND) */
static const uint8_t COPY[2][METAL_LAND + 1] = {{KIND_copy_plain_4, KIND_copy_land_4}, {KIND_copy_plain_1, KIND_copy_land_1}};

ARGS(args, uint64_t, uint32_t);
/* Kernel lookup is shared; compiled argument pages, pieces of the region (nccl-mesh.c metal_piece_hold), belong to
   their function's residency set, are never recycled while that function can execute, and are given back once that
   set ends. */
#define RECORD MESH_METAL_ARGUMENT_BYTES
/* a device's kernels, and its GPU's cores (mesh-fabric.c mesh_gpu_cores): the tracks pool's most
   workers, one a core, so every worker is resident beside every other and beside the GPU's other clients
   (metal_tracks) */
struct kernels { id<MTLDevice> device; id<MTLComputePipelineState> pipelines[PIPELINES]; uint32_t cores; };
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
    for (uint32_t kind = 0; kind < PIPELINES; kind++) {
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
   present (presents write disjoint lists and wait on their predecessors' words), and the pool's before and after it
   (metal_tracks).  3f2a90b7's placement less its barrier at an empty begin. */
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
static struct metal_result record_step(struct metal_program *program, uint32_t kind, void *d, void *s, void *a, void *v, struct args p,
                                       uint64_t units, int group);
static int recording_on(const struct metal_program *program);
static void recording_join(struct metal_program *program, int join);
static void recording_summed(struct metal_program *program, void *summed, size_t offset);
static struct metal_result dispatch(struct metal_program *program, uint32_t kind, void *d, void *s, void *a, void *v, struct args p,
                                    uint64_t units, int group) {
  if (kind >= KINDS) return (struct metal_result){EINVAL, "no kernel of this operation and data type (nccl-mesh-metal.m TYPED)"};
  if (!units) return METAL_DONE;
  if (recording_on(program)) return record_step(program, kind, d, s, a, v, p, units, group);
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

/* nccl-mesh-metal.h metal_boundary: the group's operands entered in the caller's schedule's frontier, the barrier that
   ordering may call for the pool's own, which follows (metal_tracks) */
void metal_boundary(struct metal_program *program,const ncclMeshAccess *uses,size_t n) {
  struct MetalReach reach[n?n:1];
  for(size_t k=0;k<n;k++)reach[k]=(struct MetalReach){(id)uses[k].storage.buffer,uses[k].storage.offset,uses[k].bytes,
      (uses[k].access&ncclMeshAccessRead?MetalReachRead:0)|(uses[k].access&ncclMeshAccessWritten?MetalReachWritten:0),-1};
  if(n)MetalEncoderOrder(program->encoder,reach,n);
  program->declared=1;
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
  /* a block's elements: a combine's lanes a thread, a merge's one row a thread (LANES) */
  const uint64_t block = (type == METAL_MERGE ? 1 : MESH_COMBINE_LANES) * width;
  size_t words = 4 + 8 * (size_t)S, runs = 0;
  for (uint32_t s = 0; s < S; s++) {
    if (!sum[s].inputs || !sum[s].in[0].count) return metal_failed(EINVAL);
    words += 3 * (size_t)sum[s].inputs + sum[s].in[0].count * (2 + 4 * (size_t)sum[s].inputs);
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
  int status = !segment || !place || metal_carve((words + 3 * blocks + S) * sizeof(uint64_t), &table) ? ENOMEM : 0;
  uint64_t *z = status ? NULL : table.contents;
  const uint64_t base = status ? 0 : ((id<MTLBuffer>)table.buffer).gpuAddress + table.offset;
  if (!status) metal_resident(program, table.buffer);
  const size_t done = words + 3 * blocks;
  uint64_t first = 0;
  size_t off = 4 + 8 * (size_t)S, g = 0;
  for (uint32_t s = 0; !status && s < S; s++) {
    const struct metal_zelloth_sum *x = sum + s;
    const uint32_t K = x->inputs;
    const size_t R = x->in[0].count;
    uint64_t mine = 0, elements = 0;
    for (size_t r = 0; r < R; r++) { mine += (x->in[0].runs[r].elements + block - 1) / block; elements = x->in[0].runs[r].first + x->in[0].runs[r].elements; }
    uint64_t *h = z + 4 + 8 * (size_t)s;
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
      of[2 * g] = 4 + 8 * (size_t)s; of[2 * g + 1] = at;
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
  }
  if (!status) {
    mesh_graph_cells(length, runs, block, segment, place);
    for (uint64_t q = 0; q < blocks; q++) {
      uint64_t *cell = z + words + 3 * q;
      cell[0] = of[2 * segment[q]]; cell[1] = of[2 * segment[q] + 1]; cell[2] = place[q];
    }
    z[0] = S; z[1] = block; z[2] = blocks; z[3] = words;
    metal_done((const _Atomic uint64_t *)(z + done), S, 1);
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

struct metal_result metal_land(struct metal_program *program, int type, int op, void *dst, size_t dst_offset,
                               void *with, size_t with_offset,
                               size_t element_bytes, mesh_pages packets, const struct mesh_landing_run *runs, size_t count, const struct metal_wait *w,
                               void *summed, size_t summed_offset) {
  recording_summed(program, summed, summed_offset);
  const uint32_t kind = type == METAL_MERGE && (with != dst || with_offset != dst_offset) ? KIND_land_merge_from : typed(OP_LAND, type);
  struct metal_result result = METAL_DONE;
  for (size_t k = 0; k < count; k++) {
    const struct mesh_landing_run run = runs[k];
    const size_t first=run.first*element_bytes,bytes=run.elements*element_bytes;
    const ncclMeshAccess uses[]={{{with,with_offset+first},bytes,ncclMeshAccessRead},
      {{w->data,w->data_offset+first},bytes,ncclMeshAccessRead},{{dst,dst_offset+first},bytes,ncclMeshAccessWritten},
      {{w->words,w->word},sizeof(uint64_t),ncclMeshAccessRead|ncclMeshAccessObserve},
      {{summed,summed_offset},summed?sizeof(uint64_t):0,ncclMeshAccessWritten}};
    metal_accesses(program,uses,sizeof uses/sizeof *uses);
    struct args p = wait_args(w, dst_offset + run.first * element_bytes, run.elements, op);
    p.src += run.first * element_bytes;
    p.input = ((id<MTLBuffer>)with).gpuAddress + with_offset + run.first * element_bytes;
    p.scalar = pages_to_word(pages_sub(w->target, packets)) + run.packet;
    p.phase = run.phase; p.step = run.step; p.mask = run.shift;
    recording_join(program, k > 0);
    result = dispatch(program, kind, dst, w->data, w->words, w->violation, p, type == METAL_MERGE ? run.elements : (run.elements + LANES[kind] - 1) / LANES[kind], 0);
    if (result.status) break;
  }
  recording_join(program, 0);
  recording_summed(program, NULL, 0);
  return result;
}

/* Deferred steps retain their compiled records through the function's residency set. */
struct metal_result metal_send_deferred(struct metal_program *program, void *operand, size_t offset, const struct metal_release *r, uint64_t bound[6]) {
  return record4(program, (id<MTLBuffer>)operand, nil, (id<MTLBuffer>)r->words, nil, release_args(r, offset), bound);
}

struct metal_result metal_land_deferred(struct metal_program *program, void *landing, size_t offset, const struct metal_wait *w, uint64_t bound[6]) {
  return record4(program, (id<MTLBuffer>)landing, (id<MTLBuffer>)landing, (id<MTLBuffer>)w->words, (id<MTLBuffer>)w->violation,
                 wait_args(w, offset, 0, 0), bound);
}

/* the tracks being recorded (nccl-mesh-metal.h metal_record): each step as its dispatch would bind it (`v` its failure
   words, else the program's; its polls the program's where it has such words), its dispatch's threads (`units`) and
   whether it is a group kernel, and whether it runs beside the step before it (`joined`: a landing's runs, disjoint
   ranges each awaiting its own pages, metal_land); each item's position, `need`, steps, dependencies and failure
   words; each track's items by cursor; the buffers named, made resident at metal_tracks */
struct host_op { void *d, *s, *a, *v; struct args p; uint64_t units; uint32_t kind; int group, joined; void *summed; size_t summed_offset; };
struct host_item { uint64_t position, need; void *v; uint32_t first_op, ops, first_dep, deps; uint32_t after_track; uint64_t after_position; int after; };
struct host_dep { uint64_t count; uint32_t other; };
struct host_track { uint32_t *item[2]; size_t count[2], room[2]; };
struct recording { struct host_op *op; size_t ops, op_room; struct host_item *item; size_t items, item_room; struct host_dep *dep; size_t deps, dep_room;
                   struct host_track *track; uint32_t tracks; uint32_t current; int on, join; NSMutableSet *named;
                   void *summed; size_t summed_offset; };

static int recording_on(const struct metal_program *program) { return program->recording && program->recording->on; }
static void recording_join(struct metal_program *program, int join) { if (program->recording) program->recording->join = join; }
/* the summed word the steps recorded next take (record_step), NULL none: a combine's landing's (metal_land) */
static void recording_summed(struct metal_program *program, void *summed, size_t offset) {
  if (program->recording) { program->recording->summed = summed; program->recording->summed_offset = offset; }
}

static void *more(void *items, size_t *room, size_t count, size_t size) {
  if (count < *room) return items;
  const size_t bigger = *room ? 2 * *room : 16;
  void *made = realloc(items, bigger * size);
  if (made) *room = bigger;
  return made;
}

static void recording_free(struct recording *r) {
  if (!r) return;
  for (uint32_t t = 0; t < r->tracks; t++) { free(r->track[t].item[0]); free(r->track[t].item[1]); }
  free(r->track); free(r->op); free(r->item); free(r->dep);
  [r->named release];
  free(r);
}

/* a step of kind `kind` over `units` threads (a group kernel's one threadgroup) recorded onto the item being recorded */
static struct metal_result record_step(struct metal_program *program, uint32_t kind, void *d, void *s, void *a, void *v, struct args p,
                                       uint64_t units, int group) {
  struct recording *r = program->recording;
  if (!v) v = (void *)program->violation;
  p.polls = v && v != d ? program->polls : 0;
  struct host_op *ops = more(r->op, &r->op_room, r->ops, sizeof *ops);
  if (!ops) return metal_failed(ENOMEM);
  r->op = ops;
  struct host_item *it = r->item + r->current;
  const struct host_op made = {d, s ? s : d, a ? a : d, v ? v : d, p, units, kind, group, r->join && it->ops, r->summed, r->summed_offset};
  r->op[r->ops++] = made;
  const id named[4] = {(id)made.d, (id)made.s, (id)made.a, (id)made.v};
  for (int k = 0; k < 4; k++) [r->named addObject:named[k]];
  if (made.summed) [r->named addObject:(id)made.summed];
  it->ops++;
  if (!it->v && p.polls) it->v = made.v;
  return METAL_DONE;
}

struct metal_result metal_record(struct metal_program *program, uint32_t track, int cursor, uint64_t position, uint64_t need) {
  if (track == METAL_NONE) { if (program->recording) program->recording->on = 0; return METAL_DONE; }
  if (!program->recording) {
    if (!(program->recording = calloc(1, sizeof *program->recording))) return metal_failed(ENOMEM);
    program->recording->named = [NSMutableSet new];
  }
  struct recording *r = program->recording;
  if (track >= r->tracks) {
    struct host_track *grown = realloc(r->track, ((size_t)track + 1) * sizeof *grown);
    if (!grown) return metal_failed(ENOMEM);
    memset(grown + r->tracks, 0, ((size_t)track + 1 - r->tracks) * sizeof *grown);
    r->track = grown; r->tracks = track + 1;
  }
  struct host_track *t = r->track + track;
  uint32_t *listed = more(t->item[cursor], &t->room[cursor], t->count[cursor], sizeof *listed);
  struct host_item *items = listed ? more(r->item, &r->item_room, r->items, sizeof *items) : NULL;
  if (listed) t->item[cursor] = listed;
  if (!items) return metal_failed(ENOMEM);
  r->item = items;
  r->item[r->items] = (struct host_item){.position = position, .need = need, .first_op = (uint32_t)r->ops, .first_dep = (uint32_t)r->deps};
  t->item[cursor][t->count[cursor]++] = (uint32_t)r->items;
  r->current = (uint32_t)r->items++;
  r->on = 1;
  return METAL_DONE;
}

struct metal_result metal_after(struct metal_program *program, uint32_t other, uint64_t position) {
  struct recording *r = program->recording;
  if (!r || !r->on) return metal_failed(EINVAL);
  struct host_item *it = r->item + r->current;
  it->after = 1; it->after_track = other; it->after_position = position;
  return METAL_DONE;
}

struct metal_result metal_depend(struct metal_program *program, uint32_t other, uint64_t count) {
  struct recording *r = program->recording;
  if (!r || !r->on) return metal_failed(EINVAL);
  struct host_dep *deps = more(r->dep, &r->dep_room, r->deps, sizeof *deps);
  if (!deps) return metal_failed(ENOMEM);
  r->dep = deps;
  r->dep[r->deps++] = (struct host_dep){count, other};
  r->item[r->current].deps++;
  return METAL_DONE;
}

/* the records as the track kernel reads them (nccl-mesh-kernels.h TRACKS: track_head, track_cursor, track_task,
   track_item, track_op, track_dep), then the state its workers share: each track's consumption count, each cursor's
   task, each task's chunks claimed and finished, the count of tasks finished, each worker's end, and each chunk's mark
   (finished) and each task's prefix of marked chunks, which a combine's landing counts its summed units by: its op's
   `summed` word, the receive's, where a SEND names it to send each unit of the sum on as it is made (nccl-mesh.c
   forwards) */
struct track_head { uint64_t cursors_off, tasks_off, items_off, ops_off, deps_off, count_off, state_off, claim_off, done_off, progress_off,
                    ended_off, marks_off, prefix_off; uint32_t cursors, polls; };
struct track_cursor { uint32_t first, end, track, kind; };
struct track_task { uint32_t first_op, ops, chunks, item, cursor, flags, marks, after; };
struct track_item { uint64_t position, need, v; uint32_t first_dep, deps; };
struct track_op { uint64_t d, s, a, v; struct args p; uint64_t word, block; uint32_t kind, chunks; uint64_t summed; };
struct track_dep { uint64_t count; uint32_t other, pad; };
_Static_assert(sizeof(struct track_head) == 112 && sizeof(struct track_cursor) == 16 && sizeof(struct track_task) == 32 &&
               sizeof(struct track_item) == 32 && sizeof(struct track_op) == sizeof(struct args) + 8 * sizeof(uint64_t) &&
               sizeof(struct track_dep) == 16, "the track records");
/* a task's first and last runs of its item, and whether its chunks' pages decide which may be claimed (TRACK_KERNEL) */
enum { TASK_FIRST = 1, TASK_LAST = 2, TASK_PACED = 4 };

static uint64_t up(uint64_t x) { return (x + 255) & ~(uint64_t)255; }

/* a step's chunks at `width` threads a worker: a group kernel's one; a landing's a block each, so each awaits its own
   pages; any other step's its dispatch's threadgroups, at most `widest`, as that dispatch's grid is */
static uint32_t op_chunks(const struct host_op *o, uint64_t width, uint64_t widest) {
  const uint64_t groups = (o->units + width - 1) / width;
  return o->group ? 1 : (uint32_t)(LANES[o->kind] || groups < widest ? groups : widest);
}

struct metal_result metal_tracks(struct metal_program *program, uint32_t tracks, const uint64_t *consumed) {
  struct recording *r = program->recording;
  program->recording = NULL;
  if (!r || !r->tracks) { recording_free(r); return METAL_DONE; }
  @autoreleasepool {
    const char *message = NULL;
    struct kernels *k = program_kernels(program, &message);
    if (!k) { recording_free(r); return (struct metal_result){EINVAL, message}; }
    id<MTLComputePipelineState> state = k->pipelines[KIND_tracks];
    /* a worker the widest threadgroup its pipeline takes, at most a group kernel's: one a core, each claim's chunk
       as wide as a core holds (a landing's block 4096 elements at 1024 threads), so a claim's scan and fences are
       spread over four times the elements of a 256-thread chunk */
    const uint64_t width = state.maxTotalThreadsPerThreadgroup < MESH_TRANSPORT_GROUP_THREADS ? state.maxTotalThreadsPerThreadgroup
                                                                                             : MESH_TRANSPORT_GROUP_THREADS;
    const uint64_t T = tracks, C = 2 * T, widest = MESH_TRANSPORT_GRID_THREADS / width;
    /* the tasks: each item's runs of joined steps (one for an item of none, which raises its count alone), and every
       chunk they hold; a step's chunks are its dispatch's threadgroups (at most `widest`, a dispatch's most), and the
       workers at most one a GPU core.  Workers spin while no task is ready, so a pool past the cores that can hold them
       at once only adds spinners: sized by its chunks (up to 4096), it ran the five-node all-reduce of 256 KiB in
       25 ms where the one-threadgroup-a-track form it replaced (e576a6f1) took 0.29 ms, and its 64 MiB in 92 ms against
       5.6 (design/measurement-regressions-2026-10-30.md) */
    uint64_t tasks = 0, chunks = 0;
    for (uint32_t t = 0; t < r->tracks; t++)
      for (int c = 0; c < 2; c++)
        for (size_t x = 0; x < r->track[t].count[c]; x++) {
          const struct host_item *it = r->item + r->track[t].item[c][x];
          tasks += !it->ops; chunks += !it->ops;
          for (uint32_t o = 0; o < it->ops; o++) {
            tasks += !r->op[it->first_op + o].joined;
            chunks += op_chunks(r->op + it->first_op + o, width, widest);
          }
        }
    const uint64_t workers = chunks < k->cores ? chunks : k->cores;
    const uint64_t at_cursors = up(sizeof(struct track_head)), at_tasks = up(at_cursors + C * sizeof(struct track_cursor));
    const uint64_t at_items = up(at_tasks + tasks * sizeof(struct track_task)), at_ops = up(at_items + r->items * sizeof(struct track_item));
    const uint64_t at_deps = up(at_ops + r->ops * sizeof(struct track_op)), at_count = up(at_deps + r->deps * sizeof(struct track_dep));
    const uint64_t at_state = up(at_count + T * sizeof(uint64_t)), at_claim = up(at_state + C * sizeof(uint32_t));
    const uint64_t at_done = up(at_claim + tasks * sizeof(uint32_t)), at_progress = up(at_done + tasks * sizeof(uint32_t));
    const uint64_t at_ended = up(at_progress + sizeof(uint32_t)), at_marks = up(at_ended + workers * sizeof(uint64_t));
    const uint64_t at_prefix = up(at_marks + chunks * sizeof(uint32_t)), bytes = up(at_prefix + tasks * sizeof(uint32_t));
    /* the records, a carved piece of the region, done with once every worker has ended (each stores its end word) */
    struct metal_bytes records;
    if (metal_carve(bytes, &records)) { recording_free(r); return metal_failed(ENOMEM); }
    unsigned char *base = records.contents;
    memset(base + at_state, 0, bytes - at_state);
    metal_done((const _Atomic uint64_t *)(base + at_ended), (uint32_t)workers, 1);
    const uint64_t address = ((id<MTLBuffer>)records.buffer).gpuAddress + records.offset;
    *(struct track_head *)base = (struct track_head){at_cursors, at_tasks, at_items, at_ops, at_deps, at_count, at_state, at_claim, at_done,
                                                     at_progress, at_ended, at_marks, at_prefix, (uint32_t)C, program->polls};
    struct track_cursor *cursor = (struct track_cursor *)(base + at_cursors);
    struct track_task *task = (struct track_task *)(base + at_tasks);
    struct track_item *item = (struct track_item *)(base + at_items);
    uint32_t *current = (uint32_t *)(base + at_state), placed = 0, laid = 0, marked = 0;
    /* each placed item's track, cursor, position and first task, for the items whose chunks follow another's (after) */
    struct placed_item { uint32_t track, cursor, task; uint64_t position; const struct host_item *it; };
    struct placed_item *placing = calloc(r->items + 1, sizeof *placing);
    if (!placing) { recording_free(r); return metal_failed(ENOMEM); }
    for (uint32_t t = 0; t < T; t++)
      for (uint32_t c = 0; c < 2; c++) {
        const uint32_t cu = 2 * t + c;
        cursor[cu] = (struct track_cursor){laid, laid, t, c};
        current[cu] = laid;
        for (size_t x = 0; t < r->tracks && x < r->track[t].count[c]; x++, placed++) {
          const struct host_item *it = r->item + r->track[t].item[c][x];
          item[placed] = (struct track_item){it->position, it->need, it->v ? ((id<MTLBuffer>)it->v).gpuAddress : 0, it->first_dep, it->deps};
          placing[placed] = (struct placed_item){t, c, laid, it->position, it};
          if (!it->ops) task[laid++] = (struct track_task){.chunks = 1, .item = placed, .cursor = cu, .flags = TASK_FIRST | TASK_LAST, .marks = marked++,
                                                            .after = ~0u};
          for (uint32_t o = 0; o < it->ops;) {
            struct track_task *made = task + laid++;
            *made = (struct track_task){.first_op = it->first_op + o, .item = placed, .cursor = cu, .flags = o ? 0 : TASK_FIRST, .marks = marked,
                                        .after = ~0u};
            do {
              const struct host_op *op = r->op + it->first_op + o;
              made->ops++; made->chunks += op_chunks(op, width, widest);
              if (LANES[op->kind]) made->flags |= TASK_PACED;
            } while (++o < it->ops && r->op[it->first_op + o].joined);
            if (o == it->ops) made->flags |= TASK_LAST;
            marked += made->chunks;
          }
        }
        cursor[cu].end = laid;
      }
    /* a consumption whose chunks follow another's (metal_after): its first task's chunk c is claimed once that one's
       first task has finished its first c + 1 chunks (TRACK_KERNEL's prefix), the two the same elements in the same
       runs (nccl-mesh.c consume); one whose earlier consumption is not in this pool fails as a value */
    for (uint32_t x = 0; x < placed; x++) {
      if (!placing[x].it->after) continue;
      uint32_t found = ~0u;
      for (uint32_t y = 0; y < placed && found == ~0u; y++)
        if (placing[y].cursor == 1 && placing[y].track == placing[x].it->after_track && placing[y].position == placing[x].it->after_position)
          found = placing[y].task;
      if (found == ~0u || task[found].chunks != task[placing[x].task].chunks) {
        free(placing); recording_free(r);
        return (struct metal_result){EINVAL, "a combine follows an earlier one chunk by chunk that its tracks pool does not hold alike"};
      }
      task[placing[x].task].after = found;
    }
    free(placing);
    struct track_op *op = (struct track_op *)(base + at_ops);
    for (size_t x = 0; x < r->ops; x++) {
      const struct host_op *o = r->op + x;
      void *const word = WORD[o->kind] == WORD_A ? o->a : WORD[o->kind] == WORD_D && o->p.scalar ? o->d : NULL;
      op[x] = (struct track_op){((id<MTLBuffer>)o->d).gpuAddress, ((id<MTLBuffer>)o->s).gpuAddress, ((id<MTLBuffer>)o->a).gpuAddress,
                                ((id<MTLBuffer>)o->v).gpuAddress, o->p, word ? ((id<MTLBuffer>)word).gpuAddress + o->p.aux : 0,
                                LANES[o->kind] * width, o->kind, op_chunks(o, width, widest),
                                o->summed ? ((id<MTLBuffer>)o->summed).gpuAddress + o->summed_offset : 0};
    }
    struct track_dep *dep = (struct track_dep *)(base + at_deps);
    for (size_t x = 0; x < r->deps; x++) dep[x] = (struct track_dep){r->dep[x].count, r->dep[x].other, 0};
    uint64_t *count = (uint64_t *)(base + at_count);
    for (uint32_t t = 0; t < T; t++) count[t] = consumed[t];
    metal_resident(program, records.buffer);
    for (id b in r->named) metal_resident(program, (void *)b);
    [program->table setAddress:address atIndex:0];
    [program->table setAddress:address atIndex:1];
    id<MTL4ComputeCommandEncoder> encoder = program->encoder;
    [encoder setComputePipelineState:state];
    [encoder setArgumentTable:program->table];
    metal_barrier(program);
    [encoder dispatchThreadgroups:MTLSizeMake(workers, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    /* after the pool: a barrier, unless its group declared its operands to the caller's schedule (metal_boundary) */
    if (!program->declared) metal_barrier(program);
    program->declared = 0;
  }
  recording_free(r);
  return METAL_DONE;
}

size_t metal_tracks_open(const void *records, const void *ended, uint32_t workers, char *out, size_t room) {
  const unsigned char *base = records;
  const struct track_head *h = records;
  if (!room || (uintptr_t)ended != (uintptr_t)base + h->ended_off) return 0;
  const _Atomic uint64_t *end = ended;
  uint32_t live = 0;
  for (uint32_t w = 0; w < workers; w++) live += !atomic_load_explicit(end + w, memory_order_acquire);
  if (!live) return 0;
  const struct track_cursor *cursor = (const struct track_cursor *)(base + h->cursors_off);
  const struct track_task *task = (const struct track_task *)(base + h->tasks_off);
  const struct track_item *item = (const struct track_item *)(base + h->items_off);
  const struct track_op *op = (const struct track_op *)(base + h->ops_off);
  const struct track_dep *dep = (const struct track_dep *)(base + h->deps_off);
  const _Atomic uint64_t *count = (const _Atomic uint64_t *)(base + h->count_off);
  const _Atomic uint32_t *state = (const _Atomic uint32_t *)(base + h->state_off), *claim = (const _Atomic uint32_t *)(base + h->claim_off),
                         *done = (const _Atomic uint32_t *)(base + h->done_off);
  /* a pool no worker has begun (every cursor at its first task, none of it claimed) names nothing: the pools of the
     work encoded after a stall are all such */
  int begun = live < workers;
  for (uint32_t cu = 0; !begun && cu < h->cursors; cu++) {
    const uint32_t j = atomic_load_explicit(state + cu, memory_order_acquire);
    begun = j != cursor[cu].first || (j < cursor[cu].end && atomic_load_explicit(claim + j, memory_order_acquire));
  }
  if (!begun) return 0;
  size_t used = 0;
#define TOLD(...) do { if (used + 1 < room) { const int n_ = snprintf(out + used, room - used, __VA_ARGS__); \
                         if (n_ > 0) used += (size_t)n_ < room - used ? (size_t)n_ : room - used - 1; } } while (0)
  TOLD("tracks pool with %u of %u workers not ended, its tracks' counts", live, workers);
  for (uint32_t t = 0; t < h->cursors / 2; t++) TOLD("%s%llu", t ? ", " : " ", (unsigned long long)atomic_load_explicit(count + t, memory_order_acquire));
  for (uint32_t cu = 0; cu < h->cursors; cu++) {
    const uint32_t j = atomic_load_explicit(state + cu, memory_order_acquire);
    if (j >= cursor[cu].end) continue;
    const struct track_task *x = task + j;
    const struct track_item *it = item + x->item;
    TOLD("; track %u's %s at task %u of [%u, %u): position %llu, need %llu, %u of %u chunks claimed and %u done",
         cursor[cu].track, cursor[cu].kind ? "consumptions" : "publications", j, cursor[cu].first, cursor[cu].end,
         (unsigned long long)it->position, (unsigned long long)it->need, atomic_load_explicit(claim + j, memory_order_acquire), x->chunks,
         atomic_load_explicit(done + j, memory_order_acquire));
    for (uint32_t k = 0; k < it->deps; k++)
      TOLD(", awaiting track %u's count %llu (now %llu)", dep[it->first_dep + k].other, (unsigned long long)dep[it->first_dep + k].count,
           (unsigned long long)atomic_load_explicit(count + dep[it->first_dep + k].other, memory_order_acquire));
    if (x->ops && op[x->first_op].kind < KINDS)
      TOLD(", its %s step's count %llu%s", NAME[op[x->first_op].kind], (unsigned long long)op[x->first_op].p.scalar,
           op[x->first_op].word ? " awaited" : "");
  }
#undef TOLD
  return used;
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
  recording_free(program->recording);
  program->recording = NULL;
  if (program->added) [program->residency commit];
  [program->encoder release]; [program->table release]; [program->residency release]; [program->violation release];
}

