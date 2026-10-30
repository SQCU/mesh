#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <IOKit/IOKitLib.h>
#include "nccl-mesh-metal.h"
#include "nccl-mesh-kernels.h"
#include "mesh_metal.h"
#include "metal-capabilities.h"
#include "nccl.h"
#include "metal-recording.h"
#include "mesh-landing.h"
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/* a typed kernel's kind by its operation and its ncclDataType_t (int8 uint8 int32 uint32 int64 uint64 float16 float32
   float64 bfloat16 e4m3 e5m2, then the partials merge's rows, METAL_MERGE), KINDS where none is compiled: float64's,
   truncdiv's but the integers', premul's of the merge */
enum { OP_LAND, OP_PREMUL, OP_TRUNCDIV, OPS };
#define INTEGERS(op) KIND_##op##_i8, KIND_##op##_u8, KIND_##op##_i32, KIND_##op##_u32, KIND_##op##_i64, KIND_##op##_u64
#define FLOATS(op) KIND_##op##_f16, KIND_##op##_f32, KINDS, KIND_##op##_bf16, KIND_##op##_e4m3, KIND_##op##_e5m2
static const uint8_t TYPED[OPS][METAL_MERGE + 1] = {
  {INTEGERS(land), FLOATS(land), KIND_land_merge}, {INTEGERS(premul), FLOATS(premul), KINDS},
  {INTEGERS(truncdiv), KINDS, KINDS, KINDS, KINDS, KINDS, KINDS, KINDS}};
/* a copy's kind by its width (0 words of 4 bytes, 1 bytes) and its loads (METAL_PLAIN, METAL_LAND) */
static const uint8_t COPY[2][METAL_LAND + 1] = {{KIND_copy_plain_4, KIND_copy_land_4}, {KIND_copy_plain_1, KIND_copy_land_1}};

ARGS(args, uint64_t, uint32_t);
/* Kernel lookup is shared; compiled argument pages, pieces of the region (nccl-mesh.c metal_piece_hold), belong to
   their function's residency set, are never recycled while that function can execute, and are given back once that
   set ends. */
#define RECORD MESH_METAL_ARGUMENT_BYTES
/* a device's kernels, and its GPU's cores (IORegistry gpu-core-count, read once with them): the tracks pool's most
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
    long long cores = 0;
    io_service_t gpu = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"));
    CFTypeRef counted = gpu ? IORegistryEntryCreateCFProperty(gpu, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0) : NULL;
    if (counted && CFGetTypeID(counted) == CFNumberGetTypeID()) CFNumberGetValue(counted, kCFNumberLongLongType, &cores);
    if (counted) CFRelease(counted);
    if (gpu) IOObjectRelease(gpu);
    if (cores <= 0 || cores > UINT32_MAX) { *message = "IORegistry's AGXAccelerator states no gpu-core-count"; return NULL; }
    struct kernels *made = calloc(1, sizeof *made);
    if (!made) { *message = "out of memory"; return NULL; }
    made->cores = (uint32_t)cores;
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
/* nccl-mesh-metal.h metal_queue, metal_submission_make: the library's own Metal 4 work */
struct metal_submission {
  id<MTL4CommandQueue> queue;
  id<MTL4CommandAllocator> allocator;
  id<MTL4CommandBuffer> buffer;
  id<MTL4ComputeCommandEncoder> encoder;
  id<MTL4ArgumentTable> table;
  id<MTLResidencySet> residency;
  _Atomic uint32_t references;
  _Atomic int ended;
  int committed;
  double seconds;
};
void *metal_queue(void) { return [(id<MTLDevice>)metal_device() newMTL4CommandQueue]; }
struct metal_submission *metal_submission_retain(struct metal_submission *s) {
  if (s) atomic_fetch_add_explicit(&s->references, 1, memory_order_relaxed);
  return s;
}
void metal_submission_release(struct metal_submission *s) {
  if (!s || atomic_fetch_sub_explicit(&s->references, 1, memory_order_acq_rel) != 1) return;
  if (!s->committed && s->encoder) { [s->encoder endEncoding]; [s->buffer endCommandBuffer]; }
  [s->encoder release]; [s->table release]; [s->residency release]; [s->buffer release]; [s->allocator release]; [s->queue release];
  free(s);
}
int metal_submission_ended(const struct metal_submission *s) {
  return atomic_load_explicit(&((struct metal_submission *)s)->ended, memory_order_acquire);
}
struct metal_submission *metal_submission_make(void *queue, void **encoder, void **table, void **residency, const char **message) {
  @autoreleasepool {
    id<MTLDevice> device = (id<MTLDevice>)metal_device();
    struct metal_submission *s = calloc(1, sizeof *s);
    if (!s) { *message = "out of memory"; return NULL; }
    atomic_init(&s->references, 1);
    s->queue = [(id<MTL4CommandQueue>)queue retain];
    s->allocator = [device newCommandAllocator];
    s->buffer = [device newCommandBuffer];
    MTL4ArgumentTableDescriptor *bindings = [[MTL4ArgumentTableDescriptor new] autorelease];
    bindings.maxBufferBindCount = 6;   /* d, s, the record, a, v, progress: the kernels' six (record4) */
    NSError *error = nil;
    s->table = [device newArgumentTableWithDescriptor:bindings error:&error];
    if (s->table) s->residency = [device newResidencySetWithDescriptor:[[MTLResidencySetDescriptor new] autorelease] error:&error];
    if (s->queue && s->allocator && s->buffer && s->residency) {
      [s->buffer beginCommandBufferWithAllocator:s->allocator];
      s->encoder = [[s->buffer computeCommandEncoder] retain];
    }
    if (!s->encoder) {
      *message = say("the library's Metal 4 submission (its queue, allocator, command buffer, argument table, residency set and encoder): %s",
                     error ? error.localizedDescription.UTF8String : "one was not made");
      metal_submission_release(s);
      return NULL;
    }
    *encoder = s->encoder; *table = s->table; *residency = s->residency;
    return s;
  }
}
/* the submission committed on its queue, `event` (nil: none) signalled to `value` after it, its end and GPU seconds
   taken from its commit feedback */
static void submission_commit(struct metal_submission *s, id<MTLEvent> event, uint64_t value) {
  @autoreleasepool {
    [s->encoder endEncoding];
    [s->buffer useResidencySet:s->residency];
    [s->buffer endCommandBuffer];
    s->committed = 1;
    MTL4CommitOptions *options = [[MTL4CommitOptions new] autorelease];
    metal_submission_retain(s);
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
      s->seconds = feedback.GPUEndTime - feedback.GPUStartTime;
      atomic_store_explicit(&s->ended, feedback.error ? -1 : 1, memory_order_release);
      metal_submission_release(s);
    }];
    const id<MTL4CommandBuffer> buffers[1] = {s->buffer};
    [s->queue commit:buffers count:1 options:options];
    if (event) [s->queue signalEvent:event value:value];
  }
}

/* a stream's polls as a program's bound: 0, or more than NCCL_MESH_POLLS, is NCCL_MESH_POLLS
   (design/transport.md) */
static uint32_t bound_of(uint32_t polls) { return polls && polls < NCCL_MESH_POLLS ? polls : NCCL_MESH_POLLS; }

struct metal_program *metal_begin(struct metal_program *program, void *encoder, void *table, void *residency, void *violation, uint32_t polls,
                                  struct metal_submission *own, const char **message) {
  if (!kernels_of(((id<MTLResidencySet>)residency).device, message)) return NULL;
  *program = (struct metal_program){0};
  program->own = own;
  program->encoder = [(id<MTL4ComputeCommandEncoder>)encoder retain];
  program->table = [(id<MTL4ArgumentTable>)table retain];
  program->residency = [(id<MTLResidencySet>)residency retain];
  program->violation = [(id<MTLBuffer>)violation retain];
  program->polls = bound_of(polls);
  metal_resident(program, violation);
  [program->encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
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

/* the Metal 4 form's dispatch: its record (record4), every buffer bound by address through the caller's table, then a
   barrier before the next */
static struct metal_result dispatch4(struct metal_program *program, id<MTLComputePipelineState> state, id<MTLBuffer> d, id<MTLBuffer> s,
                                     id<MTLBuffer> a, id<MTLBuffer> words, struct args p, uint64_t units, int group) {
  uint64_t bound[6];
  const struct metal_result recorded = record4(program, d, s, a, words, p, bound);
  if (recorded.status) return recorded;
  for (NSUInteger k = 0; k < 6; k++) [program->table setAddress:bound[k] atIndex:k];
  id<MTL4ComputeCommandEncoder> encoder = program->encoder;
  [encoder setComputePipelineState:state];
  [encoder setArgumentTable:program->table];
  const NSUInteger most = group ? MESH_TRANSPORT_GROUP_THREADS : MESH_TRANSPORT_ELEMENT_THREADS, width = state.maxTotalThreadsPerThreadgroup < most ? state.maxTotalThreadsPerThreadgroup : most;
  if (group) [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  else [encoder dispatchThreads:MTLSizeMake(units < MESH_TRANSPORT_GRID_THREADS ? (NSUInteger)units : MESH_TRANSPORT_GRID_THREADS, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
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
  return dispatch(program, COPY[!words][mode], dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = words ? bytes / 4 : bytes},
                  words ? bytes / 4 : bytes, 0);
}

/* operation `op`'s kernel of `type` (TYPED), KINDS where none is compiled */
static uint32_t typed(int op, int type) { return (unsigned)type <= METAL_MERGE ? TYPED[op][type] : KINDS; }

struct metal_result metal_premultiply(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                      uint64_t scalar) {
  return dispatch(program, typed(OP_PREMUL, type), dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = n, .scalar = scalar},
                  n, 0);
}

struct metal_result metal_truncdiv(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                   uint64_t divisor) {
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

static void record_uses(struct metal_program *, void *const *, size_t);
void metal_uses(struct metal_program *program, void *const *buffers, size_t n) {
  if (recording_on(program)) { record_uses(program, buffers, n); return; }
  for (size_t k = 0; k < n; k++) metal_resident(program, buffers[k]);
}

struct metal_result metal_publish(struct metal_program *program, const struct metal_release *r) {
  return dispatch(program, KIND_publish, r->words, NULL, NULL, NULL, release_args(r, 0), 1, 1);
}

struct metal_result metal_open_gate(struct metal_program *program, void *words, size_t word, uint64_t value) {
  return dispatch(program, KIND_publish, words, NULL, NULL, NULL, (struct args){.scalar = value, .aux = word}, 1, 1);
}

/* design/transport.md section 5: the seconds one GPU wait's load takes on `device`, read once in the process: the spin
   kernel's bounded wait on a count that never reaches its target, 2^16 loads, timed by its command buffer, the second of
   two (the GPU's clock risen), each on fresh failure words; 0 where it could not be read */
double metal_poll_seconds(void *device) {
  static double period;
  static dispatch_once_t once;
  dispatch_once(&once, ^{ @autoreleasepool {
    const char *message = NULL;
    struct kernels *k = kernels_of((id<MTLDevice>)device, &message);
    id<MTL4CommandQueue> queue = k ? [(id<MTLDevice>)device newMTL4CommandQueue] : nil;
    /* the counts at 0, the failure words at 64, the argument record at 128, the progress word at 256 */
    id<MTLBuffer> words = [(id<MTLDevice>)device newBufferWithLength:384 options:MTLResourceStorageModeShared];
    const struct args p = {.scalar = 1, .polls = 1u << 16};
    _Static_assert(sizeof p <= 128, "the argument record fits between the failure words and the progress word");
    for (int pass = 0; queue && words && pass < 2; pass++) {
      memset(words.contents, 0, 384);
      memcpy((char *)words.contents + 128, &p, sizeof p);
      void *encoder = NULL, *table = NULL, *residency = NULL;
      struct metal_submission *made = metal_submission_make(queue, &encoder, &table, &residency, &message);
      if (!made) break;
      [(id<MTLResidencySet>)residency addAllocation:words];
      [(id<MTLResidencySet>)residency commit];
      const uint64_t at = words.gpuAddress, bound[6] = {at, at, at + 128, at, at + 64, at + 256};
      for (NSUInteger i = 0; i < 6; i++) [(id<MTL4ArgumentTable>)table setAddress:bound[i] atIndex:i];
      [(id<MTL4ComputeCommandEncoder>)encoder setComputePipelineState:k->pipelines[KIND_spin]];
      [(id<MTL4ComputeCommandEncoder>)encoder setArgumentTable:(id<MTL4ArgumentTable>)table];
      [(id<MTL4ComputeCommandEncoder>)encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      submission_commit(made, nil, 0);
      const uint64_t until = mesh_clock_ns() + MESH_PLATFORM_NS;
      while (!metal_submission_ended(made) && mesh_clock_ns() < until) usleep(100);
      period = metal_submission_ended(made) == 1 ? made->seconds / p.polls : 0;
      metal_submission_release(made);
    }
    [queue release]; [words release];
  }});
  return period;
}

struct metal_result metal_spin(struct metal_program *program, const struct metal_wait *w) {
  metal_uses(program, &w->data, w->data != NULL);
  return dispatch(program, KIND_spin, w->words, w->words, NULL, w->violation, wait_args(w, 0, 1, 0), 1, 1);
}

struct metal_result metal_land(struct metal_program *program, int type, int op, void *dst, size_t dst_offset,
                               void *with, size_t with_offset,
                               size_t element_bytes, mesh_pages packets, const struct mesh_landing_run *runs, size_t count, const struct metal_wait *w) {
  if (count) metal_uses(program, &with, 1);
  const uint32_t kind = type == METAL_MERGE && (with != dst || with_offset != dst_offset) ? KIND_land_merge_from : typed(OP_LAND, type);
  for (size_t k = 0; k < count; k++) {
    const struct mesh_landing_run run = runs[k];
    struct args p = wait_args(w, dst_offset + run.first * element_bytes, run.elements, op);
    p.src += run.first * element_bytes;
    p.input = ((id<MTLBuffer>)with).gpuAddress + with_offset + run.first * element_bytes;
    p.scalar = pages_to_word(pages_sub(w->target, packets)) + run.packet;
    p.phase = run.phase; p.step = run.step; p.mask = run.shift;
    recording_join(program, k > 0);
    const struct metal_result result = dispatch(program, kind, dst, w->data, w->words, w->violation, p,
      type == METAL_MERGE ? run.elements : (run.elements + 3) / 4, 0);
    if (result.status) { recording_join(program, 0); return result; }
  }
  recording_join(program, 0);
  return METAL_DONE;
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
struct host_op { void *d, *s, *a, *v; struct args p; uint64_t units; uint32_t kind; int group, joined; };
struct host_item { uint64_t position, need; void *v; uint32_t first_op, ops, first_dep, deps; };
struct host_dep { uint64_t count; uint32_t other; };
struct host_track { uint32_t *item[2]; size_t count[2], room[2]; };
struct recording { struct host_op *op; size_t ops, op_room; struct host_item *item; size_t items, item_room; struct host_dep *dep; size_t deps, dep_room;
                   struct host_track *track; uint32_t tracks; uint32_t current; int on, join; NSMutableSet *named; };

static int recording_on(const struct metal_program *program) { return program->recording && program->recording->on; }
static void recording_join(struct metal_program *program, int join) { if (program->recording) program->recording->join = join; }

static void record_uses(struct metal_program *program, void *const *buffers, size_t n) {
  for (size_t k = 0; k < n; k++) if (buffers[k]) [program->recording->named addObject:(id)buffers[k]];
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
  const struct host_op made = {d, s ? s : d, a ? a : d, v ? v : d, p, units, kind, group, r->join && it->ops};
  r->op[r->ops++] = made;
  const id named[4] = {(id)made.d, (id)made.s, (id)made.a, (id)made.v};
  for (int k = 0; k < 4; k++) [r->named addObject:named[k]];
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
   task, each task's chunks claimed and finished, the count of tasks finished and each worker's end */
struct track_head { uint64_t cursors_off, tasks_off, items_off, ops_off, deps_off, count_off, state_off, claim_off, done_off, progress_off,
                    ended_off; uint32_t cursors, polls; };
struct track_cursor { uint32_t first, end, track, kind; };
struct track_task { uint32_t first_op, ops, chunks, item, cursor, flags, pad[2]; };
struct track_item { uint64_t position, need, v; uint32_t first_dep, deps; };
struct track_op { uint64_t d, s, a, v; struct args p; uint64_t word, block; uint32_t kind, chunks; };
struct track_dep { uint64_t count; uint32_t other, pad; };
_Static_assert(sizeof(struct track_head) == 96 && sizeof(struct track_cursor) == 16 && sizeof(struct track_task) == 32 &&
               sizeof(struct track_item) == 32 && sizeof(struct track_op) == sizeof(struct args) + 7 * sizeof(uint64_t) &&
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
    const uint64_t width = state.maxTotalThreadsPerThreadgroup < MESH_TRANSPORT_ELEMENT_THREADS ? state.maxTotalThreadsPerThreadgroup
                                                                                               : MESH_TRANSPORT_ELEMENT_THREADS;
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
    const uint64_t at_ended = up(at_progress + sizeof(uint32_t)), bytes = up(at_ended + workers * sizeof(uint64_t));
    /* the records, a carved piece of the region, done with once every worker has ended (each stores its end word) */
    struct metal_bytes records;
    if (metal_carve(bytes, &records)) { recording_free(r); return metal_failed(ENOMEM); }
    unsigned char *base = records.contents;
    memset(base + at_state, 0, bytes - at_state);
    metal_done((const _Atomic uint64_t *)(base + at_ended), (uint32_t)workers, 1);
    const uint64_t address = ((id<MTLBuffer>)records.buffer).gpuAddress + records.offset;
    *(struct track_head *)base = (struct track_head){at_cursors, at_tasks, at_items, at_ops, at_deps, at_count, at_state, at_claim, at_done,
                                                     at_progress, at_ended, (uint32_t)C, program->polls};
    struct track_cursor *cursor = (struct track_cursor *)(base + at_cursors);
    struct track_task *task = (struct track_task *)(base + at_tasks);
    struct track_item *item = (struct track_item *)(base + at_items);
    uint32_t *current = (uint32_t *)(base + at_state), placed = 0, laid = 0;
    for (uint32_t t = 0; t < T; t++)
      for (uint32_t c = 0; c < 2; c++) {
        const uint32_t cu = 2 * t + c;
        cursor[cu] = (struct track_cursor){laid, laid, t, c};
        current[cu] = laid;
        for (size_t x = 0; t < r->tracks && x < r->track[t].count[c]; x++, placed++) {
          const struct host_item *it = r->item + r->track[t].item[c][x];
          item[placed] = (struct track_item){it->position, it->need, it->v ? ((id<MTLBuffer>)it->v).gpuAddress : 0, it->first_dep, it->deps};
          if (!it->ops) task[laid++] = (struct track_task){.chunks = 1, .item = placed, .cursor = cu, .flags = TASK_FIRST | TASK_LAST};
          for (uint32_t o = 0; o < it->ops;) {
            struct track_task *made = task + laid++;
            *made = (struct track_task){.first_op = it->first_op + o, .item = placed, .cursor = cu, .flags = o ? 0 : TASK_FIRST};
            do {
              const struct host_op *op = r->op + it->first_op + o;
              made->ops++; made->chunks += op_chunks(op, width, widest);
              if (LANES[op->kind]) made->flags |= TASK_PACED;
            } while (++o < it->ops && r->op[it->first_op + o].joined);
            if (o == it->ops) made->flags |= TASK_LAST;
          }
        }
        cursor[cu].end = laid;
      }
    struct track_op *op = (struct track_op *)(base + at_ops);
    for (size_t x = 0; x < r->ops; x++) {
      const struct host_op *o = r->op + x;
      void *const word = WORD[o->kind] == WORD_A ? o->a : WORD[o->kind] == WORD_D && o->p.scalar ? o->d : NULL;
      op[x] = (struct track_op){((id<MTLBuffer>)o->d).gpuAddress, ((id<MTLBuffer>)o->s).gpuAddress, ((id<MTLBuffer>)o->a).gpuAddress,
                                ((id<MTLBuffer>)o->v).gpuAddress, o->p, word ? ((id<MTLBuffer>)word).gpuAddress + o->p.aux : 0,
                                LANES[o->kind] * width, o->kind, op_chunks(o, width, widest)};
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
    [encoder dispatchThreadgroups:MTLSizeMake(workers, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
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

struct metal_result metal_present(struct metal_program *program,void *lists,size_t list_offset,
    void *records,size_t record_offset,size_t count,uint64_t first,void *words,size_t taken,size_t presented) {
  if(count) {
    // The bridge consumes this list in order. The kernel stores the group's end only once the list's count has
    // reached its first entry (its predecessor's end), so a later group's count never passes an earlier group's
    // unwritten entries, in whatever command buffer each runs; here the count is read as well as written.
    const size_t slot=first%MESH_STREAM_ENTRIES;
    const size_t head=count<MESH_STREAM_ENTRIES-slot?count:MESH_STREAM_ENTRIES-slot;
    const struct MetalReach reaches[]={
      {(id)lists,list_offset+slot*sizeof(struct mesh_entry),head*sizeof(struct mesh_entry),MetalReachWritten,-1},
      {(id)lists,list_offset,(count-head)*sizeof(struct mesh_entry),MetalReachWritten,-1},
      {(id)words,taken,sizeof(uint64_t),MetalReachRead,-1},
      {(id)words,presented,sizeof(uint64_t),MetalReachWritten,-1}};
    MetalEncoderReaches(program->encoder,reaches,sizeof reaches/sizeof *reaches);
  }
  return dispatch(program,KIND_present,lists,records,words,NULL,
    (struct args){.dst=list_offset,.src=record_offset,.n=count,.scalar=first,.aux=taken,
                  .at=presented,.count=MESH_STREAM_ENTRIES},count,1);
}

void metal_end(struct metal_program *program, void *event, uint64_t value) {
  recording_free(program->recording);
  program->recording = NULL;
  if (program->added) [program->residency commit];
  if (program->own && event) submission_commit(program->own, (id<MTLEvent>)event, value);
  [program->encoder release]; [program->table release]; [program->residency release]; [program->violation release];
}

