#import <Metal/Metal.h>
#import <objc/runtime.h>
#include "nccl-mesh-metal.h"
#include "nccl-mesh-kernels.h"
#include "mesh_metal.h"
#include "metal-capabilities.h"
#include "nccl.h"
#include "metal-recording.h"
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/* a typed kernel's kind by its operation and its ncclDataType_t (int8 uint8 int32 uint32 int64 uint64 float16 float32
   float64 bfloat16 e4m3 e5m2, then the partials merge's rows, METAL_MERGE), KINDS where none is compiled: float64's,
   truncdiv's but the integers', combine3's and premul's of the merge */
enum { OP_LAND, OP_COMBINE3, OP_PREMUL, OP_TRUNCDIV, OPS };
#define INTEGERS(op) KIND_##op##_i8, KIND_##op##_u8, KIND_##op##_i32, KIND_##op##_u32, KIND_##op##_i64, KIND_##op##_u64
#define FLOATS(op) KIND_##op##_f16, KIND_##op##_f32, KINDS, KIND_##op##_bf16, KIND_##op##_e4m3, KIND_##op##_e5m2
static const uint8_t TYPED[OPS][METAL_MERGE + 1] = {
  {INTEGERS(land), FLOATS(land), KIND_land_merge}, {INTEGERS(combine3), FLOATS(combine3), KINDS}, {INTEGERS(premul), FLOATS(premul), KINDS},
  {INTEGERS(truncdiv), KINDS, KINDS, KINDS, KINDS, KINDS, KINDS, KINDS}};
/* a copy's kind by its width (0 words of 4 bytes, 1 bytes) and its loads (METAL_PLAIN, METAL_LAND) */
static const uint8_t COPY[2][METAL_LAND + 1] = {{KIND_copy_plain_4, KIND_copy_land_4}, {KIND_copy_plain_1, KIND_copy_land_1}};

ARGS(args, uint64_t, uint32_t);
/* Kernel lookup is shared; compiled argument pages belong to their function's
   residency set and are never recycled while that function can execute. */
#define RECORD MESH_METAL_ARGUMENT_BYTES
struct kernels { id<MTLDevice> device; id<MTLComputePipelineState> pipelines[PIPELINES]; };
@interface MeshArgumentPages : NSObject {
@public
  id<MTLBuffer> page;
  NSUInteger used;
}
@end
@implementation MeshArgumentPages
- (void)dealloc { [page release]; [super dealloc]; }
@end
static char argumentPagesKey;
_Static_assert(sizeof(struct args) <= RECORD, "an argument record");

static id<MTLDevice> device_;
static NSMutableDictionary *kernels_;
static NSMutableArray *retiring_;
/* the text of this thread's last failure here that is composed (say): a result's message (struct metal_result), its
   caller's to carry before the thread's next step */
static _Thread_local char said_[512];

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
    if (!kernels_) { kernels_ = [NSMutableDictionary new]; retiring_ = [NSMutableArray new]; }
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
    program->kernels = kernels_of(program->encoder4 ? program->residency.device : (id<MTLDevice>)metal_device(), message);
  return program->kernels;
}

void *metal_wrap(void *address, size_t bytes) {
  return [(id<MTLDevice>)metal_device() newBufferWithBytesNoCopy:address length:bytes options:MTLResourceStorageModeShared deallocator:nil];
}

void *metal_scratch(size_t bytes) {
  return [(id<MTLDevice>)metal_device() newBufferWithLength:bytes ? bytes : 16 options:MTLResourceStorageModePrivate];
}

void *metal_shared(size_t bytes) {
  id<MTLBuffer> made = [(id<MTLDevice>)metal_device() newBufferWithLength:bytes ? bytes : 16 options:MTLResourceStorageModeShared];
  if (made) memset(made.contents, 0, made.length);
  return made;
}

void *metal_event(void) { return [(id<MTLDevice>)metal_device() newSharedEvent]; }
uint64_t metal_signaled(void *event) { return ((id<MTLSharedEvent>)event).signaledValue; }
int metal_wait(void *event, uint64_t value, uint64_t ns) { return [(id<MTLSharedEvent>)event waitUntilSignaledValue:value timeoutMS:ns / 1000000]; }
void metal_release(void *object) { [(id)object release]; }
void *metal_retain(void *object) { return [(id)object retain]; }
void *metal_command(const struct metal_program *program) { return [program->buffer retain]; }
int metal_ended(void *command) {
  const MTLCommandBufferStatus status = ((id<MTLCommandBuffer>)command).status;
  return status == MTLCommandBufferStatusError ? -1 : status == MTLCommandBufferStatusCompleted ? 1 : status < MTLCommandBufferStatusCommitted ? 2 : 0;
}

/* a stream's polls as a program's bound: 0, or more than NCCL_MESH_POLLS, is NCCL_MESH_POLLS
   (design/transport.md) */
static uint32_t bound_of(uint32_t polls) { return polls && polls < NCCL_MESH_POLLS ? polls : NCCL_MESH_POLLS; }

struct metal_program *metal_begin(struct metal_program *program, void *command_buffer, void *encoder, void *residency, void *violation, uint32_t polls, const char **message) {
  if (residency && !kernels_of(((id<MTLResidencySet>)residency).device, message)) return NULL;
  *program = (struct metal_program){0};
  program->buffer = [(id<MTLCommandBuffer>)command_buffer retain];
  if (encoder) { program->encoder = [(id<MTLComputeCommandEncoder>)encoder retain]; program->borrowed = 1; }
  if (!encoder) program->kept = [NSMutableArray new];
  program->violation = [(id<MTLBuffer>)violation retain];
  program->polls = bound_of(polls);
  /* Deferred records and the payloads they name belong to this residency set. */
  if (residency) {
    program->residency = [(id<MTLResidencySet>)residency retain];
    metal_resident(program, violation);
  }
  return program;
}

struct metal_program *metal_begin4(struct metal_program *program, void *encoder, void *table, void *residency, void *violation, uint32_t polls, const char **message) {
  if (!kernels_of(((id<MTLResidencySet>)residency).device, message)) return NULL;
  *program = (struct metal_program){0};
  program->encoder4 = [(id<MTL4ComputeCommandEncoder>)encoder retain];
  program->table = [(id<MTL4ArgumentTable>)table retain];
  program->residency = [(id<MTLResidencySet>)residency retain];
  program->violation = [(id<MTLBuffer>)violation retain];
  program->polls = bound_of(polls);
  program->borrowed = 1;
  metal_resident(program, violation);
  [program->encoder4 barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
  return program;
}

int metal_metal4(const struct metal_program *program) { return program->encoder4 != nil; }

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

static void close_encoder(struct metal_program *program) {
  if (!program->encoder) return;
  if (!program->borrowed) [program->encoder endEncoding];
  [program->encoder release];
  program->encoder = nil;
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
  if(!pages->page || pages->used+RECORD>pages->page.length) {
    [pages->page release];
    pages->page=[program->residency.device newBufferWithLength:65536 options:MTLResourceStorageModeShared];
    pages->used=0;
    if(!pages->page)return metal_failed(ENOMEM);
    metal_resident(program,pages->page);
  }
  id<MTLBuffer> v = words ? words : program->violation ? program->violation : d;
  p.polls = v != d ? program->polls : 0;
  p.sequence=0;
  const uint64_t argument=pages->page.gpuAddress+pages->used;
  memcpy((char *)pages->page.contents+pages->used,&p,sizeof p);pages->used+=RECORD;
  for (id<MTLBuffer> b in @[d, s ? s : d, a ? a : d, v]) metal_resident(program, b);
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
  id<MTL4ComputeCommandEncoder> encoder = program->encoder4;
  [encoder setComputePipelineState:state];
  [encoder setArgumentTable:program->table];
  const NSUInteger most = group ? 1024 : 256, width = state.maxTotalThreadsPerThreadgroup < most ? state.maxTotalThreadsPerThreadgroup : most;
  if (group) [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  else [encoder dispatchThreads:MTLSizeMake(units < (1u << 20) ? (NSUInteger)units : (1u << 20), 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
  return METAL_DONE;
}

/* kernel `kind` (KINDS_OF) on buffers d, s, a and v (0, 1, 3 and 4; an absent one d, and v, where none is given, the
   program's failure words: metal_gate) and the arguments p (2, its polls the program's where v is such words), the
   progress word (5; the classic form binds d there, which no kernel then writes): a GROUP kernel (`group`: publish,
   spin) as one threadgroup of up to 1024 threads, any other over `units` threads */
static struct metal_result record_step(struct metal_program *program, uint32_t kind, void *d, void *s, void *a, void *v, struct args p);
static int recording_on(const struct metal_program *program);
static struct metal_result dispatch(struct metal_program *program, uint32_t kind, void *d, void *s, void *a, void *v, struct args p,
                                    uint64_t units, int group) {
  if (kind >= KINDS) return (struct metal_result){EINVAL, "no kernel of this operation and data type (nccl-mesh-metal.m TYPED)"};
  if (!units) return METAL_DONE;
  if (recording_on(program)) return record_step(program, kind, d, s, a, v, p);
  const char *message = NULL;
  struct kernels *k = program_kernels(program, &message);
  if (!k) return (struct metal_result){EINVAL, message};
  id<MTLComputePipelineState> state = k->pipelines[kind];
  @autoreleasepool {
    if (program->encoder4) return dispatch4(program, state, (id<MTLBuffer>)d, (id<MTLBuffer>)s, (id<MTLBuffer>)a, (id<MTLBuffer>)v, p, units, group);
    if (!program->encoder) program->encoder = [[program->buffer computeCommandEncoder] retain];
    id<MTLComputeCommandEncoder> encoder = program->encoder;
    if (!v) v = (void *)program->violation;
    p.polls = v && v != d ? program->polls : 0;
    [encoder setComputePipelineState:state];
    [encoder setBuffer:(id<MTLBuffer>)d offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)(s ? s : d) offset:0 atIndex:1];
    [encoder setBytes:&p length:sizeof p atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)(a ? a : d) offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)(v ? v : d) offset:0 atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)d offset:0 atIndex:5];
    const NSUInteger most = group ? 1024 : 256, width = state.maxTotalThreadsPerThreadgroup < most ? state.maxTotalThreadsPerThreadgroup : most;
    if (group) [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    else [encoder dispatchThreads:MTLSizeMake(units < (1u << 20) ? (NSUInteger)units : (1u << 20), 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  }
  return METAL_DONE;
}

struct metal_result metal_copy(struct metal_program *program, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes) {
  const int words = !((dst_offset | src_offset | bytes) & 3);
  if (mode < METAL_PLAIN || mode > METAL_LAND) return metal_failed(EINVAL);
  return dispatch(program, COPY[!words][mode], dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = words ? bytes / 4 : bytes},
                  words ? bytes / 4 : bytes, 0);
}

/* operation `op`'s kernel of `type` (TYPED), KINDS where none is compiled */
static uint32_t typed(int op, int type) { return (unsigned)type <= METAL_MERGE ? TYPED[op][type] : KINDS; }

struct metal_result metal_combine3(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset,
                                   void *with, size_t with_offset, size_t n) {
  return dispatch(program, typed(OP_COMBINE3, type), dst, src, with, NULL,
                  (struct args){.dst = dst_offset, .src = src_offset, .n = n, .op = (uint32_t)op, .aux = with_offset}, n, 0);
}

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

void metal_uses(struct metal_program *program, void *const *buffers, size_t n, int write) {
  if (!n || recording_on(program) || program->encoder4) return;
  @autoreleasepool {
    if (!program->encoder) program->encoder = [[program->buffer computeCommandEncoder] retain];
    const MTLResourceUsage usage = write ? (MTLResourceUsageRead | MTLResourceUsageWrite) : MTLResourceUsageRead;
    for (size_t k = 0; k < n; k++)
      if (buffers[k]) [program->encoder useResource:(id<MTLBuffer>)buffers[k] usage:usage];
  }
}

struct metal_result metal_publish(struct metal_program *program, const struct metal_release *r) {
  return dispatch(program, KIND_publish, r->words, NULL, NULL, NULL, release_args(r, 0), 1, 1);
}

struct metal_result metal_open_gate(struct metal_program *program, void *words, size_t word, uint64_t value) {
  return dispatch(program, KIND_publish, words, NULL, NULL, NULL, (struct args){.scalar = value, .aux = word}, 1, 1);
}

struct metal_result metal_spin(struct metal_program *program, const struct metal_wait *w) {
  metal_uses(program, &w->data, w->data != NULL, 1);
  return dispatch(program, KIND_spin, w->words, w->words, NULL, w->violation, wait_args(w, 0, 1, 0), 1, 1);
}

struct metal_result metal_land(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, size_t n, mesh_pages pages,
                               uint64_t page, const struct metal_wait *w) {
  struct args p = wait_args(w, dst_offset, n, op);
  if (type == METAL_MERGE) return dispatch(program, typed(OP_LAND, type), dst, w->data, w->words, w->violation, p, n, 0);
  p.scalar = pages_to_word(pages_sub(w->target, pages)) + 1;
  p.count = (uint32_t)pages_to_word(pages);
  p.mask = (uint32_t)page;
  return dispatch(program, typed(OP_LAND, type), dst, w->data, w->words, w->violation, p, (n + 3) / 4, 0);
}

/* Deferred steps retain their compiled records through the function's residency set. */
struct metal_result metal_send_deferred(struct metal_program *program, void *operand, size_t offset, const struct metal_release *r, uint64_t bound[6]) {
  if (!program->residency) return (struct metal_result){EINVAL, "a step run by its caller's kernel is the Metal 4 form's, or a classic stream's with its residency set"};
  return record4(program, (id<MTLBuffer>)operand, nil, (id<MTLBuffer>)r->words, nil, release_args(r, offset), bound);
}

struct metal_result metal_land_deferred(struct metal_program *program, void *landing, size_t offset, const struct metal_wait *w, uint64_t bound[6]) {
  if (!program->residency) return (struct metal_result){EINVAL, "a step run by its caller's kernel is the Metal 4 form's, or a classic stream's with its residency set"};
  return record4(program, (id<MTLBuffer>)landing, (id<MTLBuffer>)landing, (id<MTLBuffer>)w->words, (id<MTLBuffer>)w->violation,
                 wait_args(w, offset, 0, 0), bound);
}

/* the tracks being recorded (nccl-mesh-metal.h metal_record): each step's kernel and bindings as its dispatch would bind
   them (`v` its failure words, else the program's; its polls the program's where it has such words), each item's
   position, `need`, steps, dependencies and the landing a consumption awaits (its first wait on `landed`: spin's word
   in d, a landing's in a), each track's items by cursor; the buffers named, made resident at metal_tracks */
struct host_op { void *d, *s, *a, *v; struct args p; uint32_t kind; };
struct host_item { uint64_t position, need, target; void *word_buffer, *v; size_t word; uint32_t first_op, ops, first_dep, deps, wait_op, cursor; };
struct host_dep { uint64_t count; uint32_t other; };
struct host_track { uint32_t *item[2]; size_t count[2], room[2]; };
struct recording { struct host_op *op; size_t ops, op_room; struct host_item *item; size_t items, item_room; struct host_dep *dep; size_t deps, dep_room;
                   struct host_track *track; uint32_t tracks; uint32_t current; int on; NSMutableSet *named; };

static int recording_on(const struct metal_program *program) { return program->recording && program->recording->on; }

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

/* the pages of a landing of `pages` its item awaits before its track runs it, the rest awaited in the kernel page by
   page (nccl-mesh-kernels.h LANDS): the least r with r * r at least `pages`, so a landing of a page or two is awaited
   whole and a larger one is summed from its first r pages on */
static uint64_t first_pages(uint64_t pages) {
  uint64_t r = 1;
  while (r * r < pages) r++;
  return r < pages ? r : pages;
}

/* a step of kind `kind` recorded onto the item being recorded: a consumption's first step that awaits a word (WORD) its
   item's wait */
static struct metal_result record_step(struct metal_program *program, uint32_t kind, void *d, void *s, void *a, void *v, struct args p) {
  struct recording *r = program->recording;
  if (!v) v = (void *)program->violation;
  p.polls = v && v != d ? program->polls : 0;
  struct host_op *ops = more(r->op, &r->op_room, r->ops, sizeof *ops);
  if (!ops) return metal_failed(ENOMEM);
  r->op = ops;
  const struct host_op made = {d, s ? s : d, a ? a : d, v ? v : d, p, kind};
  r->op[r->ops++] = made;
  for (id b in @[(id)made.d, (id)made.s, (id)made.a, (id)made.v]) [r->named addObject:b];
  struct host_item *it = r->item + r->current;
  it->ops++;
  if (it->cursor == METAL_CONSUME && it->wait_op == UINT32_MAX && WORD[kind] && p.scalar) {
    it->wait_op = (uint32_t)(r->ops - 1);
    it->word_buffer = WORD[kind] == WORD_D ? made.d : made.a;
    it->word = p.aux;
    it->target = WORD[kind] == WORD_P ? p.scalar + first_pages(p.count) - 1 : p.scalar;
  }
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
  r->item[r->items] = (struct host_item){.position = position, .need = need, .first_op = (uint32_t)r->ops, .first_dep = (uint32_t)r->deps,
                                         .wait_op = UINT32_MAX, .cursor = (uint32_t)cursor};
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

/* the records as the track kernel reads them (TRACKS: track_head, track, track_item, track_op, track_dep) */
struct track_head { uint64_t tracks_off, items_off, ops_off, deps_off, count_off, sequence, progress_off; uint32_t tracks, polls; };
struct track_rec { uint32_t first[2], count[2]; };
struct track_item { uint64_t position, need, target, word, v; uint32_t first_op, ops, first_dep, deps, wait_op, cursor; };
struct track_op { uint64_t d, s, a, v; struct args p; uint32_t kind, pad; };
struct track_dep { uint64_t count; uint32_t other, pad; };
_Static_assert(sizeof(struct track_head) == 64 && sizeof(struct track_rec) == 16 && sizeof(struct track_item) == 64 &&
               sizeof(struct track_op) == 120 && sizeof(struct track_dep) == 16, "the track records");

static uint64_t up(uint64_t x) { return (x + 255) & ~(uint64_t)255; }

struct metal_result metal_tracks(struct metal_program *program, uint32_t tracks, const uint64_t *consumed) {
  struct recording *r = program->recording;
  program->recording = NULL;
  if (!r || !r->tracks) { recording_free(r); return METAL_DONE; }
  @autoreleasepool {
    const char *message = NULL;
    struct kernels *k = program_kernels(program, &message);
    if (!k) { recording_free(r); return (struct metal_result){EINVAL, message}; }
    id<MTLComputePipelineState> state = k->pipelines[KIND_tracks];
    id<MTLDevice> device = k->device;
    const uint64_t T = r->tracks, at_tracks = up(sizeof(struct track_head)), at_items = up(at_tracks + T * sizeof(struct track_rec));
    const uint64_t at_ops = up(at_items + r->items * sizeof(struct track_item)), at_deps = up(at_ops + r->ops * sizeof(struct track_op));
    const uint64_t at_count = up(at_deps + r->deps * sizeof(struct track_dep)), bytes = up(at_count + T * sizeof(uint64_t));
    const uint64_t sequence=0,first=64;
    id<MTLBuffer> records=[device newBufferWithLength:first+bytes options:MTLResourceStorageModeShared];
    if(!records){recording_free(r);return metal_failed(ENOMEM);}
    memset(records.contents,0,first);
    unsigned char *base=(unsigned char *)records.contents+first;
    const uint64_t address=records.gpuAddress;
    *(struct track_head *)base = (struct track_head){at_tracks, at_items, at_ops, at_deps, first + at_count, sequence, 0, (uint32_t)T, program->polls};
    struct track_rec *rec = (struct track_rec *)(base + at_tracks);
    struct track_item *item = (struct track_item *)(base + at_items);
    uint32_t placed = 0;
    for (uint32_t t = 0; t < T; t++)
      for (int c = 0; c < 2; c++) {
        rec[t].first[c] = placed; rec[t].count[c] = (uint32_t)r->track[t].count[c];
        for (size_t x = 0; x < r->track[t].count[c]; x++) {
          const struct host_item *it = r->item + r->track[t].item[c][x];
          item[placed++] = (struct track_item){it->position, it->need, it->target,
                                               it->word_buffer ? ((id<MTLBuffer>)it->word_buffer).gpuAddress + it->word : 0,
                                               it->v ? ((id<MTLBuffer>)it->v).gpuAddress : 0, it->first_op, it->ops, it->first_dep, it->deps, it->wait_op,
                                               it->cursor};
        }
      }
    struct track_op *op = (struct track_op *)(base + at_ops);
    for (size_t x = 0; x < r->ops; x++) {
      const struct host_op *o = r->op + x;
      op[x] = (struct track_op){((id<MTLBuffer>)o->d).gpuAddress, ((id<MTLBuffer>)o->s).gpuAddress, ((id<MTLBuffer>)o->a).gpuAddress,
                                ((id<MTLBuffer>)o->v).gpuAddress, o->p, o->kind, 0};
    }
    struct track_dep *dep = (struct track_dep *)(base + at_deps);
    for (size_t x = 0; x < r->deps; x++) dep[x] = (struct track_dep){r->dep[x].count, r->dep[x].other, 0};
    uint64_t *count = (uint64_t *)(base + at_count);
    for (uint32_t t = 0; t < T; t++) count[t] = t < tracks ? consumed[t] : 0;
    const NSUInteger width = state.maxTotalThreadsPerThreadgroup < 1024 ? state.maxTotalThreadsPerThreadgroup : 1024;
    if (program->encoder4) {
      metal_resident(program, records);
      for (id b in r->named) metal_resident(program, (void *)b);
      [program->table setAddress:address + first atIndex:0];
      [program->table setAddress:address atIndex:1];
      id<MTL4ComputeCommandEncoder> encoder = program->encoder4;
      [encoder setComputePipelineState:state];
      [encoder setArgumentTable:program->table];
      [encoder dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    } else {
      if (!program->encoder) program->encoder = [[program->buffer computeCommandEncoder] retain];
      id<MTLComputeCommandEncoder> encoder = program->encoder;
      for (id b in r->named) [encoder useResource:(id<MTLBuffer>)b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
      [encoder setComputePipelineState:state];
      [encoder setBuffer:records offset:first atIndex:0];
      [encoder setBuffer:records offset:0 atIndex:1];
      [encoder dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    }
    [records release];
  }
  recording_free(r);
  return METAL_DONE;
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

void metal_keep(struct metal_program *program, void *object) { [program->kept addObject:(id)object]; }

struct metal_result metal_present(struct metal_program *program,void *lists,size_t list_offset,
    void *records,size_t record_offset,size_t count,uint64_t first,void *words,size_t taken,size_t presented) {
  if(program->encoder4&&count) {
    // The bridge consumes this list in order. A later prepared function must
    // not replace its publication count before an earlier encoder uses it.
    const size_t slot=first%MESH_STREAM_ENTRIES;
    const size_t head=count<MESH_STREAM_ENTRIES-slot?count:MESH_STREAM_ENTRIES-slot;
    const struct MetalReach reaches[]={
      {(id)lists,list_offset+slot*sizeof(struct mesh_entry),head*sizeof(struct mesh_entry),MetalReachWritten},
      {(id)lists,list_offset,(count-head)*sizeof(struct mesh_entry),MetalReachWritten},
      {(id)words,taken,sizeof(uint64_t),MetalReachRead},
      {(id)words,presented,sizeof(uint64_t),MetalReachWritten}};
    MetalEncoderReaches(program->encoder4,reaches,sizeof reaches/sizeof *reaches);
  }
  return dispatch(program,KIND_present,lists,records,words,NULL,
    (struct args){.dst=list_offset,.src=record_offset,.n=count,.scalar=first,.aux=taken,
                  .at=presented,.count=MESH_STREAM_ENTRIES},count,1);
}

/* the program's end signals `event` to `value`; what it kept is released once the event has passed it
   (metal_collect); one with no event (a failed group's, nccl-mesh.c program_end) signals nothing, and what its own
   encoder's commands kept is held by its command buffer (a completion handler's capture) until that buffer completes
   or is freed; a program in a borrowed encoder keeps nothing; the residency set (the Metal 4 form's, a fused classic
   program's) committed where the program made a buffer resident */
void metal_end(struct metal_program *program, void *event, uint64_t value) {
  recording_free(program->recording);
  program->recording = NULL;
  close_encoder(program);
  if (event) {
    [program->buffer encodeSignalEvent:(id<MTLEvent>)event value:value];
    @autoreleasepool { [retiring_ addObject:@[@(value), program->kept]]; }
  } else if (!program->borrowed && program->buffer) {
    NSMutableArray *kept = program->kept;
    [program->buffer addCompletedHandler:^(id<MTLCommandBuffer> done) { (void)done; (void)kept; }];
  }
  if (program->added) [program->residency commit];
  [program->kept release];
  [program->buffer release];
  [program->encoder4 release]; [program->table release]; [program->residency release]; [program->violation release];
}

void metal_collect(void *event) {
  const uint64_t signaled = metal_signaled(event);
  @autoreleasepool {
    NSIndexSet *done = [retiring_ indexesOfObjectsPassingTest:^BOOL(NSArray *entry, NSUInteger index, BOOL *stop) {
      (void)index; (void)stop; return [entry[0] unsignedLongLongValue] <= signaled; }];
    [retiring_ removeObjectsAtIndexes:done];
  }
}
