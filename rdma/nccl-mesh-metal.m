#import <Metal/Metal.h>
#include "nccl-mesh-metal.h"
#include "nccl-mesh-kernels.h"
#include "nccl.h"
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/* a typed kernel's kind by its operation and its ncclDataType_t (int8 uint8 int32 uint32 int64 uint64 float16 float32
   float64 bfloat16 e4m3 e5m2, then the partials merge's rows, METAL_MERGE), KINDS where none is compiled: float64's,
   truncdiv's but the integers', combine3's and premul's of the merge */
enum { OP_COMBINE, OP_LAND, OP_COMBINE3, OP_PREMUL, OP_TRUNCDIV, OPS };
#define INTEGERS(op) KIND_##op##_i8, KIND_##op##_u8, KIND_##op##_i32, KIND_##op##_u32, KIND_##op##_i64, KIND_##op##_u64
#define FLOATS(op) KIND_##op##_f16, KIND_##op##_f32, KINDS, KIND_##op##_bf16, KIND_##op##_e4m3, KIND_##op##_e5m2
static const uint8_t TYPED[OPS][METAL_MERGE + 1] = {
  {INTEGERS(combine), FLOATS(combine), KIND_combine_merge}, {INTEGERS(land), FLOATS(land), KIND_land_merge},
  {INTEGERS(combine3), FLOATS(combine3), KINDS}, {INTEGERS(premul), FLOATS(premul), KINDS},
  {INTEGERS(truncdiv), KINDS, KINDS, KINDS, KINDS, KINDS, KINDS, KINDS}};
/* a copy's kind by its width (0 words of 4 bytes, 1 bytes) and its mode (METAL_PLAIN, METAL_LAND, METAL_SEND); a
   counted step's by its width and its kind (nccl-mesh-metal.h METAL_COUNTED_*) */
static const uint8_t COPY[2][METAL_SEND + 1] = {{KIND_copy_plain_4, KIND_copy_land_4, KIND_copy_send_4},
                                                {KIND_copy_plain_1, KIND_copy_land_1, KIND_copy_send_1}};
static const uint8_t COUNTED[2][METAL_COUNTED_OWN + 1] = {
  {KIND_counted_start, KIND_counted_check, KIND_counted_arrived, KIND_counted_send_4, KIND_counted_land_4, KIND_counted_own_4},
  {KIND_counted_start, KIND_counted_check, KIND_counted_arrived, KIND_counted_send_1, KIND_counted_land_1, KIND_counted_own_1}};

ARGS(args, uint64_t, uint32_t);
/* a program: the classic form's command buffer and compute encoder (made here, or the caller's: `borrowed`) and what it
   keeps until the program's end; or the Metal 4 form's caller's encoder, argument table and residency set, the
   device's argument ring, and whether a buffer was made resident since the set's last commit; and either form's
   failure words, those of the communicator whose group it encodes (metal_gate; none: nothing recorded), and
   its waits' bound (`polls`, never 0); its device's kernels (`kernels`, found at its first step: program_kernels) */
struct ring;
struct recording;
struct kernels;
struct metal_program { id<MTLCommandBuffer> buffer; id<MTLComputeCommandEncoder> encoder; NSMutableArray *kept; int borrowed;
                       id<MTL4ComputeCommandEncoder> encoder4; id<MTL4ArgumentTable> table; id<MTLResidencySet> residency;
                       id<MTLBuffer> violation; struct ring *ring; int added; uint32_t polls; struct recording *recording;
                       struct kernels *kernels; };

/* a device's kernels: the device and its pipelines by kind (kernels_of); and its argument ring (the Metal 4 form's
   arguments: RECORDS records of RECORD bytes in `records`, the `next` one written, which the GPU's dispatch of sequence q reads from record (q - 1) mod RECORDS, `progress` the sequence of the last
   dispatch the GPU began, every one before it finished) */
#define RECORDS 65536u
#define RECORD 128u
struct ring { id<MTLBuffer> records, progress; uint64_t next; };
/* a device's track ring (metal_tracks): TRACK_RING bytes, its first 64 the progress word the track kernel stores its
   dispatch's sequence into as it begins (every dispatch before it finished), then each dispatch's records in turn,
   `held` the spans still read: a span of sequence q is free once the progress word reads more than q */
#define TRACK_RING (64u << 20)
struct span { uint64_t sequence, first, end; };
struct track_ring { id<MTLBuffer> buffer; uint64_t sequence, head; struct span *held; size_t helds, room; };
struct kernels { id<MTLDevice> device; id<MTLComputePipelineState> pipelines[PIPELINES]; struct ring ring;
                 struct track_ring tracks; };
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

/* `device`'s argument ring, made at its first Metal 4 program; NULL, *message what failed, where it cannot be */
static struct ring *ring_of(id<MTLDevice> device, const char **message) {
  struct kernels *k = kernels_of(device, message);
  if (!k) return NULL;
  if (!k->ring.records) {
    k->ring.records = [device newBufferWithLength:(NSUInteger)RECORDS * RECORD options:MTLResourceStorageModeShared];
    k->ring.progress = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    if (!k->ring.records || !k->ring.progress) {
      [k->ring.records release]; [k->ring.progress release]; k->ring.records = k->ring.progress = nil;
      *message = say("the argument ring of %u records", RECORDS);
      return NULL;
    }
    memset(k->ring.progress.contents, 0, 64);
  }
  return &k->ring;
}

void metal_records(uint64_t *issued, uint64_t *capacity) {
  const char *message = NULL;
  struct ring *ring = ring_of(metal_device(), &message);
  *issued = ring ? ring->next : 0;
  *capacity = RECORDS - 1;
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

/* a stream's polls as a program's bound: 0, or more than NCCL_MESH_POLLS, is NCCL_MESH_POLLS
   (design/transport.md) */
static uint32_t bound_of(uint32_t polls) { return polls && polls < NCCL_MESH_POLLS ? polls : NCCL_MESH_POLLS; }

struct metal_program *metal_begin(void *command_buffer, void *encoder, void *violation, uint32_t polls, const char **message) {
  struct metal_program *program = calloc(1, sizeof *program);
  if (!program) { *message = "out of memory"; return NULL; }
  program->buffer = [(id<MTLCommandBuffer>)command_buffer retain];
  if (encoder) { program->encoder = [(id<MTLComputeCommandEncoder>)encoder retain]; program->borrowed = 1; }
  program->kept = [NSMutableArray new];
  program->violation = [(id<MTLBuffer>)violation retain];
  program->polls = bound_of(polls);
  return program;
}

struct metal_program *metal_begin4(void *encoder, void *table, void *residency, void *violation, uint32_t polls, const char **message) {
  struct ring *ring = ring_of(((id<MTLResidencySet>)residency).device, message);
  if (!ring) return NULL;
  struct metal_program *program = calloc(1, sizeof *program);
  if (!program) { *message = "out of memory"; return NULL; }
  program->encoder4 = [(id<MTL4ComputeCommandEncoder>)encoder retain];
  program->table = [(id<MTL4ArgumentTable>)table retain];
  program->residency = [(id<MTLResidencySet>)residency retain];
  program->violation = [(id<MTLBuffer>)violation retain];
  program->polls = bound_of(polls);
  program->ring = ring;
  program->borrowed = 1;
  metal_resident(program, ring->records);
  metal_resident(program, ring->progress);
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

/* the Metal 4 form's record of a step: its arguments the ring's next record (refused where the GPU has not begun the
   dispatch that read that record RECORDS dispatches ago: its command buffers not committed, or that far behind), its
   buffers made resident and their addresses in binding order (0 d, 1 s, 2 the record, 3 a, 4 v: the failure words a
   wait names, else the program's (metal_gate), the waits bounded by the program's polls and failures recorded where
   there are such words; 5 progress) */
static struct metal_result record4(struct metal_program *program, id<MTLBuffer> d, id<MTLBuffer> s, id<MTLBuffer> a, id<MTLBuffer> words,
                                   struct args p, uint64_t bound[6]) {
  struct ring *ring = program->ring;
  const uint64_t begun = *(volatile uint64_t *)ring->progress.contents;
  if (ring->next - begun >= RECORDS - 1)
    return (struct metal_result){EOVERFLOW, say("the GPU began none of the library's last %u dispatches (dispatch %llu written, %llu begun): "
                                                "their command buffers are not committed, or that far behind", RECORDS - 1,
                                                (unsigned long long)ring->next, (unsigned long long)begun)};
  id<MTLBuffer> v = words ? words : program->violation ? program->violation : d;
  p.polls = v != d ? program->polls : 0;
  p.sequence = ++ring->next;
  memcpy((char *)ring->records.contents + ((p.sequence - 1) % RECORDS) * RECORD, &p, sizeof p);
  for (id<MTLBuffer> b in @[d, s ? s : d, a ? a : d, v]) metal_resident(program, b);
  const uint64_t made[6] = {d.gpuAddress, (s ? s : d).gpuAddress, ring->records.gpuAddress + ((p.sequence - 1) % RECORDS) * RECORD,
                            (a ? a : d).gpuAddress, v.gpuAddress, ring->progress.gpuAddress};
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
   progress word (5; the classic form binds d there, which no kernel then writes): a GROUP kernel (`group`) as one
   threadgroup of up to 1024 threads over a block's piece, any other over `units` threads */
static struct metal_result dispatch_as(struct metal_program *program, uint32_t kind, void *d, void *s, void *a, void *v, struct args p,
                                       uint64_t units, int group);
/* a GROUP kernel where `a` is given (a stream class's ready or landed words) */
static struct metal_result dispatch(struct metal_program *program, uint32_t kind, void *d, void *s, void *a, void *v, struct args p,
                                    uint64_t units) {
  return dispatch_as(program, kind, d, s, a, v, p, units, a != NULL);
}
static struct metal_result record_step(struct metal_program *program, uint32_t kind, void *d, void *s, void *a, void *v, struct args p);
static int recording_on(const struct metal_program *program);
static struct metal_result dispatch_as(struct metal_program *program, uint32_t kind, void *d, void *s, void *a, void *v, struct args p,
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
  return dispatch(program, COPY[!words][mode], dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = words ? bytes / 4 : bytes},
                  words ? bytes / 4 : bytes);
}

/* operation `op`'s kernel of `type` (TYPED), KINDS where none is compiled */
static uint32_t typed(int op, int type) { return (unsigned)type <= METAL_MERGE ? TYPED[op][type] : KINDS; }

struct metal_result metal_combine(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset,
                                  size_t n) {
  const uint32_t kind = typed(OP_COMBINE, type);
  return dispatch(program, kind, dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = n, .op = (uint32_t)op}, n);
}

struct metal_result metal_combine3(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset,
                                   void *with, size_t with_offset, size_t n) {
  const uint32_t kind = typed(OP_COMBINE3, type);
  return dispatch_as(program, kind, dst, src, with, NULL,
                     (struct args){.dst = dst_offset, .src = src_offset, .n = n, .op = (uint32_t)op, .aux = with_offset}, n, 0);
}

struct metal_result metal_premultiply(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                      uint64_t scalar) {
  const uint32_t kind = typed(OP_PREMUL, type);
  return dispatch(program, kind, dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = n, .scalar = scalar}, n);
}

struct metal_result metal_truncdiv(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                   uint64_t divisor) {
  const uint32_t kind = typed(OP_TRUNCDIV, type);
  return dispatch(program, kind, dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = n, .scalar = divisor}, n);
}

/* a wait's arguments (nccl-mesh-metal.h struct metal_wait): its word at aux for 1 + scalar, its run's first unit at src
   (a landing's piece), its position and channel for its failure's record */
static struct args wait_args(const struct metal_wait *w, size_t dst_offset, size_t n, int op) {
  return (struct args){.dst = dst_offset, .src = w->slot, .n = n, .scalar = w->cycle, .aux = w->word, .op = (uint32_t)op, .at = w->at,
                       .channel = w->channel};
}

/* a release's arguments (struct metal_release): its slot at dst, its count words at aux each scalar */
static struct args release_args(const struct metal_release *r, size_t src_offset, size_t bytes) {
  return (struct args){.dst = r->slot, .src = src_offset, .n = bytes, .scalar = r->value, .aux = r->word, .count = r->count};
}

struct metal_result metal_publish(struct metal_program *program, const struct metal_release *r) {
  return dispatch(program, KIND_publish, r->buffer, NULL, NULL, NULL, release_args(r, 0, 1), 1);
}

struct metal_result metal_spin(struct metal_program *program, const struct metal_wait *w) {
  return dispatch(program, KIND_spin, w->buffer, w->buffer, NULL, w->violation, wait_args(w, 0, 1, 0), 1);
}

struct metal_result metal_send_small(struct metal_program *program, void *src, size_t src_offset, size_t bytes, const struct metal_release *r) {
  return dispatch(program, KIND_send_small, r->buffer, src, r->buffer, NULL, release_args(r, src_offset, bytes), 1);
}

struct metal_result metal_land(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, size_t n, const struct metal_wait *w) {
  const uint32_t kind = type < 0 ? (uint32_t)KIND_land_copy : typed(OP_LAND, type);
  return dispatch(program, kind, dst, w->buffer, w->buffer, w->violation, wait_args(w, dst_offset, n, op), 1);
}

struct metal_result metal_send_deferred(struct metal_program *program, size_t bytes, const struct metal_release *r, uint64_t bound[6]) {
  if (!program->encoder4) return (struct metal_result){EINVAL, "a step run by its caller's kernel is the Metal 4 form's"};
  return record4(program, (id<MTLBuffer>)r->buffer, nil, (id<MTLBuffer>)r->buffer, nil, release_args(r, 0, bytes), bound);
}

struct metal_result metal_land_deferred(struct metal_program *program, int op, void *dst, size_t dst_offset, size_t n, const struct metal_wait *w,
                                        uint64_t bound[6]) {
  if (!program->encoder4) return (struct metal_result){EINVAL, "a step run by its caller's kernel is the Metal 4 form's"};
  return record4(program, (id<MTLBuffer>)dst, (id<MTLBuffer>)w->buffer, (id<MTLBuffer>)w->buffer, (id<MTLBuffer>)w->violation,
                 wait_args(w, dst_offset, n, op), bound);
}

/* a counted all-to-all's dispatch (nccl-mesh-metal.h metal_counted): its kernel by kind (COUNTED), its copies' width the
   widest word that divides every offset, row and piece it moves; each one's arguments as the kernel above reads them */
struct metal_result metal_counted(struct metal_program *program, int kind, const struct metal_counted *k) {
  if (kind < METAL_COUNTED_START || kind > METAL_COUNTED_OWN) return metal_failed(EINVAL);
  const int words = !((k->dst | k->src | k->row | k->bytes | k->aux) & 3);
  const struct args p = {.dst = k->dst, .src = k->src, .n = k->bytes, .scalar = k->scalar, .aux = k->aux, .first = k->first, .slot = k->row,
                         .at = k->limit, .count = k->entries, .mask = kind == METAL_COUNTED_ARRIVED && k->a};
  return dispatch(program, COUNTED[!words][kind], k->d, k->s, kind >= METAL_COUNTED_SEND ? k->a : NULL, kind == METAL_COUNTED_ARRIVED ? k->a : NULL,
                  p, kind == METAL_COUNTED_START ? (k->bytes ? k->bytes : 1) : 1);
}

/* the tracks being recorded (nccl-mesh-metal.h metal_record): each step's kernel and bindings as its dispatch would bind
   them (`v` its failure words, else the program's; its polls the program's where it has such words), each item's
   position, credit, steps, dependencies and the landing a consumption awaits (its first wait: spin's word in d, a
   landing's in a), each track's items by cursor; the buffers named, made resident at metal_tracks */
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
  if (it->cursor == METAL_CONSUME && it->wait_op == UINT32_MAX && WORD[kind]) {
    it->wait_op = (uint32_t)(r->ops - 1);
    it->word_buffer = WORD[kind] == WORD_D ? made.d : made.a;
    it->word = p.aux;
    it->target = 1 + p.scalar;
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
               sizeof(struct track_op) == 136 && sizeof(struct track_dep) == 16, "the track records");

static uint64_t up(uint64_t x) { return (x + 255) & ~(uint64_t)255; }

/* `bytes` of the device's track ring for the dispatch of sequence `sequence`: its offset, or UINT64_MAX where the GPU
   still reads every span that place would take (its dispatches not begun: their command buffers not committed, or that
   far behind) */
static uint64_t ring_take(struct track_ring *t, uint64_t sequence, uint64_t bytes) {
  const uint64_t begun = *(volatile uint64_t *)t->buffer.contents;
  size_t kept = 0;
  for (size_t k = 0; k < t->helds; k++) if (t->held[k].sequence >= begun) t->held[kept++] = t->held[k];
  t->helds = kept;
  uint64_t first = t->head + bytes <= TRACK_RING ? t->head : 64;
  if (first < 64) first = 64;
  if (first + bytes > TRACK_RING) return UINT64_MAX;
  for (size_t k = 0; k < t->helds; k++)
    if (first < t->held[k].end && t->held[k].first < first + bytes) return UINT64_MAX;
  struct span *held = more(t->held, &t->room, t->helds, sizeof *held);
  if (!held) return UINT64_MAX;
  t->held = held;
  t->held[t->helds++] = (struct span){sequence, first, first + bytes};
  t->head = first + bytes;
  return first;
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
    id<MTLDevice> device = k->device;
    struct track_ring *ring = &k->tracks;
    if (!ring->buffer) {
      ring->buffer = [device newBufferWithLength:TRACK_RING options:MTLResourceStorageModeShared];
      if (!ring->buffer) { recording_free(r); return (struct metal_result){ENOMEM, say("the track ring of %u bytes", TRACK_RING)}; }
      memset(ring->buffer.contents, 0, 64);
      ring->head = 64;
    }
    const uint64_t T = r->tracks, at_tracks = up(sizeof(struct track_head)), at_items = up(at_tracks + T * sizeof(struct track_rec));
    const uint64_t at_ops = up(at_items + r->items * sizeof(struct track_item)), at_deps = up(at_ops + r->ops * sizeof(struct track_op));
    const uint64_t at_count = up(at_deps + r->deps * sizeof(struct track_dep)), bytes = up(at_count + T * sizeof(uint64_t));
    const uint64_t sequence = ++ring->sequence, first = ring_take(ring, sequence, bytes);
    if (first == UINT64_MAX) {
      recording_free(r);
      return (struct metal_result){EOVERFLOW, say("the track ring holds no %llu bytes free: the GPU has not begun the tracks that read it (their "
                                                  "command buffers not committed, or that far behind)", (unsigned long long)bytes)};
    }
    unsigned char *base = (unsigned char *)ring->buffer.contents + first;
    const uint64_t address = ring->buffer.gpuAddress;
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
      metal_resident(program, ring->buffer);
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
      [encoder setBuffer:ring->buffer offset:first atIndex:0];
      [encoder setBuffer:ring->buffer offset:0 atIndex:1];
      [encoder dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    }
  }
  recording_free(r);
  return METAL_DONE;
}

void *metal_contents(void *buffer) { return buffer ? ((id<MTLBuffer>)buffer).contents : NULL; }

const char *metal_steps(void) { return STEPS.UTF8String; }

void metal_keep(struct metal_program *program, void *object) { [program->kept addObject:(id)object]; }

/* the program's end signals `event` to `value`; what it kept is released once the event has passed it
   (metal_collect); one with no event (a failed group's, nccl-mesh.c program_end) signals nothing, and what its own
   encoder's commands kept is held by its command buffer (a completion handler's capture) until that buffer completes
   or is freed; a program in a borrowed encoder keeps nothing; the Metal 4 form's residency set committed where the
   program made a buffer resident */
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
  free(program);
}

void metal_collect(void *event) {
  const uint64_t signaled = metal_signaled(event);
  @autoreleasepool {
    NSIndexSet *done = [retiring_ indexesOfObjectsPassingTest:^BOOL(NSArray *entry, NSUInteger index, BOOL *stop) {
      (void)index; (void)stop; return [entry[0] unsignedLongLongValue] <= signaled; }];
    [retiring_ removeObjectsAtIndexes:done];
  }
}
