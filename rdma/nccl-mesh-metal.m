#import <Metal/Metal.h>
#include "nccl-mesh-metal.h"
#include "mesh_metal.h"
#include "mesh.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#define STR(...) #__VA_ARGS__
#define XSTR(...) STR(__VA_ARGS__)
/* a kernel's arguments, one layout for the host (struct args) and the kernels (struct mesh_args) */
#define ARGS(NAME, U64, U32) struct NAME { U64 dst, src, n, scalar, aux; U32 op, polls; U64 first, slot, at, sequence; U32 count, mask, channel, depth; }

/* a step of a compiled program, the library's one implementation of it (metal-microbench docs/acceptance-20261014.md
   A39), after rdma/mesh.metal: libnccl-mesh's kernels call these, and so does a caller's kernel that runs a step its
   group handed it (ncclMeshFused: a product's finish that publishes its piece, a layer's finish that lands its peer's).
   mesh_step_enter: in the Metal 4 form a kernel first stores its record's sequence in `progress` (every dispatch before
   it has finished: a barrier stands between each two), whether or not its program has violation words (the classic
   form's sequence is 0, and it stores none), and, where it has them (p.polls), goes on only while violation word [2]
   holds no failure.  mesh_step_wait: a wait on a completion word is mesh_remote_wait (which the session's canceller
   ends) or, where p.polls is set (a bounded program, metal_begin), mesh_remote_wait_bounded; mesh_step_check: a word
   short of its target past the bound is recorded in violation words [2], [3] and [4], else the positions the record's
   check names have their slots checked against the filler protocol (the stamp; violation words [0] and [1]).
   mesh_step_await: a landing awaited by thread 0 and checked, then the threadgroup past a device barrier and a
   system-scope fence (its payload then read with mesh_payload).  mesh_step_release: a piece the threadgroup stored
   into its slot, its cell released by thread 0 after a device barrier.  One threadgroup a step.  A caller's kernel
   stores its piece's element k with mesh_step_store and reads its peer's fourth k with mesh_step_read (float16), so
   it names no transport qualifier of its own (they are compiled here, under mesh.metal's pragma: a header after it,
   MetalPerformancePrimitives', may end that pragma). */
static NSString *const STEPS =
  @"\nusing namespace metal;\n"
  XSTR(ARGS(mesh_args, ulong, uint)) ";\n"
  "static inline ulong mesh_step_stamp(ulong t) { return " XSTR(METAL_STAMP(t)) "; }\n"
  "static inline bool mesh_step_enter(constant mesh_args &p, device uchar *vb, device ulong *progress, bool first) { "
  "if (p.sequence && first) *progress = p.sequence; return !(p.polls && ((volatile device const ulong *)vb)[2]); }\n"
  "static inline ulong mesh_step_wait(constant mesh_args &p, device uchar *word, ulong cycle) { "
  "return p.polls ? mesh_remote_wait_bounded((volatile coherent(system) device const ulong *)word, cycle, p.polls) "
  ": mesh_remote_wait((volatile coherent(system) device const ulong *)word, cycle); }\n"
  "static inline void mesh_step_check(constant mesh_args &p, ulong status, device uchar *ring, device uchar *vb, ulong cycle) { "
  "volatile coherent(system) device ulong *v = (volatile coherent(system) device ulong *)vb; "
  "if (p.polls && status != ~0UL && status < 1 + cycle) { if (v[2] == 0) { v[3] = ulong(p.channel); v[4] = ulong(p.polls); mesh_publish(v + 2, p.at + 1); } } "
  "else if (status != ~0UL) for (uint c = 0; c < p.count; c++) { ulong t = p.first + c; "
  "ulong seen = mesh_payload((volatile coherent(system) device const ulong *)(ring + (t % p.depth) * p.slot)); bool data = (p.mask >> c) & 1u; "
  "if (data == (seen == mesh_step_stamp(t)) && v[0] == 0) { v[1] = (ulong(p.channel) << 1) | ulong(data); mesh_publish(v, t + 1); } } }\n"
  "static inline bool mesh_step_await(constant mesh_args &p, device uchar *words, device uchar *ring, device uchar *vb, uint i) { "
  "if (i == 0) { ulong status = mesh_step_wait(p, words + p.aux, p.scalar); mesh_step_check(p, status, ring, vb, p.scalar); } "
  "threadgroup_barrier(mem_flags::mem_device); "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3)); "
  "return !(p.polls && ((volatile device const ulong *)vb)[2]); }\n"
  "static inline void mesh_step_release(constant mesh_args &p, device uchar *cells, uint i) { "
  "threadgroup_barrier(mem_flags::mem_device); if (i == 0) { *(volatile coherent(system) device uint *)(cells + p.aux + " XSTR(MESH_SEND_LENGTH) ") = uint(p.n); "
  "mesh_publish((volatile coherent(system) device ulong *)(cells + p.aux), p.scalar); } }\n"
  "static inline void mesh_step_store(constant mesh_args &p, device uchar *slot, ulong k, half value) { "
  "((volatile coherent(system) device half *)(slot + p.dst))[k] = value; }\n"
  "static inline half4 mesh_step_read(constant mesh_args &p, device uchar *slot, ulong k) { "
  "return mesh_payload((volatile coherent(system) device const half4 *)(slot + p.src) + k); }\n";

/* the kernels, after STEPS: each operator's arithmetic is written once (op_int, op_float, op_f8), a landing being the
   wait and then the combine's body; it is the host's (nccl-mesh.c combine_into, premultiply, truncdiv): the small
   floating types in float32 rounded to nearest even, float8 decoded and encoded by mesh.metal's f8_to and f8_from,
   integers wrapping */
static NSString *const KERNELS =
  @"\n"
  "#define ENTER(first) if (!mesh_step_enter(p, vb, progress, first)) return;\n"
  "#define HEAD(name) kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]], uint i [[thread_position_in_grid]], uint g [[threads_per_grid]])\n"
  "#define GROUP(name) kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *a [[buffer(3)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]], uint i [[thread_position_in_threadgroup]], uint g [[threads_per_threadgroup]])\n"
  "#define LOOP for (ulong k = i; k < p.n; k += g)\n"
  "#define AWAIT if (!mesh_step_await(p, a, s, vb, i)) return;\n"
  "#define COPY(Z, W) "
  "HEAD(copy_plain_##Z) { ENTER(i == 0) device W *x = (device W *)(d + p.dst); device const W *y = (device const W *)(s + p.src); LOOP x[k] = y[k]; } "
  "HEAD(copy_land_##Z) { ENTER(i == 0) device W *x = (device W *)(d + p.dst); volatile coherent(system) device const W *y = (volatile coherent(system) device const W *)(s + p.src); LOOP x[k] = mesh_payload(y + k); } "
  "HEAD(copy_send_##Z) { ENTER(i == 0) volatile coherent(system) device W *x = (volatile coherent(system) device W *)(d + p.dst); device const W *y = (device const W *)(s + p.src); LOOP x[k] = y[k]; }\n"
  "COPY(4, uint)\n"
  "COPY(1, uchar)\n"
  "template <typename T> T op_int(T a, T b, uint op) { return op == 0 ? T(ulong(a) + ulong(b)) : op == 1 ? T(ulong(a) * ulong(b)) : (op == 2 ? b > a : b < a) ? b : a; }\n"
  "template <typename T> T op_float(T a, T b, uint op) { float x = float(a), y = float(b); return op == 0 ? T(x + y) : op == 1 ? T(x * y) : (op == 2 ? y > x : y < x) ? b : a; }\n"
  "template <bool E5> uchar op_f8(uchar a, uchar b, uint op) { float x = f8_to(a, E5), y = f8_to(b, E5); "
  "return op == 0 ? f8_from(x + y, E5) : op == 1 ? f8_from(x * y, E5) : (op == 2 ? y > x : y < x) ? b : a; }\n"
  "#define BODY(T, OP) device T *x = (device T *)(d + p.dst); volatile coherent(system) device const T *y = (volatile coherent(system) device const T *)(s + p.src); "
  "LOOP x[k] = OP(x[k], mesh_payload(y + k), p.op);\n"
  "#define COMBINE(name, T, OP) HEAD(combine_##name) { ENTER(i == 0) BODY(T, OP) } GROUP(land_##name) { ENTER(i == 0) AWAIT BODY(T, OP) }\n"
  "#define INTEGER(name, T, SIGNED) COMBINE(name, T, op_int<T>) "
  "HEAD(premul_##name) { ENTER(i == 0) device T *x = (device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); ulong c = ulong(T(p.scalar)); LOOP x[k] = T(ulong(y[k]) * c); } "
  "HEAD(truncdiv_##name) { ENTER(i == 0) device T *x = (device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); LOOP { T v = y[k]; bool negative = SIGNED && v < T(0); "
  "ulong m = negative ? 0UL - ulong(v) : ulong(v), q = m / p.scalar; x[k] = negative ? T(0UL - q) : T(q); } }\n"
  "#define FLOATING(name, T) COMBINE(name, T, op_float<T>) "
  "HEAD(premul_##name) { ENTER(i == 0) device T *x = (device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); float c = as_type<float>(uint(p.scalar)); LOOP x[k] = T(float(y[k]) * c); }\n"
  "#define FLOAT8(name, E5) COMBINE(name, uchar, op_f8<E5>) "
  "HEAD(premul_##name) { ENTER(i == 0) device uchar *x = (device uchar *)(d + p.dst); device const uchar *y = (device const uchar *)(s + p.src); float c = as_type<float>(uint(p.scalar)); LOOP x[k] = f8_from(f8_to(y[k], E5) * c, E5); }\n"
  "INTEGER(i8, char, true)\n"
  "INTEGER(u8, uchar, false)\n"
  "INTEGER(i32, int, true)\n"
  "INTEGER(u32, uint, false)\n"
  "INTEGER(i64, long, true)\n"
  "INTEGER(u64, ulong, false)\n"
  "FLOATING(f16, half)\n"
  "FLOATING(f32, float)\n"
  "FLOATING(bf16, bfloat)\n"
  "FLOAT8(e4m3, false)\n"
  "FLOAT8(e5m2, true)\n"
  "kernel void publish(device uchar *cells [[buffer(0)]], constant mesh_args &p [[buffer(2)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]]) { "
  "ENTER(true) mesh_publish((volatile coherent(system) device ulong *)(cells + p.dst), p.scalar); }\n"
  "kernel void publish_filler(device uchar *cells [[buffer(0)]], device uchar *ring [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]]) { "
  "ENTER(true) *(volatile coherent(system) device ulong *)(ring + p.src) = mesh_step_stamp(p.first); "
  "*(volatile coherent(system) device uint *)(cells + p.dst + " XSTR(MESH_SEND_LENGTH) ") = uint(p.n); "
  "mesh_publish((volatile coherent(system) device ulong *)(cells + p.dst), p.scalar); }\n"
  "GROUP(send_small) { ENTER(i == 0) volatile coherent(system) device uchar *x = (volatile coherent(system) device uchar *)(d + p.dst); device const uchar *y = (device const uchar *)(s + p.src); LOOP x[k] = y[k]; "
  "mesh_step_release(p, a, i); }\n"
  "GROUP(land_copy) { ENTER(i == 0) AWAIT device uchar *x = d + p.dst; volatile coherent(system) device const uchar *y = (volatile coherent(system) device const uchar *)(s + p.src); LOOP x[k] = mesh_payload(y + k); }\n"
  "kernel void spin(device uchar *words [[buffer(0)]], device uchar *ring [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]]) { "
  "ENTER(true) mesh_step_check(p, mesh_step_wait(p, words + p.dst, p.n), ring, vb, p.n); }\n";

/* ncclDataType_t's order: int8 uint8 int32 uint32 int64 uint64 float16 float32 float64 bfloat16 e4m3 e5m2 */
static const char *const TYPE[12] = {"i8", "u8", "i32", "u32", "i64", "u64", "f16", "f32", NULL, "bf16", "e4m3", "e5m2"};
static const char *const MODE[3] = {"plain", "land", "send"};

ARGS(args, uint64_t, uint32_t);
/* a program: the classic form's command buffer and compute encoder (made here, or the caller's: `borrowed`) and what it
   keeps until the program's end; or the Metal 4 form's caller's encoder, argument table and residency set, the
   device's argument ring, and whether a buffer was made resident since the set's last commit; and either form's
   session's violation words and its waits' bound (`polls`; no words or 0: the waits unbounded and nothing recorded) */
struct ring;
struct metal_program { id<MTLCommandBuffer> buffer; id<MTLComputeCommandEncoder> encoder; NSMutableArray *kept; int borrowed;
                       id<MTL4ComputeCommandEncoder> encoder4; id<MTL4ArgumentTable> table; id<MTLResidencySet> residency;
                       id<MTLBuffer> violation; struct ring *ring; int added; uint32_t polls; };

/* a device's kernels: its library and pipelines by name; and its argument ring (the Metal 4 form's arguments: RECORDS
   records of RECORD bytes in `records`, the `next` one written, which the GPU's dispatch of sequence q reads from record
   (q - 1) mod RECORDS, `progress` the sequence of the last dispatch the GPU began, every one before it finished) */
#define RECORDS 65536u
#define RECORD 128u
struct ring { id<MTLBuffer> records, progress; uint64_t next; };
struct kernels { id<MTLLibrary> library; NSMutableDictionary *pipelines; struct ring ring; };
_Static_assert(sizeof(struct args) <= RECORD, "an argument record");

static id<MTLDevice> device_;
static NSMutableDictionary *kernels_;
static NSMutableArray *retiring_;
static char error_[512];

void *metal_device(void) {
  static dispatch_once_t once;
  dispatch_once(&once, ^{ device_ = MTLCreateSystemDefaultDevice(); });
  return device_;
}

const char *metal_error(void) { return error_; }

/* `device`'s kernels, compiled at its first use */
static struct kernels *kernels_of(id<MTLDevice> device) {
  @autoreleasepool {
    if (!kernels_) { kernels_ = [NSMutableDictionary new]; retiring_ = [NSMutableArray new]; }
    NSValue *key = [NSValue valueWithPointer:(__bridge const void *)device];
    NSValue *held = kernels_[key];
    if (held) return held.pointerValue;
    MTLCompileOptions *options = [[MTLCompileOptions new] autorelease];
    options.mathMode = MTLMathModeSafe;
    NSError *error = nil;
    NSString *primitives = [[[NSString alloc] initWithBytes:mesh_metal length:mesh_metal_len encoding:NSUTF8StringEncoding] autorelease];
    id<MTLLibrary> library = [device newLibraryWithSource:[[primitives stringByAppendingString:STEPS] stringByAppendingString:KERNELS] options:options error:&error];
    if (!library) { snprintf(error_, sizeof error_, "the kernels: %s", error.localizedDescription.UTF8String); return NULL; }
    struct kernels *made = calloc(1, sizeof *made);
    if (!made) { [library release]; return NULL; }
    made->library = library;
    made->pipelines = [NSMutableDictionary new];
    [device retain];
    kernels_[key] = [NSValue valueWithPointer:made];
    return made;
  }
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, NSString *name) {
  @autoreleasepool {
    struct kernels *k = kernels_of(device);
    if (!k) return nil;
    id<MTLComputePipelineState> made = k->pipelines[name];
    if (made) return made;
    id<MTLFunction> function = [[k->library newFunctionWithName:name] autorelease];
    NSError *error = nil;
    made = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
    if (!made) { snprintf(error_, sizeof error_, "kernel %s: %s", name.UTF8String, error ? error.localizedDescription.UTF8String : "absent"); return nil; }
    k->pipelines[name] = made;
    [made release];
    return made;
  }
}

/* `device`'s argument ring, made at its first Metal 4 program */
static struct ring *ring_of(id<MTLDevice> device) {
  struct kernels *k = kernels_of(device);
  if (!k) return NULL;
  if (!k->ring.records) {
    k->ring.records = [device newBufferWithLength:(NSUInteger)RECORDS * RECORD options:MTLResourceStorageModeShared];
    k->ring.progress = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    if (!k->ring.records || !k->ring.progress) {
      [k->ring.records release]; [k->ring.progress release]; k->ring.records = k->ring.progress = nil;
      snprintf(error_, sizeof error_, "the argument ring of %u records", RECORDS);
      return NULL;
    }
    memset(k->ring.progress.contents, 0, 64);
  }
  return &k->ring;
}

void metal_records(uint64_t *issued, uint64_t *capacity) {
  struct ring *ring = ring_of(metal_device());
  *issued = ring ? ring->next : 0;
  *capacity = RECORDS - 1;
}

void *metal_wrap(void *address, size_t bytes) {
  return [(id<MTLDevice>)metal_device() newBufferWithBytesNoCopy:address length:bytes options:MTLResourceStorageModeShared deallocator:nil];
}

void *metal_scratch(size_t bytes) {
  return [(id<MTLDevice>)metal_device() newBufferWithLength:bytes ? bytes : 16 options:MTLResourceStorageModePrivate];
}

void *metal_event(void) { return [(id<MTLDevice>)metal_device() newSharedEvent]; }
uint64_t metal_signaled(void *event) { return ((id<MTLSharedEvent>)event).signaledValue; }
int metal_wait(void *event, uint64_t value, uint64_t ns) { return [(id<MTLSharedEvent>)event waitUntilSignaledValue:value timeoutMS:ns / 1000000]; }
void metal_release(void *object) { [(id)object release]; }

struct metal_program *metal_begin(void *command_buffer, void *encoder, void *violation, uint32_t polls) {
  struct metal_program *program = calloc(1, sizeof *program);
  if (!program) return NULL;
  program->buffer = [(id<MTLCommandBuffer>)command_buffer retain];
  if (encoder) { program->encoder = [(id<MTLComputeCommandEncoder>)encoder retain]; program->borrowed = 1; }
  program->kept = [NSMutableArray new];
  program->violation = [(id<MTLBuffer>)violation retain];
  program->polls = polls;
  return program;
}

struct metal_program *metal_begin4(void *encoder, void *table, void *residency, void *violation, uint32_t polls) {
  struct ring *ring = ring_of(((id<MTLResidencySet>)residency).device);
  struct metal_program *program = ring ? calloc(1, sizeof *program) : NULL;
  if (!program) return NULL;
  program->encoder4 = [(id<MTL4ComputeCommandEncoder>)encoder retain];
  program->table = [(id<MTL4ArgumentTable>)table retain];
  program->residency = [(id<MTLResidencySet>)residency retain];
  program->violation = [(id<MTLBuffer>)violation retain];
  program->polls = polls;
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

static void close_encoder(struct metal_program *program) {
  if (!program->encoder) return;
  if (!program->borrowed) [program->encoder endEncoding];
  [program->encoder release];
  program->encoder = nil;
}

/* the Metal 4 form's record of a step: its arguments the ring's next record (refused where the GPU has not begun the
   dispatch that read that record RECORDS dispatches ago: its command buffers not committed, or that far behind), its
   buffers made resident and their addresses in binding order (0 d, 1 s, 2 the record, 3 a, 4 v: the words a check
   names, else the program's session's, the waits bounded by the program's polls and failures recorded where there are
   such words; 5 progress) */
static int record4(struct metal_program *program, id<MTLBuffer> d, id<MTLBuffer> s, id<MTLBuffer> a, id<MTLBuffer> words, struct args p,
                   uint64_t bound[6]) {
  struct ring *ring = program->ring;
  const uint64_t begun = *(volatile uint64_t *)ring->progress.contents;
  if (ring->next - begun >= RECORDS - 1) {
    snprintf(error_, sizeof error_, "the GPU began none of the library's last %u dispatches (dispatch %llu written, %llu begun): their command "
             "buffers are not committed, or that far behind", RECORDS - 1, (unsigned long long)ring->next, (unsigned long long)begun);
    return EOVERFLOW;
  }
  id<MTLBuffer> v = words ? words : program->violation ? program->violation : d;
  p.polls = v != d ? program->polls : 0;
  p.sequence = ++ring->next;
  memcpy((char *)ring->records.contents + ((p.sequence - 1) % RECORDS) * RECORD, &p, sizeof p);
  for (id<MTLBuffer> b in @[d, s ? s : d, a ? a : d, v]) metal_resident(program, b);
  const uint64_t made[6] = {d.gpuAddress, (s ? s : d).gpuAddress, ring->records.gpuAddress + ((p.sequence - 1) % RECORDS) * RECORD,
                            (a ? a : d).gpuAddress, v.gpuAddress, ring->progress.gpuAddress};
  memcpy(bound, made, sizeof made);
  return 0;
}

/* the Metal 4 form's dispatch: its record (record4), every buffer bound by address through the caller's table, then a
   barrier before the next */
static int dispatch4(struct metal_program *program, id<MTLComputePipelineState> state, id<MTLBuffer> d, id<MTLBuffer> s, id<MTLBuffer> a,
                     id<MTLBuffer> words, struct args p, uint64_t units, int group) {
  uint64_t bound[6];
  const int recorded = record4(program, d, s, a, words, p, bound);
  if (recorded) return recorded;
  for (NSUInteger k = 0; k < 6; k++) [program->table setAddress:bound[k] atIndex:k];
  id<MTL4ComputeCommandEncoder> encoder = program->encoder4;
  [encoder setComputePipelineState:state];
  [encoder setArgumentTable:program->table];
  const NSUInteger most = group ? 1024 : 256, width = state.maxTotalThreadsPerThreadgroup < most ? state.maxTotalThreadsPerThreadgroup : most;
  if (group) [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  else [encoder dispatchThreads:MTLSizeMake(units < (1u << 20) ? (NSUInteger)units : (1u << 20), 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
  return 0;
}

/* kernel `name` on buffers d, s, a and v (0, 1, 3 and 4; an absent one d, and v the program's violation words where
   its waits are bounded) and the arguments p (2, its polls the program's where v is such words), the progress word
   (5; the classic form binds d there, which no kernel then writes): a GROUP kernel (`a` given, the cells or the
   completion words) as one threadgroup of up to 1024 threads over a block's piece, any other over `units` threads */
static int dispatch(struct metal_program *program, NSString *name, void *d, void *s, void *a, void *v, struct args p, uint64_t units) {
  if (!units) return 0;
  id<MTLComputePipelineState> state = pipeline(program->encoder4 ? program->residency.device : (id<MTLDevice>)metal_device(), name);
  if (!state) return EINVAL;
  @autoreleasepool {
    if (program->encoder4) return dispatch4(program, state, (id<MTLBuffer>)d, (id<MTLBuffer>)s, (id<MTLBuffer>)a, (id<MTLBuffer>)v, p, units, a != NULL);
    if (!program->encoder) program->encoder = [[program->buffer computeCommandEncoder] retain];
    id<MTLComputeCommandEncoder> encoder = program->encoder;
    if (!v && program->polls) v = (void *)program->violation;
    p.polls = v && v != d ? program->polls : 0;
    [encoder setComputePipelineState:state];
    [encoder setBuffer:(id<MTLBuffer>)d offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)(s ? s : d) offset:0 atIndex:1];
    [encoder setBytes:&p length:sizeof p atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)(a ? a : d) offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)(v ? v : d) offset:0 atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)d offset:0 atIndex:5];
    const NSUInteger most = a ? 1024 : 256, width = state.maxTotalThreadsPerThreadgroup < most ? state.maxTotalThreadsPerThreadgroup : most;
    if (a) [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    else [encoder dispatchThreads:MTLSizeMake(units < (1u << 20) ? (NSUInteger)units : (1u << 20), 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  }
  return 0;
}

int metal_copy(struct metal_program *program, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes) {
  const int words = !((dst_offset | src_offset | bytes) & 3);
  NSString *name = [NSString stringWithFormat:@"copy_%s_%d", MODE[mode], words ? 4 : 1];
  return dispatch(program, name, dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = words ? bytes / 4 : bytes}, words ? bytes / 4 : bytes);
}

static NSString *typed(const char *what, int type) {
  return (unsigned)type < 12 && TYPE[type] ? [NSString stringWithFormat:@"%s_%s", what, TYPE[type]] : nil;
}

int metal_combine(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n) {
  NSString *name = typed("combine", type);
  return name ? dispatch(program, name, dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = n, .op = (uint32_t)op}, n) : EINVAL;
}

int metal_premultiply(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t scalar) {
  NSString *name = typed("premul", type);
  return name ? dispatch(program, name, dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = n, .scalar = scalar}, n) : EINVAL;
}

int metal_truncdiv(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t divisor) {
  NSString *name = type <= 5 ? typed("truncdiv", type) : nil;
  return name ? dispatch(program, name, dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = n, .scalar = divisor}, n) : EINVAL;
}

static struct args checked(struct args a, const struct metal_check *check) {
  if (check) {
    a.first = check->first; a.slot = check->slot; a.count = check->count; a.mask = check->mask; a.channel = check->channel;
    a.depth = check->depth; a.at = check->at;
  }
  return a;
}

int metal_publish(struct metal_program *program, void *cells, size_t offset, uint64_t argument) {
  return dispatch(program, @"publish", cells, NULL, NULL, NULL, (struct args){.dst = offset, .n = 1, .scalar = argument}, 1);
}

int metal_publish_filler(struct metal_program *program, void *cells, size_t offset, uint64_t argument, void *slot, size_t slot_offset,
                         uint64_t position, size_t length) {
  return dispatch(program, @"publish_filler", cells, slot, NULL, NULL,
                  (struct args){.dst = offset, .src = slot_offset, .n = length, .scalar = argument, .first = position}, 1);
}

int metal_spin(struct metal_program *program, void *words, size_t offset, uint64_t cycle, const struct metal_check *check) {
  return dispatch(program, @"spin", words, check ? check->ring : NULL, NULL, check ? check->violation : NULL,
                  checked((struct args){.dst = offset, .n = cycle}, check), 1);
}

static struct args send_args(size_t slot_offset, size_t src_offset, size_t bytes, size_t cell_offset, uint64_t value) {
  return (struct args){.dst = slot_offset, .src = src_offset, .n = bytes, .scalar = value, .aux = cell_offset};
}

static struct args land_args(int op, size_t dst_offset, size_t slot_offset, size_t n, size_t word_offset, uint64_t cycle,
                             const struct metal_check *check) {
  return checked((struct args){.dst = dst_offset, .src = slot_offset, .n = n, .scalar = cycle, .aux = word_offset, .op = (uint32_t)op}, check);
}

int metal_send_small(struct metal_program *program, void *slot, size_t slot_offset, void *src, size_t src_offset, size_t bytes,
                     void *cells, size_t cell_offset, uint64_t value) {
  return dispatch(program, @"send_small", slot, src, cells, NULL, send_args(slot_offset, src_offset, bytes, cell_offset, value), 1);
}

int metal_land(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *slot, size_t slot_offset, size_t n,
               void *words, size_t word_offset, uint64_t cycle, const struct metal_check *check) {
  NSString *name = type < 0 ? @"land_copy" : typed("land", type);
  return name ? dispatch(program, name, dst, slot, words, check ? check->violation : NULL,
                         land_args(op, dst_offset, slot_offset, n, word_offset, cycle, check), 1)
              : EINVAL;
}

int metal_send_deferred(struct metal_program *program, void *slot, size_t slot_offset, size_t bytes, void *cells, size_t cell_offset,
                        uint64_t value, uint64_t bound[6]) {
  if (!program->encoder4) { snprintf(error_, sizeof error_, "a step run by its caller's kernel is the Metal 4 form's"); return EINVAL; }
  return record4(program, (id<MTLBuffer>)slot, nil, (id<MTLBuffer>)cells, nil, send_args(slot_offset, 0, bytes, cell_offset, value), bound);
}

int metal_land_deferred(struct metal_program *program, int op, void *dst, size_t dst_offset, void *slot, size_t slot_offset, size_t n,
                        void *words, size_t word_offset, uint64_t cycle, const struct metal_check *check, uint64_t bound[6]) {
  if (!program->encoder4) { snprintf(error_, sizeof error_, "a step run by its caller's kernel is the Metal 4 form's"); return EINVAL; }
  return record4(program, (id<MTLBuffer>)dst, (id<MTLBuffer>)slot, (id<MTLBuffer>)words, check ? (id<MTLBuffer>)check->violation : nil,
                 land_args(op, dst_offset, slot_offset, n, word_offset, cycle, check), bound);
}

const char *metal_steps(void) { return STEPS.UTF8String; }

void metal_keep(struct metal_program *program, void *object) { [program->kept addObject:(id)object]; }

/* the program's end signals `event` to `value`; what it kept is released once the event has passed it
   (metal_collect); a program in a borrowed encoder (event NULL) signals nothing and keeps nothing; the Metal 4 form's
   residency set committed where the program made a buffer resident */
void metal_end(struct metal_program *program, void *event, uint64_t value) {
  close_encoder(program);
  if (event) {
    [program->buffer encodeSignalEvent:(id<MTLEvent>)event value:value];
    @autoreleasepool { [retiring_ addObject:@[@(value), program->kept]]; }
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
