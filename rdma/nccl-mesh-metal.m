#import <Metal/Metal.h>
#include "nccl-mesh-metal.h"
#include "mesh_metal.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

/* the kernels, after rdma/mesh.metal (its text embedded at build: .build/mesh_metal.h): a wait on a completion word is
   its mesh_remote_wait, a landed slot's read its mesh_payload and a cell's release its mesh_publish; the filler protocol
   (STAMP, CHECK) is this library's.  Each operator's arithmetic is written once (op_int, op_float, op_f8), a landing
   being the wait and then the combine's body; it is the host's (nccl-mesh.c combine_into, premultiply, truncdiv): the
   small floating types in float32 rounded to nearest even, float8 decoded and encoded by f8_to and f8_from line for
   line, integers wrapping */
static NSString *const KERNELS =
  @"\nusing namespace metal;\n"
  "struct args { ulong dst, src, n, scalar, aux; uint op, unused; ulong first, slot; uint count, mask, channel, depth; };\n"
  "#define STAMP(t) (0x6d6573682d66696cUL ^ ((t) * 0x9e3779b97f4a7c15UL))\n"
  "#define CHECK(ring, status) if (status != ~0UL) for (uint c = 0; c < p.count; c++) { ulong t = p.first + c; "
  "ulong seen = mesh_payload((volatile coherent(system) device const ulong *)(ring + (t % p.depth) * p.slot)); bool data = (p.mask >> c) & 1u; "
  "if (data == (seen == STAMP(t)) && v[0] == 0) { v[1] = (ulong(p.channel) << 1) | ulong(data); mesh_publish(v, t + 1); } }\n"
  "#define HEAD(name) kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant args &p [[buffer(2)]], uint i [[thread_position_in_grid]], uint g [[threads_per_grid]])\n"
  "#define GROUP(name) kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant args &p [[buffer(2)]], device uchar *a [[buffer(3)]], device uchar *vb [[buffer(4)]], uint i [[thread_position_in_threadgroup]], uint g [[threads_per_threadgroup]])\n"
  "#define LOOP for (ulong k = i; k < p.n; k += g)\n"
  "#define AWAIT if (i == 0) { ulong status = mesh_remote_wait((volatile coherent(system) device const ulong *)(a + p.aux), p.scalar); "
  "volatile coherent(system) device ulong *v = (volatile coherent(system) device ulong *)vb; CHECK(s, status) } threadgroup_barrier(mem_flags::mem_device); "
  "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3));\n"
  "#define COPY(Z, W) "
  "HEAD(copy_plain_##Z) { device W *x = (device W *)(d + p.dst); device const W *y = (device const W *)(s + p.src); LOOP x[k] = y[k]; } "
  "HEAD(copy_land_##Z) { device W *x = (device W *)(d + p.dst); volatile coherent(system) device const W *y = (volatile coherent(system) device const W *)(s + p.src); LOOP x[k] = mesh_payload(y + k); } "
  "HEAD(copy_send_##Z) { volatile coherent(system) device W *x = (volatile coherent(system) device W *)(d + p.dst); device const W *y = (device const W *)(s + p.src); LOOP x[k] = y[k]; }\n"
  "COPY(4, uint)\n"
  "COPY(1, uchar)\n"
  "template <typename T> T op_int(T a, T b, uint op) { return op == 0 ? T(ulong(a) + ulong(b)) : op == 1 ? T(ulong(a) * ulong(b)) : (op == 2 ? b > a : b < a) ? b : a; }\n"
  "template <typename T> T op_float(T a, T b, uint op) { float x = float(a), y = float(b); return op == 0 ? T(x + y) : op == 1 ? T(x * y) : (op == 2 ? y > x : y < x) ? b : a; }\n"
  "float f8_to(uchar v, bool e5) { int mbits = e5 ? 2 : 3, bias = e5 ? 15 : 7; int expf = (v >> mbits) & (e5 ? 31 : 15), mant = v & ((1 << mbits) - 1); float r; "
  "if (e5 && expf == 31) r = as_type<float>(mant ? 0x7fc00000u : 0x7f800000u); else if (!e5 && expf == 15 && mant == 7) r = as_type<float>(0x7fc00000u); "
  "else if (!expf) r = ldexp(float(mant), 1 - bias - mbits); else r = ldexp(1.0f + float(mant) / float(1 << mbits), expf - bias); return (v & 0x80) ? -r : r; }\n"
  "uchar f8_from(float x, bool e5) { int mbits = e5 ? 2 : 3, bias = e5 ? 15 : 7; uint u = as_type<uint>(x), a = u & 0x7fffffffu; uchar sign = (u >> 31) ? 0x80 : 0; "
  "if (a > 0x7f800000u) return sign | 0x7f; if (a == 0x7f800000u) return e5 ? (sign | 0x7c) : (sign | 0x7f); if (a == 0) return sign; "
  "int e = max(int(a >> 23) - 127, 1 - bias); float n = rint(ldexp(as_type<float>(a), mbits - e)); if (n >= float(1 << (mbits + 1))) { n *= 0.5f; e += 1; } "
  "int expf, mant; if (n < float(1 << mbits)) { expf = 0; mant = int(n); } else { expf = e + bias; mant = int(n) - (1 << mbits); } "
  "if (e5 ? expf >= 31 : (expf > 15 || (expf == 15 && mant == 7))) return e5 ? (sign | 0x7c) : (sign | 0x7f); return sign | uchar(expf << mbits) | uchar(mant); }\n"
  "template <bool E5> uchar op_f8(uchar a, uchar b, uint op) { float x = f8_to(a, E5), y = f8_to(b, E5); "
  "return op == 0 ? f8_from(x + y, E5) : op == 1 ? f8_from(x * y, E5) : (op == 2 ? y > x : y < x) ? b : a; }\n"
  "#define BODY(T, OP) device T *x = (device T *)(d + p.dst); volatile coherent(system) device const T *y = (volatile coherent(system) device const T *)(s + p.src); "
  "LOOP x[k] = OP(x[k], mesh_payload(y + k), p.op);\n"
  "#define COMBINE(name, T, OP) HEAD(combine_##name) { BODY(T, OP) } GROUP(land_##name) { AWAIT BODY(T, OP) }\n"
  "#define INTEGER(name, T, SIGNED) COMBINE(name, T, op_int<T>) "
  "HEAD(premul_##name) { device T *x = (device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); ulong c = ulong(T(p.scalar)); LOOP x[k] = T(ulong(y[k]) * c); } "
  "HEAD(truncdiv_##name) { device T *x = (device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); LOOP { T v = y[k]; bool negative = SIGNED && v < T(0); "
  "ulong m = negative ? 0UL - ulong(v) : ulong(v), q = m / p.scalar; x[k] = negative ? T(0UL - q) : T(q); } }\n"
  "#define FLOATING(name, T) COMBINE(name, T, op_float<T>) "
  "HEAD(premul_##name) { device T *x = (device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); float c = as_type<float>(uint(p.scalar)); LOOP x[k] = T(float(y[k]) * c); }\n"
  "#define FLOAT8(name, E5) COMBINE(name, uchar, op_f8<E5>) "
  "HEAD(premul_##name) { device uchar *x = (device uchar *)(d + p.dst); device const uchar *y = (device const uchar *)(s + p.src); float c = as_type<float>(uint(p.scalar)); LOOP x[k] = f8_from(f8_to(y[k], E5) * c, E5); }\n"
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
  "kernel void publish(device uchar *cells [[buffer(0)]], constant args &p [[buffer(2)]]) { mesh_publish((volatile coherent(system) device ulong *)(cells + p.dst), p.scalar); }\n"
  "kernel void publish_filler(device uchar *cells [[buffer(0)]], device uchar *ring [[buffer(1)]], constant args &p [[buffer(2)]]) { "
  "*(volatile coherent(system) device ulong *)(ring + p.src) = STAMP(p.first); mesh_publish((volatile coherent(system) device ulong *)(cells + p.dst), p.scalar); }\n"
  "GROUP(send_small) { volatile coherent(system) device uchar *x = (volatile coherent(system) device uchar *)(d + p.dst); device const uchar *y = (device const uchar *)(s + p.src); LOOP x[k] = y[k]; "
  "threadgroup_barrier(mem_flags::mem_device); if (i == 0) mesh_publish((volatile coherent(system) device ulong *)(a + p.aux), p.scalar); }\n"
  "GROUP(land_copy) { AWAIT device uchar *x = d + p.dst; volatile coherent(system) device const uchar *y = (volatile coherent(system) device const uchar *)(s + p.src); LOOP x[k] = mesh_payload(y + k); }\n"
  "kernel void spin(device uchar *words [[buffer(0)]], device uchar *ring [[buffer(1)]], constant args &p [[buffer(2)]], device uchar *vb [[buffer(3)]]) { "
  "ulong status = mesh_remote_wait((volatile coherent(system) device const ulong *)(words + p.dst), p.n); "
  "volatile coherent(system) device ulong *v = (volatile coherent(system) device ulong *)vb; CHECK(ring, status) }\n";

/* ncclDataType_t's order: int8 uint8 int32 uint32 int64 uint64 float16 float32 float64 bfloat16 e4m3 e5m2 */
static const char *const TYPE[12] = {"i8", "u8", "i32", "u32", "i64", "u64", "f16", "f32", NULL, "bf16", "e4m3", "e5m2"};
static const char *const MODE[3] = {"plain", "land", "send"};

struct args { uint64_t dst, src, n, scalar, aux; uint32_t op, unused; uint64_t first, slot; uint32_t count, mask, channel, depth; };
struct metal_program { id<MTLCommandBuffer> buffer; id<MTLComputeCommandEncoder> encoder; NSMutableArray *kept; int borrowed; };

static id<MTLDevice> device_;
static id<MTLLibrary> library_;
static NSMutableDictionary *pipelines_;
static NSMutableArray *retiring_;
static char error_[512];

void *metal_device(void) {
  static dispatch_once_t once;
  dispatch_once(&once, ^{ device_ = MTLCreateSystemDefaultDevice(); });
  return device_;
}

const char *metal_error(void) { return error_; }

static id<MTLComputePipelineState> pipeline(NSString *name) {
  @autoreleasepool {
    if (!library_) {
      MTLCompileOptions *options = [[MTLCompileOptions new] autorelease];
      options.mathMode = MTLMathModeSafe;
      NSError *error = nil;
      NSString *primitives = [[[NSString alloc] initWithBytes:mesh_metal length:mesh_metal_len encoding:NSUTF8StringEncoding] autorelease];
      library_ = [(id<MTLDevice>)metal_device() newLibraryWithSource:[primitives stringByAppendingString:KERNELS] options:options error:&error];
      if (!library_) { snprintf(error_, sizeof error_, "the kernels: %s", error.localizedDescription.UTF8String); return nil; }
      pipelines_ = [NSMutableDictionary new];
      retiring_ = [NSMutableArray new];
    }
    id<MTLComputePipelineState> made = pipelines_[name];
    if (made) return made;
    id<MTLFunction> function = [[library_ newFunctionWithName:name] autorelease];
    NSError *error = nil;
    made = function ? [device_ newComputePipelineStateWithFunction:function error:&error] : nil;
    if (!made) { snprintf(error_, sizeof error_, "kernel %s: %s", name.UTF8String, error ? error.localizedDescription.UTF8String : "absent"); return nil; }
    pipelines_[name] = made;
    [made release];
    return made;
  }
}

void *metal_wrap(void *address, size_t bytes) {
  return [(id<MTLDevice>)metal_device() newBufferWithBytesNoCopy:address length:bytes options:MTLResourceStorageModeShared deallocator:nil];
}

void *metal_scratch(size_t bytes) {
  return [(id<MTLDevice>)metal_device() newBufferWithLength:bytes ? bytes : 16 options:MTLResourceStorageModePrivate];
}

void *metal_event(void) { return [(id<MTLDevice>)metal_device() newSharedEvent]; }
uint64_t metal_signaled(void *event) { return ((id<MTLSharedEvent>)event).signaledValue; }
void metal_release(void *object) { [(id)object release]; }

struct metal_program *metal_begin(void *command_buffer, void *encoder) {
  struct metal_program *program = calloc(1, sizeof *program);
  if (!program) return NULL;
  program->buffer = [(id<MTLCommandBuffer>)command_buffer retain];
  if (encoder) { program->encoder = [(id<MTLComputeCommandEncoder>)encoder retain]; program->borrowed = 1; }
  program->kept = [NSMutableArray new];
  return program;
}

static void close_encoder(struct metal_program *program) {
  if (!program->encoder) return;
  if (!program->borrowed) [program->encoder endEncoding];
  [program->encoder release];
  program->encoder = nil;
}

static int dispatch(struct metal_program *program, NSString *name, void *d, void *s, void *v, struct args a, uint64_t units) {
  if (!units) return 0;
  id<MTLComputePipelineState> state = pipeline(name);
  if (!state) return EINVAL;
  @autoreleasepool {
    if (!program->encoder) program->encoder = [[program->buffer computeCommandEncoder] retain];
    id<MTLComputeCommandEncoder> encoder = program->encoder;
    [encoder setComputePipelineState:state];
    [encoder setBuffer:(id<MTLBuffer>)d offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)(s ? s : d) offset:0 atIndex:1];
    [encoder setBytes:&a length:sizeof a atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)(v ? v : d) offset:0 atIndex:3];
    const NSUInteger width = state.maxTotalThreadsPerThreadgroup < 256 ? state.maxTotalThreadsPerThreadgroup : 256;
    [encoder dispatchThreads:MTLSizeMake(units < (1u << 20) ? (NSUInteger)units : (1u << 20), 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  }
  return 0;
}

int metal_copy(struct metal_program *program, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes) {
  const int words = !((dst_offset | src_offset | bytes) & 3);
  NSString *name = [NSString stringWithFormat:@"copy_%s_%d", MODE[mode], words ? 4 : 1];
  return dispatch(program, name, dst, src, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = words ? bytes / 4 : bytes}, words ? bytes / 4 : bytes);
}

static NSString *typed(const char *what, int type) {
  return (unsigned)type < 12 && TYPE[type] ? [NSString stringWithFormat:@"%s_%s", what, TYPE[type]] : nil;
}

int metal_combine(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n) {
  NSString *name = typed("combine", type);
  return name ? dispatch(program, name, dst, src, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = n, .op = (uint32_t)op}, n) : EINVAL;
}

int metal_premultiply(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t scalar) {
  NSString *name = typed("premul", type);
  return name ? dispatch(program, name, dst, src, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = n, .scalar = scalar}, n) : EINVAL;
}

int metal_truncdiv(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t divisor) {
  NSString *name = type <= 5 ? typed("truncdiv", type) : nil;
  return name ? dispatch(program, name, dst, src, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = n, .scalar = divisor}, n) : EINVAL;
}

uint64_t metal_stamp(uint64_t position) { return 0x6d6573682d66696cULL ^ (position * 0x9e3779b97f4a7c15ULL); }

static struct args checked(struct args a, const struct metal_check *check) {
  if (check) { a.first = check->first; a.slot = check->slot; a.count = check->count; a.mask = check->mask; a.channel = check->channel; a.depth = check->depth; }
  return a;
}

int metal_publish(struct metal_program *program, void *cells, size_t offset, uint64_t argument) {
  return dispatch(program, @"publish", cells, NULL, NULL, (struct args){.dst = offset, .n = 1, .scalar = argument}, 1);
}

int metal_publish_filler(struct metal_program *program, void *cells, size_t offset, uint64_t argument, void *slot, size_t slot_offset,
                         uint64_t position) {
  return dispatch(program, @"publish_filler", cells, slot, NULL,
                  (struct args){.dst = offset, .src = slot_offset, .n = 1, .scalar = argument, .first = position}, 1);
}

int metal_spin(struct metal_program *program, void *words, size_t offset, uint64_t cycle, const struct metal_check *check) {
  return dispatch(program, @"spin", words, check ? check->ring : NULL, check ? check->violation : NULL,
                  checked((struct args){.dst = offset, .n = cycle}, check), 1);
}

/* one threadgroup over a block's piece: `a` the cells (send) or the completion words (land) at aux */
static int dispatch_group(struct metal_program *program, NSString *name, void *d, void *s, void *a, void *v, struct args p) {
  id<MTLComputePipelineState> state = pipeline(name);
  if (!state) return EINVAL;
  @autoreleasepool {
    if (!program->encoder) program->encoder = [[program->buffer computeCommandEncoder] retain];
    id<MTLComputeCommandEncoder> encoder = program->encoder;
    [encoder setComputePipelineState:state];
    [encoder setBuffer:(id<MTLBuffer>)d offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)(s ? s : d) offset:0 atIndex:1];
    [encoder setBytes:&p length:sizeof p atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)a offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)(v ? v : d) offset:0 atIndex:4];
    const NSUInteger width = state.maxTotalThreadsPerThreadgroup < 1024 ? state.maxTotalThreadsPerThreadgroup : 1024;
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  }
  return 0;
}

int metal_send_small(struct metal_program *program, void *slot, size_t slot_offset, void *src, size_t src_offset, size_t bytes,
                     void *cells, size_t cell_offset, uint64_t value) {
  return dispatch_group(program, @"send_small", slot, src, cells, NULL,
                        (struct args){.dst = slot_offset, .src = src_offset, .n = bytes, .scalar = value, .aux = cell_offset});
}

int metal_land(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *slot, size_t slot_offset, size_t n,
               void *words, size_t word_offset, uint64_t cycle, const struct metal_check *check) {
  NSString *name = type < 0 ? @"land_copy" : typed("land", type);
  return name ? dispatch_group(program, name, dst, slot, words, check ? check->violation : NULL,
                               checked((struct args){.dst = dst_offset, .src = slot_offset, .n = n, .scalar = cycle, .aux = word_offset,
                                                     .op = (uint32_t)op}, check))
              : EINVAL;
}

void metal_keep(struct metal_program *program, void *object) { [program->kept addObject:(id)object]; }

/* the program's end signals `event` to `value`; what it kept is released once the event has passed it
   (metal_collect); a program in a borrowed encoder (event NULL) signals nothing and keeps nothing */
void metal_end(struct metal_program *program, void *event, uint64_t value) {
  close_encoder(program);
  if (event) {
    [program->buffer encodeSignalEvent:(id<MTLEvent>)event value:value];
    @autoreleasepool { [retiring_ addObject:@[@(value), program->kept]]; }
  }
  [program->kept release];
  [program->buffer release];
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
