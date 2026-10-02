#import <Metal/Metal.h>
#include "nccl-mesh-metal.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

/* the kernels: a landed slot is read through volatile coherent(system) loads and a slot a SEND reads is written
   through coherent(system) stores, its cell released after a system-scope fence (metal-microbench
   metal_recording.m MetalPayloadRead and MetalPublicationStores); the arithmetic is the host's (nccl-mesh.c
   combine_into, premultiply, truncdiv): the small floating types in float32 rounded to nearest even, integers
   wrapping */
static NSString *const SOURCE =
  @"#include <metal_stdlib>\n"
  "#pragma METAL internals : enable\n"
  "using namespace metal;\n"
  "#define SYS volatile coherent(system) device\n"
  "#define FENCE atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3))\n"
  "struct args { ulong dst, src, n, scalar; uint op, unused; };\n"
  "#define HEAD(name) kernel void name(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant args &p [[buffer(2)]], uint i [[thread_position_in_grid]], uint g [[threads_per_grid]])\n"
  "#define LOOP for (ulong k = i; k < p.n; k += g)\n"
  "#define COPY(name, W, TO, FROM) HEAD(name) { TO W *x = (TO W *)(d + p.dst); FROM const W *y = (FROM const W *)(s + p.src); LOOP x[k] = y[k]; }\n"
  "COPY(copy_plain_4, uint, device, device)\n"
  "COPY(copy_plain_1, uchar, device, device)\n"
  "COPY(copy_land_4, uint, device, SYS)\n"
  "COPY(copy_land_1, uchar, device, SYS)\n"
  "COPY(copy_send_4, uint, SYS, device)\n"
  "COPY(copy_send_1, uchar, SYS, device)\n"
  "#define INTEGER(name, T, W, U, SIGNED) "
  "HEAD(combine_##name) { device T *x = (device T *)(d + p.dst); SYS const W *y = (SYS const W *)(s + p.src); LOOP { W w = y[k]; T a = x[k], b = as_type<T>(w); "
  "if (p.op == 0) x[k] = T(U(a) + U(b)); else if (p.op == 1) x[k] = T(U(a) * U(b)); else if (p.op == 2) { if (b > a) x[k] = b; } else if (b < a) x[k] = b; } } "
  "HEAD(premul_##name) { device T *x = (device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); U c = U(T(p.scalar)); LOOP x[k] = T(U(y[k]) * c); } "
  "HEAD(truncdiv_##name) { device T *x = (device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); LOOP { T v = y[k]; bool negative = SIGNED && v < T(0); "
  "U m = negative ? U(0) - U(v) : U(v); U q = m / U(p.scalar); x[k] = negative ? T(U(0) - q) : T(q); } }\n"
  "#define FLOATING(name, T, W) "
  "HEAD(combine_##name) { device T *x = (device T *)(d + p.dst); SYS const W *y = (SYS const W *)(s + p.src); LOOP { W w = y[k]; float a = float(x[k]), b = float(as_type<T>(w)); "
  "if (p.op == 0) x[k] = T(a + b); else if (p.op == 1) x[k] = T(a * b); else if (p.op == 2) { if (b > a) x[k] = as_type<T>(w); } else if (b < a) x[k] = as_type<T>(w); } } "
  "HEAD(premul_##name) { device T *x = (device T *)(d + p.dst); device const T *y = (device const T *)(s + p.src); float c = as_type<float>(uint(p.scalar)); LOOP x[k] = T(float(y[k]) * c); }\n"
  "INTEGER(i8, char, uchar, uint, true)\n"
  "INTEGER(u8, uchar, uchar, uint, false)\n"
  "INTEGER(i32, int, uint, uint, true)\n"
  "INTEGER(u32, uint, uint, uint, false)\n"
  "INTEGER(i64, long, ulong, ulong, true)\n"
  "INTEGER(u64, ulong, ulong, ulong, false)\n"
  "FLOATING(f16, half, ushort)\n"
  "FLOATING(f32, float, uint)\n"
  "FLOATING(bf16, bfloat, ushort)\n"
  "kernel void publish(device uchar *cells [[buffer(0)]], constant args &p [[buffer(2)]]) { FENCE; *(SYS ulong *)(cells + p.dst) = p.scalar; FENCE; }\n";

/* ncclDataType_t's order: int8 uint8 int32 uint32 int64 uint64 float16 float32 float64 bfloat16 e4m3 e5m2 */
static const char *const TYPE[12] = {"i8", "u8", "i32", "u32", "i64", "u64", "f16", "f32", NULL, "bf16", NULL, NULL};
static const char *const MODE[3] = {"plain", "land", "send"};

struct args { uint64_t dst, src, n, scalar; uint32_t op, unused; };
struct metal_program { id<MTLCommandBuffer> buffer; id<MTLComputeCommandEncoder> encoder; NSMutableArray *kept; };

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
      library_ = [(id<MTLDevice>)metal_device() newLibraryWithSource:SOURCE options:options error:&error];
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
void metal_signal(void *event, uint64_t value) { ((id<MTLSharedEvent>)event).signaledValue = value; }
uint64_t metal_signaled(void *event) { return ((id<MTLSharedEvent>)event).signaledValue; }
void metal_release(void *object) { [(id)object release]; }

struct metal_program *metal_begin(void *command_buffer) {
  struct metal_program *program = calloc(1, sizeof *program);
  if (!program) return NULL;
  program->buffer = [(id<MTLCommandBuffer>)command_buffer retain];
  program->kept = [NSMutableArray new];
  return program;
}

static void close_encoder(struct metal_program *program) {
  if (!program->encoder) return;
  [program->encoder endEncoding];
  [program->encoder release];
  program->encoder = nil;
}

static int dispatch(struct metal_program *program, NSString *name, void *d, void *s, struct args a, uint64_t units) {
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
    const NSUInteger width = state.maxTotalThreadsPerThreadgroup < 256 ? state.maxTotalThreadsPerThreadgroup : 256;
    [encoder dispatchThreads:MTLSizeMake(units < (1u << 20) ? (NSUInteger)units : (1u << 20), 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  }
  return 0;
}

int metal_copy(struct metal_program *program, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes) {
  const int words = !((dst_offset | src_offset | bytes) & 3);
  NSString *name = [NSString stringWithFormat:@"copy_%s_%d", MODE[mode], words ? 4 : 1];
  return dispatch(program, name, dst, src, (struct args){dst_offset, src_offset, words ? bytes / 4 : bytes, 0, 0, 0}, words ? bytes / 4 : bytes);
}

static NSString *typed(const char *what, int type) {
  return (unsigned)type < 12 && TYPE[type] ? [NSString stringWithFormat:@"%s_%s", what, TYPE[type]] : nil;
}

int metal_combine(struct metal_program *program, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n) {
  NSString *name = typed("combine", type);
  return name ? dispatch(program, name, dst, src, (struct args){dst_offset, src_offset, n, 0, (uint32_t)op, 0}, n) : EINVAL;
}

int metal_premultiply(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t scalar) {
  NSString *name = typed("premul", type);
  return name ? dispatch(program, name, dst, src, (struct args){dst_offset, src_offset, n, scalar, 0, 0}, n) : EINVAL;
}

int metal_truncdiv(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n, uint64_t divisor) {
  NSString *name = type <= 5 ? typed("truncdiv", type) : nil;
  return name ? dispatch(program, name, dst, src, (struct args){dst_offset, src_offset, n, divisor, 0, 0}, n) : EINVAL;
}

int metal_publish(struct metal_program *program, void *cells, size_t offset, uint64_t argument) {
  return dispatch(program, @"publish", cells, NULL, (struct args){offset, 0, 1, argument, 0, 0}, 1);
}

void metal_wait(struct metal_program *program, void *event, uint64_t value) {
  close_encoder(program);
  [program->buffer encodeWaitForEvent:(id<MTLEvent>)event value:value];
}

void metal_keep(struct metal_program *program, void *object) { [program->kept addObject:(id)object]; }

/* the program's end signals `event` to `value`; what it kept is released once the event has passed it
   (metal_collect) */
void metal_end(struct metal_program *program, void *event, uint64_t value) {
  close_encoder(program);
  [program->buffer encodeSignalEvent:(id<MTLEvent>)event value:value];
  @autoreleasepool { [retiring_ addObject:@[@(value), program->kept]]; }
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
