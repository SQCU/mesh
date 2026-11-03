#pragma once
#import <Metal/Metal.h>
#include "metal-capabilities.h"
#include "mesh-host.h"
#include "nccl.h"
#import "mesh-tensor-layout.h"

#ifdef __cplusplus
extern "C" {
#endif

// Build operands from the selected Metal configuration, shared by frontends.
MTLCompileOptions *MetalDeviceCompilerOptions(id<MTLDevice> device);

// Preparation storage from the shared bridge region. The caller owns the buffer.
id<MTLBuffer> MetalLeaseBuffer(id<MTLDevice>, NSUInteger bytes) NS_RETURNS_RETAINED;
// docs/kernels.md Any program
// Observe qualifies only monotone progress reads. Initializing the word remains
// a dependency; concurrent Atomic publications do not order these observations.
// Exact data-page reaches carry producer ordering. The prepared observation
// threshold is an operand, not a join of all publishers of its progress word.
// Observations do not create anti-dependencies on later publications either.
enum { MetalReachRead = 1, MetalReachWritten = 2, MetalReachBinding = 4, MetalReachResident = 8, MetalReachAddress = 16,
    MetalReachObserve = 32, MetalReachNext = 64, MetalReachAtomic = 128 };
// Atomic qualifies an order-independent contribution (with Written, and Read
// for read-modify-write). The caller declares that peer contributions commute.
// Contributors retain prior ordinary-access edges but do not order each other;
// later ordinary reads and writes depend on every overlapping contributor.
// This is a preparation relation, not permission to reorder arbitrary atomics.
// Next qualifies a read of the next declared writer of each selected page
// range. Preparation binds that version and orders its producer before the
// reader, even when the producer is declared later. It has no runtime flag.
// Binding reaches name their actual argument index. Several disjoint ranges
// can describe one view; all follow that argument's prepared invocation slot.
// Address reaches use binding = the command address-record ordinal.
// A non-binding reach uses binding = -1 (memory addressed through an operand).
// Resident-only reaches retain storage without declaring a data dependency.
struct MetalReach {
    __unsafe_unretained id<MTLBuffer> buffer;
    NSUInteger offset, length;
    uint32_t access;
    int32_t binding;
};
// Preparation only: the union of a strided view's byte accesses, returned as
// packed MetalReach records. Strides are bytes; traversal order is irrelevant
// to dependencies. Empty views retain a zero-length binding declaration.
NSData *MetalViewReaches(id<MTLBuffer>, NSUInteger offset, const NSUInteger *shape,
    const NSInteger *strides, size_t rank, NSUInteger elementBytes, uint32_t access, int32_t binding);
// Preparation-only address lowering. The result maps a logical linear index
// to an element offset; shape/stride descriptors do not enter the kernel.
NSString *MetalViewIndexExpression(const NSUInteger *shape, const NSInteger *strides,
    size_t rank, NSString *index);
NSString *MetalViewIndexPrelude(void);
// design/compile-once-2026-10-29.md: the index prelude whose layouts are function constants, one text for every command,
// and an affine view's values for it ([index, type (0 bool, 1 uint, 2 long, 3 ulong), value] each; nil: use the source form)
NSString *MetalViewIndexConstantPrelude(void);
NSArray<NSArray<NSNumber *> *> *MetalViewIndexConstants(NSUInteger binding, const NSUInteger *shape, const NSInteger *strides,
    size_t rank, NSUInteger elementOffset, NSInteger relativeOffset);
// Required API strides are equivalent when they address the same elements.
// Empty and singleton axes do not impose a different physical layout.
BOOL MetalViewLayoutMatches(const NSUInteger *shape, const NSInteger *strides,
    const NSInteger *required, size_t rank);
// AOT copy tiles, packed as (origin[rank], shape[rank]) words per tile.
NSData *MetalViewTiles(const NSUInteger *shape, size_t rank, NSUInteger elements);
// Factor a power-of-two butterfly into local pieces. A stage transforms
// `width` values separated by `step` in the logical row. The operation's
// width is independent of the selected device's local storage capacity.
struct MetalButterflyStage { NSUInteger width, step; };
NSData *MetalButterflyPlan(NSUInteger width, NSUInteger localElements);
// Eligibility of the configured fused schedule, never a limit on the graph.
BOOL MetalHadamardLocal(NSUInteger columns, NSUInteger mix, NSUInteger threads, NSUInteger memory);
// A logical slice of an encoded axis of `stored` entries. The axis's domain is
// its `count` present entries, `present` (ascending stored positions), and
// `order` maps each to its logical index; no entry outside the domain is named.
// Bounds/step are the normalized slice; block is the format's indivisible
// stored group. Returns int32 retained block indices ("blocks"), and the
// result's own domain: its present positions in the retained blocks
// ("present", ascending) and each one's logical index ("order"). A retained
// block's other lanes are outside that domain.
// Only immutable index metadata is read. Encoded numerical values stay in place.
NSDictionary<NSString *, NSData *> *MetalViewBlockSlice(const int32_t *present, const int32_t *order, size_t count,
    size_t stored, NSInteger first, NSInteger end, NSInteger step, NSUInteger block);
// Explicit logical selections, including reordered or repeated indices. A
// stored block can be named more than once when it supplies repeated outputs.
NSDictionary<NSString *, NSData *> *MetalViewBlockGather(const int32_t *present, const int32_t *order, size_t count,
    size_t stored, const int32_t *selected, size_t selectedCount, NSUInteger block);
NSString *MetalViewIndexDeclaration(NSUInteger binding, const NSUInteger *shape, const NSInteger *strides,
    size_t rank, NSUInteger elementOffset);
// Axis maps contain signed offsets, in the same units as strides. A mapped
// axis replaces that axis's multiplication by stride. Maps are consumed at
// preparation: generated shaders contain arithmetic or one immutable lookup.
// Normalize affine maps into strides and a base offset before API binding.
NSDictionary<NSString *, id> *MetalViewNormalizeAxes(const NSInteger *strides, size_t rank,
    NSDictionary<NSNumber *, NSData *> *maps);
// Composed factor coordinates are emitted source expressions with a proved
// vector span; nil coordinates use the ordinary mixed-radix digits of shape.
NSString *MetalViewMappedIndexDeclaration(NSUInteger binding, const NSUInteger *shape, const NSInteger *strides,
    size_t rank, NSUInteger elementOffset, NSDictionary<NSNumber *, NSData *> *maps,
    NSArray<NSString *> *coordinates, NSUInteger vectorSpan, NSInteger relativeOffset);
// Declared index relations, such as an encoding's row positions. These are
// compiler metadata, never a substitute for a computed index tensor. The
// relation is defined on its `count` values alone: mesh_position<B>(i) for
// i < mesh_position_count<B>(), a compiled constant.
NSString *MetalViewIndexValuesDeclaration(NSUInteger binding, const NSInteger *values, NSUInteger count);
// Adapt an already emitted mesh_index<0> declaration to the position ABI.
// Layout relations and explicit value arrays share this source emitter.
NSString *MetalViewPositionDeclaration(NSUInteger binding, NSString *indices, NSUInteger count, NSUInteger span);
NSData *MetalViewMappedReaches(id<MTLBuffer> buffer, NSUInteger offset, const NSUInteger *shape,
    const NSInteger *strides, size_t rank, NSUInteger elementBytes, uint32_t access, int32_t binding,
    NSDictionary<NSNumber *, NSData *> *maps);
// Union prepared element spans with the same byte-access representation.
NSData *MetalTensorSpanReaches(id<MTLBuffer> buffer, NSUInteger base, NSData *spans,
    NSUInteger elementBytes, uint32_t access, int32_t binding);
/* The bytes of the type a pipeline's kernel declares at buffer `index` (its reflection's bufferDataSize; 0 where its
   pipeline was not made by MetalPipelineMake, or the kernel uses no buffer there), and its label */
NSUInteger MetalPipelineDeclaredBytes(id<MTLComputePipelineState>, NSUInteger index);
NSString *MetalPipelineName(id<MTLComputePipelineState>);
// An arithmetic pipeline's declaration: the library it is specialized from, its function,
// its function-constant values ([index, MTLDataType, bytes] each) and its descriptor's fields (label; threads, the most
// threads a threadgroup holds; multiple, a threadgroup size a multiple of the execution width; indirect, support for
// indirect commands). MetalPipelineMake consumes these preparation operands,
// using `archives` where available. The caller owns the pipeline.
id<MTLComputePipelineState> MetalPipelineMake(id<MTLLibrary> library, NSString *function, NSArray<NSArray *> *constants,
    NSDictionary<NSString *, id> *descriptor, NSArray<id<MTLBinaryArchive>> *archives, NSError **error);
#ifdef __cplusplus
}
#endif
