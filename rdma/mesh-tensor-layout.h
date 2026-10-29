#pragma once
#import <Metal/Metal.h>
#include "nccl.h"
NS_ASSUME_NONNULL_BEGIN

struct MetalTensorSpan { NSInteger offset; NSUInteger count; };

// Preparation data, never a runtime tensor or a descriptor followed by a kernel.
// Every coordinate denotes one logical element. Its storage relation can be
// affine, separable, or joint across axes; these are normal forms of one type.
// Offsets and maps use elements. Buffer binding supplies the element's byte type.
@interface MetalTensorLayout : NSObject <NSCopying>
@property(nonatomic, readonly) NSArray<NSNumber *> *shape;
// Bounds, strides and maps of the index factors. coordinates selects their
// digits from a logical flat index using constant integer arithmetic. This
// compiler data is emitted as source; no relation descriptor reaches execution.
@property(nonatomic, readonly) NSArray<NSNumber *> *indexShape;
@property(nonatomic, readonly) NSArray *coordinates;
@property(nonatomic, readonly) BOOL factored;
@property(nonatomic, readonly) NSArray<NSNumber *> *strides;
@property(nonatomic, readonly) NSDictionary<NSNumber *, NSData *> *maps;
@property(nonatomic, readonly) NSInteger offset;
@property(nonatomic, readonly) NSUInteger count;
@property(nonatomic, readonly) BOOL affine;
+ (instancetype)shape:(NSArray<NSNumber *> *)shape strides:(NSArray<NSNumber *> *)strides
    offset:(NSInteger)offset maps:(NSDictionary<NSNumber *, NSData *> *)maps indices:(nullable NSData *)indices;
// Restore/compose flat factor coordinates, retaining no parent view. Coordinate
// expressions contain integer constants, "i", arithmetic, and the scalar
// selections emitted from piecewise integer relations. These are preparation
// terms; execution receives their compiled index source, never this tree.
- (MetalTensorLayout *)shaped:(NSArray<NSNumber *> *)shape coordinates:(NSArray *)coordinates;
- (NSInteger)index:(NSUInteger)logical;
// Compare the selected element positions as integer relations, even when
// their factorizations differ. This operation runs only during preparation.
- (BOOL)equivalent:(MetalTensorLayout *)other;
// Compose transformations now; the result never retains a parent descriptor.
- (MetalTensorLayout *)select:(NSUInteger)axis indices:(NSData *)indices;
// Ordered (first, width, step, count) runs. A repeated stripe is composed as
// integer coordinates; its repetitions do not become an element-index table.
- (MetalTensorLayout *)select:(NSUInteger)axis runs:(NSArray<NSArray<NSNumber *> *> *)runs;
- (MetalTensorLayout *)slice:(NSUInteger)axis first:(NSInteger)first count:(NSUInteger)count step:(NSInteger)step;
- (MetalTensorLayout *)element:(NSUInteger)axis at:(NSUInteger)index;
- (MetalTensorLayout *)permute:(NSArray<NSNumber *> *)axes;
- (MetalTensorLayout *)reshape:(NSArray<NSNumber *> *)shape;
- (MetalTensorLayout *)gather:(NSArray<NSNumber *> *)shape indices:(NSData *)indices;
// Subdivide each typed element into fixed-width components (e.g. bytes).
- (MetalTensorLayout *)components:(NSUInteger)count;
// Combine the declared last axis of adjacent, aligned scalar components into
// one typed element. The relation determines this property, not its strides'
// particular factorization.
- (MetalTensorLayout *)elements:(NSUInteger)count;
- (MetalTensorLayout *)shift:(NSInteger)elements;
// Compose a declared alias with its producer's assigned storage. Both source
// layouts describe the same logical elements. Only preparation calls this.
- (MetalTensorLayout *)rebase:(MetalTensorLayout *)declared onto:(MetalTensorLayout *)placed;
// Required strides are an API operand. Indexed consumers accept this layout;
// fixed-layout consumers either bind it directly or get a prepared copy edge.
// positive_strided requires an affine view with positive strides. Unused axes
// can change their stride metadata without changing any selected value.
- (BOOL)matches:(NSArray<NSNumber *> *)strides;
- (MetalTensorLayout *)required:(NSString *)kind strides:(NSArray<NSNumber *> *)strides;
// Intersect explicit use/API requirements. No requirement is discarded by a
// later lowering. Conflicts are preparation errors, before any numerical work.
- (MetalTensorLayout *)requirements:(NSArray<NSDictionary *> *)requirements;
// Address expressions and, for affine views, per-axis stride constants for
// numerical APIs. Both describe the assigned operand, after page placement.
- (NSString *)declaration:(NSUInteger)binding;
// The binding's layout as the constants form's function-constant values (metal-recording.h MetalViewIndexConstants), nil
// where the view is mapped or does not fit it.
- (NSArray<NSArray<NSNumber *> *> *)constants:(NSUInteger)binding;
// Lower the declared storage indices of a frontend IR to this page selection.
// The source index is compiler structure, not a payload index tensor.
- (NSString *)declaration:(NSUInteger)binding fromStorage:(MetalTensorLayout *)source;
// Logical traversal order, including repeated reads; unlike dependency union.
- (NSData *)orderedSpans;
- (NSData *)reaches:(nullable id<MTLBuffer>)buffer base:(NSUInteger)base elementBytes:(NSUInteger)bytes
    access:(uint32_t)access binding:(int32_t)binding;
@end

// Preparation maps ordered storage parts onto the caller's logical intervals.
// Cuts from every participant are united before a fragment is requested. The
// fragment is then a direct storage binding; execution never sees this object.
struct MetalTensorFragment { NSUInteger part, count; NSInteger offset; };
@interface MetalTensorTraversal : NSObject
@property(nonatomic, readonly) NSArray<NSNumber *> *cuts;
+ (instancetype)layouts:(NSArray<MetalTensorLayout *> *)layouts domains:(NSArray<NSArray<NSNumber *> *> *)domains;
- (struct MetalTensorFragment)fragment:(NSUInteger)first end:(NSUInteger)end;
@end

// Place one storage domain from declared use layouts. All indices and widths
// use the caller's common scalar unit. `values` preserves typed element fields
// and alignment; `uses`/`targets` impose relative positions for consumers.
// A nil result means no common zero-copy placement exists. No data moves.
FOUNDATION_EXPORT MetalTensorLayout * _Nullable MetalTensorPlace(NSUInteger count,
    NSArray<MetalTensorLayout *> *uses, NSArray<MetalTensorLayout *> *targets,
    NSArray<MetalTensorLayout *> *values, NSArray<NSNumber *> *widths);

// The storage domain is the bridge's existing page table. This is compiler
// metadata: it resolves local frontend handles once and emits backend bindings.
// Neither this object nor a tensor layout is followed during execution.
@interface MetalTensorPages : NSObject
@property(nonatomic, readonly) ncclMeshPageTable_t table;
@property(nonatomic, readonly) id<MTLBuffer> buffer;
@property(nonatomic, readonly) NSUInteger pageBytes, pageCount, origin;
+ (instancetype)storage:(id<MTLBuffer>)buffer;
+ (instancetype)host:(const void *)address;
- (MetalTensorLayout *)bind:(MetalTensorLayout *)layout elementBytes:(NSUInteger)bytes;
@end
NS_ASSUME_NONNULL_END
