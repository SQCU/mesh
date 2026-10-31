#pragma once
#import <Metal/Metal.h>
#include "nccl.h"
NS_ASSUME_NONNULL_BEGIN

struct MetalTensorSpan { NSInteger offset; NSUInteger count; };

// Resolve a configuration's scalar integer term during preparation. Extents,
// domain size and named parameters use the layout expression's same algebra;
// logical coordinates are not inputs to a scalar storage requirement.
FOUNDATION_EXPORT NSInteger MetalTensorConstant(id expression, NSArray<NSNumber *> *shape,
    NSDictionary<NSString *, NSNumber *> *parameters);

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
// Smallest enclosing storage interval, in elements. This is an image-bounds
// query during preparation; it does not enumerate pages or dependency ranges.
@property(nonatomic, readonly) struct MetalTensorSpan storageSpan;
+ (instancetype)shape:(NSArray<NSNumber *> *)shape strides:(NSArray<NSNumber *> *)strides
    offset:(NSInteger)offset maps:(NSDictionary<NSNumber *, NSData *> *)maps indices:(nullable NSData *)indices;
// The same integer terms as coordinates, with bound configuration operands.
// "axis.N", "extent.N" and "count" name logical coordinates, axis extents and
// domain size. Parameters bind constants or other index relations over this
// domain. isl composes them before emission; no relation lookup runs at execution.
+ (instancetype)shape:(NSArray<NSNumber *> *)shape index:(id)expression parameters:(NSDictionary<NSString *, id> *)parameters;
// A compact ordered selection of logical positions, independent of a buffer.
+ (instancetype)runs:(NSArray<NSArray<NSNumber *> *> *)runs;
// Set intersection of logical positions, in ascending order, for preparation.
- (MetalTensorLayout *)intersection:(MetalTensorLayout *)other;
// Union of reached positions as a set; it does not concatenate value order.
- (MetalTensorLayout *)union:(MetalTensorLayout *)other;
// Ordinal positions whose selected value belongs to `values`. Repeated reads
// keep all of their distinct ordinals. This relation is resolved at preparation.
- (MetalTensorLayout *)preimage:(MetalTensorLayout *)values;
@property(nonatomic, readonly) NSDictionary *configuration;
+ (instancetype)configuration:(NSDictionary *)configuration;
// Ordered logical selections shared by graph dependency analysis and final
// page binding. The resolver supplies frontend configuration extents; nil
// accepts integers, products and bound {parameter,value,scale,offset} extents.
// The latter retain substitution identity through repeated preparation. Only
// preparation calls this.
+ (instancetype)selection:(NSDictionary *)declaration extent:(NSInteger (^ _Nullable)(id))extent;
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
// A requirement's `layout` can supply an exact physical relation, including
// a provider's padded axes reshaped onto the operand's logical axes.
- (MetalTensorLayout *)requirements:(NSArray<NSDictionary *> *)requirements;
// Address expressions and, for affine views, per-axis stride constants for
// numerical APIs. Both describe the assigned operand, after page placement.
- (NSString *)declaration:(NSUInteger)binding;
// A separately declared pointer base, in this layout's scalar units. The
// index expression retains every offset from that base to the selected pages.
- (NSString *)declaration:(NSUInteger)binding origin:(NSInteger)origin;
// Emit a declared index-valued relation as mesh_position<B>, with its logical
// count and consecutive-run alignment. The relation stays compiler data;
// arithmetic source is emitted without expanding a per-element index array.
- (NSString *)positionDeclaration:(NSUInteger)binding;
// The binding's layout as the constants form's function-constant values (metal-recording.h MetalViewIndexConstants), nil
// where the view is mapped or does not fit it.
- (NSArray<NSArray<NSNumber *> *> *)constants:(NSUInteger)binding;
- (NSArray<NSArray<NSNumber *> *> *)constants:(NSUInteger)binding origin:(NSInteger)origin;
// Lower the declared storage indices of a frontend IR to this page selection.
// The source index is compiler structure, not a payload index tensor.
- (NSString *)declaration:(NSUInteger)binding fromStorage:(MetalTensorLayout *)source;
// Logical traversal order, including repeated reads; unlike dependency union.
- (NSData *)orderedSpans;
- (NSData *)reaches:(nullable id<MTLBuffer>)buffer base:(NSUInteger)base elementBytes:(NSUInteger)bytes
    access:(uint32_t)access binding:(int32_t)binding;
@end

// Unite declared view equalities, quotienting repeated positions in either
// root. Classes follow source-root order, then unmatched target positions.
// A member keeps its ordered shape and repeated reads; nil member returns
// the combined root domain. Preparation only.
FOUNDATION_EXPORT MetalTensorLayout *MeshIndexUnion(MetalTensorLayout *sourceRoot, MetalTensorLayout *targetRoot,
    MetalTensorLayout *source, MetalTensorLayout *target, MetalTensorLayout * _Nullable member, BOOL sourceMember);
// Join explicit position-to-page assignments. Returns covered logical positions,
// their assigned page positions in that order, and the remaining logical domain.
// Conflicting assignments are preparation errors; no values are copied.
FOUNDATION_EXPORT NSArray<MetalTensorLayout *> *MetalTensorAssignments(NSUInteger count,
    NSArray<MetalTensorLayout *> *uses, NSArray<MetalTensorLayout *> *targets);
// Identify equal ordered uses within one domain. Returns the compact domain
// and every original ordinal's image in it, including transitive equalities.
FOUNDATION_EXPORT NSArray<MetalTensorLayout *> *MetalTensorIdentify(NSUInteger count,
    MetalTensorLayout *source, MetalTensorLayout *target);

// The domain maps concatenated storage ordinals to logical positions; selection
// maps requested ordinals to those positions. Preparation unites physical cuts
// from every participant before requesting direct bindings. Execution never
// sees this object.
struct MetalTensorFragment { NSUInteger part, count; NSInteger offset; };
@interface MetalTensorTraversal : NSObject
@property(nonatomic, readonly) NSArray<NSNumber *> *cuts;
+ (instancetype)layouts:(NSArray<MetalTensorLayout *> *)layouts domain:(MetalTensorLayout *)domain
    selection:(MetalTensorLayout *)selection;
- (struct MetalTensorFragment)fragment:(NSUInteger)first end:(NSUInteger)end;
@end

// Disjoint position sets with identical membership among the supplied domains.
// Each record contains `members` (domain indices) and `domain` (ordered positions).
// The sets stay symbolic until emitted as compact index runs.
FOUNDATION_EXPORT NSArray<NSDictionary *> *MetalTensorCover(NSArray<MetalTensorLayout *> *domains);

// Place one storage domain from declared use layouts. All indices and widths
// use the caller's common scalar unit. `values` preserves typed element fields
// and alignment; `uses`/`targets` impose relative positions for consumers.
// Each footprint is [first, count, alignment] in target coordinates. It reserves
// API writes beyond logical values and aligns the footprint's first address;
// [0, 0, 1] adds neither padding nor an alignment constraint.
// Relations retain value domains without per-value arrays. Connected groups
// pack in source-root order at their first aligned, nonoverlapping origins.
// A nil result means no common zero-copy placement exists. No data moves.
FOUNDATION_EXPORT MetalTensorLayout * _Nullable MetalTensorPlace(NSUInteger count,
    NSArray<MetalTensorLayout *> *uses, NSArray<MetalTensorLayout *> *targets,
    NSArray<NSArray<NSNumber *> *> *footprints, NSArray<MetalTensorLayout *> *values, NSArray<NSNumber *> *widths);

// The storage domain is the bridge's existing page table. This is compiler
// metadata: it resolves local frontend handles once and emits backend bindings.
// Neither this object nor a tensor layout is followed during execution.
@interface MetalTensorPages : NSObject
@property(nonatomic, readonly) ncclMeshPageTable_t table;
@property(nonatomic, readonly) id<MTLBuffer> buffer;
@property(nonatomic, readonly) NSUInteger pageBytes, pageCount, origin;
+ (instancetype)storage:(id<MTLBuffer>)buffer;
+ (instancetype)host:(const void *)address;
// Describe unique physical pages, or compose a declared order of those pages.
+ (MetalTensorLayout *)domain:(NSArray<MetalTensorPages *> *)sources;
+ (instancetype)compose:(NSArray<MetalTensorPages *> *)sources order:(MetalTensorLayout *)order alignment:(NSUInteger)alignment;
- (MetalTensorLayout *)physical:(MetalTensorLayout *)layout elementBytes:(NSUInteger)bytes;
- (MetalTensorLayout *)bind:(MetalTensorLayout *)layout elementBytes:(NSUInteger)bytes;
@end
NS_ASSUME_NONNULL_END
