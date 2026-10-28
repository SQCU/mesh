#pragma once
#import <Metal/Metal.h>
#include "nccl.h"
NS_ASSUME_NONNULL_BEGIN

// Preparation data, never a runtime tensor or a descriptor followed by a kernel.
// Every coordinate denotes one logical element. Its storage relation can be
// affine, separable, or joint across axes; these are normal forms of one type.
// Offsets and maps use elements. Buffer binding supplies the element's byte type.
@interface MetalTensorLayout : NSObject <NSCopying>
@property(nonatomic, readonly) NSArray<NSNumber *> *shape;
@property(nonatomic, readonly) NSArray<NSNumber *> *strides;
@property(nonatomic, readonly) NSDictionary<NSNumber *, NSData *> *maps;
@property(nonatomic, readonly, nullable) NSData *indices;
@property(nonatomic, readonly) NSInteger offset;
@property(nonatomic, readonly) NSUInteger count;
@property(nonatomic, readonly) BOOL affine;
+ (instancetype)shape:(NSArray<NSNumber *> *)shape strides:(NSArray<NSNumber *> *)strides
    offset:(NSInteger)offset maps:(NSDictionary<NSNumber *, NSData *> *)maps indices:(nullable NSData *)indices;
- (NSInteger)index:(NSUInteger)logical;
// Compose transformations now; the result never retains a parent descriptor.
- (MetalTensorLayout *)select:(NSUInteger)axis indices:(NSData *)indices;
- (MetalTensorLayout *)slice:(NSUInteger)axis first:(NSInteger)first count:(NSUInteger)count step:(NSInteger)step;
- (MetalTensorLayout *)element:(NSUInteger)axis at:(NSUInteger)index;
- (MetalTensorLayout *)permute:(NSArray<NSNumber *> *)axes;
- (MetalTensorLayout *)reshape:(NSArray<NSNumber *> *)shape;
- (MetalTensorLayout *)gather:(NSArray<NSNumber *> *)shape indices:(NSData *)indices;
// Subdivide each typed element into fixed-width components (e.g. bytes).
- (MetalTensorLayout *)components:(NSUInteger)count;
- (MetalTensorLayout *)shift:(NSInteger)elements;
// Compose a declared alias with its producer's assigned storage. Both source
// layouts describe the same logical elements. Only preparation calls this.
- (MetalTensorLayout *)rebase:(MetalTensorLayout *)declared onto:(MetalTensorLayout *)placed;
// Required strides are an API operand. Indexed consumers accept this layout;
// fixed-layout consumers either bind it directly or get a prepared copy edge.
- (BOOL)matches:(NSArray<NSNumber *> *)strides;
- (MetalTensorLayout *)required:(NSString *)kind strides:(NSArray<NSNumber *> *)strides;
// Intersect explicit use/API requirements. No requirement is discarded by a
// later lowering. Conflicts are preparation errors, before any numerical work.
- (MetalTensorLayout *)requirements:(NSArray<NSDictionary *> *)requirements;
- (NSString *)declaration:(NSUInteger)binding;
- (NSData *)reaches:(nullable id<MTLBuffer>)buffer base:(NSUInteger)base elementBytes:(NSUInteger)bytes
    access:(uint32_t)access binding:(int32_t)binding;
@end

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
