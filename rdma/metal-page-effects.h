#pragma once
#import "metal-recording.h"
typedef struct isl_map isl_map;
typedef struct isl_ctx isl_ctx;

/* One preparation language for every numerical frontend. Relations use one
   caller-owned ISL context. A root is the canonical physical storage identity;
   relation ranges are byte positions in that root, not virtual addresses.
   No object declared in this header is consumed by numerical execution. */
enum MetalPageBackend { MetalPageMetal, MetalPageCPU, MetalPageANE };

@interface MetalPageAccess : NSObject {
@public
    id root;
    NSInteger slot;
    uint32_t access;
    isl_map *relation;                 /* flat dispatch group -> physical byte */
}
+ (instancetype)root:(id)root slot:(NSInteger)slot access:(uint32_t)access relation:(isl_map *)relation;
@end

/* A source recipe describes one finite iteration domain twice: the group
   which performs each access, and the byte it uses. Both are ordinary layout
   relations, including repeated/gathered positions. This is not a whole-buffer
   approximation and does not infer access geometry from a compiled pipeline. */
@interface MetalPageAccessLayout : NSObject {
@public
    id<MTLBuffer> buffer;
    NSInteger slot, invocation;
    uint32_t access;
    MetalTensorLayout *groups, *bytes;
}
+ (instancetype)buffer:(id<MTLBuffer>)buffer slot:(NSInteger)slot access:(uint32_t)access
    groups:(MetalTensorLayout *)groups bytes:(MetalTensorLayout *)bytes invocation:(NSInteger)invocation;
- (MetalPageAccess *)resolve:(isl_ctx *)context displacement:(NSInteger)displacement;
@end

@interface MetalPageCommand : NSObject {
@public
    NSUInteger order, groups, metadataSlot;
    enum MetalPageBackend backend;
    BOOL coherentAccesses;             /* retained body already uses shared coherent load/store helpers */
    NSArray<NSString *> *argumentNames; /* source pointer names indexed by binding slot; empty entries are not pointers */
    NSString *source, *function;
    MTLSize grid;
    NSArray<MetalPageAccess *> *accesses;
}
@property(nonatomic) BOOL coherentAccesses;
@property(nonatomic, copy) NSArray<NSString *> *argumentNames;
/* Source is the retained numerical entry, before page effects and backend
   binding. CPU entries name MeshCpuArguments and mesh_cpu_entry; ANE entries
   use the shared mesh-ane-entry.inc recipe. GPU entries retain their declared
   function and source templates. groups is the exact flattened dispatch count. */
+ (instancetype)order:(NSUInteger)order groups:(NSUInteger)groups backend:(enum MetalPageBackend)backend
    source:(NSString *)source function:(NSString *)function grid:(MTLSize)grid metadataSlot:(NSUInteger)slot
    accesses:(NSArray<MetalPageAccess *> *)accesses;
@end

@interface MetalPageTransfer : NSObject {
@public
    id owner, root;
    NSUInteger order;
    ncclMeshPageEffect effect;
    id<MTLBuffer> payload;
    MetalTensorLayout *layout;          /* same selected bytes in payload-local offsets */
    isl_map *selection;                /* zero-based selected byte -> physical byte */
}
/* owner identifies the declared collective occurrence, not a runtime handle.
   effect.first is the source/destination logical offset of selection byte 0. */
+ (instancetype)owner:(id)owner order:(NSUInteger)order effect:(ncclMeshPageEffect)effect
    root:(id)root selection:(isl_map *)selection payload:(id<MTLBuffer>)payload layout:(MetalTensorLayout *)layout;
@end

@interface MetalPageInitial : NSObject {
@public
    MetalPageTransfer *transfer;
    NSUInteger first, bytes;           /* relative to transfer selection */
    NSArray<MetalPageTransfer *> *observations;
}
@end

@interface MetalPageEffects : NSObject {
@public
    NSArray<MetalPageCommand *> *commands;
    NSArray<MetalPageInitial *> *initial;
    NSArray<MetalPageTransfer *> *results;
@package
    NSArray *domains, *solved;
}
@end

/* Solve first readers/last writers, channel-prefix contributions, true entry
   inputs and terminal observations. All selected bytes are accounted for;
   earlier arriving values become input prerequisites rather than dummy GPU
   work. Input ordering is source order; independent dispatches stay unordered. */
#ifdef __cplusplus
extern "C" {
#endif
/* Canonicalize declared byte uses through the bridge page mapping once.
   local is borrowed and maps any caller domain to absolute buffer-local byte
   offsets. The owned result keeps that domain and maps to region byte offsets,
   using local's existing ISL context. Aliases, repeated positions, masks and
   sub-page selections retain their exact relation; no access role is inferred.
   Selection is the same conversion for an ordered selected-byte layout. */
id MetalPageRegionRoot(void);
isl_map *MetalPageRegionRelation(isl_map *local, id<MTLBuffer> buffer);
isl_map *MetalPageRegionSelection(isl_ctx *context, id<MTLBuffer> buffer, MetalTensorLayout *bytes);
/* Relabel a source access by an explicit finite new-group -> old-group
   relation. Both layouts have the same domain; neither is numerical data. */
MetalPageAccess *MetalPageAccessGroups(MetalPageAccess *access,
    MetalTensorLayout *oldLabels, MetalTensorLayout *newGroups);
/* Project a retained work-coordinate access onto the target dispatch groups.
   axes has one entry per work dimension: 0,1,2 selects that launch axis and -1
   leaves it inside the group. Reduction dimensions also stay inside a group.
   access is borrowed; the result is owned in the same enclosing ISL context. */
isl_map *MetalPageGroupRelation(isl_map *access, const NSUInteger threads[3],
    const NSUInteger counts[3], const NSInteger *axes, NSUInteger axisCount,
    const NSUInteger *reductions, NSUInteger reductionCount);

// Public adapters hold the ordinary enclosing preparation context without
// importing vendor headers. Objects are released before their context.
isl_ctx *isl_ctx_alloc(void);
void isl_ctx_free(isl_ctx *);
MetalPageEffects *MetalPageEffectsSolve(NSArray<MetalPageCommand *> *commands,
    NSArray<MetalPageTransfer *> *transfers);
// Place one retained source block into the complete function's order. Equal
// local positions stay equal; no command or relation is reconstructed.
void MetalPageEffectsOrder(NSArray<MetalPageCommand *> *commands,
    NSArray<MetalPageTransfer *> *transfers, NSUInteger *next);
#ifdef __cplusplus
}
#endif

@interface MetalPageSource : NSObject {
@public
    NSString *source;
    id<MTLBuffer> metadata;
    NSUInteger metadataOffset;
    NSArray<NSNumber *> *observedSlots, *publishedSlots;
    NSData *reaches;                   /* extra MetalReach records, including initial reads */
@package
    NSArray *owners;
}
@property(nonatomic, readonly) NSString *source;
@property(nonatomic, readonly) id<MTLBuffer> metadata;
@property(nonatomic, readonly) NSUInteger metadataOffset;
@property(nonatomic, readonly) NSData *reaches;
@end

/* Allocate fixed metadata pages and specialize one retained entry. The
   returned metadata is one zero-copy page composition, bound at metadataSlot
   for Metal and the appended `uint8_t *mesh_effects` argument for CPU/ANE.
   The result owns its composed page view and all prefix state. It must be held
   by the resulting ordinary command, alongside its other page operands.
   This function creates no pipeline, request, host task or execution record. */
#ifdef __cplusplus
extern "C" {
#endif
MetalPageSource *MetalPageEffectsSource(MetalPageEffects *effects, NSUInteger command,
    id<MTLDevice> device);
/* Source-only stage; the fragments already use fixed metadata offsets. */
NSString *MetalPageEffectsCompose(MetalPageCommand *command, NSString *helpers, NSString *before,
    NSString *after, NSIndexSet *observedSlots, NSIndexSet *publishedSlots);
#ifdef __cplusplus
}
#endif
