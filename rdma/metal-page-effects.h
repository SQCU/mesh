#pragma once
#import "metal-recording.h"
typedef struct isl_map isl_map;
typedef struct isl_ctx isl_ctx;

/* One preparation language for every numerical frontend. Relations use one
   caller-owned ISL context. A root is the canonical physical storage identity;
   relation ranges are byte positions in that root, not virtual addresses.
   No object declared in this header is consumed by numerical execution. */
enum MetalPageBackend { MetalPageMetal, MetalPageCPU, MetalPageANE };
enum MetalPageSiteScope { MetalPageSiteLane, MetalPageSiteThreadgroup, MetalPageSiteSimdgroup };

/* The index producer declares the actual contribution domain as ordinary
   pages. active and expected have one UInt32 per semantic transfer cut. Each
   physical transfer retains its coverageCut projection. A masked cut has
   active=expected=0. Counts are
   values, never the size of a possible-address union. The same active operand
   is consumed by the matched source/landing transport declaration. */
@interface MetalPageCoverage : NSObject {
@public
    ncclMeshPageCoverage declaration;
@package
    NSArray *identity;
}
+ (instancetype)declaration:(ncclMeshPageCoverage)declaration;
@end

/* An existing source continuation may establish imported coverage operands.
   Its ordinary typed reads carry their availability relation. relation maps
   phase-local cells to coverage ordinals; flags are 1 for expected and 4 for
   active. Numerical writers of these pages need no such annotation: Solve
   already finds their defining accesses. */
@interface MetalPageAnnouncement : NSObject {
@public
    MetalPageCoverage *coverage;
    NSUInteger phase;
    uint32_t flags;
    isl_map *relation;
}
+ (instancetype)coverage:(MetalPageCoverage *)coverage phase:(NSUInteger)phase
    flags:(uint32_t)flags relation:(isl_map *)relation;
@end

/* One fixed-width row per source cell. Each Int64 is an actual selected cut;
   -1 is an inactive slot. Repeated cut indices are separate contributions.
   The numerical source consumes these same index pages. Their contents may
   change; their shape and page binding are preparation operands. */
@interface MetalPageContribution : NSObject {
@public
    MetalPageCoverage *coverage;
    ncclMeshBuffer word, indices;      /* publication domain and selected local row ordinals */
    NSUInteger phase, cells, width;
}
+ (instancetype)coverage:(MetalPageCoverage *)coverage word:(ncclMeshBuffer)word phase:(NSUInteger)phase
    cells:(NSUInteger)cells width:(NSUInteger)width indices:(ncclMeshBuffer)indices;
@end

@interface MetalPageAccess : NSObject {
@public
    id root;
    NSInteger slot;
    uint32_t access;
    NSUInteger phase;                 /* entry, then declared source-local continuations */
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
    uint32_t frame;                    /* retained source instance within the enclosing occurrence */
    enum MetalPageBackend backend;
    BOOL coherentAccesses;             /* retained body already uses shared coherent load/store helpers */
    NSArray<NSString *> *argumentNames; /* source pointer names indexed by binding slot; empty entries are not pointers */
    NSString *source, *function;
    MTLSize grid;
    NSArray<MetalPageAccess *> *accesses;
    NSArray<NSDictionary *> *sites;    /* named source-local cell domains; preparation only */
    NSArray<MetalPageContribution *> *contributions;
    NSArray<MetalPageAnnouncement *> *announcements;
}
@property(nonatomic) BOOL coherentAccesses;
@property(nonatomic) uint32_t frame;
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
    MetalPageCoverage *coverage;
    NSUInteger coverageCut;
}
@property(nonatomic, retain) MetalPageCoverage *coverage;
@property(nonatomic) NSUInteger coverageCut;
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
/* Project a retained work-coordinate access onto the target dispatch groups.
   axes has one entry per work dimension: 0,1,2 selects that launch axis and -1
   leaves it inside the group. Reduction dimensions also stay inside a group.
   access is borrowed; the result is owned in the same enclosing ISL context. */
isl_map *MetalPageGroupRelation(isl_map *access, const NSUInteger threads[3],
    const NSUInteger counts[3], const NSInteger *axes, NSUInteger axisCount,
    const NSUInteger *reductions, NSUInteger reductionCount);
/* Persistent source import: the declaration contains the serialized logical
   relation and its grouped/active/reductions operands. Parse once into the
   enclosing context, project with the target launch, then compose its selected
   byte view and physical page mapping. In-process clients pass relations above. */
MetalPageAccess *MetalPageImportAccess(isl_ctx *context, NSDictionary *declaration,
    id<MTLBuffer> buffer, MetalTensorLayout *bytes, const NSUInteger threads[3],
    const NSUInteger counts[3]);

// Public adapters hold the ordinary enclosing preparation context without
// importing vendor headers. Objects are released before their context.
isl_ctx *isl_ctx_alloc(void);
void isl_ctx_free(isl_ctx *);
MetalPageEffects *MetalPageEffectsSolve(NSArray<MetalPageCommand *> *commands,
    NSArray<MetalPageTransfer *> *transfers);
/* The selected-cut domain consumed by contribution index pages. Each array
   has one physical publication word and increasing thresholds. The solver
   consumes these same rows; adapters never infer cut ordinals from addresses. */
NSArray<NSArray<MetalPageTransfer *> *> *MetalPagePublicationDomains(NSArray<MetalPageTransfer *> *transfers);
/* A source-local continuation uses the same fixed cell domain and exact page
   relations as an entry dispatch. Read-only cells may be masked or repeated.
   Static writer cells are reached once by the declared ownership relation;
   changing or repeated writes carry explicit coverage/contribution operands.
   Execution order may depend on ordinary index pages. The source declares MESH_PAGE_SITE(name) and calls
   MESH_PAGE_ENTER/LEAVE(name,metadata,cell,lane). Scope names the participating
   lanes; lane is local to that scope. CPU source cells are owned by one native
   invocation. GPU subgroups smaller than a SIMD group declare individual lane
   cells with their exact access relations. This is not a possible-address
   union and does not assert coverage for an arbitrary duplicate scatter. */
void MetalPageCommandSite(MetalPageCommand *, NSString *name, NSUInteger cells,
    enum MetalPageSiteScope scope, NSArray<MetalPageAccess *> *accesses);
/* Resource analysis and a wholly local site's empty effects use this same
   source ABI. Final specialization replaces these macros with compiled calls. */
NSString *MetalPageSitesPrelude(void);
/* Choose from the target entry's declared bindings, not reflection liveness.
   generatedBindings retains slots introduced by explicit parameter recipes.
   Capacity is the backend manifest operand. The caller retains this placement
   through source emission and binding; NSNotFound means all slots are occupied. */
NSUInteger MetalPageMetadataSlot(NSString *source, NSString *function, NSIndexSet *generatedBindings, NSUInteger capacity);
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
NSString *MetalPageEffectsCompose(MetalPageCommand *command, NSString *helpers, NSArray<NSString *> *before,
    NSArray<NSString *> *after, NSIndexSet *observedSlots, NSIndexSet *publishedSlots);
#ifdef __cplusplus
}
#endif
